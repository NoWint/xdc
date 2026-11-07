"""Optimisation primitives shared by coordinator and workers.

DiLoCo in one paragraph: every worker starts from the same global weights, performs
H inner AdamW steps on its own data shard, and returns ``delta = local_after -
global_before``.  The coordinator token-weights the deltas and applies an outer
SGD/Nesterov-momentum step.  Only one model-sized payload is exchanged per H steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import torch

from .. import native

from .data import sample_batch, val_batches
from .gpt import GPT, ModelConfig


@dataclass
class LocalTrainResult:
    delta: dict[str, torch.Tensor]
    loss: float
    steps: int
    tokens: int
    elapsed_s: float
    device: str
    update_norm: float = 0.0
    extra: dict = field(default_factory=dict)


def set_thread_count(threads: int | None) -> None:
    if threads and threads > 0:
        torch.set_num_threads(threads)


def pick_device(preferred: str, allow_fallback: bool = True) -> torch.device:
    preferred = (preferred or "auto").lower()
    if preferred == "auto":
        if torch.cuda.is_available():
            preferred = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            preferred = "mps"
        else:
            preferred = "cpu"
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        if not allow_fallback:
            raise RuntimeError("CUDA requested but unavailable")
        preferred = "cpu"
    if preferred == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        if not allow_fallback:
            raise RuntimeError("MPS requested but unavailable")
        preferred = "cpu"
    return torch.device(preferred)


def train_local(
    model: GPT,
    base_state: Mapping[str, torch.Tensor],
    data: bytes,
    *,
    local_steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    device: str | torch.device,
    seed: int,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
) -> LocalTrainResult:
    import time

    start = time.time()
    device = torch.device(device)
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    losses = []
    tokens = 0
    for _ in range(local_steps):
        x, y = sample_batch(data, batch_size, seq_len, gen, device="cpu")
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        tokens += x.numel()
    if device.type == "mps":
        torch.mps.synchronize()
    delta = {}
    norm_sq = 0.0
    for name, param in model.named_parameters():
        if name not in base_state:
            raise ValueError(f"parameter {name} missing from base state")
        d = param.detach().to("cpu", dtype=torch.float32) - base_state[name].detach().to("cpu", dtype=torch.float32)
        if not torch.isfinite(d).all():
            raise ValueError(f"non-finite update in {name}")
        delta[name] = d.contiguous()
        norm_sq += float(torch.sum(d * d))
    return LocalTrainResult(
        delta=delta,
        loss=sum(losses) / max(len(losses), 1),
        steps=local_steps,
        tokens=tokens,
        elapsed_s=time.time() - start,
        device=str(device),
        update_norm=math.sqrt(norm_sq),
    )


@torch.no_grad()
def evaluate_loss(
    state: Mapping[str, torch.Tensor],
    cfg: ModelConfig,
    val_data: bytes,
    *,
    batch_size: int = 4,
    seq_len: int = 128,
    max_batches: int = 8,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> float:
    device = torch.device(device)
    model = GPT(cfg)
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    model.to(device).eval()
    total = 0.0
    total_tokens = 0
    for x, y in val_batches(val_data, seq_len, max_batches=max_batches, seed=seed):
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        n = y.numel()
        total += float(loss) * n
        total_tokens += n
    return total / max(total_tokens, 1)


def update_norm(delta: Mapping[str, torch.Tensor]) -> float:
    s = 0.0
    for value in delta.values():
        s += float(torch.sum(value.float() * value.float()))
    return math.sqrt(s)


def robust_filter(
    deltas: list[Mapping[str, torch.Tensor]],
    weights: list[float],
    *,
    max_norm_ratio: float = 4.0,
    min_updates: int = 3,
) -> tuple[list[int], list[float], list[float]]:
    """Drop finite-but-extreme updates by L2 norm.

    The median absolute deviation guard needs at least three updates; with two
    workers there is no meaningful robust statistic, so only finite/shape checks
    (done earlier) apply.  This is an outlier guard, not Byzantine consensus.
    """
    norms = [update_norm(d) for d in deltas]
    if len(deltas) < min_updates:
        return list(range(len(deltas))), weights, norms
    median = sorted(norms)[len(norms) // 2]
    threshold = max(median * max_norm_ratio, 1e-6)
    keep = [i for i, n in enumerate(norms) if n <= threshold and n > 0]
    if not keep:
        keep = list(range(len(deltas)))
    return keep, [weights[i] for i in keep], norms


def aggregate_deltas(
    base_state: Mapping[str, torch.Tensor],
    deltas: list[Mapping[str, torch.Tensor]],
    weights: list[float],
    *,
    outer_lr: float = 1.0,
    momentum: float = 0.9,
    nesterov: bool = True,
    momentum_state: Mapping[str, torch.Tensor] | None = None,
    max_norm_ratio: float = 4.0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
    """Token-weighted outer update.

    Returns ``(new_state, new_momentum_state, stats)``.  The caller is responsible
    for saving the state and momentum atomically.
    """
    if not deltas:
        raise ValueError("no updates to aggregate")
    if len(deltas) != len(weights):
        raise ValueError("deltas/weights length mismatch")
    keys = list(base_state.keys())
    for idx, delta in enumerate(deltas):
        if set(delta.keys()) != set(keys):
            missing = set(keys) - set(delta.keys())
            extra = set(delta.keys()) - set(keys)
            raise ValueError(f"update {idx} parameter mismatch: missing={len(missing)} extra={len(extra)}")
        for key in keys:
            if delta[key].shape != base_state[key].shape:
                raise ValueError(f"update {idx} shape mismatch for {key}")
    if native.AVAILABLE:
        # One contiguous float32 buffer per model instead of ~50 tiny NumPy
        # conversions.  This cuts Python/kernel-launch overhead and lets the C++
        # merge run a single parallel pass over all parameters.
        sizes = [int(base_state[k].numel()) for k in keys]
        base_flat = torch.cat(
            [base_state[k].detach().reshape(-1).float() for k in keys]
        ).contiguous()
        delta_flats = []
        for delta in deltas:
            flat = torch.cat(
                [delta[k].detach().reshape(-1).float() for k in keys]
            ).contiguous()
            delta_flats.append(flat)
        prev_flat = None
        if momentum > 0 and momentum_state:
            prev_flat = torch.cat(
                [momentum_state[k].detach().reshape(-1).float() for k in keys]
            ).contiguous()
        out_flat, mom_flat, nstats = native.aggregate(
            base_flat.numpy(),
            [flat.numpy() for flat in delta_flats],
            weights,
            prev_flat.numpy() if prev_flat is not None else None,
            outer_lr,
            momentum,
            nesterov,
            max_norm_ratio,
        )
        out_t = torch.from_numpy(out_flat)
        mom_t = torch.from_numpy(mom_flat) if momentum > 0 else None
        new_state, new_mom = {}, {}
        offset = 0
        for key, size in zip(keys, sizes):
            # Views share the one output buffer; no per-tensor clone.
            new_state[key] = out_t.narrow(0, offset, size).view_as(base_state[key])
            if mom_t is not None:
                new_mom[key] = mom_t.narrow(0, offset, size).view_as(base_state[key])
            offset += size
        if not torch.isfinite(out_t).all():
            raise ValueError("aggregation produced non-finite parameters")
        stats = dict(nstats)
        stats["native"] = True
        return new_state, new_mom, stats

    for idx, delta in enumerate(deltas):
        for key in keys:
            if not torch.isfinite(delta[key]).all():
                raise ValueError(f"update {idx} has non-finite values in {key}")

    keep, kept_weights, norms = robust_filter(deltas, weights, max_norm_ratio=max_norm_ratio)
    total_w = sum(kept_weights)
    if total_w <= 0:
        raise ValueError("all update weights are zero")
    avg = {}
    for key in keys:
        acc = torch.zeros_like(base_state[key], dtype=torch.float32)
        for i in keep:
            acc += deltas[i][key].float() * (kept_weights[i] / total_w)
        avg[key] = acc
    new_state = {}
    new_mom = {}
    prev = momentum_state or {}
    for key in keys:
        if momentum > 0:
            buf = prev.get(key)
            buf = avg[key].clone() if buf is None else momentum * buf.float() + avg[key]
            step = avg[key] + momentum * buf if nesterov else buf
            new_mom[key] = buf.contiguous()
        else:
            step = avg[key]
        new_state[key] = (base_state[key].float() + outer_lr * step).contiguous()
        if not torch.isfinite(new_state[key]).all():
            raise ValueError(f"aggregation produced non-finite parameter {key}")
    stats = {
        "updates_received": len(deltas),
        "updates_used": len(keep),
        "dropped_indices": sorted(set(range(len(deltas))) - set(keep)),
        "norms": norms,
        "weights": weights,
        "total_weight": total_w,
        "native": False,
    }
    return new_state, new_mom, stats


def grad_conflict_stats(deltas: Iterable[Mapping[str, torch.Tensor]]) -> float:
    """Mean pairwise cosine similarity of pseudo-gradients (diagnostic only)."""
    vecs = []
    for d in deltas:
        flat = torch.cat([v.float().reshape(-1) for v in d.values()])
        n = torch.linalg.vector_norm(flat)
        vecs.append(flat / (n + 1e-12))
    if len(vecs) < 2:
        return 1.0
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(float(torch.dot(vecs[i], vecs[j])))
    return sum(sims) / len(sims)
