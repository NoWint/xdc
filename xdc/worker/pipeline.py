"""Worker-side pipeline stage: parameters, activations, gradients and Adam state.

A stage owns exactly one contiguous slice of the model.  It keeps that slice's
parameters, gradients and Adam moments resident across rounds; the coordinator
never asks it to load the full model.  Forward activations stay in autograd
graphs until the corresponding backward RPC arrives (GPipe-style microbatch
scheduling), and the coordinator can abort a round and restore the last
committed stage/optimizer files.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save

from ..model.llama import LlamaStage
from ..model.gpt import PipelineStage
from ..model.registry import apply_lora_if_requested, model_config, trainable_parameters
from ..tensorwire import decode_array, encode_array

log = logging.getLogger("xdc.worker.pipeline")


class StageError(RuntimeError):
    pass


class StageVersionMismatch(StageError):
    pass


@dataclass
class PipelineSession:
    job_id: str
    stage_index: int
    layer_start: int
    layer_end: int
    kind: str
    model: Any
    optimizer: torch.optim.Optimizer
    device: torch.device
    version: int
    stage_sha256: str = ""
    lora_rank: int = 0
    lora_alpha: float = 0.0
    round_id: str | None = None
    round_version: int = 0
    pending: dict[int, tuple[Any, Any, Any]] = field(default_factory=dict)
    losses: list[float] = field(default_factory=list)
    tokens: int = 0
    activation_bytes: int = 0
    last_metrics: dict[str, Any] = field(default_factory=dict)
    round_started: float = 0.0
    last_used: float = field(default_factory=time.time)

    @property
    def is_first(self) -> bool:
        return self.layer_start == 0

    @property
    def is_last(self) -> bool:
        return bool(getattr(self.model, "is_last", False))

    def param_bytes(self) -> int:
        return sum(int(p.numel()) * int(p.element_size()) for p in self.model.parameters())

    def trainable_bytes(self) -> int:
        return sum(
            int(p.numel()) * int(p.element_size())
            for p in trainable_parameters(self.model)
        )

    def grad_bytes(self) -> int:
        total = 0
        for p in trainable_parameters(self.model):
            if p.grad is not None:
                total += int(p.grad.numel()) * int(p.grad.element_size())
        return total

    def optimizer_bytes(self) -> int:
        total = 0
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += int(value.numel()) * int(value.element_size())
        return total

    def state_hash(self) -> str:
        h = hashlib.sha256()
        for name, tensor in self.model.state_dict().items():
            t = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
            h.update(name.encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.numpy().tobytes())
        return h.hexdigest()


class PipelineRunner:
    def __init__(self, runtime, cache_dir: Path):
        self.runtime = runtime
        self.client = runtime.client
        self.cache_dir = cache_dir
        self.sessions: dict[str, PipelineSession] = {}

    # ------------------------------------------------------------------ load
    async def load(self, payload: dict[str, Any], force: bool = False) -> dict[str, Any]:
        job_id = str(payload["job_id"])
        stage_index = int(payload["stage_index"])
        stage_sha = str(payload["stage_sha256"])
        base_version = int(payload["base_version"])
        current = self.sessions.get(job_id)
        if (
            current is not None
            and not force
            and current.stage_index == stage_index
            and current.stage_sha256 == stage_sha
            and current.version == base_version
        ):
            current.last_used = time.time()
            return self._status(current, cached=True)

        kind = str(payload.get("model_kind", "hf"))
        spec = payload.get("model_spec") or {}
        cfg = model_config(spec)
        start = int(payload["layer_start"])
        end = int(payload["layer_end"])
        lora_rank = int(payload.get("lora_rank", 0))
        lora_alpha = float(payload.get("lora_alpha", 0.0))

        stage_dest = self.cache_dir / "pipeline" / f"{stage_sha}.safetensors"
        await self.client.download(str(payload["stage_url"]), stage_sha, stage_dest)
        opt_path = None
        if payload.get("optimizer_url") and payload.get("optimizer_sha256"):
            opt_sha = str(payload["optimizer_sha256"])
            opt_path = self.cache_dir / "pipeline" / f"opt-{opt_sha}.safetensors"
            await self.client.download(str(payload["optimizer_url"]), opt_sha, opt_path)

        if current is not None:
            self._release(current)

        def build():
            state = load_file(str(stage_dest))
            base_state = {k: v for k, v in state.items() if "lora_" not in k}
            if kind == "hf":
                stage = LlamaStage(cfg, start, end)  # type: ignore[arg-type]
            else:
                stage = PipelineStage(cfg, start, end)  # type: ignore[arg-type]
            stage.load_state_dict(base_state, strict=True)
            adapted = apply_lora_if_requested(stage, lora_rank, lora_alpha)
            if adapted:
                missing, unexpected = stage.load_state_dict(state, strict=False)
                missing = [k for k in missing if "lora_" not in k]
                if missing or unexpected:
                    raise StageError(
                        f"bad LoRA stage slice: missing={missing[:8]} unexpected={unexpected[:8]}"
                    )
            params = [p for p in trainable_parameters(stage)]
            if not params:
                raise StageError("stage has no trainable parameters")
            param_count = sum(int(p.numel()) for p in stage.parameters())
            device = self.runtime.device_for(param_count)
            stage.to(device)
            stage.train()
            optimizer = torch.optim.AdamW(
                params,
                lr=float(payload.get("lr", 3e-4)),
                betas=(0.9, 0.95),
                weight_decay=float(payload.get("weight_decay", 0.01)),
            )
            if opt_path is not None:
                self._load_optimizer(optimizer, stage, opt_path)
            return stage, optimizer, device, adapted

        with self.runtime.compute_lock:
            stage, optimizer, device, adapted = await asyncio.to_thread(build)
        session = PipelineSession(
            job_id=job_id,
            stage_index=stage_index,
            layer_start=start,
            layer_end=end,
            kind=kind,
            model=stage,
            optimizer=optimizer,
            device=device,
            version=base_version,
            stage_sha256=stage_sha,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.sessions[job_id] = session
        log.info(
            "pipeline stage loaded job=%s stage=%s layers=%s..%s device=%s params=%.1fM lora=%s",
            job_id,
            stage_index,
            start,
            end,
            device,
            sum(int(p.numel()) for p in stage.parameters()) / 1e6,
            adapted,
        )
        return self._status(session, cached=False)

    @staticmethod
    def _load_optimizer(
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        path: Path,
    ) -> None:
        data = load_file(str(path))
        loaded = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            exp_avg = data.get(f"{name}.exp_avg")
            exp_avg_sq = data.get(f"{name}.exp_avg_sq")
            step = data.get(f"{name}.step")
            if exp_avg is None or exp_avg_sq is None or step is None:
                continue
            if tuple(exp_avg.shape) != tuple(param.shape):
                raise StageError(f"optimizer shape mismatch for {name}")
            optimizer.state[param]["step"] = step.detach().to(param.device).float().reshape(())
            optimizer.state[param]["exp_avg"] = exp_avg.detach().to(param.device).float()
            optimizer.state[param]["exp_avg_sq"] = exp_avg_sq.detach().to(param.device).float()
            loaded += 1
        if loaded == 0:
            raise StageError("optimizer file contained no matching parameter state")

    def _status(self, session: PipelineSession, cached: bool) -> dict[str, Any]:
        return {
            "job_id": session.job_id,
            "stage_index": session.stage_index,
            "layer_start": session.layer_start,
            "layer_end": session.layer_end,
            "version": session.version,
            "cached": cached,
            "device": str(session.device),
            "parameters": sum(int(p.numel()) for p in session.model.parameters()),
            "trainable_parameters": sum(int(p.numel()) for p in trainable_parameters(session.model)),
            "param_bytes": session.param_bytes(),
            "optimizer_bytes": session.optimizer_bytes(),
            "grad_bytes": session.grad_bytes(),
            "measured_total_bytes": session.param_bytes() + session.optimizer_bytes() + session.grad_bytes(),
            "lora_rank": session.lora_rank,
        }

    # ------------------------------------------------------------------ round
    def _session(self, payload: dict[str, Any]) -> PipelineSession:
        job_id = str(payload["job_id"])
        session = self.sessions.get(job_id)
        if session is None:
            raise StageError("pipeline stage is not loaded")
        if session.stage_index != int(payload.get("stage_index", session.stage_index)):
            raise StageError("stage index mismatch")
        return session

    def begin(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._session(payload)
        round_id = str(payload["round_id"])
        base_version = int(payload["base_version"])
        log.info("pipeline_begin stage=%s round=%s base=%s held=%s", session.stage_index, round_id, base_version, session.version)
        if session.round_id is not None:
            if session.round_id == round_id:
                return {"ok": True, "duplicate": True, "version": session.version}
            raise StageError("another pipeline round is active")
        if session.version != base_version:
            raise StageVersionMismatch(
                f"stage holds version {session.version}, coordinator expects {base_version}"
            )
        session.round_id = round_id
        session.round_version = base_version
        session.pending.clear()
        session.losses = []
        session.tokens = 0
        session.activation_bytes = 0
        session.model.train()
        session.optimizer.zero_grad(set_to_none=True)
        session.round_started = time.time()
        session.last_used = time.time()
        return {"ok": True, "version": session.version, "param_bytes": session.param_bytes()}

    def forward(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._session(payload)
        if session.round_id is None:
            raise StageError("no active pipeline round")
        round_id = str(payload.get("round_id", ""))
        if round_id and round_id != session.round_id:
            raise StageError("round id mismatch")
        mb_id = int(payload["mb_id"])
        if mb_id in session.pending:
            raise StageError("microbatch already forwarded")
        log.info("pipeline_forward stage=%s mb=%s last=%s", session.stage_index, mb_id, session.is_last)

        def compute():
            with torch.enable_grad():
                device = session.device
                hidden_in = None
                if payload.get("hidden") is not None:
                    arr = decode_array(payload["hidden"])
                    hidden_in = torch.from_numpy(arr).to(device)
                    hidden_in.requires_grad_(True)
                ids = None
                if payload.get("input_ids") is not None:
                    ids = torch.tensor(
                        payload["input_ids"], dtype=torch.long, device=device
                    )
                out, _ = session.model.forward_features(
                    input_ids=ids,
                    hidden=hidden_in,
                    pos_offset=0,
                    use_cache=False,
                )
                loss = None
                if session.is_last:
                    targets = torch.tensor(
                        payload["targets"], dtype=torch.long, device=device
                    )
                    logits = session.model.head_logits(out)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.reshape(-1),
                        reduction="mean",
                    ) * float(payload.get("loss_scale", 1.0))
                    session.pending[mb_id] = (hidden_in, out, loss)
                    value = float(loss.detach().cpu())
                    session.losses.append(value)
                else:
                    session.pending[mb_id] = (hidden_in, out, None)
                    value = None
                tokens = int(ids.numel()) if ids is not None else 0
                session.tokens += tokens
                return out, value, tokens

        with self.runtime.compute_lock:
            out, loss_value, tokens = compute()
        wire_dtype = "float16" if payload.get("activation_dtype") == "fp16" else "float32"
        result: dict[str, Any] = {
            "mb_id": mb_id,
            "tokens": tokens,
            "loss": loss_value,
            "shape": list(out.shape),
        }
        if not session.is_last:
            arr = out.detach().float().cpu().numpy()
            result["hidden"] = encode_array(arr, dtype=wire_dtype)
            session.activation_bytes += int(arr.size) * (2 if wire_dtype == "float16" else 4)
        session.last_used = time.time()
        return result

    def backward(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._session(payload)
        round_id = str(payload.get("round_id", ""))
        if session.round_id is None or (round_id and round_id != session.round_id):
            raise StageError("no matching active pipeline round")
        mb_id = int(payload["mb_id"])
        item = session.pending.pop(mb_id, None)
        if item is None:
            raise StageError("microbatch was not forwarded")
        hidden_in, out, loss = item
        log.info("pipeline_backward stage=%s mb=%s last=%s", session.stage_index, mb_id, session.is_last)

        def compute():
            with torch.enable_grad():
                if session.is_last:
                    if loss is None:
                        raise StageError("final stage has no loss graph")
                    loss.backward()
                else:
                    if payload.get("grad_hidden") is None:
                        raise StageError("non-final stage requires grad_hidden")
                    grad = torch.from_numpy(decode_array(payload["grad_hidden"])).to(session.device)
                    torch.autograd.backward(out, grad)
                if hidden_in is not None and hidden_in.grad is not None:
                    return hidden_in.grad.detach().float().cpu().numpy(), loss
                return None, loss

        with self.runtime.compute_lock:
            grad, loss = compute()
        wire_dtype = "float16" if payload.get("activation_dtype") == "fp16" else "float32"
        result: dict[str, Any] = {"mb_id": mb_id}
        if grad is not None:
            result["grad_hidden"] = encode_array(grad, dtype=wire_dtype)
            session.activation_bytes += int(grad.size) * (2 if wire_dtype == "float16" else 4)
        session.last_used = time.time()
        return result

    def finalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._session(payload)
        if session.round_id is None:
            raise StageError("no active pipeline round")
        if session.pending:
            raise StageError(f"{len(session.pending)} microbatches still pending backward")
        params = list(trainable_parameters(session.model))
        log.info("pipeline_finalize stage=%s params=%s", session.stage_index, len(params))

        def step():
            grads = [p.grad for p in params if p.grad is not None]
            if len(grads) != len(params):
                missing = sum(1 for p in params if p.grad is None)
                raise StageError(f"{missing} trainable parameters received no gradient")
            total_sq = 0.0
            for g in grads:
                if not torch.isfinite(g).all():
                    raise StageError("non-finite gradient")
                total_sq += float(g.detach().float().pow(2).sum())
            norm = math.sqrt(total_sq)
            if not math.isfinite(norm):
                raise StageError("non-finite gradient norm")
            grad_clip = float(payload.get("grad_clip", 1.0))
            grad_bytes = sum(
                int(p.grad.numel()) * int(p.grad.element_size())
                for p in params
                if p.grad is not None
            )
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            session.optimizer.step()
            session.optimizer.zero_grad(set_to_none=True)
            return norm, grad_bytes

        with self.runtime.compute_lock:
            grad_norm, grad_bytes = step()
        log.info("pipeline_finalize stage=%s stepped grad_norm=%s", session.stage_index, grad_norm)
        session.version += 1
        session.round_id = None
        session.last_used = time.time()
        param_hash = session.state_hash()
        log.info("pipeline_finalize stage=%s hashed", session.stage_index)
        metrics = {
            "stage_index": session.stage_index,
            "version": session.version,
            "grad_norm": grad_norm,
            "loss": sum(session.losses) / max(len(session.losses), 1),
            "tokens": session.tokens,
            "activation_bytes": session.activation_bytes,
            "param_bytes": session.param_bytes(),
            "optimizer_bytes": session.optimizer_bytes(),
            "grad_bytes": grad_bytes,
            "trainable_parameters": sum(int(p.numel()) for p in params),
            "parameters": sum(int(p.numel()) for p in session.model.parameters()),
            "device": str(session.device),
            "param_hash": param_hash,
            "elapsed_s": time.time() - session.round_started,
        }
        session.last_metrics = metrics
        return metrics

    async def export(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._session(payload)
        round_id = str(payload["round_id"])
        version = int(payload["version"])
        if session.version != version:
            raise StageVersionMismatch(
                f"stage holds version {session.version}, cannot export {version}"
            )
        if session.round_id is not None:
            raise StageError("cannot export while a round is active")

        def serialize_params() -> bytes:
            state = {
                name: tensor.detach().to("cpu").contiguous()
                for name, tensor in session.model.state_dict().items()
            }
            return save(state)

        def serialize_optimizer() -> bytes:
            out = {}
            for name, param in session.model.named_parameters():
                if not param.requires_grad:
                    continue
                state = session.optimizer.state.get(param)
                if not state:
                    continue
                out[f"{name}.exp_avg"] = state["exp_avg"].detach().to("cpu").float().contiguous()
                out[f"{name}.exp_avg_sq"] = state["exp_avg_sq"].detach().to("cpu").float().contiguous()
                out[f"{name}.step"] = torch.tensor(
                    float(state["step"].item() if torch.is_tensor(state["step"]) else state["step"]),
                    dtype=torch.float32,
                )
            return save(out)

        params = await asyncio.to_thread(serialize_params)
        optimizer = await asyncio.to_thread(serialize_optimizer)
        round_meta = {
            "job_id": session.job_id,
            "round_id": round_id,
            "node_id": self.client.identity.node_id,
            "stage_index": session.stage_index,
            "version": version,
            "nonce": os.urandom(12).hex(),
        }
        import base64
        import json

        uploads = []
        for kind, blob in (("params", params), ("optimizer", optimizer)):
            meta = dict(round_meta, kind=kind, size=len(blob))
            header = base64.b64encode(json.dumps(meta, separators=(",", ":")).encode()).decode()
            await self.client.post_raw(
                "/v1/pipeline/upload",
                blob,
                {"X-XDC-Pipeline-Meta": header},
                timeout=600.0,
            )
            sha = hashlib.sha256(blob).hexdigest()
            if kind == "params":
                session.stage_sha256 = sha
            uploads.append({"kind": kind, "bytes": len(blob), "sha256": sha})
        return {"stage_index": session.stage_index, "version": version, "uploads": uploads}

    async def abort(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._session(payload)
        session.round_id = None
        session.pending.clear()
        session.losses = []
        session.tokens = 0
        session.activation_bytes = 0
        session.optimizer.zero_grad(set_to_none=True)
        restored = False
        if payload.get("stage_url") and payload.get("stage_sha256"):
            await self.load(payload, force=True)
            restored = True
        else:
            session.version = int(payload.get("base_version", session.version))
        return {
            "stage_index": session.stage_index,
            "restored": restored,
            "version": session.version,
        }

    def status(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.sessions.get(str(payload.get("job_id", "")))
        if session is None:
            return {"loaded": False}
        out = self._status(session, cached=True)
        out["loaded"] = True
        out["round_id"] = session.round_id
        out["last_metrics"] = session.last_metrics
        return out

    def release(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = str(payload.get("job_id", ""))
        session = self.sessions.pop(job_id, None)
        if session is not None:
            self._release(session)
        return {"released": bool(session)}

    @staticmethod
    def _release(session: PipelineSession) -> None:
        session.pending.clear()
        session.model.cpu()
        session.optimizer.state.clear()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
