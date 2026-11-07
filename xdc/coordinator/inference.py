"""True layer-sharded, pipeline-parallel inference.

Layers are allocated to workers by measured compute score.  Stage 0 runs the
embedding, the final stage runs norm + LM head, and only hidden activations cross
the network.  KV caches stay on the worker that owns the corresponding layers.
The coordinator samples tokens and streams raw UTF-8 bytes to the client.
"""

from __future__ import annotations

import base64
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import torch

from .. import native
from ..config import CoordinatorConfig
from ..model.checkpoint import load_state, save_state, sha256_file
from ..model.gpt import ModelConfig, PipelineStage
from ..tensorwire import decode_array, encode_array
from ..protocol import InferenceRequest
from .scheduler import StagePlan, allocate_stages, inference_candidates
from .state import JobRecord, StateStore
from .tunnel import TunnelError, TunnelHub

log = logging.getLogger("xdc.infer")


class InferenceError(RuntimeError):
    pass


class StageFailure(InferenceError):
    def __init__(self, node_id: str, message: str):
        super().__init__(message)
        self.node_id = node_id


@dataclass
class ResolvedModel:
    job: JobRecord
    cfg: ModelConfig
    path: Path
    state: dict[str, torch.Tensor]
    sha256: str


@dataclass
class StageToken:
    node_id: str
    path: Path
    expires_at: float


@dataclass
class GenerateResult:
    tokens: list[int] = field(default_factory=list)
    text: str = ""
    plan: list[dict] = field(default_factory=list)
    first_token_ms: float | None = None
    total_ms: float = 0.0
    distributed: bool = False
    attempts: int = 1


class InferenceEngine:
    def __init__(self, store: StateStore, tunnel: TunnelHub, cfg: CoordinatorConfig):
        self.store = store
        self.tunnel = tunnel
        self.cfg = cfg
        self.stage_tokens: dict[str, StageToken] = {}

    # ------------------------------------------------------------------ model
    def resolve(self, req: InferenceRequest) -> ResolvedModel:
        if not req.job_id:
            raise ValueError("job_id is required (checkpoints are addressed through their training job)")
        job = self.store.get_job(req.job_id)
        if not job:
            raise KeyError(f"job {req.job_id} not found")
        cfg = ModelConfig.from_dict(job.model)
        if req.checkpoint:
            p = Path(req.checkpoint)
            if not p.is_absolute():
                p = self.store.data_dir / p
        elif job.latest_checkpoint:
            p = self.store.data_dir / job.latest_checkpoint
        else:
            p = Path(job.model_path)
        p = p.resolve()
        data_root = self.store.data_dir.resolve()
        if data_root not in p.parents and p != data_root:
            raise PermissionError("checkpoint path escapes coordinator data directory")
        if not p.exists():
            raise FileNotFoundError(f"checkpoint {p} not found")
        state = load_state(p)
        # Validate early: corrupt checkpoints should fail before workers are called.
        probe = PipelineStage(cfg, 0, cfg.n_layer)
        probe.load_global_state(state)
        return ResolvedModel(job=job, cfg=cfg, path=p, state=state, sha256=sha256_file(p))

    async def prepare_plan(
        self, model: ResolvedModel, max_stages: int, exclude: set[str] | None = None
    ) -> tuple[list[StagePlan], dict[str, str]]:
        exclude = exclude or set()
        nodes = [
            n
            for n in inference_candidates(self.store, self.cfg.heartbeat_timeout_s)
            if n.node_id not in exclude and self.tunnel.connected(n.node_id)
        ]
        if not nodes:
            raise InferenceError("no inference workers with active tunnels")
        plans = allocate_stages(nodes, model.cfg.n_layer, max_stages=max_stages)
        tokens: dict[str, str] = {}
        for plan in plans:
            stage = PipelineStage(model.cfg, plan.layer_start, plan.layer_end)
            local_state = stage.layer_keys(model.state)
            out_dir = self.store.data_dir / "inference" / model.sha256[:24]
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"stage_{plan.layer_start}_{plan.layer_end}.safetensors"
            if not path.exists():
                save_state(path, local_state, {"kind": "xdc-pipeline-stage"})
            token = secrets.token_urlsafe(24)
            self.stage_tokens[token] = StageToken(
                node_id=plan.node.node_id, path=path, expires_at=time.time() + 120
            )
            tokens[plan.node.node_id] = f"/v1/inference/stage/{token}"
        self._expire_tokens()
        return plans, tokens

    def stage_token(self, token: str, node_id: str) -> Path:
        item = self.stage_tokens.get(token)
        if not item or item.expires_at < time.time():
            raise PermissionError("stage token missing or expired")
        if item.node_id != node_id:
            raise PermissionError("stage token is bound to another node")
        return item.path

    def _expire_tokens(self) -> None:
        now = time.time()
        for token, item in list(self.stage_tokens.items()):
            if item.expires_at < now:
                self.stage_tokens.pop(token, None)

    def plan_view(self, plans: list[StagePlan]) -> list[dict]:
        return [
            {
                "stage": p.stage_index,
                "node_id": p.node.node_id,
                "node_name": p.node.name,
                "backend": p.node.caps().backend,
                "device": p.node.caps().device_name,
                "layer_start": p.layer_start,
                "layer_end": p.layer_end,
                "layers": p.num_layers,
            }
            for p in plans
        ]

    # ------------------------------------------------------------------ generate
    async def generate_events(
        self, req: InferenceRequest, exclude: set[str] | None = None, attempt: int = 1
    ) -> AsyncIterator[dict]:
        started = time.perf_counter()
        model = self.resolve(req)
        exclude = exclude or set()
        plans, stage_urls = await self.prepare_plan(model, req.max_stages, exclude)
        session_id = "sess-" + secrets.token_hex(10)
        loaded: list[str] = []
        first_token_at = None
        produced = 0
        emitted = 0
        try:
            stage_views = self.plan_view(plans)
            for plan in plans:
                token = stage_urls[plan.node.node_id].rsplit("/", 1)[-1]
                payload = {
                    "session_id": session_id,
                    "stage_index": plan.stage_index,
                    "layer_start": plan.layer_start,
                    "layer_end": plan.layer_end,
                    "model_spec": model.cfg.to_dict(),
                    "stage_url": stage_urls[plan.node.node_id],
                    "stage_sha256": sha256_file(self.stage_tokens[token].path),
                }
                try:
                    load_result = await self.tunnel.call(
                        plan.node.node_id, "stage_load", payload, timeout=120
                    )
                except TunnelError as exc:
                    raise StageFailure(plan.node.node_id, str(exc)) from exc
                stage_views[plan.stage_index]["device"] = load_result.get("device", plan.node.caps().device_name)
                stage_views[plan.stage_index]["backend"] = load_result.get("device", "").split(":")[0]
                loaded.append(plan.node.node_id)
            yield {
                "event": "start",
                "data": {
                    "session_id": session_id,
                    "stages": stage_views,
                    "distributed": len(plans) > 1,
                    "model_sha256": model.sha256,
                    "attempt": attempt,
                    "context_limit": model.cfg.block_size,
                },
            }

            prompt = req.prompt.encode("utf-8") or b"\n"
            max_new = min(req.max_new_tokens, model.cfg.block_size - 1)
            if len(prompt) > model.cfg.block_size - max_new:
                prompt = prompt[-(model.cfg.block_size - max_new) :]
            if max_new < 1:
                raise ValueError("prompt uses the entire context window")
            tokens = list(prompt)
            prompt_len = len(tokens)
            hidden = None
            logits = None
            for idx, plan in enumerate(plans):
                payload = {
                    "session_id": session_id,
                    "stage_index": plan.stage_index,
                    "use_cache": True,
                    "pos_offset": 0,
                }
                if idx == 0:
                    payload["input_ids"] = tokens
                else:
                    payload["hidden"] = encode_array(hidden)
                try:
                    result = await self.tunnel.call(plan.node.node_id, "stage_forward", payload, timeout=60)
                except TunnelError as exc:
                    raise StageFailure(plan.node.node_id, str(exc)) from exc
                hidden = decode_array(result["hidden"]) if result.get("hidden") is not None else None
                logits = decode_array(result["logits"]) if result.get("logits") is not None else None
            if logits is None:
                raise InferenceError("final stage returned no logits")

            sampled = self._sample(logits, req, seed=req.seed + 0)
            tokens.append(sampled)
            produced += 1
            first_token_at = time.perf_counter()
            yield {"event": "token", "data": {"bytes_b64": base64.b64encode(bytes([sampled])).decode()}}
            emitted += 1

            for step in range(1, max_new):
                feed_token = sampled
                pos = prompt_len + step - 1
                hidden = None
                logits = None
                for idx, plan in enumerate(plans):
                    payload = {
                        "session_id": session_id,
                        "stage_index": plan.stage_index,
                        "use_cache": True,
                        "pos_offset": pos,
                    }
                    if idx == 0:
                        payload["input_ids"] = [feed_token]
                    else:
                        payload["hidden"] = encode_array(hidden)
                    try:
                        result = await self.tunnel.call(
                            plan.node.node_id, "stage_forward", payload, timeout=60
                        )
                    except TunnelError as exc:
                        raise StageFailure(plan.node.node_id, str(exc)) from exc
                    hidden = decode_array(result["hidden"]) if result.get("hidden") is not None else None
                    logits = decode_array(result["logits"]) if result.get("logits") is not None else None
                if logits is None:
                    raise InferenceError("final stage returned no logits")
                sampled = self._sample(logits, req, seed=req.seed + step)
                tokens.append(sampled)
                produced += 1
                yield {
                    "event": "token",
                    "data": {"bytes_b64": base64.b64encode(bytes([sampled])).decode()},
                }
                emitted += 1

            generated_bytes = bytes(tokens[prompt_len:])
            total_s = time.perf_counter() - started
            yield {
                "event": "done",
                "data": {
                    "tokens": produced,
                    "bytes": len(generated_bytes),
                    "text": generated_bytes.decode("utf-8", errors="replace"),
                    "first_token_ms": (first_token_at - started) * 1000 if first_token_at else None,
                    "total_ms": total_s * 1000,
                    "tokens_per_s": produced / max(total_s - (first_token_at - started if first_token_at else 0), 1e-9),
                    "distributed": len(plans) > 1,
                    "stages": stage_views,
                    "attempt": attempt,
                },
            }
        finally:
            for node_id in loaded:
                try:
                    await self.tunnel.call(node_id, "stage_release", {"session_id": session_id}, timeout=5)
                except Exception:
                    pass

    async def generate_text(self, req: InferenceRequest) -> GenerateResult:
        """Non-streaming API with full restart on worker failure."""
        exclude: set[str] = set()
        last_error: Exception | None = None
        for attempt in range(1, 4 if req.allow_retry else 2):
            tokens = bytearray()
            plan: list[dict] = []
            try:
                async for ev in self.generate_events(req, exclude=exclude, attempt=attempt):
                    if ev["event"] == "start":
                        plan = ev["data"]["stages"]
                    elif ev["event"] == "token":
                        tokens.extend(base64.b64decode(ev["data"]["bytes_b64"]))
                    elif ev["event"] == "done":
                        return GenerateResult(
                            tokens=list(tokens),
                            text=tokens.decode("utf-8", errors="replace"),
                            plan=plan,
                            first_token_ms=ev["data"].get("first_token_ms"),
                            total_ms=ev["data"]["total_ms"],
                            distributed=ev["data"]["distributed"],
                            attempts=attempt,
                        )
            except StageFailure as exc:
                last_error = exc
                log.warning("inference attempt %s failed on node %s: %s", attempt, exc.node_id, exc)
                exclude.add(exc.node_id)
                self._mark_node_failure(exc.node_id, str(exc))
                continue
            except Exception as exc:
                last_error = exc
                log.exception("inference attempt %s failed", attempt)
                continue
        raise InferenceError(f"inference failed after retries: {last_error}")

    def _sample(self, logits: np.ndarray, req: InferenceRequest, seed: int) -> int:
        arr = np.asarray(logits, dtype=np.float32).reshape(-1)
        if req.temperature <= 0:
            return int(np.argmax(arr))
        value = native.topk_sample(arr, req.top_k, req.temperature, seed)
        if value is not None:
            return value
        t = torch.from_numpy(arr)
        t = t / req.temperature
        if req.top_k > 0:
            k = min(req.top_k, t.numel())
            vals, _ = torch.topk(t, k)
            t[t < vals[-1]] = -float("inf")
        probs = torch.softmax(t, dim=-1)
        gen = torch.Generator().manual_seed(seed & 0x7FFFFFFF)
        return int(torch.multinomial(probs, 1, generator=gen).item())

    def _mark_node_failure(self, node_id: str, reason: str) -> None:
        node = self.store.get_node(node_id)
        if node:
            node.last_error = reason[:500]
            node.reputation = max(0.0, node.reputation - 0.1)
            self.store.save()


