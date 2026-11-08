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
from ..model.llama import LlamaConfig, LlamaModel, apply_lora
from ..model.registry import model_kind, stage_state_from_global
from ..model.tokenizer import ByteTokenizer, HFTokenizer, TokenizerInfo, load_tokenizer
from ..tensorwire import decode_array, encode_array
from ..protocol import InferenceRequest
from .scheduler import StagePlan, allocate_stages, group_by_island, inference_candidates
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
    cfg: ModelConfig | LlamaConfig
    path: Path
    state: dict[str, torch.Tensor]
    sha256: str
    kind: str = "tinygpt"
    tokenizer: ByteTokenizer | HFTokenizer | None = None
    lora_rank: int = 0
    lora_alpha: float = 0.0
    context_limit: int = 0


@dataclass
class StageToken:
    node_id: str
    path: Path
    expires_at: float


@dataclass
class GenerateResult:
    tokens: list[int] = field(default_factory=list)
    text: str = ""
    raw: bytes = b""
    token_count: int = 0
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
        self._resolved: ResolvedModel | None = None
        self._resolved_key: str = ""

    # ------------------------------------------------------------------ model
    def resolve(self, req: InferenceRequest) -> ResolvedModel:
        if not req.job_id:
            raise ValueError("job_id is required (checkpoints are addressed through their training job)")
        job = self.store.get_job(req.job_id)
        if not job:
            raise KeyError(f"job {req.job_id} not found")
        kind = job.model_kind or model_kind(job.model)
        if kind == "hf":
            cfg = LlamaConfig.from_dict(job.model)
        else:
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
        stat = p.stat()
        cache_key = f"{job.job_id}:{p}:{stat.st_size}:{stat.st_mtime_ns}"
        if self._resolved is not None and self._resolved_key == cache_key:
            return self._resolved
        state = load_state(p)
        rank = int(job.model.get("lora_rank", 0))
        alpha = float(job.model.get("lora_alpha", 0.0))
        tokenizer = None
        if kind == "hf":
            model_dir = self._model_dir(job)
            tokenizer = load_tokenizer(model_dir, TokenizerInfo.from_dict(job.tokenizer))
            probe = LlamaModel(cfg)  # type: ignore[arg-type]
            if rank:
                apply_lora(probe.model, r=rank, alpha=alpha)
            probe.load_hf_state(state)
            context_limit = int(cfg.max_position_embeddings)  # type: ignore[union-attr]
        else:
            # Validate early: corrupt checkpoints should fail before workers are called.
            probe = PipelineStage(cfg, 0, cfg.n_layer)  # type: ignore[arg-type]
            probe.load_global_state(state)
            tokenizer = ByteTokenizer()
            context_limit = int(cfg.block_size)  # type: ignore[union-attr]
        resolved = ResolvedModel(
            job=job,
            cfg=cfg,
            path=p,
            state=state,
            sha256=sha256_file(p),
            kind=kind,
            tokenizer=tokenizer,
            lora_rank=rank,
            lora_alpha=alpha,
            context_limit=context_limit,
        )
        # Keep one checkpoint resident.  Validation and tokenizer construction
        # are seconds of work on a 135M-parameter model and must not be repeated
        # for every generation request.
        self._resolved = resolved
        self._resolved_key = cache_key
        return resolved

    def _model_dir(self, job: JobRecord) -> Path:
        if job.model_dir and Path(job.model_dir).is_dir():
            return Path(job.model_dir)
        name = str(job.model.get("model_dir") or job.model.get("model_name") or "")
        if not name:
            raise InferenceError("HF job has no model directory")
        return self.store.data_dir / "models" / name

    async def prepare_plan(
        self, model: ResolvedModel, max_stages: int, exclude: set[str] | None = None
    ) -> tuple[list[StagePlan], dict[str, str]]:
        exclude = exclude or set()
        cands = inference_candidates(self.store, self.cfg.heartbeat_timeout_s)
        log.info(
            "inference candidates=%s cutoff=%.1f",
            [(n.node_id, n.reputation, n.banned, n.last_seen) for n in cands],
            self.cfg.heartbeat_timeout_s,
        )
        island = model.job.spec().island
        nodes = [
            n
            for n in cands
            if n.node_id not in exclude
            and self.tunnel.connected(n.node_id)
            and (not island or n.labels.get("island", "default") == island)
        ]
        nodes = group_by_island(nodes)
        if not nodes:
            raise InferenceError(
                "no inference workers with active tunnels: "
                f"candidates={[(n.node_id, self.tunnel.connected(n.node_id), n.node_id in exclude) for n in self.store.list_nodes()]} "
                f"hub={self.tunnel.states()}"
            )
        layers = (
            model.cfg.num_hidden_layers if model.kind == "hf" else model.cfg.n_layer  # type: ignore[union-attr]
        )
        plans = allocate_stages(nodes, layers, max_stages=max_stages)
        tokens: dict[str, str] = {}
        latest = Path(model.job.model_path).resolve() if model.job.model_path else None
        records = (
            (model.job.stage_checkpoints or {}).get(str(model.job.version), {})
            if latest is not None and model.path.resolve() == latest
            else {}
        )
        by_range = {
            (int(r["layer_start"]), int(r["layer_end"])): r
            for r in records.values()
            if r.get("params_path")
        }
        tied = bool(
            model.kind == "hf"
            and isinstance(model.cfg, LlamaConfig)
            and model.cfg.tie_word_embeddings
        )
        for plan in plans:
            # Training already wrote a stage-local checkpoint at the exact same
            # layer range.  Serving it avoids rebuilding a 100 MB+ file on every
            # inference session.  The final tied stage is the exception: its
            # duplicated head is resynchronised from the global embedding.
            record = by_range.get((plan.layer_start, plan.layer_end))
            reuse = bool(
                record
                and Path(self.store.data_dir / record["params_path"]).exists()
                and not (tied and plan.layer_end == layers)
            )
            if reuse:
                path = self.store.data_dir / record["params_path"]
            else:
                local_state = stage_state_from_global(
                    model.kind, model.cfg, model.state, plan.layer_start, plan.layer_end
                )
                out_dir = self.store.data_dir / "inference" / model.sha256[:24]
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / (
                    f"stage_{plan.layer_start}_{plan.layer_end}_{model.sha256[:12]}.safetensors"
                )
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
                "island": p.node.labels.get("island", "default"),
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
                    "model_spec": model.job.model,
                    "model_kind": model.kind,
                    "stage_url": stage_urls[plan.node.node_id],
                    "stage_sha256": sha256_file(self.stage_tokens[token].path),
                    "lora_rank": model.lora_rank,
                    "lora_alpha": model.lora_alpha,
                    "tokenizer": model.job.tokenizer,
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
                    "context_limit": model.context_limit,
                },
            }

            assert model.tokenizer is not None
            prompt_ids = model.tokenizer.encode(req.prompt)
            if not prompt_ids:
                bos = model.job.tokenizer.get("bos_token_id")
                prompt_ids = [int(bos)] if bos is not None else [1]
            max_new = min(req.max_new_tokens, model.context_limit - 1)
            if len(prompt_ids) > model.context_limit - max_new:
                prompt_ids = prompt_ids[-(model.context_limit - max_new) :]
            if max_new < 1:
                raise ValueError("prompt uses the entire context window")
            tokens = list(prompt_ids)
            prompt_len = len(tokens)
            generated_ids: list[int] = []
            emitted_text = ""
            emitted_raw = 0
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
            generated_ids.append(sampled)
            produced += 1
            first_token_at = time.perf_counter()
            delta = self._token_text(model, generated_ids, emitted_text, emitted_raw)
            emitted_text = delta["full"]
            emitted_raw += len(delta["bytes"])
            if delta["bytes"]:
                yield {
                    "event": "token",
                    "data": {"bytes_b64": base64.b64encode(delta["bytes"]).decode()},
                }
                emitted += len(delta["bytes"])

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
                generated_ids.append(sampled)
                produced += 1
                delta = self._token_text(model, generated_ids, emitted_text, emitted_raw)
                emitted_text = delta["full"]
                emitted_raw += len(delta["bytes"])
                if delta["bytes"]:
                    yield {
                        "event": "token",
                        "data": {"bytes_b64": base64.b64encode(delta["bytes"]).decode()},
                    }
                    emitted += len(delta["bytes"])

            generated_bytes = (
                emitted_text.encode("utf-8")
                if model.kind == "hf"
                else bytes(int(i) & 0xFF for i in generated_ids)
            )
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
                            tokens=[],
                            text=tokens.decode("utf-8", errors="replace"),
                            raw=bytes(tokens),
                            token_count=int(ev["data"].get("tokens", 0)),
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

    @staticmethod
    def _token_text(
        model: ResolvedModel, ids: list[int], previous: str, emitted_raw: int
    ) -> dict:
        assert model.tokenizer is not None
        if model.kind == "hf":
            full = model.tokenizer.decode(ids)
            raw = full.encode("utf-8")
            prev = previous.encode("utf-8")
            if raw.startswith(prev):
                delta = raw[len(prev) :]
            else:
                # The tokenizer rewrote an earlier partial character; send the
                # whole suffix from the point the client has consumed.
                delta = raw[emitted_raw:] if emitted_raw <= len(raw) else raw
            return {"bytes": delta, "full": full}
        data = bytes(int(i) & 0xFF for i in ids)
        delta = data[emitted_raw:] if emitted_raw <= len(data) else data
        return {"bytes": delta, "full": data.decode("utf-8", errors="replace")}

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


