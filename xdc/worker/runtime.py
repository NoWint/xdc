"""Worker-local training and pipeline-stage inference runtimes."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..config import WorkerConfig
from ..model.checkpoint import load_state, serialize_state_bytes
from ..model.gpt import GPT, ModelConfig, PipelineStage
from ..model.train import set_thread_count, train_local
from ..protocol import Assignment, Capabilities, UpdateMeta
from ..tensorwire import decode_array, encode_array
from .client import ApiError, SigningClient

log = logging.getLogger("xdc.worker.runtime")


@dataclass
class StageSession:
    stage: PipelineStage
    cache: Any = None
    stage_index: int = 0
    last_used: float = 0.0
    device: str = "cpu"


class WorkerRuntime:
    def __init__(
        self,
        cfg: WorkerConfig,
        client: SigningClient,
        caps: Capabilities,
        device: torch.device,
        cache_dir: Path,
    ):
        self.cfg = cfg
        self.client = client
        self.caps = caps
        self.device = device
        self.cache_dir = cache_dir
        self.compute_lock = threading.Lock()
        self.sessions: dict[str, StageSession] = {}
        self.stage_cache: dict[str, PipelineStage] = {}
        self.training = False
        self.last_error: str | None = None
        self.last_training: dict[str, Any] = {}
        set_thread_count(cfg.threads or None)
        self._mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        self._cuda = torch.cuda.is_available()
        self._xpu = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())

    def device_for(self, params: int) -> torch.device:
        """Choose an execution device honestly.

        Apple MPS has high per-kernel dispatch overhead for small tensors; on an
        M-series CPU, Accelerate frequently wins by 5-10x for the tiny GPT slices
        used by the demo.  Auto therefore keeps small workloads on CPU and moves
        large ones to Metal.  Explicit ``--backend mps`` always wins.
        """
        if (self.cfg.backend or "auto").lower() != "auto":
            return self.device
        if self._cuda:
            return torch.device("cuda:0")
        if self._xpu:
            return torch.device("xpu")
        if self._mps and params >= max(1, self.cfg.mps_cpu_threshold_params):
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------ training
    async def run_assignment(self, assignment: Assignment) -> bool:
        started = time.time()
        self.training = True
        self.last_error = None
        try:
            if time.time() > assignment.deadline:
                raise TimeoutError("assignment expired before start")
            model_dest = self.cache_dir / "models" / f"{assignment.model_sha256}.safetensors"
            shard_sha = assignment.shard_sha256 or assignment.dataset_sha256
            data_dest = (
                self.cache_dir
                / "shards"
                / f"{shard_sha}-{assignment.shard_start}-{assignment.shard_end}.bin"
            )
            await asyncio.gather(
                self.client.download(assignment.model_path, assignment.model_sha256, model_dest),
                self.client.download(assignment.dataset_path, shard_sha, data_dest),
            )
            result, body = await asyncio.to_thread(self._train_sync, assignment, model_dest, data_dest)
            if time.time() > assignment.deadline:
                raise TimeoutError("local training exceeded round deadline")
            meta = UpdateMeta(
                job_id=assignment.job_id,
                round_id=assignment.round_id,
                node_id=self.client.identity.node_id,
                base_version=assignment.base_version,
                loss=result.loss,
                steps=result.steps,
                tokens=result.tokens,
                elapsed_s=result.elapsed_s,
                update_norm=result.update_norm,
                device=result.device,
                compressed="fp16" if assignment.compression == "fp16" else "fp32",
                nonce=os.urandom(12).hex(),
            )
            encoded_meta = base64.b64encode(meta.model_dump_json().encode()).decode()
            await self.client.post_raw(
                "/v1/worker/updates", body, {"X-XDC-Meta": encoded_meta}
            )
            self.last_training = {
                "job_id": assignment.job_id,
                "round_id": assignment.round_id,
                "round": assignment.round_index,
                "loss": result.loss,
                "tokens": result.tokens,
                "device": result.device,
                "elapsed_s": result.elapsed_s,
                "upload_s": time.time() - started - result.elapsed_s,
                "completed_at": time.time(),
                "update_norm": result.update_norm,
            }
            log.info(
                "round %s job=%s loss=%.4f tokens=%s device=%s",
                assignment.round_index,
                assignment.job_id,
                result.loss,
                result.tokens,
                result.device,
            )
            return True
        except ApiError as exc:
            self.last_error = f"upload rejected: {exc}"
            log.warning("assignment upload rejected: %s", exc)
            return False
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: {exc}"
            log.exception("training assignment failed")
            return False
        finally:
            self.training = False

    def _train_sync(self, assignment: Assignment, model_path: Path, data_path: Path):
        cfg = ModelConfig.from_dict(assignment.model_spec)
        data = data_path.read_bytes()
        if len(data) < assignment.seq_len + 2:
            raise ValueError("assigned data shard is too small")
        state = load_state(model_path)
        model = GPT(cfg)
        model.load_state_dict(state, strict=True)
        with self.compute_lock:
            device = self.device_for(model.num_parameters())
            result = train_local(
                model,
                state,
                data,
                local_steps=assignment.local_steps,
                batch_size=assignment.batch_size,
                seq_len=assignment.seq_len,
                lr=assignment.lr,
                device=device,
                seed=assignment.seed,
                weight_decay=assignment.weight_decay,
                grad_clip=assignment.grad_clip,
            )
        dtype = torch.float16 if assignment.compression == "fp16" else None
        body = serialize_state_bytes(result.delta, dtype=dtype)
        return result, body

    # ------------------------------------------------------------------ inference RPC
    async def stage_load(self, payload: dict[str, Any]) -> dict[str, Any]:
        cfg = ModelConfig.from_dict(payload["model_spec"])
        start = int(payload["layer_start"])
        end = int(payload["layer_end"])
        sha = str(payload["stage_sha256"])
        url = str(payload["stage_url"])
        dest = self.cache_dir / "stages" / f"{sha}.safetensors"
        await self.client.download(url, sha, dest)
        session_id = str(payload["session_id"])
        stage_index = int(payload["stage_index"])

        cache_key = f"{sha}:{start}:{end}"

        def load():
            cached = self.stage_cache.get(cache_key)
            if cached is not None:
                return cached, str(next(cached.parameters()).device)
            state = load_state(dest)
            stage = PipelineStage(cfg, start, end)
            stage.load_state_dict(state, strict=True)
            device = self.device_for(sum(p.numel() for p in stage.parameters()))
            stage.to(device).eval()
            self.stage_cache[cache_key] = stage
            return stage, str(device)

        with self.compute_lock:
            stage, actual_device = await asyncio.to_thread(load)
        self.sessions[session_id] = StageSession(
            stage=stage,
            cache=None,
            stage_index=stage_index,
            last_used=time.time(),
            device=actual_device,
        )
        self._evict_sessions()
        return {
            "session_id": session_id,
            "stage_index": stage_index,
            "layers": [start, end],
            "device": actual_device,
            "parameters": sum(p.numel() for p in stage.parameters()),
        }

    async def stage_forward(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload["session_id"])
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError("unknown inference session")
        if int(payload["stage_index"]) != session.stage_index:
            raise ValueError("stage index mismatch")
        stage = session.stage
        input_ids = payload.get("input_ids")
        hidden = payload.get("hidden")
        pos_offset = int(payload.get("pos_offset", 0))
        use_cache = bool(payload.get("use_cache", True))

        def compute():
            with self.compute_lock, torch.inference_mode():
                ids_t = None
                hidden_t = None
                session_device = torch.device(session.device)
                if input_ids is not None:
                    ids_t = torch.tensor([input_ids], dtype=torch.long, device=session_device)
                if hidden is not None:
                    hidden_t = torch.from_numpy(decode_array(hidden)).to(session_device)
                out = stage(
                    input_ids=ids_t,
                    hidden=hidden_t,
                    past_kv=session.cache,
                    pos_offset=pos_offset,
                    use_cache=use_cache,
                )
                hidden_out, logits, presents = out
                session.cache = presents
                session.last_used = time.time()
                result = {}
                if not stage.is_last:
                    result["hidden"] = encode_array(hidden_out.detach().float().cpu().numpy())
                else:
                    last = logits[:, -1, :].detach().float().cpu().numpy()[0]
                    result["logits"] = encode_array(last)
                    result["hidden"] = encode_array(hidden_out.detach().float().cpu().numpy())
                return result

        return await asyncio.to_thread(compute)

    async def stage_release(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload["session_id"])
        self.sessions.pop(session_id, None)
        return {"released": session_id}

    def _evict_sessions(self) -> None:
        max_sessions = max(1, self.cfg.max_inference_sessions)
        while len(self.sessions) > max_sessions:
            oldest = min(self.sessions.items(), key=lambda kv: kv[1].last_used)[0]
            self.sessions.pop(oldest, None)
            log.info("evicted inference session %s", oldest)

    def release_all(self) -> None:
        self.sessions.clear()
        self.stage_cache.clear()
        if self.device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    async def handle_rpc(self, method: str, payload: dict[str, Any]) -> Any:
        if method == "ping":
            return {"pong": time.time(), "node_id": self.client.identity.node_id}
        if method == "stage_load":
            return await self.stage_load(payload)
        if method == "stage_forward":
            return await self.stage_forward(payload)
        if method == "stage_release":
            return await self.stage_release(payload)
        if method == "status":
            return {
                "node_id": self.client.identity.node_id,
                "backend": self.caps.backend,
                "device": self.caps.device_name,
                "training": self.training,
                "sessions": len(self.sessions),
                "last_error": self.last_error,
            }
        raise ValueError(f"unknown RPC method {method}")
