"""Coordinator-side pipeline-parallel training.

The model is split into contiguous layer stages.  Each stage owns its parameters
and Adam moments on a worker; only hidden activations and their gradients cross
the tunnel.  A round is a GPipe-style schedule over ``micro_batches``:

    forward  mb0: stage0 -> stage1 -> ... -> stageN-1 (loss)
    forward  mb1: ...
    backward mb1: stageN-1 -> ... -> stage0
    backward mb0: ...
    finalize: every stage clips gradients and performs its own AdamW step
    commit:   stages upload their shard + optimizer state; coordinator assembles
              a global checkpoint and advances the version

Nothing in this path downloads a full model to a worker.  Failures abort the
round and restore the last committed shard/optimizer on every surviving stage.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..config import CoordinatorConfig
from ..model.checkpoint import load_state, save_state, sha256_file
from ..model.gpt import GPT, PipelineStage
from ..model.llama import LlamaConfig, LlamaModel, LlamaStage, apply_lora
from ..model.registry import (
    model_config,
    stage_layer_count,
    stage_state_from_global,
)
from ..model.tokenizer import fixed_token_batches, read_tokens, sample_token_batch
from ..protocol import PipelineAssignment
from ..tensorwire import decode_array, encode_array
from .scheduler import node_score, training_candidates
from .state import JobRecord, NodeRecord, RoundRecord, StateStore
from .tunnel import TunnelError, TunnelHub

log = logging.getLogger("xdc.pipeline")


class PipelinePlanError(RuntimeError):
    """The available devices cannot hold the model with this stage count."""


class PipelineRoundError(RuntimeError):
    pass


class StageWorkerError(PipelineRoundError):
    def __init__(self, node_id: str, stage_index: int, message: str):
        super().__init__(message)
        self.node_id = node_id
        self.stage_index = stage_index


@dataclass
class StageArtifact:
    stage_index: int
    node: NodeRecord
    layer_start: int
    layer_end: int
    stage_path: Path
    stage_sha256: str
    optimizer_path: Path | None
    optimizer_sha256: str | None
    estimated_bytes: int
    parameters: int
    trainable_parameters: int


@dataclass
class StageToken:
    node_id: str
    path: Path
    kind: str
    expires_at: float


@dataclass
class PipelineUpload:
    job_id: str
    round_id: str
    node_id: str
    stage_index: int
    version: int
    kind: str
    path: Path
    sha256: str
    size: int
    created_at: float


@dataclass
class RoundResult:
    metrics: list[dict[str, Any]] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    tokens: int = 0
    activation_bytes: int = 0
    wall_s: float = 0.0
    uploads: dict[str, dict[str, PipelineUpload]] = field(default_factory=dict)


class PipelineEngine:
    def __init__(self, store: StateStore, tunnel: TunnelHub, cfg: CoordinatorConfig, owner):
        self.store = store
        self.tunnel = tunnel
        self.cfg = cfg
        self.owner = owner
        self.stage_tokens: dict[str, StageToken] = {}
        self.uploads: dict[tuple[str, str, int, str], PipelineUpload] = {}

    # ------------------------------------------------------------------ public
    async def advance(self, job: JobRecord) -> None:
        spec = job.spec()
        if job.next_round >= spec.rounds:
            self.owner._complete(job)
            return
        if job.current_round is not None:
            # StateStore discards volatile rounds on restart; a live round should
            # be handled by the running coroutine, not reopened here.
            return
        await self._run_round(job)

    def stage_token(self, token: str, node_id: str, kind: str) -> Path:
        item = self.stage_tokens.get(token)
        if not item or item.expires_at < time.time():
            raise PermissionError("pipeline stage token missing or expired")
        if item.node_id != node_id or item.kind != kind:
            raise PermissionError("pipeline stage token is bound to another node or file")
        return item.path

    def record_upload(self, meta: dict[str, Any], body: bytes) -> dict[str, Any]:
        job = self.store.get_job(str(meta.get("job_id", "")))
        if job is None:
            raise KeyError("job not found")
        rnd = job.current_round
        if job.state != "running" or rnd is None or rnd.round_id != meta.get("round_id"):
            raise ValueError("round is not active")
        if job.spec().parallelism != "pipeline":
            raise ValueError("job is not a pipeline job")
        node_id = str(meta["node_id"])
        stages = {s["node_id"]: s for s in rnd.pipeline.get("stages", [])}
        stage = stages.get(node_id)
        if stage is None:
            raise PermissionError("node has no stage in this round")
        if int(stage["stage_index"]) != int(meta["stage_index"]):
            raise ValueError("stage index mismatch")
        version = int(meta["version"])
        if version != job.version + 1:
            raise ValueError("pipeline upload version mismatch")
        kind = str(meta["kind"])
        if kind not in {"params", "optimizer"}:
            raise ValueError("invalid pipeline upload kind")
        expected_size = int(meta.get("size", -1))
        if expected_size != len(body) or expected_size <= 0:
            raise ValueError("pipeline upload size mismatch")
        sha = hashlib.sha256(body).hexdigest()
        if str(meta.get("sha256", "")) not in {"", sha}:
            raise ValueError("pipeline upload hash mismatch")
        dest_dir = Path(job.model_path).parent / "pipeline" / rnd.round_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"stage_{int(meta['stage_index'])}.{kind}.safetensors"
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        import os

        os.replace(tmp, path)
        record = PipelineUpload(
            job_id=job.job_id,
            round_id=rnd.round_id,
            node_id=node_id,
            stage_index=int(meta["stage_index"]),
            version=version,
            kind=kind,
            path=path,
            sha256=sha,
            size=len(body),
            created_at=time.time(),
        )
        self.uploads[(rnd.round_id, node_id, int(meta["stage_index"]), kind)] = record
        rnd.pipeline.setdefault("uploads", {})[f"{node_id}:{kind}"] = {
            "path": str(path),
            "sha256": sha,
            "size": len(body),
            "stage_index": int(meta["stage_index"]),
        }
        self.store.save()
        return {"ok": True, "kind": kind, "sha256": sha, "bytes": len(body)}

    def plan_view(self, job: JobRecord) -> list[dict[str, Any]]:
        return list(job.stage_plan)

    def assignment_for(self, job: JobRecord, node_id: str) -> PipelineAssignment | None:
        rnd = job.current_round
        if not rnd:
            return None
        for stage in rnd.pipeline.get("stages", []):
            if stage.get("node_id") == node_id:
                return PipelineAssignment(
                    job_id=job.job_id,
                    job_name=job.name,
                    round_id=rnd.round_id,
                    round_index=rnd.index,
                    base_version=rnd.base_version,
                    stage_index=int(stage["stage_index"]),
                    num_stages=len(rnd.pipeline.get("stages", [])),
                    layer_start=int(stage["layer_start"]),
                    layer_end=int(stage["layer_end"]),
                    model_spec=job.model,
                    model_kind=job.model_kind,
                    stage_path=stage["stage_path"],
                    stage_sha256=stage["stage_sha256"],
                    optimizer_path=stage.get("optimizer_path"),
                    optimizer_sha256=stage.get("optimizer_sha256"),
                    micro_batches=job.spec().micro_batches,
                    batch_size=job.spec().batch_size,
                    seq_len=job.spec().seq_len,
                    lr=job.spec().lr,
                    weight_decay=job.spec().weight_decay,
                    grad_clip=job.spec().grad_clip,
                    seed=job.spec().seed,
                    deadline=rnd.deadline,
                    activation_dtype=job.spec().activation_dtype,
                    lora_rank=job.spec().lora_rank,
                    lora_alpha=job.spec().lora_alpha,
                    island=job.spec().island,
                    tokenizer=job.tokenizer,
                    estimated_bytes=int(stage.get("estimated_bytes", 0)),
                )
        return None

    # ------------------------------------------------------------------ planning
    def _nodes_for(self, job: JobRecord) -> list[NodeRecord]:
        spec = job.spec()
        nodes = [
            n
            for n in training_candidates(self.store, self.cfg.heartbeat_timeout_s, spec.max_workers)
            if self.tunnel.connected(n.node_id)
        ]
        if spec.island:
            nodes = [n for n in nodes if n.labels.get("island", "default") == spec.island]
        return nodes

    def _memory_budget(self, node: NodeRecord) -> int:
        caps = node.caps()
        budget_mb = caps.available_memory_mb * self.cfg.pipeline_memory_fraction
        budget_mb = max(0.0, budget_mb - self.cfg.pipeline_reserve_mb)
        return int(budget_mb * 1024 * 1024)

    def _allocate(self, job: JobRecord, nodes: list[NodeRecord], stages: int) -> list[StageArtifact]:
        spec = job.spec()
        cfg = model_config(job.model)
        total_layers = stage_layer_count(cfg)
        if stages > total_layers or stages > len(nodes):
            raise PipelinePlanError("more stages than layers or available workers")
        weights = [max(node_score(n), 0.1) for n in nodes[:stages]]
        if stages > 1:
            weights[0] *= 0.9
            weights[-1] *= 0.9
        total_w = sum(weights)
        counts = [max(1, int(total_layers * w / total_w)) for w in weights]
        while sum(counts) > total_layers:
            idx = max((i for i in range(stages) if counts[i] > 1), key=lambda i: counts[i], default=None)
            if idx is None:
                break
            counts[idx] -= 1
        while sum(counts) < total_layers:
            fractions = [total_layers * weights[i] / total_w - counts[i] for i in range(stages)]
            counts[max(range(stages), key=lambda i: fractions[i])] += 1

        def ranges(cs: list[int]) -> list[tuple[int, int]]:
            out, start = [], 0
            for c in cs:
                out.append((start, start + c))
                start += c
            return out

        def estimate(node: NodeRecord, start: int, end: int) -> dict:
            return self._estimate(job, start, end)

        # Move layers away from stages that exceed their host's memory budget.
        for _ in range(max(1, total_layers * stages * 2)):
            rs = ranges(counts)
            over = []
            for i, (start, end) in enumerate(rs):
                est = estimate(nodes[i], start, end)
                if est["total_bytes"] * spec.stage_memory_headroom > self._memory_budget(nodes[i]):
                    over.append(i)
            if not over:
                return self._artifacts(job, nodes, ranges(counts), stages)
            moved = False
            for i in over:
                if counts[i] <= 1:
                    continue
                candidates = [j for j in (i - 1, i + 1) if 0 <= j < stages]
                candidates.sort(
                    key=lambda j: estimate(nodes[j], *ranges(counts)[j])["total_bytes"]
                )
                for j in candidates:
                    trial = list(counts)
                    trial[i] -= 1
                    trial[j] += 1
                    if trial[j] < 1:
                        continue
                    trs = ranges(trial)
                    if (
                        estimate(nodes[j], *trs[j])["total_bytes"] * spec.stage_memory_headroom
                        <= self._memory_budget(nodes[j])
                    ):
                        counts = trial
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break
        rs = ranges(counts)
        details = []
        for i, (start, end) in enumerate(rs):
            est = estimate(nodes[i], start, end)
            required = int(est["total_bytes"] * spec.stage_memory_headroom)
            details.append(
                f"stage {i} layers {start}..{end} needs ~{required / 1e9:.2f} GB, "
                f"host budget ~{self._memory_budget(nodes[i]) / 1e9:.2f} GB"
            )
        raise PipelinePlanError(
            "cannot fit the model into the available stage memory: " + "; ".join(details)
        )

    def _estimate(self, job: JobRecord, start: int, end: int) -> dict:
        spec = job.spec()
        return self._estimate_cfg(
            job, start, end, rank=spec.lora_rank, batch=spec.batch_size, seq=spec.seq_len
        )

    def _estimate_cfg(self, job: JobRecord, start: int, end: int, *, rank: int, batch: int, seq: int) -> dict:
        from ..model.registry import estimate_stage

        return estimate_stage(
            job.model_kind, model_config(job.model), start, end,
            lora_rank=rank, batch_size=batch, seq_len=seq,
        )

    def _artifacts(
        self,
        job: JobRecord,
        nodes: list[NodeRecord],
        ranges: list[tuple[int, int]],
        stages: int,
    ) -> list[StageArtifact]:
        spec = job.spec()
        out = []
        for i, (start, end) in enumerate(ranges):
            est = self._estimate(job, start, end)
            out.append(
                StageArtifact(
                    stage_index=i,
                    node=nodes[i],
                    layer_start=start,
                    layer_end=end,
                    stage_path=Path(),
                    stage_sha256="",
                    optimizer_path=None,
                    optimizer_sha256=None,
                    estimated_bytes=int(est["total_bytes"] * spec.stage_memory_headroom),
                    parameters=int(est["parameters"]),
                    trainable_parameters=int(est["trainable_parameters"]),
                )
            )
        return out

    def _choose_plan(self, job: JobRecord, nodes: list[NodeRecord]) -> list[StageArtifact]:
        spec = job.spec()
        cfg = model_config(job.model)
        layers = stage_layer_count(cfg)
        if not nodes:
            raise PipelinePlanError("no pipeline workers with active tunnels")
        # Reuse a committed layer split when possible.  This preserves per-stage
        # optimizer state across rounds and lets a replacement worker take over a
        # range without resetting the other stages.
        if job.stage_plan:
            ranges = [(int(s["layer_start"]), int(s["layer_end"])) for s in job.stage_plan]
            if len(ranges) <= len(nodes):
                return self._artifacts(job, nodes, ranges, len(ranges))
        # ``pipeline_stages`` is a preference, not a hard requirement: after a
        # worker loss there may be fewer nodes, and a single larger stage can
        # still be the correct (and only) plan.
        preferred = min(spec.pipeline_stages or len(nodes), len(nodes), layers)
        errors = []
        for extra in range(preferred, min(len(nodes), layers) + 1):
            try:
                return self._allocate(job, nodes, extra)
            except PipelinePlanError as exc:
                errors.append(str(exc))
        raise PipelinePlanError(errors[-1] if errors else "no feasible pipeline plan")

    # ------------------------------------------------------------------ artifacts
    def _prepare_stage_files(self, job: JobRecord, plan: list[StageArtifact]) -> None:
        version = job.version
        cfg = model_config(job.model)
        global_path = Path(job.model_path)
        if not global_path.exists():
            raise PipelineRoundError("global model file is missing")
        global_sha = job.model_sha256 or sha256_file(global_path)
        state = None
        out_dir = global_path.parent / "pipeline" / f"v{version:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        committed = job.stage_checkpoints.get(str(version), {})
        tied = bool(
            job.model_kind == "hf"
            and isinstance(cfg, LlamaConfig)
            and cfg.tie_word_embeddings
        )
        for stage in plan:
            record = committed.get(str(stage.stage_index))
            reuse = (
                record is not None
                and int(record.get("layer_start", -1)) == stage.layer_start
                and int(record.get("layer_end", -1)) == stage.layer_end
                and bool(record.get("params_path"))
                and Path(self.store.data_dir / record.get("params_path", "")).exists()
                and not (
                    tied
                    and isinstance(cfg, LlamaConfig)
                    and stage.layer_end == cfg.num_hidden_layers
                )
            )
            if reuse:
                stage.stage_path = self.store.data_dir / record["params_path"]
                stage.stage_sha256 = record["params_sha256"]
            else:
                if state is None:
                    state = load_state(global_path)
                path = out_dir / f"stage_{stage.stage_index}_{stage.layer_start}_{stage.layer_end}.safetensors"
                local_state = stage_state_from_global(
                    job.model_kind, cfg, state, stage.layer_start, stage.layer_end
                )
                # Tied embeddings are physically duplicated on the final stage
                # (two devices cannot share one tensor).  Re-sync the head from
                # the committed embedding at every round so the copies stay equal.
                if tied and isinstance(cfg, LlamaConfig) and stage.layer_end == cfg.num_hidden_layers:
                    local_state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
                meta = {
                    "kind": "xdc-pipeline-stage",
                    "model_kind": job.model_kind,
                    "layer_start": stage.layer_start,
                    "layer_end": stage.layer_end,
                    "global_version": version,
                    "global_sha256": global_sha,
                    "parameters": sum(int(t.numel()) for t in local_state.values()),
                }
                stage.stage_sha256 = save_state(path, local_state, meta)
                path.with_suffix(".json").write_text(
                    json.dumps(meta, sort_keys=True), encoding="utf-8"
                )
                stage.stage_path = path
            if (
                record is not None
                and bool(record.get("optimizer_path"))
                and Path(self.store.data_dir / record["optimizer_path"]).exists()
                and int(record.get("layer_start", -1)) == stage.layer_start
                and int(record.get("layer_end", -1)) == stage.layer_end
            ):
                stage.optimizer_path = self.store.data_dir / record["optimizer_path"]
                stage.optimizer_sha256 = record["optimizer_sha256"]
                continue
            opt_dir = global_path.parent / "pipeline" / "optimizers" / f"v{version:05d}"
            opt_path = opt_dir / f"stage_{stage.stage_index}_{stage.layer_start}_{stage.layer_end}.safetensors"
            if opt_path.exists():
                stage.optimizer_path = opt_path
                stage.optimizer_sha256 = sha256_file(opt_path)
            else:
                stage.optimizer_path = None
                stage.optimizer_sha256 = None

    @staticmethod
    def _add_event(job: JobRecord, level: str, message: str, **data) -> None:
        """Throttle repeated waiting/plan warnings to one per minute."""
        last = job.events[-1] if job.events else None
        if (
            last
            and last.get("message") == message
            and time.time() - float(last.get("ts", 0)) < 60
        ):
            return
        job.add_event(level, message, **data)

    def _make_token(self, node_id: str, path: Path, kind: str) -> str:
        token = secrets.token_urlsafe(24)
        self.stage_tokens[token] = StageToken(
            node_id=node_id, path=path, kind=kind, expires_at=time.time() + 600
        )
        self._expire_tokens()
        return token

    def _expire_tokens(self) -> None:
        now = time.time()
        for token, item in list(self.stage_tokens.items()):
            if item.expires_at < now:
                self.stage_tokens.pop(token, None)

    def _rpc_timeout(self, job: JobRecord) -> float:
        return max(30.0, min(job.spec().round_timeout_s, 900.0))

    async def _call(self, node: NodeRecord, method: str, payload: dict[str, Any], timeout: float) -> Any:
        try:
            return await self.tunnel.call(node.node_id, method, payload, timeout=timeout)
        except TunnelError as exc:
            raise StageWorkerError(node.node_id, int(payload.get("stage_index", -1)), str(exc)) from exc
        except Exception as exc:
            raise StageWorkerError(node.node_id, int(payload.get("stage_index", -1)), str(exc)) from exc

    # ------------------------------------------------------------------ round
    async def _run_round(self, job: JobRecord) -> None:
        spec = job.spec()
        started = time.time()
        rnd = RoundRecord(
            round_id=self.store.new_round_id(),
            index=job.next_round,
            base_version=job.version,
            expected=[],
            deadline=time.time() + spec.round_timeout_s,
        )
        plan: list[StageArtifact] = []
        try:
            nodes = self._nodes_for(job)
            if len(nodes) < spec.min_workers:
                if job.state != "waiting":
                    job.state = "waiting"
                    self._add_event(
                        job,
                        "warning",
                        "waiting for pipeline workers",
                        online=len(nodes),
                        required=spec.min_workers,
                    )
                    self.store.save()
                return
            plan = self._choose_plan(job, nodes)
            if len(plan) < spec.min_workers:
                raise PipelinePlanError(
                    f"only {len(plan)} stage(s) fit; min_workers={spec.min_workers}"
                )
            self._prepare_stage_files(job, plan)
            self._refresh_stage_plan(job, plan)
            rnd.expected = [s.node.node_id for s in plan]
            rnd.pipeline = {
                "stages": [
                    {
                        "stage_index": s.stage_index,
                        "node_id": s.node.node_id,
                        "node_name": s.node.name,
                        "layer_start": s.layer_start,
                        "layer_end": s.layer_end,
                        "stage_path": f"/v1/pipeline/stage/{self._make_token(s.node.node_id, s.stage_path, 'params')}",
                        "stage_sha256": s.stage_sha256,
                        "optimizer_path": (
                            f"/v1/pipeline/optimizer/{self._make_token(s.node.node_id, s.optimizer_path, 'optimizer')}"
                            if s.optimizer_path
                            else None
                        ),
                        "optimizer_sha256": s.optimizer_sha256,
                        "estimated_bytes": s.estimated_bytes,
                        "parameters": s.parameters,
                        "trainable_parameters": s.trainable_parameters,
                    }
                    for s in plan
                ]
            }
            job.current_round = rnd
            if job.state != "running":
                job.state = "running"
            job.add_event(
                "info",
                "pipeline round opened",
                round=rnd.index,
                version=job.version,
                stages=len(plan),
                layers=[[s.layer_start, s.layer_end] for s in plan],
                micro_batches=spec.micro_batches,
            )
            self.store.save()

            log.info("pipeline load stages job=%s round=%s", job.job_id, rnd.round_id)
            await self._load_stages(job, rnd, plan)
            log.info("pipeline begin job=%s round=%s", job.job_id, rnd.round_id)
            await self._begin_all(job, rnd, plan)
            log.info("pipeline sample/train job=%s round=%s", job.job_id, rnd.round_id)
            X, Y = self._sample_batches(job, spec, rnd.index)
            result = await self._train_microbatches(job, rnd, plan, X, Y)
            await self._commit(job, rnd, plan, result, started)
        except PipelinePlanError as exc:
            await self._abort(job, rnd, plan)
            job.current_round = None
            job.state = "waiting"
            self._add_event(job, "warning", "pipeline plan not feasible", error=str(exc))
            self.store.save()
        except Exception as exc:
            log.exception("pipeline round failed job=%s", job.job_id)
            await self._abort(job, rnd, plan)
            job.current_round = None
            job.round_attempts += 1
            job.add_event(
                "error",
                "pipeline round failed",
                round=rnd.index,
                attempt=job.round_attempts,
                error=str(exc),
            )
            if job.round_attempts >= 3:
                self.owner._fail_job(job, f"pipeline round failed after 3 attempts: {exc}")
            else:
                self.store.save()

    async def _load_stages(self, job: JobRecord, rnd: RoundRecord, plan: list[StageArtifact]) -> None:
        spec = job.spec()
        timeout = self._rpc_timeout(job)

        async def one(stage: StageArtifact, wire: dict[str, Any]) -> dict[str, Any]:
            payload = {
                "job_id": job.job_id,
                "stage_index": stage.stage_index,
                "layer_start": stage.layer_start,
                "layer_end": stage.layer_end,
                "model_kind": job.model_kind,
                "model_spec": job.model,
                "stage_url": wire["stage_path"],
                "stage_sha256": stage.stage_sha256,
                "optimizer_url": wire.get("optimizer_path"),
                "optimizer_sha256": stage.optimizer_sha256,
                "base_version": job.version,
                "lr": spec.lr,
                "weight_decay": spec.weight_decay,
                "lora_rank": spec.lora_rank,
                "lora_alpha": spec.lora_alpha,
                "tokenizer": job.tokenizer,
            }
            return await self._call(stage.node, "pipeline_load", payload, timeout)

        results = await asyncio.gather(
            *(one(s, w) for s, w in zip(plan, rnd.pipeline["stages"]))
        )
        for stage, res in zip(plan, results):
            stage.parameters = int(res.get("parameters", stage.parameters))
            stage.trainable_parameters = int(
                res.get("trainable_parameters", stage.trainable_parameters)
            )
            job.stage_plan[stage.stage_index].update(
                {
                    "device": res.get("device", ""),
                    "measured_param_bytes": int(res.get("param_bytes", 0)),
                    "measured_optimizer_bytes": int(res.get("optimizer_bytes", 0)),
                    "measured_total_bytes": int(res.get("measured_total_bytes", 0)),
                    "cached": bool(res.get("cached", False)),
                }
            )
        self.store.save()

    async def _begin_all(self, job: JobRecord, rnd: RoundRecord, plan: list[StageArtifact]) -> None:
        payload = {
            "job_id": job.job_id,
            "round_id": rnd.round_id,
            "base_version": job.version,
        }

        async def one(stage: StageArtifact):
            log.debug("pipeline_begin -> node=%s stage=%s", stage.node.node_id, stage.stage_index)
            result = await self._call(
                stage.node,
                "pipeline_begin",
                {**payload, "stage_index": stage.stage_index},
                self._rpc_timeout(job),
            )
            log.debug("pipeline_begin <- node=%s stage=%s result=%s", stage.node.node_id, stage.stage_index, result)
            return result

        await asyncio.gather(*(one(s) for s in plan))

    def _sample_batches(self, job: JobRecord, spec, round_index: int):
        path = job.data_train_tokens_path
        if not path:
            raise PipelineRoundError("pipeline job has no tokenized training data")
        tokens = read_tokens(path)
        gen = torch.Generator().manual_seed(spec.seed + round_index * 100003)
        X, Y = [], []
        for _ in range(spec.micro_batches):
            x, y = sample_token_batch(tokens, spec.batch_size, spec.seq_len, gen)
            X.append(x)
            Y.append(y)
        return X, Y

    async def _train_microbatches(
        self,
        job: JobRecord,
        rnd: RoundRecord,
        plan: list[StageArtifact],
        X: list[torch.Tensor],
        Y: list[torch.Tensor],
    ) -> RoundResult:
        spec = job.spec()
        timeout = self._rpc_timeout(job)
        stages = len(plan)
        losses: list[float] = []
        wire_bytes = 0
        produced: list[int] = []

        # Forward wavefront: one microbatch traverses every stage in order.
        for mb in range(spec.micro_batches):
            hidden = None
            for i, stage in enumerate(plan):
                payload = {
                    "job_id": job.job_id,
                    "round_id": rnd.round_id,
                    "stage_index": stage.stage_index,
                    "mb_id": mb,
                    "loss_scale": 1.0 / spec.micro_batches,
                    "activation_dtype": spec.activation_dtype,
                }
                if i == 0:
                    payload["input_ids"] = X[mb].tolist()
                else:
                    payload["hidden"] = encode_array(hidden, dtype=spec.activation_dtype)
                    wire_bytes += int(hidden.size) * (2 if spec.activation_dtype == "fp16" else 4)
                if i == stages - 1:
                    payload["targets"] = Y[mb].tolist()
                log.debug("pipeline_forward -> mb=%s stage=%s node=%s", mb, i, stage.node.node_id)
                res = await self._call(stage.node, "pipeline_forward", payload, timeout)
                log.debug("pipeline_forward <- mb=%s stage=%s loss=%s", mb, i, res.get("loss"))
                if i < stages - 1:
                    hidden = decode_array(res["hidden"])
                else:
                    if res.get("loss") is None:
                        raise PipelineRoundError("final stage returned no loss")
                    losses.append(float(res["loss"]))
                    produced.append(int(res.get("tokens", 0)))
            if time.time() > rnd.deadline:
                raise PipelineRoundError("pipeline round exceeded its deadline")

        # Backward wavefront: reverse microbatch order, reverse stage order.
        log.info("pipeline backward start round=%s", rnd.round_id)
        for mb in reversed(range(spec.micro_batches)):
            grad = None
            for i in reversed(range(stages)):
                stage = plan[i]
                payload = {
                    "job_id": job.job_id,
                    "round_id": rnd.round_id,
                    "stage_index": stage.stage_index,
                    "mb_id": mb,
                    "activation_dtype": spec.activation_dtype,
                }
                if i < stages - 1:
                    if grad is None:
                        raise PipelineRoundError("missing hidden gradient")
                    payload["grad_hidden"] = encode_array(grad, dtype=spec.activation_dtype)
                    wire_bytes += int(grad.size) * (2 if spec.activation_dtype == "fp16" else 4)
                log.debug("pipeline_backward -> mb=%s stage=%s", mb, i)
                res = await self._call(stage.node, "pipeline_backward", payload, timeout)
                log.debug("pipeline_backward <- mb=%s stage=%s grad=%s", mb, i, res.get("grad_hidden") is not None)
                if i > 0:
                    grad = decode_array(res["grad_hidden"]) if res.get("grad_hidden") is not None else None

        log.info("pipeline finalize start round=%s", rnd.round_id)
        finalize = await asyncio.gather(
            *(
                self._call(
                    stage.node,
                    "pipeline_finalize",
                    {
                        "job_id": job.job_id,
                        "stage_index": stage.stage_index,
                        "grad_clip": spec.grad_clip,
                    },
                    timeout * 2,
                )
                for stage in plan
            )
        )
        log.info("pipeline finalize done round=%s", rnd.round_id)
        await asyncio.gather(
            *(
                self._call(
                    stage.node,
                    "pipeline_export",
                    {
                        "job_id": job.job_id,
                        "round_id": rnd.round_id,
                        "stage_index": stage.stage_index,
                        "version": job.version + 1,
                    },
                    timeout * 4,
                )
                for stage in plan
            )
        )
        log.info("pipeline export done round=%s", rnd.round_id)
        result = RoundResult(
            metrics=[dict(m) for m in finalize],
            losses=losses,
            tokens=int(spec.micro_batches * spec.batch_size * spec.seq_len),
            activation_bytes=wire_bytes,
            wall_s=time.time() - rnd.started_at,
        )
        for metric in result.metrics:
            idx = int(metric["stage_index"])
            uploads = rnd.pipeline.setdefault("uploads", {})
            for kind in ("params", "optimizer"):
                key = f"{plan[idx].node.node_id}:{kind}"
                if key not in uploads:
                    raise PipelineRoundError(f"stage {idx} did not upload {kind}")
            job.stage_plan[idx].update(
                {
                    "measured_param_bytes": int(metric.get("param_bytes", 0)),
                    "measured_optimizer_bytes": int(metric.get("optimizer_bytes", 0)),
                    "measured_grad_bytes": int(metric.get("grad_bytes", 0)),
                    "measured_total_bytes": int(
                        metric.get("param_bytes", 0)
                        + metric.get("optimizer_bytes", 0)
                        + metric.get("grad_bytes", 0)
                    ),
                    "device": metric.get("device", job.stage_plan[idx].get("device", "")),
                    "measured_parameters": int(metric.get("parameters", 0)),
                    "grad_norm": float(metric.get("grad_norm", 0.0)),
                    "trainable_parameters": int(metric.get("trainable_parameters", 0)),
                }
            )
        return result

    # ------------------------------------------------------------------ commit
    def _assemble_global(self, job: JobRecord, plan: list[StageArtifact], rnd: RoundRecord) -> dict[str, torch.Tensor]:
        base = load_state(job.model_path)
        cfg = model_config(job.model)
        for stage in plan:
            rec = rnd.pipeline["uploads"].get(f"{stage.node.node_id}:params")
            if not rec:
                raise PipelineRoundError(f"missing params upload for stage {stage.stage_index}")
            local = load_state(rec["path"])
            if job.model_kind == "hf":
                stage_mod = LlamaStage(cfg, stage.layer_start, stage.layer_end)  # type: ignore[arg-type]
            else:
                stage_mod = PipelineStage(cfg, stage.layer_start, stage.layer_end)  # type: ignore[arg-type]
            for local_key, tensor in local.items():
                if job.model_kind == "hf":
                    global_key = stage_mod.global_key(local_key)  # type: ignore[attr-defined]
                else:
                    global_key = self._tiny_global_key(stage, local_key)
                if global_key is None:
                    continue
                if global_key in base and tuple(base[global_key].shape) != tuple(tensor.shape):
                    raise PipelineRoundError(f"shape mismatch for {global_key}")
                base[global_key] = tensor.to(torch.float32)
        if (
            job.model_kind == "hf"
            and isinstance(cfg, LlamaConfig)
            and cfg.tie_word_embeddings
        ):
            # Tied checkpoints omit the duplicate head; loaders derive it.
            base.pop("lm_head.weight", None)
        return base

    @staticmethod
    def _tiny_global_key(stage: StageArtifact, local_key: str) -> str | None:
        if local_key in {"tok_emb.weight", "pos_emb.weight", "norm.weight", "norm.bias", "head.weight"}:
            return local_key
        if local_key.startswith("blocks."):
            _, idx, rest = local_key.split(".", 2)
            return f"blocks.{stage.layer_start + int(idx)}.{rest}"
        return None

    async def _commit(
        self,
        job: JobRecord,
        rnd: RoundRecord,
        plan: list[StageArtifact],
        result: RoundResult,
        started: float,
    ) -> None:
        spec = job.spec()
        version = job.version + 1
        new_state = await asyncio.to_thread(self._assemble_global, job, plan, rnd)
        sha = save_state(job.model_path, new_state, {"kind": "xdc-global", "version": str(version)})
        ckpt_dir = Path(job.model_path).parent / "checkpoints"
        ckpt_path = ckpt_dir / f"v{version:05d}.safetensors"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        if ckpt_path.exists():
            ckpt_path.unlink()
        try:
            # save_state uses atomic replace, so the old inode survives in the
            # checkpoint while the global path advances to the new version.
            import os

            os.link(job.model_path, ckpt_path)
        except OSError:
            import shutil

            shutil.copy2(job.model_path, ckpt_path)
        ckpt_sha = sha
        stage_records = {}
        for stage in plan:
            rec = rnd.pipeline["uploads"].get(f"{stage.node.node_id}:optimizer")
            params_rec = rnd.pipeline["uploads"].get(f"{stage.node.node_id}:params")
            if not rec or not params_rec:
                raise PipelineRoundError(f"missing optimizer upload for stage {stage.stage_index}")
            stage.optimizer_path = Path(rec["path"])
            stage.optimizer_sha256 = rec["sha256"]
            stage_records[str(stage.stage_index)] = {
                "layer_start": stage.layer_start,
                "layer_end": stage.layer_end,
                "params_path": str(Path(params_rec["path"]).relative_to(self.store.data_dir)),
                "params_sha256": params_rec["sha256"],
                "optimizer_path": str(Path(rec["path"]).relative_to(self.store.data_dir)),
                "optimizer_sha256": rec["sha256"],
                "param_bytes": int(stage.parameters) * 4,
                "estimated_bytes": stage.estimated_bytes,
            }
        val_loss = None
        if spec.eval_every > 0 and (version % spec.eval_every == 0 or version >= spec.rounds):
            val_loss = await asyncio.to_thread(self._evaluate, job, new_state)
        avg_loss = sum(result.losses) / max(len(result.losses), 1)
        job.version = version
        job.next_round = rnd.index + 1
        job.current_round = None
        job.model_sha256 = sha
        job.latest_checkpoint = str(ckpt_path.relative_to(self.store.data_dir))
        job.tokens_trained += result.tokens
        job.round_attempts = 0
        job.stage_checkpoints[str(version)] = stage_records
        job.checkpoint_history.append(
            {
                "version": version,
                "path": job.latest_checkpoint,
                "sha256": ckpt_sha,
                "size_bytes": ckpt_path.stat().st_size,
                "val_loss": val_loss,
                "created_at": time.time(),
            }
        )
        if val_loss is not None:
            job.best_loss = min(job.best_loss if job.best_loss is not None else val_loss, val_loss)
            job.final_loss = val_loss
        metrics = {
            "round": rnd.index,
            "version": version,
            "workers": [s.node.node_id for s in plan],
            "updates_received": len(plan),
            "updates_used": len(plan),
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "tokens": result.tokens,
            "micro_batches": spec.micro_batches,
            "stages": len(plan),
            "activation_bytes": result.activation_bytes,
            "round_wall_s": time.time() - started,
            "stage_metrics": result.metrics,
            "stage_parameters": [int(s.parameters) for s in plan],
            "stage_measured_bytes": [int(m.get("param_bytes", 0) + m.get("optimizer_bytes", 0) + m.get("grad_bytes", 0)) for m in result.metrics],
            "optimizer_sharded": True,
            "parallelism": "pipeline",
            "timestamp": time.time(),
        }
        job.metrics.append(metrics)
        for stage in plan:
            node = self.store.get_node(stage.node.node_id)
            if node:
                node.accepted_updates += 1
                node.reputation = min(1.0, node.reputation + 0.02)
        job.add_event(
            "info",
            "pipeline round committed",
            round=rnd.index,
            version=version,
            train_loss=avg_loss,
            val_loss=val_loss,
            stages=len(plan),
            activation_mb=round(result.activation_bytes / 1e6, 2),
        )
        self._write_job_meta(job)
        self.owner._prune_checkpoints(job)
        self.store.save()
        if job.next_round >= spec.rounds:
            self.owner._complete(job)

    def _evaluate(self, job: JobRecord, state: dict[str, torch.Tensor]) -> float | None:
        try:
            spec = job.spec()
            cfg = model_config(job.model)
            path = job.data_val_tokens_path
            if not path:
                return None
            tokens = read_tokens(path)
            batches = fixed_token_batches(
                tokens, spec.seq_len, max_batches=spec.eval_batches, seed=spec.seed
            )
            rank = int(job.model.get("lora_rank", 0))
            alpha = float(job.model.get("lora_alpha", 0.0))
            if job.model_kind == "hf":
                model = LlamaModel(cfg)  # type: ignore[arg-type]
                if rank:
                    apply_lora(model.model, r=rank, alpha=alpha)
                model.load_hf_state(state)
            else:
                model = GPT(cfg)  # type: ignore[arg-type]
                if rank:
                    apply_lora(model, r=rank, alpha=alpha, targets=("qkv", "proj", "fc1", "fc2"))
                missing, unexpected = model.load_state_dict(state, strict=False)
                missing = [k for k in missing if "lora_" not in k]
                if missing or unexpected:
                    raise ValueError(f"evaluation state mismatch: {missing[:4]} {unexpected[:4]}")
            model.to(self.cfg.eval_device).eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for x, y in batches:
                    x, y = x.to(self.cfg.eval_device), y.to(self.cfg.eval_device)
                    _, loss = model(x, targets=y)
                    total += float(loss) * int(y.numel())
                    count += int(y.numel())
            return total / max(count, 1)
        except Exception as exc:
            log.warning("pipeline evaluation failed for %s: %s", job.job_id, exc)
            job.add_event("warning", "pipeline evaluation failed", error=str(exc))
            return None

    def _refresh_stage_plan(self, job: JobRecord, plan: list[StageArtifact]) -> None:
        stages = []
        for s in plan:
            stages.append(
                {
                    "stage_index": s.stage_index,
                    "node_id": s.node.node_id,
                    "node_name": s.node.name,
                    "layer_start": s.layer_start,
                    "layer_end": s.layer_end,
                    "parameters": s.parameters,
                    "trainable_parameters": s.trainable_parameters,
                    "estimated_bytes": s.estimated_bytes,
                    "device": s.node.caps().device_name,
                    "backend": s.node.caps().backend,
                }
            )
        job.stage_plan = stages

    async def _abort(self, job: JobRecord, rnd: RoundRecord, plan: list[StageArtifact]) -> None:
        if not plan or not rnd.pipeline.get("stages"):
            return

        async def one(stage: StageArtifact, wire: dict) -> None:
            payload = {
                "job_id": job.job_id,
                "stage_index": stage.stage_index,
                "layer_start": stage.layer_start,
                "layer_end": stage.layer_end,
                "base_version": job.version,
                "model_kind": job.model_kind,
                "model_spec": job.model,
                "stage_url": wire.get("stage_path"),
                "stage_sha256": stage.stage_sha256,
                "optimizer_url": wire.get("optimizer_path"),
                "optimizer_sha256": stage.optimizer_sha256,
                "lr": job.spec().lr,
                "weight_decay": job.spec().weight_decay,
                "lora_rank": job.spec().lora_rank,
                "lora_alpha": job.spec().lora_alpha,
                "tokenizer": job.tokenizer,
            }
            try:
                await self.tunnel.call(
                    stage.node.node_id, "pipeline_abort", payload, timeout=120
                )
            except Exception as exc:
                log.info("pipeline abort failed on %s: %s", stage.node.node_id, exc)

        await asyncio.gather(
            *(one(s, w) for s, w in zip(plan, rnd.pipeline.get("stages", [])))
        )

    def _write_job_meta(self, job: JobRecord) -> None:
        self.owner._write_job_meta(job)
