"""DiLoCo/Decoupled-DiLoCo training coordinator.

Training is a sequence of rounds.  In each round every selected worker starts from
the same committed global weights, performs local AdamW steps on a disjoint data
shard, and uploads a pseudo-gradient.  The coordinator aggregates once a quorum
(plus an adaptive grace window) has reported, or once the round deadline expires.
Stragglers do not hold the whole job hostage; a configurable minimum worker count
decides whether to commit the round or retry with replacements.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import traceback
from pathlib import Path

import torch

from ..config import CoordinatorConfig
from ..model.checkpoint import (
    load_state,
    save_state,
    sha256_bytes,
    save_json,
)
from ..model.data import shard_bounds, split_train_val
from ..model.gpt import GPT, ModelConfig
from ..model.train import aggregate_deltas, evaluate_loss
from ..protocol import (
    Assignment,
    DatasetInfo,
    JobCreate,
    JobView,
    TrainingSpec,
    UpdateAccepted,
    UpdateMeta,
)
from .scheduler import training_candidates
from .state import JobRecord, NodeRecord, RoundRecord, StateStore
from .tunnel import TunnelHub

log = logging.getLogger("xdc.train")


class TrainingEngine:
    def __init__(self, store: StateStore, tunnel: TunnelHub, cfg: CoordinatorConfig):
        self.store = store
        self.tunnel = tunnel
        self.cfg = cfg
        self.task: asyncio.Task | None = None
        self.stopping = False
        self._last_wait_note: dict[str, float] = {}
        self._job_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.task = asyncio.create_task(self._loop(), name="xdc-training-loop")

    async def stop(self) -> None:
        self.stopping = True
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    def lock_for(self, job_id: str) -> asyncio.Lock:
        return self._job_locks.setdefault(job_id, asyncio.Lock())

    async def _loop(self) -> None:
        while not self.stopping:
            try:
                for job in self.store.list_jobs():
                    if job.state in {"running", "waiting"}:
                        # Serialize round decisions with uploads for the same
                        # job.  Upload validation runs in a thread; without the
                        # lock a commit could race a late safetensors parse.
                        async with self.lock_for(job.job_id):
                            await self._advance(job)
            except Exception as exc:  # keep coordinator alive; job has its own errors
                log.exception("training loop error: %s", exc)
            await asyncio.sleep(self.cfg.job_tick_s)

    # ------------------------------------------------------------------ job API
    def create_job(self, req: JobCreate) -> JobRecord:
        with self.store.lock:
            ds = self.store.datasets.get(req.dataset_id)
            if not ds or not Path(ds.path).exists():
                raise ValueError(f"dataset {req.dataset_id} not found")
            mcfg = ModelConfig.from_dict(req.model)
            spec = req.training
            if spec.seq_len > mcfg.block_size:
                raise ValueError("training.seq_len exceeds model.block_size")
            raw = Path(ds.path).read_bytes()
            train_data, val_data = split_train_val(raw, spec.val_fraction, min_val_bytes=max(spec.seq_len * 8, 512))
            if len(train_data) < spec.seq_len + 2:
                raise ValueError("dataset too small after train/validation split")
            job_id = self.store.new_job_id()
            jdir = self.store.job_dir(job_id)
            train_path = jdir / "train.bin"
            val_path = jdir / "val.bin"
            train_path.write_bytes(train_data)
            val_path.write_bytes(val_data)
            model_path = jdir / "global.safetensors"
            optimizer_path = jdir / "outer_optimizer.safetensors"

            if req.init_checkpoint_job:
                src = self.store.jobs.get(req.init_checkpoint_job)
                if not src or not src.latest_checkpoint:
                    raise ValueError("init_checkpoint_job has no checkpoint")
                src_path = self.store.data_dir / src.latest_checkpoint
                state = load_state(src_path)
                gpt = GPT(mcfg)
                gpt.load_state_dict(state, strict=True)
            else:
                torch.manual_seed(spec.seed)
                gpt = GPT(mcfg)
            params = gpt.num_parameters()
            init_path = jdir / "init.safetensors"
            save_state(init_path, gpt.state_dict(), {"kind": "xdc-init", "version": "0"})
            model_sha = save_state(model_path, gpt.state_dict(), {"kind": "xdc-global", "version": "0"})
            job = JobRecord(
                job_id=job_id,
                name=req.name,
                state="queued",
                created_at=time.time(),
                updated_at=time.time(),
                model=mcfg.to_dict(),
                training=spec.model_dump(mode="json"),
                dataset_id=ds.dataset_id,
                dataset_name=ds.name,
                dataset_sha256=sha256_bytes(train_data),
                data_train_path=str(train_path),
                data_val_path=str(val_path),
                model_path=str(model_path),
                optimizer_path=str(optimizer_path),
                init_checkpoint=str(init_path.relative_to(self.store.data_dir)),
                model_sha256=model_sha,
                parameters=params,
                init_checkpoint_job=req.init_checkpoint_job,
            )
            job.add_event("info", "job created", parameters=params, dataset=ds.name)
            initial_loss = self._evaluate(job, train=False)
            job.initial_loss = initial_loss
            job.best_loss = initial_loss
            self.store.create_job(job)
            if req.auto_start:
                job.state = "running"
                job.started_at = time.time()
                job.add_event("info", "training started")
            self.store.save()
            return job

    def start_job(self, job_id: str) -> JobRecord:
        job = self._job(job_id)
        if job.state in {"completed", "cancelled"}:
            raise ValueError(f"cannot start job in state {job.state}")
        if job.state == "running":
            return job
        job.state = "running"
        job.error = None
        job.traceback = None
        job.started_at = job.started_at or time.time()
        job.add_event("info", "training started/resumed", version=job.version)
        self.store.save()
        return job

    def pause_job(self, job_id: str) -> JobRecord:
        job = self._job(job_id)
        if job.state == "running":
            job.state = "paused"
            job.current_round = None
            job.add_event("warning", "training paused; in-flight round discarded")
            self.store.save()
        return job

    def cancel_job(self, job_id: str) -> JobRecord:
        job = self._job(job_id)
        if job.state not in {"completed", "cancelled"}:
            job.state = "cancelled"
            job.current_round = None
            job.completed_at = time.time()
            job.add_event("warning", "training cancelled")
            self.store.save()
        return job

    def _job(self, job_id: str) -> JobRecord:
        job = self.store.get_job(job_id)
        if not job:
            raise KeyError(f"job {job_id} not found")
        return job

    def job_view(self, job: JobRecord) -> JobView:
        ds = self.store.datasets.get(job.dataset_id)
        dsinfo = ds.info() if ds else DatasetInfo(
            dataset_id=job.dataset_id,
            name=job.dataset_name,
            size_bytes=0,
            sha256=job.dataset_sha256,
            created_at=0,
        )
        return JobView(
            job_id=job.job_id,
            name=job.name,
            state=job.state,
            created_at=job.created_at,
            updated_at=job.updated_at,
            model=job.model,
            training=job.spec(),
            dataset=dsinfo,
            current_round=job.current_round.view() if job.current_round else None,
            next_round=job.next_round,
            version=job.version,
            initial_loss=job.initial_loss,
            final_loss=job.final_loss,
            best_loss=job.best_loss,
            latest_checkpoint=job.latest_checkpoint,
            initial_checkpoint=job.init_checkpoint,
            error=job.error,
            metrics=job.metrics[-100:],
            events=job.events[-100:],
            tokens_trained=job.tokens_trained,
            parameters=job.parameters,
            traceback=job.traceback,
        )

    # ------------------------------------------------------------------ round flow
    async def _advance(self, job: JobRecord) -> None:
        spec = job.spec()
        if job.next_round >= spec.rounds:
            self._complete(job)
            return
        if job.current_round is None:
            await self._open_round(job, spec)
            return
        rnd = job.current_round
        now = time.time()
        # Detect lost workers without waiting for the full timeout.  A live
        # tunnel counts as liveness: a busy consumer machine can delay its
        # heartbeat loop for many seconds while it is still training.  Treating
        # that as death throws away real work and can bounce a round for minutes.
        for node_id in list(rnd.expected):
            if node_id in rnd.submitted or node_id in rnd.failed:
                continue
            node = self.store.get_node(node_id)
            if node is None or not self._node_alive(node):
                if node_id not in rnd.failed:
                    rnd.failed.append(node_id)
                    rnd.failed_at[node_id] = now
                    if node:
                        node.failed_rounds += 1
                        node.reputation = max(0.0, node.reputation - 0.02)
                    job.add_event("warning", "worker lost during round", node=node_id, round=rnd.index)
        accounted = len(rnd.submitted) + len(rnd.failed)
        quorum = self._quorum(job, rnd)
        if len(rnd.submitted) >= quorum and rnd.quorum_reached_at is None:
            rnd.quorum_reached_at = now
        grace_done = (
            rnd.quorum_reached_at is not None
            and (now - rnd.quorum_reached_at) >= spec.grace_s
        )
        all_done = accounted >= len(rnd.expected)
        deadline_done = now >= rnd.deadline
        should_commit = False
        if len(rnd.submitted) >= quorum and (all_done or grace_done):
            should_commit = True
        # Failure-aware quorum: once every expected worker has either submitted or
        # been declared lost, do not wait for the full deadline when enough live
        # updates exist to satisfy min_workers.  This is the practical fault-
        # tolerance path used by consumer devices that sleep or lose Wi-Fi.
        if all_done and len(rnd.submitted) >= spec.min_workers and rnd.failed:
            should_commit = True
        # If every remaining worker was declared lost and there are not enough
        # updates to satisfy min_workers, the round cannot commit.  Wait a short
        # recovery window so an in-flight upload from a misclassified worker can
        # still be re-admitted, then retry instead of stalling until the deadline.
        if (
            not should_commit
            and all_done
            and rnd.failed
            and len(rnd.submitted) < spec.min_workers
            and now - max(rnd.failed_at.values(), default=rnd.started_at) >= self._recovery_s()
        ):
            await self._retry_round(job, rnd, "all remaining workers lost with too few updates")
            return
        if deadline_done:
            if len(rnd.submitted) >= spec.min_workers:
                should_commit = True
            else:
                await self._retry_round(job, rnd, "deadline with too few updates")
                return
        if not should_commit:
            return
        if not rnd.submitted:
            await self._retry_round(job, rnd, "no updates")
            return
        rnd.aggregated = True
        self.store.save()
        try:
            await asyncio.to_thread(self._commit_round, job, rnd)
        except Exception as exc:
            log.exception("aggregation failed for %s", job.job_id)
            job.error = f"aggregation failed: {exc}"
            job.traceback = traceback.format_exc()
            job.add_event("error", "aggregation failed", error=str(exc))
            await self._retry_round(job, rnd, f"aggregation error: {exc}")
        else:
            self.store.save()

    async def _open_round(self, job: JobRecord, spec: TrainingSpec) -> None:
        candidates = training_candidates(self.store, self.cfg.heartbeat_timeout_s, spec.max_workers)
        if len(candidates) < spec.min_workers:
            if job.state != "waiting":
                job.state = "waiting"
                job.add_event(
                    "warning",
                    "waiting for workers",
                    online=len(candidates),
                    required=spec.min_workers,
                )
                self.store.save()
            return
        if job.state == "waiting":
            job.state = "running"
        train_blob = Path(job.data_train_path).read_bytes()
        total = len(train_blob)
        if total <= spec.seq_len + 1:
            self._fail_job(job, "training split too small for configured seq_len")
            return
        expected = [n.node_id for n in candidates]
        rnd = RoundRecord(
            round_id=self.store.new_round_id(),
            index=job.next_round,
            base_version=job.version,
            expected=expected,
            deadline=time.time() + spec.round_timeout_s,
        )
        for rank, node in enumerate(candidates):
            start, end = shard_bounds(total, rank, len(candidates))
            if end - start <= spec.seq_len + 1:
                # A tiny tail shard would be unusable; let it see the whole split
                # rather than fail the round.  Data is still never sent to more
                # than the workers selected for this round.
                start, end = 0, total
            rnd.assignments[node.node_id] = {
                "shard_start": start,
                "shard_end": end,
                "shard_sha256": sha256_bytes(train_blob[start:end]),
                "shard_index": rank,
                "world_size": len(candidates),
                "dispatched_at": None,
            }
        job.current_round = rnd
        job.add_event(
            "info",
            "round opened",
            round=rnd.index,
            version=rnd.base_version,
            workers=expected,
            quorum=self._quorum(job, rnd),
        )
        self.store.save()

    def _quorum(self, job: JobRecord, rnd: RoundRecord) -> int:
        spec = job.spec()
        if spec.algorithm == "diloco":
            return len(rnd.expected)
        q = math.ceil(len(rnd.expected) * spec.quorum_fraction)
        q = max(q, spec.min_workers)
        return min(q, len(rnd.expected))

    async def _retry_round(self, job: JobRecord, rnd: RoundRecord, reason: str) -> None:
        job.round_attempts += 1
        job.add_event("warning", "round retry", round=rnd.index, reason=reason, attempt=job.round_attempts)
        job.current_round = None
        for node_id in rnd.failed:
            node = self.store.get_node(node_id)
            if node:
                node.failed_rounds += 1
        if job.round_attempts >= 3:
            self._fail_job(job, f"round {rnd.index} failed after 3 attempts: {reason}")
            return
        self.store.save()

    def _commit_round(self, job: JobRecord, rnd: RoundRecord) -> None:
        """Runs in a worker thread: CPU/disk heavy aggregation + evaluation."""
        base = load_state(job.model_path)
        updates = []
        weights = []
        used_nodes = []
        tokens = 0
        losses = []
        local_times = []
        for node_id, path in rnd.submitted.items():
            try:
                delta = load_state(path)
            except Exception as exc:
                self._reject_in_commit(job, node_id, f"unreadable update: {exc}")
                continue
            updates.append(delta)
            meta = rnd.assignments.get(node_id, {}).get("meta", {})
            tokens += int(meta.get("tokens", 1))
            weights.append(max(1, int(meta.get("tokens", 1))))
            losses.append(float(meta.get("loss", 0.0)))
            local_times.append(float(meta.get("elapsed_s", 0.0)))
            used_nodes.append(node_id)
        if not updates:
            raise RuntimeError("no valid updates in round")
        prev_mom = None
        if Path(job.optimizer_path).exists():
            try:
                prev_mom = load_state(job.optimizer_path)
            except Exception:
                prev_mom = None
        spec = job.spec()
        agg_started = time.perf_counter()
        new_state, new_mom, stats = aggregate_deltas(
            base,
            updates,
            [float(w) for w in weights],
            outer_lr=spec.outer_lr,
            momentum=spec.outer_momentum,
            nesterov=spec.nesterov,
            momentum_state=prev_mom,
            max_norm_ratio=spec.max_update_norm_ratio,
        )
        aggregate_s = time.perf_counter() - agg_started
        version = job.version + 1
        sha = save_state(job.model_path, new_state, {"kind": "xdc-global", "version": str(version)})
        save_state(job.optimizer_path, new_mom, {"kind": "xdc-outer-momentum", "version": str(version)})
        ckpt_dir = Path(job.model_path).parent / "checkpoints"
        ckpt_path = ckpt_dir / f"v{version:05d}.safetensors"
        ckpt_sha = save_state(ckpt_path, new_state, {"kind": "xdc-checkpoint", "version": str(version)})
        rel_ckpt = str(ckpt_path.relative_to(self.store.data_dir))
        val_loss = None
        if spec.eval_every > 0 and (version % spec.eval_every == 0 or version >= spec.rounds):
            val_loss = self._evaluate(job, train=False, state=new_state)
        job.version = version
        job.next_round = rnd.index + 1
        job.current_round = None
        job.model_sha256 = sha
        job.latest_checkpoint = rel_ckpt
        job.tokens_trained += tokens
        job.round_attempts = 0
        job.checkpoint_history.append(
            {
                "version": version,
                "path": rel_ckpt,
                "sha256": ckpt_sha,
                "size_bytes": ckpt_path.stat().st_size,
                "val_loss": val_loss,
                "created_at": time.time(),
            }
        )
        if val_loss is not None:
            job.best_loss = min(job.best_loss if job.best_loss is not None else val_loss, val_loss)
            job.final_loss = val_loss
        metric = {
            "round": rnd.index,
            "version": version,
            "workers": used_nodes,
            "updates_received": stats.get("updates_received"),
            "updates_used": stats.get("updates_used"),
            "dropped": stats.get("dropped_indices", []),
            "norm": (sum(float(v) ** 2 for v in stats.get("norms", [])) ** 0.5),
            "train_loss": sum(losses) / max(len(losses), 1),
            "val_loss": val_loss,
            "tokens": tokens,
            "aggregate_s": aggregate_s,
            "local_elapsed_max_s": max(local_times, default=0.0),
            "local_elapsed_sum_s": sum(local_times),
            "native": stats.get("native", False),
            "timestamp": time.time(),
        }
        job.metrics.append(metric)
        for node_id in used_nodes:
            node = self.store.get_node(node_id)
            if node:
                node.accepted_updates += 1
                node.reputation = min(1.0, node.reputation + 0.02)
        job.add_event(
            "info",
            "round committed",
            round=rnd.index,
            version=version,
            val_loss=val_loss,
            train_loss=metric["train_loss"],
            workers=len(used_nodes),
        )
        self._prune_checkpoints(job)
        self._write_job_meta(job)
        if job.next_round >= job.spec().rounds:
            self._complete(job)
        self.store.save()

    def _reject_in_commit(self, job: JobRecord, node_id: str, reason: str) -> None:
        node = self.store.get_node(node_id)
        if node:
            node.rejected_updates += 1
            node.reputation = max(0.0, node.reputation - 0.2)
            node.last_error = reason[:500]
        job.add_event("error", "invalid worker update", node=node_id, reason=reason)

    def _evaluate(self, job: JobRecord, *, train: bool = False, state=None) -> float | None:
        try:
            spec = job.spec()
            cfg = ModelConfig.from_dict(job.model)
            if state is None:
                state = load_state(job.model_path)
            data = Path(job.data_train_path if train else job.data_val_path).read_bytes()
            loss = evaluate_loss(
                state,
                cfg,
                data,
                batch_size=spec.batch_size,
                seq_len=spec.seq_len,
                max_batches=spec.eval_batches,
                seed=spec.seed,
                device=self.cfg.eval_device,
            )
            return float(loss)
        except Exception as exc:
            log.warning("evaluation failed for %s: %s", job.job_id, exc)
            job.add_event("warning", "evaluation failed", error=str(exc))
            return None

    def _complete(self, job: JobRecord) -> None:
        if job.state == "completed":
            return
        job.state = "completed"
        job.completed_at = time.time()
        job.current_round = None
        if job.final_loss is None:
            job.final_loss = self._evaluate(job, train=False)
        job.add_event("info", "training completed", version=job.version, final_loss=job.final_loss)
        self._write_job_meta(job)
        self.store.save()

    def _fail_job(self, job: JobRecord, error: str) -> None:
        job.state = "failed"
        job.error = error
        job.current_round = None
        job.add_event("error", "job failed", error=error)
        self.store.save()

    def _recovery_s(self) -> float:
        """How long to keep a round open for late uploads after all workers fail.

        Short enough that a dead laptop does not stall a job, long enough that a
        heartbeat blip on a loaded machine does not throw away a completed round.
        """
        return max(5.0, min(30.0, self.cfg.heartbeat_timeout_s * 2.0))

    def _node_alive(self, node: NodeRecord) -> bool:
        if node.banned:
            return False
        # A live outbound tunnel proves the worker process is connected even if
        # its heartbeat loop was delayed by CPU pressure or a sleeping display.
        try:
            if self.tunnel.connected(node.node_id):
                return True
        except Exception:
            pass
        return (time.time() - node.last_seen) <= self.cfg.heartbeat_timeout_s

    def _prune_checkpoints(self, job: JobRecord) -> None:
        keep = self.cfg.checkpoint_keep
        if keep <= 0 or len(job.checkpoint_history) <= keep:
            return
        removed = job.checkpoint_history[:-keep]
        job.checkpoint_history = job.checkpoint_history[-keep:]
        for item in removed:
            if item["path"] == job.latest_checkpoint:
                continue
            try:
                (self.store.data_dir / item["path"]).unlink(missing_ok=True)
            except Exception:
                log.warning("could not prune checkpoint %s", item["path"])

    def _write_job_meta(self, job: JobRecord) -> None:
        path = Path(job.model_path).parent / "job.json"
        save_json(
            path,
            {
                "job_id": job.job_id,
                "name": job.name,
                "state": job.state,
                "model": job.model,
                "training": job.training,
                "version": job.version,
                "next_round": job.next_round,
                "initial_loss": job.initial_loss,
                "final_loss": job.final_loss,
                "best_loss": job.best_loss,
                "parameters": job.parameters,
                "tokens_trained": job.tokens_trained,
                "latest_checkpoint": job.latest_checkpoint,
                "checkpoint_sha256": job.model_sha256,
            },
        )

    # ------------------------------------------------------------------ worker API
    def assignment_for(self, node_id: str) -> Assignment | None:
        now = time.time()
        for job in self.store.list_jobs():
            if job.state != "running" or not job.current_round:
                continue
            rnd = job.current_round
            if node_id not in rnd.assignments or node_id in rnd.submitted or node_id in rnd.failed:
                continue
            if rnd.aggregated or now > rnd.deadline:
                continue
            a = rnd.assignments[node_id]
            spec = job.spec()
            start = int(a["shard_start"])
            end = int(a["shard_end"])
            return Assignment(
                job_id=job.job_id,
                job_name=job.name,
                round_id=rnd.round_id,
                round_index=rnd.index,
                base_version=rnd.base_version,
                model_path=f"/v1/jobs/{job.job_id}/weights?version={rnd.base_version}",
                model_sha256=job.model_sha256,
                model_spec=job.model,
                dataset_path=f"/v1/jobs/{job.job_id}/dataset",
                dataset_id=job.dataset_id,
                dataset_sha256=job.dataset_sha256,
                shard_sha256=a.get("shard_sha256", ""),
                shard_start=start,
                shard_end=end,
                shard_index=int(a["shard_index"]),
                world_size=int(a["world_size"]),
                local_steps=spec.local_steps,
                batch_size=spec.batch_size,
                seq_len=spec.seq_len,
                lr=spec.lr,
                weight_decay=spec.weight_decay,
                grad_clip=spec.grad_clip,
                seed=(spec.seed + rnd.index * 100003 + int(a["shard_index"]) * 7919) & 0x7FFFFFFF,
                max_update_norm_ratio=spec.max_update_norm_ratio,
                deadline=rnd.deadline,
                compression=spec.compression,
            )

    def accept_update(self, node_id: str, meta: UpdateMeta, body: bytes) -> UpdateAccepted:
        job = self.store.get_job(meta.job_id)
        if job is None:
            raise KeyError("job not found")
        rnd = job.current_round
        if job.state != "running" or rnd is None or rnd.round_id != meta.round_id:
            raise ValueError("round is not active")
        if node_id not in rnd.expected:
            raise PermissionError("node was not assigned to this round")
        if node_id in rnd.submitted:
            raise ValueError("duplicate update")
        if rnd.aggregated:
            raise ValueError("round already aggregated")
        if node_id in rnd.failed:
            # The worker was declared lost while its update was still in flight
            # (common on a loaded/sleeping consumer machine).  The round is still
            # open and the update is bound to this round/version, so re-admit it
            # rather than discarding real work.
            rnd.failed.remove(node_id)
            rnd.failed_at.pop(node_id, None)
            job.add_event("info", "late update re-admitted", round=rnd.index, node=node_id)
        if time.time() > rnd.deadline + 5:
            raise ValueError("update arrived after deadline")
        if meta.base_version != rnd.base_version:
            raise ValueError("update base version mismatch")
        if not body:
            raise ValueError("empty update body")
        from ..model.checkpoint import deserialize_state_bytes

        try:
            delta = deserialize_state_bytes(body)
        except Exception as exc:
            raise ValueError(f"invalid safetensors payload: {exc}") from exc
        base = load_state(job.model_path)
        if set(delta.keys()) != set(base.keys()):
            raise ValueError("update parameter names do not match global model")
        for key, tensor in delta.items():
            if tuple(tensor.shape) != tuple(base[key].shape):
                raise ValueError(f"shape mismatch for {key}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"non-finite update tensor {key}")
        # Enforce an absolute sanity ceiling even when there are only two workers
        # (the adaptive median filter needs three).
        norm = math.sqrt(sum(float((t.float() ** 2).sum()) for t in delta.values()))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("invalid update norm")
        max_allowed = 1e6
        if norm > max_allowed:
            raise ValueError("update norm exceeds safety ceiling")
        update_dir = Path(job.model_path).parent / "updates" / rnd.round_id
        update_dir.mkdir(parents=True, exist_ok=True)
        path = update_dir / f"{node_id}.safetensors"
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        import os

        os.replace(tmp, path)
        rnd.submitted[node_id] = str(path)
        rnd.assignments[node_id]["meta"] = {
            "loss": meta.loss,
            "tokens": meta.tokens,
            "steps": meta.steps,
            "elapsed_s": meta.elapsed_s,
            "device": meta.device,
            "update_norm": meta.update_norm,
            "compression": meta.compressed,
        }
        node = self.store.get_node(node_id)
        if node:
            node.reputation = min(1.0, node.reputation + 0.01)
        job.add_event(
            "info",
            "update received",
            round=rnd.index,
            node=node_id,
            loss=meta.loss,
            tokens=meta.tokens,
            norm=meta.update_norm,
        )
        self.store.save()
        return UpdateAccepted(job_id=job.job_id, round_id=rnd.round_id, status="accepted")

    def active_models(self) -> dict[str, dict]:
        out = {}
        for job in self.store.list_jobs():
            out[job.job_id] = {
                "job_id": job.job_id,
                "state": job.state,
                "version": job.version,
                "latest_checkpoint": job.latest_checkpoint,
                "model": job.model,
            }
        return out
