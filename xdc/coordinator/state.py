"""Durable coordinator state (nodes, jobs, datasets, checkpoints).

A single JSON document is enough for one coordinator process and keeps recovery
auditable.  Writes are atomic (fsync + rename).  Tensor data never goes into JSON.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..model.checkpoint import load_json, save_json, sha256_bytes
from ..protocol import Capabilities, DatasetInfo, TrainingSpec


def _now() -> float:
    return time.time()


@dataclass
class NodeRecord:
    node_id: str
    name: str
    public_key: str
    capabilities: dict[str, Any]
    labels: dict[str, str] = field(default_factory=dict)
    first_seen: float = field(default_factory=_now)
    last_seen: float = field(default_factory=_now)
    tunnel_token_hash: str = ""
    tunnel_connected: bool = False
    reputation: float = 0.5
    accepted_updates: int = 0
    rejected_updates: int = 0
    failed_rounds: int = 0
    banned: bool = False
    last_error: str | None = None
    active_sessions: int = 0
    training_jobs: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "NodeRecord":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def caps(self) -> Capabilities:
        return Capabilities(**self.capabilities)


@dataclass
class DatasetRecord:
    dataset_id: str
    name: str
    path: str
    size_bytes: int
    sha256: str
    created_at: float
    text_lines: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetRecord":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            dataset_id=self.dataset_id,
            name=self.name,
            size_bytes=self.size_bytes,
            sha256=self.sha256,
            created_at=self.created_at,
            text_lines=self.text_lines,
        )


@dataclass
class RoundRecord:
    round_id: str
    index: int
    base_version: int
    expected: list[str]
    submitted: dict[str, str] = field(default_factory=dict)  # node -> update path
    failed: list[str] = field(default_factory=list)
    failed_at: dict[str, float] = field(default_factory=dict)
    assignments: dict[str, dict] = field(default_factory=dict)
    deadline: float = 0.0
    started_at: float = field(default_factory=_now)
    quorum_reached_at: float | None = None
    aggregated: bool = False
    attempt: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "RoundRecord":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def view(self):
        from ..protocol import RoundView

        return RoundView(
            round_id=self.round_id,
            index=self.index,
            base_version=self.base_version,
            expected=self.expected,
            submitted=list(self.submitted.keys()),
            failed=self.failed,
            deadline=self.deadline,
            aggregated=self.aggregated,
        )


@dataclass
class JobRecord:
    job_id: str
    name: str
    state: str
    created_at: float
    updated_at: float
    model: dict[str, Any]
    training: dict[str, Any]
    dataset_id: str
    dataset_name: str
    dataset_sha256: str
    data_train_path: str
    data_val_path: str
    model_path: str
    optimizer_path: str
    init_checkpoint: str | None = None
    model_sha256: str = ""
    parameters: int = 0
    next_round: int = 0
    version: int = 0
    current_round: RoundRecord | None = None
    initial_loss: float | None = None
    final_loss: float | None = None
    best_loss: float | None = None
    latest_checkpoint: str | None = None
    checkpoint_history: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    traceback: str | None = None
    metrics: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    tokens_trained: int = 0
    init_checkpoint_job: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    round_attempts: int = 0
    restart_count: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "JobRecord":
        valid = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in valid}
        if clean.get("current_round"):
            clean["current_round"] = RoundRecord.from_dict(clean["current_round"])
        return cls(**clean)

    def spec(self) -> TrainingSpec:
        return TrainingSpec(**self.training)

    def add_event(self, level: str, message: str, **extra: Any) -> None:
        self.events.append({"ts": _now(), "level": level, "message": message, **extra})
        if len(self.events) > 500:
            self.events = self.events[-500:]
        self.updated_at = _now()


class StateStore:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "state.json"
        self.lock = threading.RLock()
        self.nodes: dict[str, NodeRecord] = {}
        self.jobs: dict[str, JobRecord] = {}
        self.datasets: dict[str, DatasetRecord] = {}
        self.schema = 1
        self._last_save = 0.0
        self.load()

    def load(self) -> None:
        with self.lock:
            raw = load_json(self.path, default={}) or {}
            self.nodes = {k: NodeRecord.from_dict(v) for k, v in (raw.get("nodes") or {}).items()}
            # No WebSocket survives a coordinator restart; do not report a stale
            # tunnel as live.
            for node in self.nodes.values():
                node.tunnel_connected = False
            self.jobs = {k: JobRecord.from_dict(v) for k, v in (raw.get("jobs") or {}).items()}
            self.datasets = {k: DatasetRecord.from_dict(v) for k, v in (raw.get("datasets") or {}).items()}
            # A previous process may have died mid-round.  Drop the volatile round;
            # the engine will re-issue it from the last committed global weights.
            for job in self.jobs.values():
                if job.state in {"running", "waiting"}:
                    if job.current_round is not None:
                        job.current_round = None
                        job.add_event("warning", "coordinator restarted; uncommitted round discarded")
                    job.state = "running"
                    job.restart_count += 1

    def save(self, min_interval: float = 0.0) -> None:
        """Persist state atomically.

        ``min_interval`` is for high-frequency liveness updates (heartbeats):
        skipping an fsync does not change correctness because the next save
        flushes the same in-memory state.  Round/commit writes always use 0.
        """
        with self.lock:
            if min_interval > 0:
                now = time.monotonic()
                if now - self._last_save < min_interval:
                    return
            data = {
                "schema": self.schema,
                "saved_at": _now(),
                "nodes": {k: asdict(v) for k, v in self.nodes.items()},
                "jobs": {k: asdict(v) for k, v in self.jobs.items()},
                "datasets": {k: asdict(v) for k, v in self.datasets.items()},
            }
            save_json(self.path, data)
            self._last_save = time.monotonic()

    def upsert_node(self, record: NodeRecord) -> None:
        with self.lock:
            self.nodes[record.node_id] = record
            self.save()

    def get_node(self, node_id: str) -> NodeRecord | None:
        with self.lock:
            return self.nodes.get(node_id)

    def list_nodes(self) -> list[NodeRecord]:
        with self.lock:
            return list(self.nodes.values())

    def mark_node_error(self, node_id: str, error: str) -> None:
        with self.lock:
            node = self.nodes.get(node_id)
            if node:
                node.last_error = error[:500]
                node.reputation = max(0.0, node.reputation - 0.02)

    def reward_node(self, node_id: str, delta: float) -> None:
        with self.lock:
            node = self.nodes.get(node_id)
            if node:
                node.reputation = min(1.0, max(0.0, node.reputation + delta))

    def save_dataset(self, name: str, data: bytes, text_lines: int | None = None) -> DatasetRecord:
        sha = sha256_bytes(data)
        dataset_id = "ds-" + sha[:20]
        with self.lock:
            existing = self.datasets.get(dataset_id)
            if existing and Path(existing.path).exists():
                return existing
            path = self.data_dir / "datasets" / f"{dataset_id}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
            rec = DatasetRecord(
                dataset_id=dataset_id,
                name=name,
                path=str(path),
                size_bytes=len(data),
                sha256=sha,
                created_at=_now(),
                text_lines=text_lines,
            )
            self.datasets[dataset_id] = rec
            self.save()
            return rec

    def job_dir(self, job_id: str) -> Path:
        path = self.data_dir / "jobs" / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_job(self, job: JobRecord) -> None:
        with self.lock:
            if job.job_id in self.jobs:
                raise ValueError("job already exists")
            self.jobs[job.job_id] = job
            self.save()

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.lock:
            return self.jobs.get(job_id)

    def list_jobs(self) -> list[JobRecord]:
        with self.lock:
            return list(self.jobs.values())

    def touch(self, job: JobRecord, save: bool = False) -> None:
        job.updated_at = _now()
        if save:
            self.save()

    def delete_round_updates(self, job_id: str, round_id: str) -> None:
        # Kept for diagnostics by default; retention is bounded by checkpoint_keep
        # only for checkpoints, not updates.  Updates are small in the demo model.
        pass

    @staticmethod
    def new_job_id() -> str:
        return "job-" + uuid.uuid4().hex[:12]

    @staticmethod
    def new_round_id() -> str:
        return "rnd-" + uuid.uuid4().hex[:12]
