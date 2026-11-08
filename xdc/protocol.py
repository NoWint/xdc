"""Wire models shared by coordinator, worker and CLI."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .crypto import canonical_json


class Capabilities(BaseModel):
    platform: str = ""
    machine: str = ""
    hostname: str = ""
    python: str = ""
    torch: str = ""
    backend: str = "cpu"
    device_name: str = ""
    cpu_model: str = ""
    cpu_count: int = 0
    total_memory_mb: int = 0
    available_memory_mb: int = 0
    gpu_memory_mb: int = 0
    compute_score: float = 1.0
    backends: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class NodeRegistration(BaseModel):
    node_id: str
    name: str
    public_key: str
    capabilities: Capabilities
    labels: dict[str, str] = Field(default_factory=dict)
    timestamp: int
    nonce: str
    signature: str

    def signed_payload(self) -> bytes:
        return canonical_json(self.model_dump(mode="json", exclude={"signature"}))


class NodeRegistrationResponse(BaseModel):
    ok: bool = True
    node_id: str
    tunnel_token: str
    heartbeat_interval_s: float
    coordinator_version: str
    server_time: float


class HeartbeatRequest(BaseModel):
    capabilities: Capabilities | None = None
    active_sessions: int = 0
    training_jobs: int = 0
    tunnel_connected: bool = False
    last_error: str | None = None


class HeartbeatResponse(BaseModel):
    ok: bool = True
    server_time: float
    offline: bool = False


class BackendInfo(BaseModel):
    kind: str
    name: str
    available: bool
    memory_mb: int = 0
    reason: str = ""


class NodeView(BaseModel):
    node_id: str
    name: str
    online: bool
    tunnel_connected: bool
    banned: bool
    reputation: float
    last_seen_s: float
    capabilities: Capabilities
    labels: dict[str, str] = Field(default_factory=dict)
    accepted_updates: int = 0
    rejected_updates: int = 0
    failed_rounds: int = 0
    current_assignment: str | None = None


class DatasetInfo(BaseModel):
    dataset_id: str
    name: str
    size_bytes: int
    sha256: str
    created_at: float
    text_lines: int | None = None


class TrainingSpec(BaseModel):
    algorithm: Literal["diloco", "diloco_decoupled", "pipeline"] = "diloco_decoupled"
    parallelism: Literal["diloco", "pipeline"] = "diloco"
    rounds: int = Field(20, ge=1, le=100000)
    local_steps: int = Field(20, ge=1, le=100000)
    batch_size: int = Field(4, ge=1, le=4096)
    seq_len: int = Field(128, ge=2, le=8192)
    lr: float = Field(3e-4, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    grad_clip: float = Field(1.0, ge=0)
    outer_lr: float = Field(1.0, gt=0)
    outer_momentum: float = Field(0.9, ge=0, le=0.9999)
    nesterov: bool = False
    min_workers: int = Field(1, ge=1)
    max_workers: int = Field(4, ge=1)
    quorum_fraction: float = Field(1.0, gt=0, le=1.0)
    grace_s: float = Field(0.0, ge=0)
    round_timeout_s: float = Field(180.0, gt=1)
    eval_every: int = Field(1, ge=0)
    eval_batches: int = Field(8, ge=1, le=512)
    val_fraction: float = Field(0.1, gt=0, le=0.5)
    seed: int = 1234
    compression: Literal["fp32", "fp16"] = "fp16"
    max_update_norm_ratio: float = Field(4.0, gt=0)
    mode: Literal["pretrain", "sft"] = "pretrain"
    # Pipeline-parallel training.  One optimizer step per round over
    # ``micro_batches`` activation microbatches.
    micro_batches: int = Field(4, ge=1, le=256)
    pipeline_stages: int = Field(0, ge=0, le=64)
    activation_dtype: Literal["fp32", "fp16"] = "fp32"
    lora_rank: int = Field(0, ge=0, le=256)
    lora_alpha: float = Field(0.0, ge=0)
    island: str | None = None
    stage_memory_headroom: float = Field(1.25, ge=1.0, le=8.0)

    @field_validator("max_workers")
    @classmethod
    def _max_workers(cls, v: int) -> int:
        return max(1, v)

    def model_post_init(self, __context) -> None:  # type: ignore[override]
        if self.min_workers > self.max_workers:
            raise ValueError("min_workers cannot exceed max_workers")
        if self.eval_every < 0:
            raise ValueError("eval_every must be >= 0")
        if self.parallelism == "pipeline" and self.algorithm != "pipeline":
            object.__setattr__(self, "algorithm", "pipeline")
        if self.parallelism == "diloco" and self.algorithm == "pipeline":
            object.__setattr__(self, "parallelism", "pipeline")
        if self.lora_rank and self.lora_alpha <= 0:
            object.__setattr__(self, "lora_alpha", float(self.lora_rank * 2))


class JobCreate(BaseModel):
    name: str = "xdc-job"
    dataset_id: str
    model: dict[str, Any]
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    auto_start: bool = True
    init_checkpoint_job: str | None = None


class RoundView(BaseModel):
    round_id: str
    index: int
    base_version: int
    expected: list[str]
    submitted: list[str]
    failed: list[str]
    deadline: float
    aggregated: bool = False


class JobView(BaseModel):
    job_id: str
    name: str
    state: str
    created_at: float
    updated_at: float
    model: dict[str, Any]
    training: TrainingSpec
    dataset: DatasetInfo
    current_round: RoundView | None = None
    next_round: int = 0
    version: int = 0
    initial_loss: float | None = None
    final_loss: float | None = None
    best_loss: float | None = None
    latest_checkpoint: str | None = None
    initial_checkpoint: str | None = None
    error: str | None = None
    metrics: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    tokens_trained: int = 0
    parameters: int = 0
    traceback: str | None = None
    model_kind: str = "tinygpt"
    model_dir: str | None = None
    tokenizer: dict[str, Any] = Field(default_factory=dict)
    stage_plan: list[dict[str, Any]] = Field(default_factory=list)
    stage_checkpoints: dict[str, Any] = Field(default_factory=dict)



class Assignment(BaseModel):
    type: Literal["train_round"] = "train_round"
    job_id: str
    job_name: str
    round_id: str
    round_index: int
    base_version: int
    model_path: str
    model_sha256: str
    model_spec: dict[str, Any] = Field(default_factory=dict)
    dataset_path: str
    dataset_id: str
    dataset_sha256: str
    shard_sha256: str = ""
    shard_start: int
    shard_end: int
    shard_index: int
    world_size: int
    local_steps: int
    batch_size: int
    seq_len: int
    lr: float
    weight_decay: float
    grad_clip: float
    seed: int
    max_update_norm_ratio: float
    deadline: float
    compression: Literal["fp32", "fp16"] = "fp16"


class PipelineStagePlan(BaseModel):
    stage_index: int
    node_id: str
    node_name: str = ""
    layer_start: int
    layer_end: int
    stage_path: str = ""
    stage_sha256: str = ""
    optimizer_path: str | None = None
    optimizer_sha256: str | None = None
    estimated_bytes: int = 0
    parameters: int = 0
    trainable_parameters: int = 0
    device: str = ""
    param_bytes: int = 0
    optimizer_bytes: int = 0
    grad_bytes: int = 0
    measured_total_bytes: int = 0


class PipelineAssignment(BaseModel):
    """Direct tunnel assignment for one pipeline stage.

    Pipeline rounds are coordinator-driven RPC, not polled HTTP assignments; this
    model documents the plan exchanged with the worker and is stored in the job
    view for inspection.
    """

    type: Literal["pipeline_stage"] = "pipeline_stage"
    job_id: str
    job_name: str
    round_id: str
    round_index: int
    base_version: int
    stage_index: int
    num_stages: int
    layer_start: int
    layer_end: int
    model_spec: dict[str, Any] = Field(default_factory=dict)
    model_kind: str = "hf"
    stage_path: str
    stage_sha256: str
    optimizer_path: str | None = None
    optimizer_sha256: str | None = None
    micro_batches: int
    batch_size: int
    seq_len: int
    lr: float
    weight_decay: float
    grad_clip: float
    seed: int
    deadline: float
    activation_dtype: Literal["fp32", "fp16"] = "fp32"
    lora_rank: int = 0
    lora_alpha: float = 0.0
    island: str | None = None
    tokenizer: dict[str, Any] = Field(default_factory=dict)
    estimated_bytes: int = 0


class UpdateMeta(BaseModel):
    job_id: str
    round_id: str
    node_id: str
    base_version: int
    loss: float
    steps: int = Field(ge=1, le=1_000_000)
    tokens: int = Field(ge=1, le=10**12)
    elapsed_s: float = Field(ge=0)
    update_norm: float = Field(ge=0)
    device: str = ""
    compressed: Literal["fp32", "fp16"] = "fp32"
    nonce: str


class UpdateAccepted(BaseModel):
    ok: bool = True
    job_id: str
    round_id: str
    status: str


class InferenceRequest(BaseModel):
    job_id: str | None = None
    checkpoint: str | None = None
    prompt: str = ""
    max_new_tokens: int = Field(32, ge=1, le=4096)
    temperature: float = Field(0.0, ge=0.0, le=10.0)
    top_k: int = Field(0, ge=0, le=1000000)
    seed: int = 0
    stream: bool = True
    allow_retry: bool = True
    max_stages: int = Field(0, ge=0, le=128)


class InferencePlanView(BaseModel):
    stages: list[dict[str, Any]]
    nodes: list[str]
    model_sha256: str
    total_layers: int
    pipeline_latency_ms: float | None = None


class CheckpointView(BaseModel):
    version: int
    path: str
    size_bytes: int
    sha256: str
    val_loss: float | None = None
    created_at: float


class AdminStats(BaseModel):
    nodes_online: int
    nodes_total: int
    jobs_running: int
    jobs_total: int
    pooled_cpu_threads: int
    pooled_memory_mb: int
    pooled_gpu_memory_mb: int
    total_tokens_trained: int
    uptime_s: float
    version: str
