"""YAML configuration loading with strict, readable defaults."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _dataclass_from_dict(cls, data: dict):
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in valid})


@dataclass
class TLSConfig:
    enabled: bool = False
    cert_file: str | None = None
    key_file: str | None = None
    ca_file: str | None = None


@dataclass
class CoordinatorConfig:
    host: str = "127.0.0.1"
    port: int = 8787
    data_dir: str = "./xdc-data/coordinator"
    admin_token: str = ""
    join_token: str = ""
    public_url: str | None = None
    heartbeat_timeout_s: float = 15.0
    offline_grace_s: float = 5.0
    job_tick_s: float = 0.5
    max_upload_mb: int = 512
    max_dataset_mb: int = 512
    max_model_upload_mb: int = 4096
    eval_device: str = "cpu"
    log_level: str = "INFO"
    checkpoint_keep: int = 20
    trust_proxy_headers: bool = False
    pipeline_memory_fraction: float = 0.7
    pipeline_reserve_mb: int = 512
    tls: TLSConfig = field(default_factory=TLSConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> "CoordinatorConfig":
        raw: dict[str, Any] = {}
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        raw = raw.get("coordinator", raw) if "coordinator" in raw else raw
        tls = _dataclass_from_dict(TLSConfig, raw.pop("tls", {}) or {})
        cfg = _dataclass_from_dict(cls, raw)
        cfg.tls = tls
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not (0 < self.port < 65536):
            raise ValueError("invalid coordinator port")
        if not self.admin_token or not self.join_token:
            raise ValueError("admin_token and join_token must be configured")
        if self.tls.enabled and (not self.tls.cert_file or not self.tls.key_file):
            raise ValueError("TLS enabled but cert_file/key_file missing")
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class WorkerConfig:
    coordinator_url: str = "http://127.0.0.1:8787"
    join_token: str = ""
    data_dir: str = "./xdc-data/worker"
    name: str = ""
    backend: str = "auto"
    threads: int = 0
    max_memory_mb: int = 0
    tls_ca: str | None = None
    allow_insecure: bool = False
    heartbeat_interval_s: float = 5.0
    poll_interval_s: float = 1.0
    labels: dict[str, str] = field(default_factory=dict)
    log_level: str = "INFO"
    inference_cache: bool = True
    max_inference_sessions: int = 4
    mps_cpu_threshold_params: int = 50_000_000

    @classmethod
    def load(cls, path: str | Path | None) -> "WorkerConfig":
        raw: dict[str, Any] = {}
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        if "worker" in raw:
            raw = raw["worker"]
        cfg = _dataclass_from_dict(cls, raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.join_token:
            raise ValueError("worker join_token must be configured")
        if not self.coordinator_url.startswith(("http://", "https://")):
            raise ValueError("coordinator_url must start with http:// or https://")
        if self.coordinator_url.startswith("http://") and not self.allow_insecure:
            raise ValueError(
                "plain HTTP worker is disabled by default; enable TLS or set allow_insecure: true for localhost/LAN development"
            )
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class CoordinatorRuntime:
    """Compatibility container for CLI flags merged with file config."""

    file: CoordinatorConfig


DEFAULT_COORDINATOR_YAML = """\
coordinator:
  host: 127.0.0.1
  port: 8787
  data_dir: ./xdc-data/coordinator
  admin_token: REPLACE_ADMIN_TOKEN
  join_token: REPLACE_JOIN_TOKEN
  heartbeat_timeout_s: 15
  max_upload_mb: 512
  eval_device: cpu
  tls:
    enabled: false
    cert_file: null
    key_file: null
    ca_file: null
"""

DEFAULT_WORKER_YAML = """\
worker:
  coordinator_url: https://127.0.0.1:8787
  join_token: REPLACE_JOIN_TOKEN
  data_dir: ./xdc-data/worker
  name: ""
  backend: auto
  threads: 0
  max_memory_mb: 0
  tls_ca: ./xdc-data/tls/ca.pem
  allow_insecure: false
  heartbeat_interval_s: 5
  poll_interval_s: 1
  labels: {}
"""
