"""Shared fixtures for XDC tests, including a real multi-process cluster."""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Cluster:
    def __init__(
        self,
        root: Path,
        tls: bool = False,
        workers: int = 2,
        backends: dict[str, str] | None = None,
        labels: dict[str, dict[str, str]] | None = None,
    ):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.tls = tls
        self.port = free_port()
        self.admin_token = secrets.token_urlsafe(24)
        self.join_token = secrets.token_urlsafe(24)
        self.coord_proc: subprocess.Popen | None = None
        self.worker_procs: dict[str, subprocess.Popen] = {}
        self.worker_configs: dict[str, Path] = {}
        self.scheme = "https" if tls else "http"
        self.url = f"{self.scheme}://127.0.0.1:{self.port}"
        self.ca = None
        if tls:
            from xdc.crypto import generate_tls_material

            paths = generate_tls_material(root / "tls", hosts=["127.0.0.1"])
            self.ca = paths["ca"]
            self.cert = paths["cert"]
            self.key = paths["key"]
        coord = {
            "coordinator": {
                "host": "127.0.0.1",
                "port": self.port,
                "data_dir": str(root / "coordinator-data"),
                "admin_token": self.admin_token,
                "join_token": self.join_token,
                "heartbeat_timeout_s": 15.0,
                "job_tick_s": 0.2,
                "eval_device": "cpu",
                "max_upload_mb": 64,
                "max_dataset_mb": 64,
                "tls": {
                    "enabled": tls,
                    "cert_file": getattr(self, "cert", None),
                    "key_file": getattr(self, "key", None),
                    "ca_file": self.ca,
                },
            }
        }
        self.coord_config = root / "coordinator.yaml"
        self.coord_config.write_text(yaml.safe_dump(coord, sort_keys=False))
        for i in range(1, workers + 1):
            name = f"worker-{i}"
            wcfg = {
                "worker": {
                    "coordinator_url": self.url,
                    "join_token": self.join_token,
                    "data_dir": str(root / name),
                    "name": name,
                    "backend": (backends or {}).get(name, "cpu"),
                    "labels": (labels or {}).get(name, {}),
                    "threads": 2,
                    "heartbeat_interval_s": 1.0,
                    "poll_interval_s": 0.15,
                    "tls_ca": self.ca,
                    "allow_insecure": not tls,
                    "max_inference_sessions": 4,
                }
            }
            path = root / f"{name}.yaml"
            path.write_text(yaml.safe_dump(wcfg, sort_keys=False))
            self.worker_configs[name] = path
        self.log_dir = root / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.client = httpx.Client(
            base_url=self.url,
            headers={"Authorization": f"Bearer {self.admin_token}"},
            verify=self.ca if tls else True,
            trust_env=False,
            timeout=120.0,
        )

    def _popen(self, args, log_name):
        log = open(self.log_dir / log_name, "ab")
        return subprocess.Popen(
            args,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=(os.name != "nt"),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    def start_coordinator(self) -> None:
        self.coord_proc = self._popen(
            [PYTHON, "-m", "xdc", "coordinator", "--config", str(self.coord_config)],
            "coordinator.log",
        )
        deadline = time.time() + 120
        last = None
        while time.time() < deadline:
            try:
                r = httpx.get(
                    self.url + "/healthz",
                    verify=self.ca if self.tls else True,
                    trust_env=False,
                    timeout=2,
                )
                if r.status_code == 200:
                    return
            except Exception as exc:
                last = exc
            if self.coord_proc.poll() is not None:
                raise RuntimeError(f"coordinator exited: {self.tail('coordinator.log')}")
            time.sleep(0.2)
        raise RuntimeError(f"coordinator did not start: {last}\n{self.tail('coordinator.log')}")

    def start_worker(self, name: str) -> None:
        if name in self.worker_procs and self.worker_procs[name].poll() is None:
            return
        self.worker_procs[name] = self._popen(
            [PYTHON, "-m", "xdc", "worker", "--config", str(self.worker_configs[name])],
            f"{name}.log",
        )

    def start_workers(self) -> None:
        for name in self.worker_configs:
            self.start_worker(name)
        self.wait_nodes(len(self.worker_configs))

    def wait_nodes(self, count: int, timeout: float = 60) -> list[dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                nodes = self.client.get("/v1/nodes").json()
                online = [n for n in nodes if n["online"] and n["tunnel_connected"]]
                if len(online) >= count:
                    return nodes
            except Exception:
                pass
            time.sleep(0.2)
        raise RuntimeError(f"only saw {len(self.client.get('/v1/nodes').json())} nodes\n{self.dump_logs()}")

    def wait_worker_offline(self, name: str, timeout: float = 60) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            nodes = self.client.get("/v1/nodes").json()
            for n in nodes:
                if n["name"] == name and not n["online"]:
                    return
            time.sleep(0.3)
        raise RuntimeError(f"{name} not marked offline")

    def stop_worker(self, name: str, graceful: bool = False) -> None:
        proc = self.worker_procs.get(name)
        if not proc:
            return
        if graceful:
            proc.terminate()
        else:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def stop_coordinator(self, graceful: bool = True) -> None:
        proc = self.coord_proc
        if not proc:
            return
        if graceful:
            proc.terminate()
        else:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        self.coord_proc = None

    def restart_coordinator(self) -> None:
        self.stop_coordinator(graceful=True)
        self.start_coordinator()
        self.wait_nodes(len([p for p in self.worker_procs.values() if p.poll() is None]))

    def cleanup(self) -> None:
        for proc in self.worker_procs.values():
            if proc.poll() is None:
                proc.kill()
        for proc in self.worker_procs.values():
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        if self.coord_proc and self.coord_proc.poll() is None:
            self.coord_proc.kill()
            try:
                self.coord_proc.wait(timeout=5)
            except Exception:
                pass
        self.client.close()

    def tail(self, name: str, n: int = 80) -> str:
        path = self.log_dir / name
        if not path.exists():
            return ""
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])

    def dump_logs(self) -> str:
        parts = ["=== coordinator ===", self.tail("coordinator.log")]
        for name in self.worker_configs:
            parts += [f"=== {name} ===", self.tail(f"{name}.log", 40)]
        return "\n".join(parts)

    # convenience API helpers
    def upload_dataset(self, data: bytes, name: str = "test") -> dict:
        r = self.client.post("/v1/admin/datasets", params={"name": name}, content=data)
        assert r.status_code == 200, r.text
        return r.json()

    def create_job(self, **overrides) -> dict:
        data = overrides.pop("data", None)
        if data is not None:
            ds = self.upload_dataset(data)
        else:
            ds = overrides.pop("dataset")
        spec = {
            "rounds": 4,
            "local_steps": 6,
            "batch_size": 2,
            "seq_len": 64,
            "lr": 3e-4,
            "outer_lr": 1.0,
            "outer_momentum": 0.9,
            "min_workers": 1,
            "max_workers": 2,
            "quorum_fraction": 1.0,
            "grace_s": 0.3,
            "round_timeout_s": 120,
            "eval_every": 1,
            "eval_batches": 2,
            "seed": 7,
            "compression": "fp16",
        }
        spec.update(overrides.pop("training", {}))
        body = {
            "name": overrides.pop("name", "test-job"),
            "dataset_id": ds["dataset_id"],
            "model": {
                "vocab_size": 256,
                "n_layer": overrides.pop("layers", 2),
                "n_head": overrides.pop("heads", 4),
                "n_embd": overrides.pop("dim", 64),
                "block_size": overrides.pop("block_size", 64),
                "dropout": 0.0,
                "bias": True,
            },
            "training": spec,
            "auto_start": True,
        }
        body.update(overrides)
        r = self.client.post("/v1/jobs", content=__import__("json").dumps(body), headers={"Content-Type": "application/json"})
        assert r.status_code == 200, r.text
        return r.json()

    def wait_job(self, job_id: str, timeout: float = 300) -> dict:
        deadline = time.time() + timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                job = self.client.get(f"/v1/jobs/{job_id}", timeout=60).json()
            except (httpx.HTTPError, ValueError) as exc:
                # A single slow poll must not abort a long CPU-bound job.
                last_error = exc
                time.sleep(0.5)
                continue
            if job["state"] in {"completed", "failed", "cancelled"}:
                return job
            time.sleep(0.2)
        raise RuntimeError(f"job {job_id} did not finish (last error: {last_error!r})\n{self.dump_logs()}")

    def infer(self, job_id: str, prompt: str, max_tokens: int = 8, stream: bool = False) -> dict:
        payload = {
            "job_id": job_id,
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
            "stream": stream,
            "allow_retry": True,
        }
        r = self.client.post("/v1/infer", content=__import__("json").dumps(payload), headers={"Content-Type": "application/json"})
        assert r.status_code == 200, r.text
        return r.json()


@pytest.fixture
def corpus() -> bytes:
    data = (ROOT / "datasets" / "tiny_stories.txt").read_bytes()
    return data[:120_000]


@pytest.fixture
def cluster(tmp_path, corpus):
    c = Cluster(tmp_path, tls=False, workers=2)
    try:
        c.start_coordinator()
        c.start_workers()
        yield c
    finally:
        c.cleanup()


@pytest.fixture
def tls_cluster(tmp_path):
    c = Cluster(tmp_path / "tls-cluster", tls=True, workers=1)
    try:
        c.start_coordinator()
        c.start_workers()
        yield c
    finally:
        c.cleanup()
