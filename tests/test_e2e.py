from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import httpx
import pytest
import torch

from xdc.crypto import NodeIdentity, sign_request
from xdc.model.checkpoint import load_state, sha256_file
from xdc.model.data import shard_bounds
from xdc.model.gpt import GPT, ModelConfig


def _model(cfg_dict):
    return GPT(ModelConfig.from_dict(cfg_dict)).eval()


def _local_greedy(checkpoint: Path, cfg_dict: dict, prompt: str, n: int) -> bytes:
    model = _model(cfg_dict)
    model.load_state_dict(load_state(checkpoint))
    prompt_bytes = prompt.encode()
    ids = torch.tensor([list(prompt_bytes)], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(ids, n, temperature=0)
    return bytes(out[0, len(prompt_bytes) :].tolist())


def _assert_pipeline_matches(cluster, job: dict, prompt: str, n: int, distributed: bool):
    out = cluster.infer(job["job_id"], prompt, n)
    assert out["distributed"] is distributed
    got = base64.b64decode(out["bytes_b64"])
    ckpt = Path(cluster.root) / "coordinator-data" / job["latest_checkpoint"]
    expected = _local_greedy(ckpt, job["model"], prompt, n)
    assert got == expected, (got.hex(), expected.hex(), out["text"])
    return out


@pytest.mark.e2e
def test_two_worker_training_and_distributed_inference(cluster, corpus):
    job = cluster.create_job(
        data=corpus[:160_000],
        name="two-worker",
        training={
            "rounds": 6,
            "local_steps": 8,
            "batch_size": 2,
            "seq_len": 64,
            "min_workers": 2,
            "max_workers": 2,
            "quorum_fraction": 1.0,
            "grace_s": 0.2,
        },
    )
    done = cluster.wait_job(job["job_id"])
    assert done["state"] == "completed", done
    assert done["version"] == 6
    assert done["initial_loss"] is not None and done["final_loss"] is not None
    assert done["final_loss"] < done["initial_loss"], (done["initial_loss"], done["final_loss"])
    assert done["tokens_trained"] == 6 * 2 * 8 * 2 * 64
    assert len(done["metrics"]) == 6
    workers = set()
    for metric in done["metrics"]:
        assert metric["updates_received"] == 2
        assert metric["updates_used"] == 2
        workers.update(metric["workers"])
    assert len(workers) == 2
    assert done["latest_checkpoint"]
    job_dir = Path(cluster.root) / "coordinator-data" / "jobs" / job["job_id"]
    initial = job_dir / "init.safetensors"
    final = Path(cluster.root) / "coordinator-data" / done["latest_checkpoint"]
    assert initial.exists() and final.exists()
    assert sha256_file(initial) != sha256_file(final)
    # Initial and final parameter tensors genuinely differ.
    a, b = load_state(initial), load_state(final)
    assert any(not torch.equal(a[k], b[k]) for k in a)

    # True two-stage layer-sharded inference and exact parity with one process.
    out = _assert_pipeline_matches(cluster, done, "Mira found a brass key near", 8, True)
    assert len(out["stages"]) == 2
    assert {s["layer_start"] for s in out["stages"]} == {0, 1}
    assert out["first_token_ms"] is not None

    # Streaming returns the same raw bytes and real token events.
    payload = {
        "job_id": done["job_id"],
        "prompt": "Mira found a brass key near",
        "max_new_tokens": 8,
        "temperature": 0.0,
        "stream": True,
        "allow_retry": True,
    }
    tokens = bytearray()
    started = False
    finished = False
    with cluster.client.stream("POST", "/v1/infer", content=json.dumps(payload), headers={"Content-Type": "application/json"}) as r:
        assert r.status_code == 200, r.read()
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                if event == "token":
                    tokens.extend(base64.b64decode(data["bytes_b64"]))
                elif event == "start":
                    started = True
                    assert data["distributed"] is True
                elif event == "done":
                    finished = True
    assert started and finished
    assert bytes(tokens) == base64.b64decode(out["bytes_b64"])


@pytest.mark.e2e
def test_dataset_download_is_shard_only(cluster, corpus):
    """A worker must never receive the full training split, only its shard."""
    job = cluster.create_job(
        data=corpus[:160_000],
        name="shard-privacy",
        training={
            "rounds": 2,
            "local_steps": 100,
            "batch_size": 2,
            "seq_len": 64,
            "min_workers": 2,
            "max_workers": 2,
            "quorum_fraction": 1.0,
            "grace_s": 0.2,
            "round_timeout_s": 120,
        },
    )
    job_dir = Path(cluster.root) / "coordinator-data" / "jobs" / job["job_id"]
    train = (job_dir / "train.bin").read_bytes()
    total = len(train)
    identity = NodeIdentity.load_or_create(cluster.root / "worker-1")
    path = f"/v1/jobs/{job['job_id']}/dataset"

    response = None
    deadline = time.time() + 60
    while time.time() < deadline:
        cur = cluster.client.get(f"/v1/jobs/{job['job_id']}").json()
        if cur["current_round"] is None:
            time.sleep(0.1)
            continue
        # A fresh nonce is required for every signed request.
        headers = sign_request(identity, "GET", path, b"")
        r = httpx.get(cluster.url + path, headers=headers, trust_env=False, timeout=30)
        if r.status_code == 200:
            response = r
            break
        # The round may have committed between the two requests; wait for the next.
        time.sleep(0.2)
    assert response is not None, "could not fetch a shard while a round was open"
    body = response.content
    assert len(body) < total, "worker received the entire training split"
    expected = []
    for rank in range(2):
        start, end = shard_bounds(total, rank, 2)
        expected.append(train[start:end])
    assert body in expected
    assert response.headers.get("X-XDC-Shard-SHA256") == hashlib.sha256(body).hexdigest()


@pytest.mark.e2e
def test_checkpoint_resume_after_coordinator_restart(cluster, corpus):
    job = cluster.create_job(
        data=corpus[:120_000],
        name="resume-job",
        training={
            "rounds": 8,
            "local_steps": 10,
            "batch_size": 2,
            "seq_len": 64,
            "min_workers": 2,
            "max_workers": 2,
            "grace_s": 0.2,
        },
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        cur = cluster.client.get(f"/v1/jobs/{job['job_id']}").json()
        if cur["version"] >= 1 and cur["current_round"] is None:
            break
        time.sleep(0.15)
    else:
        raise AssertionError("no completed round")
    paused = cluster.client.post(f"/v1/jobs/{job['job_id']}/pause").json()
    assert paused["state"] == "paused"
    paused_version = paused["version"]
    ckpt = paused["latest_checkpoint"]
    assert ckpt
    assert (Path(cluster.root) / "coordinator-data" / ckpt).exists()

    cluster.restart_coordinator()
    resumed = cluster.client.post(f"/v1/jobs/{job['job_id']}/start").json()
    assert resumed["state"] in {"running", "waiting"}
    done = cluster.wait_job(job["job_id"], timeout=180)
    assert done["state"] == "completed", done
    assert done["version"] >= paused_version + 1
    assert done["latest_checkpoint"] != ckpt or done["version"] > paused_version
    # The resumed checkpoint is still valid and inference works from it.
    _assert_pipeline_matches(cluster, done, "The harbor at dawn was quiet", 6, True)


@pytest.mark.e2e
def test_worker_failure_detection_and_inference_reschedule(cluster, corpus):
    job = cluster.create_job(
        data=corpus[:140_000],
        name="fault-job",
        training={
            "rounds": 8,
            "local_steps": 14,
            "batch_size": 2,
            "seq_len": 64,
            "min_workers": 1,
            "max_workers": 2,
            "grace_s": 1.0,
            "round_timeout_s": 30,
        },
    )
    deadline = time.time() + 120
    saw_round = False
    while time.time() < deadline:
        cur = cluster.client.get(f"/v1/jobs/{job['job_id']}").json()
        if cur["version"] >= 1:
            saw_round = True
            break
        time.sleep(0.1)
    assert saw_round
    cluster.stop_worker("worker-2", graceful=False)
    cluster.wait_worker_offline("worker-2", timeout=20)
    done = cluster.wait_job(job["job_id"], timeout=240)
    assert done["state"] == "completed", done
    assert done["version"] == 8
    lost_events = [e for e in done["events"] if "worker lost" in e["message"] or "retry" in e["message"]]
    assert lost_events, done["events"][-20:]
    # With one healthy worker left, XDC reschedules all layers to it.  This is a
    # single-node pipeline fallback, so distributed must be false and correct.
    out = _assert_pipeline_matches(cluster, done, "A brass key waited in the mossy library", 6, False)
    assert len(out["stages"]) == 1


@pytest.mark.e2e
def test_tls_end_to_end(tls_cluster, corpus):
    assert tls_cluster.url.startswith("https://")
    stats = tls_cluster.client.get("/v1/stats").json()
    assert stats["nodes_online"] == 1
    job = tls_cluster.create_job(
        data=corpus[:60_000],
        name="tls-job",
        training={
            "rounds": 2,
            "local_steps": 4,
            "batch_size": 2,
            "seq_len": 64,
            "min_workers": 1,
            "max_workers": 1,
            "grace_s": 0.1,
        },
    )
    done = tls_cluster.wait_job(job["job_id"])
    assert done["state"] == "completed"
    assert done["latest_checkpoint"]
    # TLS connection with an untrusted certificate is rejected by httpx.
    with pytest.raises(httpx.TransportError):
        httpx.get(tls_cluster.url + "/healthz", verify=True, trust_env=False, timeout=3)


@pytest.mark.e2e
def test_admin_auth_required(cluster):
    with httpx.Client(base_url=cluster.url, trust_env=False, verify=True) as anon:
        assert anon.get("/v1/stats").status_code == 401
        assert anon.get("/v1/nodes").status_code == 401
        assert anon.get("/healthz").status_code == 200
        # Worker endpoints reject unsigned traffic.
        assert anon.get("/v1/worker/assignment").status_code == 401
