"""Pipeline-parallel correctness, memory sharding and failure tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from xdc.model.checkpoint import load_state
from xdc.model.hf import SafetensorStore, write_stage_file
from xdc.model.llama import LlamaConfig, LlamaModel, LlamaStage
from xdc.model.tokenizer import read_tokens, sample_token_batch
from tests.conftest import Cluster
from tests.hf_fixture import build_tiny_hf


def _copy_model(c, src: Path) -> None:
    dest = c.root / "coordinator-data" / "models" / "tinytest"
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file():
            (dest / path.name).write_bytes(path.read_bytes())


def _create_pipeline_job(
    c,
    dataset: dict,
    *,
    rounds: int = 1,
    micro_batches: int = 2,
    batch_size: int = 2,
    seq_len: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 0.0,
    eval_every: int = 0,
    lora_rank: int = 0,
    min_workers: int = 2,
    max_workers: int = 2,
    pipeline_stages: int = 2,
    island: str | None = None,
) -> dict:
    body = {
        "name": "pipeline-test",
        "dataset_id": dataset["dataset_id"],
        "model": {"kind": "hf", "model_name": "tinytest"},
        "training": {
            "parallelism": "pipeline",
            "algorithm": "pipeline",
            "rounds": rounds,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "micro_batches": micro_batches,
            "lr": lr,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "min_workers": min_workers,
            "max_workers": max_workers,
            "pipeline_stages": pipeline_stages,
            "eval_every": eval_every,
            "eval_batches": 2,
            "round_timeout_s": 60,
            "seed": 7,
            "lora_rank": lora_rank,
            "island": island,
        },
        "auto_start": True,
    }
    r = c.client.post(
        "/v1/jobs",
        content=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 200, r.text
    return c.wait_job(r.json()["job_id"], timeout=180)


def test_llama_logits_and_gradients_match_transformers(tmp_path):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig as HFCfg
    from transformers import LlamaForCausalLM

    torch.manual_seed(0)
    hc = HFCfg(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
    )
    ref = LlamaForCausalLM(hc).eval()
    state = {k: v.float() for k, v in ref.state_dict().items()}
    cfg = LlamaConfig.from_dict(hc.to_dict())
    ours = LlamaModel(cfg)
    ours.load_hf_state(state)
    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        assert torch.equal(ref(ids).logits, ours(ids)[0])

    ref.zero_grad()
    loss_ref = ref(ids, labels=ids).loss
    loss_ref.backward()
    ours.zero_grad()
    _, loss_ours = ours(ids[:, :-1], targets=ids[:, 1:])
    loss_ours.backward()
    assert abs(loss_ref.item() - loss_ours.item()) < 1e-5
    ref_params = dict(ref.named_parameters())
    for name, param in ours.named_parameters():
        local = name[len("model.") :] if name.startswith("model.") else name
        key = ours.model.global_key(local)
        assert torch.allclose(param.grad, ref_params[key].grad, atol=1e-6, rtol=1e-5)
    assert ours.num_parameters() == cfg.num_parameters()


def test_stage_files_really_shard_the_model(tmp_path):
    build_tiny_hf(tmp_path, layers=6, hidden=64)
    cfg = LlamaConfig.from_dict(json.loads((tmp_path / "config.json").read_text()))
    full = tmp_path / "full.safetensors"
    from xdc.model.hf import write_full_file

    with SafetensorStore(tmp_path) as store:
        write_full_file(store, cfg, full)
        stage_paths = []
        for index, (start, end) in enumerate([(0, 2), (2, 4), (4, 6)]):
            path = tmp_path / f"stage-{index}.safetensors"
            write_stage_file(store, cfg, start, end, path)
            stage_paths.append(path)
    full_size = full.stat().st_size
    stage_params = 0
    for (start, end), path in zip([(0, 2), (2, 4), (4, 6)], stage_paths):
        state = load_file(str(path))
        stage = LlamaStage(cfg, start, end)
        stage.load_state_dict(state, strict=True)
        stage_params += sum(int(p.numel()) for p in stage.parameters())
        assert path.stat().st_size < full_size
    assert stage_params == cfg.num_parameters()

    # The three stages compose exactly to the monolithic model.
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    full_model = LlamaModel(cfg)
    full_model.load_hf_state(load_state(full))
    with torch.no_grad():
        expected = full_model(x)[0]
        hidden = None
        for (start, end), path in zip([(0, 2), (2, 4), (4, 6)], stage_paths):
            stage = LlamaStage(cfg, start, end)
            stage.load_state_dict(load_file(str(path)), strict=True)
            if start == 0:
                hidden, _ = stage.forward_features(input_ids=x)
            else:
                hidden, _ = stage.forward_features(hidden=hidden)
        actual = stage.head_logits(hidden)
    assert torch.equal(expected, actual)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    dest = tmp_path_factory.mktemp("tiny-hf")
    build_tiny_hf(dest, layers=4, hidden=64)
    return dest


@pytest.mark.e2e
def test_pipeline_step_matches_monolithic(cluster, corpus, tiny_model):
    _copy_model(cluster, tiny_model)
    dataset = cluster.upload_dataset(corpus, "pipeline-exact")
    job = _create_pipeline_job(cluster, dataset)
    assert job["state"] == "completed", job.get("error")
    assert job["version"] == 1
    cfg = LlamaConfig.from_dict(job["model"])
    init = load_state(cluster.root / "coordinator-data" / job["initial_checkpoint"])
    final = load_state(cluster.root / "coordinator-data" / job["latest_checkpoint"])

    tokens = read_tokens(
        cluster.root / "coordinator-data" / "jobs" / job["job_id"] / "train.tokens"
    )
    gen = torch.Generator().manual_seed(7)
    xs, ys = [], []
    for _ in range(2):
        x, y = sample_token_batch(tokens, 2, 16, gen)
        xs.append(x)
        ys.append(y)
    x, y = torch.cat(xs), torch.cat(ys)

    model = LlamaModel(cfg)
    model.load_hf_state(init)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0
    )
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, targets=y)
    loss.backward()
    opt.step()
    expected = model.state_dict_global()
    assert set(expected) == set(final)
    worst = 0.0
    for key in expected:
        diff = (expected[key].float() - final[key].float()).abs().max().item()
        worst = max(worst, diff)
    assert worst < 2e-5, f"pipeline diverged from monolithic AdamW by {worst}"

    metric = job["metrics"][-1]
    assert metric["parallelism"] == "pipeline"
    assert metric["optimizer_sharded"] is True
    assert metric["tokens"] == 2 * 2 * 16
    assert len(metric["stage_metrics"]) == 2
    full_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    for stage in metric["stage_metrics"]:
        assert stage["param_bytes"] < full_param_bytes
        assert stage["optimizer_bytes"] > 0
        assert stage["grad_bytes"] > 0
    assert sum(s["param_bytes"] for s in metric["stage_metrics"]) < 2 * full_param_bytes


@pytest.mark.e2e
def test_pipeline_tied_head_is_synchronized(cluster, corpus, tmp_path):
    tied = tmp_path / "tied"
    build_tiny_hf(tied, layers=4, hidden=64, tied=True)
    _copy_model(cluster, tied)
    # The coordinator resolves the model by directory name; use a separate job
    # name and replace the directory contents in place.
    dest = cluster.root / "coordinator-data" / "models" / "tinytest"
    assert (dest / "config.json").exists()
    dataset = cluster.upload_dataset(corpus, "pipeline-tied")
    job = _create_pipeline_job(cluster, dataset, rounds=2, eval_every=1)
    assert job["state"] == "completed", job.get("error")
    assert job["final_loss"] is not None
    state = load_state(cluster.root / "coordinator-data" / job["latest_checkpoint"])
    # Tied checkpoints omit the duplicate head; the final stage file carries a
    # synchronized copy derived from the committed embedding.
    assert "lm_head.weight" not in state
    version = str(job["version"])
    last = job["stage_checkpoints"][version]["1"]
    stage_state = load_file(
        str(cluster.root / "coordinator-data" / last["params_path"])
    )
    assert torch.allclose(
        stage_state["lm_head.weight"], state["model.embed_tokens.weight"], atol=0.02
    )
    assert job["metrics"][-1]["val_loss"] is not None


@pytest.mark.e2e
def test_pipeline_lora_freezes_base_weights(cluster, corpus, tiny_model):
    _copy_model(cluster, tiny_model)
    dataset = cluster.upload_dataset(corpus, "pipeline-lora")
    job = _create_pipeline_job(cluster, dataset, lora_rank=4)
    assert job["state"] == "completed", job.get("error")
    init = load_state(cluster.root / "coordinator-data" / job["initial_checkpoint"])
    final = load_state(cluster.root / "coordinator-data" / job["latest_checkpoint"])
    lora_keys = [k for k in final if "lora_" in k]
    assert lora_keys
    for key, value in init.items():
        assert torch.equal(final[key], value), f"frozen base weight changed: {key}"
    assert any(bool(final[key].abs().max() > 0) for key in lora_keys if key.endswith("lora_B"))
    metric = job["metrics"][-1]
    assert metric["stage_metrics"][0]["trainable_parameters"] < metric["stage_metrics"][0]["parameters"]
    out = cluster.infer(job["job_id"], "the cat", max_tokens=4, stream=False)
    assert out["tokens"] == 4
    assert out["text"]


@pytest.mark.slow
@pytest.mark.e2e
def test_pipeline_recovers_after_stage_worker_loss(cluster, corpus, tiny_model):
    _copy_model(cluster, tiny_model)
    dataset = cluster.upload_dataset(corpus, "pipeline-failure")
    body = {
        "name": "pipeline-failure",
        "dataset_id": dataset["dataset_id"],
        "model": {"kind": "hf", "model_name": "tinytest"},
        "training": {
            "parallelism": "pipeline",
            "algorithm": "pipeline",
            "rounds": 3,
            "batch_size": 2,
            "seq_len": 16,
            "micro_batches": 8,
            "lr": 1e-3,
            "min_workers": 1,
            "max_workers": 2,
            "pipeline_stages": 2,
            "eval_every": 0,
            "round_timeout_s": 60,
            "seed": 11,
        },
        "auto_start": True,
    }
    r = cluster.client.post(
        "/v1/jobs", content=json.dumps(body), headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    # Wait until a two-stage round exists, then kill the worker that owns the
    # second stage while the round is in flight.
    deadline = time.time() + 60
    killed = None
    while time.time() < deadline:
        job = cluster.client.get(f"/v1/jobs/{job_id}").json()
        stages = job.get("stage_plan") or []
        if len(stages) >= 2 and job.get("current_round"):
            name = stages[1]["node_name"]
            proc = cluster.worker_procs.get(name)
            if proc and proc.poll() is None:
                proc.kill()
                killed = name
                break
        if job["state"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.02)
    assert killed, "never observed an in-flight two-stage round"
    job = cluster.wait_job(job_id, timeout=240)
    assert job["state"] == "completed", (job.get("error"), job.get("events")[-4:])
    assert job["version"] == 3
    assert any(e.get("level") in {"error", "warning"} for e in job["events"])
    # After the loss, a one-stage re-plan runs the model on the surviving node.
    assert any(len(m.get("stage_metrics", [])) == 1 for m in job["metrics"])


@pytest.mark.e2e
def test_hf_pipeline_inference_matches_monolithic(cluster, corpus, tiny_model):
    _copy_model(cluster, tiny_model)
    dataset = cluster.upload_dataset(corpus, "pipeline-infer")
    job = _create_pipeline_job(cluster, dataset)
    assert job["state"] == "completed", job.get("error")
    prompt = "the cat and the dog"
    out = cluster.infer(job["job_id"], prompt, max_tokens=8, stream=False)
    assert out["stages"]
    assert out["tokens"] == 8

    from xdc.model.tokenizer import TokenizerInfo, load_tokenizer

    cfg = LlamaConfig.from_dict(job["model"])
    model = LlamaModel(cfg)
    model.load_hf_state(
        load_state(cluster.root / "coordinator-data" / job["latest_checkpoint"])
    )
    model.eval()
    tok = load_tokenizer(
        cluster.root / "coordinator-data" / "models" / "tinytest",
        TokenizerInfo.from_dict(job["tokenizer"]),
    )
    ids = tok.encode(prompt)
    generated = []
    with torch.no_grad():
        for _ in range(8):
            logits = model(torch.tensor([ids]))[0][0, -1]
            nxt = int(torch.argmax(logits).item())
            generated.append(nxt)
            ids.append(nxt)
    expected = tok.decode(generated)
    assert out["text"] == expected
    assert out["distributed"] is True
    assert len(out["stages"]) == 2


@pytest.mark.slow
@pytest.mark.e2e
def test_hybrid_mps_cpu_pipeline(tmp_path, corpus):
    if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    model_dir = tmp_path / "tiny-hf"
    build_tiny_hf(model_dir, layers=4, hidden=64)
    c = Cluster(
        tmp_path / "hybrid-cluster",
        workers=2,
        backends={"worker-1": "mps", "worker-2": "cpu"},
    )
    try:
        c.start_coordinator()
        c.start_workers()
        _copy_model(c, model_dir)
        dataset = c.upload_dataset(corpus, "hybrid")
        job = _create_pipeline_job(c, dataset, eval_every=1)
        assert job["state"] == "completed", job.get("error")
        devices = {s["device"] for s in job["metrics"][-1]["stage_metrics"]}
        assert devices == {"mps", "cpu"}
        assert job["final_loss"] is not None
        out = c.infer(job["job_id"], "the cat", max_tokens=4, stream=False)
        assert out["tokens"] == 4
        assert {s["backend"] for s in out["stages"]} == {"mps", "cpu"}
    finally:
        c.cleanup()


@pytest.mark.slow
@pytest.mark.e2e
def test_island_placement_constraint_and_inference(tmp_path, corpus):
    model_dir = tmp_path / "tiny-hf"
    build_tiny_hf(model_dir, layers=4, hidden=64)
    c = Cluster(
        tmp_path / "island-cluster",
        workers=2,
        labels={
            "worker-1": {"island": "island-a"},
            "worker-2": {"island": "island-b"},
        },
    )
    try:
        c.start_coordinator()
        c.start_workers()
        _copy_model(c, model_dir)
        dataset = c.upload_dataset(corpus, "island")
        job = _create_pipeline_job(
            c,
            dataset,
            min_workers=1,
            max_workers=2,
            island="island-a",
        )
        assert job["state"] == "completed", job.get("error")
        assert len(job["stage_plan"]) == 1
        assert {s["node_name"] for s in job["stage_plan"]} == {"worker-1"}
        out = c.infer(job["job_id"], "the cat", max_tokens=3, stream=False)
        assert out["tokens"] == 3
        assert {s["node_name"] for s in out["stages"]} == {"worker-1"}
    finally:
        c.cleanup()


@pytest.mark.slow
@pytest.mark.e2e
def test_island_blocks_minimize_cross_island_hops(tmp_path, corpus):
    model_dir = tmp_path / "tiny-hf"
    build_tiny_hf(model_dir, layers=4, hidden=64)
    c = Cluster(
        tmp_path / "island-block-cluster",
        workers=4,
        labels={
            "worker-1": {"island": "a"},
            "worker-2": {"island": "b"},
            "worker-3": {"island": "a"},
            "worker-4": {"island": "b"},
        },
    )
    try:
        c.start_coordinator()
        c.start_workers()
        _copy_model(c, model_dir)
        dataset = c.upload_dataset(corpus, "island-blocks")
        job = _create_pipeline_job(
            c,
            dataset,
            min_workers=4,
            max_workers=4,
            pipeline_stages=4,
        )
        assert job["state"] == "completed", job.get("error")
        islands = [s["island"] for s in job["stage_plan"]]
        assert len(islands) == 4
        assert sum(a != b for a, b in zip(islands, islands[1:])) <= 1
        assert job["metrics"][-1]["island_hops"] <= 1
        out = c.infer(job["job_id"], "the cat", max_tokens=3, stream=False)
        infer_islands = [s["island"] for s in out["stages"]]
        assert sum(a != b for a, b in zip(infer_islands, infer_islands[1:])) <= 1
    finally:
        c.cleanup()
