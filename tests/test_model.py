from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from xdc import native
from xdc.model import train as train_mod
from xdc.model.checkpoint import load_state, save_state, serialize_state_bytes, deserialize_state_bytes, sha256_file
from xdc.model.data import shard_bounds, split_train_val
from xdc.model.gpt import GPT, ModelConfig, PipelineStage
from xdc.model.train import aggregate_deltas, evaluate_loss, train_local


def tiny_cfg():
    return ModelConfig(vocab_size=256, n_layer=4, n_head=4, n_embd=64, block_size=32, dropout=0.0)


def test_pipeline_matches_monolithic_forward_and_generation():
    torch.manual_seed(3)
    cfg = tiny_cfg()
    model = GPT(cfg).eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    stages = [PipelineStage(cfg, 0, 2).eval(), PipelineStage(cfg, 2, 4).eval()]
    for stage in stages:
        stage.load_global_state(state)
    prompt = torch.tensor([[10, 20, 30, 40, 50]])
    with torch.no_grad():
        ref_logits, _ = model(prompt)
        hidden = None
        logits = None
        for i, stage in enumerate(stages):
            hidden, logits = stage(input_ids=prompt if i == 0 else None, hidden=hidden)
    assert torch.allclose(ref_logits, logits, atol=1e-6)

    # KV-cached greedy generation must match stage-by-stage decoding.
    with torch.no_grad():
        expected = model.generate(prompt, 6, temperature=0)
    caches = [None, None]
    ids = prompt
    got = []
    pos = 0
    with torch.no_grad():
        for step in range(6):
            cond = ids if step == 0 else ids[:, -1:]
            hidden = None
            logits = None
            for i, stage in enumerate(stages):
                hidden, logits, cache = stage(
                    input_ids=cond if i == 0 else None,
                    hidden=hidden,
                    past_kv=caches[i],
                    pos_offset=pos,
                    use_cache=True,
                )
                caches[i] = cache
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            got.append(int(nxt))
            pos = ids.shape[1] - 1
    assert got == expected[0, prompt.shape[1] :].tolist()


def test_noisy_generation_still_matches_with_pipeline():
    torch.manual_seed(9)
    cfg = tiny_cfg()
    model = GPT(cfg).eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    s0, s1 = PipelineStage(cfg, 0, 2).eval(), PipelineStage(cfg, 2, 4).eval()
    s0.load_global_state(state)
    s1.load_global_state(state)
    prompt = torch.randint(0, 256, (1, 7))
    with torch.no_grad():
        expected = model.generate(prompt, 5, temperature=0)
        h = None
        logits = None
        for i, stage in enumerate([s0, s1]):
            h, logits = stage(input_ids=prompt if i == 0 else None, hidden=h)
    assert expected[0, prompt.shape[1]] == logits[:, -1].argmax(-1)[0]


def test_checkpoint_roundtrip_and_fp16_compression(tmp_path):
    state = {f"p{i}": torch.randn(3, 4) for i in range(4)}
    path = tmp_path / "s.safetensors"
    sha = save_state(path, state)
    assert sha == sha256_file(path)
    loaded = load_state(path)
    for key in state:
        assert torch.equal(loaded[key], state[key])
    body = serialize_state_bytes({k: v for k, v in state.items()}, dtype=torch.float16)
    back = deserialize_state_bytes(body)
    assert all(v.dtype == torch.float16 for v in back.values())
    assert all(back[k].shape == state[k].shape for k in state)


def test_aggregate_deltas_native_and_python_parity(monkeypatch):
    torch.manual_seed(2)
    base = {"w": torch.randn(100), "b": torch.randn(10)}
    d1 = {"w": torch.randn(100) * 0.01, "b": torch.randn(10) * 0.01}
    d2 = {"w": torch.randn(100) * 0.01, "b": torch.randn(10) * 0.01}
    kwargs = dict(outer_lr=0.8, momentum=0.9, nesterov=True)
    n_state, n_mom, n_stats = aggregate_deltas(base, [d1, d2], [1.0, 3.0], **kwargs)
    assert n_stats["native"] is native.AVAILABLE
    monkeypatch.setattr(train_mod.native, "AVAILABLE", False)
    p_state, p_mom, p_stats = aggregate_deltas(base, [d1, d2], [1.0, 3.0], **kwargs)
    for key in base:
        assert torch.allclose(n_state[key].float(), p_state[key].float(), atol=1e-6)
        assert torch.allclose(n_mom[key].float(), p_mom[key].float(), atol=1e-5)


def test_outlier_update_filtered_with_three_workers():
    base = {"w": torch.zeros(32)}
    good = [{"w": torch.ones(32) * 0.01} for _ in range(3)]
    bad = {"w": torch.ones(32) * 100.0}
    state, _, stats = aggregate_deltas(base, good + [bad], [1, 1, 1, 1], momentum=0.0)
    assert stats["updates_used"] == 3
    assert 3 in stats["dropped_indices"]
    assert torch.allclose(state["w"], torch.full((32,), 0.01), atol=1e-6)


def test_invalid_update_rejected():
    base = {"w": torch.zeros(4)}
    with pytest.raises(ValueError):
        aggregate_deltas(base, [{"w": torch.tensor([float("nan"), 0, 0, 0])}], [1], momentum=0.0)
    with pytest.raises(ValueError):
        aggregate_deltas(base, [{"other": torch.zeros(4)}], [1], momentum=0.0)


def test_local_training_changes_parameters(tmp_path):
    cfg = ModelConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=32, block_size=32)
    torch.manual_seed(0)
    model = GPT(cfg)
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    data = (Path(__file__).parents[1] / "datasets" / "tiny_stories.txt").read_bytes()[:20000]
    res = train_local(
        model,
        base,
        data,
        local_steps=3,
        batch_size=2,
        seq_len=32,
        lr=1e-3,
        device="cpu",
        seed=1,
        grad_clip=1.0,
    )
    assert res.tokens == 3 * 2 * 32
    assert res.update_norm > 0
    assert any(not torch.equal(base[k], base[k] + res.delta[k]) for k in base)
    assert math.isfinite(res.loss)


def test_validation_loss_computable():
    cfg = ModelConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=32, block_size=32)
    torch.manual_seed(0)
    model = GPT(cfg)
    raw = (Path(__file__).parents[1] / "datasets" / "tiny_stories.txt").read_bytes()[:30000]
    train_data, val_data = split_train_val(raw, 0.2, min_val_bytes=2048)
    loss = evaluate_loss(model.state_dict(), cfg, val_data, batch_size=2, seq_len=32, max_batches=2)
    assert 0 < loss < 20
    a, b = shard_bounds(100, 0, 2), shard_bounds(100, 1, 2)
    assert a == (0, 50) and b == (50, 100)


def test_native_quantize_roundtrip():
    if not native.AVAILABLE:
        pytest.skip("native extension not built")
    import numpy as np

    x = np.random.default_rng(0).standard_normal(10000).astype("float32")
    q, scales = native.quantize_int8(x, 1024)
    y = native.dequantize_int8(q, scales, 1024)
    err = np.abs(x - y)
    assert err.max() < 0.05
    assert q.nbytes * 4 == x.nbytes  # int8 codes are 4x smaller than float32
