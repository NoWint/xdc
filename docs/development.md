# XDC Development Guide

## 1. Layout

```
xdc/
  cli.py                 CLI
  config.py              YAML config dataclasses
  crypto.py              Ed25519 identities, request signing, TLS bootstrap
  protocol.py            Pydantic wire models
  native.py              optional C++ facade + fallbacks
  tensorwire.py          pickle-free float32 wire encoding
  model/                 GPT, pipeline stage, data, checkpoint, training
  coordinator/           FastAPI app, state, scheduler, DiLoCo engine, inference, tunnel
  worker/                daemon, signed client, hardware, runtime
  dashboard/index.html   monitoring UI
native/xdc_native.cpp    C++17 extension
tests/                   unit + multi-process e2e
benchmarks/benchmark.py  single vs multi comparison
```

## 2. Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,hf]"
python setup.py build_ext --inplace
python -m pytest -q -m "not e2e"
```

On Windows PowerShell, activate `.venv\Scripts\Activate.ps1`.

The native extension is optional. `xdc.native.AVAILABLE` is false if compilation failed; Python fallbacks keep tests passing.

## 3. Tests

- `tests/test_crypto.py`: identity, signatures, replay, TLS material.
- `tests/test_model.py`: monolithic/pipeline equivalence, KV-cache generation, checkpoints, FP16 compression, native/Python aggregation parity, outlier rejection, local training, data split.
- `tests/test_e2e.py`: real coordinator + two worker subprocesses, two-worker training, exact distributed inference, checkpoints, coordinator restart/resume, worker kill/failure, TLS, auth.
- `tests/test_pipeline.py`: Llama/HF parity, stage slicing, pipeline AdamW vs monolithic, tied heads, LoRA, HF inference parity, worker-loss recovery, MPS+CPU hybrid, island blocks.
- `tests/conftest.py`: `Cluster` harness (free port, generated tokens, subprocess lifecycle, log capture, API helpers).

The e2e tests use separate CPU workers so they are deterministic and do not require MPS. They run on Windows/macOS/Linux in CI. Tests that need unavailable hardware are skipped or explicitly marked unverified in the acceptance report.

Step-by-step manual/e2e/HF/pipeline test guide (中文):
[testing-tutorial.md](testing-tutorial.md).

Useful commands:

```bash
pytest -q                                  # everything
pytest -q -m "not e2e"                     # fast
pytest -q -m e2e                           # real processes
pytest tests/test_e2e.py::test_tls_end_to_end -q -s
```

When an e2e test fails, the `Cluster` harness has already written `coordinator.log` and `worker-N.log` under the pytest temp directory. Call `cluster.dump_logs()` in the test/fixture for a consolidated tail.

## 4. Adding a backend

1. Extend `xdc/worker/hardware.py::_backend_entries` with a truthful availability probe and device entry.
2. Add it to `_pick` / `_benchmark`.
3. Ensure `xdc/worker/runtime.py::device_for` returns it only when available.
4. Add it to `xdc/cli.py` choices if users can force it.
5. Add a test that verifies unavailable hardware is reported unavailable (do not test hardware you do not have).
6. Document the verification status.

## 5. Adding a model

The pipeline assumes a decoder-only causal LM with contiguous layer slices and per-stage KV caches. To add another architecture:

1. Provide a `PipelineStage`-compatible module with `layer_start/layer_end`, `layer_keys`, `load_global_state`, and `forward(input_ids|hidden, past_kv, pos_offset, use_cache)`.
2. Keep state-dict keys stable across monolithic and stage modules. For HF safetensors, add an explicit `global_key` / `layer_state` mapping and a `full_state` loader that rejects unexpected tensors.
3. Add monolithic/pipeline equivalence tests against greedy generation.
4. Register the architecture in `xdc/model/registry.py`; add a `config.json` parser and a `tokenizer.json` tokenizer adapter. The engine supports both the built-in byte tokenizer and HF `tokenizers`.
5. Reject unsupported features explicitly (e.g. `rope_scaling`, quantization formats, remote code) instead of producing wrong logits.
6. Update `ModelConfig` / `LlamaConfig` validation and documentation.

## 6. Altering the training algorithm

`xdc/coordinator/engine.py` is intentionally the only place that decides when a round commits. Keep these invariants:

- The global model version is advanced atomically after aggregation.
- A worker update is bound to a round and base version.
- Late updates never apply to a newer model.
- A restart discards an uncommitted round.
- Checkpoint and optimizer state are written before the job advances in memory (or clearly documented otherwise).

Current resume durability is "last committed round"; an in-flight round is intentionally lost.

## 7. C++ extension

`setup.py` builds `xdc._native` with pybind11 if a compiler exists. `OptionalBuildExt` catches compilation failure and leaves the Python fallback.

Functions:

- `aggregate(...)`: flat-buffer robust aggregation + outer momentum; this is the primary path used by `aggregate_deltas`.
- `aggregate_tensors(...)`: per-tensor equivalent retained for compatibility and parity tests.
- `topk_sample(...)`: deterministic sampling.
- `byte_batch(...)`: byte shard sampling.
- `quantize_int8/dequantize_int8(...)`: block quantizer (not wired to transport by default).

`aggregate_deltas` concatenates the state dict into one float32 buffer, performs
non-finite validation inside C++, and returns views of one output buffer. Python
fallbacks remain available when the extension is absent.

Build locally:

```bash
python setup.py build_ext --inplace
python -c "from xdc import native; print(native.VERSION)"
```

Windows needs MSVC Build Tools. The CI matrix verifies compilation on windows-latest.

## 8. Benchmark

```bash
python benchmarks/benchmark.py \
  --workdir /tmp/xdc-bench \
  --rounds 3 --local-steps 10 --workers 2 \
  --output benchmarks/latest.json
```

It launches coordinator + workers, runs training, compares an equivalent single-process token workload, warms the pipeline, compares distributed greedy bytes to monolithic CPU/MPS, and measures native aggregation. It always labels the result as same-host.

## 9. Release checks

- `pytest -q` green on macOS arm64/Intel, Windows x64 and Linux x64. The CI
  workflow requires the native extension to build, not just fall back.
- On macOS x86_64, torch is 2.2.2 and needs NumPy 1.x; the platform marker in
  `pyproject.toml` pins it. Do not replace it with a global NumPy pin.
- `xdc init`, `xdc coordinator`, `xdc worker`, `xdc train`, `xdc infer` manually exercised.
- `xdc doctor` shows only truthfully available backends.
- `git status` contains no keys, tokens, datasets from private users, checkpoints or temporary logs.
- Update `docs/acceptance.md` with new evidence; do not change an "unverified" row to "verified" without hardware evidence.
