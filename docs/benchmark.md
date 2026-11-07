# XDC Benchmarks

These are **same-host** measurements from one Apple M4 Mac. They prove that the
distributed paths execute and quantify their overhead; they are **not** a
multi-machine scaling claim. Wall-clock numbers move with background load, so
treat them as one reproducible sample, not a hardware rating.

Raw result: [`benchmarks/latest.json`](../benchmarks/latest.json).

## 1. Environment

| | |
|---|---|
| Host | MacBook Air, Apple M4, 10 CPU cores, 16 GB unified memory |
| OS | macOS 27.2 arm64 |
| Python / PyTorch | 3.11.15 / 2.11.0 |
| XDC | 0.1.0, native extension `xdc-native 0.1.0` |
| Workers | 2 independent CPU worker processes, 2 threads each |
| Model | 4-layer, 128-dim, 4-head byte GPT, 875,264 parameters |
| Training | DiLoCo/decoupled, FP16 updates, 3 rounds, 10 local steps/round |
| Batch | 4 sequences × 128 bytes = 512 tokens/step, 30,720 tokens total |
| Inference | 24 greedy tokens from `Mira found a brass key near` |

The distributed run and the single-process baseline use the same total token
workload. The baseline runs 60 AdamW steps in one process with 4 threads, matching
the 2 workers × 2 threads used by the distributed run.

## 2. Training

| Metric | Distributed (2 workers) | Single process (4 threads) |
|---|---:|---:|
| Total job wall clock | 5.884 s | 4.290 s |
| Round open → commit wall | 5.272 s | — |
| Tokens / round wall | 5,827 /s | 7,161 /s |
| Local-compute critical path | 3.976 s | — |
| Aggregation (3 rounds) | 0.046 s | — |
| Compute-only wall (local + aggregate) | 4.022 s | — |
| Compute-only tokens / s | 7,637 /s | 7,161 /s |
| End-to-end training ratio | **0.814×** | 1.000× |
| Compute-path ratio | **1.067×** | 1.000× |
| Initial → final validation loss | 5.552 → 2.952 | — |
| Updates per round | 2 / 2 used in all 3 rounds | — |

Reading the table:

- The **compute path is at parity or slightly faster** (1.067×) than a
  single process with the same number of CPU threads. The two independent
  processes avoid some single-process scheduling effects; this is not evidence
  of linear scaling.
- **End-to-end training is slower (0.814×)**. The gap is coordinator
  evaluation, checkpoint fsync, weight/download transfer and round
  coordination, not aggregation: C++/native aggregation costs only 46 ms for
  three rounds of 875k parameters.
- On one host, data-parallel DiLoCo cannot create more CPU cores. Its value is
  pooling **memory across devices** and tolerating heterogeneous/sleeping
  workers, which this API-level benchmark cannot show. A two-physical-device
  run is still unverified; see the acceptance report.

Per-round local times were 2.42 s, 0.80 s and 0.75 s; the first round includes
model/cache warm-up. Native aggregation per round was ~15 ms.

## 3. Inference

The model is split into two contiguous stages, one per worker. Stage 0 owns the
embeddings and blocks 0–1; stage 1 owns blocks 2–3, the final norm and the head.
KV caches stay on the stage owner. The coordinator samples the final logits and
streams bytes with SSE.

| Path | Tokens | Time to first token | Total | Throughput |
|---|---:|---:|---:|---:|
| Distributed, 2 stages, CPU | 24 | 48.2 ms | 130.3 ms | 184.2 tok/s |
| Monolithic local CPU | 24 | — | 33.0 ms | 726.6 tok/s |
| Monolithic local Apple MPS | 24 | — | 411.8 ms | 58.3 tok/s |

The distributed greedy output is **byte-for-byte identical** to the monolithic
CPU output (`IHRoZSB0IHRoZSB0aGUgdCB0aGUgdGhl` base64) and was asserted by the
benchmark. It is 0.254× local CPU on this tiny model because every token crosses
the coordinator and a WebSocket hop, while a 0.9M model fits trivially in one
process. It is 3.16× the MPS path here because tiny Metal kernels are
dispatch-bound. The pipeline's purpose is to run a model whose stage weights do
not fit on one device, not to make a tiny model faster on one box.

## 4. Native C++ aggregation

Measured through the real `aggregate_deltas` state-dict path (not a synthetic
array microbenchmark), 875,264 parameters, 2 updates, 20 iterations:

| Path | Time per merge |
|---|---:|
| C++ flat-buffer merge | 9.84 ms |
| Python fallback | 29.21 ms |
| Speedup | **2.97×** |

The C++ path flattens all parameter tensors into one contiguous float32 buffer,
computes global update norms and the robust filter in C++, performs the
token-weighted outer update and Nesterov momentum in one parallel pass, and
returns views of one output buffer. If the extension is unavailable, the Python
path remains correct and is used automatically.

## 5. Reproduce

```bash
python -m pip install -e ".[dev]"
python setup.py build_ext --inplace          # optional native extension
python benchmarks/benchmark.py \
  --workdir /tmp/xdc-bench \
  --output benchmarks/latest.json \
  --rounds 3 --local-steps 10 --workers 2
```

The benchmark starts a real coordinator and real worker daemons, uploads the
corpus, trains, compares distributed greedy bytes with monolithic CPU/MPS bytes,
and measures native aggregation. It requires no privileged operations.

## 6. Caveats

- Same physical host. Workers share CPU, memory bandwidth and the page cache.
- CPU only for the training comparison. CUDA/ROCm/XPU code paths are unverified
  on this hardware.
- The MPS comparison uses the tiny demo model and is explicitly pessimistic;
  `auto` selects CPU below 50M parameters for this reason.
- No multi-machine latency, packet loss, NAT relay or heterogeneous-device
  numbers are claimed. The acceptance report lists those as unverified.
- The data-parallel algorithm's communication advantage grows with model size
  and network distance; this benchmark does not extrapolate.
