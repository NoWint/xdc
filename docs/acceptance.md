# XDC Acceptance Report

**Date:** 2026-11-08
**Version:** 0.1.0
**Scope:** one physical Apple M4 Mac running a real coordinator and two real
worker processes. This report distinguishes what was executed from what remains
unverified. Nothing below is claimed from code inspection alone.

## 1. Verdict

The core protocol is functional and tested end to end on macOS: two independent
workers register with Ed25519 identities, open outbound tunnels, train on
disjoint data shards, upload real pseudo-gradients, the coordinator aggregates
them, parameters change, checkpoints survive a coordinator restart, a killed
worker is detected and the job completes, and two-stage layer-sharded inference
matches monolithic greedy generation byte for byte.

A **second physical device**, Windows execution, CUDA/ROCm/Intel XPU, Intel Mac,
LoRA, RL and tensor parallelism are **not verified**. They must not be described
as working. See §5.

## 2. Automated tests

```text
$ pytest -q
18 passed, 9 warnings in 61.06s
```

The suite launches real subprocesses with separate identities, ports, data
directories and caches. It is not an in-process mock.

| Test | What it actually proves |
|---|---|
| `test_two_worker_training_and_distributed_inference` | 2 workers, 6 rounds, 12,288 trained tokens, 2/2 updates every round, loss falls, init/final SHA and tensors differ, 2-stage inference matches one-process greedy bytes, SSE bytes match non-stream bytes |
| `test_checkpoint_resume_after_coordinator_restart` | Pause after a committed round, graceful coordinator restart, resume, complete more rounds, new checkpoint exists, inference from the resumed model matches |
| `test_worker_failure_detection_and_inference_reschedule` | Kill worker 2 mid-job, observe loss/retry events, complete 8 rounds on the remaining worker, single-stage fallback (`distributed=false`) still matches monolithic bytes |
| `test_dataset_download_is_shard_only` | Worker-signed request returns only one contiguous shard, length `<` full split, bytes equal that shard, header SHA matches |
| `test_tls_end_to_end` | HTTPS coordinator, CA verified by worker, training completes, untrusted certificate is rejected |
| `test_admin_auth_required` | Admin stats/nodes return 401 anonymously; unsigned worker assignment returns 401; `/healthz` stays public |
| `test_crypto.py` | Identity, canonical signing, replay/timestamp rejection, TLS material |
| `test_model.py` | Monolithic vs pipeline exactness, KV-cache generation, checkpoint round-trip, FP16, native/Python aggregation parity, outlier rejection, local training, data split, int8 round-trip |

## 3. Benchmark evidence

Full details: [benchmark.md](benchmark.md). Same host, 2 CPU workers, 3 rounds,
30,720 tokens:

| Measurement | Result |
|---|---:|
| Distributed total wall clock | 5.884 s |
| Distributed round throughput | 5,827 tok/s |
| Single-process baseline (4 threads) | 7,161 tok/s |
| End-to-end training ratio | 0.814× |
| Compute-path ratio (local + aggregation) | 1.067× |
| Loss (3 rounds) | 5.552 → 2.952 |
| Distributed inference, 24 tokens, 2 stages | 130.3 ms, 184.2 tok/s |
| Monolithic CPU inference | 33.0 ms, 726.6 tok/s |
| Native C++ merge vs Python | 9.84 ms vs 29.21 ms (2.97×) |

Distributed greedy bytes exactly equal monolithic CPU bytes. The distributed
inference path is slower than one-process CPU for the 0.9M demo model; its value
is stage memory distribution, not single-host speed.

## 4. Manual smoke tests

- `xdc init` generated tokens, TLS CA/server certificate, coordinator and worker
  YAML; `xdc coordinator` started over HTTPS with the generated config.
- `/healthz` → 200, `/dashboard` → 200 (10,770-byte HTML), `/v1/stats` with the
  admin bearer → 200, `/metrics` → 200, `/v1/nodes` → 200.
- `xdc doctor --json` truthfully reports CPU and MPS available; CUDA, ROCm, XPU
  and NPU unavailable with reasons; MLX detected but not used by the engine.

## 5. Requirement matrix

| Requirement | Status | Evidence / limitation |
|---|---|---|
| Two nodes discover and connect | **verified across processes** | 2 worker daemons, distinct Ed25519 IDs, outbound tunnels; physical 2-device run unverified |
| Both nodes contribute compute | **verified** | every round `updates_received=2`, `updates_used=2`; metric worker set has 2 IDs |
| Genuine distributed training | **verified across processes** | each worker runs its own AdamW steps on its own shard; coordinator only aggregates |
| Parameters change | **verified** | init/final safetensors SHA differ and tensors differ; loss decreases |
| Checkpoint save/restore | **verified** | pause → coordinator restart → resume → completion → inference |
| Worker exit handling | **verified** | kill worker; lost/retry events; job completes with remaining nodes |
| Cross-device sharded inference correct | **verified across processes** | contiguous 2-stage pipeline, KV cache on workers, exact greedy byte parity |
| Single vs multi benchmark | **verified** | §3; honest slowdown for end-to-end, compute at parity |
| macOS install/build/basic use | **verified locally** | Apple M4, Python 3.11, native extension compiled |
| Windows install/build/basic use | **unverified** | GitHub Actions matrix is defined but has not run in this environment |
| Second physical device over LAN | **unverified** | only one physical host available; no multi-machine latency claim |
| NVIDIA CUDA | **unverified** | code path and honest detection present, no hardware |
| AMD ROCm | **unverified** | code path and honest detection present, no hardware |
| Intel XPU | **unverified** | code path and honest detection present, no hardware |
| Intel Mac / Windows arm64 | **unverified** | no hardware |
| TLS + signed requests | **verified** | TLS e2e, auth e2e, replay/timestamp unit tests |
| Shard-only dataset delivery | **verified** | dedicated e2e test |
| Dashboard | **verified via HTTP smoke test** | served 200; browser interaction not automated |
| LoRA | **not implemented** | no adapter training or adapter transport |
| RL / preference optimization | **not implemented** | no reward model, rollouts or policy-gradient trainer |
| Tensor parallelism | **not implemented** | pipeline parallelism only |
| SFT | **basic data path only** | JSONL instruction records are formatted into text and trained with next-token loss; no chat template or prompt-token masking |
| MLX execution | **detected, not used** | engine runs PyTorch CPU/MPS/CUDA/ROCm/XPU |
| Public swarm / DHT / hole punching | **not implemented** | coordinator relay only; workers need outbound reachability to it |
| Byzantine fault tolerance | **not claimed** | norm sanity and median filtering with ≥3 updates; not proof-of-learning or secure aggregation |
| Confidential computing / encrypted storage | **not implemented** | TLS in transit, plain data at rest |

## 6. Issues found and fixed during acceptance

- A stale dashboard path returned HTTP 500; fixed and re-tested over TLS.
- Aggressive heartbeat-only liveness discarded completed work on a loaded host.
  Rounds now treat a live tunnel as liveness, re-admit a valid late update from a
  worker declared lost, and retry only after a bounded recovery window.
- Worker heartbeat used the 60 s general request timeout and forced
  re-registration on any transient error; it now fails fast and preserves its
  tunnel token on transient failures.
- Coordinator heartbeats fsynced the whole state document every request; liveness
  saves are now throttled (round commits are never throttled).
- Upload validation ran safetensors parsing and norm computation on the event
  loop; it now runs in a worker thread under the per-job lock.
- The native aggregation path paid per-tensor Python/NumPy overhead; it now uses
  one flat contiguous buffer and validates finiteness inside C++ (2.97×).

## 7. Reproduction

```bash
python -m pip install -e ".[dev]"
python setup.py build_ext --inplace
pytest -q
python benchmarks/benchmark.py --workdir /tmp/xdc-bench \
  --output benchmarks/latest.json --rounds 3 --local-steps 10 --workers 2
```

Hardware-specific claims can only be upgraded by running the relevant test or
`xdc doctor` on that hardware and recording the output here.
