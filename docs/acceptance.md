# XDC Acceptance Report

**Date:** 2026-11-08
**Version:** 0.1.0
**Scope:** one physical Apple M4 Mac running a real coordinator and two real
worker processes. This report distinguishes what was executed from what remains
unverified. Nothing below is claimed from code inspection alone.

## 1. Verdict

The core protocol is functional and tested end to end on macOS and in a
four-platform CI matrix (macOS 15 arm64, macOS 15 Intel, Windows x64, Ubuntu
x86_64): two independent workers register with Ed25519 identities, open outbound
tunnels, train on disjoint data shards, upload real pseudo-gradients, the
coordinator aggregates them, parameters change, checkpoints survive a coordinator
restart, a killed worker is detected and the job completes, and two-stage
layer-sharded inference matches monolithic greedy generation byte for byte.

Pipeline-parallel training, a self-contained Llama/Qwen HF loader, LoRA pipeline
training and HF layer-sharded inference were added and verified on this host.
A **second physical device**, CUDA/ROCm/Intel XPU, physical Windows 10/11
desktop behavior, Windows arm64, QLoRA, RL, tensor parallelism, islands and
intra-node CPU/GPU hybrid stages are **not verified**. See §5.

## 2. Automated tests

```text
$ pytest -q
30 passed, 19 warnings
```

The P1 pipeline module adds six real-process/unit tests and one inference parity
test:

| Test | What it proves |
|---|---|
| `test_llama_logits_and_gradients_match_transformers` | self-contained Llama matches HF `LlamaForCausalLM` logits and every gradient |
| `test_stage_files_really_shard_the_model` | three stage files are each smaller than the full checkpoint, parameter counts sum exactly, and composed logits are identical |
| `test_pipeline_step_matches_monolithic` | one real two-worker pipeline step matches monolithic AdamW within 4e-6 max abs error (fp32 reduction order) |
| `test_pipeline_tied_head_is_synchronized` | tied global checkpoints omit the duplicate head and the final stage carries a synchronized copy |
| `test_pipeline_lora_freezes_base_weights` | every base tensor is bitwise unchanged; adapters train; LoRA inference succeeds |
| `test_hf_pipeline_inference_matches_monolithic` | tokenized HF prompt through two stages + KV cache produces exactly the monolithic greedy text |
| `test_pipeline_recovers_after_stage_worker_loss` | killing the second-stage owner mid-round leads to abort, one-stage re-plan on the survivor, and completion |
| `test_hybrid_mps_cpu_pipeline` | stage 0 on Metal, stage 1 on CPU; both train, evaluate and serve inference |
| `test_island_placement_constraint_and_inference` | a pinned job and its inference never leave the requested island |
| `test_island_blocks_minimize_cross_island_hops` | 4 workers in 2 islands produce a contiguous island block and `island_hops <= 1` |

The suite launches real subprocesses with separate identities, ports, data
directories and caches. It is not an in-process mock.

### 2.1 GitHub Actions matrix

Run [`42408886904`](https://github.com/NoWint/xdc/actions/runs/42408886904) on
commit `caec8bb` passed on all four runners. Each job installed the package,
built the native extension, ran `xdc doctor`, initialised a cluster, ran the 14
unit tests and the 6 real-process e2e tests.

| Runner | Reported platform | Native | Unit | E2E |
|---|---|---|---|---|
| `macos-15` | macOS 15.8.2 arm64 | `available: true` | 14 passed, 41.50 s | 6 passed, 111.96 s |
| `macos-15-intel` | macOS 15.7.9 x86_64 | `available: true` | 14 passed, 38.05 s | 6 passed, 118.79 s |
| `windows-latest` | Windows build 26100 x64 | `available: true` | 14 passed, 3.71 s | 6 passed, 77.97 s |
| `ubuntu-latest` | Linux 7.0 x86_64 (glibc 2.39) | `available: true` | 14 passed, 1.94 s | 6 passed, 47.65 s |

The Windows runner is Windows Server, not a physical Windows 10/11 desktop.
The code path, native build, process lifecycle, TCP/TLS and e2e training are
verified there; desktop-specific firewall/installer behavior is not.

### 2.2 Test coverage

| Test | What it actually proves |
|---|---|
| `test_two_worker_training_and_distributed_inference` | 2 workers, 6 rounds, 12,288 trained tokens, 2/2 updates every round, loss falls, init/final SHA and tensors differ, 2-stage inference matches one-process greedy bytes, SSE bytes match non-stream bytes |
| `test_checkpoint_resume_after_coordinator_restart` | Pause after a committed round, graceful coordinator restart, resume, complete more rounds, new checkpoint exists, inference from the resumed model matches |
| `test_worker_failure_detection_and_inference_reschedule` | Kill worker 2 mid-job, observe loss/retry events, complete 8 rounds on the remaining worker, single-stage fallback (`distributed=false`) still matches monolithic bytes |
| `test_dataset_download_is_shard_only` | Worker-signed request returns only one contiguous shard, length `<` full split, bytes equal that shard, header SHA matches |
| `test_tls_end_to_end` | HTTPS coordinator, CA verified by worker, training completes, untrusted certificate is rejected |
| `test_admin_auth_required` | Admin stats/nodes return 401 anonymously; unsigned worker assignment returns 401; `/healthz` stays public |
| `test_crypto.py` | Identity, canonical signing, replay/timestamp rejection, TLS material |
| `test_model.py` | Monolithic vs pipeline exactness, KV-cache generation, checkpoint round-trip, FP16, native/Python aggregation parity, outlier rejection, update-metadata bounds, local training, data split, int8 round-trip |
| `test_tunnel.py` | A replaced tunnel does not fail RPCs belonging to the newer connection |

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

### 3.1 SmolLM2-135M real pipeline run

`benchmarks/smollm2_135m_pipeline.json` contains the machine-readable record.
Two real worker processes on one Apple M4, CPU backend, LoRA rank 4:

| Measurement | Result |
|---|---:|
| Model | `HuggingFaceTB/SmolLM2-135M-Instruct`, 134,515,008 params, tied embeddings |
| Stage plan | 17 layers on worker-1 + 13 layers on worker-2 |
| Round 0 | train 1.9865, val 3.7262, 40.6 s |
| Round 1 | train 1.9758, val 3.7043, 37.7 s |
| Stage 0 | 356.7 MB params, 5.5 MB Adam, 2.77 MB grads, 691,968 trainable |
| Stage 1 | 299.5 MB params, 4.2 MB Adam, 2.12 MB grads, 529,152 trainable |
| Activation wire | 589,824 bytes per round (fp16 hidden states) |
| Full-model Adam would have been | ~1.08 GB of optimizer state on one worker |
| Warm inference | 8 tokens, TTFT 5.86 s, 6.31 s total (CPU, 19+11 layers) |
| Warm inference after cache | 2.63 s TTFT, 3.13 s total |

The full 134.5M-parameter AdamW optimizer would need
`2 * 134.5M * 4 bytes = 1.08 GB`. The run sharded it into 5.5 MB + 4.2 MB of
adapter optimizer state; even full fine-tuning would have split it across two
processes. The full model file (257 MB) was never loaded by a worker; each
worker loaded only its 338 MB / 284 MB stage slice.

### 3.2 Hybrid MPS + CPU SmolLM2 run

`benchmarks/smollm2_135m_hybrid.json`. One worker process used Metal (MPS), the
other CPU; the stage planner put 18 layers on MPS and 12 on CPU:

| Measurement | Result |
|---|---:|
| Initial / final val loss | 3.7625 → 3.7446 |
| Tokens | 256 |
| End-to-end wall | 43.5 s |
| MPS stage | 371.1 MB params, 5.86 MB Adam, 18 layers |
| CPU stage | 285.1 MB params, 3.91 MB Adam, 12 layers |
| Inference | 8 tokens; 17.95 s first token (cold stage generation + MPS warmup) |
| Inference stages | 16 layers MPS + 14 layers CPU |

This proves a real heterogeneous device path across processes. It does **not**
prove inter-node GPU speedup or WAN island latency.

## 4. Manual smoke tests

- `xdc init` generated tokens, TLS CA/server certificate, coordinator and worker
  YAML; `xdc coordinator` started over HTTPS with the generated config.
- `/healthz` → 200, `/dashboard` → 200 (10,770-byte HTML), `/v1/stats` with the
  admin bearer → 200, `/metrics` → 200, `/v1/nodes` → 200.
- `xdc doctor --json` truthfully reports CPU and MPS available; CUDA, ROCm, XPU
  and NPU unavailable with reasons; MLX detected but not used by the engine.
- Full CLI smoke: `xdc init` → coordinator + two workers → `xdc train` (1 round,
  512 tokens, completed, loss 5.535 → 4.975) → `xdc jobs` → `xdc infer` (8
  tokens, 2 stages, `distributed=true`). The dashboard/HTTP endpoints were
  exercised in the same run.

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
| macOS Apple Silicon install/build/basic use | **verified locally and in CI** | Apple M4 local run; `macos-15` arm64 CI: 12 unit + 6 e2e, native built |
| macOS Intel install/build/basic use | **verified in CI** | `macos-15-intel` x86_64 CI: 12 unit + 6 e2e, native built; no Metal GPU claim |
| Windows x64 install/build/basic use | **verified in CI** | `windows-latest` (Windows build 26100) x64: 12 unit + 6 e2e, native built; physical Windows 10/11 desktop not separately exercised |
| Linux x86_64 install/build/basic use | **verified in CI** | `ubuntu-latest`: 12 unit + 6 e2e, native built |
| Second physical device over LAN | **unverified** | only one physical host available; no multi-machine latency claim |
| NVIDIA CUDA | **unverified** | code path and honest detection present, no hardware |
| AMD ROCm | **unverified** | code path and honest detection present, no hardware |
| Intel XPU | **unverified** | code path and honest detection present, no hardware |
| Windows arm64 | **unverified** | no runner/hardware |
| TLS + signed requests | **verified** | TLS e2e, auth e2e, replay/timestamp unit tests |
| Shard-only dataset delivery | **verified** | dedicated e2e test |
| Dashboard | **verified via HTTP smoke test** | served 200; browser interaction not automated |
| LoRA | **verified** | pipeline adapters train, base weights bitwise unchanged, LoRA inference tested; QLoRA not implemented |
| Pipeline-parallel full fine-tune | **verified across processes** | exact monolithic AdamW parity, 2 stages, stage-only memory |
| HF `safetensors` loader / tokenizer.json | **verified** | SmolLM2-135M and Tiny Llama fixtures; no remote code |
| HF layer-sharded inference | **verified** | SmolLM2 text generated through 2 stages; tiny model exact parity test |
| Island-aware placement | **verified across processes** | label pinning; contiguous island blocks with `island_hops <= 1`; inference honours the same constraint. WAN routing/latency probing not implemented |
| Hybrid CPU/GPU stage placement | **verified across processes** | SmolLM2 stage 0 on MPS, stage 1 on CPU; devices reported in metrics and inference views. One process cannot host two stages |
| RL / preference optimization | **not implemented** | no reward model, rollouts or policy-gradient trainer |
| Tensor parallelism | **not implemented** | pipeline parallelism only |
| SFT | **basic data path only** | JSONL instruction records are formatted into text and trained with next-token loss; no chat template or prompt-token masking |
| Pipeline scheduling | **verified** | contiguous stage planning by memory/compute, stage-count reduction after loss, microbatch forward/backward, atomic commit |
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
python -m pip install -e ".[dev,hf]"
python setup.py build_ext --inplace
pytest -q
python benchmarks/benchmark.py --workdir /tmp/xdc-bench \
  --output benchmarks/latest.json --rounds 3 --local-steps 10 --workers 2
```

Hardware-specific claims can only be upgraded by running the relevant test or
`xdc doctor` on that hardware and recording the output here.
