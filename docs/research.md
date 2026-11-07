# XDC Technical Research Report

**Date:** 2026-11-08  
**Scope:** consumer-grade heterogeneous distributed training and inference  
**Conclusion up front:** no existing system simultaneously satisfies open consumer Windows/macOS support, low-communication heterogeneous training, true layer-sharded inference, NAT-friendly connectivity, and a manageable security surface. XDC therefore implements a deliberately narrow, testable stack: **PyTorch execution + DiLoCo/Decoupled-DiLoCo coordination + coordinator-mediated outbound tunnels + contiguous layer-pipeline inference + an optional C++ data-plane kernel**.

This document records what was studied, what was rejected, and why. It is not a claim that XDC reproduces the scale or performance of the cited systems.

---

## 1. DeepSeek infrastructure

### 1.1 DeepEP

- Purpose: expert-parallel all-to-all dispatch/combine for MoE, with FP8 and low-latency modes.
- Requirements from the upstream README: Linux, Python 3.10+, **Hopper (SM90) or newer** NVIDIA GPUs, CUDA 13.1+, C++20, PyTorch 2.10+, NCCL 2.32.3+; NVLink for intra-node and RDMA for inter-node.
- License: MIT.
- Verdict: **not reusable for consumer devices**. A MacBook has no CUDA/NVLink; even an RTX 4090 consumer card is not the target hardware. The design lesson is that compute/communication overlap and topology awareness matter, but XDC cannot import these kernels.

### 1.2 DeepGEMM

- Purpose: FP8/FP4/BF16 GEMMs, grouped GEMM, fused MoE, MQA scoring.
- Requirements: NVIDIA **SM90 or SM100**, CUDA 12.9+, CUTLASS 4.0+, Python 3.8+.
- License: MIT.
- Verdict: rejected as a dependency. XDC's tiny GPT uses standard PyTorch linear layers and Metal/Accelerate/oneDNN kernels. DeepGEMM is a reference for future high-end GPU backends, not a portable path.

### 1.3 DualPipe

- Purpose: bidirectional pipeline schedule with full compute/communication overlap and smaller bubbles.
- Key formulas from upstream: 1F1B bubble `(PP−1)(F+B)`, ZB1P `(PP−1)(F+B−2W)`, DualPipe `(PP/2−1)(F&B+B−3W)`; memory cost roughly 2× parameters per device and PP+1 activations.
- Requirements: PyTorch 2.0+, a custom `overlapped_forward_backward` for each module.
- License: MIT.
- Verdict: **algorithmic inspiration only**. It assumes fast homogeneous interconnects and training-scale stages; XDC inference is request-driven token-by-token with variable participation. Implementing DualPipe without overlapping collectives would not help on a WAN. XDC instead uses a simple, fault-tolerant contiguous pipeline.

### 1.4 DeepInfer

- Purpose: production inference framework for DeepSeek models; Lineage from vLLM, ideas from SGLang/TensorRT-LLM.
- Requirements: NVIDIA **SM100** or Huawei Ascend 950, Linux, CUDA 13/CANN 9.2, Python 3.12. Open release focuses on offline inference; KV-cache service and disaggregation are incomplete.
- License: Apache-2.0.
- Verdict: not usable on consumer hardware. Useful design references: batch-invariant deterministic sampling, bounded replay, asynchronous scheduling.

### 1.5 TileKernels

- Purpose: TileLang kernels for MoE routing, quantization, RoPE, Engram, etc.
- Requirements: SM90/SM100 or Ascend 950, CUDA 13.1+/CANN 9.2, PyTorch 2.13+.
- License: MIT.
- Verdict: rejected for the same hardware reasons. Its multi-backend idea (same API, different hardware) is reflected in XDC's backend abstraction, though XDC uses PyTorch as the portability layer.

---

## 2. Low-communication distributed training

### 2.1 DiLoCo (Google DeepMind)

Paper: *DiLoCo: Distributed Low-Communication Training of Language Models*, arXiv:2311.08105.

Key claims and design:

- Workers are "islands" that run many local inner steps.
- Inner optimizer: AdamW.
- Outer optimizer: Nesterov momentum, periodically applied to averaged pseudo-gradients.
- Reported parity with fully synchronous training on C4 while communicating **500× less**, using 8 workers.
- Robust to per-worker data distributions and to resources becoming unavailable; can use resources that appear during training.

Why it fits XDC:

- Payload per round is model-sized, not gradient-per-step sized.
- Heterogeneous devices can run a different number of local steps within a round budget.
- Consumer links can be slow; DiLoCo amortizes the link cost over H compute steps.

Limitations that shaped XDC:

- Not fully asynchronous: a round barrier still exists.
- Whole-model transfer is practical only for small/medium models. XDC's demo model is 0.9M parameters; for 100M+ parameter models a shard/fragment protocol is required.
- The paper's scale is clusters of GPUs, not phones and laptops.

### 2.2 Decoupled DiLoCo

Paper: *Decoupled DiLoCo for Resilient Distributed Pre-training*, arXiv:2604.21428 (Douillard et al., 2026).

Key ideas used by XDC:

- Multiple independent learners send parameter fragments/updates to a central synchronizer.
- **Minimum quorum** instead of waiting for every participant.
- **Adaptive grace window** to accept slightly late updates.
- **Dynamic token-weighted merging**.
- Designed for failures and stragglers with zero global downtime in the paper's simulated experiments.

XDC implementation mapping:

- `quorum_fraction` and `min_workers` control how many updates are required.
- `grace_s` accepts additional updates after quorum.
- Round commit weights updates by tokens processed.
- Once every expected worker has either submitted or been declared lost, a round may commit immediately with `>= min_workers` live updates.
- This is a **decoupled quorum barrier**, not a full fragment-level asynchronous pipeline. XDC does not claim the paper's billion-parameter fault tolerance.

### 2.3 OpenDiLoCo

- Apache-2.0; built on PyTorch FSDP + Hivemind DHT; supports 150M and 1B models across machines; inner AdamW, outer SGD momentum; checkpointing; no longer maintained upstream.
- Lessons: DiLoCo is implementable with common PyTorch primitives; Hivemind DHT is a heavy operational dependency; FSDP assumes a relatively homogeneous, tightly coupled GPU cluster.
- Verdict: use the algorithmic structure (local steps, outer optimizer), not the code stack.

### 2.4 Hivemind and Petals

- Hivemind (MIT): decentralized DHT, fault-tolerant forward/backward, parameter averaging, decentralized MoE. Linux primary; macOS partial; Windows 10+ experimental via WSL. Active development but a complex dependency.
- Petals (MIT): layer-sharded model hosting over a public swarm; pipeline parallelism, low single-batch throughput (single-digit tokens/s for Llama-2-70B), privacy caveat: data is processed by volunteers.
- Verdict: XDC adopts Petals' "layer slice per participant" and Hivemind's fault-tolerance mindset, but rejects a DHT and native libp2p dependency for v0.1. A coordinator-mediated relay is simpler to secure and test; it is a deliberate centralization of discovery, not of model computation.

### 2.5 PyTorch Distributed / FSDP / DTensor

- DTensor is explicitly **alpha**; supports DeviceMesh, Shard/Replicate/Partial; distributed RNG does not support CPU; mixed DTensor/plain tensors raise errors; uneven sharding is experimental.
- FSDP2 is excellent for homogeneous clusters with collective communication; it is not designed for high-latency, churn-prone consumer links.
- Verdict: use plain PyTorch modules and explicit safetensors payloads for portability. A future tensor-parallel backend can use DTensor internally, but it is not the v0.1 transport.

---

## 3. Hardware and execution backends

### 3.1 Apple MLX / Metal / MPS

- MLX (MIT) is Apple-silicon only, unified memory, lazy computation, active and fast-moving. The upstream README does not describe multi-node distributed training.
- PyTorch MPS is available on Apple Silicon through PyTorch 2.x. XDC verifies MPS execution.
- Measured XDC microbenchmark (Apple M4, 10 cores, 16GB):
  - Monolithic tiny GPT inference, 12 tokens: CPU ≈ 0.034s, MPS ≈ 1.76s.
  - Tiny GPT training, 5 steps: MPS was modestly faster at ~0.9M parameters but slower at 3.3M and 13M parameters in this specific batch/sequence configuration.
- This is why `auto` on MPS hosts chooses CPU below `mps_cpu_threshold_params` (default 50M) and MPS above it. This is a measured, workload-aware policy, not an assertion that Metal is faster for all LLMs. Explicit `--backend mps` overrides it.

### 3.2 NVIDIA CUDA / AMD ROCm / Intel XPU

- PyTorch exposes CUDA (`torch.cuda`), ROCm through HIP (`torch.version.hip`), and Intel XPU (`torch.xpu`) where the installed build supports it.
- XDC detects each and reports unavailable backends with a reason. It never advertises a backend it cannot execute.
- No NVIDIA, AMD, or Intel GPU was present on the test host. Those paths are **unverified** (see acceptance report).
- Why not NCCL/RCCL: they require a tightly coupled cluster and native libraries; DiLoCo's model-level round payloads are coordinator-mediated and work across NAT.

### 3.3 NPU

- There is no vendor-neutral PyTorch NPU backend on macOS/Windows. XDC reports NPU as unavailable with the reason, rather than pretending.

---

## 4. Networking and scheduling

### 4.1 Ray

- Apache-2.0, strong general-purpose distributed runtime and scheduling, broad platform support.
- It assumes trusted cluster nodes and a fairly homogeneous runtime; volunteer churn, untrusted code, and consumer NAT are not its primary model.
- Verdict: too heavy for the client; XDC implements only the scheduling it needs (heartbeats, reputation, compute-score weighting, quorum rounds).

### 4.2 libp2p

- MIT, battle-tested P2P stack, NAT traversal and circuit relay concepts.
- A Go/rust libp2p node is a large runtime dependency for a Python worker; writing a secure DHT/relay in v0.1 would increase attack surface without improving the core training/inference demonstration.
- Verdict: v0.1 uses **outbound WebSocket tunnels to a coordinator**. This is relay-based NAT traversal: workers never need an inbound port, which is the dominant consumer constraint. A libp2p transport is a future option for federation.

### 4.3 Coordinator-mediated design

- Discovery: signed registration with a join token.
- Liveness: signed heartbeat + heartbeat timeout.
- Training transport: worker-initiated HTTP downloads/uploads.
- Inference control: multiplexed outbound WebSocket RPC.
- Failure: node reputation, lost-round handling, quorum commit, inference retry excluding failed nodes.

This is not "decentralized" in the DHT sense. It is a hybrid: centralized discovery/coordination, decentralized computation, and no centralization of user data beyond the job's split (each worker receives only its shard).

---

## 5. Security research

Threats considered:

1. Impersonation of a worker.
2. Replay/tampering of heartbeats, assignments, model downloads, and updates.
3. Malicious optimizer updates.
4. Unauthorized job submission or data theft.
5. Credential leakage.
6. A malicious coordinator (out of scope for a self-hosted deployment).

Mechanisms selected:

- Per-node Ed25519 identity; private key never leaves the device.
- Signature over `method`, path+query, timestamp, nonce, and SHA-256(body) for every worker API call.
- Timestamp skew limit + per-node nonce replay cache.
- Join token for enrolment; admin bearer token for management.
- TLS server authentication with a private CA generated by `xdc init`; worker verifies the CA.
- Hashed tunnel tokens.
- Safetensors only (no pickle), shape/name/finite checks, and size limits.
- L2-norm outlier filtering (needs ≥3 updates for a median-based guard); absolute norm ceiling; token weights.
- Worker reputation excludes repeatedly failing nodes from scheduling.
- Isolated execution: workers only run the fixed GPT training/pipeline code; assignments cannot execute arbitrary code.

Limitations, stated explicitly:

- With only two workers, There is no Byzantine consensus. A malicious worker can still bias the average; the guard only removes gross outliers. Production open-network training would need redundancy/verification (e.g., replicated computation, proof-of-learning, or trusted islands).
- The coordinator is trusted. A compromised coordinator can read datasets and serve malicious model weights.
- Dataset confidentiality depends on join-token trust and TLS; public volunteer swarms are not suitable for sensitive data.
- Traffic analysis and denial of service are not fully mitigated.
- TLS uses server authentication, not client certificates; worker authentication is at the signed-request layer.

---

## 6. C++ acceleration decision

A pure-Python coordinator adds overhead on:
- per-round aggregation of many small tensors,
- token sampling in the streaming loop,
- byte-level batching.

XDC includes an optional pybind11 C++17 extension with:
- fused per-tensor robust aggregation + outer momentum,
- deterministic top-k/temperature sampling,
- byte batch sampling,
- block-wise int8 quantizer (available for future wire compression).

The extension is strictly optional. If the host has no compiler, installation falls back to Python and `xdc doctor` reports `python-fallback`. This is the only honest way to support locked-down Windows machines while offering native performance where available.

Native aggregation was measured against the Python implementation on the actual state-dict shape. Because the test machine was under load, absolute numbers vary; relative results are recorded in `docs/benchmark.md`. The C++ path removes Python-level per-tensor loops; it is not a GPU kernel and cannot make a 2MB payload network-free.

---

## 7. Final selection

| Layer | Choice | Reason |
|---|---|---|
| Model execution | PyTorch 2.x | One API covers CPU, CUDA, ROCm, MPS, XPU; portable Windows/macOS wheels |
| Training algorithm | DiLoCo with decoupled quorum + grace + token weights | Low communication, heterogeneous straggler tolerance, real convergence |
| Outer optimizer | SGD momentum / optional Nesterov | Matches DiLoCo; stable for small models |
| Payloads | safetensors, FP16 optional | No pickle, mmap-friendly, deterministic |
| Inference | contiguous layer pipeline | True model sharding; bounded KV cache per stage; matches monolithic output |
| Control plane | FastAPI + uvicorn | Mature, testable, Windows/macOS |
| NAT | outbound WebSocket tunnel + HTTP | No inbound worker port; simpler than a DHT |
| Coordination | single coordinator + durable JSON state | Auditable, easy self-hosting |
| Identity | Ed25519 | Cheap signatures, no shared worker secret |
| TLS | private CA + server cert | Works on LAN without public DNS |
| Native hot path | optional C++17 extension | High performance without mandatory toolchain |

## 8. Explicit non-goals for v0.1

- Full tensor parallelism.
- LoRA/QLoRA and reinforcement-learning trainers.
- Byzantine fault tolerance without trusted redundancy.
- Public anonymous open swarm / crypto incentives.
- Training 7B+ models across consumer WAN links.
- AutoNAT/hole punching beyond coordinator relay.
- NPU execution.

These are not silently implied to work.

## 9. References

- Douillard et al., *DiLoCo: Distributed Low-Communication Training of Language Models*, arXiv:2311.08105. https://arxiv.org/abs/2311.08105
- Douillard et al., *Decoupled DiLoCo for Resilient Distributed Pre-training*, arXiv:2604.21428. https://arxiv.org/abs/2604.21428
- *Streaming DiLoCo with overlapping communication*, arXiv:2501.18512. https://arxiv.org/abs/2501.18512
- Jaghouar, Ong, Hagemann, *OpenDiLoCo*, arXiv:2407.07852; repository https://github.com/PrimeIntellect-ai/OpenDiLoCo (Apache-2.0)
- Hivemind, https://github.com/learning-at-home/hivemind (MIT)
- Petals, https://github.com/bigscience-workshop/petals (MIT)
- DeepEP, https://github.com/deepseek-ai/DeepEP (MIT)
- DeepGEMM, https://github.com/deepseek-ai/DeepGEMM (MIT)
- DualPipe, https://github.com/deepseek-ai/DualPipe (MIT)
- DeepInfer, https://github.com/deepseek-ai/DeepInfer (Apache-2.0)
- TileKernels, https://github.com/deepseek-ai/TileKernels (MIT)
- PyTorch DTensor documentation, https://docs.pytorch.org/docs/stable/distributed.tensor.html
- PyTorch FSDP2 documentation, https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- Apple MLX, https://github.com/ml-explore/mlx (MIT)
- Ray, https://github.com/ray-project/ray (Apache-2.0)
- libp2p, https://github.com/libp2p/go-libp2p (MIT)

All URLs accessed 2026-11-08. Requirements and claims are based on upstream READMEs/papers at that date; the ecosystem changes quickly.
