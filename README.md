# XDC — consumer-grade distributed AI compute

[![xdc-ci](https://github.com/NoWint/xdc/actions/workflows/ci.yml/badge.svg)](https://github.com/NoWint/xdc/actions/workflows/ci.yml)

**XDC (Xiatian Distributed Compute)** lets ordinary Macs and Windows PCs pool idle CPU/GPU memory and compute to train and run a model **together**. It is not a prompt-forwarding wrapper around one machine's model server.

- **Training:** workers run local AdamW steps on disjoint data shards; the coordinator performs DiLoCo-style periodic pseudo-gradient aggregation with a decoupled quorum, grace window, token weighting and failure-aware commits.
- **Inference:** a model is split into contiguous layer stages across workers; hidden activations travel through an outbound WebSocket tunnel; KV caches stay on the stage owner; tokens stream back via SSE. Distributed output is tested byte-for-byte against monolithic greedy generation.
- **Heterogeneous:** PyTorch CUDA / ROCm / Intel XPU / Apple MPS / CPU detection with honest fallback and a workload-aware auto policy. Unsupported NPU is reported, not faked.
- **Network:** coordinator + worker relay. Workers make outbound connections, so NAT-port-forwarding is not required on worker devices.
- **Security:** Ed25519 node identity, signed HTTP requests with replay protection, TLS with a generated private CA, join/admin tokens, safetensors-only payloads, update validation, reputation and outlier filtering.

```
┌──────────────── Coordinator ────────────────┐
│ scheduler · DiLoCo engine · pipeline engine │
│ TLS · Ed25519 request auth · dashboard      │
└──────┬───────────────────────────┬──────────┘
       │ signed HTTP               │ outbound WS tunnel
 ┌─────▼─────┐                ┌────▼──────┐
 │ Worker A  │                │ Worker B  │
 │ local AdamW│               │ stage 1/2 │
 │ shard 0   │                │ KV cache  │
 └───────────┘                └───────────┘
```

## Status at a glance

| Capability | State |
|---|---|
| Coordinator / worker CLI / daemon | working |
| Two worker processes discovered and connected | working |
| Real two-process DiLoCo training, loss decrease, parameter change | working |
| Checkpoint save / coordinator restart / resume | working |
| Worker loss detection + quorum commit / single-node reschedule | working |
| Two-stage layer-sharded inference with KV cache + SSE | working, exact-monolithic parity |
| TLS + signed registration/heartbeat/update | working, tested |
| C++ native aggregation/sampling/data batches | working, optional |
| Metrics, dashboard, logs, doctor | working; dashboard HTTP smoke-tested |
| macOS Apple Silicon CPU/MPS | tested locally + CI |
| macOS Intel CPU | verified in CI (`macos-15-intel`) |
| Windows x64 install/build/test | verified in CI (Windows Server build 26100) |
| Linux x86_64 install/build/test | verified in CI (`ubuntu-latest`) |
| Native C++ build on all four CI runners | verified |
| Two **physical** devices | not available in this environment — **unverified** |
| NVIDIA CUDA / AMD ROCm / Intel XPU | code path present, **unverified** hardware |
| Windows 10/11 physical desktop, Windows arm64 | **unverified** |
| LoRA / RL / tensor parallelism | **not implemented in v0.1** |
| Byzantine fault tolerance | not claimed; 2-worker outlier guard is limited |

Full evidence and caveats: [docs/acceptance.md](docs/acceptance.md).

## Quickstart (macOS / Linux)

```bash
git clone https://github.com/NoWint/xdc.git
cd xdc
bash scripts/install.sh

xdc init --dir ./xdc-demo --host 0.0.0.0
xdc coordinator --config ./xdc-demo/coordinator.yaml --access-log
```

In two more terminals:

```bash
xdc worker --config ./xdc-demo/worker1.yaml
xdc worker --config ./xdc-demo/worker2.yaml
```

Train a byte-level tiny GPT on the included original corpus:

```bash
xdc train --config ./xdc-demo/coordinator.yaml \
  --dataset datasets/tiny_stories.txt \
  --rounds 10 --local-steps 20 --min-workers 2 --wait
```

Generate through the two-stage pipeline:

```bash
xdc infer --config ./xdc-demo/coordinator.yaml \
  --job-id <job-id> --prompt "Mira found a brass key near" \
  --max-tokens 32 --temperature 0
```

Monitor at `http://127.0.0.1:8787/dashboard` (paste the admin token) or `xdc status --config ./xdc-demo/coordinator.yaml`.

### Windows

```powershell
.\scripts\install.ps1
.\.venv\Scripts\xdc.exe init --dir .\xdc-demo --host 0.0.0.0
.\.venv\Scripts\xdc.exe coordinator --config .\xdc-demo\coordinator.yaml
```

See [docs/deployment.md](docs/deployment.md) for LAN, Internet exposure (reverse proxy / WireGuard), firewall, autostart and troubleshooting.

## CLI

```
xdc init / coordinator / worker / doctor
xdc status / nodes / jobs / job / checkpoints
xdc train / start / pause / cancel / infer / plan
xdc dataset upload|list
xdc ban / health / monitor
```

`xdc doctor --json` reports CPU, RAM, selected backend, every backend's availability and the native extension status.

## What "distributed" means here

- **Training is genuinely distributed:** each worker computes its own local steps on its own shard; the coordinator never computes the local optimizer steps. The coordinator aggregates updates and advances the global model.
- **Inference is genuinely layer-sharded:** stage 0 owns embeddings and part of the transformer; stage 1 owns the rest plus the head. Only hidden activations cross the wire. This is not "send the prompt to another machine's local model."
- **Same-host caveat:** the acceptance environment has one physical Mac. The end-to-end tests use two independent worker processes with separate identities, sockets and caches. They prove the protocol and fault behavior, not multi-machine speedup. A real two-device test requires a second machine; instructions are in the deployment guide.

## Reproducible verification

```bash
make unit        # model, crypto, native, data
make e2e         # launches real coordinator + workers, trains, infers, kills a worker, restarts
make test        # both; 19 tests on the reference host
python benchmarks/benchmark.py --workdir /tmp/xdc-bench
```

Expected high-level evidence:

- training loss decreases from ~5.55 to ~3.0 on the demo;
- both workers appear in every round's `updates_received: 2`;
- initial and final safetensors hashes and tensors differ;
- distributed greedy bytes equal monolithic greedy bytes;
- streaming SSE bytes equal non-streaming bytes;
- killing a worker leads to an event and completion with the remaining node;
- a worker-signed dataset request returns only its assigned shard;
- TLS is required and a self-signed certificate is rejected without the CA;
- admin endpoints reject anonymous requests while `/healthz` stays public.

See [docs/acceptance.md](docs/acceptance.md) for actual recorded runs.

## Performance

Small models on one host are compute-bound on CPU and latency-bound on the pipeline. XDC reports this honestly. In the latest same-host run (2 CPU workers, 30,720 tokens, raw data in [`benchmarks/latest.json`](benchmarks/latest.json)):

- end-to-end training was **0.814×** a single 4-thread process because evaluation, checkpointing and transfer dominate at this scale, while the **local-compute + aggregation path was 1.067×** (effectively parity);
- two-stage distributed inference produced **184 tok/s** vs **727 tok/s** monolithic CPU and **58 tok/s** MPS for the 0.9M model. The pipeline's value is **memory aggregation across devices**, not making a tiny model faster on one box;
- the optional C++ merge was **2.97×** faster than the Python fallback (9.84 ms vs 29.21 ms for 875k parameters).

Full method and caveats: [docs/benchmark.md](docs/benchmark.md).

## Security

Read [docs/security.md](docs/security.md). In short: signed worker requests, TLS, join/admin tokens, safe tensor parsing, bounded updates, reputation, outlier filtering. Not claimed: two-worker Byzantine tolerance, anonymous public swarms, confidential computing.

Report vulnerabilities privately to the repository owner; do not open a public issue containing exploits or secrets.

## Documentation

- [Technical research](docs/research.md)
- [Architecture](docs/architecture.md)
- [Deployment](docs/deployment.md)
- [Security model](docs/security.md)
- [Development](docs/development.md)
- [Benchmarks](docs/benchmark.md)
- [Acceptance report](docs/acceptance.md)

## License

Apache-2.0. Third-party notices: [NOTICE](NOTICE).
