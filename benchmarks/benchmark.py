#!/usr/bin/env python3
"""Reproducible single-process vs multi-worker benchmark for XDC.

This benchmark starts a real coordinator and two real worker daemons on the local
machine, runs DiLoCo training, then compares:

  * distributed training wall-clock / token throughput against an equivalent
    single-process training workload (same total tokens),
  * two-stage pipeline inference against monolithic local inference on CPU (and
    Apple MPS when available),
  * C++ fused aggregation against the Python fallback.

It intentionally reports that same-host workers share memory bandwidth/CPU; this
is a functional and overhead benchmark, not a claim of linear multi-node speedup.
"""

from __future__ import annotations

import argparse
import base64
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tests.conftest import Cluster
from xdc import __version__, native
from xdc.model.checkpoint import load_state
from xdc.model.gpt import GPT, ModelConfig
from xdc.model.train import train_local


def human(x):
    return f"{x:,.2f}"


def bench_native_aggregation(params: int = 875_264, updates: int = 2, iters: int = 20):
    from xdc.model import train as trainmod

    cfg = ModelConfig(vocab_size=256, n_layer=4, n_head=4, n_embd=128, block_size=128)
    torch.manual_seed(0)
    model = GPT(cfg)
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ups = [{k: torch.randn_like(v) * 1e-3 for k, v in base.items()} for _ in range(updates)]
    kwargs = dict(outer_lr=1.0, momentum=0.9, nesterov=True)
    trainmod.aggregate_deltas(base, ups, [1.0] * updates, **kwargs)
    if not native.AVAILABLE:
        return {"available": False, "params": sum(v.numel() for v in base.values())}
    t0 = time.perf_counter()
    for _ in range(iters):
        trainmod.aggregate_deltas(base, ups, [1.0] * updates, **kwargs)
    native_s = (time.perf_counter() - t0) / iters
    was = trainmod.native.AVAILABLE
    trainmod.native.AVAILABLE = False
    trainmod.aggregate_deltas(base, ups, [1.0] * updates, **kwargs)
    t0 = time.perf_counter()
    for _ in range(iters):
        trainmod.aggregate_deltas(base, ups, [1.0] * updates, **kwargs)
    py_s = (time.perf_counter() - t0) / iters
    trainmod.native.AVAILABLE = was
    return {
        "available": True,
        "params": sum(v.numel() for v in base.values()),
        "updates": updates,
        "iterations": iters,
        "native_ms": native_s * 1000,
        "python_fallback_ms": py_s * 1000,
        "speedup": py_s / native_s,
        "version": native.VERSION,
        "method": "real state_dict aggregation",
    }


def local_training_baseline(init_path: Path, cfg_dict, data: bytes, spec: dict, workers: int, threads: int):
    torch.set_num_threads(threads)
    cfg = ModelConfig.from_dict(cfg_dict)
    torch.manual_seed(spec["seed"])
    model = GPT(cfg)
    model.load_state_dict(load_state(init_path))
    base = {k: v.clone() for k, v in model.state_dict().items()}
    total_steps = spec["rounds"] * workers * spec["local_steps"]
    t0 = time.perf_counter()
    res = train_local(
        model,
        base,
        data,
        local_steps=total_steps,
        batch_size=spec["batch_size"],
        seq_len=spec["seq_len"],
        lr=spec["lr"],
        device="cpu",
        seed=spec["seed"],
        weight_decay=spec["weight_decay"],
        grad_clip=spec["grad_clip"],
    )
    elapsed = time.perf_counter() - t0
    return {
        "device": "cpu",
        "threads": threads,
        "steps": total_steps,
        "tokens": res.tokens,
        "elapsed_s": elapsed,
        "tokens_per_s": res.tokens / elapsed,
        "final_local_loss": res.loss,
    }


def local_inference_baseline(ckpt: Path, cfg_dict, prompt: str, n_tokens: int, device: str, threads: int = 4):
    if device == "cpu":
        torch.set_num_threads(threads)
    cfg = ModelConfig.from_dict(cfg_dict)
    model = GPT(cfg)
    model.load_state_dict(load_state(ckpt))
    model = model.to(device).eval()
    ids = torch.tensor([list(prompt.encode())], dtype=torch.long, device=device)
    with torch.no_grad():
        model.generate(ids, 2, temperature=0)
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        out = model.generate(ids, n_tokens, temperature=0)
        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0
    tokens = out[0, ids.shape[1] :].tolist()
    return {
        "device": device,
        "tokens": n_tokens,
        "elapsed_s": elapsed,
        "tokens_per_s": n_tokens / elapsed,
        "bytes_b64": base64.b64encode(bytes(tokens)).decode(),
    }


def run(args):
    corpus = (ROOT / "datasets" / "tiny_stories.txt").read_bytes()[: args.corpus_bytes]
    root = Path(args.workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "xdc_version": __version__,
        "native": native.VERSION,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "workers": args.workers,
        "caveat": "Workers share one physical host; this measures correctness and coordination overhead, not multi-machine speedup.",
    }
    cluster = Cluster(root / "cluster", tls=False, workers=args.workers)
    try:
        cluster.start_coordinator()
        cluster.start_workers()
        job = cluster.create_job(
            data=corpus,
            name="benchmark",
            layers=args.layers,
            dim=args.dim,
            block_size=args.seq_len,
            training={
                "rounds": args.rounds,
                "local_steps": args.local_steps,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "min_workers": args.workers,
                "max_workers": args.workers,
                "quorum_fraction": 1.0,
                "grace_s": 0.2,
                "round_timeout_s": 300,
                "eval_every": 1,
                "eval_batches": 4,
                "seed": 1234,
            },
        )
        t0 = time.perf_counter()
        done = cluster.wait_job(job["job_id"], timeout=600)
        wall = time.perf_counter() - t0
        if done["state"] != "completed":
            raise RuntimeError(f"benchmark training failed: {done.get('error')}\n{cluster.dump_logs()}")
        spec = done["training"]
        tokens = done["tokens_trained"]
        report["distributed_training"] = {
            "job_id": done["job_id"],
            "rounds": done["version"],
            "workers": args.workers,
            "tokens": tokens,
            "wall_s": wall,
            "tokens_per_s": tokens / wall,
            "initial_loss": done["initial_loss"],
            "final_loss": done["final_loss"],
            "per_round": done["metrics"],
            "algorithm": spec["algorithm"],
            "compression": spec["compression"],
        }

        data_path = root / "cluster" / "coordinator-data" / "jobs" / done["job_id"] / "train.bin"
        init_path = (
            root / "cluster" / "coordinator-data" / "jobs" / done["job_id"] / "init.safetensors"
        )
        round_durations = []
        opened = None
        for event in done["events"]:
            if event.get("message") == "round opened":
                opened = event["ts"]
            elif event.get("message") == "round committed" and opened is not None:
                round_durations.append(event["ts"] - opened)
                opened = None
        round_wall = sum(round_durations)
        report["distributed_training"]["round_wall_s"] = round_wall
        report["distributed_training"]["round_tokens_per_s"] = (
            tokens / round_wall if round_wall > 0 else None
        )
        report["distributed_training"]["round_durations_s"] = round_durations
        # Isolate the algorithmic compute path: slowest worker per round plus
        # aggregation.  This excludes evaluation, checkpoint fsync and download
        # time, which are reported separately by the wall-clock figures above.
        metrics = done.get("metrics") or []
        local_critical = sum(float(m.get("local_elapsed_max_s") or 0.0) for m in metrics)
        aggregate_s = sum(float(m.get("aggregate_s") or 0.0) for m in metrics)
        compute_s = local_critical + aggregate_s
        report["distributed_training"]["local_critical_s"] = local_critical
        report["distributed_training"]["aggregation_s"] = aggregate_s
        report["distributed_training"]["compute_wall_s"] = compute_s
        report["distributed_training"]["compute_tokens_per_s"] = (
            tokens / compute_s if compute_s > 0 else None
        )
        report["single_process_training"] = local_training_baseline(
            init_path, done["model"], data_path.read_bytes(), spec, args.workers, args.workers * 2
        )
        report["training_speedup"] = (
            report["distributed_training"]["round_tokens_per_s"]
            / report["single_process_training"]["tokens_per_s"]
        )
        report["compute_training_speedup"] = (
            report["distributed_training"]["compute_tokens_per_s"]
            / report["single_process_training"]["tokens_per_s"]
            if report["distributed_training"]["compute_tokens_per_s"]
            else None
        )

        ckpt = root / "cluster" / "coordinator-data" / done["latest_checkpoint"]
        prompt = "Mira found a brass key near"
        n = args.infer_tokens
        # Warm stage cache, then measure a real second request.
        cluster.infer(done["job_id"], prompt, n)  # warm stage cache
        t0 = time.perf_counter()
        dist = cluster.infer(done["job_id"], prompt, n)
        wall_infer = time.perf_counter() - t0
        report["distributed_inference"] = {
            "tokens": dist["tokens"],
            "first_token_ms": dist["first_token_ms"],
            "total_ms": dist["total_ms"],
            "client_wall_s": wall_infer,
            "tokens_per_s": dist["tokens"] / (dist["total_ms"] / 1000 if dist["total_ms"] else wall_infer),
            "distributed": dist["distributed"],
            "stages": dist["stages"],
            "bytes_b64": dist["bytes_b64"],
        }
        report["local_inference_cpu"] = local_inference_baseline(
            ckpt, done["model"], prompt, n, "cpu", args.workers * 2
        )
        if torch.backends.mps.is_available():
            report["local_inference_mps"] = local_inference_baseline(ckpt, done["model"], prompt, n, "mps")
        assert base64.b64decode(dist["bytes_b64"]) == base64.b64decode(
            report["local_inference_cpu"]["bytes_b64"]
        ), "distributed pipeline output differs from monolithic"
        report["inference_speedup_vs_cpu"] = (
            report["distributed_inference"]["tokens_per_s"]
            / report["local_inference_cpu"]["tokens_per_s"]
        )
        report["native_aggregation"] = bench_native_aggregation()

        out = Path(args.output)
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print(f"\nwrote {out}")
        return report
    finally:
        cluster.cleanup()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", default=str(ROOT / ".benchmark-work"))
    p.add_argument("--output", default=str(ROOT / "benchmarks" / "latest.json"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--local-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--corpus-bytes", type=int, default=160_000)
    p.add_argument("--infer-tokens", type=int, default=24)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
