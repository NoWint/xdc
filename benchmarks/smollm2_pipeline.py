#!/usr/bin/env python
"""Reproduce the SmolLM2-135M LoRA pipeline run in benchmarks/.

This starts a real coordinator and two worker processes on one host.  It is an
acceptance benchmark, not a mock: stage files are fetched over HTTP, activations
and gradients cross the WebSocket tunnel, and the coordinator assembles the
global checkpoint from stage uploads.

    python benchmarks/smollm2_pipeline.py --model-dir /path/to/SmolLM2-135M-Instruct
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.conftest import Cluster  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--corpus", default=str(ROOT / "datasets" / "tiny_stories.txt"))
    parser.add_argument("--name", default="smollm2-135m")
    parser.add_argument("--workdir", default="/tmp/xdc-smollm-bench")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--micro-batches", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument(
        "--backends",
        default="cpu,cpu",
        help="comma-separated backend for worker-1,worker-2 (cpu,mps,cuda,rocm,xpu)",
    )
    args = parser.parse_args()
    backends = dict(zip(("worker-1", "worker-2"), args.backends.split(",")))

    source = Path(args.model_dir)
    work = Path(args.workdir)
    if work.exists():
        shutil.rmtree(work)
    c = Cluster(work, workers=2, backends=backends)
    conf = yaml.safe_load(c.coord_config.read_text())
    conf["coordinator"].update({"max_upload_mb": 2048, "pipeline_memory_fraction": 0.75})
    c.coord_config.write_text(yaml.safe_dump(conf, sort_keys=False))
    models = c.root / "coordinator-data" / "models" / args.name
    models.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, models / path.name)

    try:
        c.start_coordinator()
        c.start_workers()
        dataset = c.upload_dataset(Path(args.corpus).read_bytes(), "smollm-bench")
        body = {
            "name": "smollm2-pipeline-lora",
            "dataset_id": dataset["dataset_id"],
            "model": {"kind": "hf", "model_name": args.name},
            "training": {
                "parallelism": "pipeline",
                "algorithm": "pipeline",
                "rounds": args.rounds,
                "batch_size": 1,
                "seq_len": args.seq_len,
                "micro_batches": args.micro_batches,
                "lr": 2e-4,
                "min_workers": 2,
                "max_workers": 2,
                "pipeline_stages": 2,
                "lora_rank": args.lora_rank,
                "lora_alpha": float(args.lora_rank * 2),
                "activation_dtype": "fp16",
                "eval_every": 1,
                "eval_batches": 2,
                "round_timeout_s": 600,
                "seed": 7,
            },
            "auto_start": True,
        }
        response = c.client.post(
            "/v1/jobs",
            content=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        started = time.time()
        job = c.wait_job(response.json()["job_id"], timeout=1800)
        print(
            f"state={job['state']} version={job['version']} "
            f"loss={job['initial_loss']}->{job['final_loss']} "
            f"tokens={job['tokens_trained']} wall={time.time() - started:.1f}s"
        )
        for metric in job["metrics"]:
            print(f"round {metric['round']}: train={metric['train_loss']} val={metric['val_loss']}")
            for stage in metric["stage_metrics"]:
                print(
                    f"  stage {stage['stage_index']} {stage['device']}: "
                    f"params={stage['param_bytes']} opt={stage['optimizer_bytes']} "
                    f"trainable={stage['trainable_parameters']}"
                )
        infer = c.infer(job["job_id"], "Once upon a time", max_tokens=8, stream=False)
        print(
            f"infer: tokens={infer['tokens']} ttft={infer['first_token_ms']:.0f}ms "
            f"total={infer['total_ms']:.0f}ms text={infer['text']!r}"
        )
        print(json.dumps({"stage_plan": job["stage_plan"], "inference": infer}, indent=2))
        return 0 if job["state"] == "completed" else 1
    finally:
        c.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
