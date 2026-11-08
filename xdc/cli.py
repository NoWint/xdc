"""`xdc` command line interface."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from . import __version__, native
from .config import CoordinatorConfig, WorkerConfig
from .crypto import generate_tls_material, new_token
from .protocol import InferenceRequest, JobCreate, TrainingSpec

log = logging.getLogger("xdc.cli")


# ---------------------------------------------------------------------- helpers
def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _derive_url(cfg: CoordinatorConfig | None, explicit: str | None) -> str | None:
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("XDC_COORDINATOR_URL")
    if env:
        return env.rstrip("/")
    if cfg:
        host = cfg.host
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        scheme = "https" if cfg.tls.enabled else "http"
        return f"{scheme}://{host}:{cfg.port}"
    return None


def _admin_context(args) -> tuple[str, str, str | None, bool]:
    cfg = None
    if getattr(args, "config", None):
        cfg = CoordinatorConfig.load(args.config)
    url = _derive_url(cfg, getattr(args, "url", None))
    token = getattr(args, "token", None) or os.environ.get("XDC_ADMIN_TOKEN") or (cfg.admin_token if cfg else None)
    ca = getattr(args, "ca", None) or (cfg.tls.ca_file if cfg and cfg.tls.enabled else None)
    insecure = bool(getattr(args, "insecure", False))
    if not url:
        raise SystemExit("coordinator URL required: use --url, XDC_COORDINATOR_URL, or --config")
    if not token:
        raise SystemExit("admin token required: use --token, XDC_ADMIN_TOKEN, or --config")
    return url, token, ca, insecure


def admin_client(args) -> tuple[httpx.Client, str]:
    url, token, ca, insecure = _admin_context(args)
    verify: bool | str = True
    if url.startswith("https://"):
        if insecure:
            verify = False
        elif ca:
            verify = ca
    return (
        httpx.Client(
            base_url=url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(600.0, connect=15.0),
            verify=verify,
            trust_env=False,
        ),
        url,
    )


def _check(resp: httpx.Response) -> dict:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise SystemExit(f"HTTP {resp.status_code}: {detail}")
    if resp.status_code == 204 or not resp.content:
        return {}
    return resp.json()


def _print_job(job: dict) -> None:
    print(f"job       {job['job_id']} ({job['name']})")
    print(f"state     {job['state']}")
    print(f"progress  round {job['next_round']}/{job['training']['rounds']}  version {job['version']}")
    kind = job.get("model_kind", "tinygpt")
    layers = job["model"].get("n_layer", job["model"].get("num_hidden_layers"))
    print(f"model     {kind} {layers} layers, {job['parameters']} params")
    if job.get("stage_plan"):
        ranges = [f"{s['layer_start']}-{s['layer_end']}@{s['node_name']}" for s in job["stage_plan"]]
        print(f"stages    {len(ranges)}: {' '.join(ranges)}")
    print(f"loss      initial={job.get('initial_loss')} current={job.get('final_loss')} best={job.get('best_loss')}")
    print(f"tokens    {job['tokens_trained']:,}")
    print(f"workers   {', '.join(job['current_round']['expected']) if job.get('current_round') else '-'}")
    print(f"checkpoint {job.get('latest_checkpoint')}")
    if job.get("error"):
        print(f"error     {job['error']}")


def wait_job(client: httpx.Client, job_id: str, interval: float = 2.0) -> dict:
    last = None
    while True:
        job = _check(client.get(f"/v1/jobs/{job_id}"))
        state = (
            job["state"],
            job["next_round"],
            job["version"],
            job.get("final_loss"),
        )
        if state != last:
            rnd = job.get("current_round")
            extra = f" round={rnd['index']}" if rnd else ""
            print(
                f"[{time.strftime('%H:%M:%S')}] state={job['state']} "
                f"round={job['next_round']}/{job['training']['rounds']} "
                f"version={job['version']} val_loss={job.get('final_loss')}{extra}",
                flush=True,
            )
            last = state
        if job["state"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(interval)


# ---------------------------------------------------------------------- init
def cmd_init(args) -> int:
    root = Path(args.dir).resolve()
    if (root / "coordinator.yaml").exists() and not args.force:
        raise SystemExit(f"{root} already initialized (use --force to overwrite configs)")
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    tls_dir = root / "tls"
    if args.tls:
        paths = generate_tls_material(tls_dir, hosts=[args.host])
        cert, key, ca = paths["cert"], paths["key"], paths["ca"]
    else:
        cert = key = ca = None
    admin_token = new_token(32)
    join_token = new_token(32)
    coord = {
        "coordinator": {
            "host": args.host,
            "port": args.port,
            "data_dir": str(data / "coordinator"),
            "admin_token": admin_token,
            "join_token": join_token,
            "heartbeat_timeout_s": 15,
            "max_upload_mb": 512,
            "eval_device": "cpu",
            "tls": {
                "enabled": bool(args.tls),
                "cert_file": cert,
                "key_file": key,
                "ca_file": ca,
            },
        }
    }
    public_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    scheme = "https" if args.tls else "http"
    worker = {
        "worker": {
            "coordinator_url": f"{scheme}://{public_host}:{args.port}",
            "join_token": join_token,
            "data_dir": str(data / "worker1"),
            "name": "worker-1",
            "backend": "auto",
            "threads": 0,
            "max_memory_mb": 0,
            "tls_ca": ca,
            "allow_insecure": not args.tls,
            "heartbeat_interval_s": 5,
            "poll_interval_s": 1,
            "labels": {},
            "mps_cpu_threshold_params": 50000000,
        }
    }
    worker2 = json.loads(json.dumps(worker))
    worker2["worker"]["data_dir"] = str(data / "worker2")
    worker2["worker"]["name"] = "worker-2"
    for name, payload in [
        ("coordinator.yaml", coord),
        ("worker1.yaml", worker),
        ("worker2.yaml", worker2),
    ]:
        path = root / name
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        try:
            path.chmod(0o600)
        except Exception:
            pass
    print(f"Initialized XDC at {root}")
    print(f"  TLS: {'enabled' if args.tls else 'disabled (loopback/dev only)'}")
    print(f"  Admin token: {admin_token}")
    print(f"  Join token:  {join_token}")
    print("\nRun coordinator:")
    print(f"  xdc coordinator --config {root / 'coordinator.yaml'}")
    print("Run workers (two terminals or two machines):")
    print(f"  xdc worker --config {root / 'worker1.yaml'}")
    print(f"  xdc worker --config {root / 'worker2.yaml'}")
    print("\nUpload dataset and train:")
    print(f"  xdc dataset upload --config {root / 'coordinator.yaml'} --name tiny datasets/tiny_stories.txt")
    print(f"  xdc train --config {root / 'coordinator.yaml'} --dataset datasets/tiny_stories.txt --rounds 10 --wait")
    if args.tls:
        print("\nFor another LAN device, copy its worker YAML, set coordinator_url to this machine's LAN IP,")
        print("and keep tls_ca pointing to the generated CA (or copy tls/ca.pem).")
    return 0


# ---------------------------------------------------------------------- run services
def cmd_coordinator(args) -> int:
    import uvicorn

    cfg = CoordinatorConfig.load(args.config)
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    setup_logging(args.log_level, args.log_file)
    from .coordinator.app import create_app

    app = create_app(cfg)
    kwargs: dict[str, Any] = {
        "host": cfg.host,
        "port": cfg.port,
        "log_level": cfg.log_level.lower(),
        "access_log": args.access_log,
        # Consumer laptops and Wi-Fi can pause briefly under load.  A 20s ping
        # timeout turns that into a tunnel churn loop; keep the socket open
        # longer and let the heartbeat/round deadline make liveness decisions.
        "ws_ping_interval": 30.0,
        "ws_ping_timeout": 90.0,
    }
    if cfg.tls.enabled:
        kwargs["ssl_certfile"] = cfg.tls.cert_file
        kwargs["ssl_keyfile"] = cfg.tls.key_file
    log.info("starting coordinator on %s://%s:%s", "https" if cfg.tls.enabled else "http", cfg.host, cfg.port)
    uvicorn.run(app, **kwargs)
    return 0


def cmd_worker(args) -> int:
    import asyncio

    cfg = WorkerConfig.load(args.config)
    if args.name:
        cfg.name = args.name
    if args.backend:
        cfg.backend = args.backend
    if args.coordinator_url:
        cfg.coordinator_url = args.coordinator_url
    setup_logging(args.log_level, args.log_file)
    from .worker.daemon import run_worker

    try:
        asyncio.run(run_worker(cfg))
    except KeyboardInterrupt:
        pass
    return 0


# ---------------------------------------------------------------------- doctor
def cmd_doctor(args) -> int:
    from .worker.hardware import detect, node_backend_summary

    try:
        detected, caps = detect(args.backend, args.max_memory_mb)
    except Exception as exc:
        print(f"hardware detection failed: {exc}", file=sys.stderr)
        return 2
    report = {
        "xdc_version": __version__,
        "native": {"available": native.AVAILABLE, "version": native.VERSION},
        "platform": caps.platform,
        "machine": caps.machine,
        "hostname": caps.hostname,
        "cpu": caps.cpu_model,
        "cpu_count": caps.cpu_count,
        "memory_mb": caps.total_memory_mb,
        "available_memory_mb": caps.available_memory_mb,
        "selected_backend": caps.backend,
        "device_name": caps.device_name,
        "compute_score_gflops": caps.compute_score,
        "backends": caps.backends,
        "notes": caps.notes,
        "torch": caps.torch,
        "python": caps.python,
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"XDC {__version__}  native={native.VERSION}")
    print(f"CPU       {caps.cpu_model} ({caps.cpu_count} threads)")
    print(f"Memory    {caps.total_memory_mb} MB total / {caps.available_memory_mb} MB available")
    print(f"Selected  {node_backend_summary(caps)}")
    print("Backends:")
    for b in caps.backends:
        mark = "OK " if b["available"] else "-- "
        reason = f"  ({b['reason']})" if b.get("reason") else ""
        print(f"  [{mark}] {b['kind']:<5} {b['name']}{reason}")
    for note in caps.notes:
        print(f"note: {note}")
    if getattr(args, "config", None):
        try:
            cfg = WorkerConfig.load(args.config)
            print(f"config   {args.config} -> {cfg.coordinator_url}  backend={cfg.backend}")
        except Exception as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2
    return 0


# ---------------------------------------------------------------------- admin commands
def cmd_status(args) -> int:
    client, url = admin_client(args)
    stats = _check(client.get("/v1/stats"))
    nodes = _check(client.get("/v1/nodes"))
    if args.json:
        print(json.dumps({"url": url, "stats": stats, "nodes": nodes}, indent=2))
        return 0
    print(f"Coordinator {url}  v{stats['version']}  uptime {stats['uptime_s']:.0f}s")
    print(
        f"Nodes {stats['nodes_online']}/{stats['nodes_total']} online  "
        f"CPU threads {stats['pooled_cpu_threads']}  "
        f"RAM {stats['pooled_memory_mb']} MB  GPU {stats['pooled_gpu_memory_mb']} MB"
    )
    print(f"Jobs {stats['jobs_running']} running / {stats['jobs_total']} total  tokens {stats['total_tokens_trained']:,}")
    if nodes:
        print("\nNodes:")
        print(f"{'NAME':24} {'STATE':8} {'BACKEND':7} {'DEVICE':28} {'SCORE':>8} {'REP':>5} {'UPD':>6}")
        for n in nodes:
            state = "banned" if n["banned"] else ("online" if n["online"] else "offline")
            print(
                f"{n['name'][:24]:24} {state:8} {n['capabilities']['backend']:7} "
                f"{n['capabilities']['device_name'][:28]:28} {n['capabilities']['compute_score']:8.1f} "
                f"{n['reputation']:5.2f} {n['accepted_updates']:6d}"
            )
    return 0


def cmd_nodes(args) -> int:
    client, _ = admin_client(args)
    nodes = _check(client.get("/v1/nodes"))
    if args.json:
        print(json.dumps(nodes, indent=2))
    else:
        for n in nodes:
            print(json.dumps(n, ensure_ascii=False))
    return 0


def cmd_jobs(args) -> int:
    client, _ = admin_client(args)
    jobs = _check(client.get("/v1/jobs"))
    if args.json:
        print(json.dumps(jobs, indent=2))
        return 0
    for j in jobs:
        print(
            f"{j['job_id']}  {j['state']:<9} {j['next_round']:>3}/{j['training']['rounds']:<3} "
            f"loss={j.get('final_loss')}  {j['name']}"
        )
    return 0


def cmd_job(args) -> int:
    client, _ = admin_client(args)
    job = _check(client.get(f"/v1/jobs/{args.job_id}"))
    print(json.dumps(job, indent=2))
    return 0


def cmd_start(args) -> int:
    client, _ = admin_client(args)
    job = _check(client.post(f"/v1/jobs/{args.job_id}/start"))
    _print_job(job)
    if args.wait:
        job = wait_job(client, args.job_id)
        _print_job(job)
        return 0 if job["state"] == "completed" else 1
    return 0


def cmd_pause(args) -> int:
    client, _ = admin_client(args)
    job = _check(client.post(f"/v1/jobs/{args.job_id}/pause"))
    _print_job(job)
    return 0


def cmd_cancel(args) -> int:
    client, _ = admin_client(args)
    job = _check(client.post(f"/v1/jobs/{args.job_id}/cancel"))
    _print_job(job)
    return 0


def cmd_checkpoints(args) -> int:
    client, _ = admin_client(args)
    items = _check(client.get(f"/v1/jobs/{args.job_id}/checkpoints"))
    print(json.dumps(items, indent=2))
    return 0


def cmd_ban(args) -> int:
    client, _ = admin_client(args)
    node = _check(client.post(f"/v1/admin/nodes/{args.node_id}/ban", params={"banned": not args.unban}))
    print(json.dumps(node, indent=2))
    return 0


def cmd_dataset(args) -> int:
    client, _ = admin_client(args)
    if args.action == "list":
        print(json.dumps(_check(client.get("/v1/admin/datasets")), indent=2))
        return 0
    path = Path(args.path)
    data = path.read_bytes()
    resp = client.post(
        "/v1/admin/datasets",
        params={"name": args.name or path.name},
        content=data,
        headers={"Content-Type": "application/octet-stream"},
    )
    print(json.dumps(_check(resp), indent=2))
    return 0


MODEL_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
}


def _upload_model_tree(client: httpx.Client, root: Path, name: str) -> dict:
    files = [
        p
        for p in sorted(root.iterdir())
        if p.is_file()
        and (p.name in MODEL_FILES or (p.name.endswith(".safetensors") and p.name != "model.safetensors.index.json"))
    ]
    if not files:
        raise SystemExit(f"no model files found in {root}")
    for path in files:
        print(f"uploading {path.name} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        _check(
            client.put(
                f"/v1/admin/models/{name}/files/{path.name}",
                content=path.read_bytes(),
                headers={"Content-Type": "application/octet-stream"},
                timeout=1800,
            )
        )
    return _check(client.post(f"/v1/admin/models/{name}/finalize"))


def cmd_model(args) -> int:
    client, _ = admin_client(args)
    if args.action == "list":
        print(json.dumps(_check(client.get("/v1/admin/models")), indent=2))
        return 0
    if args.action == "show":
        name = args.name or args.target
        if not name:
            raise SystemExit("model show requires a name")
        print(json.dumps(_check(client.post(f"/v1/admin/models/{name}/finalize")), indent=2))
        return 0
    if args.action == "add":
        path = args.target
        if not path:
            raise SystemExit("model add requires a directory")
        name = args.name or Path(path).resolve().name
        summary = _upload_model_tree(client, Path(path), name)
        print(json.dumps(summary, indent=2))
        return 0
    if args.action == "pull":
        repo = args.repo or args.target
        if not repo:
            raise SystemExit("model pull requires a repo id")
        name = args.name or repo.rstrip("/").split("/")[-1]
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit("model pull requires: pip install 'xdc[hf]'") from exc
        import tempfile

        with tempfile.TemporaryDirectory(prefix="xdc-model-") as tmp:
            snapshot_download(
                repo,
                revision=args.revision,
                local_dir=tmp,
                allow_patterns=[
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "generation_config.json",
                    "*.safetensors",
                    "*.safetensors.index.json",
                ],
                ignore_patterns=["*.bin", "*.pt", "*.pth", "*.ckpt", "*.msgpack"],
            )
            summary = _upload_model_tree(client, Path(tmp), name)
        print(json.dumps(summary, indent=2))
        return 0
    raise SystemExit(f"unknown model action {args.action}")


def cmd_train(args) -> int:
    client, _ = admin_client(args)
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"dataset not found: {dataset_path}")
    data = dataset_path.read_bytes()
    ds = _check(
        client.post(
            "/v1/admin/datasets",
            params={"name": args.dataset_name or dataset_path.name},
            content=data,
            headers={"Content-Type": "application/octet-stream"},
        )
    )
    if args.hf_model:
        model = {"kind": "hf", "model_name": args.hf_model}
    else:
        model = {
            "vocab_size": 256,
            "n_layer": args.layers,
            "n_head": args.heads,
            "n_embd": args.dim,
            "block_size": args.block_size,
            "dropout": 0.0,
            "bias": True,
        }
    algorithm = "pipeline" if args.parallelism == "pipeline" or args.hf_model else args.algorithm
    spec = TrainingSpec(
        algorithm=algorithm,
        parallelism=args.parallelism,
        rounds=args.rounds,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        outer_lr=args.outer_lr,
        outer_momentum=args.outer_momentum,
        nesterov=False,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        quorum_fraction=args.quorum,
        grace_s=args.grace,
        round_timeout_s=args.timeout,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        seed=args.seed,
        compression=args.compression,
        mode=args.mode,
        micro_batches=args.micro_batches,
        pipeline_stages=args.pipeline_stages,
        activation_dtype=args.activation_dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        island=args.island,
        stage_memory_headroom=args.stage_headroom,
    )
    body = JobCreate(
        name=args.name,
        dataset_id=ds["dataset_id"],
        model=model,
        training=spec,
        auto_start=True,
        init_checkpoint_job=args.resume_from,
    )
    job = _check(client.post("/v1/jobs", content=body.model_dump_json(), headers={"Content-Type": "application/json"}))
    if args.json:
        print(json.dumps(job, indent=2))
    else:
        _print_job(job)
        print(f"dataset   {ds['dataset_id']} ({ds['size_bytes']} bytes)")
    if args.wait or not args.json:
        job = wait_job(client, job["job_id"])
        _print_job(job)
        return 0 if job["state"] == "completed" else 1
    return 0


def cmd_infer(args) -> int:
    client, _ = admin_client(args)
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    req = InferenceRequest(
        job_id=args.job_id,
        checkpoint=args.checkpoint,
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        stream=not args.no_stream,
        allow_retry=True,
    )
    if args.no_stream:
        data = _check(client.post("/v1/infer", content=req.model_dump_json(), headers={"Content-Type": "application/json"}))
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(data["text"], end="")
            print(
                f"\n[tokens={data['tokens']} ttft={data['first_token_ms']:.0f}ms "
                f"total={data['total_ms']:.0f}ms distributed={data['distributed']} stages={len(data['stages'])}]",
                file=sys.stderr,
            )
        return 0
    # SSE
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    with client.stream("POST", "/v1/infer", content=req.model_dump_json(), headers=headers) as resp:
        if resp.status_code >= 400:
            try:
                detail = resp.read().decode()
            except Exception:
                detail = "stream error"
            raise SystemExit(f"HTTP {resp.status_code}: {detail}")
        event = None
        out = sys.stdout.buffer if not args.json else None
        collected = bytearray()
        done = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload = json.loads(line.split(":", 1)[1].strip())
                if event == "token":
                    raw = base64.b64decode(payload["bytes_b64"])
                    collected.extend(raw)
                    if out is not None:
                        out.write(raw)
                        out.flush()
                elif event == "start":
                    print(
                        f"[pipeline {len(payload['stages'])} stages distributed={payload['distributed']}]",
                        file=sys.stderr,
                    )
                elif event == "restart":
                    print("[worker failed; restarting pipeline on remaining nodes]", file=sys.stderr)
                    collected.clear()
                elif event == "error":
                    print(f"\nerror: {payload.get('error')}", file=sys.stderr)
                    return 1
                elif event == "done":
                    done = payload
        if args.json:
            print(json.dumps({"text": collected.decode("utf-8", errors="replace"), **(done or {})}, indent=2))
        else:
            if out is not None:
                out.write(b"\n")
                out.flush()
            if done:
                print(
                    f"[tokens={done['tokens']} ttft={done['first_token_ms'] or 0:.0f}ms "
                    f"total={done['total_ms']:.0f}ms distributed={done['distributed']} "
                    f"stages={len(done['stages'])}]",
                    file=sys.stderr,
                )
    return 0


def cmd_plan(args) -> int:
    client, _ = admin_client(args)
    data = _check(client.get("/v1/inference/plan", params={"job_id": args.job_id, "max_stages": args.max_stages}))
    print(json.dumps(data, indent=2))
    return 0


def cmd_health(args) -> int:
    client, url = admin_client(args)
    data = _check(client.get("/healthz"))
    print(json.dumps(data, indent=2))
    return 0


def cmd_monitor(args) -> int:
    _, url = admin_client(args)
    print(f"{url}/dashboard")
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xdc", description="XDC — consumer-grade distributed AI compute")
    p.add_argument("--version", action="version", version=f"XDC {__version__} (native: {native.VERSION})")
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create tokens, TLS material and configs")
    init.add_argument("--dir", default="./xdc-demo")
    init.add_argument("--host", default="127.0.0.1")
    init.add_argument("--port", type=int, default=8787)
    init.add_argument("--tls", dest="tls", action="store_true", default=True)
    init.add_argument("--no-tls", dest="tls", action="store_false")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    co = sub.add_parser("coordinator", help="run the coordinator")
    co.add_argument("--config", required=True)
    co.add_argument("--host")
    co.add_argument("--port", type=int)
    co.add_argument("--log-level", default="INFO")
    co.add_argument("--log-file")
    co.add_argument("--access-log", action="store_true")
    co.set_defaults(func=cmd_coordinator)

    wo = sub.add_parser("worker", help="run a compute worker")
    wo.add_argument("--config", required=True)
    wo.add_argument("--name")
    wo.add_argument("--backend", choices=["auto", "cpu", "cuda", "cuda:0", "mps", "xpu", "rocm"])
    wo.add_argument("--coordinator-url")
    wo.add_argument("--log-level", default="INFO")
    wo.add_argument("--log-file")
    wo.set_defaults(func=cmd_worker)

    doc = sub.add_parser("doctor", help="detect hardware and show honest backend support")
    doc.add_argument("--config")
    doc.add_argument("--backend", default="auto")
    doc.add_argument("--max-memory-mb", type=int, default=0)
    doc.add_argument("--json", action="store_true")
    doc.set_defaults(func=cmd_doctor)

    # shared admin args
    def admin_args(sp):
        sp.add_argument("--config", help="coordinator YAML (URL/token/CA defaults)")
        sp.add_argument("--url")
        sp.add_argument("--token")
        sp.add_argument("--ca")
        sp.add_argument("--insecure", action="store_true")

    st = sub.add_parser("status", help="cluster status")
    admin_args(st)
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_status)

    nd = sub.add_parser("nodes", help="list nodes")
    admin_args(nd)
    nd.add_argument("--json", action="store_true")
    nd.set_defaults(func=cmd_nodes)

    jb = sub.add_parser("jobs", help="list training jobs")
    admin_args(jb)
    jb.add_argument("--json", action="store_true")
    jb.set_defaults(func=cmd_jobs)

    job = sub.add_parser("job", help="show one job")
    admin_args(job)
    job.add_argument("job_id")
    job.set_defaults(func=cmd_job)

    start = sub.add_parser("start", help="start or resume a job")
    admin_args(start)
    start.add_argument("job_id")
    start.add_argument("--wait", action="store_true")
    start.set_defaults(func=cmd_start)

    pause = sub.add_parser("pause", help="pause a job")
    admin_args(pause)
    pause.add_argument("job_id")
    pause.set_defaults(func=cmd_pause)

    cancel = sub.add_parser("cancel", help="cancel a job")
    admin_args(cancel)
    cancel.add_argument("job_id")
    cancel.set_defaults(func=cmd_cancel)

    ck = sub.add_parser("checkpoints", help="list checkpoints")
    admin_args(ck)
    ck.add_argument("job_id")
    ck.set_defaults(func=cmd_checkpoints)

    ban = sub.add_parser("ban", help="ban a node (or unban with --unban)")
    admin_args(ban)
    ban.add_argument("node_id")
    ban.add_argument("--unban", action="store_true")
    ban.set_defaults(func=cmd_ban)

    ds = sub.add_parser("dataset", help="upload/list datasets")
    admin_args(ds)
    ds.add_argument("action", choices=["upload", "list"])
    ds.add_argument("path", nargs="?")
    ds.add_argument("--name")
    ds.set_defaults(func=cmd_dataset)

    mdl = sub.add_parser("model", help="list/upload/pull Hugging Face safetensors models")
    admin_args(mdl)
    mdl.add_argument("action", choices=["list", "show", "add", "pull"])
    mdl.add_argument("target", nargs="?", help="model directory, repo id, or model name")
    mdl.add_argument("--name")
    mdl.add_argument("--repo")
    mdl.add_argument("--revision")
    mdl.set_defaults(func=cmd_model)

    tr = sub.add_parser("train", help="upload dataset and launch a DiLoCo or pipeline job")
    admin_args(tr)
    tr.add_argument("--dataset", required=True)
    tr.add_argument("--dataset-name")
    tr.add_argument("--name", default="xdc-demo")
    tr.add_argument("--algorithm", choices=["diloco", "diloco_decoupled", "pipeline"], default="diloco_decoupled")
    tr.add_argument("--parallelism", choices=["diloco", "pipeline"], default="diloco")
    tr.add_argument("--hf-model", help="name of a model uploaded with `xdc model add/pull`")
    tr.add_argument("--micro-batches", type=int, default=4)
    tr.add_argument("--pipeline-stages", type=int, default=0)
    tr.add_argument("--activation-dtype", choices=["fp32", "fp16"], default="fp32")
    tr.add_argument("--lora-rank", type=int, default=0)
    tr.add_argument("--lora-alpha", type=float, default=0.0)
    tr.add_argument("--island")
    tr.add_argument("--stage-headroom", type=float, default=1.25)
    tr.add_argument("--rounds", type=int, default=10)
    tr.add_argument("--local-steps", type=int, default=20)
    tr.add_argument("--batch-size", type=int, default=4)
    tr.add_argument("--seq-len", type=int, default=128)
    tr.add_argument("--block-size", type=int, default=128)
    tr.add_argument("--layers", type=int, default=4)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--dim", type=int, default=128)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--outer-lr", type=float, default=1.0)
    tr.add_argument("--outer-momentum", type=float, default=0.9)
    tr.add_argument("--weight-decay", type=float, default=0.01)
    tr.add_argument("--grad-clip", type=float, default=1.0)
    tr.add_argument("--min-workers", type=int, default=1)
    tr.add_argument("--max-workers", type=int, default=4)
    tr.add_argument("--quorum", type=float, default=1.0)
    tr.add_argument("--grace", type=float, default=0.5)
    tr.add_argument("--timeout", type=float, default=180)
    tr.add_argument("--eval-every", type=int, default=1)
    tr.add_argument("--eval-batches", type=int, default=4)
    tr.add_argument("--seed", type=int, default=1234)
    tr.add_argument("--compression", choices=["fp32", "fp16"], default="fp16")
    tr.add_argument("--mode", choices=["pretrain", "sft"], default="pretrain")
    tr.add_argument("--resume-from", help="initialize from another job's latest checkpoint")
    tr.add_argument("--wait", action="store_true")
    tr.add_argument("--json", action="store_true")
    tr.set_defaults(func=cmd_train)

    infer = sub.add_parser("infer", help="pipeline-sharded generation")
    admin_args(infer)
    infer.add_argument("--job-id", required=True)
    infer.add_argument("--checkpoint")
    infer.add_argument("--prompt", default="")
    infer.add_argument("--prompt-file")
    infer.add_argument("--max-tokens", type=int, default=32)
    infer.add_argument("--temperature", type=float, default=0.0)
    infer.add_argument("--top-k", type=int, default=0)
    infer.add_argument("--seed", type=int, default=0)
    infer.add_argument("--no-stream", action="store_true")
    infer.add_argument("--json", action="store_true")
    infer.set_defaults(func=cmd_infer)

    plan = sub.add_parser("plan", help="show pipeline layer allocation")
    admin_args(plan)
    plan.add_argument("job_id")
    plan.add_argument("--max-stages", type=int, default=0)
    plan.set_defaults(func=cmd_plan)

    health = sub.add_parser("health", help="coordinator health")
    admin_args(health)
    health.set_defaults(func=cmd_health)

    mon = sub.add_parser("monitor", help="print dashboard URL")
    admin_args(mon)
    mon.set_defaults(func=cmd_monitor)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        log.debug("command failed", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
