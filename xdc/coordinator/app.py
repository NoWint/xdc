"""Coordinator application service and HTTP/WebSocket API."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse

from .. import __version__
from ..config import CoordinatorConfig
from ..crypto import (
    NonceCache,
    constant_time_equal,
    node_id_from_public_key,
    token_hash,
    verify_node_signature,
    verify_signed_request,
)
from ..model.checkpoint import sha256_file
from ..model.hf import HFModelError, validate_model_dir
from ..protocol import (
    AdminStats,
    CheckpointView,
    DatasetInfo,
    HeartbeatRequest,
    HeartbeatResponse,
    InferencePlanView,
    InferenceRequest,
    JobCreate,
    JobView,
    NodeRegistration,
    NodeRegistrationResponse,
    NodeView,
    UpdateAccepted,
    UpdateMeta,
)
from .engine import TrainingEngine
from .inference import InferenceEngine, StageFailure
from .state import StateStore
from .tunnel import TunnelHub
from .scheduler import pooled_resources
from .. import native

log = logging.getLogger("xdc.api")


class Service:
    def __init__(self, cfg: CoordinatorConfig):
        self.cfg = cfg
        self.store = StateStore(cfg.data_dir)
        self.tunnel = TunnelHub()
        self.training = TrainingEngine(self.store, self.tunnel, cfg)
        self.inference = InferenceEngine(self.store, self.tunnel, cfg)
        self.nonces = NonceCache()
        self.started_at = time.time()

    async def start(self) -> None:
        await self.training.start()
        log.info("coordinator ready data_dir=%s", self.store.data_dir)

    async def stop(self) -> None:
        await self.training.stop()
        await self.tunnel.close_all()
        self.store.save()

    # ------------------------------------------------------------------ auth
    def require_admin(self, authorization: str | None) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="admin bearer token required")
        token = authorization.removeprefix("Bearer ").strip()
        if not constant_time_equal(token, self.cfg.admin_token):
            raise HTTPException(status_code=403, detail="invalid admin token")

    async def verify_worker(self, request: Request, body: bytes):
        node_id = request.headers.get("x-xdc-node", "")
        node = self.store.get_node(node_id)
        if not node:
            raise HTTPException(status_code=401, detail="unknown node")
        if node.banned:
            raise HTTPException(status_code=403, detail="node is banned")
        public = request.headers.get("x-xdc-public-key", "")
        if public != node.public_key:
            raise HTTPException(status_code=401, detail="public key mismatch")
        path_q = request.url.path + (("?" + request.url.query) if request.url.query else "")
        ok, reason = verify_signed_request(
            node.public_key, request.method, path_q, body, request.headers, self.nonces
        )
        if not ok:
            raise HTTPException(status_code=401, detail=f"signature rejected: {reason}")
        return node

    def node_view(self, node) -> NodeView:
        tunnel_live = self.tunnel.connected(node.node_id)
        online = (
            not node.banned
            and (
                tunnel_live
                or (time.time() - node.last_seen) <= self.cfg.heartbeat_timeout_s
            )
        )
        current = None
        for job in self.store.list_jobs():
            rnd = job.current_round
            if rnd and node.node_id in rnd.assignments and node.node_id not in rnd.submitted:
                current = f"{job.job_id}:{rnd.round_id}"
                break
        return NodeView(
            node_id=node.node_id,
            name=node.name,
            online=online,
            tunnel_connected=tunnel_live,
            banned=node.banned,
            reputation=node.reputation,
            last_seen_s=max(0.0, time.time() - node.last_seen),
            capabilities=node.caps(),
            labels=node.labels,
            accepted_updates=node.accepted_updates,
            rejected_updates=node.rejected_updates,
            failed_rounds=node.failed_rounds,
            current_assignment=current,
        )

    def stats(self) -> AdminStats:
        nodes = self.store.list_nodes()
        jobs = self.store.list_jobs()
        online = [n for n in nodes if (time.time() - n.last_seen) <= self.cfg.heartbeat_timeout_s and not n.banned]
        pooled = pooled_resources(nodes, self.cfg.heartbeat_timeout_s, self.store)
        return AdminStats(
            nodes_online=len(online),
            nodes_total=len(nodes),
            jobs_running=sum(1 for j in jobs if j.state in {"running", "waiting"}),
            jobs_total=len(jobs),
            pooled_cpu_threads=pooled["pooled_cpu_threads"],
            pooled_memory_mb=pooled["pooled_memory_mb"],
            pooled_gpu_memory_mb=pooled["pooled_gpu_memory_mb"],
            total_tokens_trained=sum(j.tokens_trained for j in jobs),
            uptime_s=time.time() - self.started_at,
            version=__version__,
        )


def create_app(cfg: CoordinatorConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = Service(cfg)
        app.state.service = service
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(
        title="XDC Coordinator",
        version=__version__,
        description="Consumer-grade distributed AI compute — DiLoCo training + pipeline inference",
        lifespan=lifespan,
    )

    def svc(request: Request) -> Service:
        return request.app.state.service

    # ------------------------------------------------------------------ public
    @app.get("/healthz")
    async def healthz(request: Request):
        s = svc(request)
        return {
            "ok": True,
            "version": __version__,
            "uptime_s": time.time() - s.started_at,
            "nodes_online": s.stats().nodes_online,
            "native": native.AVAILABLE,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics(request: Request):
        s = svc(request)
        st = s.stats()
        lines = [
            "# HELP xdc_nodes_online Nodes with a fresh heartbeat",
            "# TYPE xdc_nodes_online gauge",
            f"xdc_nodes_online {st.nodes_online}",
            "# HELP xdc_jobs_running Running/waiting training jobs",
            "# TYPE xdc_jobs_running gauge",
            f"xdc_jobs_running {st.jobs_running}",
            "# HELP xdc_pooled_cpu_threads Sum of worker CPU threads",
            "# TYPE xdc_pooled_cpu_threads gauge",
            f"xdc_pooled_cpu_threads {st.pooled_cpu_threads}",
            "# HELP xdc_pooled_memory_mb Sum of available worker memory",
            "# TYPE xdc_pooled_memory_mb gauge",
            f"xdc_pooled_memory_mb {st.pooled_memory_mb}",
            "# HELP xdc_tokens_trained Total tokens processed in DiLoCo rounds",
            "# TYPE xdc_tokens_trained counter",
            f"xdc_tokens_trained {st.total_tokens_trained}",
        ]
        return "\n".join(lines) + "\n"

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        path = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
        if not path.exists():
            raise HTTPException(404, "dashboard asset missing from this installation")
        return HTMLResponse(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ registration/tunnel
    @app.post("/v1/nodes/register", response_model=NodeRegistrationResponse)
    async def register(req: NodeRegistration, request: Request, x_xdc_join_token: str | None = Header(default=None)):
        s = svc(request)
        if not x_xdc_join_token or not constant_time_equal(x_xdc_join_token, s.cfg.join_token):
            raise HTTPException(status_code=403, detail="invalid join token")
        expected_id = node_id_from_public_key(req.public_key)
        if expected_id != req.node_id:
            raise HTTPException(status_code=400, detail="node_id does not match public key")
        if abs(time.time() - req.timestamp) > 120:
            raise HTTPException(status_code=400, detail="registration timestamp outside allowed skew")
        if not s.nonces.check_and_add("reg:" + req.node_id + ":" + req.nonce):
            raise HTTPException(status_code=400, detail="replayed registration")
        if not verify_node_signature(req.public_key, req.signed_payload(), req.signature):
            raise HTTPException(status_code=401, detail="invalid registration signature")
        existing = s.store.get_node(req.node_id)
        token = __import__("secrets").token_urlsafe(32)
        if existing:
            if existing.public_key != req.public_key:
                raise HTTPException(status_code=409, detail="node key conflict")
            existing.name = req.name
            existing.capabilities = req.capabilities.model_dump(mode="json")
            existing.labels = req.labels
            existing.last_seen = time.time()
            existing.tunnel_token_hash = token_hash(token)
            node = existing
        else:
            from .state import NodeRecord

            node = NodeRecord(
                node_id=req.node_id,
                name=req.name,
                public_key=req.public_key,
                capabilities=req.capabilities.model_dump(mode="json"),
                labels=req.labels,
                last_seen=time.time(),
                tunnel_token_hash=token_hash(token),
                reputation=0.5,
            )
            s.store.nodes[node.node_id] = node
        s.store.save()
        log.info("node registered %s (%s) backend=%s", node.name, node.node_id, req.capabilities.backend)
        return NodeRegistrationResponse(
            node_id=node.node_id,
            tunnel_token=token,
            heartbeat_interval_s=5.0,
            coordinator_version=__version__,
            server_time=time.time(),
        )

    @app.websocket("/v1/tunnel")
    async def tunnel(websocket: WebSocket, node_id: str):
        s: Service = websocket.app.state.service
        auth = websocket.headers.get("authorization", "")
        token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
        node = s.store.get_node(node_id)
        if not node or not token or not constant_time_equal(token_hash(token), node.tunnel_token_hash):
            await websocket.close(code=4401, reason="invalid tunnel token")
            return
        await websocket.accept()
        conn = await s.tunnel.attach(node_id, websocket)
        node.tunnel_connected = True
        node.last_seen = time.time()
        try:
            await s.tunnel.reader(conn)
        except WebSocketDisconnect:
            pass
        finally:
            # A replaced tunnel's reader also exits; only clear the flag if this
            # connection is still the one registered in the hub.
            if s.tunnel.connections.get(node_id) is conn:
                node.tunnel_connected = False
            s.store.save()

    @app.post("/v1/worker/heartbeat", response_model=HeartbeatResponse)
    async def heartbeat(req: HeartbeatRequest, request: Request):
        s = svc(request)
        body = await request.body()
        node = await s.verify_worker(request, body)
        node.last_seen = time.time()
        if req.capabilities:
            node.capabilities = req.capabilities.model_dump(mode="json")
        node.active_sessions = req.active_sessions
        node.training_jobs = req.training_jobs
        node.last_error = req.last_error
        s.store.save(min_interval=1.0)
        return HeartbeatResponse(server_time=time.time())

    @app.get("/v1/worker/assignment")
    async def assignment(request: Request):
        s = svc(request)
        body = await request.body()
        node = await s.verify_worker(request, body)
        node.last_seen = time.time()
        a = s.training.assignment_for(node.node_id)
        if a is None:
            from fastapi import Response

            return Response(status_code=204)
        return a

    @app.get("/v1/jobs/{job_id}/weights")
    async def weights(job_id: str, version: int, request: Request):
        s = svc(request)
        body = await request.body()
        node = await s.verify_worker(request, body)
        job = s.store.get_job(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        rnd = job.current_round
        if not rnd or node.node_id not in rnd.assignments or rnd.base_version != version:
            raise HTTPException(403, "node has no assignment for this model version")
        path = Path(job.model_path)
        if not path.exists() or sha256_file(path) != job.model_sha256:
            raise HTTPException(500, "model file integrity check failed")
        return FileResponse(path, media_type="application/octet-stream", filename="model.safetensors")

    @app.get("/v1/jobs/{job_id}/dataset")
    async def dataset(job_id: str, request: Request):
        s = svc(request)
        body = await request.body()
        node = await s.verify_worker(request, body)
        job = s.store.get_job(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        rnd = job.current_round
        if not rnd or node.node_id not in rnd.assignments:
            raise HTTPException(403, "node has no active assignment for this dataset")
        a = rnd.assignments[node.node_id]
        blob = Path(job.data_train_path).read_bytes()
        start, end = int(a["shard_start"]), int(a["shard_end"])
        return Response(
            content=blob[start:end],
            media_type="application/octet-stream",
            headers={"X-XDC-Shard-SHA256": a.get("shard_sha256", "")},
        )

    @app.post("/v1/worker/updates", response_model=UpdateAccepted)
    async def upload_update(request: Request, x_xdc_meta: str | None = Header(default=None)):
        s = svc(request)
        length = int(request.headers.get("content-length", "0") or 0)
        if length > s.cfg.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, "update too large")
        body = await request.body()
        node = await s.verify_worker(request, body)
        if not x_xdc_meta:
            raise HTTPException(400, "missing X-XDC-Meta")
        try:
            meta = UpdateMeta(**json.loads(base64.b64decode(x_xdc_meta)))
        except Exception as exc:
            raise HTTPException(400, f"invalid update metadata: {exc}") from exc
        if meta.node_id != node.node_id:
            raise HTTPException(403, "metadata node mismatch")
        try:
            # Parsing safetensors and computing the update norm is CPU/disk work;
            # keep it off the event loop so heartbeats and tunnels stay live.
            async with s.training.lock_for(meta.job_id):
                result = await asyncio.to_thread(
                    s.training.accept_update, node.node_id, meta, body
                )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            node.rejected_updates += 1
            node.reputation = max(0.0, node.reputation - 0.1)
            s.store.save()
            raise HTTPException(400, str(exc)) from exc
        return result

    @app.get("/v1/inference/stage/{token}")
    async def stage_model(token: str, request: Request):
        s = svc(request)
        body = await request.body()
        node = await s.verify_worker(request, body)
        try:
            path = s.inference.stage_token(token, node.node_id)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        return FileResponse(path, media_type="application/octet-stream", filename="stage.safetensors")

    @app.get("/v1/pipeline/stage/{token}")
    async def pipeline_stage_file(token: str, request: Request):
        s = svc(request)
        body = await request.body()
        node = await s.verify_worker(request, body)
        try:
            path = s.training.pipeline.stage_token(token, node.node_id, "params")
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        return FileResponse(path, media_type="application/octet-stream", filename="stage.safetensors")

    @app.get("/v1/pipeline/optimizer/{token}")
    async def pipeline_optimizer_file(token: str, request: Request):
        s = svc(request)
        body = await request.body()
        node = await s.verify_worker(request, body)
        try:
            path = s.training.pipeline.stage_token(token, node.node_id, "optimizer")
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        return FileResponse(path, media_type="application/octet-stream", filename="optimizer.safetensors")

    @app.post("/v1/pipeline/upload")
    async def pipeline_upload(request: Request, x_xdc_pipeline_meta: str | None = Header(default=None)):
        s = svc(request)
        length = int(request.headers.get("content-length", "0") or 0)
        if length > s.cfg.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, "pipeline shard too large")
        body = await request.body()
        node = await s.verify_worker(request, body)
        if not x_xdc_pipeline_meta:
            raise HTTPException(400, "missing X-XDC-Pipeline-Meta")
        try:
            meta = json.loads(base64.b64decode(x_xdc_pipeline_meta))
        except Exception as exc:
            raise HTTPException(400, f"invalid pipeline metadata: {exc}") from exc
        if str(meta.get("node_id")) != node.node_id:
            raise HTTPException(403, "metadata node mismatch")
        try:
            # Do not take the job round lock here: the training coroutine holds
            # it while awaiting this upload, so acquiring it would deadlock.
            return await asyncio.to_thread(s.training.pipeline.record_upload, meta, body)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # ------------------------------------------------------------------ models
    @app.get("/v1/admin/models")
    async def list_models(request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        root = s.store.data_dir / "models"
        out = []
        if root.exists():
            for path in sorted(root.iterdir()):
                if not path.is_dir():
                    continue
                try:
                    summary = validate_model_dir(path)
                    out.append({"name": path.name, **summary})
                except Exception as exc:
                    out.append({"name": path.name, "error": str(exc)})
        return out

    @app.put("/v1/admin/models/{name}/files/{filename}")
    async def upload_model_file(
        name: str,
        filename: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        s = svc(request)
        s.require_admin(authorization)
        if not name.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise HTTPException(400, "invalid model name")
        if Path(filename).name != filename or not (
            filename in {
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "generation_config.json",
                "model.safetensors",
                "model.safetensors.index.json",
            }
            or (filename.endswith(".safetensors") and filename != "model.safetensors.index.json")
        ):
            raise HTTPException(400, "file is not on the model allow-list")
        length = int(request.headers.get("content-length", "0") or 0)
        if length <= 0 or length > s.cfg.max_model_upload_mb * 1024 * 1024:
            raise HTTPException(413, "model file is empty or too large")
        root = (s.store.data_dir / "models" / name).resolve()
        if (s.store.data_dir / "models").resolve() not in root.parents:
            raise HTTPException(400, "invalid model path")
        root.mkdir(parents=True, exist_ok=True)
        from ..model.checkpoint import save_json  # noqa: F401

        data = await request.body()
        tmp = root / (filename + ".tmp")
        tmp.write_bytes(data)
        import os

        os.replace(tmp, root / filename)
        return {"ok": True, "name": name, "filename": filename, "bytes": len(data)}

    @app.post("/v1/admin/models/{name}/finalize")
    async def finalize_model(name: str, request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        root = (s.store.data_dir / "models" / name).resolve()
        if (s.store.data_dir / "models").resolve() not in root.parents or not root.is_dir():
            raise HTTPException(404, "model directory not found")
        try:
            return {"name": name, **await asyncio.to_thread(validate_model_dir, root)}
        except HFModelError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/v1/admin/models/{name}/files")
    async def model_files(name: str, request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        root = (s.store.data_dir / "models" / name).resolve()
        if (s.store.data_dir / "models").resolve() not in root.parents or not root.is_dir():
            raise HTTPException(404, "model directory not found")
        return [
            {"filename": p.name, "bytes": p.stat().st_size}
            for p in sorted(root.iterdir())
            if p.is_file() and not p.name.endswith(".tmp")
        ]

    @app.get("/v1/jobs/{job_id}/stage/{stage_index}")
    async def job_stage_file(
        job_id: str,
        stage_index: int,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        s = svc(request)
        s.require_admin(authorization)
        job = s.store.get_job(job_id)
        if not job or job.model_kind != "hf":
            raise HTTPException(404, "HF job not found")
        plan = {int(p["stage_index"]): p for p in job.stage_plan}
        item = plan.get(stage_index)
        if not item:
            raise HTTPException(404, "stage not found")
        path = Path(job.model_path).parent / "pipeline" / f"v{job.version:05d}"
        candidates = sorted(path.glob(f"stage_{stage_index}_*.safetensors"))
        if not candidates:
            raise HTTPException(404, "stage shard not materialized")
        return FileResponse(candidates[0], media_type="application/octet-stream", filename=candidates[0].name)

    # ------------------------------------------------------------------ admin datasets/jobs
    @app.post("/v1/admin/datasets", response_model=DatasetInfo)
    async def upload_dataset(request: Request, name: str = "dataset", authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        length = int(request.headers.get("content-length", "0") or 0)
        if length > s.cfg.max_dataset_mb * 1024 * 1024:
            raise HTTPException(413, "dataset too large")
        data = await request.body()
        if not data:
            raise HTTPException(400, "empty dataset")
        text_lines = data.count(b"\n") + 1
        ds = s.store.save_dataset(name, data, text_lines)
        return ds.info()

    @app.get("/v1/admin/datasets", response_model=list[DatasetInfo])
    async def list_datasets(request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        return [d.info() for d in s.store.datasets.values()]

    @app.post("/v1/jobs", response_model=JobView)
    async def create_job(req: JobCreate, request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        try:
            job = await asyncio.to_thread(s.training.create_job, req)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        return s.training.job_view(job)

    @app.get("/v1/jobs", response_model=list[JobView])
    async def list_jobs(request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        return [s.training.job_view(j) for j in sorted(s.store.list_jobs(), key=lambda j: j.created_at, reverse=True)]

    @app.get("/v1/jobs/{job_id}", response_model=JobView)
    async def get_job(job_id: str, request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        job = s.store.get_job(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return s.training.job_view(job)

    @app.post("/v1/jobs/{job_id}/start", response_model=JobView)
    async def start_job(job_id: str, request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        try:
            return s.training.job_view(s.training.start_job(job_id))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/v1/jobs/{job_id}/pause", response_model=JobView)
    async def pause_job(job_id: str, request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        try:
            return s.training.job_view(s.training.pause_job(job_id))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobView)
    async def cancel_job(job_id: str, request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        try:
            return s.training.job_view(s.training.cancel_job(job_id))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/v1/jobs/{job_id}/checkpoints", response_model=list[CheckpointView])
    async def checkpoints(job_id: str, request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        job = s.store.get_job(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return [
            CheckpointView(
                version=int(c["version"]),
                path=c["path"],
                size_bytes=int(c["size_bytes"]),
                sha256=c["sha256"],
                val_loss=c.get("val_loss"),
                created_at=float(c["created_at"]),
            )
            for c in job.checkpoint_history
        ]

    # ------------------------------------------------------------------ inference
    @app.post("/v1/infer")
    async def infer(req: InferenceRequest, request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        if req.job_id and not s.store.get_job(req.job_id):
            raise HTTPException(404, "job not found")
        if req.stream:
            async def events():
                try:
                    async for ev in s.inference.generate_events(req):
                        payload = json.dumps(ev["data"], separators=(",", ":"))
                        yield f"event: {ev['event']}\ndata: {payload}\n\n"
                except Exception as exc:
                    log.exception("streaming inference failed")
                    payload = json.dumps({"error": str(exc)}, separators=(",", ":"))
                    yield f"event: error\ndata: {payload}\n\n"

            return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})
        try:
            result = await s.inference.generate_text(req)
        except StageFailure as exc:
            raise HTTPException(503, f"worker {exc.node_id} failed: {exc}") from exc
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc
        return {
            "text": result.text,
            "tokens": result.token_count,
            "bytes_b64": base64.b64encode(result.raw).decode(),
            "first_token_ms": result.first_token_ms,
            "total_ms": result.total_ms,
            "distributed": result.distributed,
            "stages": result.plan,
            "attempts": result.attempts,
        }

    # ------------------------------------------------------------------ nodes/stats
    @app.get("/v1/admin/tunnels")
    async def tunnels(request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        return s.tunnel.states()

    @app.get("/v1/nodes", response_model=list[NodeView])
    async def nodes(request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        return [s.node_view(n) for n in sorted(s.store.list_nodes(), key=lambda n: n.first_seen)]

    @app.get("/v1/stats", response_model=AdminStats)
    async def stats(request: Request, authorization: str | None = Header(default=None)):
        s = svc(request)
        s.require_admin(authorization)
        return s.stats()

    @app.post("/v1/admin/nodes/{node_id}/ban")
    async def ban_node(
        node_id: str,
        request: Request,
        banned: bool = True,
        authorization: str | None = Header(default=None),
    ):
        s = svc(request)
        s.require_admin(authorization)
        node = s.store.get_node(node_id)
        if not node:
            raise HTTPException(404, "node not found")
        node.banned = banned
        if banned:
            node.reputation = 0.0
            try:
                await s.tunnel.call(node_id, "shutdown", {}, timeout=2)
            except Exception:
                pass
        s.store.save()
        return s.node_view(node)

    @app.post("/v1/admin/nodes/{node_id}/reputation")
    async def set_reputation(
        node_id: str,
        request: Request,
        value: float = 0.5,
        authorization: str | None = Header(default=None),
    ):
        s = svc(request)
        s.require_admin(authorization)
        node = s.store.get_node(node_id)
        if not node:
            raise HTTPException(404, "node not found")
        node.reputation = max(0.0, min(1.0, value))
        s.store.save()
        return s.node_view(node)

    @app.get("/v1/inference/plan", response_model=InferencePlanView)
    async def inference_plan(
        request: Request,
        job_id: str,
        max_stages: int = 0,
        authorization: str | None = Header(default=None),
    ):
        s = svc(request)
        s.require_admin(authorization)
        req = InferenceRequest(job_id=job_id, max_new_tokens=1)
        model = await asyncio.to_thread(s.inference.resolve, req)
        plans, _ = await s.inference.prepare_plan(model, max_stages)
        return InferencePlanView(
            stages=s.inference.plan_view(plans),
            nodes=[p.node.node_id for p in plans],
            model_sha256=model.sha256,
            total_layers=model.cfg.n_layer,
        )

    return app
