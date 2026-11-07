"""Worker daemon: registration, heartbeat, assignment polling and tunnel RPC."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import psutil
import websockets

from .. import __version__, native
from ..config import WorkerConfig
from ..crypto import NodeIdentity
from ..protocol import Assignment
from .client import ApiError, SigningClient
from .hardware import detect
from .runtime import WorkerRuntime

log = logging.getLogger("xdc.worker")


class WorkerDaemon:
    def __init__(self, cfg: WorkerConfig):
        self.cfg = cfg
        self.identity = NodeIdentity.load_or_create(cfg.data_dir)
        detected, caps = detect(cfg.backend, cfg.max_memory_mb)
        self.detected = detected
        self.caps = caps
        self.device = __import__("torch").device(detected.selected)
        self.name = cfg.name or f"{socket.gethostname()}-{self.identity.node_id[-6:]}"
        self.client = SigningClient(
            cfg.coordinator_url,
            self.identity,
            join_token=cfg.join_token,
            tls_ca=cfg.tls_ca,
            allow_insecure=cfg.allow_insecure,
        )
        self.runtime = WorkerRuntime(
            cfg,
            self.client,
            caps,
            self.device,
            Path(cfg.data_dir) / "cache",
        )
        self.stop_event = asyncio.Event()
        self.tunnel_token: str | None = None
        self.register_lock = asyncio.Lock()
        self.tasks: list[asyncio.Task] = []
        self.started_at = time.time()
        self.tunnel_generation = 0

    # ------------------------------------------------------------------ setup
    def _tunnel_url(self) -> str:
        parsed = urlparse(self.cfg.coordinator_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        netloc = parsed.netloc
        base_path = parsed.path.rstrip("/")
        return f"{scheme}://{netloc}{base_path}/v1/tunnel?node_id={self.identity.node_id}"

    def _refresh_caps(self) -> None:
        vm = psutil.virtual_memory()
        avail = int(vm.available // (1024 * 1024))
        if self.cfg.max_memory_mb > 0:
            avail = min(avail, self.cfg.max_memory_mb)
        self.caps.available_memory_mb = avail

    def _write_status(self) -> None:
        try:
            path = Path(self.cfg.data_dir) / "worker_status.json"
            data = {
                "node_id": self.identity.node_id,
                "name": self.name,
                "backend": self.caps.backend,
                "device": self.caps.device_name,
                "coordinator": self.cfg.coordinator_url,
                "started_at": self.started_at,
                "updated_at": time.time(),
                "tunnel_generation": self.tunnel_generation,
                "training": self.runtime.training,
                "last_training": self.runtime.last_training,
                "last_error": self.runtime.last_error,
                "sessions": len(self.runtime.sessions),
                "native": native.VERSION,
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, path)
        except Exception:
            pass

    # ------------------------------------------------------------------ registration
    async def ensure_registered(self, force: bool = False) -> str:
        async with self.register_lock:
            if self.tunnel_token and not force:
                return self.tunnel_token
            backoff = 1.0
            while not self.stop_event.is_set():
                try:
                    self._refresh_caps()
                    resp = await self.client.register(self.name, self.caps, self.cfg.labels)
                    self.tunnel_token = resp["tunnel_token"]
                    log.info(
                        "registered %s with %s backend=%s device=%s",
                        self.identity.node_id,
                        self.cfg.coordinator_url,
                        self.caps.backend,
                        self.caps.device_name,
                    )
                    self._write_status()
                    return self.tunnel_token
                except ApiError as exc:
                    if exc.status in {401, 403, 404}:
                        log.error("registration rejected: %s", exc)
                        # Join token / policy failures are not transient.
                        if exc.status == 403 and "banned" in str(exc).lower():
                            await asyncio.sleep(5)
                        else:
                            await asyncio.sleep(min(backoff, 30))
                    else:
                        log.warning("registration failed: %s", exc)
                        await asyncio.sleep(min(backoff, 30))
                except Exception as exc:
                    log.warning("registration failed: %s", exc)
                    await asyncio.sleep(min(backoff, 30))
                backoff = min(backoff * 1.7, 30)
            raise RuntimeError("worker stopping")

    # ------------------------------------------------------------------ loops
    async def heartbeat_loop(self) -> None:
        interval = max(1.0, self.cfg.heartbeat_interval_s)
        while not self.stop_event.is_set():
            try:
                await self.ensure_registered()
                self._refresh_caps()
                payload = {
                    "capabilities": self.caps.model_dump(mode="json"),
                    "active_sessions": len(self.runtime.sessions),
                    "training_jobs": 1 if self.runtime.training else 0,
                    "tunnel_connected": False,
                    "last_error": self.runtime.last_error,
                }
                try:
                    await self.client.heartbeat(payload)
                except ApiError as exc:
                    if exc.status in {401, 403}:
                        self.tunnel_token = None
                        await self.ensure_registered(force=True)
                        await self.client.heartbeat(payload)
                    else:
                        raise
                self._write_status()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A transient timeout is not an authentication failure; keep the
                # tunnel token so a brief coordinator stall does not force a
                # re-registration and a tunnel handshake.
                log.warning("heartbeat failed: %s: %s", exc.__class__.__name__, exc)
            await asyncio.sleep(interval)

    async def poll_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.runtime.training:
                await asyncio.sleep(0.3)
                continue
            try:
                data = await self.client.assignment()
                if data is None:
                    await asyncio.sleep(self.cfg.poll_interval_s)
                    continue
                assignment = Assignment(**data)
                log.info(
                    "assignment job=%s round=%s shard=%s..%s workers=%s",
                    assignment.job_id,
                    assignment.round_index,
                    assignment.shard_start,
                    assignment.shard_end,
                    assignment.world_size,
                )
                ok = await self.runtime.run_assignment(assignment)
                self._write_status()
                await asyncio.sleep(0.2 if ok else 2.0)
            except asyncio.CancelledError:
                raise
            except ApiError as exc:
                if exc.status in {401, 403}:
                    self.tunnel_token = None
                    await self.ensure_registered(force=True)
                else:
                    log.warning("assignment poll failed: %s", exc)
                    await asyncio.sleep(2.0)
            except Exception as exc:
                log.warning("assignment poll failed: %s", exc)
                await asyncio.sleep(2.0)

    async def tunnel_loop(self) -> None:
        backoff = 1.0
        while not self.stop_event.is_set():
            token = None
            try:
                token = await self.ensure_registered()
                url = self._tunnel_url()
                async with websockets.connect(
                    url,
                    additional_headers={"Authorization": f"Bearer {token}"},
                    ssl=self.client.ssl_context,
                    max_size=64 * 1024 * 1024,
                    compression=None,
                    open_timeout=15,
                    ping_interval=30,
                    ping_timeout=90,
                    proxy=None,
                ) as ws:
                    self.tunnel_generation += 1
                    backoff = 1.0
                    await ws.send(
                        json.dumps(
                            {
                                "type": "hello",
                                "version": __version__,
                                "backend": self.caps.backend,
                                "device": self.caps.device_name,
                            }
                        )
                    )
                    self._write_status()
                    log.info("tunnel connected to coordinator")
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        if msg.get("type") != "req":
                            continue
                        req_id = msg.get("id")
                        method = msg.get("method", "")
                        payload = msg.get("payload") or {}
                        if method == "shutdown":
                            await ws.send(json.dumps({"type": "resp", "id": req_id, "ok": True, "result": {"bye": True}}))
                            log.warning("coordinator requested worker shutdown")
                            self.stop_event.set()
                            break
                        try:
                            result = await self.runtime.handle_rpc(method, payload)
                            response = {"type": "resp", "id": req_id, "ok": True, "result": result}
                        except Exception as exc:
                            response = {
                                "type": "resp",
                                "id": req_id,
                                "ok": False,
                                "error": f"{exc.__class__.__name__}: {exc}",
                            }
                        await ws.send(json.dumps(response, separators=(",", ":")))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # 401 during handshake is surfaced as InvalidStatus.
                log.warning("tunnel disconnected/error: %s", exc)
                if "401" in str(exc) or "403" in str(exc) or "4401" in str(exc):
                    self.tunnel_token = None
                self._write_status()
                await asyncio.sleep(min(backoff, 15))
                backoff = min(backoff * 1.8, 15)

    # ------------------------------------------------------------------ lifecycle
    async def run(self) -> None:
        log.info(
            "starting worker node_id=%s backend=%s device=%s score=%.1f",
            self.identity.node_id,
            self.caps.backend,
            self.caps.device_name,
            self.caps.compute_score,
        )
        await self.ensure_registered()
        self.tasks = [
            asyncio.create_task(self.heartbeat_loop(), name="heartbeat"),
            asyncio.create_task(self.poll_loop(), name="poll"),
            asyncio.create_task(self.tunnel_loop(), name="tunnel"),
        ]
        try:
            await self.stop_event.wait()
        finally:
            for task in self.tasks:
                task.cancel()
            for task in self.tasks:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            self.runtime.release_all()
            await self.client.aclose()
            self._write_status()
            log.info("worker stopped")

    def stop(self) -> None:
        self.stop_event.set()


async def run_worker(cfg: WorkerConfig) -> None:
    daemon = WorkerDaemon(cfg)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, daemon.stop)
        except (NotImplementedError, RuntimeError):
            pass
    await daemon.run()
