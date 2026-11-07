"""Outbound worker WebSocket tunnels.

Workers connect *to* the coordinator, so the design works behind consumer NAT and
does not require port forwarding for inference RPC.  Training payloads use regular
HTTP requests for streaming and retry semantics; latency-sensitive pipeline stage
calls use this multiplexed tunnel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

log = logging.getLogger("xdc.tunnel")


class TunnelError(RuntimeError):
    pass


class TunnelUnavailable(TunnelError):
    pass


@dataclass
class Connection:
    node_id: str
    websocket: WebSocket
    connected_at: float = field(default_factory=time.time)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class TunnelHub:
    def __init__(self) -> None:
        self.connections: dict[str, Connection] = {}
        self.pending: dict[str, tuple[str, asyncio.Future]] = {}
        self._lock = asyncio.Lock()

    def connected(self, node_id: str) -> bool:
        conn = self.connections.get(node_id)
        if not conn:
            return False
        try:
            return conn.websocket.client_state.name == "CONNECTED"
        except Exception:
            return True

    async def attach(self, node_id: str, websocket: WebSocket) -> Connection:
        conn = Connection(node_id=node_id, websocket=websocket)
        old: Connection | None
        async with self._lock:
            old = self.connections.get(node_id)
            self.connections[node_id] = conn
        if old is not None:
            try:
                await old.websocket.close(code=4000, reason="replaced by newer tunnel")
            except Exception:
                pass
        log.info("tunnel attached node=%s", node_id)
        return conn

    async def detach(self, conn: Connection) -> None:
        async with self._lock:
            if self.connections.get(conn.node_id) is conn:
                self.connections.pop(conn.node_id, None)
        self._fail_node(conn.node_id, "tunnel disconnected")
        log.info("tunnel detached node=%s", conn.node_id)

    def _fail_node(self, node_id: str, reason: str) -> None:
        for req_id, (owner, future) in list(self.pending.items()):
            if owner == node_id and not future.done():
                future.set_exception(TunnelUnavailable(reason))
                self.pending.pop(req_id, None)

    async def reader(self, conn: Connection) -> None:
        try:
            while True:
                raw = await conn.websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except Exception:
                    log.warning("invalid tunnel frame from %s", conn.node_id)
                    continue
                if msg.get("type") == "resp":
                    req_id = msg.get("id")
                    item = self.pending.pop(req_id, None)
                    if item is None:
                        continue
                    _, future = item
                    if future.done():
                        continue
                    if msg.get("ok"):
                        future.set_result(msg.get("result"))
                    else:
                        future.set_exception(TunnelError(str(msg.get("error", "worker error"))))
                elif msg.get("type") == "log":
                    log.info("worker %s: %s", conn.node_id, msg.get("message", ""))
                elif msg.get("type") == "hello":
                    log.info("worker %s tunnel hello: %s", conn.node_id, msg.get("version", "?"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.info("tunnel reader stopped node=%s error=%s", conn.node_id, exc)
        finally:
            await self.detach(conn)

    async def call(self, node_id: str, method: str, payload: dict[str, Any], timeout: float = 30.0) -> Any:
        conn = self.connections.get(node_id)
        if conn is None:
            raise TunnelUnavailable(f"node {node_id} has no active tunnel")
        req_id = uuid.uuid4().hex
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending[req_id] = (node_id, future)
        message = {"type": "req", "id": req_id, "method": method, "payload": payload}
        try:
            async with conn.send_lock:
                await conn.websocket.send_text(json.dumps(message, separators=(",", ":")))
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TunnelError(f"worker {node_id} timed out on {method}") from exc
        finally:
            self.pending.pop(req_id, None)

    async def close_all(self) -> None:
        conns = list(self.connections.values())
        self.connections.clear()
        for conn in conns:
            try:
                await conn.websocket.close(code=1001)
            except Exception:
                pass
        for _, future in self.pending.values():
            if not future.done():
                future.set_exception(TunnelUnavailable("coordinator shutting down"))
        self.pending.clear()
