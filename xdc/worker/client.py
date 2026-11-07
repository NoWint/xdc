"""Worker HTTP client: signed requests, downloads, registration."""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import time
from pathlib import Path
from typing import Any

import httpx

from ..crypto import NodeIdentity, canonical_json, sign_request
from ..model.checkpoint import sha256_file
from ..protocol import Capabilities


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class SigningClient:
    def __init__(
        self,
        base_url: str,
        identity: NodeIdentity,
        *,
        join_token: str,
        tls_ca: str | None = None,
        allow_insecure: bool = False,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.identity = identity
        self.join_token = join_token
        verify: bool | str = True
        if self.base_url.startswith("https://"):
            if tls_ca:
                verify = tls_ca
            elif allow_insecure:
                verify = False
        elif not allow_insecure:
            raise ValueError("plain HTTP coordinator URL requires allow_insecure=true")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=15.0),
            verify=verify,
            trust_env=False,
        )
        self.ssl_context = self._ssl_context(tls_ca, allow_insecure)

    def _ssl_context(self, tls_ca: str | None, allow_insecure: bool) -> ssl.SSLContext | None:
        if not self.base_url.startswith("https://"):
            return None
        ctx = ssl.create_default_context(cafile=tls_ca) if tls_ca else ssl.create_default_context()
        if allow_insecure and not tls_ca:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    async def aclose(self) -> None:
        await self.client.aclose()

    # ------------------------------------------------------------------ signed primitives
    def _headers(self, method: str, path: str, body: bytes) -> dict[str, str]:
        headers = sign_request(self.identity, method, path, body)
        headers["Content-Type"] = "application/json"
        return headers

    async def get_json(self, path: str) -> dict[str, Any] | None:
        resp = await self.client.get(path, headers=self._headers("GET", path, b""))
        if resp.status_code == 204:
            return None
        self._raise(resp)
        return resp.json()

    async def post_json(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        extra = {} if timeout is None else {"timeout": timeout}
        resp = await self.client.post(path, content=body, headers=self._headers("POST", path, body), **extra)
        self._raise(resp)
        return resp.json()

    async def post_raw(self, path: str, body: bytes, extra_headers: dict[str, str] | None = None) -> dict[str, Any]:
        headers = self._headers("POST", path, body)
        if extra_headers:
            headers.update(extra_headers)
        resp = await self.client.post(path, content=body, headers=headers)
        self._raise(resp)
        return resp.json()

    @staticmethod
    def _raise(resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise ApiError(resp.status_code, str(detail))

    # ------------------------------------------------------------------ registration
    async def register(self, name: str, caps: Capabilities, labels: dict[str, str]) -> dict[str, Any]:
        ts = int(time.time())
        nonce = os.urandom(16).hex()
        body = {
            "node_id": self.identity.node_id,
            "name": name,
            "public_key": self.identity.public_key,
            "capabilities": caps.model_dump(mode="json"),
            "labels": labels,
            "timestamp": ts,
            "nonce": nonce,
        }
        signature = self.identity.sign(canonical_json(body))
        body["signature"] = signature
        raw = json.dumps(body, separators=(",", ":"), sort_keys=True).encode()
        resp = await self.client.post(
            "/v1/nodes/register",
            content=raw,
            headers={"Content-Type": "application/json", "X-XDC-Join-Token": self.join_token},
        )
        self._raise(resp)
        return resp.json()

    async def heartbeat(self, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
        # Liveness must fail fast: a heartbeat that blocks for the full 60s
        # request timeout cannot tell the coordinator anything useful.
        return await self.post_json("/v1/worker/heartbeat", payload, timeout=timeout)

    async def assignment(self):
        return await self.get_json("/v1/worker/assignment")

    # ------------------------------------------------------------------ files
    async def download(self, path: str, expected_sha: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha256_file(dest) == expected_sha:
            return dest
        tmp = dest.with_name(dest.name + f".part-{os.getpid()}")
        h = hashlib.sha256()
        headers = self._headers("GET", path, b"")
        try:
            async with self.client.stream("GET", path, headers=headers) as resp:
                self._raise(resp)
                with open(tmp, "wb") as fh:
                    async for chunk in resp.aiter_bytes(1024 * 1024):
                        fh.write(chunk)
                        h.update(chunk)
            if h.hexdigest() != expected_sha:
                raise ApiError(0, f"sha256 mismatch for {path}")
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return dest

    async def fetch_bytes(self, path: str, expected_sha: str) -> bytes:
        headers = self._headers("GET", path, b"")
        resp = await self.client.get(path, headers=headers)
        self._raise(resp)
        data = resp.content
        if hashlib.sha256(data).hexdigest() != expected_sha:
            raise ApiError(0, f"sha256 mismatch for {path}")
        return data
