"""Node identity, request signing, replay protection and TLS bootstrap.

No shared worker secret is stored on the coordinator.  Every node generates an
Ed25519 key locally; the private key never leaves the device.  Registration binds
the public key to a deterministic node id, and every state-changing/read request
afterwards carries a signature over method, path, timestamp, nonce and body hash.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.x509.oid import NameOID

SIGNATURE_VERSION = "xdc-v1"


def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def canonical_json(data: Mapping) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def node_id_from_public_key(public_key_b64: str) -> str:
    raw = b64d(public_key_b64)
    return "node-" + hashlib.sha256(raw).hexdigest()[:24]


def public_key_b64(private_key: ed25519.Ed25519PrivateKey) -> str:
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return b64e(raw)


@dataclass
class NodeIdentity:
    node_id: str
    private_key: ed25519.Ed25519PrivateKey
    public_key: str

    @classmethod
    def generate(cls) -> "NodeIdentity":
        key = ed25519.Ed25519PrivateKey.generate()
        pub = public_key_b64(key)
        return cls(node_id=node_id_from_public_key(pub), private_key=key, public_key=pub)

    @classmethod
    def load_or_create(cls, data_dir: str | Path) -> "NodeIdentity":
        path = Path(data_dir) / "identity" / "node_key.pem"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raw = path.read_bytes()
            key = serialization.load_pem_private_key(raw, password=None)
            if not isinstance(key, ed25519.Ed25519PrivateKey):
                raise ValueError(f"{path} is not an Ed25519 key")
        else:
            key = ed25519.Ed25519PrivateKey.generate()
            raw = key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
        pub = public_key_b64(key)
        return cls(node_id=node_id_from_public_key(pub), private_key=key, public_key=pub)

    def sign(self, payload: bytes) -> str:
        return b64e(self.private_key.sign(payload))

    def verify(self, payload: bytes, signature_b64: str) -> bool:
        try:
            self.private_key.public_key().verify(b64d(signature_b64), payload)
            return True
        except Exception:
            return False


def verify_node_signature(public_key_b64_value: str, payload: bytes, signature_b64: str) -> bool:
    try:
        key = ed25519.Ed25519PublicKey.from_public_bytes(b64d(public_key_b64_value))
        key.verify(b64d(signature_b64), payload)
        return True
    except Exception:
        return False


def request_payload(method: str, path_with_query: str, timestamp: str, nonce: str, body: bytes) -> bytes:
    body_hash = hashlib.sha256(body).hexdigest()
    return "\n".join([SIGNATURE_VERSION, method.upper(), path_with_query, timestamp, nonce, body_hash]).encode()


def sign_request(identity: NodeIdentity, method: str, path_with_query: str, body: bytes = b"") -> dict[str, str]:
    ts = str(int(time.time()))
    nonce = secrets.token_hex(16)
    sig = identity.sign(request_payload(method, path_with_query, ts, nonce, body))
    return {
        "X-XDC-Node": identity.node_id,
        "X-XDC-Public-Key": identity.public_key,
        "X-XDC-Timestamp": ts,
        "X-XDC-Nonce": nonce,
        "X-XDC-Signature": sig,
    }


@dataclass
class NonceCache:
    ttl_s: int = 300
    seen: dict[str, float] = field(default_factory=dict)

    def check_and_add(self, key: str) -> bool:
        now = time.time()
        if len(self.seen) > 20000:
            self.seen = {k: v for k, v in self.seen.items() if now - v < self.ttl_s}
        if key in self.seen and now - self.seen[key] < self.ttl_s:
            return False
        self.seen[key] = now
        return True


def verify_signed_request(
    public_key: str,
    method: str,
    path_with_query: str,
    body: bytes,
    headers: Mapping[str, str],
    nonces: NonceCache,
    max_skew_s: int = 120,
) -> tuple[bool, str]:
    headers = {str(k).lower(): v for k, v in headers.items()}
    try:
        ts = headers["x-xdc-timestamp"]
        nonce = headers["x-xdc-nonce"]
        sig = headers["x-xdc-signature"]
    except KeyError:
        return False, "missing signature headers"
    try:
        if abs(time.time() - int(ts)) > max_skew_s:
            return False, "timestamp outside allowed skew"
    except ValueError:
        return False, "invalid timestamp"
    payload = request_payload(method, path_with_query, ts, nonce, body)
    # Verify first so unauthenticated traffic cannot grow the replay cache.
    if not verify_node_signature(public_key, payload, sig):
        return False, "bad signature"
    if not nonces.check_and_add(headers.get("x-xdc-node", "") + ":" + nonce):
        return False, "replayed nonce"
    return True, "ok"


def generate_tls_material(out_dir: str | Path, hosts: list[str] | None = None) -> dict[str, str]:
    """Create a private CA and a server certificate; returns file paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hosts = hosts or []
    now = _utcnow()
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "XDC Local CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now.replace(year=now.year + 10))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    server_key = ec.generate_private_key(ec.SECP256R1())
    alt_names: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ipaddress.ip_address("::1")),
    ]
    for host in hosts:
        if not host or host in {"0.0.0.0", "::"}:
            continue
        try:
            alt_names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            alt_names.append(x509.DNSName(host))
    try:
        alt_names.append(x509.DNSName(socket.gethostname()))
    except Exception:
        pass
    try:
        for ip in _local_ipv4s():
            alt_names.append(x509.IPAddress(ipaddress.ip_address(ip)))
    except Exception:
        pass
    dedup = []
    seen = set()
    for name in alt_names:
        key = repr(name)
        if key not in seen:
            seen.add(key)
            dedup.append(name)
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "XDC Coordinator")])
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now.replace(year=now.year + 3))
        .add_extension(x509.SubjectAlternativeName(dedup), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    paths = {
        "ca": str(out / "ca.pem"),
        "cert": str(out / "server.crt"),
        "key": str(out / "server.key"),
    }
    _write_private(out / "ca.key", ca_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    _write_private(paths["key"], server_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    Path(paths["ca"]).write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    Path(paths["cert"]).write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    return paths

def _write_private(path: str | Path, data: bytes) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def _utcnow():
    import datetime

    return datetime.datetime.now(datetime.timezone.utc)


def _local_ipv4s() -> list[str]:
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    try:
        # No packet is sent; connect() merely asks the kernel to select a route.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
        finally:
            s.close()
    except Exception:
        pass
    try:
        # Multi-homed laptops (Wi-Fi + Ethernet + VPN): include every local
        # IPv4 address so a worker on any interface can verify the certificate.
        import psutil

        for addrs in psutil.net_if_addrs().values():
            for addr in addrs:
                if addr.family == socket.AF_INET and addr.address:
                    ips.add(addr.address)
    except Exception:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127.") and ip != "0.0.0.0")


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)[:nbytes + 4]


def constant_time_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())
