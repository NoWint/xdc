# XDC Security Model

XDC is designed for **self-hosted, invite-only** compute pools. It reduces the attack surface of a volunteer network but does not claim Byzantine fault tolerance.

## 1. Trust boundaries

| Component | Trust |
|---|---|
| Coordinator operator | Trusted to run jobs, store datasets/checkpoints, issue tokens |
| Worker device owner | Semi-trusted; may be curious, faulty, or malicious |
| Network | Untrusted; TLS + signed requests |
| Dataset | Sensitive; only assigned shards are transferred to workers |
| Model weights | Public or private per deployment; workers receive full training weights or their inference stage slice |

A malicious coordinator can read data and serve poisoned weights. Self-host the coordinator on hardware you control.

## 2. Identity

- Ed25519 key is generated locally under `data_dir/identity/node_key.pem` (0600).
- `node_id` is derived from the public key; the coordinator rejects a mismatch.
- Registration is signed over the canonical fields, timestamp and nonce, and gated by the cluster join token.
- Every subsequent worker request signs `method`, path+query, timestamp, nonce, and `sha256(body)`.
- Replay protection: 300s nonce cache + 120s timestamp skew.
- Tunnel tokens are random 32-byte values stored as SHA-256 hashes; the WebSocket carries them only over TLS.
- Identity keys are never transmitted. Back up the key if you want the node ID to survive reinstalls.

## 3. Transport

- `xdc init` generates a private CA and a server certificate with SANs for localhost and discovered LAN IPs.
- The worker verifies the coordinator certificate against `tls_ca`.
- `allow_insecure: true` is only for loopback HTTP development. Production configs leave it false.
- HTTP clients in XDC disable environment proxies so a local coordinator cannot be silently routed through a system proxy. Set an explicit URL or use an overlay network if a proxy is required.
- Public exposure: put the coordinator behind a reverse proxy with a real certificate, or use WireGuard/Tailscale. Do not expose the admin token over HTTP.

Not implemented: mutual TLS, certificate pinning beyond the CA file, encrypted storage at rest.

## 4. Request and payload hardening

- Admin endpoints require `Authorization: Bearer <admin_token>`; comparison is constant-time.
- Worker endpoints validate node identity, signature, nonce, timestamp and (for assignment-bound files) the active assignment.
- Model/dataset downloads require a signed request. Data is sliced to the assigned shard; stage weights are sliced to the stage.
- Updates are raw safetensors, never pickle:
  - exact parameter-name and shape match against the current global state;
  - finite check on every tensor;
  - norm must be finite and > 0, with an absolute ceiling;
  - body size limited by `max_upload_mb`;
  - a round accepts at most one update per node; late/duplicate updates are rejected.
- Training code is fixed by the coordinator build. Assignments cannot execute arbitrary commands.
- Workers execute one training assignment at a time; inference sessions are capped by `max_inference_sessions`.
- The dashboard never persists the admin token to disk; it uses `sessionStorage` at the operator's browser.

## 5. Aggregation defenses

- Token-weighted averaging.
- L2-norm sanity checks.
- With ≥3 updates: median-based adaptive filtering (drop updates above `max_update_norm_ratio × median`).
- Reputation penalties for invalid uploads, losses and failed rounds; scheduler excludes banned/low-reputation nodes.
- Absolute update norm ceiling protects against extreme values.

Limitations:

- Two-worker averaging has no Byzantine quorum. The outlier filter is an availability/sanity guard, not proof of correctness.
- A poisoned but norm-normal update can bias the global model.
- XDC does not implement proof-of-learning, redundant verification, secure aggregation, or differential privacy.
- For sensitive data, use trusted devices and a private coordinator.

## 6. Denial of service

- Request body limits and signature checks limit trivial unauthorized work.
- A malicious registered worker can still consume download bandwidth or repeatedly fail rounds; ban it and rotate the join token.
- There is no rate limiter on `/healthz` or `/dashboard`; place a reverse proxy in front for public deployments.
- The coordinator is a single point of failure by design in v0.1.

## 7. Secrets

- The coordinator YAML holds admin/join tokens; `xdc init` writes it with mode 0600. Keep it off backups that are broadly accessible.
- The worker identity private key is mode 0600. Do not copy it between devices unless you deliberately want to clone an identity.
- Logs do not print request bodies, admin tokens, worker keys, or tunnel tokens. `xdc init` prints the generated tokens once.
- Rotate the join token by initializing a new cluster or editing the config and restarting the coordinator; existing workers do not need the join token after registration, but they do need their signed identity to remain accepted.

## 8. Reporting

Do not open a public issue with an exploit, private key, token, dataset or checkpoint. Contact the repository owner privately. The project is a research/engineering prototype; treat it accordingly.
