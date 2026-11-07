# Deploying XDC

This guide covers macOS and Windows, a LAN cluster, and exposing a coordinator across the Internet. The worker only makes outbound connections; the coordinator is the one component that needs a listening port.

## 1. Requirements

- Python 3.10–3.13 (3.11 recommended)
- ~2GB free RAM for the demo model; more for larger models
- 1GB free disk for checkpoints/cache
- A C++17 compiler is optional. Without it, XDC uses the Python fallback and says so in `xdc doctor`.
- Stable networking between each worker and the coordinator. Workers may be behind NAT.

Supported execution backends:

| Backend | Status |
|---|---|
| CPU (x86_64/arm64) | supported, tested on Apple M4 |
| Apple MPS | supported, tested on M4 |
| NVIDIA CUDA | code path present; **unverified** (no NVIDIA hardware in test environment) |
| AMD ROCm | code path present; **unverified** |
| Intel XPU | code path present; **unverified** |
| MLX | detected/reported; not used by the execution engine |
| NPU | reported unavailable; no vendor-neutral PyTorch backend |

## 2. macOS install

```bash
git clone https://github.com/NoWint/xdc.git
cd xdc
bash scripts/install.sh
```

The installer creates `.venv`, installs the package, attempts the optional native extension, and runs `xdc doctor`.

Manual equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
xdc doctor
```

## 3. Windows install

Open PowerShell in the repository:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install.ps1
```

Manual:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m xdc doctor
```

If the native extension fails to compile, installation still succeeds with the Python fallback. Install "Desktop development with C++" from Visual Studio Build Tools if you want the native path. Do not run untrusted `.ps1` files; the supplied script only creates a venv and installs this repository.

## 4. Initialise a cluster

On the coordinator machine:

```bash
xdc init --dir ./xdc-demo --host 0.0.0.0
```

This creates:

```
xdc-demo/
  coordinator.yaml
  worker1.yaml
  worker2.yaml
  data/
  tls/{ca.pem,server.crt,server.key}
```

The generated admin and join tokens are printed once and stored in mode-0600 YAML. Keep the admin token private. The join token is required to enrol workers; rotate it by re-running init only for a new cluster.

Start the coordinator:

```bash
xdc coordinator --config ./xdc-demo/coordinator.yaml --access-log
```

Open `http://127.0.0.1:8787/dashboard` (or `https://...`), paste the admin token, and start workers.

Start two local workers in two terminals:

```bash
xdc worker --config ./xdc-demo/worker1.yaml
xdc worker --config ./xdc-demo/worker2.yaml
```

Check:

```bash
xdc status --config ./xdc-demo/coordinator.yaml
xdc doctor
```

## 5. Add a second physical device

1. Copy `worker1.yaml` (and `tls/ca.pem`) to the second device over a secure channel.
2. Edit the copy:

```yaml
worker:
  coordinator_url: https://192.168.1.25:8787
  join_token: <same join token>
  data_dir: /path/on/second/device   # or C:\xdc\worker
  name: mac-studio
  tls_ca: /path/to/ca.pem
  backend: auto
```

3. Ensure the coordinator host firewall allows TCP 8787 from the LAN.
4. Start the worker. It registers, heartbeats, and opens an outbound tunnel.

The generated certificate SAN includes local IPs discovered at init time. If the coordinator's LAN IP changes, regenerate TLS material or add that IP via `xdc init --host` (use a new directory) / a real DNS certificate. A hostname mismatch is reported by the worker as a TLS verification error.

### macOS firewall

Allow incoming connections for the Python executable or explicitly allow port 8787:

```bash
# Example; interface names differ
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add $(which python3)
```

### Windows firewall

Run PowerShell as Administrator:

```powershell
New-NetFirewallRule -DisplayName "XDC Coordinator" -Direction Inbound -Protocol TCP -LocalPort 8787 -Action Allow
```

Only allow the port on trusted/private networks if immediate Internet exposure is not needed.

## 6. Train and generate

```bash
# macOS path; on Windows use .\venv\Scripts\xdc.exe
xdc train --config ./xdc-demo/coordinator.yaml \
  --dataset datasets/tiny_stories.txt \
  --rounds 10 --local-steps 20 --min-workers 2 --wait
```

The CLI uploads the dataset once, creates the job and waits. Model defaults are a 4-layer, 128-dim byte-level GPT (875k parameters) for fast consumer validation; adjust with `--layers/--dim/--heads/--block-size`.

Inspect:

```bash
xdc jobs --config ./xdc-demo/coordinator.yaml
xdc job --config ./xdc-demo/coordinator.yaml <job-id>
xdc checkpoints --config ./xdc-demo/coordinator.yaml <job-id>
```

Stop/resume:

```bash
xdc pause  --config ... <job-id>
xdc start  --config ... <job-id> --wait
```

Resume from a snapshot into a new job:

```bash
xdc train --config ... --dataset datasets/instructions_sample.jsonl \
  --resume-from <old-job-id> --rounds 5 --min-workers 2 --wait
```

Note: the current CLI's `--resume-from` requires that the new model config matches the old checkpoint's architecture exactly. The model flags default to 4×128; if the old job used different sizes, pass the same flags.

Inference:

```bash
xdc infer --config ./xdc-demo/coordinator.yaml \
  --job-id <job-id> \
  --prompt "Mira found a brass key near" \
  --max-tokens 32 --temperature 0
```

Streaming is the default and prints raw bytes as they arrive. `--no-stream` collects the result and prints it; `--json` returns exact bytes plus stage placement.

## 7. Across the Internet

XDC workers connect outbound, so a worker behind CGNAT can still join if it can reach the coordinator. The coordinator itself needs a reachable address. Choose one:

1. **Port forward** TCP 8787 on the router to the coordinator. Prefer TLS and a non-default port. Do not expose the admin token.
2. **Reverse proxy** with a public DNS name and a real certificate (Caddy example):

```caddy
xdc.example.net {
  reverse_proxy 127.0.0.1:8787
}
```

   Put `coordinator.host: 127.0.0.1` behind the proxy and set workers' `coordinator_url` to `https://xdc.example.net`. Set `trust_proxy_headers: true` only if you know the proxy strips untrusted forwarded headers.
3. **Private overlay**: Tailscale/WireGuard between devices, coordinator bound to the overlay IP. This is simpler and avoids exposing anything publicly.
4. **frp/Cloudflare Tunnel**: terminate TLS at the tunnel; set the public URL accordingly. Worker WebSockets must be supported by the tunnel (traffic is HTTP + WS).

The coordinator does not implement DHT hole punching. Its relay is the coordinator itself. If the coordinator is unreachable, workers retry with backoff but cannot train.

## 8. Autostart

### macOS launchd

Create `~/Library/LaunchAgents/ai.xiatian.xdc-coordinator.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>ai.xiatian.xdc-coordinator</string>
  <key>ProgramArguments</key><array>
    <string>/Users/YOU/xdc/.venv/bin/xdc</string>
    <string>coordinator</string><string>--config</string>
    <string>/Users/YOU/xdc-demo/coordinator.yaml</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/Users/YOU/Library/Logs/xdc-coordinator.log</string>
  <key>StandardErrorPath</key><string>/Users/YOU/Library/Logs/xdc-coordinator.log</string>
</dict></plist>
```

```bash
launchctl load ~/Library/LaunchAgents/ai.xiatian.xdc-coordinator.plist
```

### Windows Task Scheduler

```powershell
schtasks /Create /TN "XDC Coordinator" /SC ONSTART /RL HIGHEST /TR "C:\xdc\.venv\Scripts\xdc.exe coordinator --config C:\xdc-demo\coordinator.yaml" /F
```

Run it under a dedicated non-admin account where possible. Use `--log-file` for durable logs.

## 9. Operations

- Health: `curl https://host:8787/healthz`
- Prometheus text: `/metrics`
- Dashboard: `/dashboard`
- Logs: stdout/stderr by default; `--log-file path` adds a file handler.
- Data: coordinator `data_dir` (state, datasets, jobs, checkpoints); worker `data_dir` (identity, caches, status).
- Backup: copy `data_dir` while the coordinator is stopped, or at least `state.json` + `jobs/`.
- Storage growth: checkpoints are pruned to `checkpoint_keep`; update files and caches are not automatically pruned in v0.1.
- Node removal: ban the node (`POST /v1/admin/nodes/{id}/ban?banned=true`); it cannot fetch/update. Delete its data on the device separately.
- Reputation reset: `POST /v1/admin/nodes/{id}/reputation?value=0.8` (useful after repairing a device).

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `HTTP 502` with local httpx | macOS system proxy. XDC disables `trust_env`; check proxy settings if using a custom client. |
| `tls: hostname mismatch` | Certificate SAN does not include the address. Use the same hostname/IP used for `coordinator_url`, or regenerate TLS. |
| Worker registers then goes offline | Heartbeat timeout; check clock skew (<120s), firewall, and logs. |
| Job stays `waiting` | Fewer than `min_workers` online. Run `xdc status`; start more workers or lower `min_workers`. |
| Inference says no active tunnels | Worker heartbeat is fresh but its tunnel is down. Check worker logs / reload it; the coordinator uses the actual tunnel state. |
| MPS is slow for the demo | Expected. `auto` chooses CPU below 50M parameters; use `--backend cpu` explicitly or raise `mps_cpu_threshold_params` only after benchmarking. |
| Native extension unavailable | No compiler or build failure. `xdc doctor --json` shows the fallback. Core functionality remains. |
| Upload rejected `round is not active` | Worker missed the round deadline or a newer round started. It will poll for the next assignment. |
| Worker identity duplicated | Never run two worker processes with the same `data_dir`; they share the same node key and will steal each other's tunnel. Use separate data dirs. |

## 11. Upgrade

Stop the coordinator, back up `data_dir`, update the source, reinstall into the venv, restart, then restart workers. State schema is versioned; check release notes before downgrading. Checkpoints are standard safetensors and can be inspected with `safetensors`.
