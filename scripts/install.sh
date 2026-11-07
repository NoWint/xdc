#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3 is required. Install Python 3.10+ (python.org, Homebrew, or your package manager)." >&2
  exit 1
fi
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("XDC requires Python 3.10 or newer; found %s" % sys.version.split()[0])
PY

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -e ".[dev]"
python - <<'PY'
try:
    from xdc import native
    print("native:", native.VERSION)
except Exception as exc:
    print("native: Python fallback (%s)" % exc)
PY
xdc doctor
cat <<'EOF'

Installed. Next:
  xdc init --dir ./xdc-demo --host 0.0.0.0
  xdc coordinator --config ./xdc-demo/coordinator.yaml
  xdc worker --config ./xdc-demo/worker1.yaml
EOF
