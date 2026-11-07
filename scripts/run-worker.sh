#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-./xdc-demo/worker1.yaml}"
exec xdc worker --config "$CONFIG"
