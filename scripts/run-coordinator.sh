#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-./xdc-demo/coordinator.yaml}"
exec xdc coordinator --config "$CONFIG" --access-log
