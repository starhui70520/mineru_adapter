#!/usr/bin/env bash
set -euo pipefail

HOST_ADDRESS="${HOST_ADDRESS:-0.0.0.0}"
PORT="${PORT:-18000}"
export UPSTREAM_BASE_URL="${UPSTREAM_BASE_URL:-http://127.0.0.1:8000}"
export UPSTREAM_MODEL="${UPSTREAM_MODEL:-vl-model}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
export ADAPTER_DEBUG_DIR="${ADAPTER_DEBUG_DIR:-$(pwd)/debug}"

mkdir -p "$ADAPTER_DEBUG_DIR"

exec "$(pwd)/.venv/bin/python" -m uvicorn mineru_adapter.api:app --host "$HOST_ADDRESS" --port "$PORT"
