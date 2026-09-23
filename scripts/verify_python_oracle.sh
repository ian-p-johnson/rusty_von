#!/usr/bin/env bash
# One-command verification of the Python oracle (PORTING_RUST.md Stage 0).
#
# Starts the reference server (CPU-pinned, test key), replays the golden
# corpus against it byte-exactly, and runs the pytest suite in both modes
# (in-process and pointed at the server). Exit 0 = oracle fully verified.
#
# Usage: scripts/verify_python_oracle.sh [port]
set -euo pipefail

PORT="${1:-8001}"
URL="http://127.0.0.1:${PORT}"
cd "$(dirname "$0")/.."

echo "== starting reference server on ${URL} (cpu, auth keyed) =="
VON_DEVICE=cpu VON_API_KEY=golden-test-key \
    uv run uvicorn von.server:app --host 127.0.0.1 --port "$PORT" \
    > /tmp/von_oracle_server.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
    curl -sf -m 2 "${URL}/health" > /dev/null 2>&1 && break
    sleep 2
done
curl -sf "${URL}/health" > /dev/null || { echo "server failed to start"; exit 1; }

echo "== golden replay (byte-exact) =="
uv run python scripts/replay_golden.py "$URL" --auth golden-test-key

echo "== pytest against the server (remote mode) =="
VON_TEST_BASE_URL="$URL" VON_API_KEY=golden-test-key uv run pytest -q

echo "== pytest in-process (default mode) =="
uv run pytest -q

echo "== oracle verified =="
