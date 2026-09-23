#!/usr/bin/env bash
# Run the pytest suite against a running von server (Python or Rust) instead of
# the in-process engine. This is the differential harness entry point: the same
# tests, unmodified, exercise whichever implementation serves the URL.
#
# Usage: scripts/run_tests_against.sh http://localhost:8002 [pytest args...]
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <base_url> [pytest args...]" >&2
    exit 1
fi

URL="$1"
shift

cd "$(dirname "$0")/.."
VON_TEST_BASE_URL="$URL" uv run pytest "$@"
