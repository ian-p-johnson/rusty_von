"""Replay the golden corpus against any von server implementing /v1/systemone.

The verification workhorse of the Rust port: every golden request is sent to the
target, and the response is compared to the captured oracle bytes -- status code
and exact body text. Byte-exact comparison is possible because the API rounds
its probability fields to fixed decimals and both Python's json and serde_json
serialize equal f64s to equal text.

Usage:
    # Against a live server (Python reference, trivially green):
    uv run python scripts/replay_golden.py http://localhost:8001

    # Include the auth-contract cases (server must be started with the same key):
    uv run python scripts/replay_golden.py http://localhost:8002 --auth golden-test-key

Exit code 0 = full parity; 1 = any divergence; 2 = usage/goldens missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDENS = REPO_ROOT / "goldens"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_divergence(expected: str, actual: str) -> str:
    """Human-readable pointer to the first byte where two JSON texts differ."""
    n = min(len(expected), len(actual))
    for i in range(n):
        if expected[i] != actual[i]:
            start = max(0, i - 60)
            return (
                f"first diff at byte {i}:\n"
                f"  expected ...{expected[start:i + 60]!r}\n"
                f"  actual   ...{actual[start:i + 60]!r}"
            )
    if len(expected) != len(actual):
        longer, which = (expected, "expected") if len(expected) > len(actual) else (actual, "actual")
        return f"common prefix of {n} bytes; {which} has {abs(len(expected) - len(actual))} extra bytes"
    return "texts identical"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("--goldens-dir", default=str(GOLDENS))
    parser.add_argument("--auth", default=None,
                        help="API key the target was started with; runs the auth-contract cases")
    args = parser.parse_args()

    goldens = Path(args.goldens_dir)
    try:
        requests = load_jsonl(goldens / "requests.jsonl")
        responses = {r["id"]: r for r in load_jsonl(goldens / "responses.jsonl")}
    except FileNotFoundError as exc:
        print(f"error: goldens not found under {goldens} ({exc}); run benchmarks/capture_golden.py first",
              file=sys.stderr)
        return 2

    failures: List[str] = []
    ran = skipped = 0
    base = args.base_url.rstrip("/")

    with httpx.Client(timeout=120.0) as client:
        for req in requests:
            cid = req["id"]
            expected = responses[cid]

            # Auth-contract cases were captured against a server started with
            # VON_API_KEY=golden-test-key. They only run when --auth confirms
            # the target is configured the same way.
            if req.get("requires_auth") and not args.auth:
                skipped += 1
                continue

            headers = {}
            if req.get("requires_auth"):
                # Auth-contract cases: send exactly what was recorded (absent,
                # wrong, or correct header) -- the 401s ARE the test.
                if req.get("auth") is not None:
                    headers["Authorization"] = req["auth"]
            elif args.auth:
                # Normal cases were captured against a keyless server; when the
                # target enforces a key, inject it so the request reaches the
                # same evaluation path that produced the recorded body.
                headers["Authorization"] = f"Bearer {args.auth}"

            try:
                if req["method"] == "GET":
                    resp = client.get(f"{base}{req['path']}", headers=headers)
                else:
                    headers["Content-Type"] = "application/json"
                    resp = client.post(f"{base}{req['path']}", content=req["body"], headers=headers)
            except httpx.HTTPError as exc:
                failures.append(f"{cid}: request failed: {exc}")
                ran += 1
                continue

            ran += 1
            problems = []
            if resp.status_code != expected["status"]:
                problems.append(f"status {resp.status_code} != expected {expected['status']}")
            if resp.text != expected["body"]:
                problems.append(first_divergence(expected["body"], resp.text))
            if problems:
                failures.append(f"{cid}: " + "\n  ".join(problems))

    print(f"replayed {ran} cases against {base}: {ran - len(failures)} exact, {len(failures)} divergent"
          + (f", {skipped} skipped (auth; pass --auth to include)" if skipped else ""))
    for f in failures:
        print(f"FAIL {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
