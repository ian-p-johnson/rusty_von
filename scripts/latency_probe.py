"""Latency probe (Stage 4): p50/p95/p99 per request-size class against a von
endpoint. Run against BOTH runtimes on the same device and compare — never
compare across devices. Timing runs need an otherwise-idle machine (see
PORTING_RUST.md 7.1.5): no other GPU/CPU-heavy jobs while measuring.

Usage:
    uv run python scripts/latency_probe.py --url http://127.0.0.1:8003 \
        --api-key KEY [--n 60] [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import httpx

SIZES = {
    "small": "Customer requests refund for duplicate charge on invoice #100",
    "medium": ("billing invoice refund charge server down feature request " * 25),  # ~1.5k chars
    "large": "word " * 6000,  # ~6k tokens: the long-context class
}


def probe(client: httpx.Client, url: str, api_key: str | None, state: str, n: int) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": "von-latest",
        "state": state,
        "questions": {"q": {"type": "noul", "instructions": "Is this a billing issue?",
                            "criteria": {"true": "billing", "false": "not billing"}}},
    }
    # warmup (engine alloc, cudnn heuristics, http pool)
    for _ in range(5):
        client.post(f"{url.rstrip('/')}/v1/systemone", json=payload, headers=headers)
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post(f"{url.rstrip('/')}/v1/systemone", json=payload, headers=headers)
        samples.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
    samples.sort()

    def pct(p: float) -> float:
        idx = min(len(samples) - 1, int(round(p / 100 * (len(samples) - 1))))
        return samples[idx]

    return {"n": n, "mean_ms": round(statistics.mean(samples), 2),
            "p50_ms": round(pct(50), 2), "p95_ms": round(pct(95), 2),
            "p99_ms": round(pct(99), 2), "min_ms": round(samples[0], 2),
            "max_ms": round(samples[-1], 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--n", type=int, default=60, help="measured requests per size")
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--label", default=None, help="label for the JSON record")
    args = parser.parse_args()

    results = {"url": args.url, "label": args.label}
    with httpx.Client(timeout=600.0) as client:
        for name, state in SIZES.items():
            results[name] = probe(client, args.url, args.api_key, state, args.n)
            r = results[name]
            print(f"{name:7} n={r['n']}  mean={r['mean_ms']:8.2f}ms  "
                  f"p50={r['p50_ms']:8.2f}  p95={r['p95_ms']:8.2f}  p99={r['p99_ms']:8.2f}")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        print(f"results written to {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
