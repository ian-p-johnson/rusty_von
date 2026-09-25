"""Per-device parity: run the jabr v1 corpus against TWO von endpoints and
compare their predictions case by case (PORTING_RUST.md Stage 4 gate).

The plan's device gates: compare GPU-Rust vs GPU-Python on the SAME device —
never GPU vs CPU (numerics differ across devices by design; Python itself
moves between fp16/bf16/fp32 kernel paths). Gates (from the laya-derived
risk table): argmax agreement >= 99.5% and probability MAE <= 5e-3.

Usage:
    uv run python scripts/parity_devices.py \
        --a http://127.0.0.1:8004 --b http://127.0.0.1:8003 \
        [--api-key KEY] [--json PATH]

Exit 0 = gates green; 1 = red.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.jabr_cases import ALL_TASK_FUNCTIONS  # noqa: E402


def evaluate(client: httpx.Client, base: str, api_key: str | None, task) -> list[dict]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    preds = []
    for c in task.cases:
        q = task.question
        payload_q = q if isinstance(q, dict) else q.model_dump()
        payload = {
            "model": "von-latest",
            "state": c.state,
            "questions": {"q": payload_q},
        }
        r = client.post(f"{base.rstrip('/')}/v1/systemone", json=payload, headers=headers)
        r.raise_for_status()
        preds.append(r.json()["answers"]["q"])
    return preds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="endpoint A (e.g. Python on device)")
    parser.add_argument("--b", required=True, help="endpoint B (e.g. Rust on same device)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    agree = 0
    total = 0
    abs_errs: list[float] = []
    mismatches: list[str] = []
    per_task = []

    with httpx.Client(timeout=300.0) as ca, httpx.Client(timeout=300.0) as cb:
        for fn in ALL_TASK_FUNCTIONS:
            task = fn()
            pa = evaluate(ca, args.a, args.api_key, task)
            pb = evaluate(cb, args.b, args.api_key, task)
            t_agree = 0
            for i, (x, y) in enumerate(zip(pa, pb)):
                total += 1
                if task.type == "choice":
                    ok = x["choice"] == y["choice"]
                    prob_err = max(
                        abs(x["probabilities"].get(k, 0.0) - y["probabilities"].get(k, 0.0))
                        for k in set(x["probabilities"]) | set(y["probabilities"])
                    ) if x["probabilities"] else 0.0
                elif task.type == "noul":
                    ok = (x["noul"] >= 0.5) == (y["noul"] >= 0.5)
                    prob_err = abs(x["noul"] - y["noul"])
                else:  # score
                    ok = max(x["probabilities"], key=lambda k: x["probabilities"][k]) == max(
                        y["probabilities"], key=lambda k: y["probabilities"][k]
                    )
                    prob_err = max(
                        abs(x["probabilities"].get(k, 0.0) - y["probabilities"].get(k, 0.0))
                        for k in set(x["probabilities"]) | set(y["probabilities"])
                    ) if x["probabilities"] else 0.0
                agree += int(ok)
                t_agree += int(ok)
                abs_errs.append(prob_err)
                if not ok:
                    mismatches.append(f"{task.id}#{i}: argmax differs")
            per_task.append({"id": task.id, "type": task.type, "total": len(task.cases),
                             "agree": t_agree})

    agreement = agree / total if total else 1.0
    mae = sum(abs_errs) / len(abs_errs) if abs_errs else 0.0
    max_err = max(abs_errs, default=0.0)
    print(f"cases: {total}  argmax agreement: {agreement:.4f} (gate >= 0.995)")
    print(f"probability MAE: {mae:.6f} (gate <= 0.005)  max: {max_err:.6f}")
    for m in mismatches[:20]:
        print(f"  MISMATCH {m}")
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump({"agreement": agreement, "mae": mae, "max_error": max_err,
                       "cases": total, "per_task": per_task}, f, indent=2)
            f.write("\n")
        print(f"results written to {args.json_path}")
    green = agreement >= 0.995 and mae <= 0.005
    print("DEVICE GATE: GREEN" if green else "DEVICE GATE: RED")
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
