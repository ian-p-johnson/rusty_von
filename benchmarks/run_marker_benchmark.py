"""Peer benchmark runner for the Phase 3 Option-Marker Joint Model."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.jabr_cases import ALL_TASK_FUNCTIONS  # noqa: E402
from von.backends.option_marker_backend import OptionMarkerBackend  # noqa: E402

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/von-option-marker",
        help="Checkpoint directory. Pass a Hugging Face cache snapshot dir to run "
             "the shipped weights without a local checkout (e.g. for baseline "
             "recording on machines with no checkpoints/ tree).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device for the backend (default: backend auto-detection).",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Write the results as structured JSON to this path "
             "(used by the Rust-port baseline, PORTING_RUST.md Stage 0).",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("Phase 3 Option-Marker Peer Benchmark (jabr/classifier-benchmark)")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print("=" * 65)

    backend = OptionMarkerBackend(checkpoint_dir=args.checkpoint_dir, device=args.device)

    t0_suite = time.perf_counter()
    micro_correct, micro_total = 0, 0
    macro_accs = []
    per_task = []

    for fn in ALL_TASK_FUNCTIONS:
        task = fn()
        correct = 0
        total = len(task.cases)
        t0_task = time.perf_counter()

        for c in task.cases:
            if task.type == "choice":
                res = backend.evaluate_choice("c", c.state, task.question)
                pred = res.choice
                exp = c.expected
            elif task.type == "noul":
                res = backend.evaluate_noul("n", c.state, task.question)
                pred = "yes" if res.noul >= 0.5 else "no"
                exp = "yes" if c.expected else "no"
            elif task.type == "score":
                res = backend.evaluate_score("s", c.state, task.question)
                probs = res.probabilities
                pred = max(probs, key=lambda k: probs[k]) if probs else "0"
                exp = str(c.expected)
            else:
                raise ValueError(f"Unknown task type: {task.type}")

            if pred == exp:
                correct += 1

        latency_ms = (time.perf_counter() - t0_task) * 1000 / total
        acc = correct / total
        micro_correct += correct
        micro_total += total
        macro_accs.append(acc)
        per_task.append({
            "id": task.id,
            "type": task.type,
            "correct": correct,
            "total": total,
            "accuracy": round(acc, 6),
            "ms_per_case": round(latency_ms, 3),
        })
        print(f"{task.id:<24} ({task.type:<6}): {acc:.3f} ({correct}/{total}) [{latency_ms:.1f}ms/case]")

    total_time = time.perf_counter() - t0_suite
    micro_acc = micro_correct / micro_total
    macro_acc = sum(macro_accs) / len(macro_accs)

    print("-" * 65)
    print(f"Option-Marker Micro Accuracy: {micro_acc:.3f} ({micro_correct}/{micro_total})")
    print(f"Option-Marker Macro Accuracy: {macro_acc:.3f}")
    print(f"Total Wall Time: {total_time:.2f}s ({total_time*1000/micro_total:.1f}ms mean/case)")
    print("=" * 65)

    if args.json_path:
        import platform

        import torch
        import transformers
        import von
        from von.backends.option_marker_backend import VON_MODEL_ID

        record = {
            "recorded_with": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "von": von.__version__,
            },
            "model_id": VON_MODEL_ID,
            "checkpoint_dir": args.checkpoint_dir,
            "device": str(backend.device),
            "suite": "jabr v1 (benchmarks/jabr_cases.py)",
            "micro_accuracy": round(micro_acc, 6),
            "macro_accuracy": round(macro_acc, 6),
            "mean_ms_per_case": round(total_time * 1000 / micro_total, 3),
            "wall_time_s": round(total_time, 2),
            "per_task": per_task,
        }
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
        print(f"results written to {args.json_path}")

if __name__ == "__main__":
    main()
