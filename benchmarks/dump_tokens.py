"""Emit tokenizer fixtures for the Rust port (von-rs/fixtures/tokens.json).

The Rust engine must reproduce the Python tokenizer's token IDs exactly, and
the calibration feature `log_tokens` needs real `encode(state,
add_special_tokens=False)` counts. Special-token IDs are pinned from the
Python AutoTokenizer rather than discovered by alias probing (the mmBERT
alias trap in PORTING_RUST.md §4).

For every probe row in goldens/logits.jsonl this script records the encoding
of the exact packed text used at capture (plus the null-state packing for
zero-shot Noul), and the add_special_tokens=False state-token count for the
same request. Run: .venv/bin/python benchmarks/dump_tokens.py
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from benchmarks.capture_golden import SYNTHETIC_NOUL_PRIOR, _format_state, build_cases  # noqa: E402
from benchmarks.dump_logits import get_cpu_backend  # noqa: E402
from von.types import Choice, Noul, Score  # noqa: E402

OUT = ROOT / "von-rs" / "fixtures" / "tokens.json"


def main() -> int:
    tok = AutoTokenizer.from_pretrained("wfzyx/von")
    backend = get_cpu_backend()  # loads via the same HF snapshot resolution

    tokenizer_meta = {
        "class": type(tok).__name__,
        "mask_token_id": tok.mask_token_id,
        "sep_token_id": tok.sep_token_id,
        "cls_token_id": tok.cls_token_id,
        "pad_token_id": tok.pad_token_id,
        "vocab_size": tok.vocab_size,
    }
    tok_json = Path(tok.init_kwargs.get("name_or_path", "wfzyx/von"))
    snapshot_dirs = sorted(
        (Path.home() / ".cache/huggingface/hub/models--wfzyx--von/snapshots").glob("*")
    )
    revision = snapshot_dirs[-1].name if snapshot_dirs else None
    if snapshot_dirs:
        tok_path = snapshot_dirs[-1] / "tokenizer.json"
        h = hashlib.sha256()
        with open(tok_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        tokenizer_meta["tokenizer_json_sha256"] = h.hexdigest()
    tokenizer_meta["checkpoint_revision"] = revision

    # Per-probe state texts, re-derived exactly the way capture_golden.py did:
    # every HTTP case that produced a logits.jsonl row, plus the probe-only rows.
    state_by_probe: dict = {}
    for case in build_cases():
        cid = case["id"]
        if not (case["method"] == "POST" and case.get("body")):
            continue
        try:
            payload = json.loads(case["body"])
            state_text = _format_state(payload.get("state"))
            for q_id, q_data in (payload.get("questions") or {}).items():
                q_type = q_data.get("type", "choice")
                if q_type == "choice":
                    if not q_data.get("criteria"):
                        continue
                    q_obj = Choice(**q_data)
                elif q_type == "noul":
                    q_obj = Noul(**q_data)
                elif q_type == "score":
                    if not q_data.get("criteria"):
                        continue
                    q_obj = Score(**q_data)
                else:
                    continue
                state_by_probe[f"{cid}/{q_id}"] = state_text
        except Exception:
            continue
    state_by_probe["probe_only/temp_override"] = "Invoice charged twice, need it reversed."
    state_by_probe["probe_only/prior_positive"] = (
        "Payment gateway reports timeout on charge authorizations. Urgent."
    )
    state_by_probe["probe_only/prior_negative"] = "All systems nominal."

    encodings = []
    state_counts = []
    with open(ROOT / "goldens" / "logits.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pid = row["id"]

            def encode(text: str) -> dict:
                enc = tok(text, return_tensors=None)
                ids = enc["input_ids"]
                return {
                    "text": text,
                    "input_ids": ids,
                    "mask_positions": [i for i, t in enumerate(ids) if t == tok.mask_token_id],
                }

            encodings.append({"id": pid, **encode(row["packed_text"])})
            if row.get("null_packed_text") is not None:
                encodings.append({"id": f"{pid}#null", **encode(row["null_packed_text"])})

            state = state_by_probe.get(pid)
            if state is not None:
                state_counts.append({
                    "id": pid,
                    "state": state,
                    "count": len(tok.encode(state, add_special_tokens=False)),
                })

    doc = {
        "synthetic_noul_prior": SYNTHETIC_NOUL_PRIOR,
        "tokenizer": tokenizer_meta,
        "encodings": encodings,
        "state_token_counts": state_counts,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(
        f"wrote {OUT.relative_to(ROOT)}: {len(encodings)} encodings, "
        f"{len(state_counts)} state counts, revision {revision}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
