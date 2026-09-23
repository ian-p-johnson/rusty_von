"""Pre-softmax logits probe for the Option-Marker backend (the numerics stethoscope).

The public API only exposes rounded probabilities, which is too coarse to verify
a reimplementation against: two engines can agree on every rounded field while
disagreeing in the logits underneath. This module opens the backend's internals
-- the same pack/forward/temperature sequence the backend itself runs -- and
reports the pre-softmax quantities:

- raw per-option logits (fp32, exact)
- the effective temperature the calibration machinery selects for the request
- for zero-shot Noul: the context-free (null-state) logits and the debias
  correction, because the corrected logits are what actually gets softmaxed
- probabilities before API rounding

Standalone CLI:

    uv run python benchmarks/dump_logits.py \
        --state "The server is down" --instructions "Is this urgent?" --noul

Used programmatically by benchmarks/capture_golden.py; every number here must
be reproduced exactly (up to the logit gate) by any Rust port.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

# Pin the oracle to CPU fp32 before anything imports torch/von: the golden
# corpus is a CPU reference regardless of what accelerators the host has.
os_device = "cpu"

import torch  # noqa: E402
from von.backends.option_marker_backend import OptionMarkerBackend  # noqa: E402
from von.types import Choice, Noul, Score  # noqa: E402


def get_cpu_backend(checkpoint_dir: Optional[str] = None) -> OptionMarkerBackend:
    """The shared CPU-pinned backend used for all probing and capture."""
    return OptionMarkerBackend(checkpoint_dir=checkpoint_dir, device=os_device)


def _forward(backend: OptionMarkerBackend, packed_text: str) -> torch.Tensor:
    """One forward pass; returns the (K,) logit vector for the mask positions."""
    model = backend._get_model()
    tok = model.tokenizer
    inputs = tok(packed_text, return_tensors="pt").to(backend.device)
    pos_list = (inputs["input_ids"][0] == model.mask_token_id).nonzero(as_tuple=True)[0].tolist()
    with torch.no_grad():
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            mask_positions=[pos_list],
        )[0]


def probe_choice(
    backend: OptionMarkerBackend,
    state_text: str,
    q: Choice,
    temperature_override: Optional[float] = None,
) -> Dict[str, Any]:
    options = list(q.criteria.keys())
    descriptions = [q.criteria.get(o).strip() if q.criteria.get(o) else o.strip() for o in options]
    packed = backend._get_model().pack_sequence(state_text, q.instructions, descriptions)
    logits = _forward(backend, packed)
    eff_temp = backend._effective_temperature(
        logits, state_text, len(options), backend._get_model().tokenizer, temperature_override
    )
    probs = torch.softmax(logits / max(eff_temp, 1e-4), dim=-1).tolist()
    return {
        "kind": "choice",
        "packed_text": packed,
        "logits": logits.tolist(),
        "effective_temperature": eff_temp,
        "temperature_override": temperature_override,
        "probs_unrounded": probs,
    }


def probe_noul(
    backend: OptionMarkerBackend,
    state_text: str,
    q: Noul,
    temperature_override: Optional[float] = None,
) -> Dict[str, Any]:
    crit = q.criteria or {}
    pos_desc = crit.get("true") or "Yes, condition holds true."
    neg_desc = crit.get("false") or "No, condition is false."
    descriptions = [pos_desc, neg_desc]
    model = backend._get_model()
    packed = model.pack_sequence(state_text, q.instructions, descriptions)
    logits = _forward(backend, packed)

    out: Dict[str, Any] = {
        "kind": "noul",
        "packed_text": packed,
        "has_explicit_criteria": bool(crit.get("true") or crit.get("false")),
        "raw_logits": logits.tolist(),
        "temperature_override": temperature_override,
    }

    # Zero-shot path: the backend cancels the intrinsic polarity prior with a
    # second forward pass on an empty state, then corrects logit[0].
    if not (crit.get("true") or crit.get("false")):
        null_packed = model.pack_sequence("", q.instructions, descriptions)
        null_logits = _forward(backend, null_packed)
        bias = (null_logits[0] - null_logits[1]).item()
        if backend._noul_prior is not None:
            correction = backend._noul_prior["a"] * bias + backend._noul_prior["b"]
            prior_used = dict(backend._noul_prior)
        else:
            correction = 0.7 * bias
            prior_used = None
        corrected = torch.stack([logits[0] - correction, logits[1]])
        eff_temp = backend._effective_temperature(
            corrected, state_text, 2, model.tokenizer, temperature_override
        )
        probs = torch.softmax(corrected / max(eff_temp, 1e-4), dim=-1).tolist()
        out.update(
            {
                "null_packed_text": null_packed,
                "null_logits": null_logits.tolist(),
                "polarity_bias": bias,
                "noul_prior": prior_used,
                "correction": correction,
                "corrected_logits": corrected.tolist(),
                "effective_temperature": eff_temp,
                "probs_unrounded": probs,
            }
        )
    else:
        eff_temp = backend._effective_temperature(
            logits, state_text, 2, model.tokenizer, temperature_override
        )
        probs = torch.softmax(logits / max(eff_temp, 1e-4), dim=-1).tolist()
        out.update({"effective_temperature": eff_temp, "probs_unrounded": probs})

    return out


def probe_score(
    backend: OptionMarkerBackend,
    state_text: str,
    q: Score,
    temperature_override: Optional[float] = None,
) -> Dict[str, Any]:
    descriptions: List[str] = []
    for item in q.criteria:
        if isinstance(item, dict):
            what = item.get("what", "")
            examples = item.get("examples", [])
            ex_str = f" Examples: {', '.join(examples)}" if examples else ""
            descriptions.append(f"{what}{ex_str}".strip())
        else:
            descriptions.append(str(item).strip())
    packed = backend._get_model().pack_sequence(state_text, q.instructions, descriptions)
    logits = _forward(backend, packed)
    eff_temp = backend._effective_temperature(
        logits, state_text, len(descriptions), backend._get_model().tokenizer, temperature_override
    )
    probs = torch.softmax(logits / max(eff_temp, 1e-4), dim=-1).tolist()
    return {
        "kind": "score",
        "packed_text": packed,
        "logits": logits.tolist(),
        "effective_temperature": eff_temp,
        "temperature_override": temperature_override,
        "probs_unrounded": probs,
    }


def probe_question(
    backend: OptionMarkerBackend,
    state_text: str,
    question: Any,
    temperature_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Dispatch on question type; this is the per-question oracle readout."""
    if isinstance(question, Choice):
        return probe_choice(backend, state_text, question, temperature_override)
    if isinstance(question, Noul):
        return probe_noul(backend, state_text, question, temperature_override)
    if isinstance(question, Score):
        return probe_score(backend, state_text, question, temperature_override)
    raise TypeError(f"probe_question: unknown question type {type(question)!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--instructions", required=True)
    parser.add_argument("--noul", action="store_true", help="Treat the question as Noul.")
    parser.add_argument(
        "--levels",
        default=None,
        help="With --score mode: pipe-separated level descriptions, e.g. 'Low|Medium|High'.",
    )
    parser.add_argument("--score", action="store_true", help="Treat the question as Score (requires --levels).")
    parser.add_argument("--temperature", type=float, default=None, help="Explicit temperature override.")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    backend = get_cpu_backend(args.checkpoint_dir)
    q: Any
    if args.noul:
        q = Noul(instructions=args.instructions)
    elif args.score:
        if not args.levels:
            parser.error("--score requires --levels")
        q = Score(instructions=args.instructions, criteria=[s.strip() for s in args.levels.split("|")])
    else:
        q = Choice(instructions=args.instructions, criteria={"option_a": None, "option_b": None})

    print(json.dumps(probe_question(backend, args.state, q, args.temperature), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
