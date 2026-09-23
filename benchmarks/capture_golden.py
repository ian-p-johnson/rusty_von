"""Golden corpus capture from the running Python von (Stage 0 oracle baseline).

Captures, from the real weights pinned to CPU fp32:

  goldens/requests.jsonl    {id, method, path, body, auth?}  - exact request text
  goldens/responses.jsonl   {id, status, body}              - exact response text (byte-level)
  goldens/logits.jsonl      {id, ...probe readout}          - pre-softmax logits + eff temperature
  goldens/api_errors.jsonl  {id, call, error_type, message} - function-level error contracts
  goldens/manifest.json     capture metadata, checkpoint revision + file hashes

The response bodies are the literal HTTP text the server emits (captured via
TestClient), so replay comparison is an exact string diff -- the strongest wire
gate available, and trivially implementable in any language.

Coverage matrix (PORTING_RUST.md Stage 0): choice K=0,1,2,3,5,8,25; descriptions
present/absent; unicode; literal [MASK] in text; 8k-class state; noul with
explicit criteria / one-sided / zero-shot / synthetic fitted prior; score 0,2,3,5,10
levels in str and dict shapes; state as str/dict (nested list/float/bool/None/dict)/
list/int/float/bool/null/empty; fan-out ordering; model-field variations; legacy
pos_criteria folding; auth 401s; validation and eval 422s.

Deterministic by construction. Verify: run twice, diff the goldens/ tree.

Usage:
    uv run python benchmarks/capture_golden.py [--goldens-dir goldens]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Pin the oracle to CPU fp32 before anything imports torch/von: the goldens are
# a CPU reference regardless of what the host machine has.
os.environ["VON_DEVICE"] = "cpu"

from fastapi.testclient import TestClient  # noqa: E402

import von  # noqa: E402
from von.backends.option_marker_backend import _format_state  # noqa: E402
from von.server import app  # noqa: E402
from von.types import Choice, Noul, Score  # noqa: E402

from benchmarks.dump_logits import get_cpu_backend, probe_question  # noqa: E402


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

def _long_state(target_words: int = 2200) -> str:
    """Deterministic pseudo-text long enough to pack into several thousand tokens."""
    vocab = [
        "cluster", "latency", "replica", "quota", "billing", "invoice", "kernel",
        "throughput", "cache", "shard", "token", "refund", "webhook", "queue",
        "threshold", "partition", "replica", "signal", "contract", "severance",
    ]
    words = []
    i = 0
    while len(words) < target_words:
        words.append(f"{vocab[i % len(vocab)]}{i % 97}")
        i += 1
    lines = []
    for start in range(0, len(words), 12):
        lines.append(" ".join(words[start : start + 12]))
    return "\n".join(lines)


def _choice_payload(
    instructions: str,
    criteria: Dict[str, Optional[str]],
    model: str = "von-latest",
    state: Any = "Customer requests refund for duplicate charge on invoice #100",
) -> Dict[str, Any]:
    return {
        "model": model,
        "state": state,
        "questions": {"decision": {"type": "choice", "instructions": instructions, "criteria": criteria}},
    }


def _noul_payload(
    instructions: str,
    criteria: Optional[Dict[str, str]] = None,
    state: Any = "Connection pool exhausted on port 5432; handshakes timing out.",
    model: str = "von-latest",
) -> Dict[str, Any]:
    q: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria is not None:
        q["criteria"] = criteria
    return {"model": model, "state": state, "questions": {"judgment": q}}


def _score_payload(
    instructions: str,
    criteria: List[Any],
    state: Any = "Memory utilization 98% with frequent OOM killer invocations.",
    model: str = "von-latest",
) -> Dict[str, Any]:
    return {
        "model": model,
        "state": state,
        "questions": {"rating": {"type": "score", "instructions": instructions, "criteria": criteria}},
    }


def build_cases() -> List[Dict[str, Any]]:
    """Ordered corpus. Each case: id, method, path, body (exact text). Probe
    questions are derived generically from every 200 payload at capture time."""
    cases: List[Dict[str, Any]] = []

    def add(cid: str, payload: Optional[Dict[str, Any]], **extra: Any) -> None:
        raw_body = extra.pop("raw_body", None)
        case: Dict[str, Any] = {
            "id": cid,
            "method": "POST",
            "path": "/v1/systemone",
            "body": raw_body if raw_body is not None else (json.dumps(payload) if payload is not None else None),
        }
        case.update(extra)
        cases.append(case)

    # ---- Choice: option-count ladder -------------------------------------
    add("choice_k1", _choice_payload("Which one?", {"only": "The single possible answer"}))
    add("choice_k2", _choice_payload(
        "Classify the root cause domain of this incident.",
        {"billing": "Invoices, billing, duplicate charges, refunds",
         "technical": "Software bugs and technical issues"},
    ))
    add("choice_k2_no_desc", _choice_payload("Pick one.", {"alpha": None, "beta": None}))
    add("choice_k3_mixed_desc", _choice_payload(
        "Which queue should handle this request?",
        {"account": "Account access and authentication support.",
         "billing": None,
         "sales": "Sales and product evaluation."},
    ))
    add("choice_k5", _choice_payload(
        "What is the primary customer intent in the message?",
        {"refund": "Requesting money back",
         "technical_help": "Reporting a bug or API error",
         "billing_question": "Questions about invoices or plans",
         "cancellation": "Requesting account closure",
         "general_info": "Documentation or pricing questions"},
    ))
    add("choice_k8", _choice_payload(
        "Which department owns this?",
        {name: f"Handles all {name} matters" for name in
         ["billing", "shipping", "returns", "sales", "security", "legal", "partnerships", "other"]},
    ))
    add("choice_k25", _choice_payload(
        "Which city does this support ticket refer to?",
        {f"city_{i:02d}": f"Requests originating from city number {i}" for i in range(1, 26)},
    ))
    add("choice_k0_empty", _choice_payload("Nothing to choose.", {}))

    # ---- Choice: text shapes ---------------------------------------------
    add("choice_unicode", _choice_payload(
        "この問い合わせの言語は？",
        {"japanese": "日本語の問い合わせ", "chinese": "中文询问 🀄", "korean": "한국어 문의"},
        state="顧客からの問い合わせ：返金してください 🙏 — s'il vous plaît",
    ))
    add("choice_mask_literal", _choice_payload(
        "Classify the log line.",
        {"error": "An error message", "info": "An informational message"},
        state="log: [MASK] the operation completed [MASK] successfully",
    ))
    add("choice_empty_instructions", _choice_payload(
        "",
        {"yes_path": "The state suggests yes", "no_path": "The state suggests no"},
    ))
    add("choice_multiline_state", _choice_payload(
        "Which system is affected?",
        {"database": "Database systems", "network": "Network systems"},
        state="line one: primary replica lagging\nline two: 'quotes' and \"more quotes\"\nline three: 42%",
    ))
    add("choice_stale_model_field", _choice_payload(
        "Which queue?",
        {"a": "Queue A", "b": "Queue B"},
        model="von-1.0.0",
    ))
    add("choice_jev_alias_field", _choice_payload(
        "Which queue?",
        {"a": "Queue A", "b": "Queue B"},
        model="jev-latest",
    ))
    add("choice_bogus_model_field", _choice_payload(
        "Which queue?",
        {"a": "Queue A", "b": "Queue B"},
        model="totally-not-a-model",
    ))
    add("choice_long_state", _choice_payload(
        "Does this long report describe a billing issue or a technical issue?",
        {"billing": "Anything about invoices, charges, refunds, payment",
         "technical": "Anything about systems, errors, infrastructure"},
        state=_long_state(),
    ))

    # ---- Noul paths --------------------------------------------------------
    add("noul_explicit", _noul_payload(
        "Is this issue actively blocking customer operations?",
        {"true": "Blocking: outage, exhaustion, timeouts",
         "false": "Not blocking: informational, cosmetic"},
    ))
    add("noul_true_only", _noul_payload(
        "Is this data exfiltration?",
        {"true": "Bulk export or download of sensitive data"},
        state="User scheduled a full backup export of the customer table at 3am.",
    ))
    add("noul_false_only", _noul_payload(
        "Is this a security incident?",
        {"false": "Routine operational noise"},
        state="Certificate rotated successfully on schedule.",
    ))
    add("noul_zeroshot", _noul_payload(
        "Does the request require immediate SLA intervention?",
        state="Payment gateway reports timeout on charge authorizations. Urgent.",
    ))
    add("noul_zeroshot_negative", _noul_payload(
        "Is the server currently on fire?",
        state="All systems nominal. Latency within normal range for the region.",
    ))
    add("noul_legacy_criteria", {
        "model": "von-latest",
        "state": "Urgent: Payment failed on invoice 999",
        "questions": {"judgment": {"type": "noul", "instructions": "Is this urgent?",
                                   "pos_criteria": "Urgent and critical",
                                   "neg_criteria": "Routine and minor"}},
    })

    # Synthetic fitted-prior cases: capture runs with backend._noul_prior
    # injected (ids must start with "prior_" for the loop to manage it). The
    # shipped checkpoint carries no prior, so these pin the a*bias+b branch.
    add("prior_noul_positive", _noul_payload(
        "Does the request require immediate SLA intervention?",
    ))
    add("prior_noul_negative", _noul_payload(
        "Does the request require immediate SLA intervention?",
        state="All systems nominal. Latency within normal range for the region.",
    ))

    # ---- Score shapes ------------------------------------------------------
    add("score_k2", _score_payload("Rate the severity.", ["Minor", "Catastrophic"]))
    add("score_k3", _score_payload(
        "Rate where the state falls on this scale:",
        ["Cosmetic issue", "Minor slowdown", "Catastrophic outage with complete service disruption"],
    ))
    add("score_k5", _score_payload(
        "Rate the customer frustration level.",
        ["Calm and polite", "Slightly concerned", "Annoyed", "Frustrated", "Threatening legal action"],
    ))
    add("score_k10", _score_payload(
        "Rate the reading level.",
        [f"Grade {i} reading level" for i in range(1, 11)],
        state="The cat sat on the mat and looked at the bird.",
    ))
    add("score_dict_criteria", _score_payload(
        "Rate the incident severity.",
        [
            {"what": "Nominal operation", "examples": ["healthy checks", "normal traffic"]},
            {"what": "Elevated resource consumption", "examples": ["high CPU", "queue growth"]},
            {"what": "Critical threshold", "examples": ["OOM kills", "service termination"]},
        ],
    ))
    add("score_dict_criteria_int_examples", _score_payload(
        "Rate the incident severity.",
        [{"what": "Low", "examples": [1, 2]}, {"what": "High", "examples": [9, 10]}],
    ))
    add("score_empty", _score_payload("Rate the nothing.", []))
    add("score_unicode", _score_payload(
        "整理レベルを評価してください。",
        ["低い", "中程度", "高い 🔥"],
        state="システムは完全にダウンしています。",
    ))

    # ---- State shapes -------------------------------------------------------
    add("state_dict_mixed", _choice_payload(
        "Is this ticket about payments or access?",
        {"payments": "Charges, refunds, cards", "access": "Logins, SSO, permissions"},
        state={
            "ticket_id": "INC-4091",
            "customer_tier": "enterprise",
            "attempts": 3,
            "success_rate": 0.42,
            "escalated": True,
            "last_agent": None,
            "tags": ["gateway", "timeout"],
        },
    ))
    add("state_dict_nested", _choice_payload(
        "Is the configuration valid?",
        {"valid": "Configuration passes", "invalid": "Configuration has problems"},
        state={"outer": {"inner": {"deep": [1, 2.5, True, None]}}, "note": "check repr"},
    ))
    add("state_list", _choice_payload(
        "Is this list about hardware or software?",
        {"hardware": "Physical machines", "software": "Programs and services"},
        state=["postgres", "redis", "nginx"],
    ))
    add("state_int", _choice_payload(
        "Is this number even or odd?",
        {"even": "Divisible by two", "odd": "Not divisible by two"},
        state=41,
    ))
    add("state_float", _choice_payload(
        "Is this utilization high or low?",
        {"high": "Above safe threshold", "low": "Within normal range"},
        state=0.98,
    ))
    add("state_bool", _choice_payload(
        "Is this flag good or bad?",
        {"good": "Healthy state", "bad": "Unhealthy state"},
        state=False,
    ))
    add("state_null", _choice_payload(
        "Anything to act on?",
        {"act": "Action needed", "wait": "No action"},
        state=None,
    ))
    add("state_empty_string", _choice_payload(
        "Anything to act on?",
        {"act": "Action needed", "wait": "No action"},
        state="",
    ))

    # ---- Fan-out -------------------------------------------------------------
    fanout_questions = {
        "category": {"type": "choice", "instructions": "What kind of system event is this?",
                     "criteria": {"storage": "Disk or filesystem issues",
                                  "network": "DNS, latency, timeouts",
                                  "auth": "Login or permission issues"}},
        "is_blocking": {"type": "noul", "instructions": "Is this event blocking writes?"},
        "severity": {"type": "score", "instructions": "Rate the severity.",
                     "criteria": ["Informational", "Warning", "Critical"]},
        "second_noul": {"type": "noul", "instructions": "Is this a security event?",
                        "criteria": {"true": "Security compromise", "false": "Operational issue"}},
    }
    add("fanout_mixed", {
        "model": "von-latest",
        "state": {"event": "database_disk_full", "disk_free_percent": 0.01,
                  "logs": "No space left on device while writing wal segment"},
        "questions": fanout_questions,
    })

    # ---- Errors over HTTP (pinned status + body) ----------------------------
    add("err_unknown_qtype", {
        "model": "von-latest", "state": "s",
        "questions": {"q": {"type": "bogus", "instructions": "?"}},
    })
    add("err_missing_instructions", {
        "model": "von-latest", "state": "s",
        "questions": {"q": {"type": "noul"}},
    })
    add("err_choice_missing_criteria", {
        "model": "von-latest", "state": "s",
        "questions": {"q": {"type": "choice", "instructions": "?"}},
    })
    add("err_malformed_json", None, raw_body='{"model": "von-latest", "state": ')
    add("err_get_on_endpoint", None, method="GET")
    add("err_empty_questions", {
        "model": "von-latest", "state": "s", "questions": {},
    })

    # ---- Auth (401 contract; replay only with --auth) ------------------------
    for cid, headers in (("auth_no_header", None), ("auth_bad_token", "Bearer wrong-key")):
        cases.append({
            "id": cid, "method": "POST", "path": "/v1/systemone",
            "body": json.dumps(_choice_payload("Which?", {"a": "A", "b": "B"})),
            "auth": headers, "requires_auth": True,
        })
    cases.append({
        "id": "auth_ok", "method": "POST", "path": "/v1/systemone",
        "body": json.dumps(_choice_payload("Which?", {"a": "A", "b": "B"})),
        "auth": "Bearer golden-test-key", "requires_auth": True, "expect_status": 200,
    })

    # ---- Health/model-list GETs ------------------------------------------------
    for cid, path in (("get_root", "/"), ("get_health", "/health"), ("get_models", "/v1/models")):
        cases.append({"id": cid, "method": "GET", "path": path, "body": None})

    return cases


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

SYNTHETIC_NOUL_PRIOR = {"a": 0.9, "b": 0.05}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldens-dir", default="goldens")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.goldens_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import von.backends.option_marker_backend as omb

    requests_f = open(out_dir / "requests.jsonl", "w", encoding="utf-8")
    responses_f = open(out_dir / "responses.jsonl", "w", encoding="utf-8")
    logits_f = open(out_dir / "logits.jsonl", "w", encoding="utf-8")

    backend = get_cpu_backend()
    client = TestClient(app)

    cases = build_cases()
    n_probe = 0
    try:
        for case in cases:
            cid = case["id"]
            headers = {"Content-Type": "application/json"}
            prior_active = cid.startswith("prior_")

            if cid.startswith("auth_") or case.get("requires_auth"):
                os.environ["VON_API_KEY"] = "golden-test-key"
            try:
                if case["method"] == "GET":
                    resp = client.get(case["path"])
                else:
                    if case.get("auth") is not None:
                        headers["Authorization"] = case["auth"]
                    if prior_active:
                        backend._noul_prior = dict(SYNTHETIC_NOUL_PRIOR)
                    resp = client.post(case["path"], content=case["body"], headers=headers)
            finally:
                if case.get("requires_auth"):
                    os.environ.pop("VON_API_KEY", None)
                if prior_active:
                    backend._noul_prior = None

            requests_f.write(json.dumps({
                "id": cid,
                "method": case["method"],
                "path": case["path"],
                "body": case["body"],
                **({"auth": case["auth"]} if case.get("auth") is not None else {}),
                **({"requires_auth": True} if case.get("requires_auth") else {}),
            }) + "\n")
            responses_f.write(json.dumps({
                "id": cid,
                "status": resp.status_code,
                "body": resp.text,
            }) + "\n")

            # Probe the internals for every case that reached the model: derive
            # the question objects from the payload exactly the way the backend
            # does, and skip questions that short-circuit before a forward pass
            # (empty choice criteria, empty score levels).
            if resp.status_code == 200 and case["method"] == "POST" and case.get("body"):
                probe_specs: List[Tuple[str, Any]] = []
                try:
                    payload = json.loads(case["body"])
                    state_text = _format_state(payload.get("state"))
                    for q_id, q_data in (payload.get("questions") or {}).items():
                        q_type = q_data.get("type", "choice")
                        if q_type == "choice":
                            if not q_data.get("criteria"):
                                continue
                            q_obj: Any = Choice(**q_data)
                        elif q_type == "noul":
                            q_obj = Noul(**q_data)
                        elif q_type == "score":
                            if not q_data.get("criteria"):
                                continue
                            q_obj = Score(**q_data)
                        else:
                            continue
                        probe_specs.append((q_id, q_obj))
                except Exception:
                    probe_specs = []

                if probe_specs:
                    if prior_active:
                        backend._noul_prior = dict(SYNTHETIC_NOUL_PRIOR)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            for q_id, q_obj in probe_specs:
                                readout = probe_question(backend, state_text, q_obj)
                                logits_f.write(json.dumps({"id": f"{cid}/{q_id}", **readout}) + "\n")
                                n_probe += 1
                    finally:
                        if prior_active:
                            backend._noul_prior = None

        # Probe-only cases: temperature override + synthetic prior (no HTTP pair).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            override_q = Choice(instructions="Which queue?", criteria={"billing": "Billing", "tech": "Technical"})
            readout = probe_question(backend, "Invoice charged twice, need it reversed.",
                                     override_q, temperature_override=3.5)
            logits_f.write(json.dumps({"id": "probe_only/temp_override", **readout}) + "\n")
            n_probe += 1

            backend._noul_prior = dict(SYNTHETIC_NOUL_PRIOR)
            try:
                prior_q = Noul(instructions="Does the request require immediate SLA intervention?")
                readout = probe_question(
                    backend, "Payment gateway reports timeout on charge authorizations. Urgent.", prior_q)
                logits_f.write(json.dumps({"id": "probe_only/prior_positive", **readout}) + "\n")
                n_probe += 1
                readout = probe_question(backend, "All systems nominal.", prior_q)
                logits_f.write(json.dumps({"id": "probe_only/prior_negative", **readout}) + "\n")
                n_probe += 1
            finally:
                backend._noul_prior = None

    finally:
        requests_f.close()
        responses_f.close()
        logits_f.close()

    # ---- Function-level error contracts ---------------------------------
    api_errors: List[Dict[str, str]] = []
    try:
        von.decide("some state", choices=["yes", "yes"])
    except Exception as exc:
        api_errors.append({"id": "api_dup_choices", "call": "von.decide(choices=['yes','yes'])",
                           "error_type": type(exc).__name__, "message": str(exc)})
    try:
        Noul(instructions="Is it down?", criteria={"true": "Down"}, pos_criteria="Also down")
    except Exception as exc:
        api_errors.append({"id": "api_noul_conflict",
                           "call": "Noul(criteria={'true': ...}, pos_criteria=...)",
                           "error_type": type(exc).__name__, "message": str(exc)})
    try:
        von.engine.VonEngine(backend_name="option-marker")
    except Exception as exc:
        api_errors.append({"id": "api_unknown_backend", "call": "VonEngine('option-marker')",
                           "error_type": type(exc).__name__, "message": str(exc)})
    with open(out_dir / "api_errors.jsonl", "w", encoding="utf-8") as f:
        for e in api_errors:
            f.write(json.dumps(e) + "\n")

    # ---- Manifest -----------------------------------------------------------
    snapshot_dirs = list((Path.home() / ".cache/huggingface/hub/models--wfzyx--von/snapshots").glob("*"))
    revision = snapshot_dirs[0].name if snapshot_dirs else None
    hashes: Dict[str, str] = {}
    if snapshot_dirs:
        for name in ("option_marker.pt", "marker_calibration.json", "tokenizer.json", "config.json"):
            p = snapshot_dirs[0] / name
            if p.exists():
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 22), b""):
                        h.update(chunk)
                hashes[name] = h.hexdigest()

    import transformers
    manifest = {
        "captured_with": {
            "python": sys.version.split()[0],
            "torch": __import__("torch").__version__,
            "transformers": transformers.__version__,
            "von": von.__version__,
        },
        "device": "cpu",
        "device_policy": "goldens are a CPU fp32 reference; VON_DEVICE pinned at capture",
        "checkpoint_revision": revision,
        "checkpoint_sha256": hashes,
        "synthetic_noul_prior": SYNTHETIC_NOUL_PRIOR,
        "case_count": len(cases),
        "probe_count": n_probe,
        "notes": [
            "response bodies are exact HTTP text (TestClient); replay is a string diff",
            "captured weights carry calibration_map + T=2.2 and NO fitted noul prior; "
            "prior_*/probe_only cases inject the synthetic prior at backend level",
        ],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"captured {len(cases)} request cases, {n_probe} probe readouts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
