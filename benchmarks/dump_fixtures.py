"""Emit Stage 1 pure-logic fixtures for the Rust port (von-rs/fixtures/).

Everything captured here is model-free: state formatting, pack_sequence,
rounding, usage accounting, envelope shapes, error strings, pattern logic and
the FastAPI wire behavior for requests that never reach the weights. The
Rust test suite consumes these files, so this script is the Python oracle for
Stage 1. Run: .venv/bin/python benchmarks/dump_fixtures.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "von-rs" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)


def dump(name: str, obj):
    path = OUT / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {path.relative_to(ROOT)}")


# ---------------------------------------------------------------- state format
from von.backends.option_marker_backend import _format_state

FORMAT_VALUES = [
    "plain string",
    "string with 'single' quotes",
    'string with "double" quotes',
    "string with both 'single' and \"double\"",
    "back\\slash and new\nline and tab\tchar",
    "unicode: caf\u00e9 \u4e2d\u6587 \U0001f600",
    "control\u0007bell and\u001bdel next",
    "",
    0,
    7,
    -13,
    1.5,
    0.1,
    3.0,
    -0.0,
    0.30000000000000004,
    1e16,
    1e15,
    9999999999999998.0,
    1e-4,
    1e-5,
    1.2345678901234567e-7,
    2.675,
    1e300,
    5e-324,
    True,
    False,
    None,
    [],
    ["a", "b"],
    ["mixed", 2, 3.5, True, None],
    [["nested", ["deep"]], "x"],
    ["quote' in element", 'double" element'],
    ["uni \u00e9", "\U0001f600"],
    {},
    {"a": 1},
    {"k": "v", "n": 2, "f": 1.5, "b": True, "z": None},
    {"nested": {"inner": [1, "two", None]}, "list": ["x", "y"]},
    {"empty": {}, "elist": []},
    {"uni\u00e9": "caf\u00e9"},
]

state_cases = []
for v in FORMAT_VALUES:
    state_cases.append({"input": v, "formatted": _format_state(v), "str": str(v)})
dump("state_formatting.json", {"cases": state_cases})


# --------------------------------------------------------------- pack_sequence
from von.types import Choice, Noul, Score, SystemOneResponse
from von.models.option_marker import OptionMarkerModel

pack_cases = [
    ("state text", "question?", ["a", "b"]),
    ("state text", "", ["a", "b"]),
    ("", "question?", ["a", "b"]),
    ("", "", ["a"]),
    ("state", "question?", []),
    ("  padded state  ", "  padded question  ", ["  opt a  ", "opt b", ""]),
    ("caf\u00e9 \u4e2d\u6587", "which?", ["\u00e9x", "\U0001f600"]),
    ("multi\nline\nstate", "multi\nline question", ["a\nb", "c"]),
    ("s", "q", ["one", "two", "three", "four"]),
]


def pack_like_model(state: str, question: str, options):
    mask, sep = "[MASK]", "[SEP]"
    prefix = f"{question} {state}".strip() if question else state.strip()
    opts_packed = " ".join(f"{mask} {opt.strip()}" for opt in options)
    return f"{prefix} {sep} {opts_packed}"


pack_out = []
for state, question, options in pack_cases:
    packed = pack_like_model(state, question, options)
    pack_out.append({"state": state, "question": question, "options": options, "packed": packed})
dump("pack_sequence.json", {"cases": pack_out})


# -------------------------------------------------------------------- rounding
ROUND_VALUES = [
    (0.5, 4), (0.15, 4), (0.125, 3), (2.675, 2), (1.005, 2), (8.835, 2),
    (1.05, 1), (0.00005, 4), (0.00015, 4), (2.5, 0), (3.5, 0), (-0.5, 4),
    (-2.675, 2), (0.92225, 4), (0.92235, 4), (0.1, 4), (1.0, 4), (0.0, 4),
    (1e20, 2), (1e-5, 4), (0.0001, 4), (0.12345, 4), (0.12344, 4), (0.99995, 4),
    (0.99994999999, 4), (1234.56789, 2), (12.345, 3), (0.7, 4), (-0.12345, 4),
    (0.8444999999, 3), (0.8445, 3), (0.84450000001, 3), (1.0666666666, 4),
    (0.7199999999, 4), (2.005, 2), (0.5, 3), (0.05, 1), (0.15, 1), (0.25, 1),
    (0.35, 1), (0.45, 1), (4.25, 2), (4.125, 2), (4.375, 2), (-0.00005, 4),
]

round_cases = []
for value, nd in ROUND_VALUES:
    r = round(value, nd)
    round_cases.append({"value": value, "ndigits": nd, "expected": repr(r)})
dump("rounding.json", {"cases": round_cases})


# ----------------------------------------------------------------------- usage
def usage_for(state_str: str, total_q_chars: int, n_answers: int):
    state_tokens = max(1, len(state_str) // 4)
    q_tokens = max(1, total_q_chars // 4)
    return {"input_tokens": state_tokens + q_tokens, "output_tokens": n_answers}


usage_cases = []
for state_str, qchars, n in [
    ("", 0, 1), ("", 4, 2), ("abc", 0, 1), ("abcd", 0, 1), ("abcde", 0, 1),
    ("x" * 8, len("y" * 9), 3), ("caf\u00e9 \u4e2d\u6587 \U0001f600!", 7, 2),
    ("a" * 100, len("b" * 3), 5),
]:
    usage_cases.append({"state": state_str, "q_chars": qchars, "n_answers": n, "usage": usage_for(state_str, qchars, n)})
dump("usage.json", {"cases": usage_cases})


# ------------------------------------------------------------------- envelopes
def render_fastapi(model) -> str:
    return json.dumps(model.model_dump(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


envelope_cases = []

resp = SystemOneResponse(
    model="von-1.1.0",
    answers={"decision": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}, "confidence": 1.0}},
    usage={"input_tokens": 17, "output_tokens": 1},
)
envelope_cases.append({"name": "choice", "input": resp.model_dump(), "rendered": render_fastapi(resp)})

resp = SystemOneResponse(
    model="von-1.1.0",
    answers={"j": {"type": "noul", "noul": 0.4892}},
    usage={"input_tokens": 3, "output_tokens": 1},
)
envelope_cases.append({"name": "noul", "input": resp.model_dump(), "rendered": render_fastapi(resp)})

resp = SystemOneResponse(
    model="von-1.1.0",
    answers={
        "rating": {
            "type": "score",
            "score": 1.5,
            "confidence": 0.25,
            "legend": {"0": "low", "1": "mid", "2": "high"},
            "probabilities": {"0": 0.125, "1": 0.5, "2": 0.375},
        }
    },
    usage={"input_tokens": 9, "output_tokens": 1},
)
envelope_cases.append({"name": "score", "input": resp.model_dump(), "rendered": render_fastapi(resp)})

resp = SystemOneResponse(
    model="von-1.1.0",
    answers={
        "z_first": {"type": "noul", "noul": 0.5},
        "a_second": {"type": "choice", "choice": "\u00e9x", "probabilities": {"\u00e9x": 0.3333, "y": 0.6667}, "confidence": 0.333},
        "m_third": {
            "type": "score",
            "score": 0.0,
            "confidence": 0.0,
            "legend": {"0": "only", "1": ""},
            "probabilities": {"0": 0.5, "1": 0.5},
        },
    },
    usage={"input_tokens": 12, "output_tokens": 3},
)
envelope_cases.append({"name": "fanout_order", "input": resp.model_dump(), "rendered": render_fastapi(resp)})
dump("envelopes.json", {"cases": envelope_cases})


# ------------------------------------------------------------------ api errors
api_errors = []

try:
    raise ValueError("Duplicate choices found in options list: ['yes', 'yes']")
except ValueError as exc:
    api_errors.append({"id": "duplicate_choices", "error": "ValueError", "message": str(exc)})

from von.engine import VON_CURRENT_ALIASES, VON_VERSION

try:
    raise ValueError(
        f"Unknown model 'option-marker'. "
        f"Von {VON_VERSION} is the only model; accepted aliases: "
        f"{', '.join(sorted(VON_CURRENT_ALIASES))}."
    )
except ValueError as exc:
    api_errors.append({"id": "unknown_model", "error": "ValueError", "message": str(exc)})

try:
    raise ValueError("Unknown question type 'bogus'")
except ValueError as exc:
    api_errors.append({"id": "unknown_qtype", "error": "ValueError", "message": str(exc)})

try:
    ", ".join([1, 2])
except TypeError as exc:
    api_errors.append({"id": "score_join_int", "error": "TypeError", "message": str(exc)})

try:
    ", ".join(["one", None])
except TypeError as exc:
    api_errors.append({"id": "score_join_null", "error": "TypeError", "message": str(exc)})

try:
    ", ".join(["one", True])
except TypeError as exc:
    api_errors.append({"id": "score_join_bool", "error": "TypeError", "message": str(exc)})

try:
    ", ".join([])
except Exception as exc:
    api_errors.append({"id": "score_join_empty_ok", "error": "", "message": f"empty join yields: '{', '.join([])}'"})

try:
    Choice(instructions="?", criteria="not a dict")
except Exception as exc:
    api_errors.append({"id": "choice_criteria_wrong_type", "error": type(exc).__name__, "message": str(exc)})

try:
    Choice(instructions="?", criteria={"a": 5})
except Exception as exc:
    api_errors.append({"id": "choice_criteria_int_value", "error": type(exc).__name__, "message": str(exc)})

try:
    Noul(criteria={"true": "t", "false": 3})
except Exception as exc:
    api_errors.append({"id": "noul_criteria_int_value", "error": type(exc).__name__, "message": str(exc)})

try:
    Score(instructions="?", criteria=["a", 7])
except Exception as exc:
    api_errors.append({"id": "score_criteria_int_item", "error": type(exc).__name__, "message": str(exc)})

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    n = Noul(instructions="Is it down?", pos_criteria="Also down")
    api_errors.append({"id": "noul_legacy_fold_warning", "error": "DeprecationWarning", "message": str(w[0].message)})
    api_errors.append({"id": "noul_legacy_fold_result", "error": "", "message": json.dumps(n.model_dump())})

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    n = Noul(instructions="Is it down?", pos_criteria="Down", neg_criteria="Up")
    api_errors.append({"id": "noul_legacy_fold_both_result", "error": "", "message": json.dumps(n.model_dump())})

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    n = Noul(instructions="Is it down?", pos_criteria="")
    api_errors.append({"id": "noul_legacy_fold_empty_value", "error": "", "message": json.dumps(n.model_dump())})

try:
    Noul(instructions="Is it down?", criteria={"true": "Down"}, pos_criteria="Also down", neg_criteria="Up")
except Exception as exc:
    api_errors.append({"id": "noul_legacy_conflict", "error": type(exc).__name__, "message": str(exc)})

try:
    Noul(instructions="Is it down?", criteria={"false": "F"}, neg_criteria="Up")
except Exception as exc:
    api_errors.append({"id": "noul_legacy_conflict_false", "error": type(exc).__name__, "message": str(exc)})

try:
    Noul(instructions=5)
except Exception as exc:
    api_errors.append({"id": "noul_instructions_int", "error": type(exc).__name__, "message": str(exc)})

try:
    Choice(instructions="?", criteria=5)
except Exception as exc:
    api_errors.append({"id": "choice_criteria_int", "error": type(exc).__name__, "message": str(exc)})

try:
    Choice(instructions="?", criteria=None)
except Exception as exc:
    api_errors.append({"id": "choice_criteria_null", "error": type(exc).__name__, "message": str(exc)})

try:
    Score(instructions="?", criteria={"a": 1})
except Exception as exc:
    api_errors.append({"id": "score_criteria_dict", "error": type(exc).__name__, "message": str(exc)})

try:
    Score(instructions="?", criteria=["a", {"what": "w", "examples": [1, "two"]}])
except Exception as exc:
    api_errors.append({"id": "score_examples_first_int", "error": type(exc).__name__, "message": str(exc)})

try:
    Score(instructions="?", criteria=["a", {"what": "w", "examples": ["one", None]}])
except Exception as exc:
    api_errors.append({"id": "score_examples_null_item", "error": type(exc).__name__, "message": str(exc)})

dump("api_errors.json", {"cases": api_errors})


# -------------------------------------------------------------------- patterns
from von.api import system_one
from von import patterns


class FakeClient:
    def __init__(self, responses, default=None):
        self.responses = responses
        self.default = default
        self.calls = []

    def system_one(self, state, questions, model="von-latest"):
        self.calls.append({"state": state, "questions": questions, "model": model})
        key = list(questions.keys())[0]
        data = self.responses.get(key, self.default)
        if data is None:
            raise KeyError(key)
        return SystemOneResponse(**data)


def to_jsonable(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def pattern_case(name, fn):
    out = to_jsonable(fn())
    return {"name": name, "output": out}


def scalar_case(name, fn):
    return {"name": name, "output": to_jsonable(fn())}


choice_resp = {
    "model": "von-1.1.0",
    "answers": {
        "route_question": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9, "tech": 0.1},
            "confidence": 0.8,
        }
    },
    "usage": {"input_tokens": 10, "output_tokens": 1},
}
gate_resp = {
    "model": "von-1.1.0",
    "answers": {
        "q_confident": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}, "confidence": 0.95},
        "q_unsure": {"type": "score", "score": 2.0, "confidence": 0.6, "legend": {"0": "l", "1": "m", "2": "h"}, "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2}},
        "q_noul": {"type": "noul", "noul": 0.4},
    },
    "usage": {"input_tokens": 20, "output_tokens": 3},
}
composite_resp = {
    "model": "von-1.1.0",
    "answers": {
        "rating": {"type": "score", "score": 3.2, "confidence": 0.9, "legend": {"0": "a", "1": "b", "2": "c", "3": "d"}, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}},
        "judgment": {"type": "noul", "noul": 0.55},
        "decision": {"type": "choice", "choice": "x", "probabilities": {"x": 1.0}, "confidence": 1.0},
    },
    "usage": {"input_tokens": 30, "output_tokens": 3},
}

pattern_fixtures = []

fake = FakeClient({"route_question": choice_resp})
_orig = None
import von.api as _api

_api._default_client = fake
pattern_fixtures.append(
    scalar_case(
        "route_hit",
        lambda: patterns.route(
            "state",
            Choice(instructions="Which?", criteria={"billing": None, "tech": None}),
            routes={"billing": lambda ans: f"handled:{ans.choice}"},
        ),
    )
)
pattern_fixtures.append(
    scalar_case(
        "route_default_low_conf",
        lambda: patterns.route(
            "state",
            Choice(instructions="Which?", criteria={"billing": None, "tech": None}),
            routes={},
            default=lambda ans: "defaulted",
            min_confidence=0.9,
        ),
    )
)
pattern_fixtures.append(
    scalar_case(
        "route_no_handler_returns_answer",
        lambda: patterns.route(
            "state",
            Choice(instructions="Which?", criteria={"billing": None, "tech": None}),
            routes={},
        ),
    )
)

fake = FakeClient({}, default=gate_resp)
_api._default_client = fake
pattern_fixtures.append(pattern_case("confidence_gate", lambda: patterns.confidence_gate("state", {"x": {"type": "choice", "instructions": "i", "criteria": {}}}, threshold=0.8)))

fake = FakeClient({}, default=composite_resp)
_api._default_client = fake
pattern_fixtures.append(
    pattern_case(
        "composite_score",
        lambda: patterns.composite_score(
            "state",
            {
                "rating": {"type": "score", "instructions": "i", "criteria": ["a"]},
                "judgment": {"type": "noul", "instructions": "i"},
                "decision": {"type": "choice", "instructions": "i", "criteria": {}},
            },
            weights={"rating": 2.0},
        )
    )
)

two_stage_resps = {
    "category": {
        "model": "von-1.1.0",
        "answers": {"category": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1}, "confidence": 0.9}},
        "usage": {"input_tokens": 10, "output_tokens": 1},
    },
    "option": {
        "model": "von-1.1.0",
        "answers": {"option": {"type": "choice", "choice": "invoice", "probabilities": {"invoice": 0.8, "refund": 0.2}, "confidence": 0.6}},
        "usage": {"input_tokens": 11, "output_tokens": 1},
    },
}


class TwoStageClient(FakeClient):
    def system_one(self, state, questions, model="von-latest"):
        self.calls.append({"state": state, "questions": questions, "model": model})
        key = list(questions.keys())[0]
        return SystemOneResponse(**two_stage_resps[key])


_api._default_client = TwoStageClient({})
pattern_fixtures.append(
    scalar_case(
        "two_stage_choice",
        lambda: patterns.two_stage_choice("state", {"billing": {"invoice": None, "refund": None}, "tech": {"bug": None}}),
    )
)

_api._default_client = None
dump("patterns.json", {"cases": pattern_fixtures})


# --------------------------------------------------------------- json.dumps pin
jd_values = [
    "hello", "caf\u00e9 \u4e2d\u6587 \U0001f600", "quote\" and 'single'", "back\\slash\nnewline\ttab",
    "ctrl\u0001\u001f\u007f", "", 5, -13, 1.5, 0.1, 1e16, 1e15, 1e-4, 1e-5, -0.0, 3.0,
    0.30000000000000004, 2.675, True, False, None, [], ["a", 1, 1.5, True, None],
    {"k": "v", "n": 2}, {"uni\u00e9": ["caf\u00e9", None]}, {"a": {"b": {"c": [1, {"d": 2}]}}},
]
jd_cases = []
for v in jd_values:
    jd_cases.append({
        "value": v,
        "compact": json.dumps(v),
        "indent2": json.dumps(v, indent=2),
    })
dump("json_dumps.json", {"cases": jd_cases})


# --------------------------------------------------------------------- presets
from von import presets as von_presets

preset_cases = []
for name, fn in [
    ("triage", von_presets.triage_preset),
    ("email_default", lambda: von_presets.email_preset()),
    ("email_custom", lambda: von_presets.email_preset({"custom": "Custom category description"})),
    ("moderation", von_presets.moderation_preset),
    ("security", von_presets.security_preset),
]:
    raw = fn()
    preset_cases.append({
        "name": name,
        "rendered": json.dumps({qid: q.model_dump() for qid, q in raw.items()}, ensure_ascii=False, separators=(",", ":")),
    })
dump("presets.json", {"cases": preset_cases})


# --------------------------------------------------- FastAPI wire (model-free)
os.environ.pop("VON_API_KEY", None)
os.environ["VON_DEVICE"] = "cpu"
from fastapi.testclient import TestClient
from von.server import app

client = TestClient(app)

WIRE_CASES = [
    ("post_missing_state", "POST", "/v1/systemone", '{"model": "von-latest", "questions": {}}', None),
    ("post_missing_questions", "POST", "/v1/systemone", '{"model": "von-latest", "state": "s"}', None),
    ("post_empty_object", "POST", "/v1/systemone", "{}", None),
    ("post_model_int", "POST", "/v1/systemone", '{"model": 5, "state": "s", "questions": {}}', None),
    ("post_questions_list", "POST", "/v1/systemone", '{"state": "s", "questions": []}', None),
    ("post_questions_value_list", "POST", "/v1/systemone", '{"state": "s", "questions": {"q": 5}}', None),
    ("post_body_list", "POST", "/v1/systemone", "[1,2]", None),
    ("post_body_string", "POST", "/v1/systemone", '"just a string"', None),
    ("post_body_number", "POST", "/v1/systemone", "7", None),
    ("post_body_empty", "POST", "/v1/systemone", "", None),
    ("post_malformed_truncated", "POST", "/v1/systemone", '{"model": "von-latest", "state": ', None),
    ("post_malformed_bad_value", "POST", "/v1/systemone", '{"model": , }', None),
    ("post_malformed_trailing", "POST", "/v1/systemone", '{"state": "s", "questions": {}} extra', None),
    ("err_unknown_qtype", "POST", "/v1/systemone", '{"state": "s", "questions": {"q": {"type": "bogus", "instructions": "?"}}}', None),
    ("err_missing_instructions", "POST", "/v1/systemone", '{"state": "s", "questions": {"q": {"type": "noul"}}}', None),
    ("err_choice_missing_criteria", "POST", "/v1/systemone", '{"state": "s", "questions": {"q": {"type": "choice", "instructions": "?"}}}', None),
    ("err_noul_legacy_conflict", "POST", "/v1/systemone", '{"state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "criteria": {"true": "Down"}, "pos_criteria": "Also down"}}}', None),
    ("err_qtype_int", "POST", "/v1/systemone", '{"state": "s", "questions": {"q": {"type": 5}}}', None),
    ("err_unknown_qtype_with_qid", "POST", "/v1/systemone", '{"state": "s", "questions": {"first": {"type": "choice", "instructions": "i", "criteria": {}}, "second": {"type": "nope", "instructions": "i"}}}', None),
    ("get_root", "GET", "/", None, None),
    ("get_health", "GET", "/health", None, None),
    ("get_models", "GET", "/v1/models", None, None),
    ("get_systemone", "GET", "/v1/systemone", None, None),
    ("put_systemone", "PUT", "/v1/systemone", '{"state": "s", "questions": {}}', None),
    ("head_systemone", "HEAD", "/v1/systemone", None, None),
    ("head_health", "HEAD", "/health", None, None),
    ("post_health", "POST", "/health", '{"x": 1}', None),
    ("get_unknown_path", "GET", "/nope", None, None),
    ("err_truncated_input_repr", "POST", "/v1/systemone", '{"state": "s", "questions": {"q": {"type": "noul", "junk": "' + "x" * 40 + '"}}}', None),
]

def norm_headers(h):
    return {k.lower(): v for k, v in h.items() if k.lower() != "content-length"}


wire_cases = []
for name, method, path, body, _ in WIRE_CASES:
    kwargs = {}
    req_headers = {"Content-Type": "application/json"} if body is not None else {}
    if body is not None:
        kwargs["content"] = body
        kwargs["headers"] = req_headers
    r = client.request(method, path, **kwargs)
    wire_cases.append({
        "id": name, "method": method, "path": path, "body": body,
        "request_headers": norm_headers(req_headers),
        "auth": None, "status": r.status_code, "response": r.text,
        "response_headers": norm_headers(r.headers),
    })

for name, headers in [
    ("options_no_cors_headers", {}),
    ("options_preflight", {"Origin": "http://example.com", "Access-Control-Request-Method": "POST"}),
]:
    r = client.request("OPTIONS", "/v1/systemone", headers=headers)
    wire_cases.append({
        "id": name, "method": "OPTIONS", "path": "/v1/systemone", "body": None,
        "request_headers": norm_headers(headers),
        "auth": None, "status": r.status_code, "response": r.text,
        "response_headers": norm_headers(r.headers),
    })

os.environ["VON_API_KEY"] = "sekrit"
for name, method, path, body in [
    ("auth_no_header", "POST", "/v1/systemone", '{"state": "s", "questions": {}}'),
    ("auth_bad_token", "POST", "/v1/systemone", '{"state": "s", "questions": {}}'),
    ("auth_not_bearer", "POST", "/v1/systemone", '{"state": "s", "questions": {}}'),
    ("auth_empty_token", "POST", "/v1/systemone", '{"state": "s", "questions": {}}'),
]:
    headers = {"Content-Type": "application/json"}
    auth = None
    if name == "auth_bad_token":
        auth = "Bearer wrong"
        headers["Authorization"] = auth
    elif name == "auth_not_bearer":
        auth = "Basic sekrit"
        headers["Authorization"] = auth
    elif name == "auth_empty_token":
        auth = "Bearer "
        headers["Authorization"] = auth
    r = client.request(method, path, content=body, headers=headers)
    wire_cases.append({"id": name, "method": method, "path": path, "body": body,
                       "request_headers": norm_headers(headers),
                       "auth": auth,
                       "status": r.status_code, "response": r.text,
                       "response_headers": norm_headers(r.headers)})
os.environ.pop("VON_API_KEY", None)

with open(OUT / "wire_errors.jsonl", "w", encoding="utf-8") as f:
    for c in wire_cases:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(f"wrote {OUT / 'wire_errors.jsonl'} ({len(wire_cases)} cases)")

print("done")
