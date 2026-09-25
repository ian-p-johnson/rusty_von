"""Differential harness: Python oracle vs Rust server, byte-for-byte (Stage 3).

Boots both servers (or attaches to running ones), then:

1. **Corpus phase** -- replays the golden corpus against BOTH servers and
   compares (a) each server against the recorded golden bytes and (b) the two
   live servers against each other.
2. **Fuzz phase** -- generates a seeded, deterministic stream of requests
   (valid evaluations, pathological payloads, malformed JSON, protocol-level
   probes), sends each to both servers, and compares status code + exact body.

Comparison classifier: byte-equal is the goal; the only tolerated exception is
the plan's "<= 1 ulp of the 4th decimal" envelope (the two known Stage 2
flips live inside it). Gate: zero divergences and >= 99.9% byte-identical.

Usage:
    uv run python scripts/diff_servers.py                     # boot both, corpus + 10k fuzz
    uv run python scripts/diff_servers.py --fuzz 200          # smoke run
    uv run python scripts/diff_servers.py --py-url http://127.0.0.1:8001 \
        --rs-url http://127.0.0.1:8002                        # attach to running servers

Ports default to VON_DIFF_PY_PORT / VON_DIFF_RS_PORT (8001 / 8002) so
long-running processes on this machine can't collide (PORTING_RUST.md 7.1).

Exit code 0 = gate green; 1 = divergences; 2 = setup failure.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_golden import first_divergence, load_jsonl  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDENS = REPO_ROOT / "goldens"
RUST_BIN = REPO_ROOT / "von-rs" / "target" / "release" / "von"
ONNX_ARTIFACT = REPO_ROOT / "von-rs" / "artifacts" / "von-option-marker.onnx"

# Per-field tolerance envelopes. The accepted Stage 2 numerics band
# (Rust-ORT vs Python-torch logit L∞ <= 1e-4) can straddle a rounding
# boundary at each field's own API precision in rare cases, so the envelope
# is expressed per field in units of that field's rounding step:
#   probabilities / noul : 4 decimals -> 2 units (2e-4)
#   confidence           : 3 decimals -> 1 unit  (1e-3)
#   score                : 2 decimals -> 1 unit  (1e-2)
# Anything beyond its envelope is a divergence. Strings/ints/bools/key order
# are exact.
def _tol_for(path: Tuple[str, ...]) -> float:
    leaf = path[-1] if path else ""
    if leaf == "confidence":
        return 0.001 + 1e-9
    if leaf == "score":
        return 0.01 + 1e-9
    return 0.0002


def _values_within_1ulp(a: Any, b: Any, path: Tuple[str, ...] = ()) -> bool:
    if type(a) is not type(b):
        # int vs float (2 vs 2.0) never occurs post-rounding; treat as divergence
        return False
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):  # order-sensitive on purpose
            return False
        return all(_values_within_1ulp(a[k], b[k], path + (k,)) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(
            _values_within_1ulp(x, y, path) for x, y in zip(a, b)
        )
    if isinstance(a, float):
        return abs(a - b) <= _tol_for(path)
    return a == b


def classify(status: int, body: str, want_status: int, want_body: str) -> Tuple[str, str]:
    """ -> (verdict, detail); verdict in {byte_equal, within_1ulp, divergent}."""
    if status != want_status:
        return "divergent", f"status {status} != {want_status}"
    if body == want_body:
        return "byte_equal", ""
    try:
        lhs, rhs = json.loads(want_body), json.loads(body)
    except json.JSONDecodeError:
        return "divergent", first_divergence(want_body, body)
    if _values_within_1ulp(lhs, rhs):
        return "within_1ulp", first_divergence(want_body, body)
    return "divergent", first_divergence(want_body, body)


# ---------------------------------------------------------------------------
# request execution
# ---------------------------------------------------------------------------

def send_case(client: httpx.Client, base: str, case: Dict[str, Any], auth: Optional[str]) -> Tuple[int, str]:
    headers: Dict[str, str] = dict(case.get("headers", {}))
    if case.get("auth") is not None:
        headers["Authorization"] = case["auth"]
    elif auth and case.get("method") == "POST" and "auth" not in case \
            and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {auth}"

    def attempt() -> Tuple[int, str]:
        try:
            if case["method"] == "GET":
                resp = client.get(f"{base}{case['path']}", headers=headers)
            elif case["method"] == "HEAD":
                resp = client.head(f"{base}{case['path']}", headers=headers)
            elif case["method"] == "OPTIONS":
                resp = client.options(f"{base}{case['path']}", headers=headers)
            else:
                resp = client.post(
                    f"{base}{case['path']}",
                    content=case["body"].encode("utf-8"),
                    headers=headers,
                )
            return resp.status_code, resp.text
        except httpx.HTTPError as exc:
            return -1, f"transport error: {exc}"

    # One retry on transport error: pooled-connection races under concurrency
    # should not be reported as (matching) responses. A real server fault
    # reproduces and still shows up as -1 on both attempts.
    first = attempt()
    if first[0] == -1:
        return attempt()
    return first


def run_phase(
    name: str,
    cases: List[Dict[str, Any]],
    py_url: str,
    rs_url: str,
    auth: Optional[str],
    workers: int,
) -> Dict[str, Any]:
    counts = {"byte_equal": 0, "within_1ulp": 0, "divergent": 0}
    py_status: Dict[int, int] = {}
    rs_status: Dict[int, int] = {}
    failures: List[str] = []
    ulp_cases: List[str] = []
    started = time.time()

    def one(idx: int) -> None:
        case = cases[idx]
        cid = case["id"]
        py = send_case(py_client, py_url, case, auth)
        rs = send_case(rs_client, rs_url, case, auth)
        py_status[py[0]] = py_status.get(py[0], 0) + 1
        rs_status[rs[0]] = rs_status.get(rs[0], 0) + 1
        verdict, detail = classify(rs[0], rs[1], py[0], py[1])
        counts[verdict] += 1
        if verdict == "within_1ulp":
            ulp_cases.append(cid)
        elif verdict == "divergent":
            failures.append(
                f"{cid}: py={py[0]} {py[1][:160]!r}\n"
                f"{' ' * (len(cid) + 6)}rs={rs[0]} {rs[1][:160]!r}\n"
                f"{' ' * (len(cid) + 6)}{detail}"
            )
        done = sum(counts.values())
        step = max(50, len(cases) // 20)
        if done % step == 0 or done == len(cases):
            print(f"  [{name}] {done}/{len(cases)} "
                  f"(byte={counts['byte_equal']} ulp={counts['within_1ulp']} div={counts['divergent']})",
                  flush=True)

    with httpx.Client(timeout=180.0) as py_client, httpx.Client(timeout=180.0) as rs_client:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, range(len(cases))))

    elapsed = time.time() - started
    print(f"[{name}] {len(cases)} cases in {elapsed:.1f}s: "
          f"byte_equal={counts['byte_equal']} within_1ulp={counts['within_1ulp']} "
          f"divergent={counts['divergent']}")
    print(f"  status coverage: py={dict(sorted(py_status.items()))} rs={dict(sorted(rs_status.items()))}")
    for cid in ulp_cases:
        print(f"  ulp: {cid}")
    for f in failures[:100]:
        print(f"  DIVERGENT {f}")
    if len(failures) > 100:
        print(f"  ... and {len(failures) - 100} more divergences")
    return {"cases": len(cases), "elapsed_s": round(elapsed, 1), **counts,
            "ulp_case_ids": ulp_cases, "failures": failures}


# ---------------------------------------------------------------------------
# golden corpus -> harness cases
# ---------------------------------------------------------------------------

def corpus_cases(auth: Optional[str]) -> List[Dict[str, Any]]:
    requests = load_jsonl(GOLDENS / "requests.jsonl")
    responses = {r["id"]: r for r in load_jsonl(GOLDENS / "responses.jsonl")}
    cases: List[Dict[str, Any]] = []
    for req in requests:
        expected = responses[req["id"]]
        case = {
            "id": req["id"],
            "method": req["method"],
            "path": req["path"],
            "body": req["body"],
            "headers": {"Content-Type": "application/json"} if req["method"] == "POST" else {},
        }
        # Auth-contract cases send exactly what was recorded; normal cases were
        # captured keyless, so inject the key when the target enforces one.
        if req.get("requires_auth"):
            case["auth"] = req.get("auth")
        elif auth:
            case["headers"]["Authorization"] = f"Bearer {auth}"
        case["golden_status"] = expected["status"]
        case["golden_body"] = expected["body"]
        cases.append(case)
    return cases


def run_corpus(py_url: str, rs_url: str, auth: Optional[str], workers: int) -> Tuple[Dict[str, Any], bool]:
    """Live diff AND each-side-vs-golden. Returns (report, ok)."""
    cases = corpus_cases(auth)
    live = run_phase("corpus live py-vs-rs", cases, py_url, rs_url, None, workers)

    def vs_golden(side: str, url: str) -> Tuple[int, List[str]]:
        divergent: List[str] = []
        ulp = 0
        with httpx.Client(timeout=180.0) as client:
            for case in cases:
                status, text = send_case(client, url, case, auth)
                verdict, detail = classify(status, text, case["golden_status"], case["golden_body"])
                # The plan's envelope: byte-equal, else <= 1 ulp of the 4th
                # decimal (the two known Stage 2 flips live there). Only a
                # true divergence fails the leg.
                if verdict == "within_1ulp":
                    ulp += 1
                elif verdict == "divergent":
                    divergent.append(f"{case['id']}: {verdict} {detail}")
        print(f"[corpus {side}-vs-golden] {len(cases)} cases: "
              f"{len(cases) - len(divergent) - ulp} exact, {ulp} within-1ulp, "
              f"{len(divergent)} divergent")
        for f in divergent[:10]:
            print(f"  DIVERGENT {f}")
        return ulp, divergent

    rs_ulp, rust_vs_golden = vs_golden("rs", rs_url)
    py_ulp, py_vs_golden = vs_golden("py", py_url)
    live["rs_vs_golden_divergent"] = len(rust_vs_golden)
    live["py_vs_golden_divergent"] = len(py_vs_golden)
    ok = live["divergent"] == 0 and not rust_vs_golden and not py_vs_golden
    return live, ok


# ---------------------------------------------------------------------------
# fuzz generation (seeded / deterministic)
# ---------------------------------------------------------------------------

WORDS = ("billing invoice refund charge customer server down feature request bug report "
         "crash latency deploy release rollback queue worker cache token model decision "
         "policy audit score level signal status region cluster replica backup restore").split()
UNICODES = ["客户要求退款", "🔥🀄🎉", "naïve café", "עברית מימין לשמאל", "λ = α + β",
            "emoji \U0001f600 family", "ゼロから始める", "единственный выход"]
ADVERSARIAL = ["literal [MASK] in text", "has [SEP] too", 'quotes " and \' mixed',
               "back\\slash\nnewline\ttab", "control\x01\x1f\x7f", "sep\u2028\u2029char",
               "percent %r and %s braces {}", "trailing space ", " null mid❌dle"]
FLOATS = [0.0, -0.0, 0.5, 1.0, 2.0, 0.1, 0.30000000000000004, 1 / 3, 3.14159,
          1e10, -2.5e-8, 123456789.123456, 1e-4, 6.02e23, -0.5]
MODELS = ["von-latest", "von-1.1", "von-1.1.0", "jev-latest", "von-1.0", "unknown-model", ""]
QKEYS = ["decision", "judgment", "rating", "q0", "q1", "класс", "質問"]


def _rand_text(rng: random.Random, min_w: int = 3, max_w: int = 40) -> str:
    n = rng.randint(min_w, max_w)
    parts = []
    for _ in range(n):
        r = rng.random()
        if r < 0.70:
            parts.append(rng.choice(WORDS))
        elif r < 0.85:
            parts.append(rng.choice(UNICODES))
        else:
            parts.append(rng.choice(ADVERSARIAL))
    return " ".join(parts)


def _rand_scalar(rng: random.Random) -> Any:
    r = rng.random()
    if r < 0.45:
        return _rand_text(rng, 1, 12)
    if r < 0.60:
        return rng.randint(-10**6, 10**6)
    if r < 0.75:
        return rng.choice(FLOATS)
    if r < 0.85:
        return rng.choice([True, False])
    if r < 0.93:
        return None
    return rng.choice(UNICODES)


def _rand_state(rng: random.Random) -> Any:
    r = rng.random()
    if r < 0.55:
        return _rand_text(rng, 3, 60)
    if r < 0.62:
        # medium-long: ~0.1k-1.2k tokens; the multi-K-token extreme stays
        # covered by the golden corpus itself (CPU forwards there cost seconds)
        rep = rng.randint(4, 40) if rng.random() < 0.85 else rng.randint(40, 150)
        return (_rand_text(rng, 4, 8) + " ") * rep
    if r < 0.70:
        return rng.choice(ADVERSARIAL + UNICODES)
    if r < 0.78:
        return _rand_scalar(rng)
    if r < 0.86:
        n = rng.randint(0, 6)
        return [_rand_scalar(rng) for _ in range(n)]
    if r < 0.93:
        return {f"k{i}": _rand_scalar(rng) for i in range(rng.randint(1, 5))}
    # nested dict with list/bool/None/float leaves (python-repr territory)
    return {
        "outer": {f"n{i}": _rand_scalar(rng) for i in range(rng.randint(1, 3))},
        "flags": [rng.choice([True, False, None]), rng.choice(FLOATS)],
        "meta": {"depth2": {"leaf": _rand_text(rng, 1, 5)}},
    }


def _desc(rng: random.Random) -> Any:
    r = rng.random()
    if r < 0.70:
        return _rand_text(rng, 1, 10)
    if r < 0.85:
        return None  # legal for Choice criteria
    return rng.choice([123, 4.5, True, None, ["list"], {"d": 1}])  # type violations


def _choice_q(rng: random.Random) -> Dict[str, Any]:
    k = rng.choice([1, 2, 2, 3, 3, 4, 5, 8, 25] if rng.random() < 0.15
                   else [1, 2, 3, 3, 4, 5])
    criteria: Dict[str, Any] = {}
    for i in range(k):
        key = _rand_text(rng, 1, 3).replace(" ", "_") or f"opt{i}"
        if key in criteria:
            key = f"{key}_{i}"
        criteria[key] = _desc(rng)
    if rng.random() < 0.05:
        criteria = {}  # empty-choices contract: choice="", probabilities={}
    return {"type": "choice",
            "instructions": _rand_text(rng, 2, 15),
            "criteria": criteria}


def _noul_q(rng: random.Random) -> Dict[str, Any]:
    q: Dict[str, Any] = {"type": "noul", "instructions": _rand_text(rng, 2, 15)}
    r = rng.random()
    if r < 0.35:
        q["criteria"] = None
    elif r < 0.55:
        q["criteria"] = {"true": _rand_text(rng, 1, 10)}
    elif r < 0.75:
        q["criteria"] = {"true": _rand_text(rng, 1, 8), "false": _rand_text(rng, 1, 8)}
    elif r < 0.85:
        q["pos_criteria"] = _rand_text(rng, 1, 8)  # legacy fold path
    elif r < 0.95:
        q["pos_criteria"] = _rand_text(rng, 1, 6)
        q["criteria"] = {"true": _rand_text(rng, 1, 6)}  # both- conflict -> 422
    else:
        q["criteria"] = {"true": 42}  # type violation
    if rng.random() < 0.03:
        q["unknown_extra"] = "ignored?"  # pydantic extra policy
    return q


def _score_q(rng: random.Random) -> Dict[str, Any]:
    n = rng.randint(1, 10)
    shape = rng.random()
    if shape < 0.60:
        criteria = [_rand_text(rng, 1, 6) for _ in range(n)]
    elif shape < 0.85:
        criteria = [{"what": _rand_text(rng, 1, 6), "examples": _rand_text(rng, 1, 8)}
                    for _ in range(n)]
    elif shape < 0.93:
        criteria = [{"what": _rand_text(rng, 1, 5), "examples": 7}]  # join-crash 422
    else:
        criteria = [_rand_scalar(rng) for _ in range(n)]  # union-type roulette
    return {"type": "score",
            "instructions": _rand_text(rng, 2, 12),
            "criteria": criteria}


def _garbage_q(rng: random.Random) -> Dict[str, Any]:
    return rng.choice([
        {"type": "unknown_type", "instructions": "x", "criteria": None},
        {"type": "choice", "instructions": "x"},               # missing criteria
        {"instructions": "x", "criteria": None},               # missing type
        {"type": 7, "instructions": "x", "criteria": None},    # wrong type literal
        "not even a dict",
        {"type": "noul", "instructions": 5, "criteria": None},  # wrong instructions type
    ])


def fuzz_cases(n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    cases: List[Dict[str, Any]] = []
    for i in range(n):
        r = rng.random()
        cid = f"fuzz_{i:05d}"
        if r < 0.08:
            # protocol-level probes
            kind = rng.random()
            if kind < 0.30:
                case = {"id": cid, "method": "GET", "path": rng.choice(
                    ["/health", "/v1/models", "/v1/systemone", "/nope", "/health?x=1"]), "body": ""}
            elif kind < 0.45:
                case = {"id": cid, "method": "HEAD", "path": rng.choice(
                    ["/health", "/v1/models", "/v1/systemone"]), "body": ""}
            elif kind < 0.60:
                case = {"id": cid, "method": "OPTIONS", "path": "/v1/systemone",
                        "body": "",
                        "headers": {"Origin": "http://example.com",
                                    "Access-Control-Request-Method": "POST"}}
            elif kind < 0.75:
                broken = rng.choice([
                    '{"model": "von-latest", "state"',        # truncated
                    '{"model": "von-latest"} {"x": 1}',       # extra data
                    '{invalid json}',                          # scanner
                    '',                                        # empty body
                    '[1, 2, 3]',                               # wrong shape
                ])
                case = {"id": cid, "method": "POST", "path": "/v1/systemone", "body": broken,
                        "headers": {"Content-Type": "application/json"}, "auth": None}
                if rng.random() < 0.30:
                    # FastAPI parses the JSON body regardless of content-type;
                    # keep both servers honest about that.
                    case["headers"]["Content-Type"] = rng.choice(
                        ["text/plain", "application/xml", ""]) or None
                    if case["headers"]["Content-Type"] is None:
                        del case["headers"]["Content-Type"]
            else:
                case = {"id": cid, "method": "POST", "path": "/v1/systemone",
                        "body": json.dumps({"model": "von-latest", "state": "s",
                                            "questions": {"q": _choice_q(rng)}}),
                        "headers": {"Content-Type": "application/json"}}
                case["auth"] = rng.choice([None, "Bearer wrong-key", "Basic dXNlcjpwYXNz",
                                           "Bearer ", "bearer lower-case"])
            cases.append(case)
            continue
        # evaluation payload
        n_q = rng.randint(1, 4)
        questions: Dict[str, Any] = {}
        for j in range(n_q):
            key = rng.choice(QKEYS)
            if key in questions:
                key = f"{key}_{j}"
            qr = rng.random()
            if qr < 0.45:
                questions[key] = _choice_q(rng)
            elif qr < 0.75:
                questions[key] = _noul_q(rng)
            elif qr < 0.95:
                questions[key] = _score_q(rng)
            else:
                questions[key] = _garbage_q(rng)
        payload: Dict[str, Any] = {"model": rng.choice(MODELS), "state": _rand_state(rng),
                                   "questions": questions}
        case = {"id": cid, "method": "POST", "path": "/v1/systemone",
                "body": json.dumps(payload, ensure_ascii=True),
                "headers": {"Content-Type": "application/json"}}
        cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# server boot
# ---------------------------------------------------------------------------

def wait_healthy(url: str, proc: Optional[subprocess.Popen], deadline_s: float = 240.0) -> bool:
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{url}/health", timeout=2.0)
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(1.0)
    return False


def golden_revision() -> str:
    try:
        with open(GOLDENS / "manifest.json", "r", encoding="utf-8") as f:
            rev = json.load(f).get("checkpoint_revision")
        if rev:
            return rev
    except (OSError, json.JSONDecodeError):
        pass
    print("warning: goldens/manifest.json has no checkpoint_revision; "
          "servers will resolve refs/main (unpinned)", file=sys.stderr)
    return ""


def boot_servers(args: argparse.Namespace) -> Tuple[str, str, List[subprocess.Popen]]:
    procs: List[subprocess.Popen] = []
    py_port = int(os.environ.get("VON_DIFF_PY_PORT", "8001"))
    rs_port = int(os.environ.get("VON_DIFF_RS_PORT", "8002"))
    py_url = args.py_url or f"http://127.0.0.1:{py_port}"
    rs_url = args.rs_url or f"http://127.0.0.1:{rs_port}"
    # One pin for both sides: the manifest's checkpoint revision is the single
    # source of truth for what the goldens were captured against. Both runtimes
    # honor VON_HF_REVISION (Python: transformers/hf_hub; Rust: snapshot_dir).
    revision = os.environ.get("VON_HF_REVISION") or golden_revision()

    def attached(url: str) -> bool:
        try:
            return httpx.get(f"{url}/health", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    if args.py_url:
        print(f"python oracle: attached {py_url}")
    else:
        if attached(py_url):
            print(f"python oracle: reusing healthy server on {py_url}")
        else:
            env = dict(os.environ, VON_DEVICE="cpu", VON_API_KEY=args.auth or "",
                       VON_HF_REVISION=revision)
            print(f"python oracle: booting uvicorn on {py_url} "
                  f"(cpu, auth {'on' if args.auth else 'off'}, rev {revision[:12] or 'latest'})")
            procs.append(subprocess.Popen(
                ["uv", "run", "uvicorn", "von.server:app", "--host", "127.0.0.1",
                 "--port", str(py_port)],
                cwd=REPO_ROOT, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    if args.rs_url:
        print(f"rust server:   attached {rs_url}")
    else:
        if attached(rs_url):
            print(f"rust server:   reusing healthy server on {rs_url}")
        else:
            if not RUST_BIN.is_file():
                print(f"rust binary missing at {RUST_BIN}; building (cargo build --release -p von-cli)")
                subprocess.run(["cargo", "build", "--release", "-p", "von-cli"],
                               cwd=REPO_ROOT / "von-rs", check=True)
            if not ONNX_ARTIFACT.is_file():
                print(f"error: ONNX artifact missing at {ONNX_ARTIFACT} "
                      f"(regenerate via von-rs/scripts/export_onnx.py)", file=sys.stderr)
                raise SystemExit(2)
            env = dict(os.environ, VON_API_KEY=args.auth or "",
                       VON_ONNX=str(ONNX_ARTIFACT), VON_HF_REVISION=revision)
            print(f"rust server:   booting {RUST_BIN.name} on {rs_url} "
                  f"(auth {'on' if args.auth else 'off'}, rev {revision[:12] or 'latest'})")
            procs.append(subprocess.Popen(
                [str(RUST_BIN), "serve", "--host", "127.0.0.1", "--port", str(rs_port),
                 "--device", "cpu"],
                cwd=REPO_ROOT, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    for url, proc in ((py_url, procs[0] if len(procs) == 2 else None),
                      (rs_url, procs[-1] if procs else None)):
        if not wait_healthy(url, proc):
            print(f"error: server at {url} failed to become healthy", file=sys.stderr)
            raise SystemExit(2)
    return py_url, rs_url, procs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fuzz", type=int, default=10_000,
                        help="number of fuzz cases (0 disables the fuzz phase)")
    parser.add_argument("--seed", type=int, default=20260924,
                        help="fuzz generator seed (recorded in the report)")
    parser.add_argument("--auth", default="golden-test-key",
                        help="API key both servers run with ('' disables auth)")
    parser.add_argument("--py-url", default=None, help="attach to a running Python server")
    parser.add_argument("--rs-url", default=None, help="attach to a running Rust server")
    parser.add_argument("--workers", type=int, default=8, help="concurrent in-flight cases")
    parser.add_argument("--no-corpus", action="store_true", help="skip the golden corpus phase")
    args = parser.parse_args()
    auth = args.auth or None

    py_url, rs_url, procs = boot_servers(args)
    try:
        reports: Dict[str, Dict[str, Any]] = {}
        ok = True
        if not args.no_corpus:
            reports["corpus"], corpus_ok = run_corpus(py_url, rs_url, auth, args.workers)
            ok &= corpus_ok
        if args.fuzz > 0:
            cases = fuzz_cases(args.fuzz, args.seed)
            print(f"[fuzz] generated {len(cases)} cases (seed={args.seed})")
            reports["fuzz"] = run_phase("fuzz py-vs-rs", cases, py_url, rs_url, auth, args.workers)
            ok &= reports["fuzz"]["divergent"] == 0

        total = sum(r["cases"] for r in reports.values())
        byte_eq = sum(r["byte_equal"] for r in reports.values())
        ulp = sum(r["within_1ulp"] for r in reports.values())
        div = sum(r["divergent"] for r in reports.values())
        pct = 100.0 * byte_eq / total if total else 100.0
        print("=" * 72)
        print(f"seed={args.seed}  total={total}  byte_equal={byte_eq} ({pct:.2f}%)  "
              f"within_1ulp={ulp}  divergent={div}")
        # Binding gate: nothing may diverge beyond the plan's <= 1-ulp-of-4th-
        # decimal envelope (the known ORT<->libtorch last-digit rounding);
        # byte-equality below 100% is reported as a statistic.
        gate = ok and div == 0
        print("GATE: GREEN" if gate else "GATE: RED")
        return 0 if gate else 1
    finally:
        for proc in procs:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
