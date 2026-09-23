# Porting Von to Rust — Susceptibility Assessment & Staged Plan

*Assessed against the working tree at `von-sdk` 1.1.1 (commit `657f42f`), with the
full test suite executed and passing (42/42).*

> **Companion document — `../laya/PORTING_RUST.md`.** Laya is the predecessor
> architecture (von's retired-backend list still names `laya` / `laya-421m`), and its
> Rust port is **executed through phase 8b** with the same shape as this plan
> (ModernBERT-family encoder + option-marker head + packing + calibration). Where that
> document reports a *measured* finding, this plan adopts it and marks it
> **[proven in laya]** — several of von's "risks to retire" are already retired there.
>
> **Sympathy & independence.** The two ports will run side by side on one machine.
> This plan deliberately keeps the *methodology* sympathetic to laya's (oracle-and-
> gates philosophy, fixture-driven parity rings, `make parity` as the one-command
> verification story, ort-first backend order) so the operator's mental model
> transfers between them — while keeping every *artifact* independent: separate
> workspace and crate namespace (`von-*`), separate venv, separate pyo3 package
> name, no shared code or build targets. See §7.1 for coexistence rules. Nothing
> here requires or produces changes to the laya tree.

---

## 1. Verdict

**Von is highly susceptible to a Rust port, and unusually amenable to a *verified* one.**

Three structural facts drive this:

1. **There is no custom CUDA to port.** Despite the project's GPU claims, a tree-wide
   search finds **zero** `.cu/.cuh/.cpp/.h` files. Every mention of CUDA is PyTorch's
   *device selection API* (`torch.device("cuda")`), AMP autocast/`GradScaler`, and NCCL
   in the training scripts. The runtime performs a plain forward pass through
   `transformers.AutoModel`. All GPU work is delegated to the torch runtime — which
   means a Rust port does not need to reimplement a single kernel; it needs a runtime
   that can execute a ModernBERT graph.

2. **The shippable runtime is small and clean.** `src/von` is ~1,350 lines of Python,
   of which only ~565 lines (`backends/option_marker_backend.py` + `models/option_marker.py`)
   touch tensors at all. The remainder — schemas, packing, orchestration, HTTP, CLI,
   patterns, presets — is pure deterministic business logic.

3. **The project already ships its own verification instruments.** A 42-test pytest
   suite that exercises the *real* weights end-to-end (semantic assertions: correct
   argmax, probability ranges), a 49-task / 869-case accuracy benchmark (`jabr v2`),
   a ViZDoom behavioral benchmark, and a wire protocol (`/v1/systemone`) with a
   second independent client implementation (TypeScript). The HTTP contract is the
   natural seam for differential verification against the running Python von.

The only genuinely hard part is reproducing the model's *numerics* bit-faithfully.
This document lays out a staged path that makes that tractable and keeps every stage
anchored to the actual running Python von.

---

## 2. Inventory & Susceptibility by Component

### 2.1 Runtime (`src/von`, the port target)

| Component | LOC | Nature | Rust susceptibility |
|---|---:|---|---|
| `types.py` | 120 | Pydantic schemas, legacy-key folding validator | **Trivial.** serde + hand validation. The `pos_criteria`/`neg_criteria` fold-with-warning logic is 30 lines of pure logic. |
| `engine.py` | 99 | Singleton orchestrator, alias resolution, version stamping | **Trivial.** Pure logic. |
| `api.py` | 88 | `decide`/`judge`/`rate`/`system_one` convenience wrappers | **Trivial.** |
| `patterns.py` | 182 | confidence_gate / route / composite_score / two_stage_choice | **Trivial.** Pure dict/float math. (The `routes: Callable` dispatch maps to closures/fn pointers naturally.) |
| `presets.py` | 139 | Static question definitions | **Trivial.** Static data. |
| `client.py` | 119 | In-process engine or httpx → `/v1/systemone` | **Trivial.** reqwest; or dropped entirely (Rust *is* the engine). |
| `server.py` | 93 | FastAPI: `/health`, `/v1/models`, `/v1/systemone`, bearer auth, CORS | **Easy.** axum/actix. Must match status-code semantics (401 vs 422) and JSON shapes. |
| `cli.py` | 208 | click CLI: serve/decide/judge/rate/eval | **Easy.** clap. |
| `device.py` | 61 | cuda/rocm/mps/dml/cpu detection | **Easy**, but semantics change with the inference runtime chosen (see §4). |
| `backends/option_marker_backend.py` | 480 | Sequence packing, mask-position gather, calibration map, noul debias, softmax, rounding, usage accounting | **Moderate.** The logic is simple; the *numerics and rounding semantics* are the risk (see §3). |
| `models/option_marker.py` | 85 | ModernBERT-large (395M) via `AutoModel` + MLP scorer head | **The hard part.** See §4. |

### 2.2 Everything else (not the port target)

| Area | LOC | Disposition |
|---|---:|---|
| `tests/` | ~530 | **Keep as the contract.** 42 tests, all passing; they assert semantic behavior of the real weights. |
| `benchmarks/` | ~4,300 | **Keep in Python initially.** jabr v2 (869 cases), calibration fitting, noul-prior fitting, ViZDoom eval. These are the *oracle instruments*; porting them would destroy the independent-observer property. |
| `training/` | ~5,400 | **Do not port.** DDP/NCCL/AMP/HF-datasets glue; torch ecosystem is the product here. Runs rarely (retraining), never in production inference. |
| `js/` | ~220 TS | **Unaffected.** Talks HTTP; a Rust server behind the same contract is invisible to it. |
| `.github/workflows/` | 5 | Extend: `test.yml` becomes the dual-runtime gate (see Stage 3). |

### 2.3 Artifacts

- `option_marker.pt` — torch `state_dict` (pickle/zip). Load is `strict=True`.
  → needs a **one-off, committed conversion script** to safetensors (Rust-native)
  or ONNX embedding.
- `marker_calibration.json` — scalar temperature + input-conditioned calibration map
  (`bias`, `entropy`, `log_tokens`, `n_options` coefficients; `lo`/`hi` bounds with
  validation & warning rules) + optional fitted noul prior `{a, b}`. All validation
  rules must be ported exactly (§3.3).
- ModernBERT `config.json` + tokenizer from HF repo `wfzyx/von`.

---

## 3. Parity Traps (where naive ports silently diverge)

These are the specific places where a straightforward Rust translation produces
*different answers than Python von*. Each must have a pinned test.

### 3.1 Rounding semantics
- Python `round(x, 4)` is **round-half-to-even** on the decimal expansion of the
  binary float. Rust's `f64::round()` is half-away-from-zero. Probabilities
  (`round(p, 4)`), confidence (`round(c, 3)`), and score (`round(s, 2)`) all flow
  through this. **Implement decimal half-even explicitly** and golden-test it.
  **[proven in laya]**: the `{:.4}`-format-and-parse trick reproduces Python's
  decimal rounding of the binary f64 exactly; adopt it rather than hand-rolling.
- JSON float serialization: Python `json` and `serde_json` both emit shortest-roundtrip
  representations for f64, so *if the f64 values are equal the text is equal*. This
  makes exact-response comparison feasible — but only after rounding semantics match.
- **[proven in laya]** Two more serialization fidelity points that only surfaced when
  comparing *live* side by side: serde_json cannot express Python `json.dumps`
  default separators (`", "`/`": "`) — wherever von's wire payloads embed
  JSON-rendered fragments, port a spaced-JSON writer; and Python always emits
  `None` fields (`detection: null`) — drop any `skip_serializing_if` on optional
  fields that Python serializes unconditionally.

### 3.2 State formatting
`_format_state` renders dict states as `"{k}: {v}"` lines using Python `str()`:
- `True` → `"True"`, `None` → `"None"`, `1.5` → `"1.5"`, lists render with single-quoted
  repr elements (`"['a', 'b']"`), nested dicts render as Python dict repr.
HTTP requests arrive as JSON, so the domain is bounded (str/int/float/bool/null/list/dict)
and a Python-`str`-compatible formatter is fully implementable — but it must be written
deliberately and golden-tested, including float shortest-repr and nested containers.
The string is tokenized, so any divergence changes logits downstream.

### 3.3 Calibration & debias math
- `_effective_temperature`: softmax in fp32 → entropy with `clamp_min(1e-12)`, natural
  log, normalized by `ln(n)`; feature `log10(tokens)/4` where tokens come from
  `tokenizer.encode(state, add_special_tokens=False)` (not a char proxy); clamp to
  `[lo, hi]`. Must be replicated operation-for-operation in the same precision.
- Zero-shot noul debias: a **second forward pass on an empty state**, then
  `logits[0] -= a*bias + b` (or `0.7 * bias` when no fitted prior). Both forward
  outputs feed the correction — parity requires both passes to match.
- Calibration-map validation: malformed → drop wholesale (fall back to scalar T),
  never raise; out-of-sane-range bounds → warning at load. Port the exact predicates.

### 3.4 Ordering & ties
- `questions` and `answers` are insertion-ordered dicts; JSON field order follows
  insertion. `serde_json` with `preserve_order` (IndexMap) required for byte-level
  response parity. **[proven in laya]**: enable the `preserve_order` feature
  workspace-wide from day one; a `BTreeMap` slip (alphabetical `script_profile`)
  survived three parity rings because they compare *parsed* dicts, where key order
  vanishes — it took live side-by-side comparison to flush it out.
- `torch.argmax` tie-breaking: return the **first** maximal index; pin with a
  synthetic-logits unit test (ties are near-impossible with real weights, but the
  contract should not be accidental). **[proven in laya]**: Rust `max_by` keeps the
  *last* max and Python `max()` keeps the *first* — write an explicit fold with
  strict `>` over a first-occurrence-ordered sequence; this genuinely fires on
  real inputs.
- `sorted(probs, reverse=True)` stability is irrelevant post-rounding, but confidence
  = top1 − top2 clamped to [0,1] must use *unrounded* probabilities (as Python does).

### 3.5 HTTP contract details
- `model` in every response is stamped `von-1.1.0` regardless of requested model
  (old IDs accepted, never reflected).
- 401 (missing/invalid bearer when `VON_API_KEY` set) vs 422 (evaluation errors) vs
  FastAPI's own 422 for malformed bodies — the TS SDK and `test_server.py` pin these.
- Usage accounting: `input_tokens = max(1, len(state_str)//4) + max(1, total_q_chars//4)`,
  `output_tokens = len(answers)`. Deterministic, must match exactly.
- Empty-choices Choice returns `choice=""`, `probabilities={}`, `confidence=0.0`
  (no error); `decide()` raises on duplicate list choices; Score with `len < 2` levels
  returns zeros from the backend but the CLI rejects `< 2`.

---

## 4. Model Execution Strategy

The forward pass is: tokenize → ModernBERT-large (rope, 8192 ctx, alternating
local/global attention) → gather hidden states at `[MASK]` positions → MLP scorer
(LayerNorm → Linear(1024→512) → GELU → LayerNorm → Dropout(eval=noop) → Linear(512→1))
→ K logits. Two credible routes:

### Route A (recommended): ONNX export + `ort` crate
- A committed Python script exports the **fused graph** (encoder + scorer head, so
  mask-gather happens in-graph) to ONNX at a pinned opset, fp32.
- `ort` (ONNX Runtime bindings) provides CUDA, TensorRT, CoreML, DirectML, and ROCm
  execution providers — covering the entire device matrix `device.py` currently
  handles, without writing device code.
- Numerics: ORT CPU fp32 vs libtorch CPU fp32 on transformer graphs typically agree
  to ~1e-5–1e-4 — well inside the API's 4-decimal rounding envelope.
- **[proven in laya]** Laya exported this exact architecture family
  (ModernBERT encoder + decision head) as *one* graph and hit logit L∞ = **8.8e-5**
  on CPU (gate ≤ 1e-4) and 47/47 argmax agreement on CUDA. Exporter findings to
  inherit, not rediscover:
  - The **legacy exporter chokes** on the fused
    `aten::_transformer_encoder_layer_fwd` fast-path — use the **dynamo exporter**.
  - The dynamo exporter stamps opset 17 while emitting opset-18-style `Split` nodes
    (`num_outputs`) — **declare opset 18**.
  - Inline the external `.data` file into a single artifact (one fewer thing to
    deploy alongside `option_marker.pt`).
  - **Blackwell (sm_120) trap**: `ort-sys`'s downloaded prebuilt CUDA binaries abort
    with `cudaErrorNoKernelImageForDevice` (the `sm_120` strings belong to
    statically-linked cuBLAS/cuDNN, not ORT kernels). Fix is `ort/load-dynamic`
    pointed at the **official `onnxruntime-gpu` wheel's dylib** (`ORT_DYLIB_PATH`).
    Von's Stage 4 device work must start from this, not discover it.

### Route B: pure-Rust graph implementation (`candle` or `burn`)
- Candle ships BERT-family support incl. ModernBERT; weights load from safetensors
  natively. No ONNX runtime dependency; single static binary; excellent CPU story
  (and Metal/Vulkan/CUDA backends, unevenly mature).
- You own the kernel numerics and the architecture code; parity risk is higher and
  the roadmap below gates it on the same instruments.
- **[proven in laya]** The tch variant of this route (hand-written ModernBERT + head,
  `LIBTORCH_USE_PYTORCH=1` sharing the oracle venv's libtorch) reached **1.8e-5 corpus
  logit L∞** in ~1–2 days once the parity apparatus existed — von's Stage 2 Route-B
  estimate shrinks accordingly. Two landmines Laya's per-layer probe caught apply
  verbatim to von's encoder:
  - **GeGLU gate order**: in transformers 5.17's ModernBERT the gate is on the
    **first** chunk (`gelu(a) * b`) — *opposite to candle's reference
    implementation*. Any Route B port that cross-checks against candle will be wrong;
    cross-check against forward-hook captures from the Python oracle instead.
  - **Masking must use `masked_fill`, not multiplication** — `0 × (−∞) = NaN` in the
    additive attention mask propagates silently (NaNs skip `f64::max` folds).
  - Hidden-state drift grows superlinearly with depth (~7.6e-6 at embeddings → ~4.7e-1
    at layer 23 of 28) from op-order differences alone, then washes down to ~1.8e-5 at
    the logits. Gate early layers structurally (≤1e-4) and gate *behavior* on outputs,
    not on deep hidden states.
  - **Config trap**: transformers 5.x writes RoPE thetas in a nested
    `rope_parameters` object and ships explicit `layer_types`; the scalar
    `global_rope_theta`/`local_rope_theta` keys exist in *neither* checkpoint.
    Laya's parser defaults matched ModernBERT-large **by pure coincidence**. Von must
    parse the nested form, error loudly on missing thetas, and take per-layer
    global/sliding from `layer_types`.
  - **Special-token alias trap**: don't discover `[MASK]`/`[SEP]` by trying alias
    spellings in a fixed order — mmBERT's vocab carries Llama-style `<s>`/`</s>` as
    *regular* tokens, and alias-first discovery silently produced wrong ids with
    decode-identical text. Pin the ids the Python `AutoTokenizer` reports as a
    fixture instead.
- Sensible as a *later* optimization (smaller dep tree, edge deployments), not as the
  first cut.

### Tokenizer
Non-issue, favorably: HuggingFace **fast tokenizers are the Rust `tokenizers` crate**
already — Python has been calling this exact Rust code. Parity is structural; still
verify with golden token-ID sequences including the `[MASK]`/`[SEP]` round-trip and
`add_special_tokens=False` counting used by the calibration features.

---

## 5. Verification Architecture (the core of this plan)

The guiding rule: **the Python von is the oracle, forever in-tree, and every stage
must prove equivalence against it — not against a spec.**

Von already has three verification layers that a port can lean on:

1. **`tests/` (42 tests)** — semantic behavior of the real weights through the public
   API, HTTP server contract via `TestClient`, openjev wire-compatibility cases
   (probability normalization, duplicate rejection, structured state).
2. **`benchmarks/jabr_cases_v2.py` (49 tasks / 869 cases)** + `run_marker_benchmark.py`
   — accuracy as a behavioral fingerprint. A numerically divergent port shows up as
   per-task accuracy drift long before users notice.
3. **`benchmarks/fit_calibration.py` / `noul_prior_fit.py`** — regression tests for
   the calibration math specifically.

Stage 0 adds the two layers that are missing today:

4. **Golden corpus** — versioned JSONL of requests → *exact* responses captured from
   the running Python von, spanning the input space (see Stage 0).
5. **Logit-level goldens** — the sharpest instrument: a probe that imports the
   backend and dumps **pre-softmax logits** for fixed inputs. Business logic and
   numerics are then verified independently, and a failure localizes immediately.

Plus one enabling change to the Python tree (small, worth keeping permanently):
a hook so the pytest suite can target a *remote* server. `VonClient` already supports
`base_url` mode; `api._get_default_client()` currently hardcodes `local=True`. Adding
"if `VON_TEST_BASE_URL` is set, return a remote client" (~5 lines in `api.py`/`conftest.py`)
makes the **entire existing suite a differential harness**: same tests, unmodified,
run against Python and against Rust. Similarly, `run_marker_benchmark.py`'s backend
can be pointed at an HTTP endpoint.

### Gates (defined once, enforced at every stage)

| Gate | Threshold | Instrument |
|---|---|---|
| Token parity | 100% identical token IDs | Golden token sequences |
| Logit parity (CPU fp32) | max abs Δ ≤ 1e-4 (rel ≤ 1e-5 on |logit| > 1) | Logit goldens |
| Argmax agreement | 100% over corpus | Golden corpus |
| Probability fields | exact equality after API rounding on ≥ 99.9% of corpus; every exception ≤ 1 ulp of the 4th decimal | Golden corpus diff |
| Semantic test suite | 42/42 pass unmodified | pytest against Rust engine (local & HTTP) |
| Accuracy fingerprint | jabr v2 macro/micro within ±0.2pp of Python baseline; no task drops > 2pp | `run_marker_benchmark.py` |
| Wire contract | byte-identical JSON (modulo header order) over corpus + fuzz | Differential server harness |

---

## 6. Staged Plan

### Stage 0 — Baseline & instrumentation (Python-only; no Rust yet) — *2–4 days*

Deliverables:
1. **Golden capture script** (`benchmarks/capture_golden.py`): runs the real backend,
   emits `goldens/requests.jsonl` + `goldens/responses.jsonl` (exact rounded fields)
   and `goldens/logits.jsonl` (pre-softmax logits + metadata: n_options, eff_temp).
   Coverage matrix:
   - Choice: K = 1, 2, 3, 5, 8, 25 options; with/without descriptions; unicode;
     8k-token state; empty options.
   - Noul: with explicit criteria, zero-shot (debias path), fitted-prior path.
   - Score: 2–10 levels; str and dict (`what`/`examples`) criteria shapes.
   - State: str, dict (nested list/float/bool/None), list-of-strings, ints.
   - Error cases: duplicate choices, unknown question type, both- criteria conflict.
2. **Probe script** (`benchmarks/dump_logits.py`) used above — kept permanently; it is
   the numerics stethoscope.
3. **Remote-target hook** in `api.py`/`conftest.py` (`VON_TEST_BASE_URL`) + a small
   runner script `scripts/run_tests_against.sh <url>`.
4. **Record the baseline**: run `run_marker_benchmark.py`, save the per-task accuracy
   table and mean latency on this machine to `benchmarks/results/python-baseline.json`;
   pin the HF checkpoint revision hash.
5. Golden **replay verifier** (`scripts/replay_golden.py <base_url>`): posts every
   request to any HTTP endpoint, diffs exact responses, prints a parity report.

Exit criteria: goldens committed; replay verifier green against the Python server
(trivially); baseline table recorded; all 42 tests green (already confirmed).

### Stage 1 — Cargo workspace + pure-logic port — *1–2 weeks*

Deliverables:
- Workspace: `von-types` (serde schemas + validation + legacy folding),
  `von-core` (Python-compatible state formatting, `pack_sequence`, rounding
  utilities incl. decimal half-even, usage accounting, version stamping),
  `von-patterns`, `von-presets` (static data), `von-server` (axum skeleton with
  stubbed engine), `von-cli`.
- Python-`str()`-compatible renderer for JSON values (§3.2) with exhaustive unit tests
  generated from Python outputs.
- Fixtures: Python script emits expected outputs for the pure-logic units (packed
  strings, usage numbers, response envelopes, error messages) → Rust tests consume
  the fixtures.

Verification: pure-logic fixtures pass 100%. Gate "wire contract" passes for
*malformed/edge* requests against a stub-engine server (error codes, envelope shapes).

### Stage 2 — Inference engine — *2–4 weeks* (Route A) / *4–7 weeks* (Route B)

Order of work (each sub-step gated before the next):
1. **Tokenizer harness**: `tokenizers` crate; assert golden token IDs for every
   packed sequence in the corpus (incl. mask positions extraction).
2. **Export spike (Route A)**: ONNX export of fused encoder+scorer; run one golden
   input through `ort` CPU; check logit delta. *Low-risk step* — laya already proved
   this export for the architecture family (§4); this spike is mostly applying its
   recipe (dynamo exporter, opset 18, inlined data) to von's head.
   Route B equivalent: implement ModernBERT + scorer in candle against the same check,
   using oracle forward-hook captures (not candle's reference impl) as cross-reference.
3. **Full engine**: weight conversion script (`training/export/convert_weights.py`,
   one-off, committed, versioned output), safetensors load, mask gather, scorer,
   `_effective_temperature` (§3.3), noul debias dual-pass, softmax, rounding,
   answer construction.
4. **Parity run**: replay the full golden corpus through the Rust engine
   (in-process first, then via HTTP).

Verification: token gate 100%; logit gate ≤ 1e-4; argmax 100%; probabilities
exact-after-rounding ≥ 99.9% (stragglers ≤ 1 ulp at the 4th decimal); all 42 pytest
tests pass with the suite pointed at the Rust engine (both in-process via a thin
PyO3/test binary or, simpler at this stage, the Rust server of Stage 3 pulled early
in minimal form).

Exit criteria: **the complete existing test suite, unmodified, passes against the
Rust engine.** This is the stage's entire justification for existence.

### Stage 3 — Server & CLI parity — *1–2 weeks*

Deliverables:
- axum server: `/health`, `/v1/models`, `/v1/systemone`; bearer auth honoring
  `VON_API_KEY`; CORS via `VON_CORS_ORIGINS` incl. the wildcard/credentials rule;
  error semantics (401 vs 422 vs 400 for malformed JSON); response field order
  preserved; `model` stamped `von-1.1.0`.
- clap CLI mirroring `von serve/decide/judge/rate/eval` including the
  `--model` choice set derived from the alias list (the CLI-drift regression that
  `test_backends.py::test_cli_model_choices_match_engine_aliases` guards against).
- **Differential harness** (`scripts/diff_servers.py`): boots Python (uvicorn :8001)
  and Rust (:8002) servers; replays golden corpus *plus* a generated fuzz stream
  (random states/questions incl. pathological cases); asserts byte-equal JSON bodies
  and matching status codes.

Verification: differential harness green over corpus + ≥ 10k fuzz requests; pytest
suite green against Rust server via `VON_TEST_BASE_URL`; TS SDK test suite (`js/tests`)
green against the Rust server.

### Stage 4 — Performance, devices, hardening — *2–3 weeks*

Deliverables:
- Device support via ORT execution providers; `--device auto|cuda|rocm|coreml|dml|cpu`
  mapped to EP selection; document the mapping vs `device.py` semantics.
- Concurrency: the Python backend is effectively serialized around a singleton
  engine; Rust can run the session concurrently / with a session pool and optional
  micro-batching of concurrent requests (a *behavioral* change — gate: batching must
  preserve per-request logit parity, which single-session sequential execution
  guarantees by default).
- Latency benchmark p50/p95/p99 vs the Python baseline table from Stage 0.
- Soak: sustained load, memory profile (expect large drop: no Python/torch runtime).

Verification: accuracy fingerprint re-run (jabr v2) on each device backend vs
Python same-device baseline; logit gate re-run per device (GPU numerics differ from
CPU — compare GPU-Rust vs GPU-Python, not GPU vs CPU); 42-test suite green.

### Stage 5 — Distribution & cutover — *2–3 weeks*

Deliverables (choose per product goal; they are independent):
- **PyO3 wheel**: `von._rust` exposing the engine so `pip install von-sdk` users get
  Rust inference transparently (Python remains the API surface). Gate: the 42-test
  suite passes with `_rust` engine enabled via env flag, defaulting off for one
  release. **[proven in laya]**: the side-by-side methodology is what makes this
  gate strong — install the cdylib *beside* the oracle package under a distinct
  name and run one script that exercises both live against the same inputs
  (laya: 263/263 checks, which flushed out three fidelity bugs — key-order, null
  emission, `%r`-vs-`{:?}` quote rendering — that all parsed-dict comparisons had
  missed). Include the human-visible strings in the comparison: Python's `%r`
  renders single quotes where Rust `{:?}` renders double; von's error messages
  (`Unknown model '...'`) must byte-match. Operational note from laya: keep the
  maturin install crate-local (a bare `Cargo.toml` can make maturin walk up to the
  repo-root `pyproject.toml` and install over the oracle).
- **Standalone binary / container**: `von-server` image becomes the deployment
  artifact; PyPI package can slim to a pure client + server launcher.
- **CI rewiring**: `test.yml` becomes a matrix:
  - job `python-reference`: full pytest + regenerate goldens on version bump
    (goldens re-pinned only by explicit PR that updates the checkpoint hash);
  - job `rust`: cargo test + parity vs pinned goldens + full pytest via
    `VON_TEST_BASE_URL` against the freshly built Rust server + jabr v2 fingerprint
    gate.
  The Python reference build stays in CI **indefinitely** — it is what keeps the
  oracle honest.

Exit criteria: both runtimes ship green from the same tree; README documents the
Rust runtime as supported; the Python path remains the reference implementation.

### Explicitly out of scope (keep Python)
- `training/` (DDP/NCCL/AMP training, dataset preparation, AWS launchers).
- Calibration & prior fitting (`fit_calibration*.py`, `noul_prior_fit.py`) — they
  produce artifacts the Rust runtime *consumes*; keeping them Python preserves one
  source of truth.
- Benchmark suite & ViZDoom eval — they are the independent observer; porting them
  to the thing they observe would invalidate them. (Optionally add an HTTP adapter
  so they can score any endpoint, which increases their value.)

---

## 7. Effort & Risk Summary

**Total: ~8–13 weeks** of one experienced engineer for the Route A (ONNX) path;
**~4–7 weeks** for the pure-Rust Route B (revised down from 3.5–5 months: laya
executed the equivalent hand-written-encoder route in days once the parity apparatus
existed). Stages 0–3 are the critical path to a verified drop-in server; Stages 4–5
are productization and can interleave with normal feature work.

**Performance expectations, tempered by laya's measurements**: do not assume Rust is
faster. Laya's honest end-state was **31.2 ms/call (Rust, ort, CUDA) vs 25.2 ms
(Python/torch)** on the same GPU — *slower*, within 24%, before any optimization
work. At von's batch sizes the 395M encoder dominates both runtimes; Rust's wins are
concurrency (no GIL), deployment footprint, and startup, not single-call latency.
Measure and report; don't optimize (and if latency matters, the lever is session
pooling/batching and graph opts, not the language).

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ModernBERT ONNX export issues (opset/attention structure) | **Low** (was Medium) | Stage 2 delay | **[proven in laya]** same architecture family exported successfully; inherit the dynamo-exporter/opset-18/inline-data recipe and the Blackwell `load-dynamic` fix |
| GPU logit drift (different EP/kernel paths) | Medium | Latent accuracy drift | Per-device parity runs vs Python on the *same* device; accuracy fingerprint gates in CI; note Python itself flips across fp16/fp32 device paths, so gate argmax agreement (≥99.5%) + confidence MAE rather than bit-parity on GPU |
| Rounding/formatting divergence (§3) | High if untested | Wrong probabilities in the last decimal; byte-diff noise | Half-even decimal rounding util + Python-`str` formatter with generated fixtures — both in Stage 1, before any model work; solutions field-validated in laya |
| Config-format divergence (transformers 5.x nested `rope_parameters`/`layer_types`) | Medium if Route B | Silent encoder corruption | Parse the nested form, error loudly on missing thetas, take per-layer global/sliding from `layer_types` |
| GeGLU gate-order / masking bugs in Route B | High on first attempt (proven) | Encoder-wide divergence | Per-layer forward-hook probe from the Python oracle as the *only* cross-reference; never candle's implementation (its gate order is opposite to transformers 5.17) |
| Test suite depends on 1.5GB HF download | Certain | Slow/flaky CI | CI caching of the HF cache keyed on checkpoint revision (already the implicit pin) |
| Scope creep into a "better Python" rewrite | Medium | Oracle contamination, schedule slip | §6 out-of-scope list is a hard rule; phase gates are the contract (laya's defense, adopted verbatim) |
| Disk pressure from cargo `target/` + ONNX artifacts | High (proven: hit 100% twice in laya) | Blocked builds | Budget ~5 GB (target ~2.4 GB regenerable + ~1.7 GB ONNX + CUDA libs); `cargo clean` discipline; keep artifacts out of the root partition if possible |
| Calibration map / noul prior files drift schema | Low | Silent fallback to T=1.0 | Port the validators exactly; add a load-time parity test feeding malformed fixtures from Python |

### 7.1 Side-by-side coexistence with the laya port

The von port runs on the same machine as the (executed) laya port. Rules that keep
them independent — and keep laya's tree untouched:

1. **Zero shared artifacts.** Von's workspace is `von-rs/` with crates `von-types`,
   `von-core`, `von-server`, `von-cli`, … — no crate, path, fixture directory,
   or `Cargo.toml` is shared with, imported by, or referenced from `laya-rs/`.
   Parity instruments are duplicated, not shared: von gets its own
   `dump_fixtures.py`/replay scripts in `benchmarks/`+`scripts/`, even though they
   resemble laya's. Shared code would couple release cadences and oracle pins.
2. **Separate environments.** Von's oracle lives in von's own `.venv` (already
   present); laya's stays in its repo. Never `pip install` either port's pyo3
   cdylib into the other project's venv. Von's cdylib is named **`von-rs`**
   (distinct from `laya-rs`), with a **crate-local `pyproject.toml`** from day one —
   laya's phase 8b showed maturin walks up to a repo-root `pyproject.toml` and can
   install over an oracle; with two ports on one machine that failure mode doubles.
3. **Disk budget, combined.** Each port needs ~5 GB (cargo `target/` ~2.4 GB
   regenerable + ONNX artifact ~1.7 GB + CUDA libs). Laya's log records hitting 100%
   disk twice; von must not assume that space exists. Before Stage 2: check free
   space, run `cargo clean` in laya-rs if it's dormant, and keep von's heavy
   artifacts on the partition with room. Don't share `CARGO_TARGET_DIR` — the
   coupling isn't worth the space.
4. **HF cache is shared, read-only, and that's fine.** Both projects resolve
   checkpoints from `~/.cache/huggingface` by repo id (`wfzyx/von` vs laya's repos);
   the cache is content-addressed and concurrent-safe. This is the one deliberate
   overlap, and it saves ~1.5 GB.
5. **GPU exclusivity for timing gates.** Both models are ~400M params (~1.7 GB VRAM
   each, fp32) so both *fit* on one 16 GB GPU concurrently — but latency numbers are
   only comparable when measured exclusively. Von's Stage 4 timing runs get the GPU
   to themselves; never benchmark while the other port's parity/e2e suite is active.
6. **Ports and processes.** Von's differential harness binds Python :8001 / Rust
   :8002 — make both configurable via env (`VON_DIFF_PY_PORT`, `VON_DIFF_RS_PORT`)
   so long-running laya CLI/parity processes can't collide.
7. **What is deliberately *not* inherited from laya.** Its router/LRU, multilingual
   mmBERT support, language detection, and the mmBERT-specific findings (RoPE
   θ=160000/160000, Llama-alias vocabulary, 8k context) are laya-only concerns;
   von's ModernBERT-large checkpoints pin special ids from the oracle instead of
   discovering aliases. Conversely, laya never had: an HTTP server contract, a TS
   SDK, a calibration map file, the noul dual-pass debias, usage accounting, or the
   jabr/doom benchmark fingerprint — those are von-only verification surface and are
   why von's Stage 0/3 have no laya analogue to copy.

---

## 8. Recommended Stack

| Concern | Choice |
|---|---|
| Tokenizer | `tokenizers` (same Rust impl HF Python uses) |
| Inference | `ort` + ONNX (fp32; EPs for CUDA/TensorRT/CoreML/DirectML/ROCm); `candle`/`burn` as a later pure-Rust option |
| Weights | safetensors via committed one-off converter |
| Server | axum + tokio |
| Schemas | serde + serde_json (`preserve_order`) |
| CLI | clap (derive) |
| PyO3 wheel (Stage 5, optional) | pyo3 + maturin |
| Testing | cargo test + golden replay + differential server harness + (unmodified) pytest |

---

## 9. Progress log

### Stage 0 — complete (2026-09-23)

Python oracle instrumented; every exit criterion met. Working tree left uncommitted
by decision.

- **Goldens** (`goldens/`): 53 request cases + 53 exact-response bodies + 45
  pre-softmax probe readouts + 3 function-level error contracts + manifest
  (checkpoint revision `d8bb5e07…` pinned, sha256 per artifact, CPU fp32 policy
  recorded). Coverage: choice K=0/1/2/3/5/8/25, descriptions present/absent,
  unicode, literal `[MASK]`, empty instructions, ~6k-token state, noul
  explicit/one-sided/zero-shot/legacy-fold/synthetic-prior, score 2–10 levels in
  str+dict shapes, all state JSON shapes incl. nested-dict repr, 4-question
  fan-out, model-field variations, auth 401s, validation/eval 422s, 405, GETs.
  **Byte-deterministic: two capture runs diff clean.**
- **Probe** (`benchmarks/dump_logits.py`): opens the backend's pack/forward/
  temperature sequence; reports raw logits, null-state debias logits + correction,
  effective temperature, unrounded probabilities. The numerics stethoscope.
- **Replay verifier** (`scripts/replay_golden.py`): exact status+body string diff
  against any `/v1/systemone` endpoint; `--auth` mode covers the 401 contract.
  **53/53 exact against the live Python server.**
- **Remote-target hook**: `VON_TEST_BASE_URL` in `api.py` + hermetic `conftest`
  + `scripts/run_tests_against.sh`. The unmodified suite now runs against any
  contract-conformant server: **42/42 remote, 42/42 in-process.** (One fix
  required: `test_server.py` now clears ambient `VON_API_KEY` — local-app tests
  must not inherit the remote run's key.)
- **Baseline** (`benchmarks/results/python-baseline.json`): jabr v1 micro
  **0.936** / macro **0.927**, ~18–28 ms/case on this machine's GPU via the
  shipped Hub weights; `run_marker_benchmark.py` gained backward-compatible
  `--checkpoint-dir/--device/--json` flags.
- **One-command oracle check**: `scripts/verify_python_oracle.sh` — server up →
  golden replay → pytest remote → pytest in-process. Green end-to-end.

Quirks the goldens pinned (a Rust port must reproduce all of them):
1. Literal `[MASK]` in state text creates phantom option positions;
   `zip(options, probs)` silently truncates, so **reported probabilities do not
   sum to 1.0** (`choice_mask_literal`: 0.4892 + 0.2026). An argmax beyond K
   would `IndexError` → 422.
2. Score criteria dicts with non-string `examples` crash `', '.join` → 422 with
   a specific message — pinned as contract.
3. `state: null` becomes the literal string `"None"`; nested dict states render
   via Python repr (single quotes) — `state_dict_nested` pins it.
4. Empty `questions: {}` yields `input_tokens` with the `max(1, 0//4)` floor.
5. pydantic/FastAPI error bodies (incl. `errors.pydantic.dev/2.13/...` URLs) are
   byte-pinned; a Rust server must match them to pass the wire gate.

Note for later stages: package `__version__` says `1.0.1` while `pyproject.toml`
says `1.1.1` (pre-existing drift, `von.__version__` feeds the manifest only).
The server's `VON_DEVICE` must be `cpu` when replaying goldens; GPU responses
diverge by design and get their own Stage 4 gates.

### Stage 1 — complete (2026-09-23)

Cargo workspace + pure-logic port landed in `von-rs/` (six crates, edition 2024,
`serde_json/preserve_order` workspace-wide per §3.4). Every pure-logic unit is
fixture-tested against Python-emitted expectations, and the wire gate replays
**34 captured error/edge cases byte-identically** against the axum stub server.
Clippy clean (0 warnings), `cargo fmt` clean, 39 Rust tests green.

- **Fixture generator** (`benchmarks/dump_fixtures.py`, model-free): emits
  `von-rs/fixtures/*.json` + `wire_errors.jsonl` from the live Python oracle —
  state formatting (`_format_state` + `str()` for 40 values), `pack_sequence`,
  `round()` half-even (45 halfway/binary-expansion cases), usage floors,
  envelope key order (FastAPI render), API error strings, patterns logic (with a
  fake client), `json.dumps` defaults, presets, and full request/response wire
  captures (incl. response **and request** headers for CORS preflight).
- **Crates**: `von-types` (schemas, legacy `pos_criteria`/`neg_criteria`
  fold-with-warning, pydantic-2.13-compatible validation-error renderer),
  `von-core` (Python `str()`/`repr()` renderer incl. float repr & quote
  switching, `format_state`, `pack_sequence`, `{:.4}`-format-and-parse half-even
  rounding [proven in laya], usage accounting, engine aliases/stamping, the
  `Engine` trait + structure-faithful **stub engine** with uniform-probability
  outputs, `json.dumps(ensure_ascii=True)` writer), `von-presets`,
  `von-patterns` (confidence_gate/route/composite_score/two_stage_choice over
  the trait), `von-server` (axum), `von-cli` (clap; serve/decide/judge/rate/eval,
  in-process stub by default, `--base-url`/`VON_BASE_URL` → reqwest HTTP).
- **Pydantic fidelity findings** (all pinned by fixtures):
  1. `input_value` in validation errors renders the **original input dict** —
     fold mutations (pops/inserts) never appear in the message.
  2. Truncation rule: repr > 51 chars → first 25 + `...` + last 24 (52 total).
  3. Score criteria union errors emit **two** entries per bad item
     (`criteria.<i>.str` + `criteria.<i>.dict[str,any]`), str branch first.
  4. Direct-model construction vs question-dispatch changes the model name
     **and** whether the `type` key is in `input_value`.
- **Wire fidelity findings**: FastAPI `json_invalid` 422 bodies carry the
  Python scanner's char offset (`Expecting value`/`Extra data`) — reimplemented
  as a small scanner (`pyjson_scan.rs`); `HEAD` on GET routes → **405 with
  empty body + `allow` header** (Starlette does not auto-allow HEAD, so axum's
  GET-falls-through-to-HEAD had to be overridden per route); preflight OPTIONS
  → 200 `OK` `text/plain; charset=utf-8` with the Starlette method list and
  `max-age: 600`; `"Bearer "` (empty token) → "Unauthorized: invalid API key"
  while `Basic ...` → "Missing or invalid Bearer token".
- **Differential signal**: the unmodified pytest suite run against the Rust
  stub server via `VON_TEST_BASE_URL` gives **34/42 passing**; all 8 failures
  assert real-weight answer values (Stage 2's parity target), zero are
  contract/shape failures.

Known Stage 1 divergences (deliberate, revisited in Stage 3+): `--device`
accepts `auto|cpu` only (ORT EPs land in Stage 4); `--reload` rejected; CLI
help/error text is clap's, not click's; JSON numbers outside i64/u64 render as
f64 (unreachable via the HTTP domain); non-printable unicode beyond C0/DEL/U+2028/9
in `repr` strings is not escaped exhaustively (fixtures pin the JSON-bounded domain).
