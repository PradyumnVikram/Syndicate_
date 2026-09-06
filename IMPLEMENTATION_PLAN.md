# SEDS Implementation Plan

Actionable build plan derived from `md_files/seds-self-improving-architecture.md` and
`md_files/seds-implementation-spec.md`, reconciled against the **measured** capabilities of this
machine and the **probed** capabilities of the available model endpoints.

**Status:** draft for implementation. Not committed.
**Baseline measured/probed:** 2026-09-06. Re-verify §2 if the host or API keys change.

---

## 1. What we are building

> Given only a goal, a set of tools, and a way to evaluate success, the system must generate an
> agent architecture, run it, analyze where it fails, and iteratively improve its prompts, tools,
> memory, or orchestration strategy — demonstrated across multiple distinct domains with
> measurable gains in **accuracy, reliability, cost, and speed**.

Everything below is subordinate to that. Any component that does not move one of those four axes,
or does not produce evidence about them, is out of scope.

---

## 2. Verified environment baseline

### 2.1 Host (measured)

| Property | Measured value | Consequence |
|---|---|---|
| Docker | 29.7.2, **rootless** (apparmor, seccomp, cgroupns) | Sandboxing viable without root |
| `--network=none` | works | True zero-egress sandbox achievable |
| `--memory=512m` | **enforced** | Memory caps are real |
| `--cpus=0.5` | **enforced** — 5.36s vs 2.19s on a fixed loop | CPU caps are real |
| Cores | 24 | Not the binding constraint |
| RAM | 15GB total, **~11GB available** | Binding constraint on sandbox concurrency |
| Disk | 158GB free | Ample |
| Python | 3.10.12 | Target 3.10; no 3.11+ syntax |
| `openai` | 1.102.0 installed | Ready |
| `neatlogs` | 1.4.21 on PyPI, **not installed** | Phase A |
| `smallestai` | 5.12.0 on PyPI, not installed | Optional, domain-only |

### 2.2 Model endpoints (probed live — this is the dominant design constraint)

`.env` configures **two** OpenAI-compatible endpoints. The full catalog available to this project:

| Endpoint | Model | Type |
|---|---|---|
| `OPENAI_BASE_URL` (api.openai.com) | `gpt-5-nano` | reasoning chat |
| | `text-embedding-3-small`, `text-embedding-ada-002` | embeddings |
| `TENSORMUX_BASE_URL` (api.tensormux.com) | `glm-4-7-flash` | chat, vLLM 0.25.1 |

**There are exactly two chat models, both small. There is no frontier tier.** This invalidates any
plan that reserves a strong model for mutation proposals. See §7 B12 — it is now the top risk.

Probed per-model behaviour:

**`gpt-5-nano`** (resolves to `gpt-5-nano-2025-08-07`)
- `temperature: 0` → **HTTP 400**, *"does not support 0 with this model. Only the default (1)"*.
  **Deterministic decoding is not available on this model.**
- `seed` accepted without error, but no `system_fingerprint` returned → reproducibility unverifiable.
- Uses `max_completion_tokens`, not `max_tokens`.
- **Critical:** it is a reasoning model. With `max_completion_tokens: 2000` and no
  `reasoning_effort`, a trivial prompt consumed **all 2000 tokens as `reasoning_tokens` and returned
  empty content** with `finish_reason: "length"`. Setting `reasoning_effort: "minimal"` (or `"low"`)
  → `reasoning_tokens: 0`, clean `"ok"`, `finish_reason: "stop"`.
  **The broker MUST inject `reasoning_effort` on every nano call.** Without it, early runs silently
  return empty answers, score as failures, and burn the entire budget — the search would then
  optimize against pure noise.
- Reports `prompt_tokens_details.cached_tokens` → prompt-cache savings are measurable.
- **Batch API confirmed available** (`GET /v1/batches` → 200).

**`glm-4-7-flash`** (vLLM 0.25.1, `system_fingerprint: "vllm-0.25.1-c3468000"`)
- `temperature: 0` + `seed` both accepted → **this is the deterministic model.**
- Tool/function calling: accepted and exercised.
- Structured outputs (`response_format: json_schema`, `strict: true`): honored, emitted conforming JSON.
- Returns a separate `reasoning` field alongside `content`.
- `prompt_tokens_details: null` → no prompt-cache accounting.
- Batch API: assume unavailable (vLLM gateway).

**Resulting tier assignment** (replaces the earlier "cheap vs frontier" split):

| Tier | Model | Used for | Why |
|---|---|---|---|
| `deterministic` | `glm-4-7-flash` @ `temperature=0`, fixed `seed` | **all rollouts**, DoVer replay, paired evaluation | Only source of reproducibility; supports tools + structured outputs |
| `reasoner` | `gpt-5-nano` @ `reasoning_effort` set explicitly | mutation proposals, failure classification, SSF summarization | Reasoning helps code edits; Batch API for offline sweeps |

Putting rollouts on the deterministic model is what rescues B2 and B4. Do not invert this.

### 2.3 Third-party SDK surface (verified by inspecting the wheels)

**`neatlogs` 1.4.21** — OpenTelemetry-based LLM observability:
- `init(api_key, endpoint, workflow_name, instrumentations=["openai"], sample_rate, batch_size,
  flush_interval, disable_export, capture_logs, mask, pii_enabled, tracer_provider, isolate, ...)`
  — auto-instruments the OpenAI SDK; **supports fully offline operation** via `disable_export=True`.
- `@span(kind=...)`, kind ∈ `WORKFLOW | AGENT | CHAIN | TOOL | RETRIEVER | EMBEDDING | MCP_TOOL`.
  `AGENT` takes `role`/`goal`; `TOOL` takes `tool_name`.
- `normalize_span_v2()` → `TelemetrySpanV2`, canonical versioned schema
  (`TELEMETRY_CONTRACT_VERSION`, `TELEMETRY_SCHEMA_SHA256`).
- `InMemoryDiagnosticSpanExporter(max_spans=...)` — bounded, thread-safe, *"never sends data"*.
- `inject_trace_context()` / `extract_trace_context()` — cross-process stitching.
- Token capture from `response.usage`; force-sets `include_usage` on streams.
- Prompt registry with versioning: `create_prompt`, `save_as_version`, `get_prompt`, `list_prompts`,
  tags, `SystemPromptTemplate` / `UserPromptTemplate`.
- `MaskingSpanExporter` / `MaskContext`, `ByteLimitedLogExporter`.
- CLI: `python -m neatlogs doctor [--local | --probe] [--endpoint E] [--json]`.

> Note: neatlogs' OpenAI auto-instrumentation is written against the OpenAI SDK. Both endpoints are
> OpenAI-compatible, so a single instrumented `openai.OpenAI(base_url=...)` client covers both —
> but **verify that glm's response shape (`reasoning` field, null `prompt_tokens_details`) does not
> break `normalize_span_v2`.** Phase A task; fall back to explicit `@span` decorators if it does.

**`smallestai` 5.12.0** — **Waves (TTS)** + **Atoms (voice agents)**. *Not* an LLM provider.
Contributes nothing to the core loop. Optional third-party domain only; no hard dependency.

### 2.4 On `npx @neatlogs/wizard`

Exists (v0.1.7, published 2026-09-04): *"AI-powered CLI that instruments Neatlogs observability in
your project."* **Do not run it against this repo.** It instruments an existing codebase (this repo
has no source yet), it generates the default backend-export wiring (sandboxes need
`disable_export=True` + in-memory exporter because they run `--network=none`), and pointing an AI
codemod at a self-modifying agent archive is an unacceptable risk.

Use: `pip install neatlogs` → `python -m neatlogs doctor --local` → `doctor --probe --json`.
To crib its idiomatic wiring, run it once in a throwaway scratch dir and hand-port.

---

## 3. Reductions against the source specs

| Spec element | Verdict | Rationale |
|---|---|---|
| Code-space agent representation + archive tree | **Keep** | Core idea, correct |
| HGM clade selection (CMP + Thompson, α=0.6) | **Keep, + cold-start fallback** | Meaningless below ~30 nodes |
| Sandboxed execution + normalized trace IR | **Keep** | Largely adopt-not-build via neatlogs |
| SSF folding | **Keep**, over structured spans | Far more reliable than regex on stdout |
| Contract auditing | **Keep** | Cheap, deterministic, high value |
| DoVer counterfactual replay | **Keep** (Phase D) | Cache + deterministic tier make it sound (§7 B4) |
| APEX Layers 1–2 | **Keep** | Cheap, effective |
| APEX health formula constants | **Drop** | Unjustified magic numbers |
| Hyperagent self-modifying `meta_agent.py` | **Flag, default OFF** | Highest risk of search collapse (§7 B8) |
| "Network disabled" sandbox | **Keep literally** | Achievable via unix-socket broker (§4) |
| 50MB RAM / 0.5 CPU | **→ 768MB / 1.0 CPU** | 50MB cannot host a Python process |
| 1-hour per-task timeout | **→ 90s** | 1h makes the search wall-clock-infeasible |
| ≥95% CI promotion gate | **→ paired test at p<0.10** | 95% unaffordable at k≈20 (§7 B2) |
| "cheap + frontier" model tiers | **→ deterministic + reasoner** | No frontier model exists here (§2.2) |
| Source doc's performance-gain table | **Do not cite as expected results** | Unverifiable citations |

---

## 4. Architecture

The **broker** is the keystone. Credential custody, cost control, deterministic replay,
rate-limiting, and reward-hacking detection all resolve inside it. Build it first.

```
┌──────────────────────────────── HOST ────────────────────────────────┐
│  seds_manager.py                                                      │
│    Synthesizer ─▶ Executor ─▶ Diagnostic ─▶ Selector ─┐               │
│         ▲                                             │               │
│         └─────────────────── archive tree ◀───────────┘               │
│                                                                       │
│  LLM Broker  (holds all keys — none ever enter a container)          │
│    ├─ tier router: deterministic→glm | reasoner→nano                 │
│    ├─ param guard: injects reasoning_effort / temperature / seed      │
│    ├─ replay cache: sha256(resolved_model,messages,params,seed)       │
│    ├─ rate limiter (token bucket) + 429 backoff w/ jitter            │
│    ├─ spend meter + per-node hard kill-switch                        │
│    ├─ tool-call recorder (for deterministic DoVer replay)            │
│    └─ full call log (reward-hacking audit trail)                     │
│         ▲                                                             │
│         │ unix domain socket: /run/seds/llm.sock                      │
└─────────┼─────────────────────────────────────────────────────────────┘
          │ (bind mount)
┌─────────┼──── SANDBOX (one per rollout) ──────────────────────────────┐
│  docker run --network=none --read-only --user nonroot                 │
│             --memory=768m --cpus=1 --pids-limit=128                   │
│             --tmpfs /tmp --tmpfs /work                                │
│             -e PYTHONHASHSEED=0                                       │
│             -v /run/seds/llm.sock:/run/seds/llm.sock                  │
│             -v <node_out>:/out                                        │
│                                                                       │
│  agent code (candidate)                                               │
│    └─ neatlogs.init(disable_export=True,                              │
│                     exporter=InMemoryDiagnosticSpanExporter())        │
│         ├─ periodic flush + SIGTERM handler → /out/trace.json         │
│         └─ host ingests → trace_db → re-export to Neatlogs backend   │
└───────────────────────────────────────────────────────────────────────┘
```

The container has **no IP stack at all**. Credentials never cross the boundary. This satisfies the
source spec's "zero network access" literally, and is strictly stronger than an egress allowlist.

### 4.1 Directory layout

```
seds/
├── broker/            # socket server, tier router, param guard, cache, limiter, meter
├── runtime/           # in-sandbox: llm shim, tool impls, span setup, flush handler, entrypoint
├── executor/          # docker pool, preflight gates, trace ingestion
├── diagnostic/        # SSF, failure taxonomy, contract auditor, DoVer replay
├── synthesizer/       # mutation operators, prompt construction, structured-output schema
├── selector/          # archive tree, CMP/Thompson, Pareto, promotion gates
├── domains/           # TaskDomain packs (§8)
├── report/            # evidence generation
└── seds_manager.py    # outer loop (checkpointed, resumable)
agent_archive/         # git-versioned node codebases (single-writer commit queue)
data/
├── trace_db.sqlite
├── cache/             # broker replay cache (LLM + tool calls)
└── checkpoints/       # outer-loop resume state
```

---

## 5. Core contracts

Freeze these before parallel work starts.

```python
# seds/domains/base.py
@dataclass(frozen=True)
class Task:
    task_id: str
    inputs: dict[str, Any]
    reference: Any          # ground truth — HOST-SIDE ONLY, never mounted into a sandbox

@dataclass(frozen=True)
class Score:
    correct: bool
    partial: float          # 0.0–1.0
    detail: dict[str, Any]

@dataclass(frozen=True)
class ToolSpec:
    name: str
    json_schema: dict
    impl: Callable          # executed inside the sandbox; recorded for replay

@dataclass(frozen=True)
class TaskDomain:
    name: str
    goal: str
    tools: list[ToolSpec]
    train_tasks: list[Task]                  # visible to the search
    val_tasks: list[Task]                    # FROZEN; gates promotion + final selection
    evaluate: Callable[[Task, str], Score]   # runs HOST-SIDE, outside the container
```

```python
# seds/executor/result.py
@dataclass
class RolloutResult:
    node_id: str
    task_id: str
    answer: str | None
    score: Score
    spans: list[dict]        # neatlogs TelemetrySpanV2, normalized
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int    # nano bills these and they are invisible in output — track separately
    cost_usd: float
    wall_ms: int
    crashed: bool
    timed_out: bool
    error_log: str
```

`AgentEvent` from the source spec is **not** hand-rolled — `TelemetrySpanV2` is its realization.
Map `module` → span `kind`, `inputs`/`outputs` → attributes, `error_log` → span status.

### 5.1 Broker wire protocol

Newline-delimited JSON over the unix socket. Freeze early; executor and runtime are built against it
independently.

```
→ {"node_id","task_id","tier":"deterministic"|"reasoner",
   "messages":[...],"tools":[...],"max_tokens":<int>,"seed":<int>}
← {"ok":true,"cached":bool,"resolved_model":"glm-4-7-flash","response":{...},
   "input_tokens":N,"output_tokens":M,"reasoning_tokens":R,"cached_tokens":C,"cost_usd":F}
← {"ok":false,"error":"BUDGET_EXCEEDED"|"RATE_LIMITED"|"UPSTREAM_ERROR"|"TIMEOUT","detail":"..."}
```

The agent requests a **tier**, never a model ID. The broker owns model resolution and all sampling
parameters:

- `deterministic` → `glm-4-7-flash`, `temperature=0`, caller-supplied `seed`, `max_tokens`.
- `reasoner` → `gpt-5-nano`, **`reasoning_effort` injected**, `max_completion_tokens` (never
  `max_tokens`), no `temperature`.

Callers cannot override these. This makes model substitution a host-side policy knob and prevents
the §2.2 nano failure mode from ever reaching a rollout.

**Cache key must be `sha256(resolved_model, messages, tools, params, seed)`** — keyed on the
*resolved* model, not the requested tier, or a tier-policy change silently serves stale responses.

---

## 6. Phase plan

### Phase A — Substrate *(blocks everything)*

1. `.env` bootstrap (§11 blocker 1) + `python-dotenv`; `pip install neatlogs`.
2. `python -m neatlogs doctor --local`, then `--probe --json`. Record output.
3. Verify `normalize_span_v2` handles **both** response shapes (glm's `reasoning` field and null
   `prompt_tokens_details`; nano's `reasoning_tokens`). Fall back to explicit `@span` if it breaks.
4. **LLM broker**: socket server, tier router, **param guard**, replay cache, rate limiter + backoff,
   spend meter + per-node kill-switch, tool-call recorder, full call log.
5. Trace DB schema: `nodes`, `rollouts`, `spans`, `evaluations`, `llm_calls`, `tool_calls`.
   Index `(node_id, task_id)` and `(cache_key)`.
6. In-sandbox runtime shim: `seds.llm.call(tier, messages, ...)` → socket. Generated agent code
   **must not** import `openai` directly — enforce with a preflight AST check so cost accounting is
   unforgeable.
7. **`--replay-only` broker mode** (§7 B14): serve exclusively from cache, error on miss. Lets W2–W5
   develop and test at zero spend.
8. `agent_v0.py`: a **domain-parametric** ReAct/CoT agent — it reads `TaskDomain.goal` and
   `TaskDomain.tools` at runtime and is *not* specialized per domain. Anything else forfeits the
   task-agnostic claim. Must genuinely solve some tasks in every domain; a weak seed poisons the
   search.

**Exit criterion:** `agent_v0` solves ≥1 task end-to-end in a container, spans in the trace DB,
non-zero measured cost, and the same run replays from cache at $0.

### Phase B — Executor

9. Docker runner with the exact flags in §4.
10. Preflight gates (free, before any container starts): `py_compile` → `ruff` → import check → AST
    check for forbidden direct-SDK imports. Expect 20–30% of bad mutations killed at zero token cost.
11. Concurrent pool, **size 12** (memory-bound: ~11GB ÷ 768MB minus host headroom). Configurable.
12. **Trace-loss guard**: periodic span flush + SIGTERM handler with grace period before SIGKILL.
    Without this, traces are lost exactly for timeouts — the most diagnostic-worthy failures (§7 B13).
13. Trace ingestion: read `/out/trace.json`, stitch to parent via `extract_trace_context`, persist.

### Phase C — Evaluation & statistics

14. Minibatch evaluator: k=20 sampled train tasks per node evaluation.
15. **Paired comparison harness** — child vs. parent on the *same* task subset with the *same seeds*,
    on the `deterministic` tier. Largest lever on statistical power per dollar.
16. Promotion gate: paired bootstrap / McNemar at p<0.10, plus non-inferiority on cost.
17. Frozen val set (≥100 tasks), evaluated only at checkpoints and final selection. **Use the OpenAI
    Batch API (confirmed available, 50% discount) for these sweeps** — reasoner tier only. Never
    exposed to the Synthesizer's context.

### Phase D — Diagnostic

18. **SSF folding** over structured spans: retain tracebacks, diff hunks, tool-call args/errors;
    fold the rest into placeholder tags. Target ≥10× compression.
19. **Failure taxonomy classifier** — fixed labels: `tool_misuse`, `schema_violation`,
    `planning_loop`, `context_overflow`, `hallucinated_fact`, `format_error`, `timeout`, `crash`,
    `empty_output`. Aggregate per node into a histogram. *Highest value-per-token component* — it
    converts "it failed" into a targeted mutation.
    (`empty_output` exists specifically to catch §2.2-class regressions.)
20. **Contract auditor**: synthesize assertions from tool JSON schemas (arg types, required ordering,
    cross-step count consistency); return earliest failing step index. Deterministic, no LLM.
21. **DoVer replay**: restore conversation state at failure step *t*, splice patch, replay forward.
    Prefix replay is all cache hits — free and byte-identical **provided tool calls are replayed from
    the recorder too** (§7 B4). Require **n≥3** passing replays. Cap at 5 debug rounds.

### Phase E — Synthesizer

22. Mutation operators as an explicit typed menu:
    - `prompt_edit` — system prompt, exemplars, prohibition rules
    - `tool_edit` — wrappers, schema/description changes, validators
    - `memory_edit` — scratchpad, trace retrieval, episodic store
    - `orchestration_edit` — verifier stage, self-consistency, decomposer, router
    - `efficiency_edit` — tier downgrade, operator fusion, cascade with early exit
23. Emit via **structured outputs** (strict JSON schema, verified working on glm):
    `{operator, target_file, old_str, new_str, rationale}`.
24. **Best-of-N mutation sampling** (N≈5) with preflight filtering, then pick by cheap heuristic.
    Both models are small; compensate for weak single-shot proposals with volume — proposals are
    ~1% of total calls, so this is affordable (§7 B12).
25. Mutation context = parent code + folded traces + **failure-mode histogram (primary signal,
    ~200 tokens)** + ancestor performance log + remaining budget.
26. Diversity guard: reject children colliding on normalized AST + prompt hash. Optionally use
    `text-embedding-3-small` for semantic near-duplicate detection.
27. Track **valid-mutation rate**. Below ~50%, the mutation prompt is the defect, not the search.

### Phase F — Selector

28. Archive tree with the four HGM counters; lineage backprop on every outcome.
29. Expand-vs-evaluate rule `N_t^0.6 ≥ |T_t|`.
30. CMP + Thompson parent sampling with `τ(t)` schedule (flat early, sharp late).
31. **Cold-start fallback**: uniform / UCB-over-individual-score until every clade has ≥10
    evaluations; minimum-evaluations threshold for parent eligibility.
32. Pareto frontier over (accuracy, cost/task, p50 latency, reliability = 1 − crash rate).
    **Retain the frontier, not just the argmax** — it is the evidence for the multi-axis claim.
33. Final `ε`-percentile best-belief selection on the frozen val set. Do not replace with argmax.
34. Rollback ledger: every node is a git commit in `agent_archive/`, written through a
    **single-writer commit queue** (§7 B16). Prompt-side versions go in the Neatlogs prompt registry
    (`save_as_version` + tags) — Guardrail 4 and APEX L1/L2 storage for free.
35. **Outer-loop checkpointing** after every evaluation, so multi-hour searches resume (§7 B15).

### Phase G — Evidence

36. Report generator, per domain: v0 baseline vs. evolved on all four axes with CIs; failure-mode
    histogram before/after; lineage tree; cumulative cost curve; Pareto plot.
37. **Unseen-domain demo** (§8.1) — the direct evidence for "tasks it has never seen before".

### Sequencing discipline

Build a **thin vertical slice first**: one domain, seed agent, real sandbox, minibatch eval, *random*
mutation, flat archive — no CMP, no DoVer, no diagnostics. Get a measurable accuracy gain end-to-end.
Only then add the clever parts. Selection math is worthless until the evaluate→mutate→improve loop
demonstrably moves a number.

---

## 7. Bottleneck register

Ordered by likelihood of sinking the project.

**B12 — No frontier model for mutation. *(New; now the top risk.)***
Only `gpt-5-nano` and `glm-4-7-flash` are available (§2.2). Mutation quality is the engine of the
entire system, and both proposers are small models. If mutations are mostly noise, the search cannot
climb no matter how good the selection math is.
*Workarounds:* (a) heavily constrained typed operator menu + structured outputs — the model fills
slots rather than authoring free-form code; (b) **best-of-N sampling** (N≈5) with preflight
filtering — small models are cheap, so buy quality with volume; (c) keep diffs small and localized;
(d) feed the failure histogram rather than raw traces so the proposal task is nearly a classification
problem; (e) **instrument valid-mutation rate and net-improvement rate from day one** — if
improvement rate is indistinguishable from zero after the first vertical slice, escalate for a
stronger model before building Phases D–F on a foundation that cannot climb.

**B2 — Signal-to-noise in the selection statistic.**
At k=20, SE on accuracy is ~11 points; most real effects are smaller, so the search chases noise.
*Workarounds:* **run all rollouts on the `deterministic` tier (glm, `temperature=0` + fixed seed)** —
this removes decoding variance entirely and is the reason for the §2.2 tier assignment; paired
evaluation on identical subsets with identical seeds (removes task-difficulty variance, the largest
remaining component); report CIs everywhere; Batch API buys larger k at half price on the reasoner
tier; retain ε-percentile final selection.
*Residual risk:* if rollouts must run on nano, determinism is unavailable (temperature unsupported)
and variance rises sharply. Avoid.

**B1 — Evaluation cost.**
~800 evaluations × 20 tasks × ~15 calls ≈ 240k calls per domain-search.
*Workarounds:* (a) broker **replay cache** — re-evaluating an unchanged node costs $0; (b) **prompt
caching** on nano (`cached_tokens` is reported, so savings are measurable) — architect the system
prompt as an immutable prefix >1024 tokens with the task at the tail; (c) **Batch API** (confirmed
available, 50% off) for frozen-val sweeps, not the inner loop; (d) tier discipline — reasoner only
for mutation/diagnosis; (e) hard per-task token cap; (f) **`reasoning_effort` guard** — without it
nano burns 100% of the completion budget on invisible reasoning tokens (§2.2).

**B3 — Sandbox network. *(Resolved.)***
`--network=none` — no IP stack — plus a bind-mounted unix socket to the host broker. Credentials
never enter the container. Verified feasible on this host.

**B4 — DoVer replay determinism. *(Resolved, with a caveat I initially missed.)***
Prefix replay up to step *t* is all cache hits — free and byte-identical — **but only if tool calls
are replayed too.** The LLM cache alone is insufficient: any retrieval over a live index, clock read,
or stateful tool re-executes and diverges.
*Workaround:* the broker records **tool-call results as well as LLM calls**, keyed identically, and
replay serves both from the recorder. Combined with the `deterministic` tier, patch verification is
genuine rather than resampling luck. Still require n≥3 confirmations.

**B5 — Reward hacking on the eval function.**
A self-modifying agent *will* find that reading the answer key, swallowing assertions, or gaming a
lenient judge outscores reasoning. Assume it happens at least once.
*Workarounds:* `evaluate()` runs host-side, outside the container; `Task.reference` never mounted;
frozen val gates promotion; the broker call log makes "high score, suspiciously few LLM calls"
automatically detectable; per-node spend kill-switch.

**B13 — Trace loss on timeout. *(New.)***
`InMemoryDiagnosticSpanExporter` holds spans in memory. A container SIGKILLed at the 90s wall-clock
limit never flushes — so traces vanish precisely for timeout and hang failures, which are the ones
the Diagnostic module most needs.
*Workaround:* periodic flush (every N spans or T seconds) to `/out/trace.json`, plus a SIGTERM
handler with a grace period before SIGKILL. Treat a partial trace as valid input to SSF.

**B14 — No offline development mode. *(New.)***
Five workers iterating against live endpoints spend real money on every test run, and CI becomes
nondeterministic.
*Workaround:* broker `--replay-only` mode serving exclusively from cache and erroring on miss, plus a
committed fixture set of recorded calls. W2–W5 develop entirely offline; only W1 and integration runs
spend.

**B17 — Rate limits and retry semantics. *(New.)***
12 concurrent sandboxes × ~15 calls/task will hit RPM/TPM limits, and naive retries corrupt both cost
accounting (double-counted) and cache determinism (a retried call may differ).
*Workaround:* token-bucket rate limiter in the broker, exponential backoff with jitter on 429/5xx,
idempotency by cache key so a retry returns the first successful response, and retries metered once.
Surface `RATE_LIMITED` to the caller rather than hanging.

**B7 — Synthesizer context pressure.**
*Workarounds:* SSF folds **structured `TelemetrySpanV2` records**, not raw stdout, so retain/drop
operates on typed fields. `ByteLimitedLogExporter` bounds payloads at capture. Failure histogram is
the primary signal; raw excerpts secondary.

**B6 — CMP cold start.**
Clade stats are noise for the first few dozen nodes and early lucky nodes get entrenched.
*Workaround:* two-regime schedule (uniform → CMP at ≥10 evals/clade) + minimum-evaluations threshold
for parent eligibility.

**B8 — Self-modifying meta-agent → search collapse.**
A bad self-edit degrades every subsequent proposal, and lineage stats mask it because damage surfaces
downstream. Especially dangerous given B12.
*Workaround:* feature-flag **off** for the headline result; separate arm, versioned, same
non-regression gate.

**B9 — Wall-clock and concurrency.**
Memory-bound: ~11GB ÷ 768MB → pool of 12 on 24 cores. Inference is remote so the sandbox process is
thin (spec's 50MB cannot host Python; 2GB is over-provisioned).

**B10 — Mutation validity rate.**
*Workarounds:* typed operator menu + structured outputs (verified working on glm) kills malformed
diffs; `py_compile` + one-task smoke test catches semantic breakage before a full minibatch is spent.

**B15 — No resumability. *(New.)***
Multi-hour searches with no checkpoint lose everything on a crash, a rate-limit storm, or a host
restart.
*Workaround:* checkpoint outer-loop state (archive, counters, budget, RNG state) after every
evaluation to `data/checkpoints/`; `seds run --resume`.

**B16 — Archive write contention. *(New.)***
12 concurrent rollouts committing to `agent_archive/` will collide on the git index lock.
*Workaround:* single-writer commit queue; workers enqueue, one thread commits. Same pattern already
required for the trace DB.

**B18 — Python-level nondeterminism. *(New.)***
Even with `temperature=0`, dict/set iteration order and unseeded `random` make agent behaviour
irreproducible, defeating paired comparison.
*Workaround:* `PYTHONHASHSEED=0` in the container env, seed `random`/`numpy` from the task seed, and
forbid wall-clock reads in agent code via the preflight AST check.

**B19 — Host-side attack surface. *(New.)***
The container is well isolated, but `evaluate()`, the preflight AST check, and git operations all run
on the **host** against agent-produced content.
*Workaround:* `evaluate()` must never `eval`/`exec`/unpickle agent output; parse defensively. Treat
agent-authored strings as untrusted in every host-side path.

**B11 — Cost/speed axes unfalsifiable. *(Resolved.)***
Neatlogs captures per-call `input_tokens`/`output_tokens` into normalized spans and force-sets
`include_usage` on streams; the broker meters independently as a cross-check. **Track
`reasoning_tokens` separately** — on nano they are billed but invisible in output, so a
naive output-length cost model understates spend badly.

---

## 8. Domains

Three structurally different domains, each with cheap programmatic ground truth. Two is not enough
to support the task-agnostic claim; three that stress *different* mutation operators is the proof.

1. **Multi-hop QA** (HotPotQA subset) — retrieval, search orchestration, tool use. Exact-match eval,
   free. Stresses `orchestration_edit` / `memory_edit`.
2. **Code / data transformation** (synthetic ETL or SWE-style micro-tasks) — file editing, test-suite
   scoring, fully deterministic. Stresses `tool_edit` / `prompt_edit`.
3. **Structured extraction / API sequencing** against a mock service — schema-strict eval. Stresses
   reliability and tool-misuse repair.

*(Optional 4th: a `smallestai` Waves/Atoms workflow as a real external tool-sequencing domain.)*

### 8.1 Unseen-domain demo *(judging-critical)*

The brief says *"tasks it has never seen before."* Three hardcoded domains do not demonstrate that.
Add an explicit protocol:

- Adding a domain must require **only** a new `TaskDomain` instance — no framework changes.
  Enforce with a documented "add a domain in <N lines" recipe and a template.
- Hold one domain out entirely. Implement it **after** Phases A–F are frozen, then run the full
  search on it with **zero framework edits** and report the gain.
- Record the framework's git SHA before and after to prove nothing was changed. This is the single
  most convincing artifact the project can produce; budget time for it explicitly.

---

## 9. Metrics and evidence

Per domain, v0 baseline vs. best-belief evolved agent, on the frozen val set:

| Axis | Metric | Source |
|---|---|---|
| Accuracy | success rate + partial credit, 95% CI | `evaluate()` |
| Reliability | 1 − (crash + timeout + schema-violation + empty-output rate) | spans + `RolloutResult` |
| Cost | mean USD/task; total search cost; **reasoning-token share** | broker meter (cross-checked vs. spans) |
| Speed | p50 / p95 wall-ms per task | spans |

Also report: valid-mutation rate, **net-improvement rate per mutation** (the B12 early-warning
signal), cache hit rate, failure-histogram shift, archive size/depth, and the Pareto frontier.

---

## 10. Worker split

Contracts in §5 must be frozen before parallel work begins.

| Worker | Scope | Depends on |
|---|---|---|
| **W1 — Substrate** | `.env` bootstrap, neatlogs doctor + span-shape check, **broker** (router, param guard, cache, limiter, meter, tool recorder, replay-only mode), trace DB, runtime shim, domain-parametric seed agent | — *(blocks all)* |
| **W2 — Executor** | Docker pool, preflight gates, trace-loss guard, ingestion, concurrency | W1 socket protocol + `RolloutResult` |
| **W3 — Diagnostic** | SSF, failure taxonomy, contract auditor, DoVer | W1 span schema + tool recorder |
| **W4 — Synth + Select** | Operators, structured outputs, best-of-N, archive + commit queue, CMP/Thompson, Pareto, gates, checkpointing | W1 contracts |
| **W5 — Domains + report** | 3 `TaskDomain` packs, eval harness, evidence generator, unseen-domain template | `TaskDomain` contract |

W1 lands first. W2–W5 then run concurrently against frozen contracts, developing against
`--replay-only` fixtures.

---

## 11. Setup blockers

1. **`.env` does not propagate to worktrees.** It exists at
   `/home/azidozide/projects/syndicate_/.env` (containing `OPENAI_BASE_URL`, `OPENAI_API_KEY`,
   `TENSORMUX_BASE_URL`, `TENSORMUX_API_KEY`, `NEATLOGS_API_KEY`, `SMALLEST_API_KEY`) but is
   **gitignored and therefore untracked**, so it is absent from this worktree and will be absent from
   every worker worktree. Confirmed: no `.env` in the orchestrator worktree.
   **Fix:** a `make bootstrap` / `scripts/bootstrap.sh` step that symlinks the main checkout's `.env`
   into the current worktree, plus loading via an absolute-path fallback. Do this before spawning
   workers or all five fail on first API call.
2. **Neatlogs doctor never run.** `--local` → `--probe --json`, before building on the span schema.
3. **Do not run `npx @neatlogs/wizard`** against this repo (§2.4).

## 12. Open questions

- **Total search spend per domain in USD.** Sets B, and therefore achievable statistical power (B2).
- **Wall-clock deadline** for a search run — drives the `τ(t)` sharpening schedule.
- **Is a stronger model obtainable?** Given B12, this is the highest-leverage question in the
  document. If a larger model can be added to the TensorMux catalog for the `reasoner` tier only,
  it would cost little (mutations are ~1% of calls) and materially raise the ceiling on the whole
  system.
- **Is the Hyperagent self-modification arm (B8) a deliverable or a stretch experiment?**
