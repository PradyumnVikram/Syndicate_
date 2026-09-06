# SEDS User Interaction & Propagation Plan

How an end user — someone who has a goal, some tools, and a way to judge success, but does not
want to touch the Synthesizer/Executor/Diagnostic/Selector internals — plugs into the system, and
how that input flows through every phase described in `IMPLEMENTATION_PLAN.md`.

**Principle:** the user's entire surface area with SEDS is authoring one `TaskDomain` (§5 of the
implementation plan). Everything below is either (a) ways to produce a `TaskDomain` with less
friction than hand-writing the dataclass, or (b) how that one object propagates through the eight
phases once it exists. There is no second interface to design — the frozen contract already *is*
the user-facing contract; this plan makes it ergonomic and traces it end to end.

---

## 1. What the user actually provides

Four things, matching the brief exactly ("a goal, available tools, and a way to evaluate success"):

| Input | Maps to | Notes |
|---|---|---|
| Goal | `TaskDomain.goal: str` | Natural language. No structure required. |
| Tools | `TaskDomain.tools: list[ToolSpec]` | Name + JSON schema + implementation. See §2 for the network/sandbox split. |
| Evaluation | `TaskDomain.evaluate: Callable[[Task, str], Score]` | Always runs host-side (existing guardrail, §5 of the implementation plan / bottleneck B5). See §3 for lower-friction alternatives to hand-writing this. |
| Task set | `TaskDomain.train_tasks` / `TaskDomain.val_tasks` | The instances to search and validate on. See §4. |

Nothing else is required to get a working seed agent — `agent_v0` is domain-parametric
(IMPLEMENTATION_PLAN.md §6 Phase A item 8) specifically so that supplying these four things is
sufficient to produce a first working baseline before any evolution happens.

---

## 2. Connecting tools

Tools split into two kinds, and the user should not have to know which kind before writing one —
the registration step classifies it.

### 2.1 Local tools (pure compute, no network/host resource)

Examples: a calculator, a regex extractor, a unit converter, a JSON reshaper. These are safe to
ship into the sandbox verbatim (they don't violate `--network=none`), so `ToolSpec.impl` executes
*inside* the container, exactly as the current contract comment says.

```python
ToolSpec(
    name="calculator",
    json_schema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    impl=safe_eval_arithmetic,   # pure function, no I/O
)
```

### 2.2 Host-proxied tools (anything needing network, credentials, or a real external system)

Examples: a real web-search API, a database query, an internal company API, an MCP server. These
**cannot** run inside a `--network=none` container. They must execute on the host and be reached
through the same broker socket the LLM calls already use (IMPLEMENTATION_PLAN.md §4/§5.1) —
this is a deliberate extension of the existing broker, not a new mechanism:

```
sandbox agent code
    │  seds.tools.call("web_search", {"query": "..."})
    ▼
unix socket  (same /run/seds/llm.sock pattern, or a sibling /run/seds/tools.sock)
    ▼
host-side tool proxy (holds the real credentials, does the real network call)
    │  records the call the same way the broker records LLM calls
    ▼
sqlite call log + replay cache, keyed on sha256(tool_name, args)
    ▼
result returned to the sandbox
```

This is why bottleneck B4 in the implementation plan already requires "the broker records
tool-call results as well as LLM calls" — DoVer replay determinism depends on it, and this section
just makes explicit that the *same* recorder is the mechanism that lets user-supplied network tools
exist at all under the network-isolation guardrail. One registration call classifies a tool:

```python
seds.tools.register(
    name="web_search",
    json_schema={...},
    impl=my_real_search_function,   # runs on the host, never shipped into the container
    network=True,                    # -> routed through the host-side proxy instead of bundled
)
```

If the user points at an **MCP server** instead of writing a Python function, the registration step
introspects the server's tool list and auto-generates one `ToolSpec` per exposed tool (name +
schema come straight from the MCP `list_tools` response); invocation is proxied exactly like any
other network tool, since an MCP server is itself an external network endpoint.

### 2.3 Safety note carried over from the implementation plan

Tool `impl`s — local or proxied — are still subject to the preflight AST check (item 6, Phase A):
user-supplied local tools that try to import `openai`/`requests`/`httpx` directly are rejected at
registration time, same as generated agent code, because a "local" tool trying to reach the network
directly would silently defeat the sandbox boundary.

---

## 3. Connecting the evaluation metric

The frozen contract is a callable, which is correct for power users but a needless barrier for the
common case. Three tiers, all compiling down to the same `Callable[[Task, str], Score]`:

**Tier 1 — named strategy (zero code).** For the common cases, the user picks a string and the
system supplies the callable:

```python
evaluate = seds.evaluators.named("exact_match")       # str(answer) == str(task.reference)
evaluate = seds.evaluators.named("contains")           # task.reference in answer
evaluate = seds.evaluators.named("numeric_tolerance", tol=0.01)
evaluate = seds.evaluators.named("test_suite", suite_path="tests/")   # for code-transform-style domains
evaluate = seds.evaluators.named("llm_judge", rubric="...")           # last resort, see caveat below
```

**Tier 2 — small Python function (most users).** Exactly the shape already frozen:

```python
def evaluate(task: Task, answer: str) -> Score:
    correct = normalize(answer) == normalize(task.reference)
    return Score(correct=correct, partial=1.0 if correct else 0.0, detail={})
```

**Tier 3 — full custom pipeline** (multi-metric, external test harness, human-in-the-loop queue) —
the user just hands over any callable matching the signature; SEDS doesn't need to know what's
inside it.

**Caveat carried over from the implementation plan:** `llm_judge` re-introduces cost and variance
into the very signal the selector optimizes against (bottleneck B2), and is a standing invitation
for the reward-hacking failure mode in B5. It's offered because some domains have no cheaper
alternative, not because it's recommended — the onboarding flow should warn when it's selected.

In every tier, `evaluate` is invoked **only** by the host-side evaluation harness (Phase C), never
inside a sandbox, and `Task.reference` is never serialized into anything the sandboxed agent can
read. This is an existing, non-negotiable guardrail, not something the onboarding layer can loosen.

---

## 4. Connecting the task set

Three ways to produce `train_tasks`/`val_tasks`, in increasing order of effort:

1. **Inline list** — for small, hand-written task sets (the smoke-test pattern `agent_v0` already
   uses).
2. **File loader** — `seds.tasks.from_jsonl(path, reference_field="answer")`: one JSON object per
   line becomes one `Task`, with the ground-truth field pulled out into `Task.reference` and
   everything else left in `Task.inputs`. Covers the common case (a QA dataset, a set of tickets, a
   batch of documents) without writing a parser.
3. **Programmatic generator** — a callable that yields `Task` objects, for domains that synthesize
   tasks (e.g. randomly parameterized arithmetic, procedurally generated ETL fixtures).

**Split enforcement, not just convention:** the registration step (§5) rejects a `TaskDomain` whose
`val_tasks` overlap `train_tasks` by `task_id`, and warns if `val_tasks` is smaller than ~50
instances — too small a frozen set defeats its own purpose per bottleneck B2.

---

## 5. Registration: where user input enters the system

One call is the actual entry point:

```python
domain_id = seds.register_domain(domain)     # domain: TaskDomain
```

This does, synchronously, before anything expensive happens:

1. **Schema validation** — every `ToolSpec.json_schema` is valid JSON Schema; `evaluate` is
   callable; `train_tasks`/`val_tasks` are non-empty and disjoint (§4).
2. **Tool classification** — each tool is marked local or host-proxied (§2), and local tools are
   run through the same preflight AST check as generated code.
3. **Snapshotting** — `train_tasks`/`val_tasks` are hashed and written into the trace DB
   (`nodes`/`evaluations` tables already defined in Phase A) under a new `domain_id`, so the task
   set is provably frozen for the duration of the run — nobody, including a later self-modification,
   can silently swap in easier tasks mid-search.
4. **Budget & deadline capture** — the user supplies `budget_usd` and/or `wall_clock_deadline` here;
   this is what drives the `τ(t)` exploration/exploitation schedule (Phase F, item 30) and the
   per-node spend kill-switch already built into the broker (Phase A item 4).

Everything from here on is internal propagation — the user is not involved again until §7.

---

## 6. Propagation through the eight phases

```
 registered TaskDomain (domain_id)
        │
        ▼
[Phase 1: Seed]  agent_v0(domain.goal, domain.tools, task) ──► first RolloutResult
        │                                                         (uses domain.evaluate,
        │                                                          host-side, §3)
        ▼
[Phase 2-4: Synthesizer]  mutation prompts built from:
    domain.goal, domain.tools (schemas), failure-mode histogram
    — domain CONTENT flows in verbatim; the mutation MECHANISM is domain-agnostic
        │
        ▼
[Phase 5: Executor]  sandbox launched per rollout:
    - domain.goal + domain.tools (local ones bundled, proxied ones registered with the broker)
    - agent_vN.py is still domain-parametric — no per-domain code branch ever exists in the harness
        │
        ▼
[Phase 6: Diagnostic]  SSF / failure taxonomy / contract auditor operate on the normalized trace;
    contract auditor's assertions are SYNTHESIZED FROM domain.tools[i].json_schema — this is the
    one diagnostic component whose behavior is directly parameterized by user input, by design
        │
        ▼
[Phase 7: Lineage backprop]  scored via domain.evaluate — domain-agnostic bookkeeping,
    domain-specific score
        │
        ▼
[Phase 8: Selector]  archive/CMP/Pareto keyed by domain_id, so one SEDS installation can hold
    many users' domains side by side without cross-contamination of lineage statistics
        │
        ▼
   Best-belief agent for THIS domain_id
```

The load-bearing design fact that makes this whole plan possible: **every phase after registration
takes `domain.goal`/`domain.tools`/`domain.evaluate` as opaque parameters, never as something to
special-case.** `agent_v0` already had to be built this way (Phase A, item 8, "domain-parametric,
not hardcoded to any domain") specifically so this propagation holds all the way through — and the
unseen-domain protocol in §8.1 of the implementation plan is the direct test of this claim: a domain
introduced after the framework is frozen must produce a working, improvable agent with zero
framework-file changes.

---

## 7. What comes back to the user

1. **Live progress** (optional, during the run): `seds.status(domain_id)` — current archive size,
   best score so far, spend against budget, wall-clock remaining.
2. **Final deliverable**: a single-file `agent_vN.py`, the Best-Belief Agent (Phase 8), runnable
   standalone against `domain.goal`/`domain.tools` without the SEDS harness — this is the actual
   production artifact the user takes away.
3. **Evidence report** (Phase G): baseline-vs-evolved on all four axes (accuracy, reliability, cost,
   speed) with confidence intervals, the failure-histogram shift, and the Pareto frontier — so the
   user can pick a different point on the frontier (e.g. cheaper-but-slightly-less-accurate) instead
   of being forced to accept whatever the single "best" pick was.
4. **Full lineage**, if wanted: the entire `agent_archive/` git history for this domain, so the user
   can inspect *why* the agent looks the way it does, not just what it outputs.

---

## 8. Interfaces summary (concrete, minimal)

```python
import seds

domain = seds.TaskDomain(
    name="invoice-extraction",
    goal="Extract vendor, amount, and due date from an invoice PDF.",
    tools=[
        seds.ToolSpec(name="pdf_to_text", json_schema={...}, impl=local_pdf_parser),
        seds.tools.from_mcp("my-invoice-mcp-server"),   # auto-discovered, host-proxied
    ],
    train_tasks=seds.tasks.from_jsonl("invoices_train.jsonl", reference_field="fields"),
    val_tasks=seds.tasks.from_jsonl("invoices_val.jsonl", reference_field="fields"),
    evaluate=seds.evaluators.named("exact_match_fields"),
)

domain_id = seds.register_domain(domain, budget_usd=20, wall_clock_deadline_minutes=90)
seds.run(domain_id)                     # blocks or backgrounds; see status via seds.status(domain_id)
report = seds.report(domain_id)         # Phase G evidence
best_agent_path = seds.export(domain_id)  # the single-file deliverable
```

This is the entire user-facing surface. Nothing else needs to be designed — the rest of the system
(Phases 1-8) is what workers W1-W5 are already building, and this document only formalizes the door
into it and traces where the four inputs go once they're through it.
