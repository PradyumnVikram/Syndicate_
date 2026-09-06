# SEDS: System for Evidence-Based Decision Support

A framework for tool-using reasoning agents with formal verification.

## Status: Early Development (v0.1.0)

This project is in early stages with core domain contracts, broker routing, and partial agent implementations. See [progress docs](./docs) for detailed plans.

## Key Components

### Domains
- `seds.domains.base` - Core contracts (TaskDomain, Task, Score, ToolSpec, RolloutResult)
- `seds.domains.simple_qa` - Simple question answering example
- `seds.domains.multi_hop_qa` - Multi-hop question answering example

### Agent System
- `agent_v0` - Domain-parametric ReAct/CoT seed agent (Phase A item 8)
  - Reads `TaskDomain.goal` and `TaskDomain.tools` dynamically
  - Safe arithmetic expression evaluation (`safe_eval_arithmetic`)
  - Uses `seds.llm.deterministic()` for LLM calls via broker

### Runtime Layer
- `seds.runtime.llm` - LLM shim (calls broker via Unix socket)
  - `deterministic()` - glm-4-7-flash tier
  - `reasoner()` - gpt-5-nano tier

### Broker
- `seds.broker.server` - Unix socket server with tier routing
  - Tier routing: deterministic→glm-4-7-flash, reasoner→gpt-5-nano
  - Cache-based exit criterion: one real LLM call with cost > $0, then cached replay at $0

### Executor
- `seds.executor.runner` - DockerExecutor with security constraints
  - 90s timeout per task
  - 12x concurrent execution

### Verification (Phase D)
- `seds.phase_d.do_ver` - Checkpoint-Replay Counterfactual Verification
  - Captures state at failure steps
  - Splices in patches (instructions or tool arguments)
  - Replays to verify fixes (n≥3 passing replays required)

- `seds.phase_d.contract_auditor` - Tool call validation
  - Input schema compliance
  - Output schema compliance
  - State consistency
  - Rate limiting (burst detection)

### Report Generation
- `seds.report.comparison_generator` - Generates comparison reports

### Self-Improvement (Phases E-F)
- `seds.synthesizer` - Code-space mutation operator (Phase E)
  - Mutation operators: prompt_edit, tool_edit, memory_edit, orchestration_edit, efficiency_edit
  - Structured JSON schema for mutation requests

- `seds.selector` - Self-improving agent factory (Phase F)
  - Policy types: EPSILON, THOMPSON, EXP3, UCB
  - Pareto frontier optimization
  - Node event tracking and performance evaluation

## Quick Start

```bash
# Setup environment (creates/copies .env from src/project)
./scripts/bootstrap.sh

# Run smoke test (demonstrates exit criterion: real LLM call + cached replay)
python agent_v0.py
```

The smoke test in `agent_v0.py` demonstrates:
1. Domain-parametric agent with calculator tool
2. Broker-managed LLM calls with cost tracking
3. Exit criterion: one real call with nonzero cost, then cached replay at $0

## Documentation

See [docs/](./docs) for:
- Implementation plan
- User interaction plan
- Progress tracking
- neatlogs-doctor-output.md
