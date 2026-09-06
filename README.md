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

### Runtime
- `seds.executor.runner` - DockerExecutor with security constraints
- `seds.broker.server` - Unix socket server with tier routing

### Verification
- `seds.phase_d.do_ver` - Checkpoint-Replay Counterfactual Verification
- `seds.phase_d.contract_auditor` - Tool call validation

## Quick Start

```bash
# Setup environment
./scripts/bootstrap.sh

# Run smoke test
python agent_v0.py
```

## Documentation

See [docs/](./docs) for:
- Implementation plan
- User interaction plan
- Progress tracking
