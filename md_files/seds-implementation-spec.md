# SEDS Self-Improving Agent System: Architectural Implementation Spec
This document provides a concise, high-density specification for implementing a task-agnostic, recursive self-improving agent framework based on the **Synthesizer-Executor-Diagnostic-Selector (SEDS)** architecture. It is designed to be directly parsed by code-generation agents (such as `Claude Code`) to generate an implementation plan.

---

## 1. System Architecture & Directory Structure

The system is designed as a self-referential Python workspace where the agent and its meta-optimizer exist as mutable codebase files.

```
/workspace/seds-runtime/
├── agent_archive/                  # Versioned tree of evolved agent files
│   ├── agent_v0.py                 # Initial Seed Agent (Chain-of-Thought baseline)
│   └── agent_v1.py                 # Autonomously mutated child agent code
├── data_system/
│   ├── validation_set.jsonl        # Frozen evaluation minibatches
│   └── trace_db.db                 # SQLite database for tracking execution logs
├── runtimes/
│   └── sandbox/                    # Ephemeral Docker container runtimes (no network)
└── seds_manager.py                 # Outer orchestrator containing SEDS modules
```

---

## 2. Core Python API & Class Interfaces

```python
import abc
from typing import Dict, Any, List, Tuple

class AgentEvent:
    """Portable Intermediate Representation (IR) of any execution event."""
    event_id: str
    step_index: int
    agent_name: str
    module: str           # e.g., "planning", "tool_call", "thought"
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    error_log: str        # Non-empty if step failed/crashed
    timestamp: float

class AgentSystem(abc.ABC):
    """Base class for all generated agent candidates."""
    @abc.abstractmethod
    def forward(self, task_info: Dict[str, Any]) -> Tuple[str, List[AgentEvent]]:
        """Executes task and returns the final answer along with normalized event traces."""
        pass

class SEDSSynthesizer:
    """Proposes code modifications using Clade-Metaproductivity (CMP)."""
    def __init__(self, trace_db_path: str):
        self.trace_db = trace_db_path

    def sample_parent(self, archive: List[Dict[str, Any]], tau: float) -> str:
        """Applies Thompson Sampling over Beta(tau * success_clade, tau * failure_clade) to select a parent path."""
        pass

    def mutate(self, parent_code_path: str, context: Dict[str, Any]) -> str:
        """Prompts Meta-Agent to edit task_agent.py and meta_agent.py (Hyperagent metacognition)."""
        pass

class SEDSExecutor:
    """Safely executes candidate code inside containerized sandboxes."""
    def run_sandbox(self, agent_code: str, test_tasks: List[Dict[str, Any]]) -> List[AgentEvent]:
        """Launches container, copies code, executes tests, and captures stdout/stderr/traces."""
        pass

class SEDSDiagnostic:
    """Performs failure attribution and active verification."""
    def fold_trajectory(self, events: List[AgentEvent]) -> List[AgentEvent]:
        """Applies SSF to fold low-signal logs and retain only diffs and tracebacks."""
        pass

    def audit_contracts(self, events: List[AgentEvent]) -> int:
        """Evaluates schema and policy contracts step-by-step to return the earliest causal error index."""
        pass

    def dover_replay(self, events: List[AgentEvent], error_idx: int, repair_patch: str) -> bool:
        """Restores state checkpoint at error_idx, injects repair, replays, and verifies success."""
        pass

class SEDSSelector:
    """Crates populations and constructs the Pareto frontier."""
    def optimize_workflow(self, agent_code: str) -> str:
        """Applies Operator Fusion, Model Substitution, and Cascade Filtering to code."""
        pass

    def get_best_belief(self, archive: List[Dict[str, Any]], epsilon: float) -> str:
        """Calculates regularized incomplete beta percentile to return the most stable final winner."""
        pass
```

---

## 3. The 8-Phase Evolutionary Loop: Step-by-Step

The outer loop of `seds_manager.py` executes these steps sequentially under a budget constraint (e.g., $B = 800$ evaluations):

```
[Phase 1: Seed v0] ─► [Phase 2: Decouple Check (N^0.6 >= |T|?)]
                          ├── YES ─► [Phase 3: Parent CMP Selection] ─► [Phase 4: Hyperagent Mutation] ─┐
                          └── NO  ───────────────► [Phase 5: Sandbox Evaluation] ◄──────────────────────┘
                                                          │
                                                    (Test fails?)
                                                          ├── YES ─► [Phase 6: SEDS SSF/DoVer Diagnostics] ─┐
                                                          └── NO  ──────────────────────────────────────────┤
                                                                                                            ▼
                                                  [Phase 8: Budget End] ◄── [Phase 7: Lineage Backprop] ◄──┘
                                                            │
                                                            ▼
                                                [Deploy Best-Belief Agent]
```

1. **Phase 1: Seeding**: Save a functional, single-file Chain-of-Thought `agent_v0.py` as root node $a_0$. Initialize individual and clade counters to $0$.
2. **Phase 2: Decoupling Check**: Evaluate $N_t^{0.6} \ge |\mathcal{T}_t|$. If `True`, branch to **Phase 3** (Expand). If `False`, branch to **Phase 5** (Evaluate an existing node).
3. **Phase 3: Parent Selection (CMP)**: Calculate Clade-Metaproductivity for all nodes. Draw Beta-distributed Thompson samples $S_C(a)$ scaled by scheduler $\tau(t)$. Select node $a^*$ with the maximum score.
4. **Phase 4: Metacognitive Mutation**: Prompt the selected parent's `meta_agent.py` to write file-level changes to its own directory. Allow the agent to write performance-tracking JSON logs and conditional strategic planners.
5. **Phase 5: Sandboxed Rollout**: Run the newly proposed agent on a validation minibatch inside a Docker container with network access disabled, a memory limit of 50MB, and a CPU quota of 0.5 cores. Normalize logs to `AgentEvent` IR.
6. **Phase 6: Closed-Loop Debugging**: On rollout failure, trigger the auto-debugger. Perform **Semantic Saliency Folding** to mask noise, execute **Contract Auditing** to pinpoint the failure step index, and run **DoVer replay** to verify the fix.
7. **Phase 7: Lineage Backpropagation**: Score task outcomes (success $= 1$, failure $= 0$). Increment individual node counters, then traverse recursively up to the root node, updating the clade counters ($n^C_{\text{success}}, n^C_{\text{failure}}$) of all ancestors.
8. **Phase 8: Final Selection**: When the budget is exhausted, evaluate the $\epsilon$-percentile of the posterior utility for each candidate in the archive. Deploy the highest-confidence **Best-Belief Agent** as the single-file production deliverable.

---

## 4. Pre-Deployment Acceptance Gates
*   **Non-Regression Test Gate**: Any promoted candidate code edit must be evaluated on a held-out validation set and pass a statistical significance test ($\ge 95\%$ confidence interval) against the baseline to control cumulative risk.
*   **Static Compilation & Formatting Check**: Candidate code must pass `py-compile` and static linting tests within the SEDS pipeline before entering sandboxed execution.
