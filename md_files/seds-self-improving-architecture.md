# Architectural Specification: SEDS-HGM Self-Improving Agent Factory

This document provides a comprehensive systems-engineering blueprint and a detailed tooling specification for deploying a **Synthesizer-Executor-Diagnostic-Selector (SEDS)** agent factory. By integrating the theoretical rigor of the **Huxley-Gödel Machine (HGM)** with the open-ended mutability of **Hyperagents (DGM-H)**, this architecture treats agent configurations not as static configurations, but as an evolving, branching lineage of executable code [17, 23].

---

## 1. Executive Summary & Design Philosophy

The traditional development of LLM agents relies heavily on manual prompt engineering, static routing, and rigid coordination frameworks [1, 2]. These manual interventions introduce human design bottlenecks and fail to adapt dynamically when the underlying task distribution or model capabilities shift [3, 4]. 

**Automated Design of Agentic Systems (ADAS)** recasts this development cycle as an optimization search over an unbounded, non-convex program space [5, 6]. By representing the agent’s prompt, workflow, tools, and memory structures entirely in **Turing-complete code space (e.g., Python)**, the search space is maximally expressive—capable of representing and discovering arbitrary multi-agent configurations, custom memory databases, and reasoning paths [6, 7].

To manage this open-ended search without collapsing into local optima or overestimating brittle designs, we present a unified system built on **SEDS** and **Huxley-Gödel Lineage-Based Selection** [15, 23].

---

## 2. The SEDS Master Architecture

The **SEDS Framework** organizes the self-evolution lifecycle into four highly decoupled, parallelized functional modules operating over a shared, containerized runtime substrate [15, 31, 34, 44].

```
                     [Goal, Tools, Evaluation Metric]
                                    │
                                    ▼
       ┌────────────────► [SEDS Synthesizer Module] (Turing-complete Python code)
       │                            │
       │                            ▼
       │                  [SEDS Executor Module] (Sandboxed runtimes)
       │                            │
       │                            ▼
       │                 [SEDS Diagnostic Module] (Causal tracing & DoVer replays)
       │                            │
       │                            ▼
       └─────────────── [SEDS Selector Module] (Pareto frontier updates)
                                    │
                                    ▼
                        [Optimized Production Agent]
```

### 2.1. Synthesizer Module (The Architecture Proposer)
*   **Role**: Acts as the meta-agent, proposing novel configurations and compiling them into executable Python classes [1, 9].
*   **Mechanism**: Reads the current active tree of evolved agents, ancestral performance histories, and the remaining compute budget [23, 235]. It models selection as an infinite-armed bandit problem, utilizing **Thompson Sampling** on lineage statistics to select parent templates for mutation, avoiding greedy over-reliance on lucky, brittle candidates [15, 17].

### 2.2. Executor Module (The Isolated Sandbox)
*   **Role**: Isolates, executes, and validates generated agent code against target task minibatches [18, 22].
*   **Mechanism**: Spawns ephemeral, containerized runtimes (e.g., non-root Docker sandboxes) with zero network access and strict resource boundaries [22, 23]. The Executor supports **Dynamic Agent Generation**—parsing intermediate state variables to instantiate specialized subagents only on-demand, which reduces average token overhead and network latency [4, 26].

### 2.3. Diagnostic Module (Causal Failure Attribution)
*   **Role**: Pinpoints, debugs, and patches failures when a candidate agent underperforms or throws compilation errors [29, 31].
*   **Mechanism**: Applies **Semantic Saliency Folding** to compress long, noisy execution logs [30]. It executes **Global and Dynamic Contract Auditing** to causally localize the earliest unrecoverable error step, generates orchestrator-level interventions, and replays them forward in-situ via **DoVer counterfactual verification** before saving code-level patches [31, 32, 34].

### 2.4. Selector Module (Multi-Objective Optimization)
*   **Role**: Manages, prunes, and curates the active population of agent designs [49, 60].
*   **Mechanism**: Compiles candidate agents across multiple competing axes (Accuracy, Speed, Cost, and Reliability) [49]. It applies structural query optimization rewrites—such as **Same-Type Operator Fusion** to merge redundant adjacent LLM nodes, or **Model Substitution** to dynamically route simpler tasks to cheaper models—and maps the optimal configurations to a multi-dimensional **Pareto frontier** [48, 60, 61].

---

## 3. The 8-Phase Evolutionary Loop

Under the hood, SEDS coordinates a continuous, unrolled feedback loop to evolve the agent codebase recursively.

### Phase 1: Archive Initialization (Seeding the Forest)
*   **Operation**: The system initializes the **Agent Archive** (represented as an evolutionary tree $\mathcal{T}$) and seeds the root node ($a_0$) with a basic, fully-functional human-written Python agent class (e.g., a simple Chain-of-Thought or Self-Refine loop) [234, 235].
*   **Registers**: Node $a_0$ is initialized with four distinct trackers:
    *   $n_{\text{success}}(a_0) = 0$ (Individual task successes)
    *   $n_{\text{failure}}(a_0) = 0$ (Individual task failures)
    *   $n^C_{\text{success}}(a_0) = 0$ (Total successes in $a_0$'s clade)
    *   $n^C_{\text{failure}}(a_0) = 0$ (Total failures in $a_0$'s clade)

### Phase 2: Decoupling Decisions (To Expand or to Evaluate?)
*   **Operation**: Rather than wasting compute by running every new mutant agent on the entire evaluation dataset, the system uses an infinite-armed bandit rule to decide whether to spend its budget on **Expansion** (generating a new architecture) or **Evaluation** (testing an existing agent on one more task to gather cleaner statistics) [233, 244].
*   **Math**: At search step $t$, the system evaluates the global branching inequality:
    $$N_t^\alpha \ge |\mathcal{T}_t|$$
    where $N_t$ is the total number of individual task evaluations completed across all agents, $|\mathcal{T}_t|$ is the current count of unique agent codebases in the tree, and $\alpha$ is a widening parameter (typically set to $0.6$) [258, 276].
    *   *If True*: Trigger **Expansion** (Phase 3).
    *   *If False*: Trigger **Evaluation** (Phase 5).

### Phase 3: Parent Selection via Clade-Metaproductivity (CMP)
*   **Operation**: To select which agent codebase to expand, the system bypasses individual scores—which often suffer from the **Metaproductivity-Performance Mismatch (MPM)**—and instead evaluates clades [17, 235].
*   **Math**: It computes **Clade-Metaproductivity (CMP)**, aggregating the success rates of an agent's entire descendant subtree ($C(a)$) [15, 235]:
    $$\widehat{\text{CMP}}(a) = \frac{n^C_{\text{success}}(a)}{n^C_{\text{success}}(a) + n^C_{\text{failure}}(a)}$$
    Selection is sampled probabilistically via **Thompson Sampling** over Beta distributions, scaled by an increasing exploration-exploitation scheduler $\tau(t)$ [254, 255]:
    $$a^* \sim \text{Beta}\left(\tau(t) \cdot (1 + n^C_{\text{success}}(a)), \; \tau(t) \cdot (1 + n^C_{\text{failure}}(a))\right)$$
    This curve is highly flattened early in the 24-hour budget to encourage the exploration of diverse, modular abstractions (stepping stones), and sharpens toward infinity near the deadline to exploit elite, proven lineages [15, 235, 254].

### Phase 4: Metacognitive Mutation (The Hyperagent Propose Step)
*   **Operation**: The selected parent agent ($a^*$) is instantiated as a self-referential **Hyperagent** workspace, containing task-solving files (`task_agent.py`) and self-modification files (`meta_agent.py`) in an editable repository [23].
*   **Mechanism**: The meta-agent reads its own source code, the performance logs of its ancestors, and its remaining iteration budget [23, 322]. It writes direct, file-level modifications to **both** files, allowing the system to improve its own search, strategic planning, and performance logging infrastructure over time (metacognitive self-modification) [23, 320].

### Phase 5: Gated Validation & Sandboxed Execution
*   **Operation**: The compiled child codebase, $a_{\text{child}}$, is loaded inside a temporary, isolated Docker container [174, 182].
*   **Verification**: The Executor launches the child against validation tasks, enforcing strict CPU, memory, and wall-clock timeout gates (e.g., hard 1-hour timeout per task) [182, 279]. The sandbox captures a structured, sequential trace of thoughts, responses, tool calls, and observations [38, 592].

### Phase 6: Closed-Loop Automated Debugging (SEDS Diagnostics)
*   **Operation**: If $a_{\text{child}}$ fails to compile or crashes, the system triggers the automated repair loop for up to **5 debugging rounds** [110, 629]:
    1.  **Saliency Folding**: Redundant observations are masked, retaining only code patches (diff headers) and traceback error keywords [30].
    2.  **Contract Auditing**: OpenTelemetry-style spans are audited against synthesized tool schemas and policy contracts to isolate the earliest failure step [31, 32].
    3.  **Counterfactual Replay**: The sandbox environment state is restored to the failure checkpoint, a patch is applied, and the execution is replayed forward [34, 35]. If verified, the codebase is registered as a new node in the archive tree [22, 234].

### Phase 7: Lineage Outcome Propagation
*   **Operation**: Once an evaluation is completed (returning success $Z \in \{0, 1\}$), the outcome is backpropagated up the ancestral tree [255, 278].
*   **Update**: The system recursively increments the clade registers of the parent node and all of its ancestors up to the root ($a_0$) [255, 278]:
    $$n^C_{\text{success}}(a_{\text{ancestor}}) \leftarrow n^C_{\text{success}} + Z$$
    This instantly updates the selection probabilities of all lineages for subsequent iterations [255, 278].

### Phase 8: Final Candidate Selection (The Best-Belief Agent)
*   **Operation**: When the total computational or time budget is exhausted, the search terminates [257, 278]. 
*   **Math**: To prevent selecting lucky, overfitted outliers, the Selector calculates the **$\epsilon$-percentile of the posterior utility** for all nodes using the regularized incomplete beta function, returning the single most statistically robust, non-overfitted agent to deploy [257, 278]:
    $$a_{\text{final}} = \arg\max_{a \in \mathcal{T}_B} I_{\epsilon}\left(1 + n_{\text{success}}(a), \; 1 + n_{\text{failure}}(a)\right)$$

---

## 4. Multi-Axis Co-Evolution (The APEX Strategy)

To achieve high reliability, SOTA self-improving systems must avoid single-axis optimization (e.g., only optimizing prompt instructions) and instead deploy **APEX multi-axis co-evolution**, optimizing three separate layers of the agent harness simultaneously from a shared trace database [8, 16]:

$$H_{\text{APEX}} = \underbrace{\min(0.30, |\Delta| \times 0.10)}_{\text{Layer 1: Harness Review}} + \underbrace{\min(0.40, |Q| \times 0.07)}_{\text{Layer 2: Principle Distillation}} + \underbrace{\text{score}(\tau^*) \times 0.30}_{\text{Layer 3: Topology Evolution}}$$

```
                      [Shared Trace Database]
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
 [Layer 1: Harness]    [Layer 2: Principles]   [Layer 3: Topology]
 Failure mode patching  Success trace distillation  DAG orchestration search
  (Prohibition Rules)    (Behavior Guidelines)      (Message Routing Paths)
         └───────────────────────┬───────────────────────┘
                                 ▼
                     [APEX Health Aggregation]
```

1.  **Layer 1: Harness Review ($\Delta$)**: Analyzes failed execution trajectories to identify systemic failure patterns. It automatically inserts explicit, strict **prohibition rules** into the master system prompt to prevent downstream model crashes [8, 24].
2.  **Layer 2: Principle Distillation ($Q$)**: Synthesizes highly successful traces to extract abstract, **reusable behavioral guidelines** that are dynamically injected into the active runtime context [8, 24].
3.  **Layer 3: Topology Evolution ($\tau^*$)**: Uses genetic programming or Monte Carlo Tree Search over directed acyclic graphs (DAGs) to mutate the multi-agent communication topology [8, 24].
*   *Co-Evolution Constraint*: Optimizing workflow topologies (L3) without an established prompt harness foundation (L1) degrades net performance. Because these layers target orthogonal failure modes, co-evolution must occur jointly to prevent coordination collapse [24, 25].

---

## 5. Detailed Tooling Specification

To implement this architecture safely and efficiently, SEDS separates tools into three categories: **Infrastructure & Sandbox Tools** (system-level execution), **Agent-Facing Tools** (APIs exposed to the agent), and **Diagnostic Tools** (debugging and causal tracing).

```
┌────────────────────────────────────────────────────────────────────────┐
│                      1. INFRASTRUCTURE & SANDBOX TOOLS                 │
│  - temporary Docker sandboxes   - OpenTelemetry GenAI spans            │
│  - bash compiler checkers       - CPU/RAM/wall-clock limiters          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        2. AGENT-FACING TOOLS                           │
│  - bash execution API           - file editor (view/insert/replace)    │
│  - dynamic subagent generator   - memory search & retrieval engines    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        3. DIAGNOSTIC & AUDIT TOOLS                     │
│  - SSF log compression          - AgentRx relational contract audit    │
│  - DoVer counterfactual replay  - TrajAudit diff-header parsers        │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.1. Infrastructure & Sandbox Tools
These tools manage the secure compilation, execution, and instrumentation of candidate agents [111, 472].
*   **Docker-Sandbox Executor**: Provisions lightweight, ephemeral Docker runtimes [174, 182]. Enforces a strict read-only root filesystem, disables external network routing (to protect data privacy and block malicious outbound scripts), and binds a maximum RAM allocation (e.g., 512MB) and hard runtime timeout [182, 339].
*   **OpenTelemetry GenAI Span Exporter**: Normalizes all internal agent events (thoughts, tool calls, model outputs, and handoffs) into standard OpenTelemetry-compatible GenAI span schemas [38, 49]. This isolates trace capture from the underlying framework (e.g., LangGraph, CrewAI, or raw ReAct) [38, 54].
*   **Syntax & Compiler Checker**: Executes a zero-cost local Python syntax parser (`python -m py_compile`) on proposed code modifications before sandboxing, immediately filtering out syntax errors or broken imports without consuming model or evaluation tokens [22, 221].

### 5.2. Agent-Facing Tools
These represent the specific programmatic APIs exposed to the agent inside its execution sandbox [221, 322].

#### Tool 1: `bash` (Interactive Terminal)
Allows the agent to run commands in a stateful shell environment to test scripts, run compilations, or execute test suites [279, 344].
*   **Schema**:
```json
{
  "name": "bash",
  "description": "Run commands in a stateful bash shell. State persists across sequential invocations. Internet access is disabled. Long-lived commands must be run in the background (e.g. 'sleep 10 &'). Avoid commands with excessive output.",
  "input_schema": {
    "type": "object",
    "properties": {
      "command": {
        "type": "string",
        "description": "The exact bash command to execute in the terminal."
      }
    },
    "required": ["command"]
  }
}
```

#### Tool 2: `editor` (File-System Manager)
Provides atomic, structured, and stateful operations to read, write, and patch files in the repository, preventing the model from overwriting complete files and losing context [345].
*   **Schema**:
```json
{
  "name": "editor",
  "description": "Stateful tool for viewing, creating, and modifying files in the workspace. Supports local rollback via undo_edit.",
  "input_schema": {
    "type": "object",
    "properties": {
      "command": {
        "type": "string",
        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
        "description": "The specific file operation to run."
      },
      "path": {
        "type": "string",
        "description": "Absolute path to the target file or directory."
      },
      "file_text": {
        "type": "string",
        "description": "Required only for 'create'; contains the full starting content of the file."
      },
      "old_str": {
        "type": "string",
        "description": "Required only for 'str_replace'. The exact block of consecutive lines to search and replace. Must match exactly once."
      },
      "new_str": {
        "type": "string",
        "description": "Required only for 'str_replace' (replacement text) and 'insert' (text to inject)."
      },
      "insert_line": {
        "type": "integer",
        "description": "Required only for 'insert'. The line number after which new_str will be injected."
      },
      "view_range": {
        "type": "array",
        "items": { "type": "integer" },
        "description": "Optional line range for 'view' (e.g., [10, 25]). Index starts at 1."
      }
    },
    "required": ["command", "path"]
  }
}
```

#### Tool 3: `subagent_generator` (Dynamic Orchestration)
Allows the agent to dynamically spawn specialized sub-agents with custom system prompts and models, delegating heavy semantic tasks (like ensembling or reviewing) on the fly [4, 25].
*   **Schema**:
```json
{
  "name": "subagent_generator",
  "description": "Dynamically instantiates a specialized sub-agent for a single task. The subagent is destroyed immediately upon returning its output.",
  "input_schema": {
    "type": "object",
    "properties": {
      "role": {
        "type": "string",
        "description": "The persona/specialization of the subagent (e.g., 'Verifier', 'Critic', 'Ensemble Master')."
      },
      "system_prompt": {
        "type": "string",
        "description": "The exact system instructions shaping the subagent's logic and behavioral constraints."
      },
      "task_input": {
        "type": "string",
        "description": "The specific semantic query or data payload that the subagent must process."
      },
      "model_override": {
        "type": "string",
        "description": "Optional model name override (e.g., 'gpt-4o-mini' for fast formatting, 'claude-3-5-sonnet' for complex reasoning)."
      }
    },
    "required": ["role", "system_prompt", "task_input"]
  }
}
```

### 5.3. Diagnostic & Audit Tools
These tools run inside the **SEDS Diagnostic Module** to trace, isolate, and debug execution-level errors [29, 31].
*   **Semantic Saliency Folder (SSF)**: Uses regular expressions to match code diff markers (e.g., `--- a/`, `+++ b/`, and hunk boundaries like `@@ -N,M +P,Q @@`) alongside a failure-indicative keyword index (e.g., `Exception`, `Traceback`, `AssertError`, `FAIL`) [597]. Text blocks matching neither pattern are compressed into lightweight JSON placeholder tags, preventing long-context attention degradation during model analysis [30, 597].
*   **AgentRx Contract Auditor**: Normalizes multi-agent traces and compares them step-by-step against two sets of synthesized validation assertions [32, 61]:
    *   *Global Constraints* ($C_G$): Synthesized once from tool schemas and domain policies (e.g., *"The model must verify user credentials prior to invoking delete_database"*).
    *   *Dynamic Constraints* ($C_D$): Generated on-the-fly to enforce logical consistency across execution steps (e.g., *"The output element count reported in step $t$ must equal the length of the list returned by the tool at step $t-1$"*) [32].
*   **DoVer Checkpoint-Replay Harness**: Wraps the agent's chat manager with a checkpoint-aware interceptor [202]. It captures state variables (conversation history, tool parameters, file state) at each step. On failure, it restores the environment state exactly at step $t$, splices in the modified instruction or argument proposal, and replays the trajectory forward to verify if the patch resolves the failure [192, 195].

---

## 6. Pre-Deployment Guardrails & Safety Protocols

To prevent self-evolving loops from introducing destructive behaviors, performance drift, or silent regressions, SOTA systems enforce four strict pre-deployment gates [552]:

1.  **Frozen Evaluation sets**: Success is never evaluated solely on the active training set or active user interaction stream [552]. Held-out validation sets must be completely frozen and isolated; a self-modifying loop will naturally find hacks to maximize its training metric (e.g., skipping safety checks or hardcoding common outputs) while quietly regressing on unseen tasks [552].
2.  **Statistical Gödel Machine Acceptance Gates**: A candidate modification is never merged or promoted based on a raw increase in validation score alone [241, 552]. A candidate is only accepted if it passes a rigorous, non-regression test suite with a strict statistical confidence threshold (e.g., $\ge 95\%$ confidence under paired t-tests or UCB bandit boundaries), strictly controlling cumulative risk across recursive iterations [241, 552].
3.  **Strict Sandbox Isolation**: All validation runs, code evaluations, and third-party tools are executed in non-root, read-only, network-disabled Docker containers with hard resource caps, protecting the host system from security exploits or destructive loops [174, 182, 339].
4.  **Automated Rollback Ledgers**: The Selector maintains a complete git-like version control ledger of all prompts, configurations, and code files [552]. If validation metrics fall below the historical baseline at any milestone, an automated rollback is triggered immediately, restoring the last proven "best-belief" system state [552].

---

## 7. SEDS-HGM Performance Gains (Empirical Evidence)

Evolving agent configurations through SEDS code-space search and clade-level selection yields massive, measurable improvements over traditional hand-engineered designs across diverse, complex domains [484]:

| Target Domain | Evaluation Benchmark | Baseline Configuration | SEDS-HGM Evolved Configuration | Measurable Performance Gains | Citation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Reading Comprehension** | DROP Benchmark | Hand-designed CoT / Self-Refine | Meta Agent Search code-space optimization | **+13.6 F1 Score** improvement over human baselines | [3, 484] |
| **Advanced Mathematics** | AIME-2025 Benchmark | Base DSPy ChainOfThought | GEPA-optimized programmatic prompts | **+12.0% accuracy** gain on complex math sequences | [41, 484] |
| **Multilingual Translation** | MGSM Multi-Language | Hand-designed zero-shot prompts | Meta Agent Search synthesized code architectures | **+14.4% translation accuracy** on multi-language datasets | [3, 484] |
| **Software Engineering** | SWE-bench (Verified) | Fixed-architecture software agents | Darwin Gödel Machine self-evolving codebases | **SWE-bench: 20.0% to 50.0%** accuracy | [23, 484] |
| **Multi-Hop Verification** | HotPotQA Dataset | Standard ReAct prompting | Language Agent Tree Search with MCTS & reflection | **71.0% exact match** accuracy, doubling 32.0% ReAct baseline | [56, 484] |
| **CAD Assembly Design** | Tabletop Football Fabrication | General-purpose base language models | ArtiCAD dynamic multi-agent FreeCAD code loops | **97.14% assembly accuracy** vs. 48.57% for unoptimized base models | [25, 484] |

---

## References

*   **[3]** Qwen Team. *Qwen3.7: The Agent Frontier*. 2026.
*   **[8]** Ya-Chuan Chen, Tien-Jen Lai, and Hsiang-Wei Hu. *APEX: Adaptive Principle EXtraction A Three-Layer Self-Evolution Framework for Production AI Agents*. Grace AI Technology. June 13, 2026.
*   **[15]** Wenyi Wang, Piotr Piękos, Li Nanbo, Firas Laakom, Yimeng Chen, Mateusz Ostaszewski, Mingchen Zhuge, and Jürgen Schmidhuber. *Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine*. King Abdullah University of Science and Technology (KAUST). October 28, 2025.
*   **[22]** J. Lin, S. Liu, C. Pan, L. Lin, S. Dou, X. Huang, H. Yan, Z. Han, and Z. Gui. *Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses*. CoRR. 2026.
*   **[23]** Jenny Zhang, Bo Zhao, William Yang, Jakob N. Foerster, Jeff Clune, Minqi Jiang, Sam Devlin, and Tatiana Shavrina. *Hyperagents*. Meta AI. August 24, 2026.
*   **[32]** Shraddha Barke, Arnav Goyal, Alind Khare, Avaljot Singh, Suman Nath, and Chetan Bansal. *AgentRx: Diagnosing AI Agent Failures from Execution Trajectories*. Microsoft Research. 2026.
*   **[34]** Kunlun Zhu, et al. *DoVer: Intervention-Driven Auto Debugging for LLM Multi-Agent Systems*. arXiv. 2025.
*   **[49]** Kunlun Zhu, et al. *AgentDebugX: An Open-Source Toolkit for Failure Observability, Attribution, and Recovery in LLM Agents*. arXiv. July, 2026.
*   **[225]** Eric Zelikman, Eric Lorch, Lester Mackey, and Adam Tauman Kalai. *Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation*. 2024.
*   **[226]** X. Yin, X. Wang, L. Pan, L. Lin, X. Wan, and W. Y. Wang. *Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement*. 2025.
*   **[383]** Nikhil Shankar, et al. *Multi-Objective Agentic Rewrites for Unstructured Data Processing*. 2026.
