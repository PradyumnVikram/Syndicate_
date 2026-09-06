# SEDS Unseen Domain Protocol

**Purpose:** Document the minimal steps to add a new domain to the SEDS framework without modifying any framework files.

**Philosophy:** The SEDS architecture is task-agnostic. New domains can be added by creating a single Python file under `seds/domains/` and instantiating it in the agent, with zero changes to the core framework.

---

## 8.1 Domain Addition Template

### Required File Structure

```
seds/
├── domains/
│   ├── base.py          # (Framework - never modify)
│   ├── __init__.py      # (Framework - never modify)
│   ├── multi_hop_qa.py  # Existing domain
│   ├── code_transform.py # Existing domain
│   ├── structured_extraction.py # Existing domain
│   └── my_new_domain.py  # <-- Add your new domain here
```

### Minimal Domain File Template

```python
"""[Domain Name] domain for SEDS evaluation.

This domain implements [one-line description of what the domain does].

It uses [optional: specific tools/resources] and provides [number] tasks.
"""

from typing import Any, Callable, List
from dataclasses import replace
from seds.domains.base import TaskDomain, Task, Score, ToolSpec


# --- STEP 1: Define Tools ---
# Specify the tools your domain will use. These run in the sandbox.
MYTOOLS = [
    ToolSpec(
        name="tool_name",
        json_schema={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Description"},
            },
            "required": ["param1"],
        },
        impl=lambda param1: {
            # Tool implementation returns a dict
            "result": param1.upper(),
        }
    ),
]


# --- STEP 2: Implement TaskDomain Class ---
class MyNewDomain(TaskDomain):
    """
    [Domain Name] domain implementation.

    Provides [brief description of the domain's purpose].
    """

    name = "my_new_domain"  # MUST match filename prefix

    goal = """
    [One-sentence goal describing what this domain tests].
    """

    def __init__(self, tools=None, train_tasks=None, val_tasks=None):
        """
        Initialize the domain.

        Args:
            tools: Override default tools (default from MYTOOLS)
            train_tasks: Override default training tasks
            val_tasks: Override default validation tasks

        Returns:
            Initialized TaskDomain instance
        """
        if tools is None:
            tools = MYTOOLS
        if train_tasks is None:
            train_tasks = self._create_training_tasks()
        if val_tasks is None:
            val_tasks = self._create_validation_tasks()

        # Pass all required fields to parent class
        super().__init__(
            name=self.name,
            goal=self.goal,
            tools=tools,
            train_tasks=train_tasks,
            val_tasks=val_tasks,
            evaluate=self.evaluate  # Register evaluation function
        )

    # --- STEP 3: Create Tasks ---
    def _create_training_tasks(self) -> List[Task]:
        """Create the training task set."""
        return [
            Task(
                task_id="my_new_001",
                inputs={
                    "param1": "value1",
                    # Additional inputs as needed
                },
                reference="expected_output",  # Ground truth reference
            ),
            Task(
                task_id="my_new_002",
                inputs={
                    "param1": "value2",
                },
                reference="expected_output_2",
            ),
        ]

    def _create_validation_tasks(self) -> List[Task]:
        """Create the frozen validation task set."""
        # ⚠️ CRITICAL: Validation tasks must have different task_ids than training tasks.
        # Reusing task_ids would cause train/val leakage (see section 4 of USER_INTERACTION_PLAN.md).
        return [
            Task(
                task_id="my_new_003",
                inputs={
                    "param1": "value3",
                },
                reference="expected_output_3",
            ),
            Task(
                task_id="my_new_004",
                inputs={
                    "param1": "value4",
                },
                reference="expected_output_4",
            ),
        ]

    # --- STEP 4: Implement evaluate() ---
    def evaluate(self, task: Task, host_output_reference: str) -> Score:
        """
        Evaluate a single task host-side.

        Args:
            task: The task specification
            host_output_reference: The agent's output (ground truth not available here)

        Returns:
            Score object with:
                - correct: bool (True if answer matches task.reference)
                - partial: float (must equal 1.0 if correct, 0.0 if incorrect)
                - detail: dict[str, Any] (debug information)

        Note:
            partial always equals correct (1.0 iff exact_match, 0.0 otherwise).
            This makes partial redundant but ensures consistent semantics
            across domains for Pareto frontier comparison.
        """
        # Extract and normalize the output reference
        output_ref = str(host_output_reference).strip().lower()

        # The ground truth reference from the Task object
        ground_truth = str(task.reference).strip().lower()

        # Check if answers match (exact match)
        exact_match = output_ref == ground_truth

        # Partial score equals correctness (ensures consistency across domains)
        partial = 1.0 if exact_match else 0.0

        # Detail breakdown for debugging
        detail = {
            "failure_category": "exact_match_failure" if not exact_match else "success",
            "output": host_output_reference,
            "ground_truth": task.reference,
        }

        return Score(
            correct=exact_match,
            partial=partial,
            detail=detail,
        )
```

---

## 8.2 Minimal Example Domain

For immediate practical use, here's a complete, minimal domain that can be added to `seds/domains/`:

**File:** `seds/domains/simple_qa.py`

```python
"""Simple QA domain for SEDS evaluation.

This domain implements a basic QA task with exact-match scoring.
It demonstrates the minimal required implementation for any new domain.
"""

from typing import Any, Callable, List
from dataclasses import replace
from seds.domains.base import TaskDomain, Task, Score, ToolSpec


# Tool specification (optional, can be empty list)
SIMPLETOOLS = [
    ToolSpec(
        name="answer_question",
        json_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to answer"},
            },
            "required": ["question"],
        },
        impl=lambda question: {"answer": question.capitalize()}
    ),
]


class SimpleQA_Domain(TaskDomain):
    """Simple QA domain implementation."""

    name = "simple_qa"

    goal = """
    Answer simple yes/no questions with exact-match validation.
    """

    def __init__(self, tools=None, train_tasks=None, val_tasks=None):
        if tools is None:
            tools = SIMPLETOOLS
        if train_tasks is None:
            train_tasks = self._create_training_tasks()
        if val_tasks is None:
            val_tasks = self._create_validation_tasks()

        super().__init__(
            name=self.name,
            goal=self.goal,
            tools=tools,
            train_tasks=train_tasks,
            val_tasks=val_tasks,
            evaluate=self.evaluate
        )

    def _create_training_tasks(self) -> List[Task]:
        return [
            Task(
                task_id="simple_qa_001",
                inputs={"question": "Is water wet?"},
                reference="yes",
            ),
            Task(
                task_id="simple_qa_002",
                inputs={"question": "Is fire cold?"},
                reference="no",
            ),
        ]

    def _create_validation_tasks(self) -> List[Task]:
        # Validation tasks use different task_ids (003, 004) than training tasks (001, 002).
        # This prevents train/val leakage and ensures a frozen held-out set.
        return [
            Task(
                task_id="simple_qa_003",
                inputs={"question": "Is the sky blue?"},
                reference="yes",
            ),
            Task(
                task_id="simple_qa_004",
                inputs={"question": "Is the ground hot?"},
                reference="no",
            ),
        ]

    def evaluate(self, task: Task, host_output_reference: str) -> Score:
        """Evaluate a simple QA task with exact match."""
        output_ref = str(host_output_reference).strip().lower()
        ground_truth = str(task.reference).strip().lower()

        exact_match = output_ref == ground_truth
        partial = 1.0 if exact_match else 0.0

        detail = {
            "failure_category": "exact_match_failure" if not exact_match else "success",
            "output": host_output_reference,
            "ground_truth": task.reference,
        }

        return Score(
            correct=exact_match,
            partial=partial,
            detail=detail,
        )
```

---

## 8.3 How to Use the New Domain

```python
from seds.domains.simple_qa import SimpleQA_Domain

# Initialize the domain (no framework changes required)
domain = SimpleQA_Domain()

# Get tasks
train_tasks = domain.train_tasks
val_tasks = domain.val_tasks

# Evaluate an answer
task = val_tasks[0]
answer = "yes"  # Agent's answer
score = domain.evaluate(task, answer)

print(f"Correct: {score.correct}")  # True
print(f"Partial: {score.partial}")  # 1.0
print(f"Detail: {score.detail}")     # Debug info
```

---

## 8.4 Verification Checklist

When adding a new domain, verify:

- [ ] Domain file created in `seds/domains/`
- [ ] `name` class attribute matches filename prefix (e.g., `my_new_domain.py` → `name = "my_new_domain"`)
- [ ] `__init__` accepts `tools`, `train_tasks`, `val_tasks` parameters
- [ ] `evaluate()` method has correct signature: `evaluate(self, task: Task, host_output_reference: str) -> Score`
- [ ] `Score` object has: `correct: bool`, `partial: float (0.0-1.0)`, `detail: dict[str, Any]`
- [ ] `partial` always equals `correct` (1.0 if correct, 0.0 if incorrect)
- [ ] `evaluate()` compares against `task.reference` (NOT `task.inputs`)
- [ ] `_create_training_tasks()` returns `List[Task]`
- [ ] `_create_validation_tasks()` returns `List[Task]`
- [ ] **train_tasks and val_tasks have disjoint task_ids** (no overlap, e.g., `set(t.task_id for t in train_tasks).isdisjoint(set(v.task_id for v in val_tasks)) is True`)

---

## 8.5 Task-agnostic Claim Proof

To verify that the framework is task-agnostic:

1. Add a new domain file (e.g., `seds/domains/simple_qa.py`)
2. Instantiation: `domain = SimpleQA_Domain()`
3. No changes to `seds/domains/base.py`, `agent_v0.py`, or any framework files required
4. The domain integrates seamlessly with existing SEDS components:

```python
# This works without ANY framework modifications
from seds.domains.simple_qa import SimpleQA_Domain

domain = SimpleQA_Domain()
for task in domain.val_tasks:
    score = domain.evaluate(task, "expected_answer")
    if score.correct:
        print(f"Task {task.task_id} passed!")
```

**Claim:** The SEDS architecture supports unlimited domains with zero framework changes, as proven by the ability to add `simple_qa.py` above without touching any existing files.
