"""Simple QA domain for SEDS evaluation.

This domain implements simple yes/no question answering with exact-match validation.
It serves as a minimal example domain that can be instantiated without any framework modifications.
"""

from typing import Any, Callable, List
from dataclasses import replace
from domains.base import TaskDomain, Task, Score, ToolSpec

# Simple set of tools for the domain
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
