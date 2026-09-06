"""Code/data transformation domain for SEDS evaluation.

This domain implements synthetic ETL and SWE-style file-edit tasks with
deterministic test suites.
"""

from typing import Any, Callable, List
from dataclasses import replace
from seds.domains.base import TaskDomain, Task, Score, ToolSpec


# Define tool specifications for code/data transformation
CODETRANSFORM_TOOLS = [
    ToolSpec(
        name="file_read",
        json_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to file"},
            },
            "required": ["file_path"],
        },
        impl=lambda file_path: {
            "content": f"Mock content for {file_path}",
            "exists": True,
        }
    ),
    ToolSpec(
        name="file_write",
        json_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to file"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["file_path", "content"],
        },
        impl=lambda file_path, content: {
            "success": True,
            "written_bytes": len(content),
        }
    ),
    ToolSpec(
        name="test_deterministic",
        json_schema={
            "type": "object",
            "properties": {
                "test_file": {"type": "string", "description": "Path to test file"},
            },
            "required": ["test_file"],
        },
        impl=lambda test_file: {
            "passed": True,
            "metrics": {"tests_run": 5, "tests_passed": 5, "coverage": "100%"},
        }
    ),
]


class CodeTransform_Domain(TaskDomain):
    """
    Code/data transformation domain implementation.

    Provides synthetic ETL and SWE-style file-edit tasks with deterministic
    test suites.
    """

    name = "code_transform"

    goal = """
    Execute data transformations and code modifications with deterministic
    test validation.
    """

    def __init__(self, tools=None, train_tasks=None, val_tasks=None):
        """Initialize the domain with task specifications."""
        if tools is None:
            tools = CODETRANSFORM_TOOLS
        if train_tasks is None:
            train_tasks = self._create_training_tasks()
        if val_tasks is None:
            val_tasks = self._create_validation_tasks()
        super().__init__(name=self.name, goal=self.goal, tools=tools, train_tasks=train_tasks, val_tasks=val_tasks, evaluate=self.evaluate)

    def _create_training_tasks(self) -> List[Task]:
        """Create 2 training tasks."""
        return [
            Task(
                task_id="code_transform_001",
                inputs={
                    "operation": "ETL",
                    "source": "user_logs.csv",
                    "destination": "analytics.db",
                    "transform": "aggregate by timestamp",
                },
                reference="completed",
            ),
            Task(
                task_id="code_transform_002",
                inputs={
                    "operation": "file_edit",
                    "file": "src/config.json",
                    "changes": [{"path": "timeout", "value": 30}],
                },
                reference="applied",
            ),
        ]

    def _create_validation_tasks(self) -> List[Task]:
        """Create 3 validation tasks for final evaluation."""
        return [
            Task(
                task_id="code_transform_003",
                inputs={
                    "operation": "ETL",
                    "source": "sales_data.json",
                    "destination": "warehouse.csv",
                    "transform": "extract date range and sum",
                },
                reference="valid",
            ),
            Task(
                task_id="code_transform_004",
                inputs={
                    "operation": "file_edit",
                    "file": "api/endpoint.py",
                    "changes": [{"path": "endpoint", "value": "/new/path"}],
                },
                reference="applied",
            ),
            Task(
                task_id="code_transform_005",
                inputs={
                    "operation": "file_delete",
                    "file": "legacy/config.ini",
                },
                reference="deleted",
            ),
        ]

    def evaluate(self, task: Task, host_output_reference: str) -> Score:
        """
        Evaluate a single code/data transformation task host-side.

        Args:
            task: The task specification
            host_output_reference: Reference output indicating success

        Returns:
            Score with correct flag, partial score, and detailed breakdown

        Note:
            partial always equals correct (1.0 iff exact_match, 0.0 otherwise).
            This makes partial redundant but ensures consistent semantics across domains
            for Pareto frontier comparison.
        """
        # Host-side evaluation - check if task was marked as completed
        output_ref = str(host_output_reference).strip().lower()
        operation = task.inputs.get("operation", "")

        # Check if operation is recognized
        recognized_ops = ["ETL", "file_edit", "file_delete"]
        recognized = operation in recognized_ops

        # Check if reference indicates success
        exact_match = output_ref == str(task.reference).strip()

        # Partial score based on correctness (partial == correct)
        partial = 1.0 if exact_match else 0.0

        detail = {
            "failure_category": "failure" if not (recognized and exact_match) else "success",
            "operation": operation,
            "recognized": recognized,
            "output_reference": host_output_reference,
        }

        return Score(
            correct=exact_match and recognized,
            partial=partial,
            detail=detail,
        )
