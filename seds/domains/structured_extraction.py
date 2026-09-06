"""Structured extraction/API-sequencing domain for SEDS evaluation.

This domain implements mock service extraction and API sequencing tasks
with schema-strict validation.
"""

from typing import Any, Callable, List
from dataclasses import replace
from seds.domains.base import TaskDomain, Task, Score, ToolSpec


# Define tool specifications for structured extraction
STRUCTUREDEXTRACT_TOOLS = [
    ToolSpec(
        name="extract_json",
        json_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source data"},
                "schema": {"type": "object", "description": "JSON schema to validate"},
            },
            "required": ["source", "schema"],
        },
        impl=lambda source, schema: {
            "extracted": {"name": "Alice", "age": 30},
            "valid": True,
        }
    ),
    ToolSpec(
        name="api_sequence",
        json_schema={
            "type": "object",
            "properties": {
                "endpoints": {"type": "array", "items": {"type": "string"}},
                "data": {"type": "object", "description": "Input data"},
            },
            "required": ["endpoints", "data"],
        },
        impl=lambda endpoints, data: {
            "responses": [{"status": 200}, {"status": 200}],
            "total_time_ms": 120,
        }
    ),
]


class StructuredExtract_Domain(TaskDomain):
    """
    Structured extraction/API-sequencing domain implementation.

    Provides mock service extraction and API sequencing tasks with
    schema-strict validation.
    """

    name = "structured_extraction"

    goal = """
    Extract structured data from APIs with schema validation and execute
    API sequences in order.
    """

    def __init__(self, tools=None, train_tasks=None, val_tasks=None):
        """Initialize the domain with task specifications."""
        if tools is None:
            tools = STRUCTUREDEXTRACT_TOOLS
        if train_tasks is None:
            train_tasks = self._create_training_tasks()
        if val_tasks is None:
            val_tasks = self._create_validation_tasks()
        super().__init__(name=self.name, goal=self.goal, tools=tools, train_tasks=train_tasks, val_tasks=val_tasks, evaluate=self.evaluate)

    def _create_training_tasks(self) -> List[Task]:
        """Create 2 training tasks."""
        return [
            Task(
                task_id="structured_001",
                inputs={
                    "task_type": "extract",
                    "service": "user_profile",
                    "fields": ["name", "email", "age"],
                },
                reference="Alice",
            ),
            Task(
                task_id="structured_002",
                inputs={
                    "task_type": "sequence",
                    "endpoints": ["/auth", "/data"],
                    "init_data": {"token": "abc123"},
                },
                reference="completed",
            ),
        ]

    def _create_validation_tasks(self) -> List[Task]:
        """Create 3 validation tasks for final evaluation."""
        return [
            Task(
                task_id="structured_003",
                inputs={
                    "task_type": "extract",
                    "service": "inventory",
                    "fields": ["id", "quantity", "price"],
                },
                reference="12345",
            ),
            Task(
                task_id="structured_004",
                inputs={
                    "task_type": "sequence",
                    "endpoints": ["/init", "/process", "/finalize"],
                    "init_data": {"id": 1},
                },
                reference="completed",
            ),
            Task(
                task_id="structured_005",
                inputs={
                    "task_type": "extract",
                    "service": "order",
                    "fields": ["order_id", "status"],
                },
                reference="ORD-1001",
            ),
        ]

    def evaluate(self, task: Task, host_output_reference: str) -> Score:
        """
        Evaluate a single structured extraction/API sequencing task host-side.

        Args:
            task: The task specification
            host_output_reference: Reference output

        Returns:
            Score with correct flag, partial score, and detailed breakdown
        """
        # Host-side evaluation
        output_ref = str(host_output_reference).strip()
        task_type = task.inputs.get("task_type", "")

        # Validate task type
        valid_types = ["extract", "sequence"]
        valid_task_type = task_type in valid_types

        # Check for success indicators
        success_indicators = ["completed", "done", "true", "success"]
        exact_match = output_ref in success_indicators or output_ref == task_type

        partial = 1.0 if valid_task_type and exact_match else 0.0

        detail = {
            "failure_category": "failure" if not (valid_task_type and exact_match) else "success",
            "task_type": task_type,
            "recognized": valid_task_type,
            "output_reference": host_output_reference,
        }

        return Score(
            correct=exact_match and valid_task_type,
            partial=partial,
            detail=detail,
        )
