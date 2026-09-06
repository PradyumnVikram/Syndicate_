"""Arithmetic word problem domain for SEDS end-to-end self-improvement test.

This domain implements arithmetic word problems with numeric normalization-based
evaluation to avoid the strict string matching issues seen in SimpleQA.
"""

from typing import Any, Callable, List
from seds.domains.base import TaskDomain, Task, Score, ToolSpec


def normalize_number(x: Any) -> float:
    """Normalize numeric values to float for comparison.

    Handles:
    - Numeric strings like "5.5", "42", "3"
    - Actual floats
    - Exact integers represented as floats
    """
    if x is None:
        return float('nan')
    try:
        return float(x)
    except (TypeError, ValueError):
        # If not numeric, try to extract digits and decimals
        import re
        matches = re.findall(r'[+-]?(\d+\.?\d*)', str(x))
        if matches:
            return float(matches[-1])
        return float('nan')


def evaluate_arithmetic(task: Task, host_output_reference: Any) -> Score:
    """
    Evaluate arithmetic word problem with numeric normalization.

    Args:
        task: The task specification
        host_output_reference: Reference answer from the model

    Returns:
        Score with correct flag, partial score, and detailed breakdown
    """
    from seds.domains.arithmetic import normalize_number

    # Extract task inputs
    question = task.inputs.get("question", "")
    expected_number = normalize_number(task.reference)

    # Normalize the model's output
    try:
        model_output = str(host_output_reference).strip()
        # Try to extract the last number from the output (takes the last number, not the first)
        import re
        matches = re.findall(r'[+-]?(\d+\.?\d*)', model_output)
        if matches:
            actual_number = float(matches[-1])
        else:
            # If no number found, check if reference was just a number string
            actual_number = normalize_number(model_output)
    except:
        actual_number = float('nan')

    # Calculate exact match (with small tolerance for floating point)
    tolerance = 1e-6
    is_correct = (
        not (expected_number is None or expected_number != expected_number)) and \
        (actual_number != actual_number or
         abs(actual_number - expected_number) < tolerance)

    partial = 1.0 if is_correct else 0.0

    detail = {
        "failure_category": "numeric_mismatch" if not is_correct else "success",
        "expected_normalized": expected_number,
        "actual_normalized": actual_number,
        "expected_raw": task.reference,
        "actual_raw": host_output_reference,
    }

    return Score(
        correct=is_correct,
        partial=partial,
        detail=detail,
    )


def create_calculator_tool() -> ToolSpec:
    """Create a calculator tool using the safe_eval_arithmetic from domains.

    Returns ToolSpec with OpenAI function calling schema format.
    """
    from seds.domains.multi_hop_qa import safe_eval_arithmetic

    return ToolSpec(
        name="calculator",
        json_schema={
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate an arithmetic expression safely.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression to evaluate (e.g., '2 + 2 * 3')"
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
        impl=lambda expression: {
            "result": safe_eval_arithmetic(expression)
        }
    )


# Fixed task set for paired evaluation across iterations
TASKS = [
    Task(
        task_id="arithmetic_001",
        inputs={"question": "What is 5 + 7?"},
        reference=12,
    ),
    Task(
        task_id="arithmetic_002",
        inputs={"question": "Calculate 3 * 4 + 2"},
        reference=14,
    ),
    Task(
        task_id="arithmetic_003",
        inputs={"question": "What is 100 - 33?"},
        reference=67,
    ),
    Task(
        task_id="arithmetic_004",
        inputs={"question": "What is 15 / 3?"},
        reference=5,
    ),
    Task(
        task_id="arithmetic_005",
        inputs={"question": "What is 2.5 * 4?"},
        reference=10.0,
    ),
]


class Arithmetic_Domain(TaskDomain):
    """Arithmetic word problem domain for self-improvement testing."""

    name = "arithmetic"

    goal = """
    Answer arithmetic word problems correctly by performing numeric computations.
    """

    def __init__(self, tools=None, train_tasks=None, val_tasks=None):
        if tools is None:
            tools = [create_calculator_tool()]
        if train_tasks is None:
            train_tasks = TASKS[:2]  # First 2 for training (not used in this test)
        if val_tasks is None:
            val_tasks = TASKS[2:]   # Last 3 for validation (fixed, paired across iterations)

        super().__init__(
            name=self.name,
            goal=self.goal,
            tools=tools,
            train_tasks=train_tasks,
            val_tasks=val_tasks,
            evaluate=evaluate_arithmetic
        )
