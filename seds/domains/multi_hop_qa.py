"""Multi-hop QA domain for SEDS evaluation.

This domain implements exact-match local QA with Multi-hop QA-style tasks
(HotPotQA-style, 2-3 hop reasoning). Tasks are provided with reference answers.
"""

import ast
from typing import Any, Callable, List
from dataclasses import replace
from seds.domains.base import TaskDomain, Task, Score, ToolSpec


def safe_eval_arithmetic(expr: str) -> float | None:
    """Evaluate arithmetic expressions safely using a restricted AST.

    Only allows:
    - Numbers (int and float)
    - Binary operators: +, -, *, /, //, %, **
    - Unary operators: +, -
    - Parentheses for grouping
    - Whitespace as separator

    Args:
        expr: The arithmetic expression to evaluate

    Returns:
        The computed float result, or None if evaluation fails
    """
    expr = expr.strip()
    if not expr:
        return None

    try:
        tree = ast.parse(expr, mode='eval')

        class Validator(ast.NodeVisitor):
            def generic_visit(self, node):
                # Only recurse into allowed node types
                if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                                     ast.Num, ast.Constant, ast.Add, ast.Sub,
                                     ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                                     ast.Pow, ast.USub, ast.UAdd)):
                    super().generic_visit(node)
                # For disallowed node types, don't recurse (this will raise ValueError later)

        validator = Validator()
        validator.visit(tree)

        def eval_node(node):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div):
                    return left / right
                elif isinstance(node.op, ast.FloorDiv):
                    return left // right
                elif isinstance(node.op, ast.Mod):
                    return left % right
                elif isinstance(node.op, ast.Pow):
                    return left ** right
                else:
                    raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return +operand
                elif isinstance(node.op, ast.USub):
                    return -operand
                else:
                    raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            elif isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("Only numeric constants allowed")
            else:
                raise ValueError(f"Unsupported node type in value: {type(node).__name__}")

        result = eval_node(tree)
        if isinstance(result, (int, float)):
            return float(result)
        return None

    except Exception:
        return None


# Define tool specifications for multi-hop QA
MULTIHOP_TOOLS = [
    ToolSpec(
        name="web_search",
        json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
        impl=lambda query: {
            # Mock web search result - in real implementation, this would call an actual API
            "results": [
                {"title": f"Result for {query}", "snippet": f"Information about {query}"},
            ],
        }
    ),
    ToolSpec(
        name="calculation",
        json_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression"},
            },
            "required": ["expression"],
        },
        impl=lambda expression: {
            "result": safe_eval_arithmetic(expression)
        }
    ),
]


class MultiHopQA_Domain(TaskDomain):
    """
    Multi-hop QA domain implementation.

    Provides 5 fixture tasks with exact-match scoring.
    Each task requires 2-3 hop reasoning through web search and calculation.
    """

    name = "multi_hop_qa"

    goal = """
    Answer multi-hop questions by reasoning through multiple sources.
    Requires combining information from web search and calculation tools.
    """

    def __init__(self, tools=None, train_tasks=None, val_tasks=None):
        """Initialize the domain with task specifications."""
        if tools is None:
            tools = MULTIHOP_TOOLS
        if train_tasks is None:
            train_tasks = self._create_training_tasks()
        if val_tasks is None:
            val_tasks = self._create_validation_tasks()
        super().__init__(name=self.name, goal=self.goal, tools=tools, train_tasks=train_tasks, val_tasks=val_tasks, evaluate=self.evaluate)

    def _create_training_tasks(self) -> List[Task]:
        """Create 3 training tasks for learning."""
        return [
            Task(
                task_id="multi_hop_001",
                inputs={
                    "question": "Who directed the first movie set on Mars and won Best Visual Effects?",
                    "hop1_query": "first movie set on Mars",
                    "hop2_query": "director of that movie",
                },
                reference="Cosmic Voyage"
            ),
            Task(
                task_id="multi_hop_002",
                inputs={
                    "question": "What is the chemical element with atomic number 94, known for being fissile?",
                    "hop1_query": "chemical element atomic number 94",
                    "hop2_query": "fissile elements",
                },
                reference="Plutonium"
            ),
            Task(
                task_id="multi_hop_003",
                inputs={
                    "question": "In what year was the Nobel Peace Prize awarded to Malala Yousafzai?",
                    "hop1_query": "Malala Yousafzai Nobel Peace Prize",
                    "hop2_query": "year of Nobel Peace Prize 2014",
                },
                reference="2014"
            ),
        ]

    def _create_validation_tasks(self) -> List[Task]:
        """Create 2 validation tasks for final evaluation."""
        return [
            Task(
                task_id="multi_hop_004",
                inputs={
                    "question": "What compound is used as a antifreeze in automotive applications?",
                    "hop1_query": "automotive antifreeze main compound",
                    "hop2_query": "ethylene glycol properties",
                },
                reference="ethylene glycol"
            ),
            Task(
                task_id="multi_hop_005",
                inputs={
                    "question": "Which programming language was created by Guido van Rossum?",
                    "hop1_query": "programming language creator Guido van Rossum",
                    "hop2_query": "Python programming language",
                },
                reference="Python"
            ),
        ]

    def evaluate(self, task: Task, host_output_reference: str) -> Score:
        """
        Evaluate a single multi-hop QA task host-side.

        Args:
            task: The task specification
            host_output_reference: Reference answer from host (ground truth)

        Returns:
            Score with correct flag, partial score, and detailed breakdown

        Note:
            partial always equals correct (1.0 iff exact_match, 0.0 otherwise).
            This makes partial redundant but ensures consistent semantics across domains
            for Pareto frontier comparison.
        """
        # Host-side evaluation - exact match check against ground truth reference
        exact_match = str(host_output_reference).strip().lower() == str(task.reference).strip().lower()

        # Extract reasoning hops
        hop1 = task.inputs.get("hop1_query", "")
        hop2 = task.inputs.get("hop2_query", "")

        # Calculate partial score based on correctness (partial == correct)
        partial = 1.0 if exact_match else 0.0

        # Detailed breakdown
        detail = {
            "failure_category": "exact_match_failure" if not exact_match else "success",
            "hop1_retrieved": bool(hop1),
            "hop2_retrieved": bool(hop2),
            "final_answer": host_output_reference,
        }

        return Score(
            correct=exact_match,
            partial=partial,
            detail=detail,
        )
