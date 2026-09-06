"""Multi-hop QA domain for SEDS evaluation.

This domain implements exact-match local QA with Multi-hop QA-style tasks
(HotPotQA-style, 2-3 hop reasoning). Tasks are provided with reference answers.
"""

from typing import Any, Callable, List
from seds.domains.base import TaskDomain, Task, Score, ToolSpec


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
            "result": eval(expression)  # In production, use safer evaluation
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

    def __init__(self):
        """Initialize the domain with task specifications."""
        self.tools = MULTIHOP_TOOLS
        self.train_tasks = self._create_training_tasks()
        self.val_tasks = self._create_validation_tasks()

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
        """
        # Host-side evaluation - exact match check
        exact_match = str(host_output_reference).strip().lower() == str(task.inputs.get("question", "")).strip().lower()

        # Extract reasoning hops
        hop1 = task.inputs.get("hop1_query", "")
        hop2 = task.inputs.get("hop2_query", "")

        # Calculate partial score based on hop completeness
        partial = 1.0 if hop1 and hop2 else 0.0

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
