"""Core domain abstractions with frozen dataclasses for immutable task representation."""

import dataclasses
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Allow subclasses to modify specific fields if needed (but keep structure frozen)
FieldLevel = Optional[str]  # 'read-only', 'immutable', 'mutable'


class EvaluationMode(Enum):
    """How tasks should be evaluated."""
    EXACT_MATCH = "exact_match"
    LEVENSHTEIN = "levenshtein"
    HELPER_REQUIRED = "helper_required"
    SPAN = "span"


class ToolRequirement(Enum):
    """Constraint on tool availability."""
    FREE = "free"  # No external network/API calls
    LOCAL = "local"  # Local tool invocation only
    NETWORK = "network"  # Requires network access


@dataclasses.dataclass(frozen=True, slots=True)
class TaskSpec:
    """Frozen specification for a single task instance."""
    task_id: str
    input: Dict[str, Any]
    reference: str  # Ground truth answer, never exposed to agent in sandboxed runs

    def __post_init__(self):
        """Validate task structure."""
        if not self.task_id or not isinstance(self.task_id, str):
            raise ValueError("task_id must be a non-empty string")

    def with_input(self, **kwargs: Any) -> 'TaskSpec':
        """Return a new task with updated input fields."""
        if not kwargs:
            return self
        new_input = self.input.copy()
        new_input.update(kwargs)
        return dataclasses.replace(self, input=new_input)


@dataclasses.dataclass(frozen=True, slots=True)
class ToolSpec:
    """Frozen tool specification."""
    tool_name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    requirement: ToolRequirement = ToolRequirement.FREE

    def __post_init__(self):
        """Validate tool schema structure."""
        required_keys = {"type", "properties", "required"}
        if not all(key in self.input_schema for key in required_keys):
            raise ValueError("input_schema must contain type, properties, and required")
        if not all(key in self.output_schema for key in required_keys):
            raise ValueError("output_schema must contain type, properties, and required")


@dataclasses.dataclass(frozen=True, slots=True)
class Score:
    """Frozen evaluation score with metadata."""
    success: bool
    score: float  # 0.0 to 1.0 or specific metric value
    metric_name: str
    breakdown: Dict[str, Any] = dataclasses.field(default_factory=dict)
    error: Optional[str] = None
    cost_ms: float = 0.0  # Execution time in milliseconds
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """Validate score structure."""
        if self.score < 0 or self.score > 1 and "score" in self.breakdown.get("primary", {}):
            raise ValueError("score must be in range [0, 1] or represent a specific metric value")


@dataclasses.dataclass(frozen=True, slots=True)
class TaskResult:
    """Result of evaluating a task."""
    task_id: str
    spec: TaskSpec
    score: Score
    output: Optional[str] = None
    events: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """Validate result structure."""
        if self.task_id != self.spec.task_id:
            raise ValueError("task_id must match spec.task_id")


class DomainMetric(Enum):
    """Domain-specific evaluation metrics."""
    # General
    ACCURACY = "accuracy"
    RECALL = "recall"
    PRECISION = "precision"
    F1 = "f1"
    # Text-based
    EXACT_MATCH = "exact_match"
    LEVENSHTEIN_SIMILARITY = "levenshtein_similarity"
    ROUGE_L = "rouge_l"
    # Cost/efficiency
    TIME_COST = "time_cost_ms"
    TOKEN_COST = "token_cost"
    RELIABILITY = "reliability"  # Fraction of successful runs


class RolloutResult:
    """Result of evaluating a domain across multiple tasks."""
    domain_name: str
    results: List[TaskResult]
    average_score: float
    total_cost_ms: float
    total_duration_ms: float
    failure_breakdown: Dict[str, int]
    timestamp: datetime

    def __init__(
        self,
        domain_name: str,
        results: List[TaskResult],
        total_cost_ms: float = 0.0,
        total_duration_ms: float = 0.0,
        timestamp: Optional[datetime] = None,
    ):
        self.domain_name = domain_name
        self.results = results
        self.total_cost_ms = total_cost_ms
        self.total_duration_ms = total_duration_ms
        self.timestamp = timestamp or datetime.utcnow()

        # Calculate aggregate scores
        successful = [r for r in results if r.score.success]
        failed = [r for r in results if not r.score.success]

        if results:
            self.average_score = sum(r.score.score for r in results) / len(results)
        else:
            self.average_score = 0.0

        # Count failure types
        self.failure_breakdown = {}
        for r in failed:
            failure_type = r.score.breakdown.get("failure_type", "unknown")
            self.failure_breakdown[failure_type] = self.failure_breakdown.get(failure_type, 0) + 1


class TaskDomain(ABC):
    """Abstract base class for all evaluation domains."""

    domain_name: str = ""

    def __init__(self):
        """Initialize domain with tool specifications."""
        self.tools: List[ToolSpec] = []

    @abstractmethod
    def tools_for_task(self, task: TaskSpec) -> List[ToolSpec]:
        """Return relevant tools for a specific task."""
        pass

    @abstractmethod
    def evaluate_single(
        self,
        task: TaskSpec,
        run_host_side: bool = True,
        output_reference: bool = False,
    ) -> TaskResult:
        """
        Evaluate a single task and return the result.

        Args:
            task: The task specification
            run_host_side: If True, run host-side (no sandbox). If False, evaluate in sandbox.
            output_reference: Only relevant when run_host_side=False. If True, expose reference
                             to sandboxed code (should be avoided for security).

        Returns:
            TaskResult with score and metadata
        """
        pass

    @abstractmethod
    def get_metric(self, result: TaskResult) -> float:
        """
        Extract a domain-specific metric from a result.

        Returns a float in [0, 1] range for binary classification domains.
        """
        pass

    def evaluate_batch(
        self,
        tasks: List[TaskSpec],
        run_host_side: bool = True,
    ) -> List[TaskResult]:
        """Evaluate a batch of tasks."""
        results = []
        for task in tasks:
            results.append(self.evaluate_single(task, run_host_side=run_host_side))
        return results

    def compare_results(self, baseline: RolloutResult, evolved: RolloutResult) -> Dict[str, Any]:
        """
        Compare baseline and evolved results with statistical significance.

        Returns a comparison dictionary with metrics, confidence intervals,
        and statistical tests.
        """
        pass

    def compute_pareto_frontier(
        self,
        results: List[TaskResult],
        metrics: List[str],
    ) -> List[Tuple[float, float, float]]:
        """
        Compute Pareto frontier of trade-offs across multiple metrics.

        Returns list of (accuracy, speed, cost) tuples where no other result
        dominates it.
        """
        pass
