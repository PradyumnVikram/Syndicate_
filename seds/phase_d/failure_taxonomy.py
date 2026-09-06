"""SEDS Phase D: Failure taxonomy and classification (§6.1).

Provides a structured classification system for runtime failures:
- Contract violations (input/output mismatch with ToolSpec)
- Execution failures (runtime errors in tool implementations)
- Safety violations (restricted operations, unauthorized access)
- Data quality failures (invalid inputs, malformed data)
- Timeout failures (resource exhaustion, long-running operations)
- Semantic failures (incoherent behavior, hallucinations)
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import auto


logger = logging.getLogger(__name__)


class FailureCategory(enum.Enum):
    """Classifies failure types by root cause."""
    CONTRACT_VIOLATION = auto()      # Input/output violates ToolSpec
    EXECUTION_ERROR = auto()          # Tool impl raises exception
    SAFETY_VIOLATION = auto()         # Restricted operation attempted
    DATA_QUALITY = auto()             # Invalid input data
    TIMEOUT = auto()                  # Operation exceeded time limit
    SEMANTIC = auto()                 # Incoherent behavior, hallucination
    RESOURCE_EXHAUSTION = auto()      # Memory/CPU limits exceeded
    UNKNOWN = auto()                  # Unclassified failure


@dataclass
class Failure:
    """Represents a single failure event in the SEDS pipeline."""
    failure_id: str
    category: FailureCategory
    severity: FailureSeverity
    task_id: str
    node_id: str
    span_id: str | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    remediation: Optional[str] = None

    def __hash__(self):
        return hash(self.failure_id)


@dataclass
class FailureStatistics:
    """Aggregated statistics for failures across a run."""
    total_failures: int = 0
    by_category: dict[FailureCategory, int] = field(default_factory=dict)
    by_severity: dict[FailureSeverity, int] = field(default_factory=dict)
    top_messages: List[tuple[str, int]] = field(default_factory=list)

    def count(self, category: FailureCategory, severity: FailureSeverity):
        """Count a failure by category and severity."""
        self.total_failures += 1
        self.by_category[category] = self.by_category.get(category, 0) + 1
        self.by_severity[severity] = self.by_severity.get(severity, 0) + 1

    def add_message(self, message: str, count: int = 1):
        """Add a message with its count to the top messages list."""
        self.top_messages.append((message, count))

    def get_top_n_messages(self, n: int = 5) -> List[tuple[str, int]]:
        """Return the top N most frequent messages."""
        sorted_messages = sorted(self.top_messages, key=lambda x: -x[1])
        return sorted_messages[:n]


class FailureSeverity(enum.Enum):
    """Severity levels for failures."""
    CRITICAL = auto()   # Blocks all subsequent work
    ERROR = auto()      # Prevents successful completion
    WARNING = auto()    # May require attention but not blocking
    INFO = auto()       # Diagnostic information


FAILURE_SEVERITY_LEVELS = {
    FailureSeverity.CRITICAL: 1.0,
    FailureSeverity.ERROR: 0.8,
    FailureSeverity.WARNING: 0.4,
    FailureSeverity.INFO: 0.1,
}


# Registry of failure detectors
FAILURE_DETECTORS: list[Callable[[dict, dict], Optional[Failure]]] = []


def register_failure_detector(detector: Callable[[dict, dict], Optional[Failure]]):
    """Register a failure detector function.

    Args:
        detector: Function that takes (context, span_data) and returns Failure or None
    """
    FAILURE_DETECTORS.append(detector)
    logger.debug(f"Registered failure detector: {detector.__name__}")


def detect_failures(context: dict, span_data: dict) -> list[Failure]:
    """Run all registered detectors on the given span data.

    Args:
        context: Execution context (task_id, node_id, etc.)
        span_data: Span data to analyze for failures

    Returns:
        List of detected failures
    """
    failures = []
    for detector in FAILURE_DETECTORS:
        try:
            failure = detector(context, span_data)
            if failure:
                failures.append(failure)
        except Exception as e:
            logger.warning(f"Failure detector {detector.__name__} raised error: {e}")
    return failures


def categorize_error(error: Any) -> FailureCategory:
    """Categorize an error object by type.

    Args:
        error: Exception or error object

    Returns:
        FailureCategory
    """
    error_type = type(error).__name__

    # Contract violations (type mismatch, missing required fields)
    if "ValidationError" in error_type or "SchemaError" in error_type:
        return FailureCategory.CONTRACT_VIOLATION

    # Execution errors (tool implementation failure)
    if "RuntimeError" in error_type or "ToolExecutionError" in error_type:
        return FailureCategory.EXECUTION_ERROR

    # Timeout errors
    if "TimeoutError" in error_type or "Timeout" in error_type:
        return FailureCategory.TIMEOUT

    # Safety violations (filesystem, network, etc.)
    if "PermissionError" in error_type or "SecurityError" in error_type:
        return FailureCategory.SAFETY_VIOLATION

    # Memory/CPU errors
    if "MemoryError" in error_type or "ResourceExhaustedError" in error_type:
        return FailureCategory.RESOURCE_EXHAUSTION

    # Default to unknown
    return FailureCategory.UNKNOWN


def get_severity_from_category(category: FailureCategory) -> FailureSeverity:
    """Determine severity level from category.

    Args:
        category: FailureCategory

    Returns:
        FailureSeverity
    """
    mapping = {
        FailureCategory.CONTRACT_VIOLATION: FailureSeverity.ERROR,
        FailureCategory.EXECUTION_ERROR: FailureSeverity.ERROR,
        FailureCategory.SAFETY_VIOLATION: FailureSeverity.CRITICAL,
        FailureCategory.DATA_QUALITY: FailureSeverity.WARNING,
        FailureCategory.TIMEOUT: FailureSeverity.WARNING,
        FailureCategory.SEMANTIC: FailureSeverity.INFO,
        FailureCategory.RESOURCE_EXHAUSTION: FailureSeverity.ERROR,
        FailureCategory.UNKNOWN: FailureSeverity.INFO,
    }
    return mapping.get(category, FailureSeverity.INFO)


# Built-in failure detectors
def contract_violation_detector(context: dict, span_data: dict) -> Optional[Failure]:
    """Detect contract violations in tool calls (input/output schema mismatch)."""
    if span_data.get("kind") != "call" or span_data.get("error"):
        return None

    tool_spec = span_data.get("tool_spec")
    if not tool_spec:
        return None

    inputs = span_data.get("inputs", {})
    outputs = span_data.get("outputs", {})

    # Check input schema compliance
    schema = tool_spec.get("json_schema")
    if not schema:
        return None

    required = schema.get("required", [])
    for field in required:
        if field not in inputs:
            return Failure(
                failure_id=f"cv_{context.get('span_id', 'unknown')[:8]}",
                category=FailureCategory.CONTRACT_VIOLATION,
                severity=get_severity_from_category(FailureCategory.CONTRACT_VIOLATION),
                task_id=context.get("task_id", ""),
                node_id=context.get("node_id", ""),
                span_id=context.get("span_id"),
                message=f"Missing required input field: {field}",
                details={
                    "field": field,
                    "tool_spec": tool_spec.get("name"),
                    "inputs": inputs,
                },
                remediation="Ensure all required fields are provided in inputs"
            )

    # Check output schema compliance
    if outputs:
        output_schema = schema.get("properties")
        if output_schema:
            for field, value in outputs.items():
                if field not in output_schema:
                    return Failure(
                        failure_id=f"cv_{context.get('span_id', 'unknown')[:8]}",
                        category=FailureCategory.CONTRACT_VIOLATION,
                        severity=get_severity_from_category(FailureCategory.CONTRACT_VIOLATION),
                        task_id=context.get("task_id", ""),
                        node_id=context.get("node_id", ""),
                        span_id=context.get("span_id"),
                        message=f"Unexpected output field: {field}",
                        details={
                            "field": field,
                            "value": str(value)[:100],
                            "tool_spec": tool_spec.get("name"),
                        },
                        remediation="Check tool implementation for unexpected output fields"
                    )

    return None


def execution_error_detector(context: dict, span_data: dict) -> Optional[Failure]:
    """Detect execution errors in tool calls."""
    if span_data.get("kind") != "call":
        return None

    error = span_data.get("error")
    if not error:
        return None

    return Failure(
        failure_id=f"ex_{context.get('span_id', 'unknown')[:8]}",
        category=categorize_error(error),
        severity=get_severity_from_category(categorize_error(error)),
        task_id=context.get("task_id", ""),
        node_id=context.get("node_id", ""),
        span_id=context.get("span_id"),
        message=f"Tool execution failed: {type(error).__name__}: {str(error)[:100]}",
        details={
            "tool_name": span_data.get("tool_name"),
            "error_type": type(error).__name__,
            "error_details": str(error),
        },
        remediation="Check tool implementation and error logs"
    )


def timeout_detector(context: dict, span_data: dict) -> Optional[Failure]:
    """Detect timeout errors in long-running operations."""
    if span_data.get("kind") != "call":
        return None

    duration_ms = span_data.get("duration_ms", 0)
    timeout_ms = span_data.get("timeout_ms", 0)

    if timeout_ms > 0 and duration_ms >= timeout_ms:
        return Failure(
            failure_id=f"tm_{context.get('span_id', 'unknown')[:8]}",
            category=FailureCategory.TIMEOUT,
            severity=FailureSeverity.ERROR,
            task_id=context.get("task_id", ""),
            node_id=context.get("node_id", ""),
            span_id=context.get("span_id"),
            message=f"Tool call timed out after {duration_ms}ms (threshold: {timeout_ms}ms)",
            details={
                "tool_name": span_data.get("tool_name"),
                "duration_ms": duration_ms,
                "timeout_ms": timeout_ms,
            },
            remediation="Reduce timeout threshold or optimize tool implementation"
        )

    return None


def initialize_failure_taxonomy():
    """Initialize default failure detectors."""
    register_failure_detector(contract_violation_detector)
    register_failure_detector(execution_error_detector)
    register_failure_detector(timeout_detector)

    logger.info("Failure taxonomy initialized with 3 built-in detectors")
