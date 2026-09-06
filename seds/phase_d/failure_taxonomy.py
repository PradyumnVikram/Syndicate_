"""SEDS Phase D: Failure taxonomy and classification (§6.1).

Provides a structured classification system for runtime failures with
exact enum labels as specified in syndicate_-1 correction:

- tool_misuse: Incorrect tool usage or misuse
- schema_violation: Contract/JSON schema violations
- planning_loop: Recursive planning or infinite loops
- context_overflow: Context window or memory overflow
- hallucinated_fact: Invented facts not grounded in data
- format_error: Data format mismatches or parsing errors
- timeout: Operation exceeded time limit
- crash: Unexpected process crashes
- empty_output: No output generated
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import auto


logger = logging.getLogger(__name__)


class FailureCategory(enum.Enum):
    """Classifies failure types by root cause with exact spec labels."""
    TOOL_MISUSE = auto()              # Incorrect tool usage
    SCHEMA_VIOLATION = auto()          # Contract/JSON schema violations
    PLANNING_LOOP = auto()             # Recursive planning or infinite loops
    CONTEXT_OVERFLOW = auto()          # Context window or memory overflow
    HALLUCINATED_FACT = auto()         # Invented facts not grounded in data
    FORMAT_ERROR = auto()              # Data format mismatches
    TIMEOUT = auto()                   # Operation exceeded time limit
    CRASH = auto()                     # Unexpected process crashes
    EMPTY_OUTPUT = auto()              # No output generated


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

    # Schema violations
    if "ValidationError" in error_type or "SchemaError" in error_type:
        return FailureCategory.SCHEMA_VIOLATION

    # Execution crashes
    if "RuntimeError" in error_type or "ToolExecutionError" in error_type:
        return FailureCategory.CRASH

    # Timeout errors
    if "TimeoutError" in error_type or "Timeout" in error_type:
        return FailureCategory.TIMEOUT

    # Format errors
    if "FormatError" in error_type or "ParseError" in error_type:
        return FailureCategory.FORMAT_ERROR

    # Tool misuse
    if "ToolUsageError" in error_type or "BadArgumentError" in error_type:
        return FailureCategory.TOOL_MISUSE

    # Context overflow
    if "ContextOverflow" in error_type or "MemoryError" in error_type:
        return FailureCategory.CONTEXT_OVERFLOW

    # Hallucinated facts (semantic failures)
    if "HallucinationError" in error_type or "FactError" in error_type:
        return FailureCategory.HALLUCINATED_FACT

    # Empty output
    if "EmptyOutputError" in error_type:
        return FailureCategory.EMPTY_OUTPUT

    # Default to unknown
    return FailureCategory.TOOL_MISUSE


def get_severity_from_category(category: FailureCategory) -> FailureSeverity:
    """Determine severity level from category.

    Args:
        category: FailureCategory

    Returns:
        FailureSeverity
    """
    mapping = {
        FailureCategory.SCHEMA_VIOLATION: FailureSeverity.ERROR,
        FailureCategory.TOOL_MISUSE: FailureSeverity.WARNING,
        FailureCategory.PLANNING_LOOP: FailureSeverity.CRITICAL,
        FailureCategory.CONTEXT_OVERFLOW: FailureSeverity.ERROR,
        FailureCategory.HALLUCINATED_FACT: FailureSeverity.ERROR,
        FailureCategory.FORMAT_ERROR: FailureSeverity.ERROR,
        FailureCategory.TIMEOUT: FailureSeverity.WARNING,
        FailureCategory.CRASH: FailureSeverity.CRITICAL,
        FailureCategory.EMPTY_OUTPUT: FailureSeverity.WARNING,
    }
    return mapping.get(category, FailureSeverity.INFO)


# Built-in failure detectors
def schema_violation_detector(context: dict, span_data: dict) -> Optional[Failure]:
    """Detect schema/contract violations in tool calls."""
    if span_data.get("kind") != "call" or span_data.get("error"):
        return None

    tool_spec = span_data.get("tool_spec")
    if not tool_spec:
        return None

    inputs = span_data.get("inputs", {})
    outputs = span_data.get("outputs", {})

    schema = tool_spec.get("json_schema")
    if not schema:
        return None

    required = schema.get("required", [])
    for field in required:
        if field not in inputs:
            return Failure(
                failure_id=f"sv_{context.get('span_id', 'unknown')[:8]}",
                category=FailureCategory.SCHEMA_VIOLATION,
                severity=get_severity_from_category(FailureCategory.SCHEMA_VIOLATION),
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

    if outputs:
        output_schema = schema.get("properties")
        if output_schema:
            for field, value in outputs.items():
                if field not in output_schema:
                    return Failure(
                        failure_id=f"sv_{context.get('span_id', 'unknown')[:8]}",
                        category=FailureCategory.SCHEMA_VIOLATION,
                        severity=get_severity_from_category(FailureCategory.SCHEMA_VIOLATION),
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
    """Detect execution errors/crashes in tool calls."""
    if span_data.get("kind") != "call":
        return None

    error = span_data.get("error")
    if not error:
        return None

    return Failure(
        failure_id=f"cr_{context.get('span_id', 'unknown')[:8]}",
        category=categorize_error(error),
        severity=get_severity_from_category(categorize_error(error)),
        task_id=context.get("task_id", ""),
        node_id=context.get("node_id", ""),
        span_id=context.get("span_id"),
        message=f"Tool execution crashed: {type(error).__name__}: {str(error)[:100]}",
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


def format_error_detector(context: dict, span_data: dict) -> Optional[Failure]:
    """Detect format errors in data."""
    if span_data.get("kind") != "write":
        return None

    content = span_data.get("outputs", {}).get("content")
    if not content:
        return None

    # Check for common format issues
    if not content.strip():
        return Failure(
            failure_id=f"fe_{context.get('span_id', 'unknown')[:8]}",
            category=FailureCategory.FORMAT_ERROR,
            severity=FailureSeverity.WARNING,
            task_id=context.get("task_id", ""),
            node_id=context.get("node_id", ""),
            span_id=context.get("span_id"),
            message="Generated empty file content",
            details={
                "file": span_data.get("inputs", {}).get("file"),
            },
            remediation="Check output generation logic"
        )

    return None


def planning_loop_detector(context: dict, span_data: dict) -> Optional[Failure]:
    """Detect potential planning loops."""
    if span_data.get("kind") != "call":
        return None

    # Check for repeated tool calls with same arguments
    tool_name = span_data.get("tool_name")
    inputs = span_data.get("inputs", {})

    # In a real implementation, this would check against historical calls
    # For now, we just detect repeated planning-related tool calls
    if tool_name in ["plan", "plan_complex", "generate_plan"]:
        return Failure(
            failure_id=f"pl_{context.get('span_id', 'unknown')[:8]}",
            category=FailureCategory.PLANNING_LOOP,
            severity=FailureSeverity.WARNING,
            task_id=context.get("task_id", ""),
            node_id=context.get("node_id", ""),
            span_id=context.get("span_id"),
            message=f"Possible planning loop detected: {tool_name}",
            details={
                "tool_name": tool_name,
                "inputs": inputs,
            },
            remediation="Review planning strategy to avoid infinite recursion"
        )

    return None


def initialize_failure_taxonomy():
    """Initialize default failure detectors."""
    register_failure_detector(schema_violation_detector)
    register_failure_detector(execution_error_detector)
    register_failure_detector(timeout_detector)
    register_failure_detector(format_error_detector)
    register_failure_detector(planning_loop_detector)

    logger.info("Failure taxonomy initialized with 5 built-in detectors (exact labels: tool_misuse, schema_violation, planning_loop, context_overflow, hallucinated_fact, format_error, timeout, crash, empty_output)")
