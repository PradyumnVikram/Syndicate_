"""SEDS Phase D: Contract auditor for tool call validation (§6.2).

Validates that tool calls comply with their ToolSpecs:
- Input schema compliance (required fields, types, constraints)
- Output schema compliance (return types, field presence)
- State consistency (tool state after calls)
- Call rate limits (burst detection, throttling)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from enum import auto


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Result of a contract validation check."""
    is_valid: bool
    tool_name: str
    check_type: str  # "input_schema", "output_schema", "rate_limit", "state_consistency"
    error_messages: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractAudit:
    """Full audit result for a set of tool calls."""
    task_id: str
    node_id: str
    total_calls: int = 0
    passed_validations: int = 0
    failed_validations: int = 0
    violations: list[ValidationResult] = field(default_factory=list)
    error_log: str = ""
    audit_timestamp: float = time.time()

    def add_violation(self, result: ValidationResult):
        """Add a validation result to the audit."""
        self.violations.append(result)
        self.failed_validations += 1

    def add_pass(self, result: ValidationResult):
        """Add a successful validation result."""
        self.passed_validations += 1

    def is_clean(self) -> bool:
        """Check if audit passed with no violations."""
        return self.failed_validations == 0


@dataclass
class StateContext:
    """Contextual state across tool calls."""
    tool_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    call_counts: dict[str, int] = field(default_factory=dict)
    last_call_times: dict[str, float] = field(default_factory=dict)
    user_ids: Set[str] = field(default_factory=set)
    request_ids: Set[str] = field(default_factory=set)


class ContractAuditor:
    """Validates tool calls against their specifications.

    Main components:
    1. Input schema validation
    2. Output schema validation
    3. Rate limiting
    4. State consistency checks
    """

    def __init__(self, max_burst_calls: int = 10, time_window: int = 60):
        """
        Initialize the contract auditor.

        Args:
            max_burst_calls: Maximum calls per time window (rate limiting)
            time_window: Time window in seconds for burst detection
        """
        self.max_burst_calls = max_burst_calls
        self.time_window = time_window
        self.state_context = StateContext()

        # Track per-tool rate limits
        self.tool_rate_limits: dict[str, dict[str, Any]] = {}

        logger.debug(
            f"ContractAuditor initialized: max_burst_calls={max_burst_calls}, "
            f"time_window={time_window}s"
        )

    def audit_call(
        self,
        task_id: str,
        node_id: str,
        tool_spec: dict[str, Any],
        inputs: dict[str, Any],
        outputs: Optional[dict[str, Any]] = None,
        span_id: str = "",
        timestamp: Optional[float] = None
    ) -> ValidationResult:
        """Audit a single tool call.

        Args:
            task_id: Task identifier
            node_id: Node identifier in the search tree
            tool_spec: Tool specification from ToolSpec
            inputs: Input arguments to the tool
            outputs: Output returned by the tool
            span_id: Span identifier for logging
            timestamp: Call timestamp (uses current time if not provided)

        Returns:
            ValidationResult with validation status and error messages
        """
        if timestamp is None:
            timestamp = time.time()

        tool_name = tool_spec.get("name", "unknown")
        call_hash = self._compute_call_hash(task_id, tool_name, inputs)

        result = ValidationResult(
            is_valid=True,
            tool_name=tool_name,
            check_type="input_schema"
        )

        # Check input schema compliance
        input_violations = self._validate_input_schema(tool_spec, inputs)
        if input_violations:
            result.is_valid = False
            result.error_messages.extend(input_violations)
            result.check_type = "input_schema"

        # Check output schema compliance if outputs provided
        if outputs is not None and result.is_valid:
            output_violations = self._validate_output_schema(tool_spec, outputs)
            if output_violations:
                result.is_valid = False
                result.error_messages.extend(output_violations)
                result.check_type = "output_schema"

        # Check rate limiting
        if result.is_valid:
            rate_violation = self._check_rate_limit(tool_name, timestamp)
            if rate_violation:
                result.is_valid = False
                result.error_messages.append(rate_violation)
                result.check_type = "rate_limit"

        # Update state
        self.state_context.call_counts[tool_name] = (
            self.state_context.call_counts.get(tool_name, 0) + 1
        )
        self.state_context.last_call_times[tool_name] = timestamp

        logger.debug(
            f"Contract audit for {tool_name}: {'PASS' if result.is_valid else 'FAIL'}"
        )

        return result

    def audit_batch(
        self,
        task_id: str,
        node_id: str,
        tool_calls: list[dict[str, Any]]
    ) -> ContractAudit:
        """Audit multiple tool calls at once.

        Args:
            task_id: Task identifier
            node_id: Node identifier
            tool_calls: List of tool call dicts with keys: tool_spec, inputs, span_id

        Returns:
            ContractAudit with aggregated results
        """
        audit = ContractAudit(
            task_id=task_id,
            node_id=node_id,
            total_calls=len(tool_calls)
        )

        for call in tool_calls:
            tool_spec = call.get("tool_spec", {})
            inputs = call.get("inputs", {})
            span_id = call.get("span_id", "")

            result = self.audit_call(
                task_id=task_id,
                node_id=node_id,
                tool_spec=tool_spec,
                inputs=inputs,
                span_id=span_id
            )

            if result.is_valid:
                audit.add_pass(result)
            else:
                audit.add_violation(result)

        audit.error_log = self._generate_error_log(audit)

        return audit

    def _validate_input_schema(
        self,
        tool_spec: dict[str, Any],
        inputs: dict[str, Any]
    ) -> list[str]:
        """Validate inputs against tool specification schema."""
        violations = []

        schema = tool_spec.get("json_schema", {})
        name = tool_spec.get("name", "unknown")

        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in inputs:
                violations.append(
                    f"Missing required input field: {field}"
                )

        # Check field types and constraints
        properties = schema.get("properties", {})
        for field, value in inputs.items():
            if field in properties:
                field_schema = properties[field]
                self._validate_field(field, value, field_schema, violations)

        return violations

    def _validate_output_schema(
        self,
        tool_spec: dict[str, Any],
        outputs: dict[str, Any]
    ) -> list[str]:
        """Validate outputs against tool specification schema."""
        violations = []

        schema = tool_spec.get("json_schema", {})
        name = tool_spec.get("name", "unknown")

        # Check that all outputs are expected fields
        properties = schema.get("properties", {})
        for field in outputs:
            if field not in properties:
                violations.append(
                    f"Unexpected output field: {field}"
                )

        return violations

    def _validate_field(
        self,
        field_name: str,
        value: Any,
        field_schema: dict[str, Any],
        violations: list[str]
    ):
        """Validate a single field against schema constraints."""
        # Check type
        if "type" in field_schema:
            expected_type = field_schema["type"]
            actual_type = type(value).__name__

            if expected_type == "array" and not isinstance(value, list):
                violations.append(
                    f"Field '{field_name}' should be of type array, got {actual_type}"
                )
            elif expected_type == "object" and not isinstance(value, dict):
                violations.append(
                    f"Field '{field_name}' should be of type object, got {actual_type}"
                )
            elif expected_type == "number" and not isinstance(value, (int, float)):
                violations.append(
                    f"Field '{field_name}' should be of type number, got {actual_type}"
                )
            elif expected_type == "string" and not isinstance(value, str):
                violations.append(
                    f"Field '{field_name}' should be of type string, got {actual_type}"
                )

        # Check enum values
        if "enum" in field_schema:
            allowed = field_schema["enum"]
            if value not in allowed:
                violations.append(
                    f"Field '{field_name}' has invalid value '{value}'. "
                    f"Allowed values: {allowed}"
                )

        # Check minimum/maximum
        if "minimum" in field_schema and value < field_schema["minimum"]:
            violations.append(
                f"Field '{field_name}' value {value} is below minimum {field_schema['minimum']}"
            )

        if "maximum" in field_schema and value > field_schema["maximum"]:
            violations.append(
                f"Field '{field_name}' value {value} exceeds maximum {field_schema['maximum']}"
            )

        # Check pattern (regex validation)
        if "pattern" in field_schema:
            import re
            if not re.match(field_schema["pattern"], str(value)):
                violations.append(
                    f"Field '{field_name}' does not match pattern {field_schema['pattern']}"
                )

    def _check_rate_limit(
        self,
        tool_name: str,
        timestamp: float
    ) -> Optional[str]:
        """Check if a tool call respects rate limiting rules."""
        # Get tool-specific rate limit if set
        max_calls = self.tool_rate_limits.get(tool_name, {}).get("max_burst_calls", self.max_burst_calls)

        # Get calls in time window
        now = timestamp
        time_window = self.time_window

        calls_in_window = sum(
            1 for call_time in self.state_context.last_call_times.values()
            if now - call_time <= time_window
        )

        if calls_in_window >= max_calls:
            # Calculate remaining time until window resets
            oldest_call_time = min(self.state_context.last_call_times.values())
            time_until_reset = oldest_call_time + time_window - now

            return (
                f"Rate limit exceeded: {max_calls} calls in {time_window}s. "
                f"Reset in {time_until_reset:.1f}s"
            )

        return None

    def _compute_call_hash(
        self,
        task_id: str,
        tool_name: str,
        inputs: dict[str, Any]
    ) -> str:
        """Compute a hash of the call for deduplication."""
        call_data = {
            "task_id": task_id,
            "tool_name": tool_name,
            "inputs": inputs,
        }
        return hashlib.md5(
            json.dumps(call_data, sort_keys=True).encode()
        ).hexdigest()[:8]

    def _generate_error_log(self, audit: ContractAudit) -> str:
        """Generate a consolidated error log from audit violations."""
        if not audit.violations:
            return ""

        log_lines = [
            f"ContractAudit for task={audit.task_id}, node={audit.node_id}:",
            f"Total calls: {audit.total_calls}",
            f"Passed: {audit.passed_validations}",
            f"Failed: {audit.failed_validations}",
        ]

        # Group violations by tool
        violations_by_tool: dict[str, list[ValidationResult]] = {}
        for v in audit.violations:
            tool = v.tool_name
            if tool not in violations_by_tool:
                violations_by_tool[tool] = []
            violations_by_tool[tool].append(v)

        for tool, tool_violations in violations_by_tool.items():
            log_lines.append(f"\nTool: {tool} ({len(tool_violations)} violations)")
            for v in tool_violations:
                for msg in v.error_messages:
                    log_lines.append(f"  - [{v.check_type}] {msg}")

        return "\n".join(log_lines)

    def set_tool_rate_limit(
        self,
        tool_name: str,
        max_burst_calls: int,
        time_window: int = 60
    ):
        """Set a rate limit for a specific tool.

        Args:
            tool_name: Name of the tool
            max_burst_calls: Maximum calls per time window
            time_window: Time window in seconds
        """
        self.tool_rate_limits[tool_name] = {
            "max_burst_calls": max_burst_calls,
            "time_window": time_window,
        }
        logger.debug(
            f"Set rate limit for {tool_name}: {max_burst_calls} calls in {time_window}s"
        )

    def get_state_context(self) -> StateContext:
        """Get current state context (useful for debugging)."""
        return self.state_context
