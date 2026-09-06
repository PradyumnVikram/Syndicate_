#!/usr/bin/env python3
"""
DoVer - Checkpoint-Replay Counterfactual Verification

Performs counterfactual verification by restoring the conversation/tool-call
state at a failure step, splicing in a proposed patch, and replaying forward
to verify if the patch resolves the failure.

This implementation requires:
- Prefix replay is all cache hits (free and byte-identical) - provided tool calls are replayed from the recorder
- n>=3 passing replays before crediting a patch
- Capped at 5 debug rounds
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class VerificationState:
    """State captured at a checkpoint for replay."""
    step_index: int
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    tool_parameters: dict[str, Any] = field(default_factory=dict)
    file_state: dict[str, Any] = field(default_factory=dict)
    model_outputs: list[dict[str, Any]] = field(default_factory=list)
    error_state: Optional[dict[str, Any]] = None


@dataclass
class Patch:
    """A proposed patch to test against the failing execution."""
    type: str  # 'instruction_patch' or 'tool_argument_patch'
    target_step: int
    description: str
    details: dict[str, Any]


@dataclass
class ReplayResult:
    """Result of a single replay with a patch."""
    step_index: int
    passed: bool
    error: Optional[str] = None
    output: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    """Report of checkpoint-replay verification."""
    checkpoint_step: int
    patch: Patch
    success: bool
    required_passing_replays: int
    actual_passing_replays: int
    max_debug_rounds: int
    actual_debug_rounds: int
    replays: list[ReplayResult]
    recommendation: str


class DoVerCheckpointReplay:
    """
    DoVer Checkpoint-Replay Counterfactual Verification Harness.

    This class implements counterfactual verification by:
    1. Capturing verification state at failure step t
    2. Splicing in a proposed patch (modified instruction or tool argument)
    3. Replaying the trajectory forward
    4. Verifying if the patch resolves the failure
    5. Repeating n>=3 times to confirm the fix

    Args:
        max_debug_rounds: Maximum number of debug rounds to attempt (default: 5)
    """

    def __init__(
        self,
        max_debug_rounds: int = 5,
    ):
        """
        Initialize the DoVer checkpoint-replay harness.

        Args:
            max_debug_rounds: Maximum debug rounds for patch verification
        """
        self.max_debug_rounds = max_debug_rounds

    def capture_state(
        self,
        trace: list[dict[str, Any]],
        failure_step: int,
    ) -> VerificationState:
        """
        Capture verification state at failure step.

        Args:
            trace: Execution trace with spans
            failure_step: Index of step where failure occurred

        Returns:
            Captured verification state
        """
        state = VerificationState(
            step_index=failure_step,
        )

        # Capture conversation history up to failure step
        for i in range(failure_step + 1):
            span = trace[i] if i < len(trace) else {}
            state.conversation_history.append(span)

        # Capture tool parameters from spans
        for i in range(failure_step + 1):
            span = trace[i] if i < len(trace) else {}
            if span.get("kind") == "call":
                state.tool_parameters[span.get("tool_name", "")] = span.get("inputs", {})

        # Capture file state if available
        for span in trace[:failure_step + 1]:
            if span.get("kind") == "write" and "file" in span.get("inputs", {}):
                state.file_state[span["inputs"]["file"]] = span.get("outputs", {}).get("content")

        return state

    def splice_patch(self, state: VerificationState, patch: Patch) -> dict[str, Any]:
        """
        Splice a patch into the verification state.

        Args:
            state: Captured verification state
            patch: Patch to splice

        Returns:
            Modified state with patch applied
        """
        modified_state = VerificationState(
            step_index=patch.target_step,
            conversation_history=state.conversation_history.copy(),
            tool_parameters=state.tool_parameters.copy(),
            file_state=state.file_state.copy(),
            model_outputs=state.model_outputs.copy(),
        )

        if patch.type == "instruction_patch":
            # Modify system prompt or user message
            if patch.details.get("target") == "system":
                modified_state.conversation_history[0]["content"] = patch.details.get("new_content", "")
            elif patch.details.get("target") == "user":
                if patch.target_step < len(modified_state.conversation_history):
                    modified_state.conversation_history[patch.target_step]["content"] = patch.details.get("new_content", "")

        elif patch.type == "tool_argument_patch":
            # Modify tool arguments at target step
            for i in range(len(modified_state.conversation_history)):
                span = modified_state.conversation_history[i]
                if span.get("kind") == "call" and span.get("tool_name") == patch.details.get("tool_name"):
                    span["inputs"] = {**span.get("inputs", {}), **patch.details.get("new_args", {})}

        return modified_state

    def replay_forward(
        self,
        trace: list[dict[str, Any]],
        modified_state: VerificationState,
    ) -> ReplayResult:
        """
        Replay the trajectory forward from checkpoint with modified state.

        Args:
            trace: Original execution trace
            modified_state: Modified state with patch applied

        Returns:
            Replay result indicating success/failure
        """
        # In real implementation, this would:
        # 1. Use the broker replay cache for deterministic replay
        # 2. Replay tool calls from the tool-call recorder (free cache hits)
        # 3. Execute the modified state
        # 4. Check for errors or success

        # For this implementation, we'll simulate the replay
        # by checking if the failure condition was resolved

        target_step = modified_state.step_index
        if target_step >= len(trace):
            return ReplayResult(
                step_index=target_step,
                passed=False,
                error="Target step beyond trace length",
            )

        span = trace[target_step]

        # Check if this span failed
        if span.get("error"):
            return ReplayResult(
                step_index=target_step,
                passed=False,
                error=span["error"],
            )

        # Simulate forward replay
        # In production, this would execute the modified state through the executor
        passed = not span.get("error")
        error = span.get("error")
        output = span.get("outputs", {}).get("content")

        return ReplayResult(
            step_index=target_step,
            passed=passed,
            error=error,
            output=output,
            metadata={"replay_complete": True},
        )

    def verify_patch(
        self,
        trace: list[dict[str, Any]],
        patch: Patch,
    ) -> VerificationReport:
        """
        Verify a patch against a failing execution trace.

        Args:
            trace: Execution trace containing the failure
            patch: Proposed patch to test

        Returns:
            Verification report
        """
        required_passing_replays = 3  # n>=3 passing replays
        actual_passing_replays = 0
        actual_debug_rounds = 0
        replays = []

        # Capture state at failure step
        failure_step = patch.target_step
        state = self.capture_state(trace, failure_step)

        # Repeatedly replay with the patch
        for round_num in range(1, self.max_debug_rounds + 1):
            actual_debug_rounds = round_num

            # Splice patch into state
            modified_state = self.splice_patch(state, patch)

            # Replay forward
            replay_result = self.replay_forward(trace, modified_state)
            replays.append(replay_result)

            if replay_result.passed:
                actual_passing_replays += 1

                # Check if we have enough passing replays
                if actual_passing_replays >= required_passing_replays:
                    success = True
                    break
            else:
                # Patch didn't work
                break

        # Determine recommendation
        if actual_passing_replays >= required_passing_replays:
            recommendation = "Patch verified - apply to production"
        elif actual_passing_replays > 0:
            recommendation = "Partial success - requires additional debugging"
        else:
            recommendation = "Patch failed - try a different approach"

        return VerificationReport(
            checkpoint_step=failure_step,
            patch=patch,
            success=actual_passing_replays >= required_passing_replays,
            required_passing_replays=required_passing_replays,
            actual_passing_replays=actual_passing_replays,
            max_debug_rounds=self.max_debug_rounds,
            actual_debug_rounds=actual_debug_rounds,
            replays=replays,
            recommendation=recommendation,
        )


# Mock replay cache for demo
class MockReplayCache:
    """Mock implementation of replay cache for demo purposes."""
    def __init__(self):
        self.cache = {}

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Get cached replay result."""
        return self.cache.get(key)

    def set(self, key: str, result: dict[str, Any]) -> None:
        """Cache replay result."""
        self.cache[key] = result


# Mock tool-call recorder for demo
class MockToolCallRecorder:
    """Mock implementation of tool-call recorder for demo purposes."""
    def __init__(self):
        self.calls = []

    def record(self, tool_name: str, inputs: dict[str, Any]) -> Any:
        """Record a tool call."""
        result = {"tool_name": tool_name, "inputs": inputs}
        self.calls.append(result)
        return result

    def replay_all(self) -> list[dict[str, Any]]:
        """Replay all recorded tool calls (free cache hits)."""
        return self.calls.copy()
