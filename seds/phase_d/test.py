#!/usr/bin/env python3
"""
SEDS Phase D Unit Tests

Tests for the Phase D components:
- ssf: Saliency folding for preserving real errors
- do_ver: Checkpoint-replay counterfactual verification
- failure_taxonomy: Failure detection and classification
- contract_auditor: Tool contract validation
"""
from __future__ import annotations

import unittest

from seds.phase_d.do_ver import (
    DoVerCheckpointReplay,
    MockReplayCache,
    MockToolCallRecorder,
    Patch,
    ReplayResult,
    VerificationReport,
    VerificationState,
)

from seds.phase_d.failure_taxonomy import (
    Failure,
    FailureCategory,
    FailureSeverity,
    FailureStatistics,
)


class TestVerificationState(unittest.TestCase):
    """Test VerificationState dataclass."""

    def test_verification_state_creation(self):
        """Test creating a verification state."""
        state = VerificationState(
            step_index=2,
            conversation_history=[{"test": "data"}],
        )

        self.assertEqual(state.step_index, 2)
        self.assertEqual(len(state.conversation_history), 1)


class TestPatch(unittest.TestCase):
    """Test Patch dataclass."""

    def test_patch_creation(self):
        """Test creating a patch."""
        patch = Patch(
            type="instruction_patch",
            target_step=2,
            description="Fix failing test by removing print statements",
            details={"target": "user", "new_content": "execute_python(code='assert True')"},
        )

        self.assertEqual(patch.type, "instruction_patch")
        self.assertEqual(patch.target_step, 2)
        self.assertEqual(patch.description, "Fix failing test by removing print statements")
        self.assertEqual(patch.details["target"], "user")


class TestReplayResult(unittest.TestCase):
    """Test ReplayResult dataclass."""

    def test_replay_result_pass(self):
        """Test creating a passing replay result."""
        result = ReplayResult(step_index=0, passed=True, error=None)

        self.assertTrue(result.passed)
        self.assertIsNone(result.error)
        self.assertEqual(result.step_index, 0)

    def test_replay_result_fail(self):
        """Test creating a failing replay result."""
        result = ReplayResult(step_index=0, passed=False, error="RuntimeError: Test failed")

        self.assertFalse(result.passed)
        self.assertEqual(result.error, "RuntimeError: Test failed")
        self.assertEqual(result.step_index, 0)


class TestVerificationReport(unittest.TestCase):
    """Test VerificationReport dataclass."""

    def test_report_creation(self):
        """Test creating a verification report."""
        patch = Patch(
            type="instruction_patch",
            target_step=2,
            description="Test patch",
            details={"target": "user", "new_content": "execute_python(code='assert True')"},
        )
        report = VerificationReport(
            checkpoint_step=2,
            patch=patch,
            success=True,
            required_passing_replays=3,
            actual_passing_replays=3,
            max_debug_rounds=5,
            actual_debug_rounds=3,
            replays=[
                ReplayResult(step_index=0, passed=True, error=None),
                ReplayResult(step_index=1, passed=True, error=None),
                ReplayResult(step_index=2, passed=True, error=None),
            ],
            recommendation="Patch verified successfully",
        )

        self.assertEqual(report.checkpoint_step, 2)
        self.assertEqual(report.actual_debug_rounds, 3)
        self.assertEqual(report.required_passing_replays, 3)
        self.assertEqual(report.actual_passing_replays, 3)
        self.assertEqual(report.recommendation, "Patch verified successfully")
        self.assertEqual(len(report.replays), 3)


class TestFailureStatistics(unittest.TestCase):
    """Test FailureStatistics functionality."""

    def test_count_failures(self):
        """Test counting failures by category and severity."""
        stats = FailureStatistics()

        # Count some failures
        stats.count(FailureCategory.SCHEMA_VIOLATION, FailureSeverity.ERROR)
        stats.count(FailureCategory.TOOL_MISUSE, FailureSeverity.WARNING)
        stats.count(FailureCategory.TOOL_MISUSE, FailureSeverity.WARNING)

        self.assertEqual(stats.total_failures, 3)
        self.assertEqual(stats.by_category[FailureCategory.SCHEMA_VIOLATION], 1)
        self.assertEqual(stats.by_category[FailureCategory.TOOL_MISUSE], 2)
        self.assertEqual(stats.by_severity[FailureSeverity.ERROR], 1)
        self.assertEqual(stats.by_severity[FailureSeverity.WARNING], 2)

    def test_top_messages(self):
        """Test top messages functionality."""
        stats = FailureStatistics()

        # Add some messages
        stats.add_message("Error message 1", count=3)
        stats.add_message("Error message 2", count=1)
        stats.add_message("Error message 3", count=2)

        top_messages = stats.get_top_n_messages(n=2)

        self.assertEqual(len(top_messages), 2)
        self.assertEqual(top_messages[0][0], "Error message 1")
        self.assertEqual(top_messages[0][1], 3)


class TestFailureDataclass(unittest.TestCase):
    """Test Failure dataclass."""

    def test_failure_creation(self):
        """Test creating a failure."""
        failure = Failure(
            failure_id="fail_001",
            category=FailureCategory.SCHEMA_VIOLATION,
            severity=FailureSeverity.ERROR,
            task_id="task_123",
            node_id="node_1",
            span_id="s1",
            message="Test failure message",
            details={"key": "value"},
            remediation="Fix the issue",
        )

        self.assertEqual(failure.failure_id, "fail_001")
        self.assertEqual(failure.category, FailureCategory.SCHEMA_VIOLATION)
        self.assertEqual(failure.severity, FailureSeverity.ERROR)
        self.assertEqual(failure.task_id, "task_123")
        self.assertEqual(failure.node_id, "node_1")
        self.assertEqual(failure.span_id, "s1")
        self.assertEqual(failure.message, "Test failure message")
        self.assertTrue(failure.details)
        self.assertEqual(failure.remediation, "Fix the issue")


def run_tests():
    """Run all tests and return success status."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVerificationState))
    suite.addTests(loader.loadTestsFromTestCase(TestPatch))
    suite.addTests(loader.loadTestsFromTestCase(TestReplayResult))
    suite.addTests(loader.loadTestsFromTestCase(TestVerificationReport))
    suite.addTests(loader.loadTestsFromTestCase(TestFailureStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestFailureDataclass))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit(run_tests())
