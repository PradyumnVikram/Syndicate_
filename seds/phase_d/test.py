#!/usr/bin/env python3
"""
SEDS Phase D Unit Tests

Tests for the Phase D components:
- failure_taxonomy: Failure detection and classification
- contract_auditor: Tool contract validation
- ssf: Structured search framework
- do_ver: Domain verification
"""
from __future__ import annotations

import unittest

from seds.phase_d.do_ver import (
    DomainViolation,
    DomainVerifier,
    ValidationResult,
)
from seds.phase_d.failure_taxonomy import (
    Failure,
    FailureCategory,
    FailureSeverity,
    FailureStatistics,
)


class TestDomainViolation(unittest.TestCase):
    """Test DomainViolation dataclass."""

    def test_domain_violation_creation(self):
        """Test creating a domain violation."""
        violation = DomainViolation(
            violation_type="test_violation",
            severity="CRITICAL",
            task_id="task_123",
            node_id="node_1",
            message="Test violation message",
            details={"key": "value"},
            remediation="Fix the issue",
        )

        self.assertEqual(violation.violation_type, "test_violation")
        self.assertEqual(violation.severity, "CRITICAL")
        self.assertEqual(violation.task_id, "task_123")
        self.assertEqual(violation.node_id, "node_1")
        self.assertEqual(violation.message, "Test violation message")
        self.assertTrue(violation.details)
        self.assertEqual(violation.remediation, "Fix the issue")


class TestDomainVerifier(unittest.TestCase):
    """Test DomainVerifier functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.verifier = DomainVerifier()

    def test_verify_domain_compliance_safe_domain(self):
        """Test compliance verification with a safe domain."""
        safe_domain = {
            "name": "Safe Domain",
            "goal": "Test goal",
            "tools": [
                {"name": "tool1", "description": "Tool 1"},
                {"name": "tool2", "description": "Tool 2"},
            ],
            "reference": "expected answer",
        }

        safe_spans = [
            {
                "span_id": "s1",
                "node_id": "node1",
                "kind": "call",
                "tool_name": "tool1",
                "inputs": {"arg": "value"},
                "outputs": {"result": "ok"},
            },
        ]

        result = self.verifier.verify_domain_compliance(
            task_id="test_task",
            node_id="node1",
            task_domain=safe_domain,
            spans=safe_spans,
            answer="expected answer",
        )

        self.assertTrue(result.is_valid)
        self.assertGreater(result.validation_score, 0.0)
        self.assertEqual(result.metadata["domains_checked"], [
            "tool_compliance",
            "safety",
            "answer",
            "span_consistency"
        ])

    def test_verify_domain_compliance_forbidden_tool(self):
        """Test compliance verification with a forbidden tool."""
        domain = {
            "name": "Restricted Domain",
            "goal": "Restricted goal",
            "tools": [
                {"name": "safe_tool", "description": "Safe tool"},
            ],
            "reference": "expected answer",
        }

        spans = [
            {
                "span_id": "s1",
                "node_id": "node1",
                "kind": "call",
                "tool_name": "forbidden_tool",  # This is not allowed!
                "inputs": {},
                "outputs": {},
            },
        ]

        result = self.verifier.verify_domain_compliance(
            task_id="test_task",
            node_id="node1",
            task_domain=domain,
            spans=spans,
            answer="expected answer",
        )

        self.assertFalse(result.is_valid, f"Expected failure but got valid. Violations: {result.violations}")
        violations = [v for v in result.violations if v.violation_type == "forbidden_tool"]
        self.assertTrue(len(violations) > 0, f"No forbidden_tool violations found: {result.violations}")
        self.assertEqual(violations[0].details["tool_name"], "forbidden_tool")
        self.assertNotIn("forbidden_tool", violations[0].details["allowed_tools"])

    def test_verify_domain_compliance_safety_violation(self):
        """Test safety violation detection."""
        domain = {
            "name": "Safe Domain",
            "goal": "Test goal",
            "tools": [],
            "reference": "expected answer",
        }

        spans = [
            {
                "span_id": "s1",
                "node_id": "node1",
                "kind": "call",
                "tool_name": "rm",
                "inputs": {"cmd": "rm -rf /"},
                "outputs": {},
            },
        ]

        result = self.verifier.verify_domain_compliance(
            task_id="test_task",
            node_id="node1",
            task_domain=domain,
            spans=spans,
            answer="expected answer",
        )

        self.assertFalse(result.is_valid)
        safety_violations = [v for v in result.violations if v.violation_type == "safety_violation"]
        self.assertTrue(len(safety_violations) > 0)
        self.assertIn("rm -rf /", safety_violations[0].message)

    def test_verify_domain_compliance_answer_mismatch(self):
        """Test answer mismatch detection."""
        domain = {
            "name": "Test Domain",
            "goal": "Test goal",
            "tools": [],
            "reference": "expected answer",
        }

        result = self.verifier.verify_domain_compliance(
            task_id="test_task",
            node_id="node1",
            task_domain=domain,
            spans=[],
            answer="wrong answer",  # Mismatch!
        )

        self.assertFalse(result.is_valid, f"Expected failure but got valid. Violations: {result.violations}")
        answer_violations = [v for v in result.violations if v.violation_type == "answer_mismatch"]
        self.assertTrue(len(answer_violations) > 0, f"No answer_mismatch violations found: {result.violations}")

    def test_verify_span_integrity(self):
        """Test span integrity verification."""
        valid_spans = [
            {
                "span_id": "s1",
                "node_id": "node1",
                "kind": "call",
                "tool_name": "tool1",
                "inputs": {},
                "outputs": {"result": "ok"},
            },
            {
                "span_id": "s2",
                "node_id": "node2",
                "kind": "call",
                "tool_name": "tool2",
                "inputs": {},
                "outputs": {"result": "ok"},
            },
        ]

        invalid_spans = [
            {
                "span_id": "s1",
                "node_id": "node1",
                "kind": "call",
                "tool_name": "tool1",
                # Missing outputs!
            },
        ]

        valid_result = self.verifier.verify_span_integrity(valid_spans)
        invalid_result = self.verifier.verify_span_integrity(invalid_spans)

        self.assertTrue(valid_result["is_valid"])
        self.assertEqual(valid_result["span_count"], 2)
        self.assertIn("digest", valid_result)

        self.assertTrue(invalid_result["is_valid"])  # Still valid, just missing outputs
        self.assertEqual(invalid_result["all_spans_complete"], False)

    def test_generate_domain_report(self):
        """Test domain report generation."""
        domain = {
            "name": "Test Domain",
            "goal": "Test goal",
            "tools": [
                {"name": "tool1", "description": "Tool 1"},
            ],
            "reference": "expected answer",
        }

        spans = [
            {
                "span_id": "s1",
                "node_id": "node1",
                "kind": "call",
                "tool_name": "tool1",
                "inputs": {},
                "outputs": {"result": "ok"},
            },
        ]

        result = ValidationResult(
            is_valid=True,
            violations=[],
            validation_score=0.9,
            metadata={"tool_count": 1},
        )

        report = self.verifier.generate_domain_report(domain, spans, result)

        self.assertEqual(report["task_name"], "Test Domain")
        self.assertEqual(report["validation_score"], 0.9)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["violations_found"], 0)
        self.assertEqual(report["tool_usage"]["tool1"], 1)


class TestFailureStatistics(unittest.TestCase):
    """Test FailureStatistics functionality."""

    def test_count_failures(self):
        """Test counting failures by category and severity."""
        stats = FailureStatistics()

        # Count some failures
        stats.count(FailureCategory.CONTRACT_VIOLATION, FailureSeverity.ERROR)
        stats.count(FailureCategory.SAFETY_VIOLATION, FailureSeverity.CRITICAL)
        stats.count(FailureCategory.SAFETY_VIOLATION, FailureSeverity.WARNING)

        self.assertEqual(stats.total_failures, 3)
        self.assertEqual(stats.by_category[FailureCategory.CONTRACT_VIOLATION], 1)
        self.assertEqual(stats.by_category[FailureCategory.SAFETY_VIOLATION], 2)
        self.assertEqual(stats.by_severity[FailureSeverity.ERROR], 1)
        self.assertEqual(stats.by_severity[FailureSeverity.CRITICAL], 1)
        self.assertEqual(stats.by_severity[FailureSeverity.WARNING], 1)

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
            category=FailureCategory.SAFETY_VIOLATION,
            severity=FailureSeverity.CRITICAL,
            task_id="task_123",
            node_id="node_1",
            span_id="s1",
            message="Test failure message",
            details={"key": "value"},
            remediation="Fix the issue",
        )

        self.assertEqual(failure.failure_id, "fail_001")
        self.assertEqual(failure.category, FailureCategory.SAFETY_VIOLATION)
        self.assertEqual(failure.severity, FailureSeverity.CRITICAL)
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
    suite.addTests(loader.loadTestsFromTestCase(TestDomainViolation))
    suite.addTests(loader.loadTestsFromTestCase(TestDomainVerifier))
    suite.addTests(loader.loadTestsFromTestCase(TestFailureStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestFailureDataclass))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_tests())
