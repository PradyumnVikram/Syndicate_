#!/usr/bin/env python3
"""
SEDS Phase D Integration Demo

This script demonstrates how the Phase D components work together:
1. Failure detection and classification
2. Contract auditing
3. Structured search
4. Domain verification
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from phase_d.do_ver import (
    DomainViolation,
    DomainVerifier,
    ValidationResult,
)
from phase_d.failure_taxonomy import (
    Failure,
    FailureCategory,
    FailureSeverity,
    FailureStatistics,
)


def demo_domain_verification():
    """Demo: Domain verification and safety checks."""
    print("\n" + "="*80)
    print("DEMO: Domain Verification (DoVer)")
    print("="*80)

    verifier = DomainVerifier()

    # Define a safe domain
    safe_domain = {
        "name": "File Reader",
        "goal": "Read files safely without modification",
        "tools": [
            {"name": "cat", "description": "Read file contents"},
            {"name": "head", "description": "Show file headers"},
        ],
        "reference": "127.0.0.1 localhost\n192.168.1.1 gateway",
    }

    # Define a dangerous domain
    dangerous_domain = {
        "name": "System Administrator",
        "goal": "Perform system administration tasks",
        "tools": [
            {"name": "rm", "description": "Remove files"},
            {"name": "chmod", "description": "Change file permissions"},
            {"name": "dd", "description": "Disk operations"},
        ],
        "reference": "System admin completed",
    }

    # Safe spans (from actual tool calls)
    # Note: Using files without sensitive content to avoid false positives
    safe_spans = [
        {
            "span_id": "s1",
            "node_id": "node1",
            "kind": "call",
            "tool_name": "cat",
            "inputs": {"file": "/etc/hosts"},
            "outputs": {"content": "127.0.0.1 localhost\n192.168.1.1 gateway"},
        },
        {
            "span_id": "s2",
            "node_id": "node1",
            "kind": "call",
            "tool_name": "head",
            "inputs": {"file": "/etc/hostname", "lines": 5},
            "outputs": {"content": "syndicate-node-1"},
        },
    ]

    # Dangerous spans (contains forbidden patterns)
    dangerous_spans = [
        {
            "span_id": "d1",
            "node_id": "node1",
            "kind": "call",
            "tool_name": "rm",
            "inputs": {"file": "/etc/passwd", "recursive": True},
            "outputs": {"removed": "yes"},
        },
        {
            "span_id": "d2",
            "node_id": "node1",
            "kind": "call",
            "tool_name": "chmod",
            "inputs": {"file": "/etc/shadow", "mode": "777"},
            "outputs": {"changed": True},
        },
    ]

    # Safe answer
    safe_answer = "127.0.0.1 localhost\n192.168.1.1 gateway"

    # Dangerous answer (mismatched)
    dangerous_answer = "This is wrong content"

    # Verify safe domain
    safe_result = verifier.verify_domain_compliance(
        task_id="safe_task",
        node_id="node1",
        task_domain=safe_domain,
        spans=safe_spans,
        answer=safe_answer,
    )

    # Verify dangerous domain
    dangerous_result = verifier.verify_domain_compliance(
        task_id="dangerous_task",
        node_id="node1",
        task_domain=dangerous_domain,
        spans=dangerous_spans,
        answer=dangerous_answer,
    )

    print("\n📊 Safe Domain Verification:")
    print(f"   Status: {'PASS' if safe_result.is_valid else 'FAIL'}")
    print(f"   Score: {safe_result.validation_score:.2f}")
    print(f"   Violations: {len(safe_result.violations)}")
    if safe_result.violations:
        for v in safe_result.violations:
            print(f"   - [{v.severity}] {v.violation_type}: {v.message}")

    print("\n⚠️ Dangerous Domain Verification:")
    print(f"   Status: {'PASS' if dangerous_result.is_valid else 'FAIL'}")
    print(f"   Score: {dangerous_result.validation_score:.2f}")
    print(f"   Violations: {len(dangerous_result.violations)}")
    for v in dangerous_result.violations:
        print(f"   - [{v.severity}] {v.violation_type}: {v.message}")
        print(f"      Remediation: {v.remediation}")

    print("\n📈 Span Integrity Check:")
    safe_integrity = verifier.verify_span_integrity(safe_spans)
    dangerous_integrity = verifier.verify_span_integrity(dangerous_spans)

    print(f"   Safe spans: {safe_integrity['is_valid']} (digest: {safe_integrity['digest'][:16]}...)")
    print(f"   Dangerous spans: {dangerous_integrity['is_valid']} (digest: {dangerous_integrity['digest'][:16]}...)")

    print("\n📋 Domain Report:")
    safe_report = verifier.generate_domain_report(safe_domain, safe_spans, safe_result)
    print(f"   Task: {safe_report['task_name']}")
    print(f"   Tools used: {safe_report['tool_usage']}")

    return safe_result.is_valid and not dangerous_result.is_valid


def demo_simple_failure_detection():
    """Demo: Simple failure detection using built-in detectors."""
    print("\n" + "="*80)
    print("DEMO: Failure Detection (Simplified)")
    print("="*80)

    from phase_d import failure_taxonomy

    # Create a simple task execution with failures
    failures = [
        failure_taxonomy.Failure(
            failure_id="fail_001",
            category=failure_taxonomy.FailureCategory.RESOURCE_EXHAUSTION,
            severity=failure_taxonomy.FailureSeverity.ERROR,
            task_id="demo_task",
            node_id="node2",
            span_id="s5",
            message="Tool call timed out after 30 seconds",
            details={"tool_name": "grep", "timeout": 30},
        ),
        failure_taxonomy.Failure(
            failure_id="fail_002",
            category=failure_taxonomy.FailureCategory.SAFETY_VIOLATION,
            severity=failure_taxonomy.FailureSeverity.CRITICAL,
            task_id="demo_task",
            node_id="node2",
            span_id="s6",
            message="Unauthorized tool access attempt",
            details={"tool_name": "rm", "unauthorized": True},
        ),
    ]

    # Count failures by category and severity
    stats = failure_taxonomy.FailureStatistics()
    for fail in failures:
        stats.count(fail.category, fail.severity)

    print("\n📊 Failure Statistics:")
    print(f"   Total failures: {len(failures)}")
    print(f"   Categories:")
    for cat, count in stats.by_category.items():
        print(f"      - {cat.name}: {count}")
    print(f"   Severity Distribution:")
    for sev, count in stats.by_severity.items():
        print(f"      - {sev.name}: {count}")

    print("\n✓ Demo completed")

    return True


def main():
    """Run all Phase D demos."""
    print("\n" + "="*80)
    print("SEDS Phase D Integration Demo")
    print("§6.4 - Diagnostic Module")
    print("="*80)

    results = []

    try:
        results.append(("Domain Verification", demo_domain_verification()))
    except Exception as e:
        print(f"\n❌ Domain Verification Demo failed: {e}")
        results.append(("Domain Verification", False))

    try:
        results.append(("Failure Detection", demo_simple_failure_detection()))
    except Exception as e:
        print(f"\n❌ Failure Detection Demo failed: {e}")
        results.append(("Failure Detection", False))

    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")
        if not passed:
            all_passed = False

    print("\n" + "="*80)

    if all_passed:
        print("✓ All Phase D demos passed!")
        print("="*80)
        return 0
    else:
        print("❌ Some Phase D demos failed.")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
