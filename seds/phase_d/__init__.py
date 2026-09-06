"""SEDS Phase D: Diagnostic Module (§6.4).

This module provides structured failure detection, contract auditing,
and domain verification capabilities for SEDS execution.

Components:
- failure_taxonomy: FailureCategory enum, Failure, FailureStatistics, FailureSeverity
- contract_auditor: ContractAuditor for input/output validation, rate limiting, state consistency
- ssf: RolloutSearch, PatternSearcher, search strategies
- do_ver: DomainVerifier for domain compliance, safety validation, answer validation, span integrity
"""
from __future__ import annotations

from seds.phase_d.do_ver import (
    DomainViolation,
    DomainVerifier,
    ValidationResult,
)

__all__ = [
    "DomainViolation",
    "DomainVerifier",
    "ValidationResult",
]
