"""SEDS Phase D: Diagnostic Module (§6.4).

This module provides semantic saliency folding (SSF) with precision detection
and checkpoint-replay counterfactual verification (DoVer) for self-improving
system daemon execution traces.

Components:
- ssf: SaliencyFoldedSpan, SemanticSaliencyFolder, debug_fold_spans for
       preserving real errors/tracebacks in highly compressed traces
- do_ver: DoVerCheckpointReplay for state capture, patch splicing, and replay
         forward verification to test counterfactual interventions
- failure_taxonomy: FailureCategory, Failure, FailureStatistics, FailureSeverity,
                    register_failure_detector, detect_failures, etc.
- contract_auditor: ContractAuditor, ValidationResult, ContractAudit, StateContext
                    for tool contract validation
"""
from __future__ import annotations

from seds.phase_d.do_ver import (
    MockReplayCache,
    MockToolCallRecorder,
    Patch,
    ReplayResult,
    DoVerCheckpointReplay,
    VerificationReport,
    VerificationState,
)

from seds.phase_d.failure_taxonomy import (
    Failure,
    FailureCategory,
    FailureSeverity,
    FailureStatistics,
    categorize_error,
    detect_failures,
    format_error_detector,
    get_severity_from_category,
    initialize_failure_taxonomy,
    planning_loop_detector,
    register_failure_detector,
    schema_violation_detector,
    timeout_detector,
    execution_error_detector,
)

from seds.phase_d.contract_auditor import (
    ContractAuditor,
    ContractAudit,
    StateContext,
    ValidationResult,
)

from seds.phase_d.ssf import (
    SaliencyFoldedSpan,
    SemanticSaliencyFolder,
    debug_fold_spans,
)

__all__ = [
    # DoVer components
    "DoVerCheckpointReplay",
    "MockReplayCache",
    "MockToolCallRecorder",
    "Patch",
    "ReplayResult",
    "VerificationReport",
    "VerificationState",
    # Failure taxonomy
    "Failure",
    "FailureCategory",
    "FailureSeverity",
    "FailureStatistics",
    "categorize_error",
    "detect_failures",
    "format_error_detector",
    "get_severity_from_category",
    "initialize_failure_taxonomy",
    "planning_loop_detector",
    "register_failure_detector",
    "schema_violation_detector",
    "timeout_detector",
    "execution_error_detector",
    # Contract auditor
    "ContractAuditor",
    "ContractAudit",
    "StateContext",
    "ValidationResult",
    # SSF
    "SaliencyFoldedSpan",
    "SemanticSaliencyFolder",
    "debug_fold_spans",
]
