"""Statistical evaluation module for SEDS Phase C node selection.

This module provides statistical tests for comparing parent and child agent
performance, including:
- Paired comparison harness
- Bootstrap-based significance testing
- McNemar's test for binary outcomes
- Promotion gate for making decisions based on statistical significance

Author: AO Worker
Date: 2026-09-06
"""

from .statistical_gate import (
    PairedComparisonHarness,
    bootstrap_test,
    mcnemar_test,
    promote_or_reject,
)
from .results import PairedResult
from .bootstrap import (
    bootstrap_resample,
    bootstrap_mean,
    bootstrap_std,
    bootstrap_p_value,
    bootstrap_confidence_interval,
    bootstrap_effect_size,
    bootstrap_test_summary,
)
from .mcnemar import (
    build_mcnemar_table,
    mcnemar_statistic,
    mcnemar_p_value,
    mcnemar_test,
    mcnemar_effect_size,
    mcnemar_test_summary,
)

__all__ = [
    # Main gate functions
    'PairedResult',
    'PairedComparisonHarness',
    'bootstrap_test',
    'mcnemar_test',
    'promote_or_reject',

    # Bootstrap utilities
    'bootstrap_resample',
    'bootstrap_mean',
    'bootstrap_std',
    'bootstrap_p_value',
    'bootstrap_confidence_interval',
    'bootstrap_effect_size',
    'bootstrap_test_summary',

    # McNemar utilities
    'build_mcnemar_table',
    'mcnemar_statistic',
    'mcnemar_p_value',
    'mcnemar_test',
    'mcnemar_effect_size',
    'mcnemar_test_summary',
]
