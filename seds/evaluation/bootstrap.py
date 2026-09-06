"""Bootstrap-based statistical test utilities for paired comparison.

Provides functions for generating bootstrap samples and computing p-values
for paired differences between parent and child agents.

Author: AO Worker
Date: 2026-09-06
"""

import random
from typing import Any, Callable, Dict, List, Tuple
from statistics import mean, stdev

from .statistical_gate import PairedResult


def bootstrap_resample(
    samples: List[Any],
    n_samples: int = 1000,
    seed: int = 42,
) -> List[List[Any]]:
    """
    Generate bootstrap samples from the original data.

    Args:
        samples: Original samples to resample
        n_samples: Number of bootstrap samples to generate
        seed: Random seed for reproducibility

    Returns:
        List of bootstrap samples
    """
    random.seed(seed)
    return [random.choices(samples, k=len(samples)) for _ in range(n_samples)]


def bootstrap_mean(
    sample: List[Any],
    n_samples: int = 1000,
    seed: int = 42,
) -> List[float]:
    """
    Generate bootstrap distribution of sample means.

    Args:
        sample: Original sample
        n_samples: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        List of bootstrap means
    """
    bootstrap_samples = bootstrap_resample(sample, n_samples, seed)
    return [mean(s) for s in bootstrap_samples]


def bootstrap_std(
    sample: List[Any],
    n_samples: int = 1000,
    seed: int = 42,
) -> List[float]:
    """
    Generate bootstrap distribution of sample standard deviations.

    Args:
        sample: Original sample
        n_samples: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        List of bootstrap standard deviations
    """
    bootstrap_samples = bootstrap_resample(sample, n_samples, seed)
    return [stdev(s) if len(s) > 1 else 0.0 for s in bootstrap_samples]


def bootstrap_p_value(
    paired_results: List[PairedResult],
    test_statistic: str = 'improvement',
    alternative: str = 'greater',
    n_samples: int = 1000,
    seed: int = 42,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute bootstrap p-value for paired comparison.

    Args:
        paired_results: List of PairedResult objects
        test_statistic: 'improvement' (default) or 'difference'
        alternative: 'greater', 'less', or 'two-sided'
        n_samples: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (p_value, statistics)
    """
    if len(paired_results) < 2:
        return (1.0, {'n': len(paired_results)})

    random.seed(seed)

    # Calculate observed test statistics
    if test_statistic == 'improvement':
        differences = [
            r.improvement for r in paired_results if r.improvement is not None
        ]
        observed_stat = mean(differences) if differences else 0.0
    else:  # 'difference'
        diffs = []
        for r in paired_results:
            if r.parent_score and r.child_score:
                diffs.append(r.child_score.reward - r.parent_score.reward)
        observed_stat = mean(diffs) if diffs else 0.0

    if not (test_statistic == 'improvement' and differences) and len(diffs) < 2:
        return (1.0, {'n_improvements': 0, 'n_comparisons': len(paired_results)})

    # Get bootstrap samples
    bootstrap_stats = []
    for _ in range(n_samples):
        sample = random.choices(
            differences if test_statistic == 'improvement' else diffs,
            k=len(differences if test_statistic == 'improvement' else diffs)
        )
        bootstrap_stats.append(mean(sample))

    bootstrap_stats.sort()

    # Calculate p-value
    if alternative == 'greater':
        p_value = sum(1 for bs in bootstrap_stats if bs > observed_stat) / n_samples
    elif alternative == 'less':
        p_value = sum(1 for bs in bootstrap_stats if bs < observed_stat) / n_samples
    else:
        p_value = 2 * min(
            sum(1 for bs in bootstrap_stats if bs > observed_stat) / n_samples,
            sum(1 for bs in bootstrap_stats if bs < observed_stat) / n_samples
        )

    statistics = {
        'n_samples': n_samples,
        'observed_statistic': observed_stat,
        'bootstrap_mean': mean(bootstrap_stats),
        'bootstrap_std': stdev(bootstrap_stats) if len(bootstrap_stats) > 1 else 0.0,
        'min_bootstrap': min(bootstrap_stats),
        'max_bootstrap': max(bootstrap_stats),
    }

    return (p_value, statistics)


def bootstrap_confidence_interval(
    sample: List[Any],
    confidence: float = 0.95,
    n_samples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for sample mean.

    Args:
        sample: Original sample
        confidence: Confidence level (0.0-1.0)
        n_samples: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_means = bootstrap_mean(sample, n_samples, seed)
    alpha = 1.0 - confidence

    lower = max(0.0, mean(bootstrap_means) - 1.96 * stdev(bootstrap_means)
                if len(bootstrap_means) > 1 else 0.0)
    upper = min(1.0, mean(bootstrap_means) + 1.96 * stdev(bootstrap_means)
                if len(bootstrap_means) > 1 else 0.0)

    return (lower, upper)


def bootstrap_effect_size(
    sample: List[Any],
) -> float:
    """
    Compute Cohen's d-like effect size for paired differences.

    Args:
        sample: List of paired differences

    Returns:
        Effect size (standardized difference)
    """
    if len(sample) < 2:
        return 0.0

    mean_diff = mean(sample)
    std_diff = stdev(sample)

    if std_diff == 0:
        return 0.0

    # Standardize by pooled standard deviation
    effect_size = mean_diff / std_diff

    return effect_size


def bootstrap_test_summary(
    p_value: float,
    effect_size: float,
    observed_statistic: float,
    bootstrap_std: float,
) -> Dict[str, Any]:
    """
    Generate human-readable summary of bootstrap test results.

    Args:
        p_value: Bootstrap p-value
        effect_size: Effect size measure
        observed_statistic: Observed test statistic
        bootstrap_std: Standard deviation of bootstrap distribution

    Returns:
        Dictionary with human-readable summary
    """
    significance = "significant" if p_value < 0.05 else \
                    "marginally significant" if p_value < 0.10 else \
                    "not significant"

    magnitude = "large" if abs(effect_size) >= 0.8 else \
                "medium" if abs(effect_size) >= 0.5 else \
                "small"

    direction = "positive" if observed_statistic > 0 else \
                "negative" if observed_statistic < 0 else \
                "null"

    return {
        'significance': significance,
        'magnitude': magnitude,
        'direction': direction,
        'p_value': p_value,
        'effect_size': effect_size,
        'observed_statistic': observed_statistic,
        'bootstrap_std': bootstrap_std,
    }


__all__ = [
    'bootstrap_resample',
    'bootstrap_mean',
    'bootstrap_std',
    'bootstrap_p_value',
    'bootstrap_confidence_interval',
    'bootstrap_effect_size',
    'bootstrap_test_summary',
]
