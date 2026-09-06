"""McNemar's test utilities for matched-pair binary outcomes.

Provides functions for performing McNemar's test (a chi-squared test on a
2x2 contingency table of paired binary outcomes) to compare two agents.

Author: AO Worker
Date: 2026-09-06
"""

from typing import Any, Dict, List, Tuple

from .statistical_gate import PairedResult


def build_mcnemar_table(
    paired_results: List[PairedResult],
) -> Dict[str, int]:
    """
    Build 2x2 contingency table from paired binary outcomes.

    The table is:
                | Child Correct | Child Incorrect
    ----------------------------------------------
    Parent Correct |       a       |        b
    Parent Incorrect|      c        |        d

    Args:
        paired_results: List of PairedResult objects with binary correctness

    Returns:
        Dictionary with keys 'a', 'b', 'c', 'd' for the table cells
    """
    a = sum(1 for r in paired_results if r.parent_correct and r.child_correct)
    b = sum(1 for r in paired_results if r.parent_correct and not r.child_correct)
    c = sum(1 for r in paired_results if not r.parent_correct and r.child_correct)
    d = sum(1 for r in paired_results if not r.parent_correct and not r.child_correct)

    return {
        'a': a,  # Both correct
        'b': b,  # Parent correct, child incorrect
        'c': c,  # Parent incorrect, child correct
        'd': d,  # Both incorrect
    }


def mcnemar_statistic(
    table: Dict[str, int],
    continuity_correction: bool = True,
) -> Tuple[float, float]:
    """
    Compute McNemar's test statistic (chi-squared) from contingency table.

    Args:
        table: Contingency table with keys 'a', 'b', 'c', 'd'
        continuity_correction: Apply continuity correction (default: True)

    Returns:
        Tuple of (chi_squared, degrees_of_freedom)
    """
    a = table.get('a', 0)
    b = table.get('b', 0)
    c = table.get('c', 0)
    d = table.get('d', 0)

    total = a + b + c + d

    if total == 0:
        return (0.0, 1.0)

    if b + c == 0:
        return (0.0, 1.0)

    if continuity_correction:
        # With continuity correction
        chi_squared = (abs(b - c) - 0.5) ** 2 / (b + c)
    else:
        # Without continuity correction
        chi_squared = (b - c) ** 2 / (b + c + 1e-10)

    return (chi_squared, 1.0)


def mcnemar_p_value(
    chi_squared: float,
    degrees_of_freedom: int = 1,
) -> float:
    """
    Compute p-value from chi-squared statistic.

    Args:
        chi_squared: Chi-squared statistic
        degrees_of_freedom: Degrees of freedom (typically 1 for McNemar)

    Returns:
        p-value (upper-tail probability)
    """
    # Clip chi_squared to avoid floating point errors
    chi_squared = max(chi_squared, 0.0)

    # Simple approximation: for chi-squared with df=1, use tail probability
    # For df != 1, we'd need scipy.stats.chi2.cdf, but we'll use a simple approximation

    if degrees_of_freedom != 1:
        # For other df, use a rough approximation
        p_value = 1.0 - min(chi_squared / degrees_of_freedom, 999.0)
    else:
        # For df=1, use approximation: p = exp(-chi_squared/2)
        # This is accurate for small chi-squared values
        p_value = 1.0 - min(
            1.0 - __chi2_cdf_approx(chi_squared, degrees_of_freedom),
            0.9999
        )

    return p_value


def __chi2_cdf_approx(x: float, df: int) -> float:
    """
    Approximate chi-squared CDF for small x.

    Args:
        x: Chi-squared statistic
        df: Degrees of freedom

    Returns:
        Approximate CDF value
    """
    if x <= 0:
        return 0.0
    if x >= 999:
        return 1.0

    # Simple approximation using upper incomplete gamma
    if df == 1:
        # Chi-squared with df=1 is related to standard normal
        # x = z^2, so p = P(Z > sqrt(x)) = 0.5 * erfc(sqrt(x)/sqrt(2))
        import math
        z = math.sqrt(x)
        return 0.5 * (1 - math.erf(z / math.sqrt(2)))
    else:
        # General case approximation
        k = df / 2.0 - 1.0
        return 1.0 - __incomplete_gamma(k + 0.5, x / 2.0) / __gamma(k + 0.5)


def __gamma(x: float) -> float:
    """Compute gamma function for small x."""
    import math
    # Stirling's approximation for gamma function
    if x > 0.5:
        return math.sqrt(2 * math.pi / x) * (x / math.e) ** x
    else:
        # Use recursion for small x
        if x <= 0:
            return float('inf')
        return math.pi / (math.sin(math.pi * x) * __gamma(1 - x))


def __incomplete_gamma(a: float, x: float) -> float:
    """Compute upper incomplete gamma function."""
    # Simple series approximation
    if x == 0:
        return 0.0

    import math
    series_sum = 0.0
    for n in range(100):
        term = (x ** a) / math.gamma(a + n + 1)
        series_sum += term
        if term < 1e-10:
            break

    gamma_total = math.gamma(a)
    return gamma_total * series_sum


def mcnemar_test(
    paired_results: List[PairedResult],
    continuity_correction: bool = True,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Perform McNemar's test for paired binary outcomes.

    Args:
        paired_results: List of PairedResult objects with binary correctness
        continuity_correction: Apply continuity correction (default: True)

    Returns:
        Tuple of (p_value, chi_squared, statistics)

    Statistical note: McNemar's test evaluates whether the proportion of
    successes differs between two matched pairs. A significant result
    indicates that the paired comparisons are not symmetric (i.e., the
    marginal homogeneity assumption is violated).
    """
    if len(paired_results) < 4:
        # Not enough samples for reliable McNemar test
        return (1.0, 0.0, {
            'n': len(paired_results),
            'reason': 'insufficient_samples'
        })

    # Build contingency table
    table = build_mcnemar_table(paired_results)

    # Compute test statistic
    chi_squared, dof = mcnemar_statistic(table, continuity_correction)

    # Compute p-value
    p_value = mcnemar_p_value(chi_squared, dof)

    # Additional statistics
    total = table['a'] + table['b'] + table['c'] + table['d']

    statistics = {
        'a': table['a'],  # Both correct
        'b': table['b'],  # Parent correct, child incorrect
        'c': table['c'],  # Parent incorrect, child correct
        'd': table['d'],  # Both incorrect
        'total': total,
        'parent_correct': table['a'] + table['c'],
        'child_correct': table['a'] + table['b'],
        'improvement_rate': table['a'] / total if total > 0 else 0.0,
        'child_worse_than_parent': table['c'] > table['b'],
        'same_as_parent': table['b'] + table['c'] == 0,
        'chi_squared': chi_squared,
        'degrees_of_freedom': dof,
        'test': 'mcnemar',
    }

    return (p_value, chi_squared, statistics)


def mcnemar_effect_size(
    table: Dict[str, int],
) -> float:
    """
    Compute effect size for McNemar's test.

    Uses the standard effect size measure for matched-pair proportions.

    Args:
        table: Contingency table with keys 'a', 'b', 'c', 'd'

    Returns:
        Effect size (0.0 to 1.0, typically < 0.5)
    """
    b = table.get('b', 0)
    c = table.get('c', 0)
    total = b + c

    if total == 0:
        return 0.0

    # Effect size = (b - c) / (b + c)
    # This measures the direction and magnitude of change
    effect_size = (b - c) / total if total > 0 else 0.0

    # Bound between 0 and 1
    effect_size = max(0.0, min(1.0, abs(effect_size)))

    return effect_size


def mcnemar_test_summary(
    p_value: float,
    chi_squared: float,
    table: Dict[str, int],
) -> Dict[str, Any]:
    """
    Generate human-readable summary of McNemar test results.

    Args:
        p_value: p-value from McNemar test
        chi_squared: Chi-squared statistic
        table: Contingency table

    Returns:
        Dictionary with human-readable summary
    """
    significance = "significant" if p_value < 0.05 else \
                    "marginally significant" if p_value < 0.10 else \
                    "not significant"

    improvement_direction = "child outperforms parent" if table['c'] > table['b'] else \
                          "parent outperforms child" if table['b'] > table['c'] else \
                          "no change"

    improvement_proportion = (table['c'] - table['b']) / (table['b'] + table['c']) if \
                             (table['b'] + table['c']) > 0 else 0.0

    return {
        'significance': significance,
        'improvement_direction': improvement_direction,
        'improvement_proportion': improvement_proportion,
        'p_value': p_value,
        'chi_squared': chi_squared,
        'b_to_c_ratio': table['c'] / table['b'] if table['b'] > 0 else float('inf'),
        'contingency_table': table,
    }


__all__ = [
    'build_mcnemar_table',
    'mcnemar_statistic',
    'mcnemar_p_value',
    'mcnemar_test',
    'mcnemar_effect_size',
    'mcnemar_test_summary',
]
