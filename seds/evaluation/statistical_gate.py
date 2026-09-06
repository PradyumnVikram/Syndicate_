"""Statistical promotion gate for Phase C evaluation and statistics.

This module provides paired comparison harness, bootstrap/McNemar tests,
and a promotion gate that makes decisions based on statistical significance
with p<0.10 threshold and non-inferiority on cost.

Author: AO Worker
Date: 2026-09-06
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from statistics import mean, stdev

from seds.domains.base import TaskDomain, Task, Score
from seds.runtimes.sandbox.harness import EvaluationHarness


@dataclass
class PairedResult:
    """Result of paired comparison between parent and child."""
    task_id: str
    parent_score: Optional[Score]
    child_score: Optional[Score]
    domain: TaskDomain

    @property
    def parent_correct(self) -> bool:
        return self.parent_score is not None and self.parent_score.correct

    @property
    def child_correct(self) -> bool:
        return self.child_score is not None and self.child_score.correct

    @property
    def improvement(self) -> Optional[float]:
        """Calculate improvement score if both succeed."""
        if self.parent_correct and self.child_correct:
            return self.child_score.reward - self.parent_score.reward
        return None


class PairedComparisonHarness:
    """
    Paired comparison harness for comparing parent and child agents.

    Runs both agents on the same frozen task set to enable proper statistical
    analysis of performance differences.

    Handles:
    - Paired evaluation on same task subset
    - Reuse of existing EvaluationHarness
    - Result tracking and analysis
    """

    def __init__(
        self,
        domain: TaskDomain,
        n_samples_per_epoch: int = 20,
        seed: int = 42,
    ):
        """
        Initialize the paired comparison harness.

        Args:
            domain: The TaskDomain to evaluate
            n_samples_per_epoch: Number of tasks to sample per evaluation (k=20 for small samples)
            seed: Random seed for reproducibility
        """
        self.domain = domain
        self.n_samples_per_epoch = n_samples_per_epoch
        self.seed = seed
        self.harness = EvaluationHarness(domain, n_samples_per_epoch, seed)

        # Store paired results
        self.results: List[PairedResult] = []

    def compare_agents(
        self,
        agent_parent: Callable[[Task], Score],
        agent_child: Callable[[Task], Score],
        frozen_set: List[Task],
        max_same_seed_tasks: Optional[int] = None,
    ) -> List[PairedResult]:
        """
        Compare parent and child agents on the same task set.

        Args:
            agent_parent: Function that takes Task and returns Score
            agent_child: Function that takes Task and returns Score
            frozen_set: Frozen task set to evaluate on
            max_same_seed_tasks: Maximum number of tasks to evaluate with same seed
                                (useful for demonstrating seed sensitivity)

        Returns:
            List of PairedResult objects
        """
        self.results.clear()
        random.seed(self.seed)

        # Determine number of tasks to evaluate
        n_tasks = min(len(frozen_set), self.n_samples_per_epoch)
        if max_same_seed_tasks and n_tasks > max_same_seed_tasks:
            n_tasks = max_same_seed_tasks

        # Shuffle and sample
        sampled_tasks = random.sample(frozen_set, n_tasks)

        # Evaluate parent on all tasks
        parent_scores = [
            self.harness.domain.evaluate(task=task, host_output_reference="")
            for task in sampled_tasks
        ]

        # Evaluate child on all tasks
        child_scores = [
            agent_child(task)
            for task in sampled_tasks
        ]

        # Create paired results
        for i, task in enumerate(sampled_tasks):
            self.results.append(PairedResult(
                task_id=task.task_id,
                parent_score=parent_scores[i],
                child_score=child_scores[i],
                domain=self.domain
            ))

        return self.results

    def get_improvement_metrics(self) -> Dict[str, Any]:
        """
        Compute improvement metrics from paired results.

        Returns:
            Dictionary with improvement statistics
        """
        if not self.results:
            return {
                'n_evaluated': 0,
                'improvement_rate': 0.0,
                'p_value': None,
                'effect_size': None,
            }

        parent_correct = sum(1 for r in self.results if r.parent_correct)
        child_correct = sum(1 for r in self.results if r.child_correct)
        correct_improvements = sum(1 for r in self.results if r.improvement is not None)

        improvement_rate = correct_improvements / len(self.results) if self.results else 0.0

        return {
            'n_evaluated': len(self.results),
            'parent_correct': parent_correct,
            'child_correct': child_correct,
            'correct_improvements': correct_improvements,
            'improvement_rate': improvement_rate,
            'parent_pass_rate': parent_correct / len(self.results),
            'child_pass_rate': child_correct / len(self.results),
        }

    def clear_results(self) -> None:
        """Clear stored results."""
        self.results.clear()


def bootstrap_test(
    paired_results: List[PairedResult],
    n_samples: int = 1000,
    alternative: str = 'greater',
    seed: int = 42,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Perform bootstrap test for significance of paired differences.

    Args:
        paired_results: List of PairedResult objects
        n_samples: Number of bootstrap samples
        alternative: 'greater', 'less', or 'two-sided'
        seed: Random seed for reproducibility

    Returns:
        Tuple of (p_value, effect_size, statistics)
    """
    if len(paired_results) < 2:
        return (1.0, 0.0, {'n': len(paired_results)})

    random.seed(seed)

    # Calculate observed differences
    differences = [
        r.improvement for r in paired_results if r.improvement is not None
    ]

    if not differences:
        return (1.0, 0.0, {'n_improvements': 0, 'n_comparisons': len(paired_results)})

    observed_mean = mean(differences)
    observed_pvalue = 0.0

    # Bootstrap samples
    bootstrap_means = []
    for _ in range(n_samples):
        bootstrap_sample = random.choices(differences, k=len(differences))
        bootstrap_means.append(mean(bootstrap_sample))

    bootstrap_means.sort()

    # Calculate p-value
    if alternative == 'greater':
        # Proportion of bootstrap means > observed mean
        p_value = sum(1 for bm in bootstrap_means if bm > observed_mean) / n_samples
        # Include observed itself
        p_value = (p_value + (1.0 if observed_mean > mean(bootstrap_means) else 0.0)) / 2
    elif alternative == 'less':
        p_value = sum(1 for bm in bootstrap_means if bm < observed_mean) / n_samples
        p_value = (p_value + (1.0 if observed_mean < mean(bootstrap_means) else 0.0)) / 2
    else:  # two-sided
        p_value = 2 * min(
            sum(1 for bm in bootstrap_means if bm > observed_mean) / n_samples,
            sum(1 for bm in bootstrap_means if bm < observed_mean) / n_samples
        )
        p_value = (p_value + (1.0 if abs(observed_mean - mean(bootstrap_means)) > 0 else 0.0)) / 2

    # Effect size (Cohen's d equivalent)
    std_pool = (stdev(differences) if len(differences) > 1 else 1.0) / (len(differences) ** 0.5)
    effect_size = observed_mean / (std_pool + 1e-10)

    # Additional statistics
    statistics = {
        'n_improvements': len(differences),
        'n_comparisons': len(paired_results),
        'observed_mean': observed_mean,
        'bootstrap_std': stdev(bootstrap_means) if len(bootstrap_means) > 1 else 0.0,
        'bootstrap_mean': mean(bootstrap_means),
    }

    return (p_value, effect_size, statistics)


def mcnemar_test(
    paired_results: List[PairedResult],
    continuity_correction: bool = True,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Perform McNemar's test for matched-pair binary outcomes.

    Args:
        paired_results: List of PairedResult objects with binary correctness
        continuity_correction: Apply continuity correction to chi-squared

    Returns:
        Tuple of (p_value, chi_squared, statistics)
    """
    if len(paired_results) < 4:
        # Not enough samples for McNemar
        return (1.0, 0.0, {'n': len(paired_results)})

    # Build 2x2 contingency table
    #               | Child Correct | Child Incorrect
    # ----------------------------------------------
    # Parent Correct |     a         |       b
    # Parent Incorrect|     c         |       d

    a = sum(1 for r in paired_results if r.parent_correct and r.child_correct)
    b = sum(1 for r in paired_results if r.parent_correct and not r.child_correct)
    c = sum(1 for r in paired_results if not r.parent_correct and r.child_correct)
    d = sum(1 for r in paired_results if not r.parent_correct and not r.child_correct)

    # McNemar statistic (chi-squared)
    if continuity_correction:
        # With continuity correction
        chi_squared = (abs(b - c) - 0.5) ** 2 / ((b + c) / 2)
    else:
        # Without continuity correction
        chi_squared = (b - c) ** 2 / (b + c + 1e-10)

    # Convert to p-value (chi-squared with 1 degree of freedom)
    p_value = 1.0 - min(chi_squared, 999.0)  # Cap at 999 for safety

    # Statistical significance threshold (typical: p < 0.05)
    # For promotion gate, we use p < 0.10

    statistics = {
        'a': a,  # Both correct
        'b': b,  # Parent correct, child incorrect
        'c': c,  # Parent incorrect, child correct
        'd': d,  # Both incorrect
        'n_correct': a + c,  # Parent correct
        'n_child_correct': a + b,  # Child correct
        'success_rate_parent': (a + c) / len(paired_results),
        'success_rate_child': (a + b) / len(paired_results),
        'improvement_rate': a / len(paired_results) if len(paired_results) > 0 else 0.0,
    }

    return (p_value, chi_squared, statistics)


def promote_or_reject(
    parent_scores: List[Score],
    child_scores: List[Score],
    domain: TaskDomain,
    n_bootstrap: int = 1000,
    threshold: float = 0.10,
    use_mcnemar: bool = True,
) -> Tuple[str, float, float, Dict[str, Any]]:
    """
    Promotion gate: make decision to promote or reject based on statistical significance.

    Args:
        parent_scores: List of scores from parent agent
        child_scores: List of scores from child agent
        domain: TaskDomain for cost comparison
        n_bootstrap: Number of bootstrap samples for test
        threshold: Significance threshold (default: p < 0.10 for promotion)
        use_mcnemar: Use McNemar test if both agents produce binary outcomes

    Returns:
        Tuple of (decision, p_value, effect_size, statistics)
        Decision: 'PROMOTE', 'REJECT', or 'UNCERTAIN'
        p_value: Statistical significance (lower is better)
        effect_size: Magnitude of improvement
        statistics: Detailed statistics
    """
    if len(parent_scores) != len(child_scores):
        raise ValueError(f"Mismatched lengths: parent={len(parent_scores)}, child={len(child_scores)}")

    if len(parent_scores) < 2:
        return ('REJECT', 1.0, 0.0, {
            'n_samples': len(parent_scores),
            'reason': 'insufficient_samples'
        })

    # Check if both have binary outcomes (correct/incorrect)
    parent_corrects = [1 if s.correct else 0 for s in parent_scores]
    child_corrects = [1 if s.correct else 0 for s in child_scores]

    use_binary_test = (parent_corrects and child_corrects and
                      len(set(parent_corrects)) > 1 and
                      len(set(child_corrects)) > 1)

    # Choose appropriate test
    effect_size = None  # Initialize for mcnemar case
    if use_binary_test and use_mcnemar:
        p_value, chi_squared, statistics = mcnemar_test(
            [
                PairedResult(f'task_{i}', parent_scores[i], child_scores[i], domain)
                for i in range(len(parent_scores))
            ]
        )
        test_name = 'mcnemar'
        test_result = f"χ²={chi_squared:.3f}, p={p_value:.4f}"
    else:
        p_value, effect_size, statistics = bootstrap_test(
            [
                PairedResult(f'task_{i}', parent_scores[i], child_scores[i], domain)
                for i in range(len(parent_scores))
            ],
            n_samples=n_bootstrap
        )
        test_name = 'bootstrap'
        test_result = f"Mean Δ={statistics.get('observed_mean', 0):.4f}, "
        test_result += f"p={p_value:.4f}"

    # Cost comparison (non-inferiority)
    parent_cost = sum(s.detail['cost'] for s in parent_scores)
    child_cost = sum(s.detail['cost'] for s in child_scores)

    if child_cost > parent_cost * 1.2:
        # Child is significantly more expensive (20% overhead)
        return ('REJECT', p_value, effect_size, {
            **statistics,
            'test_name': test_name,
            'test_result': test_result,
            'parent_cost': parent_cost,
            'child_cost': child_cost,
            'cost_ratio': child_cost / parent_cost if parent_cost > 0 else 999.0,
            'reason': 'cost_overhead'
        })

    # Promotion decision
    if p_value < threshold:
        decision = 'PROMOTE'
    elif p_value >= (1 - threshold):
        decision = 'REJECT'
    else:
        decision = 'UNCERTAIN'

    result = {
        **statistics,
        'test_name': test_name,
        'test_result': test_result,
        'parent_cost': parent_cost,
        'child_cost': child_cost,
        'cost_ratio': child_cost / parent_cost if parent_cost > 0 else 999.0,
        'decision': decision,
        'threshold': threshold,
    }

    return (decision, p_value, effect_size, result)


__all__ = [
    'PairedResult',
    'PairedComparisonHarness',
    'bootstrap_test',
    'mcnemar_test',
    'promote_or_reject',
]
