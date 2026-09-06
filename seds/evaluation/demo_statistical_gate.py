#!/usr/bin/env python3
"""
Demo of statistical promotion gate for Phase C evaluation.

Shows:
1. Genuine improvement: child agent significantly outperforms parent on k=20 tasks
2. Noise improvement: child agent shows random improvement but no statistical significance
"""

import sys
import os
import random
from statistics import mean

# Set PYTHONPATH to include parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after setting PYTHONPATH
from seds.evaluation import (
    PairedComparisonHarness,
    promote_or_reject,
)
from seds.domains.base import TaskDomain, Task, Score

# Simulate a simple domain with small number of tasks
def create_mini_domain(n_tasks: int = 30) -> TaskDomain:
    """Create a simple test domain for demonstration."""
    return TaskDomain(
        name="MiniDomain",
        goal="Test goal",
        tools=[],
        train_tasks=[
            Task(
                task_id=f"task_{i}",
                inputs={"test_input": f"Test task {i}"},
                reference=i % 2  # 0 or 1 based on index
            )
            for i in range(n_tasks)
        ],
        val_tasks=[
            Task(
                task_id=f"task_{i}",
                inputs={"test_input": f"Test task {i}"},
                reference=i % 2  # 0 or 1 based on index
            )
            for i in range(n_tasks)
        ],
        evaluate=lambda task, agent_output: (
            Score(correct=True, partial=1.0, detail={"reason": "output contains task_id"})
            if task.task_id in agent_output
            else Score(correct=False, partial=0.0, detail={"reason": "output missing task_id"})
        )
    )


def create_parent_scores(domain: TaskDomain, n_tasks: int = 20, seed: int = 42) -> list[Score]:
    """Create synthetic scores for parent agent."""
    random.seed(seed)
    scores = []

    for i in range(n_tasks):
        task = random.choice(domain.train_tasks)
        correct = (random.random() < 0.6)  # Parent baseline: 60% success rate
        partial = 1.0 if correct else 0.0
        scores.append(Score(
            correct=correct,
            partial=partial,
            detail={
                "task_id": task.task_id,
                "reward": 1.0 if correct else 0.0,
                "cost": 1.0
            }
        ))

    return scores


def create_child_scores_real_improvement(
    parent_scores: list[Score],
    n_tasks: int = 20,
    seed: int = 42
) -> list[Score]:
    """
    Create child scores showing genuine improvement.

    Child improves on most tasks, leads to statistically significant improvement.
    """
    random.seed(seed)
    scores = []

    # Parent baseline distribution
    parent_corrects = [s.correct for s in parent_scores]

    for i in range(n_tasks):
        # Child improves on most tasks (higher success rate)
        if parent_scores[i].correct:
            # Parent was correct, child also correct (improvement)
            child_correct = True
        else:
            # Parent was wrong, child correct (significant improvement)
            child_correct = random.random() < 0.7  # 70% success on difficult tasks

        scores.append(Score(
            correct=child_correct,
            partial=1.0 if child_correct else 0.0,
            detail={
                "task_id": parent_scores[i].detail.get("task_id", f"task_{i}"),
                "reward": 1.0 if child_correct else 0.0,
                "cost": 1.0
            }
        ))

    return scores


def create_child_scores_noise_improvement(
    parent_scores: list[Score],
    n_tasks: int = 20,
    seed: int = 42
) -> list[Score]:
    """
    Create child scores showing noise improvement.

    Child improves on some tasks but overall p-value is not significant.
    """
    random.seed(seed)
    scores = []

    # Parent baseline distribution
    parent_corrects = [s.correct for s in parent_scores]

    for i in range(n_tasks):
        # Simulate noise: child just as likely to be correct as parent
        # but has random variations
        child_correct = parent_scores[i].correct

        # Add some randomness
        if random.random() < 0.15:  # 15% random chance to be correct/incorrect
            child_correct = not child_correct

        scores.append(Score(
            correct=child_correct,
            partial=1.0 if child_correct else 0.0,
            detail={
                "task_id": parent_scores[i].detail.get("task_id", f"task_{i}"),
                "reward": 1.0 if child_correct else 0.0,
                "cost": 1.0
            }
        ))

    return scores


def create_child_scores_cost_overhead(
    parent_scores: list[Score],
    n_tasks: int = 20,
    seed: int = 42
) -> list[Score]:
    """
    Create child scores with genuine improvement but excessive cost.

    p-value is significant, but cost is too high, leading to REJECT.
    """
    random.seed(seed)
    scores = []

    for i in range(n_tasks):
        # Genuine improvement on some tasks
        child_correct = parent_scores[i].correct or random.random() < 0.5

        scores.append(Score(
            correct=child_correct,
            partial=1.0 if child_correct else 0.0,
            detail={
                "task_id": parent_scores[i].detail.get("task_id", f"task_{i}"),
                "reward": 1.0 if child_correct else 0.0,
                "cost": 1.5
            }
        ))

    return scores


def main():
    print("=" * 80)
    print("Statistical Promotion Gate Demo")
    print("=" * 80)
    print()

    # Initialize domain
    print("1. Setting up test domain (k=20 tasks)")
    print("-" * 80)
    domain = create_mini_domain(n_tasks=30)
    print(f"   Created domain with {len(domain.train_tasks)} tasks")
    print()

    # Case 1: Genuine improvement - PROMOTE
    print("2. Case 1: Genuine Improvement (Should PROMOTE)")
    print("-" * 80)

    harness1 = PairedComparisonHarness(domain, n_samples_per_epoch=20, seed=42)

    parent_scores_1 = create_parent_scores(domain, n_tasks=30, seed=42)
    child_scores_1 = create_child_scores_real_improvement(parent_scores_1, n_tasks=30, seed=42)

    print(f"   Parent pass rate: {sum(1 for s in parent_scores_1 if s.correct) / 30:.2%}")
    print(f"   Child pass rate:  {sum(1 for s in child_scores_1 if s.correct) / 30:.2%}")
    print()

    decision_1, p1, es1, stats1 = promote_or_reject(
        parent_scores_1,
        child_scores_1,
        domain,
        n_bootstrap=1000,
        threshold=0.10,
    )

    print(f"   Decision: {decision_1}")
    print(f"   p-value:  {p1:.4f}")
    effect_size_str = f"{es1:.4f}" if es1 is not None else "N/A (mcnemar)"
    print(f"   Effect size: {effect_size_str}")
    print(f"   Reason: {'PROMOTED (statistically significant)' if decision_1 == 'PROMOTE' else 'REJECTED'}")
    print()

    # Case 2: Noise improvement - REJECT
    print("3. Case 2: Noise Improvement (Should REJECT)")
    print("-" * 80)

    harness2 = PairedComparisonHarness(domain, n_samples_per_epoch=20, seed=43)

    parent_scores_2 = create_parent_scores(domain, n_tasks=20, seed=43)
    child_scores_2 = create_child_scores_noise_improvement(parent_scores_2, n_tasks=20, seed=43)

    print(f"   Parent pass rate: {sum(1 for s in parent_scores_2 if s.correct) / 20:.2%}")
    print(f"   Child pass rate:  {sum(1 for s in child_scores_2 if s.correct) / 20:.2%}")
    print()

    decision_2, p2, es2, stats2 = promote_or_reject(
        parent_scores_2,
        child_scores_2,
        domain,
        n_bootstrap=1000,
        threshold=0.10,
    )

    print(f"   Decision: {decision_2}")
    print(f"   p-value:  {p2:.4f}")
    effect_size_str = f"{es2:.4f}" if es2 is not None else "N/A (mcnemar)"
    print(f"   Effect size: {effect_size_str}")
    print(f"   Reason: {'PROMOTED (statistically significant)' if decision_2 == 'PROMOTE' else 'REJECTED (noise only)'}")
    print()

    # Case 3: Cost overhead - REJECT
    print("4. Case 3: Cost Overhead (Should REJECT)")
    print("-" * 80)

    harness3 = PairedComparisonHarness(domain, n_samples_per_epoch=20, seed=44)

    parent_scores_3 = create_parent_scores(domain, n_tasks=20, seed=44)
    child_scores_3 = create_child_scores_cost_overhead(parent_scores_3, n_tasks=20, seed=44)

    print(f"   Parent pass rate: {sum(1 for s in parent_scores_3 if s.correct) / 20:.2%}")
    print(f"   Child pass rate:  {sum(1 for s in child_scores_3 if s.correct) / 20:.2%}")
    print(f"   Parent cost:     {sum(s.detail['cost'] for s in parent_scores_3):.1f}")
    print(f"   Child cost:      {sum(s.detail['cost'] for s in child_scores_3):.1f}")
    print()

    decision_3, p3, es3, stats3 = promote_or_reject(
        parent_scores_3,
        child_scores_3,
        domain,
        n_bootstrap=1000,
        threshold=0.10,
    )

    print(f"   Decision: {decision_3}")
    print(f"   p-value:  {p3:.4f}")
    effect_size_str = f"{es3:.4f}" if es3 is not None else "N/A (mcnemar)"
    print(f"   Effect size: {effect_size_str}")
    print(f"   Parent cost: {stats3['parent_cost']:.1f}")
    print(f"   Child cost:  {stats3['child_cost']:.1f}")
    print(f"   Cost ratio:  {stats3['cost_ratio']:.2f}")
    print(f"   Reason: {'PROMOTED' if decision_3 == 'PROMOTE' else 'REJECTED (cost too high)'}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    cases = [
        ("Genuine improvement", decision_1, p1, es1, stats1),
        ("Noise improvement", decision_2, p2, es2, stats2),
        ("Cost overhead", decision_3, p3, es3, stats3),
    ]

    for case_name, decision, p_value, effect_size, stats in cases:
        print(f"{case_name}:")
        print(f"  Decision: {decision}")
        print(f"  p-value: {p_value:.4f} (threshold: 0.10)")
        effect_size_str = f"{effect_size:.4f}" if effect_size is not None else "N/A (mcnemar)"
        print(f"  Effect size: {effect_size_str}")
        if decision == 'REJECT' and 'reason' in stats:
            print(f"  Reason: {stats['reason']}")
        print()

    # Calibration test: identical distributions should give ~0.10 false-promotion rate
    print("=" * 80)
    print("CALIBRATION TEST")
    print("=" * 80)
    print()
    print("Running 200 tests with identical parent/child distributions (random matching)")
    print("Expected false-promotion rate: ~0.10 (based on random matching probability)")
    print("-" * 80)

    calibration_runs = 200
    threshold = 0.10
    false_promotions = 0
    p_values = []

    # Use deterministic seed for calibration
    calibration_seed = 999

    for i in range(calibration_runs):
        # Create parent scores with same seed for all runs
        parent_scores = create_parent_scores(domain, n_tasks=20, seed=calibration_seed + i)

        # Create child scores independently from same underlying distribution
        # Each task has a difficulty level (hard/easy) determined once
        # Both parent and child are sampled independently from the SAME difficulty
        child_scores = []
        for ps in parent_scores:
            # Determine difficulty of this task (same for both parent and child)
            # This represents the task's inherent difficulty/probability of being correct
            if ps.correct:
                # True positive: task is likely "easy"
                # Probability of child being correct: higher (e.g., 60%)
                pass_rate = 0.60
            else:
                # True negative: task is likely "hard"
                # Probability of child being correct: lower (e.g., 40%)
                pass_rate = 0.40

            # Sample child correctness independently using domain-defined pass rate
            # We use uniform random to sample from this probability distribution
            import random
            child_correct = random.uniform(0, 1) < pass_rate

            # Sample partial scores independently from a different distribution
            # This gives variance in partial correctness even when overall pass rate is similar
            child_partial = random.uniform(0, 1)

            # Copy detail but with independent sampling
            child_detail = ps.detail.copy()

            child_scores.append(Score(
                correct=child_correct,
                partial=child_partial,
                detail=child_detail
            ))

        # Run promotion test
        decision, p_value, _, _ = promote_or_reject(
            parent_scores,
            child_scores,
            domain,
            n_bootstrap=1000,
            threshold=threshold,
        )

        if decision == 'PROMOTE':
            false_promotions += 1
        p_values.append(p_value)

    false_promotion_rate = false_promotions / calibration_runs
    avg_p_value = mean(p_values)
    min_p_value = min(p_values)
    max_p_value = max(p_values)

    print(f"  Calibration runs: {calibration_runs}")
    print(f"  False-promotions: {false_promotions}")
    print(f"  False-promotion rate: {false_promotion_rate:.3f} (expected ~0.10)")
    print(f"  Average p-value: {avg_p_value:.4f}")
    print(f"  Min p-value: {min_p_value:.4f}")
    print(f"  Max p-value: {max_p_value:.4f}")
    print()

    # Check calibration: with identical distributions, p-values should cluster near 1.0
    # So false-promotion rate should be very low (no significant difference expected)
    calibration_ok = (false_promotion_rate <= 0.15)
    print("=" * 80)
    if calibration_ok:
        print("✓ Calibration PASSED")
        print(f"  False-promotion rate {false_promotion_rate:.3f} (expected ~0.0 for identical distributions)")
    else:
        print("✗ Calibration FAILED")
        print(f"  False-promotion rate {false_promotion_rate:.3f} (expected <= 0.15 for identical distributions)")
    print("=" * 80)
    print()

    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
