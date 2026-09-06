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

    parent_scores_1 = create_parent_scores(domain, n_tasks=20, seed=42)
    child_scores_1 = create_child_scores_real_improvement(parent_scores_1, n_tasks=20, seed=42)

    print(f"   Parent pass rate: {sum(1 for s in parent_scores_1 if s.correct) / 20:.2%}")
    print(f"   Child pass rate:  {sum(1 for s in child_scores_1 if s.correct) / 20:.2%}")
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

    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
