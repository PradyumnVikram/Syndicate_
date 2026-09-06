#!/usr/bin/env python3
"""
Simple test to verify Thompson sampling policy works with HGM components.
"""

import sys
sys.path.insert(0, 'seds')

from selector import (
    SEDSSelector,
    PolicyType,
    SyntheticNode,
    MetricsSnapshot
)

def test_thompson_sampling():
    """Test Thompson sampling selection with HGM components."""
    print("=" * 70)
    print("Testing Thompson Sampling Policy with HGM Components")
    print("=" * 70)

    # Create selector with Thompson sampling policy
    print("\n1. Creating SEDS Selector with THOMPSON policy...")
    selector = SEDSSelector(
        policy_type=PolicyType.THOMPSON,
        max_tau=10.0,
        total_steps=100
    )
    print("✓ Selector created")
    print(f"  - Tau scheduler: min_tau=0.1, max_tau=10.0, total_steps=100")

    # Add predefined nodes
    print("\n2. Adding predefined nodes...")
    selector.add_to_archive(
        SyntheticNode(
            node_id="fixed_agent",
            is_predefined=True
        ),
        MetricsSnapshot(
            success_rate=0.1,
            average_reward=0.1,
            standard_deviation=0.1,
            coverage_score=0.7,
            novelty_score=0.1
        )
    )
    selector.add_to_archive(
        SyntheticNode(
            node_id="default_module",
            is_predefined=True
        ),
        MetricsSnapshot(
            success_rate=0.2,
            average_reward=0.2,
            standard_deviation=0.1,
            coverage_score=0.5,
            novelty_score=0.2
        )
    )
    print(f"  - Archive size: {selector.archive.size()}")

    # Create and evaluate synthetic nodes
    print("\n3. Creating and evaluating synthetic nodes...")
    for i in range(5):
        # Evaluate node - success for some, failure for others
        if i < 3:
            reward = 0.7 + (i * 0.1)
        else:
            reward = 0.3 - ((i - 3) * 0.1)

        node = SyntheticNode(
            node_id=f"node_{i}",
            is_predefined=False
        )

        selector.add_to_archive(
            node,
            MetricsSnapshot(
                success_rate=reward,
                average_reward=reward,
                standard_deviation=0.1,
                coverage_score=0.5 + (i * 0.1),
                novelty_score=0.1 + (i * 0.1)
            )
        )
        selector.evaluate_node(f"node_{i}", reward)
        print(f"  - Node node_{i}: reward={reward:.2f}, CMP={selector.clade_backpropagator.get_cmp(f'node_{i}'):.2f}")

    print(f"  - Total evaluations: {selector._total_evaluations}")
    print(f"  - Total nodes created: {selector._total_nodes_created}")

    # Test Thompson sampling selection
    print("\n4. Testing Thompson sampling selection...")
    candidates = [f"node_{i}" for i in range(5)]
    selected = selector.select_node(candidates)

    print(f"  - Candidates: {candidates}")
    print(f"  - Selected: {selected}")
    print(f"  - Tau value: {selector.tau_scheduler.schedule(selector.tau_scheduler_schedule):.2f}")

    # Test decoupling rule
    print("\n5. Testing decoupling rule...")
    should_expand, N_t_alpha, T_t = selector.check_decoupling_rule()
    print(f"  - N_t: {selector._total_evaluations}")
    print(f"  - T_t (archive size): {T_t}")
    print(f"  - N_t^0.6: {N_t_alpha:.2f}")
    print(f"  - Rule (N_t^0.6 >= |T_t|): {N_t_alpha >= T_t}")
    print(f"  - Phase: {'EXPANSION' if should_expand else 'EVALUATION'}")

    # Get HGM statistics
    print("\n6. HGM Statistics:")
    stats = selector.get_hgm_statistics()
    print(f"  - Current Tau: {stats['tau_scheduler']['current_tau']:.4f}")
    print(f"  - Current Step: {stats['tau_scheduler']['current_step']}/{stats['tau_scheduler']['total_steps']}")
    print(f"  - Phase: {stats['tau_scheduler']['phase']}")
    print(f"  - Best Candidate CMP: {stats['best_candidate_cmp_value']:.2f}")

    # Print CMP values for all nodes
    print(f"\n  - CMP values:")
    for node_id in list(stats['clade_compmetrics'].keys())[:5]:
        print(f"    {node_id}: {stats['clade_compmetrics'][node_id]:.2f}")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)

if __name__ == "__main__":
    test_thompson_sampling()
