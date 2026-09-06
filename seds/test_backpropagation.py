#!/usr/bin/env python3
"""
Direct test for clade backpropagation on lineage chains.

Creates a 3-node chain: root -> a -> b
- Evaluates node 'a' as SUCCESS
- Evaluates node 'b' as FAILURE
- Verifies root clade CMP is intermediate (~0.5), not 0.0 or 1.0
"""

import sys
import os

# Set path to seds directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selector import SEDSSelector, SyntheticNode, MetricsSnapshot

def test_clade_chain_backpropagation():
    """Test that backpropagation walks the ancestor chain correctly."""
    print("=" * 70)
    print("Testing Clade Backpropagation on Lineage Chain")
    print("=" * 70)
    print()

    # Create selector with HGM components
    selector = SEDSSelector(
        policy_type="THOMPSON",
        max_tau=10.0,
        total_steps=100,  # Total steps for tau
    )

    # Create a 3-node chain: root -> a -> b
    print("1. Creating lineage chain: root -> a -> b")
    print()

    # Root node
    root_node = SyntheticNode(
        node_id="root",
        parents=(),  # Root has no parent
    )
    selector.add_to_archive(root_node, MetricsSnapshot(
        success_rate=0.0,
        average_reward=0.0,
        standard_deviation=0.0,
        coverage_score=0.0,
        novelty_score=0.0,
        any_metric={'reward': 0.0}
    ))

    # Node a (child of root)
    a_node = SyntheticNode(
        node_id="a",
        parents=("root",),
    )
    selector.add_to_archive(a_node, MetricsSnapshot(
        success_rate=0.0,
        average_reward=0.0,
        standard_deviation=0.0,
        coverage_score=0.0,
        novelty_score=0.0,
        any_metric={'reward': 0.0}
    ))

    # Node b (child of a)
    b_node = SyntheticNode(
        node_id="b",
        parents=("a",),
    )
    selector.add_to_archive(b_node, MetricsSnapshot(
        success_rate=0.0,
        average_reward=0.0,
        standard_deviation=0.0,
        coverage_score=0.0,
        novelty_score=0.0,
        any_metric={'reward': 0.0}
    ))

    print(f"   ✓ Created nodes: {selector.archive._items.keys()}")
    print(f"   ✓ Lineage structure verified")
    print()

    # Evaluate node 'a' as SUCCESS
    print("2. Evaluating node 'a' as SUCCESS (reward=0.7)")
    selector.evaluate_node("a", 0.7)
    cmp_a_success = selector.clade_backpropagator.get_cmp("a")
    cmp_root_after_a = selector.clade_backpropagator.get_cmp("root")
    print(f"   Node 'a' CMP: {cmp_a_success:.3f}")
    print(f"   Root CMP after a: {cmp_root_after_a:.3f}")
    print()

    # Evaluate node 'b' as FAILURE
    print("3. Evaluating node 'b' as FAILURE (reward=0.3)")
    selector.evaluate_node("b", 0.3)
    cmp_b_failure = selector.clade_backpropagator.get_cmp("b")
    cmp_a_after_b = selector.clade_backpropagator.get_cmp("a")
    cmp_root_final = selector.clade_backpropagator.get_cmp("root")
    print(f"   Node 'b' CMP: {cmp_b_failure:.3f}")
    print(f"   Node 'a' CMP after b: {cmp_a_after_b:.3f}")
    print(f"   Root CMP final: {cmp_root_final:.3f}")
    print()

    # Print detailed clade stats
    print("4. Detailed Clade Statistics:")
    print("-" * 70)
    for node_id in ["root", "a", "b"]:
        success, failure = selector.clade_backpropagator.get_clade_stats(node_id)
        cmp = selector.clade_backpropagator.get_cmp(node_id)
        print(f"   {node_id}: n_success={success}, n_failure={failure}, CMP={cmp:.3f}")
    print()

    # Assertions
    print("5. Validation:")
    print("-" * 70)

    # Node 'a' should have n_success=1, n_failure=0 after success evaluation
    success_a, failure_a = selector.clade_backpropagator.get_clade_stats("a")
    assert success_a == 1, f"Node 'a' should have n_success=1, got {success_a}"
    assert failure_a == 0, f"Node 'a' should have n_failure=0, got {failure_a}"
    print(f"   ✓ Node 'a' has correct stats: n_success={success_a}, n_failure={failure_a}")

    # Node 'b' should have n_success=0, n_failure=1 after failure evaluation
    success_b, failure_b = selector.clade_backpropagator.get_clade_stats("b")
    assert success_b == 0, f"Node 'b' should have n_success=0, got {success_b}"
    assert failure_b == 1, f"Node 'b' should have n_failure=1, got {failure_b}"
    print(f"   ✓ Node 'b' has correct stats: n_success={success_b}, n_failure={failure_b}")

    # Root should have n_success=1, n_failure=1 (aggregates both descendants)
    success_root, failure_root = selector.clade_backpropagator.get_clade_stats("root")
    assert success_root == 1, f"Root should have n_success=1, got {success_root}"
    assert failure_root == 1, f"Root should have n_failure=1, got {failure_root}"
    print(f"   ✓ Root has correct aggregated stats: n_success={success_root}, n_failure={failure_root}")

    # Root CMP should be intermediate (~0.5), NOT 0.0 or 1.0
    cmp_root = selector.clade_backpropagator.get_cmp("root")
    assert 0.0 < cmp_root < 1.0, \
        f"Root CMP must be intermediate (0.0, 1.0), got {cmp_root} - BACKPROPAGATION IS BROKEN!"
    print(f"   ✓ Root CMP is intermediate: {cmp_root:.3f} (NOT 0.0 or 1.0)")

    # Node 'a' CMP should remain 1.0 after 'b' failure (only receives a's own result)
    cmp_a = selector.clade_backpropagator.get_cmp("a")
    assert cmp_a == 1.0, f"Node 'a' CMP should be 1.0, got {cmp_a}"
    print(f"   ✓ Node 'a' CMP remains 1.0 (only receives its own success)")

    # Node 'b' CMP should be 0.0
    cmp_b = selector.clade_backpropagator.get_cmp("b")
    assert cmp_b == 0.0, f"Node 'b' CMP should be 0.0, got {cmp_b}"
    print(f"   ✓ Node 'b' CMP is 0.0 (only receives its own failure)")

    print()
    print("=" * 70)
    print("ALL TESTS PASSED - Backpropagation is working correctly!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  - Root clade: n_success=1, n_failure=1 → CMP={cmp_root:.3f} ✓")
    print(f"  - Node 'a':   n_success=1, n_failure=0 → CMP=1.000 ✓")
    print(f"  - Node 'b':   n_success=0, n_failure=1 → CMP=0.000 ✓")
    print()

if __name__ == "__main__":
    try:
        test_clade_chain_backpropagation()
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"TEST FAILED: {e}")
        print("=" * 70)
        exit(1)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"ERROR: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        exit(1)
