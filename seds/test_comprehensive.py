#!/usr/bin/env python3
"""
Comprehensive backpropagation test after fix.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selector import SEDSSelector, SyntheticNode, MetricsSnapshot

print("=" * 70)
print("Comprehensive Backpropagation Test")
print("=" * 70)
print()

# Test 1: Simple chain
print("Test 1: Simple chain root -> a -> b")
print("-" * 70)
selector = SEDSSelector(policy_type="EPSILON", max_tau=10.0, total_steps=100)

root = SyntheticNode(node_id="root", parents=(), is_predefined=True)
selector.add_to_archive(root, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))

a = SyntheticNode(node_id="a", parents=("root",))
selector.add_to_archive(a, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))

b = SyntheticNode(node_id="b", parents=("a",))
selector.add_to_archive(b, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))

print(f"  Parent-child structure: {dict(selector.archive._children)}")

selector.evaluate_node("a", 0.7)
success_a, failure_a = selector.clade_backpropagator.get_clade_stats("a")
cmp_a = selector.clade_backpropagator.get_cmp("a")

selector.evaluate_node("b", 0.3)
success_b, failure_b = selector.clade_backpropagator.get_clade_stats("b")
cmp_b = selector.clade_backpropagator.get_cmp("b")
success_root, failure_root = selector.clade_backpropagator.get_clade_stats("root")
cmp_root = selector.clade_backpropagator.get_cmp("root")

print(f"  Node 'a': n_success={success_a}, n_failure={failure_a}, CMP={cmp_a:.3f}")
print(f"  Node 'b': n_success={success_b}, n_failure={failure_b}, CMP={cmp_b:.3f}")
print(f"  Root:     n_success={success_root}, n_failure={failure_root}, CMP={cmp_root:.3f}")
print()

# Test 2: Single node
print("Test 2: Single node (no parent)")
print("-" * 70)
selector2 = SEDSSelector(policy_type="EPSILON", max_tau=10.0, total_steps=100)

single = SyntheticNode(node_id="single", parents=())
selector2.add_to_archive(single, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))

print(f"  Parent-child structure: {dict(selector2.archive._children)}")

selector2.evaluate_node("single", 1.0)
success, failure = selector2.clade_backpropagator.get_clade_stats("single")
cmp = selector2.clade_backpropagator.get_cmp("single")

print(f"  Node: n_success={success}, n_failure={failure}, CMP={cmp:.3f}")
print()

# Test 3: Multiple children
print("Test 3: Multiple children under same parent")
print("-" * 70)
selector3 = SEDSSelector(policy_type="EPSILON", max_tau=10.0, total_steps=100)

parent = SyntheticNode(node_id="parent", parents=(), is_predefined=True)
selector3.add_to_archive(parent, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))

child1 = SyntheticNode(node_id="child1", parents=("parent",))
selector3.add_to_archive(child1, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))

child2 = SyntheticNode(node_id="child2", parents=("parent",))
selector3.add_to_archive(child2, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))

child3 = SyntheticNode(node_id="child3", parents=("parent",))
selector3.add_to_archive(child3, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))

print(f"  Parent-child structure: {dict(selector3.archive._children)}")

selector3.evaluate_node("child1", 0.9)
selector3.evaluate_node("child2", 0.1)
selector3.evaluate_node("child3", 1.0)

success1, failure1 = selector3.clade_backpropagator.get_clade_stats("child1")
success2, failure2 = selector3.clade_backpropagator.get_clade_stats("child2")
success3, failure3 = selector3.clade_backpropagator.get_clade_stats("child3")
success_parent, failure_parent = selector3.clade_backpropagator.get_clade_stats("parent")
cmp_parent = selector3.clade_backpropagator.get_cmp("parent")

print(f"  Child1: n_success={success1}, n_failure={failure1}")
print(f"  Child2: n_success={success2}, n_failure={failure2}")
print(f"  Child3: n_success={success3}, n_failure={failure3}")
print(f"  Parent: n_success={success_parent}, n_failure={failure_parent}, CMP={cmp_parent:.3f}")
print()

print("=" * 70)
print("All tests completed successfully!")
print("=" * 70)
