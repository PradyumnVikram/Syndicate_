#!/usr/bin/env python3
"""
Minimal backpropagation test.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selector import SEDSSelector, SyntheticNode, MetricsSnapshot

print("Creating selector...")
selector = SEDSSelector(policy_type="EPSILON", max_tau=10.0, total_steps=100)

print("Creating root...")
root = SyntheticNode(node_id="root", parents=(), is_predefined=True)
selector.add_to_archive(root, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))
print("✓ Root added")

print("Creating a...")
a = SyntheticNode(node_id="a", parents=("root",))
selector.add_to_archive(a, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))
print("✓ Node a added")

print("Creating b...")
b = SyntheticNode(node_id="b", parents=("a",))
selector.add_to_archive(b, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))
print("✓ Node b added")

print("\nEvaluating node 'a' as SUCCESS...")
selector.evaluate_node("a", 0.7)
cmp_a = selector.clade_backpropagator.get_cmp("a")
cmp_root_after_a = selector.clade_backpropagator.get_cmp("root")
print(f"  Node 'a' CMP: {cmp_a}")
print(f"  Root CMP after a: {cmp_root_after_a}")

print("\nEvaluating node 'b' as FAILURE...")
selector.evaluate_node("b", 0.3)
cmp_b = selector.clade_backpropagator.get_cmp("b")
cmp_root_final = selector.clade_backpropagator.get_cmp("root")
print(f"  Node 'b' CMP: {cmp_b}")
print(f"  Root CMP final: {cmp_root_final}")

print("\nRoot clade stats:")
success_root, failure_root = selector.clade_backpropagator.get_clade_stats("root")
print(f"  n_success={success_root}, n_failure={failure_root}, CMP={cmp_root_final}")
