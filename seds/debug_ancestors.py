#!/usr/bin/env python3
"""
Debug script to check archive tree structure and ancestor navigation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selector import SEDSSelector, SyntheticNode, MetricsSnapshot

print("Creating selector...")
selector = SEDSSelector(policy_type="EPSILON", max_tau=10.0, total_steps=100)
print("✓ Selector created")
print()

print("Creating root node...")
root_node = SyntheticNode(node_id="root", parents=(), is_predefined=True)
selector.add_to_archive(root_node, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))
print("✓ Root added")
print()

print("Creating node a...")
a_node = SyntheticNode(node_id="a", parents=("root",))
selector.add_to_archive(a_node, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))
print("✓ Node a added")
print()

print("Creating node b...")
b_node = SyntheticNode(node_id="b", parents=("a",))
selector.add_to_archive(b_node, MetricsSnapshot(
    success_rate=0.0, average_reward=0.0, standard_deviation=0.0,
    coverage_score=0.0, novelty_score=0.0, any_metric={'reward': 0.0}
))
print("✓ Node b added")
print()

print("Archive tree structure:")
print(f"  Items: {selector.archive._items.keys()}")
print(f"  Children: {selector.archive._children}")
print()

print("Item structure:")
for node_id in ["root", "a", "b"]:
    item = selector.archive._items.get(node_id)
    if item:
        print(f"  {node_id}:")
        print(f"    type: {type(item)}")
        print(f"    item.node: {item.node}")
        print(f"    item.node.parents: {getattr(item.node, 'parents', 'N/A')}")
print()

print("Checking parent lookup manually:")
for child_id in ["a", "b"]:
    # Try to find parent by checking what children each node has
    parent = None
    for parent_id, children in selector.archive._children.items():
        if child_id in children:
            parent = parent_id
            break
    print(f"  Parent of '{child_id}': {parent}")
