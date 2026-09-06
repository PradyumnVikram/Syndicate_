#!/usr/bin/env python3
"""
Demo of the SEDS Selector module.
Shows synthetic nodes creation, parent selection, and Pareto frontier construction.
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selector import (
    SEDSSelector, SyntheticNode, MetricsSnapshot, 
    NodeLineage, ArchiveTree, ParetoFrontier, PolicyType
)

def create_synthetic_nodes(num_nodes: int = 10) -> list[SyntheticNode]:
    """
    Create synthetic nodes for demonstration.
    
    Args:
        num_nodes: Number of nodes to create
        
    Returns:
        List of synthetic nodes
    """
    nodes = []
    for i in range(num_nodes):
        node = SyntheticNode(
            node_id=f"node_{i}",
            parents=(f"node_{max(0, i-1)}",) if i > 0 else (),
            creation_time=1000 + i,
            is_predefined=False
        )
        nodes.append(node)
    
    return nodes

def main():
    print("=" * 70)
    print("SEDS Selector Demo")
    print("=" * 70)
    print()
    
    # Initialize selector
    print("1. Initializing SEDS Selector")
    print("-" * 70)
    selector = SEDSSelector(
        policy_type=PolicyType.EPSILON,
        epsilon=0.2,
        allow_rollbacks=True
    )
    print(f"   Created selector with {selector.get_statistics()}")
    print()
    
    # Create and add predefined nodes
    print("2. Creating Predefined Nodes")
    print("-" * 70)
    predefined_nodes = [
        SyntheticNode(node_id="fixed_agent", is_predefined=True, parents=()),
        SyntheticNode(node_id="default_module", is_predefined=True, parents=())
    ]
    
    for node in predefined_nodes:
        selector.add_predefined_node(node)
        print(f"   ✓ Added predefined node: {node.node_id}")
    
    print(f"   Current stats: {selector.get_statistics()}")
    print()
    
    # Create and evaluate synthetic nodes
    print("3. Creating and Evaluating Synthetic Nodes")
    print("-" * 70)
    synthetic_nodes = create_synthetic_nodes(15)
    
    for i, node in enumerate(synthetic_nodes):
        # Create synthetic reward based on node position (higher for later nodes)
        reward = 0.3 + (i / len(synthetic_nodes)) * 0.7
        if random.random() < 0.1:
            reward = 0.0  # Occasional failures
        
        # Add some variance
        reward += random.uniform(-0.1, 0.1)
        reward = max(0.0, min(1.0, reward))
        
        # Create snapshot with meaningful metrics
        snapshot = MetricsSnapshot(
            success_rate=reward,
            average_reward=reward,
            standard_deviation=0.05,
            coverage_score=0.6 + i * 0.02,
            novelty_score=0.1 + i * 0.05,
            any_metric={
                'reward': reward,
                'age': 1000 + i,
                'metrics_derived': 0.5 + reward
            }
        )
        
        selector.add_to_archive(node, snapshot)
        selector.evaluate_node(node.node_id, reward)
        
        print(f"   Node {node.node_id}: reward={reward:.3f}, "
              f"coverage={snapshot.coverage_score:.3f}, "
              f"novelty={snapshot.novelty_score:.3f}")
    
    print(f"   Current stats: {selector.get_statistics()}")
    print()
    
    # Demonstrate parent selection
    print("4. Parent Selection Examples")
    print("-" * 70)
    
    # Test selecting from different candidate sets
    candidates1 = [f"node_{i}" for i in [2, 5, 7, 10]]
    selected1 = selector.select_node(candidates1)
    print(f"   Candidates: {candidates1}")
    print(f"   Selected:   {selected1}")
    print()
    
    candidates2 = [f"node_{i}" for i in [0, 1, 3, 4, 6]]
    selected2 = selector.select_node(candidates2)
    print(f"   Candidates: {candidates2}")
    print(f"   Selected:   {selected2}")
    print()
    
    # Demonstrate Pareto frontier
    print("5. Pareto Frontier Analysis")
    print("-" * 70)
    
    frontier = selector.pareto_frontier
    top_k = frontier.get_candidates(k=5)
    
    print(f"   Pareto frontier size: {frontier.size()}")
    print(f"   Top 5 candidates:")
    for i, snapshot in enumerate(top_k):
        node_id = getattr(snapshot, 'node_id', 'unknown')
        print(f"   {i+1}. {node_id}: "
              f"coverage={snapshot.coverage_score:.3f}, "
              f"novelty={snapshot.novelty_score:.3f}, "
              f"success={snapshot.success_rate:.3f}")
    
    print()
    
    # Get best candidate
    best = selector.get_best_candidate()
    print(f"   Dominant candidate: {best}")
    print()
    
    # Show lineage and archive
    print("6. Lineage and Archive Statistics")
    print("-" * 70)
    print(f"   Lineage nodes: {selector.lineage.size()}")
    print(f"   Archive items: {selector.archive.size()}")
    
    if best:
        parents = selector.lineage.get_parents(best)
        print(f"   Best candidate ({best}) parents: {[p.node_id for p in parents]}")
    print()
    
    # Show final statistics
    print("7. Final Statistics")
    print("-" * 70)
    stats = selector.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print()
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
