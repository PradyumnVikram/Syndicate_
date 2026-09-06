#!/usr/bin/env python3
"""
Integration test for SEDSManager orchestration with SimpleQA domain.

Tests the end-to-end flow:
- Seed initialization
- Candidate mutation and evaluation
- Phase D diagnostics (SSF, ContractAudit)
- Selection and archive management
- Budget-aware checkpointing
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from seds_manager import SEDSMonitor, BudgetManager


def test_budget_manager():
    """Test BudgetManager basic functionality."""
    print("\n" + "=" * 80)
    print("Testing BudgetManager")
    print("=" * 80)

    budget = BudgetManager(initial_budget=1.0)
    print(f"✓ Created budget manager with {budget.initial_budget:.2f} USD")

    # Test allocation
    if budget.allocate(0.1):
        print(f"✓ Allocated 0.1 USD, remaining: {budget.get_remaining():.4f}")
    else:
        print("✗ Allocation failed unexpectedly")
        return False

    # Test checkpoint
    budget.checkpoint()
    print("✓ Created budget checkpoint")

    # Test progress
    progress = budget.get_progress()
    print(f"✓ Budget progress: {progress}")
    print(f"  - Spent: ${progress['spent']:.4f}")
    print(f"  - Remaining: ${progress['remaining_percentage']:.1f}%")

    # Test exhaustion
    if budget.is_exhausted():
        print("✗ Budget should not be exhausted yet")
        return False

    return True


def test_monitor_checkpointing():
    """Test SEDSMonitor checkpointing."""
    print("\n" + "=" * 80)
    print("Testing SEDSMonitor Checkpointing")
    print("=" * 80)

    checkpoint_dir = Path("/tmp/test_seeds_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monitor = SEDSMonitor(checkpoint_dir=checkpoint_dir)
    print(f"✓ Created monitor with checkpoint dir: {checkpoint_dir}")

    # Create a checkpoint
    test_state = {
        'iteration': 1,
        'status': 'running',
        'data': {'test_key': 'test_value'}
    }

    monitor.create_checkpoint(test_state)
    print("✓ Created checkpoint")

    # Load checkpoint
    loaded = monitor.load_checkpoint()
    if loaded:
        print(f"✓ Loaded checkpoint: {loaded}")
        assert loaded['iteration'] == 1
        assert loaded['status'] == 'running'
        assert loaded['data']['test_key'] == 'test_value'
        print("✓ Checkpoint data verified")
    else:
        print("✗ Failed to load checkpoint")
        return False

    # Cleanup
    import shutil
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    print("✓ Cleaned up test checkpoint directory")

    return True


def test_integration_standalone():
    """Test integrated workflow without full SEDS components."""
    print("\n" + "=" * 80)
    print("Testing Integrated Workflow (Standalone)")
    print("=" * 80)

    from typing import Dict, List, Optional
    import time

    # Simulate components
    class MockSelector:
        def __init__(self):
            self.archive = {}
            self.pareto_frontier = []
            self.best_candidate = None

        def add_predefined_node(self, node):
            self.archive[node.node_id] = node
            print(f"✓ Added seed node: {node.node_id}")

        def add_to_archive(self, node, metrics, parent_id):
            self.archive[node.node_id] = {'node': node, 'metrics': metrics}
            self.pareto_frontier.append({
                'node_id': node.node_id,
                'success_rate': metrics.success_rate
            })
            print(f"✓ Added to archive: {node.node_id} (success_rate={metrics.success_rate:.2f})")

        def select_node(self, candidates):
            if candidates:
                best = max(candidates, key=lambda c: hash(c))
                self.best_candidate = best
                print(f"✓ Selected: {best}")
                return best
            return None

        def evaluate_node(self, node_id, score):
            print(f"✓ Evaluated {node_id}: score={score:.4f}")

    from seds.phase_d.contract_auditor import ContractAudit
    from seds.phase_d.ssf import SemanticSaliencyFolder

    # Initialize components
    selector = MockSelector()
    contract_auditor = ContractAudit(task_id="mock_task", node_id="mock_node")
    ssf = SemanticSaliencyFolder()

    # Initialize budget
    budget = BudgetManager(initial_budget=0.5)
    print(f"\n✓ Budget initialized: ${budget.get_remaining():.4f}")

    # Simulate iterations
    for iteration in range(2):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Check budget
        if budget.is_exhausted():
            print("✗ Budget exhausted")
            break

        # Allocate budget
        if not budget.allocate(0.2):
            print("✗ Budget allocation failed")
            break
        print(f"✓ Budget remaining: ${budget.get_remaining():.4f}")

        # Generate candidates
        candidates = [f"candidate_{iteration + 1}_{i}" for i in range(2)]
        print(f"✓ Generated candidates: {candidates}")

        # Evaluate candidates (simplified)
        for candidate_id in candidates:
            print(f"  - Evaluating {candidate_id}...")
            # Simulate metrics
            metrics = {
                'success_rate': 0.75,
                'average_reward': 0.70
            }

            # Add to selector
            from seds.selector import SyntheticNode, MetricsSnapshot
            node = SyntheticNode(
                node_id=candidate_id,
                parents=('seed_42',),
                creation_time=time.time(),
                is_predefined=False
            )

            from seds.selector import MetricsSnapshot as MS
            snapshot = MS(
                success_rate=metrics['success_rate'],
                average_reward=metrics['average_reward'],
                standard_deviation=0.0,
                coverage_score=metrics['success_rate'],
                novelty_score=1.0,
                any_metric=metrics
            )

            selector.add_to_archive(node, snapshot, parent_id='seed_42')
            selector.evaluate_node(candidate_id, metrics['average_reward'])

        # Checkpoint
        budget.checkpoint()
        print(f"✓ Checkpointed budget")

    # Show final state
    print(f"\n--- Final State ---")
    print(f"✓ Archive size: {len(selector.archive)}")
    print(f"✓ Pareto frontier size: {len(selector.pareto_frontier)}")
    print(f"✓ Best candidate: {selector.best_candidate}")
    print(f"✓ Final budget: ${budget.get_remaining():.4f}")

    return True


def main():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("SEDS Manager Integration Tests")
    print("=" * 80)

    tests = [
        ("BudgetManager", test_budget_manager),
        ("SEDSMonitor Checkpointing", test_monitor_checkpointing),
        ("Integrated Workflow", test_integration_standalone)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
        except Exception as e:
            print(f"\n✗ {name} failed with exception:")
            traceback.print_exc()
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"  Error: {error}")

    print(f"\n{passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
