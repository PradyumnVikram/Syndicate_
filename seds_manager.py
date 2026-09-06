#!/usr/bin/env python3
"""
SEDS Self-Improving Agent Manager (seds_manager.py).

Outer orchestration layer that binds together:
- SEDSSynthesizer (Phase E: mutation proposal)
- DockerExecutor (Phase B: candidate execution)
- Phase D diagnostics (SSF, failure_taxonomy, contract_auditor, doVer)
- SEDSSelector (Phase F: selection, scoring, archive management)

This module implements the 8-phase evolutionary loop with budget awareness,
checkpointing/resumability, and progress tracking.

Usage:
    python -m seds_manager.run [options]

Options:
    --budget-usd FLOAT    Budget in USD for the search (default 1.0)
    --evals-per-node INT   How many val tasks per node (default 4)
    --checkpoint-dir PATH  Directory for checkpoints (default data/checkpoints/)
    --resume              Resume from checkpoint if available
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import pickle
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHECKPOINT_DIR = Path("data/checkpoints")
DEFAULT_TRACE_DB_PATH = "data/trace_db.sqlite"
DEFAULT_ARCHIVE_DIR = "agent_archive"
LOG_FILE = Path("/tmp/seds_progress/W7-outer-loop.md")


class BudgetManager:
    """Budget-aware resource manager for SEDS iterations."""

    def __init__(self, initial_budget: float, checkpoint_file: Optional[Path] = None):
        """
        Initialize budget manager.

        Args:
            initial_budget: Initial budget in USD
            checkpoint_file: Path to checkpoint file for resume functionality
        """
        self.initial_budget = initial_budget
        self.current_budget = initial_budget
        self.checkpoint_file = checkpoint_file
        self.history: List[Dict[str, Any]] = []
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load budget state from checkpoint file if it exists."""
        if self.checkpoint_file and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    self.current_budget = checkpoint_data.get('current_budget', self.current_budget)
                    self.history = checkpoint_data.get('history', [])
                    logger.info(
                        f"Loaded checkpoint: budget={self.current_budget:.4f}, "
                        f"history_len={len(self.history)}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

    def checkpoint(self) -> None:
        """Save current budget state to checkpoint file."""
        if self.checkpoint_file:
            checkpoint_data = {
                'current_budget': self.current_budget,
                'history': self.history
            }
            try:
                with open(self.checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                logger.info(f"Checkpointed budget: ${self.current_budget:.4f}")
            except Exception as e:
                logger.error(f"Failed to checkpoint: {e}")

    def allocate(self, cost: float) -> bool:
        """
        Allocate budget cost.

        Args:
            cost: Cost to allocate

        Returns:
            True if budget sufficient, False otherwise
        """
        if self.current_budget >= cost:
            self.current_budget -= cost
            self.history.append({
                'timestamp': time.time(),
                'cost': cost,
                'remaining': self.current_budget
            })
            return True
        else:
            logger.warning(
                f"Budget insufficient: requested ${cost:.4f}, "
                f"remaining ${self.current_budget:.4f}"
            )
            return False

    def get_remaining(self) -> float:
        """Get remaining budget."""
        return self.current_budget

    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.current_budget <= 0

    def get_progress(self) -> Dict[str, Any]:
        """Get budget progress information."""
        return {
            'initial_budget': self.initial_budget,
            'current_budget': self.current_budget,
            'spent': self.initial_budget - self.current_budget,
            'spend_percentage': (self.initial_budget - self.current_budget) / self.initial_budget * 100,
            'remaining_percentage': (self.current_budget / self.initial_budget) * 100
        }


class SEDSMonitor:
    """Progress monitoring and checkpointing."""

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize monitor.

        Args:
            checkpoint_dir: Directory for saving/loading checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = LOG_FILE
        self._setup_progress_tracking()

    def _setup_progress_tracking(self):
        """Initialize progress tracking file."""
        if not self.progress_file.exists():
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, "w") as f:
                f.write("# SEDS Phase D Outer Loop Progress\n\n")
            self._log("Initialized progress tracking")

    def _log(self, message: str):
        """Log progress to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.progress_file, "a") as f:
            f.write(f"{timestamp} | {message}\n")

    def create_checkpoint(self, state: Dict[str, Any]):
        """
        Save checkpoint to disk.

        Args:
            state: Dictionary of checkpoint state
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{time.time_ns()}.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(state, f)
        self._log(f"Checkpoint saved: {checkpoint_path.name}")

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load latest checkpoint from disk.

        Returns:
            Checkpoint state dict or None if not found
        """
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not checkpoint_files:
            return None

        checkpoint_path = checkpoint_files[0]
        try:
            with open(checkpoint_path, "rb") as f:
                state = pickle.load(f)
            self._log(f"Checkpoint loaded: {checkpoint_path.name}")
            return state
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None


def load_dotenv_if_exists(dotenv_path: Optional[Path] = None) -> bool:
    """
    Load .env file if it exists.

    Args:
        dotenv_path: Path to .env file (defaults to repo root)

    Returns:
        True if .env was loaded
    """
    if dotenv_path is None:
        dotenv_path = Path(".") / ".env"

    if dotenv_path.exists():
        dotenv.load_dotenv(dotenv_path)
        logger.info(f"Loaded .env from {dotenv_path}")
        return True
    else:
        logger.warning(f".env not found at {dotenv_path}")
        return False


def main():
    """Main entry point for SEDS manager."""
    parser = argparse.ArgumentParser(
        description="SEDS Self-Improving Agent Manager - Outer evolutionary loop"
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=1.0,
        help="Budget in USD for the search (default: 1.0)",
    )
    parser.add_argument(
        "--evals-per-node",
        type=int,
        default=4,
        help="Number of validation tasks per node (default: 4)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help=f"Directory for checkpoints (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--budget-checkpoint",
        type=str,
        default=None,
        help="Path for budget checkpoint file (for resuming budget state)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint if available",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load .env if available
    load_dotenv_if_exists()

    logger.info("=" * 80)
    logger.info("SEDS Self-Improving Agent Manager")
    logger.info("=" * 80)

    # Create progress monitor
    monitor = SEDSMonitor(checkpoint_dir=Path(args.checkpoint_dir))
    monitor._log("Starting SEDS outer loop")

    # Check if resuming
    if args.resume:
        state = monitor.load_checkpoint()
        if state is not None:
            monitor._log("Resuming from checkpoint")
            # TODO: Implement resume logic
            logger.warning("Resume functionality not yet implemented")
        else:
            logger.info("No checkpoint found, starting fresh")

    # Initialize SEDS components
    from seds.synthesizer import SEDSSynthesizer
    from seds.domains.simple_qa import SimpleQA_Domain

    # Create domain
    logger.info("Initializing SimpleQA domain...")
    domain = SimpleQA_Domain()

    # Initialize components
    from seds.phase_d.contract_auditor import ContractAudit
    from seds.phase_d.ssf import SSF
    from seds.phase_d.do_ver import DoVer
    from seds.phase_d.failure_taxonomy import FailureCategory
    from seds.selector import SEDSSelector, SyntheticNode, MetricsSnapshot

    # Initialize selector
    selector = SEDSSelector(policy_type="EPSILON", epsilon=0.1)

    # Initialize Phase D components
    contract_auditor = ContractAudit()
    ssf = SSF()
    do_ver = DoVer()

    # Add seed node
    logger.info("Adding seed node...")
    seed_node = SyntheticNode(
        node_id=f"seed_{args.seed}",
        parents=(),
        creation_time=time.time(),
        is_predefined=True
    )
    selector.add_predefined_node(seed_node)

    # Initialize budget manager
    budget_manager = BudgetManager(
        initial_budget=args.budget_usd,
        checkpoint_file=Path(args.budget_checkpoint) if args.budget_checkpoint else None
    )
    logger.info(f"Starting SEDS outer loop with budget: ${args.budget_usd:.4f}")

    # Main orchestration loop
    for iteration in range(5):  # Small number of iterations for demo
        logger.info(f"Iteration {iteration + 1}/5")

        # Check if budget exhausted
        if budget_manager.is_exhausted():
            logger.info("Budget exhausted, exiting loop")
            break

        # Sample mutations from seed node
        logger.info("Sampling mutations...")
        candidate_count = 0
        candidates_to_evaluate = []

        for _ in range(3):  # Try 3 mutations per iteration
            candidate_count += 1
            candidate_id = f"candidate_{iteration + 1}_{candidate_count}"
            candidates_to_evaluate.append(candidate_id)

        logger.info(f"Generated {len(candidates_to_evaluate)} candidates: {candidates_to_evaluate}")

        # Allocate budget for this iteration
        if not budget_manager.allocate(0.1):  # Simulate 0.1 USD cost per iteration
            logger.warning("Budget allocation failed, exiting")
            break

        logger.info(f"Budget remaining: ${budget_manager.get_remaining():.4f}")

        # Evaluate candidates
        for candidate_id in candidates_to_evaluate:
            logger.info(f"Evaluating candidate: {candidate_id}")

            # Create synthetic candidate node
            node = SyntheticNode(
                node_id=candidate_id,
                parents=(seed_node.node_id,),
                creation_time=time.time(),
                is_predefined=False
            )

            # Get evaluation tasks from domain
            val_tasks = domain.val_tasks[:args.evals_per_node]
            logger.info(f"Executing {len(val_tasks)} validation tasks for {candidate_id}")

            total_correct = 0
            total_score = 0.0

            for task in val_tasks:
                # Execute candidate using DockerExecutor (placeholder)
                # In real implementation, this would:
                # 1. Serialize the candidate (e.g., as a program string or LLM prompt)
                # 2. Pass to executor
                # 3. Get back answer + metrics

                # Placeholder execution
                answer = f"Mock answer for {candidate_id} on task {task.task_id}"

                # Evaluate using domain.evaluate()
                score = domain.evaluate(task, answer)
                total_correct += 1 if score.correct else 0
                total_score += score.partial

                # Run Phase D diagnostics
                failure_category = None
                if not score.correct:
                    # Run contract audit on failures
                    failure_report = contract_auditor.audit_contract(
                        agent_answer=answer,
                        reference_answer=task.reference or "",
                        task_context=task.to_dict() if hasattr(task, 'to_dict') else str(task)
                    )
                    failure_category = failure_report['category']

                # Run SSF (Semantic Saliency Folder) to identify failure modes
                ssf_features = ssf.analyze(
                    task_input=task.to_dict() if hasattr(task, 'to_dict') else str(task),
                    candidate_id=candidate_id,
                    failure_category=failure_category
                )

                logger.info(
                    f"Task {task.task_id}: correct={score.correct}, "
                    f"partial={score.partial:.4f}, "
                    f"failure_category={failure_category}, "
                    f"ssf_features={ssf_features['primary_failure_mode']}"
                )

            # Calculate metrics
            accuracy = total_correct / len(val_tasks)
            avg_reward = total_score / len(val_tasks)

            # Calculate failure analysis
            failure_modes = ssf.compute_overall_failure_modes(
                candidate_id=candidate_id
            )

            metrics = MetricsSnapshot(
                success_rate=accuracy,
                average_reward=avg_reward,
                standard_deviation=0.0,  # TODO: calculate std dev
                coverage_score=accuracy,
                novelty_score=1.0,  # New node
                any_metric={
                    'accuracy': accuracy,
                    'avg_reward': avg_reward,
                    'failure_modes': failure_modes
                }
            )

            # Add to selector archive
            selector.add_to_archive(node, metrics, parent_id=seed_node.node_id)

            # Update stats
            selector.evaluate_node(node.node_id, avg_reward)

            logger.info(
                f"Evaluated {candidate_id}: correct={total_correct}/{len(val_tasks)}, "
                f"avg_reward={avg_reward:.4f}"
            )

        # Run selector to get best candidate
        logger.info("Running selector...")
        best_candidate = selector.select_node(candidates_to_evaluate)

        if best_candidate:
            logger.info(f"Selected best candidate: {best_candidate}")
        else:
            logger.warning("No candidates selected")

        # Check if we should exit (budget exhausted)
        logger.info("Checkpointing state...")

        # Save budget checkpoint
        budget_manager.checkpoint()

        # Save general checkpoint
        checkpoint_path = monitor.checkpoint_dir / f"checkpoint_iter_{iteration + 1}.pkl"
        monitor.create_checkpoint({
            'iteration': iteration + 1,
            'budget_remaining': budget_manager.get_remaining(),
            'budget_progress': budget_manager.get_progress(),
            'best_candidate': best_candidate,
            'selector_stats': selector.get_statistics()
        })

        # Check budget again (after checkpoint)
        if budget_manager.is_exhausted():
            logger.info("Budget exhausted, exiting loop")
            break

        logger.info(f"Budget remaining: ${budget_manager.get_remaining():.4f}")

        logger.info("SEDS outer loop completed")
        monitor._log("SEDS outer loop completed successfully")

        # Final budget checkpoint
        budget_manager.checkpoint()
        logger.info(f"Final budget progress: {budget_manager.get_progress()}")

        # Show final Pareto frontier
        logger.info(f"Final Pareto frontier:")
        for snapshot in selector.pareto_frontier.get_frontier():
            logger.info(
                f"  Node {snapshot.node_id}: "
                f"success_rate={snapshot.success_rate:.2f}, "
                f"coverage_score={snapshot.coverage_score:.2f}, "
                f"novelty_score={snapshot.novelty_score:.2f}"
            )

    # Show final statistics
    stats = selector.get_statistics()
    logger.info(f"Final statistics:")
    logger.info(f"  Total nodes created: {stats['total_nodes_created']}")
    logger.info(f"  Total evaluations: {stats['total_evaluations']}")
    logger.info(f"  Archive size: {stats['archive_size']}")
    logger.info(f"  Pareto frontier size: {stats['pareto_frontier_size']}")
    logger.info(f"  Best candidate: {stats['best_candidate']}")


if __name__ == "__main__":
    main()
