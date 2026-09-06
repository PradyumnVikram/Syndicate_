"""Phase G evaluation harness with minibatch sampling and frozen validation sets."""

import random
from typing import Any, Dict, List, Optional, Tuple
from seds.domains.base import TaskDomain, Task, Score


class EvaluationHarness:
    """
    Evaluation harness for Phase G with frozen validation sets and minibatch sampling.

    Handles:
    - Frozen validation set construction
    - Random minibatch sampling without replacement
    - Frozen val set coverage tracking
    """

    def __init__(
        self,
        domain: TaskDomain,
        n_samples_per_epoch: int = 10,
        seed: int = 42,
    ):
        """
        Initialize the evaluation harness.

        Args:
            domain: The TaskDomain to evaluate
            n_samples_per_epoch: Number of tasks to sample per evaluation epoch
            seed: Random seed for reproducibility
        """
        self.domain = domain
        self.n_samples_per_epoch = n_samples_per_epoch
        self.seed = seed

        # Track evaluation history per domain
        self.history: List[Dict[str, Any]] = []

    def create_frozen_validation_set(
        self,
        n_tasks: int = 10,
        shuffle: bool = True,
        output_path: Optional[str] = None,
    ) -> List[Task]:
        """
        Create a frozen validation set of tasks.

        Args:
            n_tasks: Number of tasks to include in the frozen set
            shuffle: Whether to shuffle the tasks
            output_path: Path to save the frozen set (optional)

        Returns:
            List of frozen Task objects
        """
        random.seed(self.seed)
        # Use val_tasks for evaluation (frozen validation set)
        domain_tasks = self.domain.val_tasks if self.domain.val_tasks else self.domain.train_tasks

        if shuffle:
            random.shuffle(domain_tasks)

        selected = domain_tasks[:min(n_tasks, len(domain_tasks))]

        frozen_set = selected  # Return Task objects directly

        if output_path:
            import json
            from pathlib import Path

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump([t.__dict__ for t in selected], f, indent=2)

        return selected

    def get_frozen_validation_set(self, frozen_path: str) -> List[Task]:
        """
        Load a frozen validation set from file.

        Args:
            frozen_path: Path to the frozen validation set file

        Returns:
            List of frozen Task objects
        """
        import json
        from pathlib import Path

        with open(frozen_path, 'r') as f:
            frozen_data = json.load(f)

        return [Task(**item) for item in frozen_data]

    def sample_minibatch(
        self,
        frozen_set: List[Task],
        batch_size: Optional[int] = None,
    ) -> List[Task]:
        """
        Sample a minibatch from the frozen validation set.

        Args:
            frozen_set: Frozen validation set of tasks
            batch_size: Number of tasks to sample (default to self.n_samples_per_epoch)

        Returns:
            List of sampled TaskSpec objects (without replacement)
        """
        batch_size = batch_size or self.n_samples_per_epoch
        random.seed(self.seed + len(self.history))  # Vary seed by epoch

        if batch_size > len(frozen_set):
            batch_size = len(frozen_set)

        # Sample without replacement
        sampled = random.sample(frozen_set, batch_size)

        return sampled

    def evaluate_domain_epoch(
        self,
        frozen_set: List[Task],
        batch_size: Optional[int] = None,
        run_host_side: bool = True,
    ) -> List[Score]:
        """
        Evaluate the domain on a given frozen set for one epoch.

        Args:
            frozen_set: Frozen validation set of tasks
            batch_size: Number of tasks per batch
            run_host_side: Whether to run host-side evaluation

        Returns:
            List of Score objects
        """
        epoch_scores: List[Score] = []
        epoch_tasks = self.sample_minibatch(frozen_set, batch_size)

        for task in epoch_tasks:
            score = self.domain.evaluate(
                task=task,
                host_output_reference=""  # Host-side evaluation
            )
            epoch_scores.append(score)

        self.history.append({
            "epoch": len(self.history) + 1,
            "tasks_evaluated": len(epoch_tasks),
            "scores": epoch_scores,
            "task_ids": [task.task_id for task in epoch_tasks],
        })

        return epoch_scores

    def compute_epoch_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics across all evaluated epochs.

        Returns:
            Dictionary with epoch-level metrics
        """
        if not self.history:
            return {"n_epochs": 0}

        all_results = []
        for epoch in self.history:
            all_results.extend(epoch["scores"])

        n_results = len(all_results)
        n_success = sum(1 for r in all_results if r.correct)

        pass_rate = n_success / n_results if n_results > 0 else 0.0

        # Compute confidence intervals (95%)
        if n_results >= 2:
            se = (pass_rate * (1 - pass_rate) / n_results) ** 0.5
            ci_lower = pass_rate - 1.96 * se
            ci_upper = pass_rate + 1.96 * se
        else:
            ci_lower = pass_rate
            ci_upper = pass_rate

        return {
            "n_epochs": len(self.history),
            "n_results": n_results,
            "n_success": n_success,
            "pass_rate": pass_rate,
            "pass_rate_ci_lower": ci_lower,
            "pass_rate_ci_upper": ci_upper,
            "n_failures": n_results - n_success,
        }

    def get_epoch_pass_rates(self) -> List[float]:
        """Get pass rate for each evaluated epoch."""
        return [
            sum(r.partial for r in epoch["scores"]) / len(epoch["scores"])
            for epoch in self.history
        ]

    def clear_history(self) -> None:
        """Clear the evaluation history."""
        self.history.clear()

    def save_history(self, output_path: str) -> None:
        """Save evaluation history to file."""
        import json
        from pathlib import Path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_history(self, input_path: str) -> None:
        """Load evaluation history from file."""
        import json
        from pathlib import Path

        with open(input_path, 'r') as f:
            self.history = json.load(f)


def split_fixed_frozen_set(
    frozen_set: List[Task],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    shuffle: bool = True,
) -> Tuple[List[Task], List[Task], List[Task]]:
    """
    Split a fixed frozen set into train/val/test splits.

    Args:
        frozen_set: Frozen validation set of tasks
        train_ratio: Fraction for training set (for self-improvement)
        val_ratio: Fraction for validation set (for evaluation)
        test_ratio: Fraction for test set (for final reporting)
        shuffle: Whether to shuffle before splitting

    Returns:
        Tuple of (train_set, val_set, test_set)
    """
    random.seed(42)
    if shuffle:
        random.shuffle(frozen_set)

    n_tasks = len(frozen_set)
    train_size = int(n_tasks * train_ratio)
    val_size = int(n_tasks * val_ratio)

    train_set = frozen_set[:train_size]
    val_set = frozen_set[train_size:train_size + val_size]
    test_set = frozen_set[train_size + val_size:]

    return train_set, val_set, test_set


__all__ = ["EvaluationHarness", "split_fixed_frozen_set"]
