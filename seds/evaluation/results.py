"""Shared data structures for evaluation results.

Defines the PairedResult class used by multiple evaluation modules.

Author: AO Worker
Date: 2026-09-06
"""

from typing import Optional

from seds.domains.base import Score, TaskDomain


class PairedResult:
    """Result of paired comparison between parent and child."""

    task_id: str
    parent_score: Optional[Score]
    child_score: Optional[Score]
    domain: TaskDomain

    @property
    def parent_correct(self) -> bool:
        """Check if parent score indicates correctness."""
        return self.parent_score is not None and self.parent_score.correct

    @property
    def child_correct(self) -> bool:
        """Check if child score indicates correctness."""
        return self.child_score is not None and self.child_score.correct
