"""Unseen-domain protocol template for SEDS evaluation.

This module defines the protocol for domain-agnostic evaluation of unseen tasks.
Each TaskDomain implementation must satisfy:
1. Provide evaluate() method that runs host-side only
2. Never expose Task.reference to sandbox
3. Provide deterministic scoring for training/evaluation stability
4. Support frozen val set construction for fair comparison

Example usage:
    domain = MultiHopQA_Domain.from_config(CONFIG)
    results = domain.evaluate(host_tasks, sandbox)
    report = ComparisonReportGenerator(
        baseline_name="v1",
        evolved_name="v2",
        baseline_results=baseline_results,
        evolved_results=evolved_results,
    ).generate_report()
"""

from .base import (
    TaskDomain,
    RolloutResult,
    Task,
    Score,
    ToolSpec,
)

__all__ = [
    "TaskDomain",
    "RolloutResult",
    "Task",
    "Score",
    "ToolSpec",
]
