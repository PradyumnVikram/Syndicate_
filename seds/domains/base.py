"""SEDS core dataclass contracts (§5).

These are frozen dataclasses that define the canonical data structures
for the entire SEDS pipeline. Keep field names and types exact.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class Task:
    """Main task instance (includes reference)."""
    task_id: str
    inputs: dict[str, Any]
    reference: Any  # ground truth — HOST-SIDE ONLY, never mounted into a sandbox


@dataclass(frozen=True)
class Score:
    correct: bool
    partial: float  # 0.0–1.0
    detail: dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    json_schema: dict
    impl: Callable  # executed inside the sandbox; recorded for replay


@dataclass(frozen=True)
class TaskDomain:
    name: str
    goal: str
    tools: list[ToolSpec]
    train_tasks: list[Task]  # visible to the search
    val_tasks: list[Task]  # FROZEN; gates promotion + final selection
    evaluate: Callable[[Task, str], Score]  # runs HOST-SIDE, outside the container


@dataclass
class RolloutResult:
    node_id: str
    task_id: str
    answer: str | None
    score: Score
    spans: list[dict]  # neatlogs TelemetrySpanV2, normalized
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int  # nano bills these and they are invisible in output — track separately
    cost_usd: float
    wall_ms: int
    crashed: bool
    timed_out: bool
    error_log: str


class AgentSystem(abc.ABC):
    """Base class for all generated agent candidates."""

    @abc.abstractmethod
    def forward(self, task_info: dict[str, Any]) -> tuple[str, list[Any]]:
        """Executes task and returns the final answer along with normalized event traces."""
        pass
