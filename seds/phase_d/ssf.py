"""SEDS Phase D: Structured Search Framework (§6.3).

Provides structured search capabilities for:
- Tool pattern search (finding effective tool usage patterns)
- Rollout exploration (breadth-first search over agent actions)
- Strategy composition (combining multiple search strategies)
"""
from __future__ import annotations

import copy
import hashlib
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import auto


logger = logging.getLogger(__name__)


@dataclass
class SearchNode:
    """Node in the search tree.

    Each node represents a state in the search space.
    """
    node_id: str
    task_id: str
    parent_id: Optional[str]
    depth: int
    rollout_id: Optional[str]

    # Search-specific state
    action: Optional[dict[str, Any]] = None
    heuristic_score: float = 0.0
    explored: bool = False
    expanded: bool = False

    # Result data
    answer: Optional[str] = None
    score: Optional[dict[str, Any]] = None
    spans: list[dict[str, Any]] = field(default_factory=list)

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        return isinstance(other, SearchNode) and self.node_id == other.node_id


@dataclass
class SearchMetrics:
    """Metrics tracked during search execution."""
    total_nodes_expanded: int = 0
    total_nodes_generated: int = 0
    total_time_ms: int = 0
    max_depth_reached: int = 0
    successful_completions: int = 0
    failed_attempts: int = 0
    frontier_sizes: list[int] = field(default_factory=list)

    def add_frontier_size(self, size: int):
        """Record frontier size at this point."""
        self.frontier_sizes.append(size)


class SearchState:
    """Abstract search state interface."""

    @abstractmethod
    def get_valid_actions(self) -> list[dict[str, Any]]:
        """Get list of valid actions for this state.

        Returns:
            List of action dictionaries
        """
        pass

    @abstractmethod
    def apply_action(self, action: dict[str, Any]) -> 'SearchState':
        """Apply an action and return the new state.

        Args:
            action: Action dictionary

        Returns:
            New state after applying the action
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if this state is terminal (solution found or dead end).

        Returns:
            True if terminal
        """
        pass

    @abstractmethod
    def get_heuristic_score(self) -> float:
        """Get heuristic estimate of how close we are to solution.

        Returns:
            Heuristic score (higher is better)
        """
        pass

    @abstractmethod
    def extract_answer(self) -> Optional[str]:
        """Extract final answer from this state.

        Returns:
            Answer string or None if not terminal
        """
        pass

    @abstractmethod
    def get_children(self) -> list['SearchState']:
        """Get child states (expanding the current node).

        Returns:
            List of child states
        """
        pass


class RolloutStrategy:
    """Base class for rollout search strategies."""

    @abstractmethod
    def select_action(self, state: SearchState) -> Optional[dict[str, Any]]:
        """Select an action from valid actions.

        Args:
            state: Current search state

        Returns:
            Selected action dict or None
        """
        pass

    @abstractmethod
    def expand_node(self, node: SearchNode, state: SearchState) -> list[SearchNode]:
        """Expand a node by generating child nodes.

        Args:
            node: Node to expand
            state: Current search state

        Returns:
            List of child nodes
        """
        pass

    @abstractmethod
    def should_continue_search(self, metrics: SearchMetrics) -> bool:
        """Check if search should continue.

        Args:
            metrics: Current search metrics

        Returns:
            True if search should continue
        """
        pass


class BreadthFirstStrategy(RolloutStrategy):
    """Breadth-first search rollout strategy.

    Explores states level by level, trying all valid actions before moving deeper.
    Good for finding complete solutions with minimal depth.
    """

    def select_action(self, state: SearchState) -> Optional[dict[str, Any]]:
        """Select first valid action (randomized)."""
        valid_actions = state.get_valid_actions()
        if not valid_actions:
            return None
        return valid_actions[0]  # Simplified - could randomize

    def expand_node(self, node: SearchNode, state: SearchState) -> list[SearchNode]:
        """Generate all child nodes by applying all valid actions."""
        valid_actions = state.get_valid_actions()
        children = []

        for action in valid_actions:
            new_state = state.apply_action(action)
            child = SearchNode(
                node_id=f"{node.node_id}_{hashlib.md5(str(action).encode()).hexdigest()[:8]}",
                task_id=node.task_id,
                parent_id=node.node_id,
                depth=node.depth + 1,
                rollout_id=node.rollout_id,
                action=action,
                heuristic_score=new_state.get_heuristic_score(),
            )
            children.append(child)

        return children

    def should_continue_search(self, metrics: SearchMetrics) -> bool:
        """Continue until frontier is exhausted or time limit reached."""
        return len(metrics.frontier_sizes) < 100  # Limit for demo


class DepthFirstStrategy(RolloutStrategy):
    """Depth-first search rollout strategy.

    Explores one path deeply before backtracking. Good for quick solutions.
    """

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth

    def select_action(self, state: SearchState) -> Optional[dict[str, Any]]:
        """Select first valid action."""
        valid_actions = state.get_valid_actions()
        if not valid_actions:
            return None
        return valid_actions[0]

    def expand_node(self, node: SearchNode, state: SearchState) -> list[SearchNode]:
        """Generate single child by applying first valid action."""
        valid_actions = state.get_valid_actions()
        if not valid_actions:
            return []

        action = valid_actions[0]
        new_state = state.apply_action(action)
        child = SearchNode(
            node_id=f"{node.node_id}_{hashlib.md5(str(action).encode()).hexdigest()[:8]}",
            task_id=node.task_id,
            parent_id=node.node_id,
            depth=node.depth + 1,
            rollout_id=node.rollout_id,
            action=action,
            heuristic_score=new_state.get_heuristic_score(),
        )
        return [child]

    def should_continue_search(self, metrics: SearchMetrics) -> bool:
        """Continue until depth limit reached."""
        return metrics.max_depth_reached < self.max_depth


class RolloutSearch:
    """Rollout-based search with multiple strategies."""

    def __init__(
        self,
        task_id: str,
        initial_state: SearchState,
        max_depth: int = 10,
        max_frontier_size: int = 100,
        strategies: Optional[list[RolloutStrategy]] = None
    ):
        """
        Initialize rollout search.

        Args:
            task_id: Task identifier
            initial_state: Initial search state
            max_depth: Maximum depth to search
            max_frontier_size: Maximum frontier size for heap management
            strategies: List of search strategies to use
        """
        self.task_id = task_id
        self.initial_state = initial_state
        self.max_depth = max_depth
        self.max_frontier_size = max_frontier_size

        # Use default strategies if none provided
        if strategies is None:
            self.strategies = [
                BreadthFirstStrategy(),
                DepthFirstStrategy(max_depth=max_depth),
            ]
        else:
            self.strategies = strategies

        # Search state
        self.root = SearchNode(
            node_id="root",
            task_id=task_id,
            parent_id=None,
            depth=0,
            rollout_id=None,
        )
        self.node_map: dict[str, SearchNode] = {self.root.node_id: self.root}
        self.frontier: deque[SearchNode] = deque([self.root])

        # Metrics
        self.metrics = SearchMetrics()
        self.start_time = time.time()

        logger.debug(f"RolloutSearch initialized for task={task_id}")

    def search(self, timeout_ms: int = 5000) -> tuple[Optional[str], list[dict[str, Any]]]:
        """Execute search with timeout.

        Args:
            timeout_ms: Maximum time to search in milliseconds

        Returns:
            Tuple of (answer, spans) or (None, []) if no solution found
        """
        start_time = time.time()

        while self.frontier:
            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                logger.warning(f"Search timeout after {elapsed_ms:.1f}ms")
                break

            # Get next node
            node = self.frontier.popleft()
            node.explored = True

            # Create state from node
            state = self._state_from_node(node)

            # Check if terminal
            if state.is_terminal():
                answer = state.extract_answer()
                spans = node.spans
                self.metrics.successful_completions += 1
                logger.info(f"Solution found: answer={answer[:50] if answer else 'None'}")
                return answer, spans

            # Expand node
            if node.depth >= self.max_depth:
                continue

            # Apply each strategy
            for strategy in self.strategies:
                child_nodes = strategy.expand_node(node, state)
                for child in child_nodes:
                    # Avoid duplicates
                    if child.node_id not in self.node_map:
                        # Clone state for child
                        new_state = state.apply_action(child.action)
                        child.spans = new_state.get_spans() if hasattr(new_state, 'get_spans') else []

                        self.node_map[child.node_id] = child
                        self.frontier.append(child)
                        self.metrics.total_nodes_generated += 1

            self.metrics.total_nodes_expanded += 1
            self.metrics.max_depth_reached = max(
                self.metrics.max_depth_reached,
                node.depth + 1
            )
            self.metrics.add_frontier_size(len(self.frontier))

        # No solution found
        elapsed_ms = (time.time() - start_time) * 1000
        self.metrics.total_time_ms = int(elapsed_ms)
        logger.info(f"Search finished: {self.metrics.total_nodes_expanded} nodes expanded")

        return None, []

    def _state_from_node(self, node: SearchNode) -> SearchState:
        """Convert node to search state."""
        # Create a mutable copy of initial state
        state = copy.deepcopy(self.initial_state)

        # Reconstruct path actions
        current_id = node.parent_id
        while current_id and current_id != "root":
            parent = self.node_map[current_id]
            state = state.apply_action(parent.action)
            current_id = parent.parent_id

        return state

    def get_node(self, node_id: str) -> Optional[SearchNode]:
        """Get node by ID."""
        return self.node_map.get(node_id)

    def get_metrics(self) -> SearchMetrics:
        """Get search metrics."""
        self.metrics.total_time_ms = int((time.time() - self.start_time) * 1000)
        return self.metrics


class PatternSearcher:
    """Search for effective tool usage patterns in collected traces."""

    def __init__(self):
        """Initialize pattern searcher."""
        self.tool_pattern_counts: dict[str, dict[str, int]] = defaultdict(dict)
        self.successful_patterns: dict[str, dict[str, Any]] = {}
        self.failure_patterns: dict[str, dict[str, Any]] = {}

    def analyze_traces(
        self,
        traces: list[dict[str, Any]],
        successful: bool = True
    ):
        """Analyze traces and extract patterns.

        Args:
            traces: List of trace dictionaries
            successful: True for successful traces, False for failures
        """
        for trace in traces:
            tool_calls = trace.get("spans", [])
            for call in tool_calls:
                tool_name = call.get("tool_name", "unknown")
                if tool_name == "unknown":
                    continue

                # Extract pattern from call arguments
                inputs = call.get("inputs", {})
                pattern = self._extract_pattern_key(inputs)

                self.tool_pattern_counts[successful][tool_name][pattern] = (
                    self.tool_pattern_counts[successful][tool_name].get(pattern, 0) + 1
                )

                # Store successful patterns
                if successful:
                    self.successful_patterns.setdefault(tool_name, set()).add(pattern)

                # Store failure patterns
                if not successful and call.get("error"):
                    self.failure_patterns.setdefault(tool_name, set()).add(pattern)

    def _extract_pattern_key(self, inputs: dict[str, Any]) -> str:
        """Extract a simple key from inputs for pattern matching."""
        # Take first few fields in a deterministic order
        key_fields = list(inputs.items())[:3]
        return ",".join(f"{k}:{v}" for k, v in key_fields)

    def get_top_patterns(self, tool_name: str, successful: bool = True, top_n: int = 5) -> list[tuple[str, int]]:
        """Get most frequent patterns for a tool.

        Args:
            tool_name: Name of the tool
            successful: True for successful, False for failure patterns
            top_n: Number of top patterns to return

        Returns:
            List of (pattern, count) tuples
        """
        patterns = self.tool_pattern_counts[successful].get(tool_name, {})
        return sorted(patterns.items(), key=lambda x: -x[1])[:top_n]

    def get_successful_patterns(self, tool_name: str) -> list[str]:
        """Get list of successful patterns for a tool."""
        return list(self.successful_patterns.get(tool_name, set()))

    def get_failure_patterns(self, tool_name: str) -> list[str]:
        """Get list of failure patterns for a tool."""
        return list(self.failure_patterns.get(tool_name, set()))

    def get_pattern_feedback(self, tool_name: str) -> dict[str, Any]:
        """Get pattern feedback for a tool.

        Returns:
            Dictionary with success rates and pattern recommendations
        """
        total = self.tool_pattern_counts[True].get(tool_name, {})
        success_count = sum(total.values())

        failures = self.tool_pattern_counts[False].get(tool_name, {})
        failure_count = sum(failures.values())

        total_calls = success_count + failure_count
        success_rate = (success_count / total_calls * 100) if total_calls > 0 else 0

        successful_patterns = self.get_successful_patterns(tool_name)
        failure_patterns = self.get_failure_patterns(tool_name)

        return {
            "tool_name": tool_name,
            "success_rate": success_rate,
            "successful_patterns": successful_patterns,
            "failure_patterns": failure_patterns,
            "recommendation": (
                "Use patterns from successful_patterns list"
                if failure_patterns
                else "All patterns successful"
            ),
        }
