# Selector Module for SEDS Self-Improving Agent Factory
# Phase F Implementation
from dataclasses import dataclass, field, replace
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto
from collections import defaultdict, deque
import math
import random
import copy

# ==================== Enums ====================

class CMP(Enum):
    LESS_THAN = auto()
    EQUAL = auto()
    GREATER_THAN = auto()

class PolicyType(Enum):
    EPSILON = auto()
    THOMPSON = auto()
    EXP3 = auto()
    UCB = auto()

# ==================== Exception Classes ====================

class SelectorError(Exception):
    """Base exception for selector module errors."""
    pass

class ParetoFrontierError(SelectorError):
    """Exception for Pareto frontier operations."""
    pass

class NodeNotFoundError(SelectorError):
    """Exception when a node is not found."""
    pass

class HistoryMismatchError(SelectorError):
    """Exception when history doesn't match expected format."""
    pass

# ==================== Core Data Structures ====================

@dataclass(frozen=True)
class NodeEvent:
    """Event associated with a synthetic node execution."""
    event_id: str
    step_index: int
    agent_name: str
    module: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error_log: Optional[str] = None
    timestamp: float = 0.0

@dataclass(frozen=True)
class MetricsSnapshot:
    """Snapshot of evaluation metrics."""
    success_rate: float
    average_reward: float
    standard_deviation: float
    coverage_score: float
    novelty_score: float
    any_metric: Dict[str, float] = field(default_factory=dict)
    node_id: str = ""

@dataclass
class SyntheticNode:
    """
    A synthetic node in the selector's execution graph.
    
    Attributes:
        node_id: Unique identifier for the node
        parents: List of parent node IDs
        creation_time: Timestamp of node creation
        is_predefined: Whether this is a predefined node
        confidence_interval: Tuple of (lower, upper) bounds for expected performance
    """
    node_id: str
    parents: Tuple[str, ...] = ()
    creation_time: float = 0.0
    is_predefined: bool = False
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    def __post_init__(self):
        """Validate node ID uniqueness."""
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("node_id must be a non-empty string")

# ==================== Pareto Frontier Implementation ====================

class ParetoFrontier:
    """
    Maintains the Pareto frontier of candidate nodes based on performance metrics.
    
    A node n is Pareto-optimal if there is no other node n' such that:
    - n' performs at least as well as n in all metrics, AND
    - n' performs strictly better than n in at least one metric
    """
    
    def __init__(self):
        """Initialize an empty Pareto frontier."""
        self._frontier: List[MetricsSnapshot] = []
        self._metrics_map: Dict[str, List[MetricsSnapshot]] = defaultdict(list)
    
    def add(self, snapshot: MetricsSnapshot, node_id: str):
        """
        Add a snapshot to the Pareto frontier.

        Args:
            snapshot: The metrics snapshot to add
            node_id: The node associated with this snapshot
        """
        # Attach node_id to snapshot before using it in comparisons
        # This is required for get_dominant_node() and removal-by-id logic
        snapshot = MetricsSnapshot(
            node_id=node_id,
            success_rate=snapshot.success_rate,
            average_reward=snapshot.average_reward,
            standard_deviation=snapshot.standard_deviation,
            coverage_score=snapshot.coverage_score,
            novelty_score=snapshot.novelty_score,
            any_metric=snapshot.any_metric
        )

        # Remove existing entries for this node if any
        self._frontier = [s for s in self._frontier
                         if getattr(s, 'node_id', '') != node_id]

        # Remove dominated entries
        self._frontier = [
            s for s in self._frontier
            if not self._is_dominated(s, snapshot)
        ]

        # Add new snapshot
        self._frontier.append(snapshot)

        # Update metrics map
        for metric_name, metric_value in snapshot.any_metric.items():
            self._metrics_map[metric_name].append(snapshot)

        # Sort frontier (in-place for efficiency)
        self._frontier.sort(key=lambda s: (
            -s.any_metric.get('coverage_score', 0.0),
            -s.any_metric.get('novelty_score', 0.0),
            -s.any_metric.get('success_rate', 0.0),
            -s.any_metric.get('average_reward', 0.0)
        ))
    
    def _is_dominated(self, candidate: MetricsSnapshot, new: MetricsSnapshot) -> bool:
        """
        Check if candidate is dominated by new snapshot.
        
        A snapshot n dominates candidate m if:
        - For all metrics, n's value >= m's value
        - For at least one metric, n's value > m's value
        """
        # Check each metric in candidate
        for metric_name in candidate.any_metric:
            cand_value = candidate.any_metric[metric_name]
            new_value = new.any_metric.get(metric_name, cand_value)
            
            # If new value is less than or equal for all metrics, new dominates
            if new_value < cand_value - 1e-9:
                return False
            if new_value > cand_value + 1e-9:
                return True  # New is better in at least one metric
        
        # Check coverage_score and novelty_score specifically
        cand_coverage = candidate.coverage_score
        new_coverage = new.coverage_score
        
        if new_coverage < cand_coverage - 1e-9:
            return False
        if new_coverage > cand_coverage + 1e-9:
            return True
        
        cand_novelty = candidate.novelty_score
        new_novelty = new.novelty_score
        
        if new_novelty < cand_novelty - 1e-9:
            return False
        if new_novelty > cand_novelty + 1e-9:
            return True
        
        # Equal in all metrics, so new does NOT dominate
        return False
    
    def get_frontier(self) -> List[MetricsSnapshot]:
        """Get the current Pareto frontier."""
        return list(self._frontier)
    
    def get_dominant_node(self) -> Optional[str]:
        """
        Get the ID of the dominant node from the Pareto frontier.
        
        Returns None if frontier is empty.
        """
        if not self._frontier:
            return None
        
        # Return the node ID from the first (best) entry
        return getattr(self._frontier[0], 'node_id', None)
    
    def get_candidates(self, k: int = 10) -> List[MetricsSnapshot]:
        """
        Get the top k candidates from the Pareto frontier.
        
        Args:
            k: Number of candidates to return
            
        Returns:
            List of MetricsSnapshot objects (sorted by quality)
        """
        return list(self._frontier[:k])
    
    def size(self) -> int:
        """Return the current size of the Pareto frontier."""
        return len(self._frontier)
    
    def clear(self):
        """Clear all entries from the Pareto frontier."""
        self._frontier.clear()
        self._metrics_map.clear()

# ==================== Node Lineage Tracking ====================

class NodeLineage:
    """
    Tracks the lineage of nodes from a seed node.
    
    Used to manage candidate rollback and mutation during selection.
    """
    
    def __init__(self):
        """Initialize a new lineage tracker."""
        self._lineage: Dict[str, SyntheticNode] = {}
        self._reverse_links: Dict[str, List[str]] = defaultdict(list)
    
    def add(self, node: SyntheticNode, parent_id: Optional[str] = None):
        """
        Add a node to the lineage.
        
        Args:
            node: The node to add
            parent_id: Optional parent node ID for lineage tracking
        """
        self._lineage[node.node_id] = node
        
        if parent_id and parent_id in self._lineage:
            self._reverse_links[parent_id].append(node.node_id)
    
    def get_lineage(self, node_id: str) -> List[SyntheticNode]:
        """
        Get the lineage of nodes starting from a given node.
        
        Args:
            node_id: The starting node ID
            
        Returns:
            List of nodes in lineage order (root first)
        """
        lineage = []
        current_id = node_id
        
        while current_id in self._lineage:
            lineage.append(self._lineage[current_id])
            # Get the parent (reverse links are from parent to children)
            children = self._reverse_links.get(current_id, [])
            if not children:
                break
            current_id = children[0]  # Get the first child
        
        return lineage
    
    def get_parents(self, node_id: str) -> List[SyntheticNode]:
        """
        Get all direct parent nodes of a given node.
        
        Args:
            node_id: The node ID
            
        Returns:
            List of parent nodes
        """
        if node_id not in self._lineage:
            raise NodeNotFoundError(f"Node {node_id} not found in lineage")
        
        parent_ids = self._reverse_links.get(node_id, [])
        return [self._lineage[pid] for pid in parent_ids]
    
    def get_children(self, node_id: str) -> List[SyntheticNode]:
        """
        Get all direct child nodes of a given node.
        
        Args:
            node_id: The node ID
            
        Returns:
            List of child nodes
        """
        return [self._lineage[child_id] 
                for child_id in self._reverse_links.get(node_id, [])]
    
    def size(self) -> int:
        """Return the number of nodes in the lineage."""
        return len(self._lineage)
    
    def is_ancestor(self, ancestor_id: str, descendant_id: str) -> bool:
        """
        Check if ancestor_id is an ancestor of descendant_id.
        
        Args:
            ancestor_id: Potential ancestor node ID
            descendant_id: Potential descendant node ID
            
        Returns:
            True if ancestor is a direct or indirect ancestor
        """
        if ancestor_id not in self._lineage or descendant_id not in self._lineage:
            return False
        
        current = descendant_id
        while current:
            if current == ancestor_id:
                return True
            current = self._reverse_links.get(current, [])[0]
        
        return False

# ==================== Archive Item ====================

@dataclass
class ArchiveItem:
    """
    An archived node with all its associated data.
    
    Used by the selector to maintain a repository of evaluated nodes.
    """
    node: SyntheticNode
    snapshot: MetricsSnapshot
    timestamps: List[float] = field(default_factory=list)
    lineage_id: Optional[str] = None
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_timestamp(self, timestamp: float):
        """Record a timestamp for this item."""
        self.timestamps.append(timestamp)
    
    def get_creation_time(self) -> float:
        """Get the earliest creation time recorded."""
        return min(self.timestamps) if self.timestamps else 0.0
    
    def get_average_interval(self) -> float:
        """Get the average time between creation and evaluation."""
        if len(self.timestamps) < 2:
            return 0.0
        return sum(self.timestamps[1:] - self.timestamps[:-1]) / (len(self.timestamps) - 1)

# ==================== Archive Tree ====================

class ArchiveTree:
    """
    A tree structure that organizes archive items.
    
    Maintains a hierarchical view of synthetic nodes for efficient lookup
    and mutation management.
    """
    
    def __init__(self):
        """Initialize a new archive tree."""
        self._root: Optional[ArchiveItem] = None
        self._items: Dict[str, ArchiveItem] = {}
        self._children: Dict[str, List[str]] = defaultdict(list)
    
    def add(self, item: ArchiveItem, parent_id: Optional[str] = None):
        """
        Add an item to the archive tree.
        
        Args:
            item: The archive item to add
            parent_id: Optional parent item ID
        """
        self._items[item.node.node_id] = item
        
        if parent_id:
            if parent_id not in self._items:
                raise NodeNotFoundError(f"Parent {parent_id} not found in archive")
            self._children[parent_id].append(item.node.node_id)
        
        if item.node.is_predefined and parent_id is None:
            self._root = item
        elif not item.node.is_predefined:
            # Add to root by default if no parent specified
            if not self._children:
                self._children['root'].append(item.node.node_id)
    
    def get(self, node_id: str) -> ArchiveItem:
        """
        Get an item by node ID.
        
        Args:
            node_id: The node ID
            
        Returns:
            The archive item
            
        Raises:
            NodeNotFoundError: If the node is not found
        """
        if node_id not in self._items:
            raise NodeNotFoundError(f"Node {node_id} not found in archive")
        return self._items[node_id]
    
    def get_children(self, node_id: str) -> List[ArchiveItem]:
        """
        Get all children of a given node.
        
        Args:
            node_id: The parent node ID
            
        Returns:
            List of child archive items
        """
        if node_id not in self._items:
            raise NodeNotFoundError(f"Node {node_id} not found in archive")
        
        child_ids = self._children.get(node_id, [])
        return [self._items[pid] for pid in child_ids]
    
    def get_ancestors(self, node_id: str) -> List[ArchiveItem]:
        """
        Get all ancestor nodes of a given node.
        
        Args:
            node_id: The node ID
            
        Returns:
            List of ancestor archive items (root first)
        """
        if node_id not in self._items:
            raise NodeNotFoundError(f"Node {node_id} not found in archive")
        
        ancestors = []
        current_id = node_id
        
        # Move up through parent links
        while True:
            item = self._items.get(current_id)
            if item is None:
                break
            
            ancestors.append(item)
            
            # Try to find parent by checking reverse links
            parent_ids = []
            for pid, children in self._children.items():
                if current_id in children:
                    parent_ids.append(pid)
                    break
            
            if not parent_ids:
                break
            
            current_id = parent_ids[0]
        
        ancestors.reverse()
        return ancestors
    
    def size(self) -> int:
        """Return the number of items in the archive."""
        return len(self._items)
    
    def root(self) -> Optional[ArchiveItem]:
        """Get the root item if it exists."""
        return self._root

# ==================== Cold Start UCB Selector ====================

class ColdStartUCBSelector:
    """
    UCB-based selector for cold-start scenarios.
    
    Uses Upper Confidence Bound to balance exploration and exploitation
    when few samples are available.
    """
    
    def __init__(self, alpha: float = 1.0, c: float = 2.0):
        """
        Initialize the ColdStartUCBSelector.
        
        Args:
            alpha: Base confidence factor
            c: Exploration parameter (default: 2.0)
        """
        self.alpha = alpha
        self.c = c
        self._counts: Dict[str, int] = defaultdict(int)
        self._values: Dict[str, float] = defaultdict(float)
        self._is_predefined: Dict[str, bool] = defaultdict(bool)
    
    def add_predefined_node(self, node_id: str):
        """Mark a node as predefined (not selected for mutation)."""
        self._is_predefined[node_id] = True
    
    def reset(self):
        """Reset the selector statistics."""
        self._counts.clear()
        self._values.clear()
        self._is_predefined.clear()
    
    def select(self) -> Optional[str]:
        """
        Select a node to evaluate using UCB.
        
        Returns:
            The selected node ID, or None if no eligible nodes
        """
        eligible_nodes = [
            node_id for node_id in self._counts
            if not self._is_predefined.get(node_id, False)
        ]
        
        if not eligible_nodes:
            return None
        
        # Apply UCB formula
        ucb_values = []
        for node_id in eligible_nodes:
            n = self._counts[node_id]
            value = self._values[node_id]
            
            if n == 0:
                # Pure exploration
                ucb = float('inf')
            else:
                ucb = value + self.c * math.sqrt(math.log(sum(self._counts.values())) / n)
            
            ucb_values.append((ucb, node_id))
        
        # Return node with highest UCB
        ucb_values.sort(key=lambda x: x[0], reverse=True)
        return ucb_values[0][1]
    
    def update(self, node_id: str, reward: float):
        """
        Update the selector statistics after an evaluation.
        
        Args:
            node_id: The node ID that was evaluated
            reward: The reward received (0 to 1)
        """
        self._counts[node_id] += 1
        self._values[node_id] += (reward - self._values[node_id]) / self._counts[node_id]
    
    def get_statistics(self, node_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific node.
        
        Args:
            node_id: The node ID
            
        Returns:
            Dictionary with statistics
        """
        count = self._counts.get(node_id, 0)
        value = self._values.get(node_id, 0.0)
        
        return {
            'node_id': node_id,
            'count': count,
            'average_reward': value,
            'is_predefined': self._is_predefined.get(node_id, False)
        }

# ==================== Epsilon Greedy Default Policy ====================

class EpsilonDefaultPolicy:
    """
    Epsilon-greedy policy for selecting nodes.
    
    With probability epsilon, selects a random node for exploration.
    With probability (1 - epsilon), selects the best-known node.
    """
    
    def __init__(self, epsilon: float = 0.1):
        """
        Initialize the EpsilonDefaultPolicy.
        
        Args:
            epsilon: Exploration probability (default: 0.1)
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be between 0.0 and 1.0")
        self.epsilon = epsilon
    
    def select(self, candidate_ids: List[str], use_best: bool = True) -> str:
        """
        Select a node using epsilon-greedy strategy.
        
        Args:
            candidate_ids: List of candidate node IDs
            use_best: If True, select best-known node; otherwise random
            
        Returns:
            The selected node ID
        """
        if not candidate_ids:
            raise ValueError("Candidate list cannot be empty")
        
        # Exploration phase
        if random.random() < self.epsilon or not use_best:
            return random.choice(candidate_ids)
        
        # Exploitation phase - use best known node
        # For simplicity, select the first node (in production, this should use actual metrics)
        return candidate_ids[0]
    
    def mutate_with_exploration(self, candidate_ids: List[str], explore_ratio: float = 0.3) -> Tuple[str, str]:
        """
        Select one node for use and one for exploration via mutation.
        
        Args:
            candidate_ids: List of candidate node IDs
            explore_ratio: Ratio of candidates to select for exploration (0.0 to 1.0)
            
        Returns:
            Tuple of (selected_id, explored_id)
        """
        explore_count = max(1, int(len(candidate_ids) * explore_ratio))
        
        # Shuffle candidates for randomness
        shuffled = random.sample(candidate_ids, len(candidate_ids))
        
        selected_id = shuffled[0]
        explored_id = random.choice(shuffled[1:1 + explore_count]) if len(shuffled) > 1 else shuffled[0]
        
        return (selected_id, explored_id)

# ==================== Rollback Ledger ====================

class RollbackLedger:
    """
    Maintains a history of successful and failed rollbacks.
    
    Used by the selector to manage state mutations and recovery.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize the rollback ledger.
        
        Args:
            max_history: Maximum number of entries to keep
        """
        self._successful: List[Dict[str, Any]] = []
        self._failed: List[Dict[str, Any]] = []
        self._max_history = max_history
        self._current_transaction: Optional[Dict[str, Any]] = None
    
    def begin_transaction(self, node_id: str, mutation_type: str):
        """
        Begin a new transaction for state mutation.
        
        Args:
            node_id: The node ID being mutated
            mutation_type: Type of mutation (e.g., "add_parent", "remove_parent")
        """
        if self._current_transaction is not None:
            raise RuntimeError("Transaction already in progress")
        
        self._current_transaction = {
            'node_id': node_id,
            'mutation_type': mutation_type,
            'timestamp': time.time(),
            'state_snapshot': {},  # Will be populated
            'status': 'pending'
        }
    
    def record_state_snapshot(self, state: Any):
        """
        Record the current state during a transaction.
        
        Args:
            state: The current state to snapshot
        """
        if self._current_transaction is None:
            raise RuntimeError("No active transaction")
        
        # Deep copy for serialization
        self._current_transaction['state_snapshot'] = copy.deepcopy(state)
    
    def commit_transaction(self) -> bool:
        """
        Commit the current transaction.
        
        Returns:
            True if committed successfully
            
        Raises:
            RuntimeError: If no active transaction
        """
        if self._current_transaction is None:
            raise RuntimeError("No active transaction")
        
        try:
            # Record successful transaction
            self._successful.append(self._current_transaction.copy())
            
            # Trim history
            if len(self._successful) > self._max_history:
                self._successful.pop(0)
            
            self._current_transaction = None
            return True
        except Exception as e:
            self._current_transaction['status'] = 'failed'
            self._current_transaction['error'] = str(e)
            return False
    
    def rollback_transaction(self) -> bool:
        """
        Roll back the current transaction.
        
        Returns:
            True if rolled back successfully
            
        Raises:
            RuntimeError: If no active transaction
        """
        if self._current_transaction is None:
            raise RuntimeError("No active transaction")
        
        try:
            # Record failed transaction
            self._failed.append(self._current_transaction.copy())
            
            # Trim history
            if len(self._failed) > self._max_history:
                self._failed.pop(0)
            
            self._current_transaction = None
            return True
        except Exception as e:
            self._current_transaction['status'] = 'failed'
            self._current_transaction['error'] = str(e)
            return False
    
    def is_transaction_active(self) -> bool:
        """Check if a transaction is currently active."""
        return self._current_transaction is not None
    
    def get_successful_count(self) -> int:
        """Return the number of successful transactions."""
        return len(self._successful)
    
    def get_failed_count(self) -> int:
        """Return the number of failed transactions."""
        return len(self._failed)

# ==================== Selector Checkpoint ====================

class SelectorCheckpoint:
    """
    Serializes and deserializes the selector's internal state.
    
    Used for saving and loading selector state across evaluations.
    """
    
    def __init__(self):
        """Initialize the checkpoint manager."""
        self._data: Dict[str, Any] = {}
    
    def create_from(self, selector, lineage: NodeLineage, archive: ArchiveTree):
        """
        Create a checkpoint from selector components.
        
        Args:
            selector: The selector instance to checkpoint
            lineage: The lineage tracker
            archive: The archive tree
        """
        self._data = {
            'class_name': selector.__class__.__name__,
            'timestamp': time.time(),
            'selector_state': selector.__dict__.copy(),
            'lineage_size': lineage.size(),
            'archive_size': archive.size(),
            'lineage_ids': list(lineage._lineage.keys()),
            'archive_ids': list(archive._items.keys())
        }
    
    def save_to_file(self, filepath: str):
        """
        Save the checkpoint to a file.
        
        Args:
            filepath: Path to save the checkpoint
        """
        with open(filepath, 'w') as f:
            json.dump(self._data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """
        Load a checkpoint from a file.
        
        Args:
            filepath: Path to load the checkpoint from
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        with open(filepath, 'r') as f:
            self._data = json.load(f)
    
    def restore_to(self, selector: Any, lineage: NodeLineage, archive: ArchiveTree):
        """
        Restore state to selector components.
        
        Args:
            selector: The selector instance to restore
            lineage: The lineage tracker to populate
            archive: The archive tree to populate
            
        Raises:
            RuntimeError: If checkpoint is corrupted or incompatible
        """
        if 'selector_state' not in self._data:
            raise RuntimeError("Checkpoint data is corrupted")
        
        # Restore selector state
        try:
            selector.__dict__.update(self._data['selector_state'])
        except Exception as e:
            raise RuntimeError(f"Failed to restore selector state: {e}")
        
        # Restore lineage
        lineage._lineage = {
            nid: copy.deepcopy(node)
            for nid, node in self._data.get('lineage_ids', {}).items()
            if isinstance(nid, str)
        }
        lineage._reverse_links = defaultdict(list)
        
        # Restore archive
        archive._items = {}
        archive._children = defaultdict(list)
        archive._root = None
        
        for nid in self._data.get('archive_ids', []):
            archive._items[nid] = copy.deepcopy(self._data['archive_ids'][nid])
    
    def is_valid(self) -> bool:
        """Check if the checkpoint is valid."""
        return (
            'class_name' in self._data and
            'timestamp' in self._data and
            'selector_state' in self._data
        )

# ==================== SEDS Selector (Main Class) ====================

import time
import json

class SEDSSelector:
    """
    Main selector for SEDS (Self-Improving Agent Factory).
    
    Integrates selection algorithms, policy management, and mutation tracking
    to select and evolve candidate nodes for agent composition.
    """
    
    def __init__(
        self,
        policy_type: PolicyType = PolicyType.EPSILON,
        alpha: float = 1.0,
        c: float = 2.0,
        epsilon: float = 0.1,
        max_archive_size: int = 1000,
        allow_rollbacks: bool = True
    ):
        """
        Initialize the SEDS selector.
        
        Args:
            policy_type: Type of selection policy to use
            alpha: Base confidence factor for UCB
            c: Exploration parameter for UCB
            epsilon: Exploration probability for epsilon-greedy
            max_archive_size: Maximum number of nodes to keep in archive
            allow_rollbacks: Enable rollback functionality
        """
        # Create policy based on type
        self.policy_type = policy_type
        
        if policy_type == PolicyType.UCB:
            self.policy = ColdStartUCBSelector(alpha=alpha, c=c)
        else:
            self.policy = EpsilonDefaultPolicy(epsilon=epsilon)
        
        self.max_archive_size = max_archive_size
        self.allow_rollbacks = allow_rollbacks
        
        # Initialize components
        self.lineage = NodeLineage()
        self.archive = ArchiveTree()
        self.pareto_frontier = ParetoFrontier()
        self.rollback_ledger = RollbackLedger() if allow_rollbacks else None
        
        # Tracking state
        self._total_nodes_created: int = 0
        self._total_evaluations: int = 0
        self._checkpoint = SelectorCheckpoint()
    
    def add_predefined_node(self, node: SyntheticNode):
        """
        Add a predefined node that cannot be mutated.
        
        Args:
            node: The predefined node to add
        """
        # Add to lineage and archive
        self.lineage.add(node)
        
        archive_item = ArchiveItem(
            node=node,
            snapshot=MetricsSnapshot(
                success_rate=0.0,
                average_reward=0.0,
                standard_deviation=0.0,
                coverage_score=0.0,
                novelty_score=0.0,
                any_metric={}
            )
        )
        self.archive.add(archive_item)
        
        # Mark as predefined in policy
        if isinstance(self.policy, ColdStartUCBSelector):
            self.policy.add_predefined_node(node.node_id)
        
        # Track total nodes
        self._total_nodes_created += 1
    
    def select_node(self, candidates: List[str]) -> Optional[str]:
        """
        Select a node from candidates.
        
        Args:
            candidates: List of candidate node IDs
            
        Returns:
            The selected node ID, or None if no candidates
        """
        if not candidates:
            return None
        
        if self.policy_type == PolicyType.UCB:
            return self.policy.select()
        else:
            return self.policy.select(candidates)
    
    def evaluate_node(self, node_id: str, reward: float):
        """
        Evaluate a node and update statistics.
        
        Args:
            node_id: The node ID to evaluate
            reward: The reward received (0 to 1)
        """
        # Update policy
        if isinstance(self.policy, ColdStartUCBSelector):
            self.policy.update(node_id, reward)
        
        self._total_evaluations += 1
    
    def add_to_archive(
        self,
        node: SyntheticNode,
        snapshot: MetricsSnapshot,
        parent_id: Optional[str] = None
    ):
        """
        Add a node with its evaluation snapshot to the archive.
        
        Args:
            node: The synthetic node
            snapshot: The metrics snapshot
            parent_id: Optional parent node ID
        """
        # Increment total nodes created
        self._total_nodes_created += 1

        # Add to archive
        archive_item = ArchiveItem(
            node=node,
            snapshot=snapshot,
            lineage_id=parent_id
        )
        self.archive.add(archive_item, parent_id=parent_id)

        # Add to Pareto frontier
        self.pareto_frontier.add(snapshot, node_id=node.node_id)
        
        # Trim archive if needed
        if self.archive.size() > self.max_archive_size:
            # Remove oldest non-predefined items
            items_to_remove = []
            for item in self.archive._items.values():
                if not item.node.is_predefined and items_to_remove:
                    del self.archive._items[items_to_remove.pop(0)]
                    for pid, children in self.archive._children.items():
                        if children:
                            children.pop()
            
            self._total_nodes_created = max(0, self._total_nodes_created - len(items_to_remove))
    
    def get_best_candidate(self) -> Optional[str]:
        """
        Get the best candidate from the Pareto frontier.
        
        Returns:
            The node ID of the best candidate, or None
        """
        return self.pareto_frontier.get_dominant_node()
    
    def checkpoint(self, filepath: str):
        """
        Save the selector state to a checkpoint file.
        
        Args:
            filepath: Path to save the checkpoint
        """
        self._checkpoint.create_from(self, self.lineage, self.archive)
        self._checkpoint.save_to_file(filepath)
    
    def restore_checkpoint(self, filepath: str):
        """
        Restore the selector state from a checkpoint file.
        
        Args:
            filepath: Path to load the checkpoint from
        """
        self._checkpoint.load_from_file(filepath)
        self._checkpoint.restore_to(self, self.lineage, self.archive)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current selector statistics.
        
        Returns:
            Dictionary with statistics
        """
        best_candidate = self.get_best_candidate()
        
        return {
            'total_nodes_created': self._total_nodes_created,
            'total_evaluations': self._total_evaluations,
            'archive_size': self.archive.size(),
            'pareto_frontier_size': self.pareto_frontier.size(),
            'lineage_size': self.lineage.size(),
            'policy_type': self.policy_type.name,
            'best_candidate': best_candidate,
            'successful_transactions': self.rollback_ledger.get_successful_count() if self.rollback_ledger else 0,
            'failed_transactions': self.rollback_ledger.get_failed_count() if self.rollback_ledger else 0
        }
