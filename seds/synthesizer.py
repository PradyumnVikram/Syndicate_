"""Synthesizer module (Phase E) — code-space mutation proposal.

Based on §5.3 (mutation_operator_menu) and §6 (items 22-24).
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("seds.synthesizer")

# ── Mutation Operator Types (typed menu) ──────────────────────────────────────
# §5.3: prompt_edit, tool_edit, memory_edit, orchestration_edit, efficiency_edit

@dataclass
class MutationRequest:
    """Strict JSON schema for structured mutation requests."""

    operator: str  # "prompt_edit" | "tool_edit" | "memory_edit" | "orchestration_edit" | "efficiency_edit"
    target_file: str  # Relative path inside the agent repo
    old_str: str  # Text to replace
    new_str: str  # Replacement text
    rationale: str  # Human-readable explanation


# ── Mutation Context Builder ──────────────────────────────────────────────────
@dataclass
class MutationContext:
    """Context for mutation generation."""

    parent_code: str
    failure_traces: List[Dict[str, Any]]
    failure_mode_histogram: Dict[str, int]  # Primary signal
    ancestor_performance_log: List[Dict[str, Any]]
    remaining_budget: float
    current_node_id: str


@dataclass
class MutationResult:
    """Result of a single mutation attempt."""

    request: MutationRequest
    success: bool
    preflight_checked: bool
    diff_hash: str  # SHA-256 of old_str->new_str diff
    ast_matches_parent: bool  # AST-level similarity
    duplicate: bool  # Duplicate of existing mutation in queue


# ──── DEPENDENCIES ────────────────────────────────────────────────────────────
# Import broker and domain contracts
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from seds.broker.server import Broker
from seds.domains.base import AgentSystem, RolloutResult, Score


# ── AST Hash for Diversity Guard ─────────────────────────────────────────────
def _ast_hash(code: str) -> str:
    """Compute AST-based hash of code for diversity checking."""
    try:
        tree = ast.parse(code)
        # Simplified AST: only keep node types and important attributes
        nodes = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                nodes.append((type(node).__name__, node.name))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                nodes.append(("assign", getattr(node.targets[0], "id", None) if node.targets else None))
        return hashlib.sha256(json.dumps(nodes).encode()).hexdigest()[:16]
    except:
        # Fallback to text hash if AST fails
        return hashlib.sha256(code.encode()).hexdigest()[:16]


def _prompt_hash(prompt: str) -> str:
    """Compute hash of prompt text (for prompt_edit operator)."""
    return hashlib.sha256(prompt.strip().encode()).hexdigest()[:16]


# ──── PREDICTIVE FILTERING (preflight) ────────────────────────────────────────
def _preflight_mutation(request: MutationRequest, max_file_size: int = 50000) -> bool:
    """Quick syntax/logic sanity check before sending to LLM."""
    # Check file size
    if len(request.new_str) > max_file_size:
        logger.warning(f"Mutation too large: {len(request.new_str)} chars > {max_file_size}")
        return False

    # Check for obvious syntax errors (simple regex checks)
    if request.operator in ["tool_edit", "memory_edit"]:
        # JSON schema must be valid
        try:
            json.loads(request.new_str)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in {request.operator}: {request.new_str[:100]}")
            return False

    if request.operator == "efficiency_edit":
        # Should not contain debugging statements or excessive logging
        if re.search(r'print\(|logger\.debug\(|debugger\(', request.new_str):
            return False

    return True


# ──── DIVERSITY GUARD ────────────────────────────────────────────────────────
class DiversityGuard:
    """Rejects AST/prompt-hash collisions."""

    def __init__(self, max_similarity: float = 0.85):
        self.max_similarity = max_similarity
        self.prompt_hashes: Set[str] = set()

    def is_duplicate_prompt(self, prompt: str) -> bool:
        """Check if prompt hash already exists in queue."""
        h = _prompt_hash(prompt)
        if h in self.prompt_hashes:
            return True
        self.prompt_hashes.add(h)
        return False

    def has_ast_collision(self, code: str, existing_hashes: Set[str]) -> bool:
        """Check if new code's AST hash matches any existing code."""
        h = _ast_hash(code)
        return h in existing_hashes


# ──── STABLE STRUCTURED OUTPUT JSON SCHEMA ────────────────────────────────────
# This is used for deterministic tier (glm-4-7-flash) with structured outputs

MUTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "operator": {"type": "string", "enum": ["prompt_edit", "tool_edit", "memory_edit", "orchestration_edit", "efficiency_edit"]},
        "target_file": {"type": "string"},
        "old_str": {"type": "string"},
        "new_str": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["operator", "target_file", "old_str", "new_str", "rationale"],
}

MUTATION_LIST_SCHEMA = {
    "type": "array",
    "items": MUTATION_SCHEMA,
}


# ──── MUTATION MENU (TYPED OPERATORS) ────────────────────────────────────────
class MutationMenu:
    """Typed mutation-operator menu."""

    OPERATORS = ["prompt_edit", "tool_edit", "memory_edit", "orchestration_edit", "efficiency_edit"]

    @staticmethod
    def operator_description(op: str) -> str:
        descriptions = {
            "prompt_edit": "Modify system prompt, task description, or few-shot examples in agent prompts.",
            "tool_edit": "Adjust tool invocation schema, parameters, or add/remove tool definitions.",
            "memory_edit": "Change memory structure (e.g., RAG embeddings, vector stores, cache keys).",
            "orchestration_edit": "Rewrite multi-agent coordination (message routing, DAG topology, state sync).",
            "efficiency_edit": "Optimize code for speed, reduce latency, or shrink context window usage.",
        }
        return descriptions.get(op, "General code modification.")


# ──── SYNTHESIZER CORE ────────────────────────────────────────────────────────
class SEDSSynthesizer:
    """Code-space mutation proposal engine with best-of-N sampling.

    Features:
    - Typed mutation-operator menu (prompt_edit, tool_edit, memory_edit, orchestration_edit, efficiency_edit)
    - Structured outputs (strict JSON schema) via deterministic tier
    - Best-of-N (N~5) sampling with preflight filtering
    - Mutation context built from parent code + folded failure traces + failure-mode histogram
    - Diversity guard rejecting AST/prompt-hash collisions
    - Valid-mutation rate tracking
    """

    def __init__(
        self,
        broker: Broker,
        best_of_n: int = 5,
        deterministic_tier: str = "deterministic",
        max_file_size: int = 50000,
        max_similar_mutations: int = 10,
    ):
        self.broker = broker
        self.best_of_n = best_of_n
        self.deterministic_tier = deterministic_tier
        self.max_file_size = max_file_size
        self.max_similar_mutations = max_similar_mutations

        self.diversity_guard = DiversityGuard()
        self.valid_mutation_rate = 0.0
        self._call_count = 0
        self._valid_count = 0

    def sample_mutations(
        self,
        context: MutationContext,
    ) -> List[MutationResult]:
        """Generate best-of-N mutations with preflight filtering and diversity guard.

        Args:
            context: Context containing parent code, failure traces, histogram, etc.

        Returns:
            List of MutationResult objects, sorted by success likelihood.
        """
        prompt = self._build_mutation_prompt(context)

        # Collect attempts
        results: List[MutationResult] = []
        attempted_hashes: Set[str] = set()
        prompt_hashes: Set[str] = set()

        for i in range(self.best_of_n):
            # Filter out duplicates at prompt level
            if len(prompt_hashes) >= self.max_similar_mutations:
                break

            # Request mutation from LLM
            req = self._request_mutation(prompt, i + 1, len(self.best_of_n))

            # Preflight check
            preflight_ok = _preflight_mutation(req, self.max_file_size)

            # Compute diff hash for AST similarity
            diff_hash = hashlib.sha256(f"{req.old_str}:{req.new_str}".encode()).hexdigest()[:16]

            # AST collision check
            ast_matches_parent = (diff_hash == _ast_hash(req.new_str) and diff_hash != _ast_hash(req.old_str))

            # Duplicate detection
            duplicate = diff_hash in attempted_hashes

            if duplicate:
                logger.debug(f"Duplicate mutation detected (hash={diff_hash[:12]})")
                results.append(MutationResult(req, success=False, preflight_checked=True, diff_hash=diff_hash, ast_matches_parent=ast_matches_parent, duplicate=True))
                continue

            attempted_hashes.add(diff_hash)
            prompt_hashes.add(_prompt_hash(req.rationale))

            # Store result
            results.append(MutationResult(req, success=preflight_ok, preflight_checked=True, diff_hash=diff_hash, ast_matches_parent=ast_matches_parent, duplicate=False))

        # Track valid-mutation rate
        if results:
            self._valid_count = max(self._valid_count, sum(1 for r in results if r.success))
            self._call_count = max(self._call_count, len(results))
            self.valid_mutation_rate = self._valid_count / self._call_count

        # Sort by success likelihood (preflight_ok first, then duplicate=False first)
        results.sort(key=lambda r: (not r.success, r.duplicate))
        return results

    def _build_mutation_prompt(self, context: MutationContext) -> str:
        """Build prompt for mutation generation based on context."""
        prompt_lines = [
            "# SEDS Mutator Task",
            "",
            "You are a code-space mutator for an agent self-improvement system.",
            "",
            "Operator Menu:",
            "- prompt_edit: Modify system prompt, task description, or few-shot examples",
            "- tool_edit: Adjust tool invocation schema, parameters, or add/remove tools",
            "- memory_edit: Change memory structure (RAG, vector stores, cache keys)",
            "- orchestration_edit: Rewrite multi-agent coordination (message routing, DAG)",
            "- efficiency_edit: Optimize code for speed, reduce latency, shrink context",
            "",
        ]

        # Context: failure mode histogram (PRIMARY SIGNAL)
        if context.failure_mode_histogram:
            prompt_lines.append("FAILURE PATTERN (Highest Priority):")
            sorted_modes = sorted(context.failure_mode_histogram.items(), key=lambda x: x[1], reverse=True)
            for mode, count in sorted_modes[:5]:
                prompt_lines.append(f"  - {mode}: {count} occurrences")
            prompt_lines.append("")

        # Context: parent code snippet
        prompt_lines.append("PARENT CODE SNIPPET:")
        prompt_lines.append("```python")
        prompt_lines.append(context.parent_code[:2000])
        prompt_lines.append("```")
        prompt_lines.append("")

        # Context: remaining budget
        prompt_lines.append(f"REMAINING EVALUATION BUDGET: {context.remaining_budget:.2f}")
        prompt_lines.append("")
        prompt_lines.append("INSTRUCTION:")
        prompt_lines.append("Propose ONE mutation that addresses the failure pattern above.")
        prompt_lines.append("Use the typed operator menu. Be concise and precise.")

        return "\n".join(prompt_lines)

    def _request_mutation(self, prompt: str, attempt_num: int, total_attempts: int) -> MutationRequest:
        """Request a single mutation from LLM using structured outputs."""
        messages = [
            {"role": "system", "content": "You are a code-space mutator. Output STRICT JSON following the schema."},
            {"role": "user", "content": prompt},
        ]

        # Use deterministic tier for structured outputs
        result = self.broker.handle_call(
            msg={
                "node_id": "synthesizer",
                "task_id": f"mutation_{attempt_num}",
                "tier": self.deterministic_tier,
                "messages": messages,
                "max_tokens": 500,
            }
        )

        if not result.get("ok"):
            # Fallback to text parsing if structured output fails
            logger.warning(f"Structured output failed, falling back to text parsing")
            content = result.get("response", {}).get("content", "")
            return self._parse_text_mutation(content)

        try:
            response = result["response"]["content"]
            data = json.loads(response)
            return MutationRequest(**data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse mutation JSON: {e}, content={response[:200]}")
            # Fallback: try to extract JSON from text
            content = result.get("response", {}).get("content", "")
            return self._parse_text_mutation(content)

    def _parse_text_mutation(self, text: str) -> MutationRequest:
        """Parse raw text to extract mutation request."""
        # Try to find JSON block
        match = re.search(r'\{[\s\S]*"operator"[\s\S]*\}', text)
        if match:
            data = json.loads(match.group())
            return MutationRequest(**data)

        # Fallback: extract operator and rationale
        operator_match = re.search(r'operator["\s:]+([a-z_]+)', text, re.IGNORECASE)
        operator = operator_match.group(1) if operator_match else "efficiency_edit"

        rationale_match = re.search(r'rationale["\s:]+(.*)$', text, re.IGNORECASE)
        rationale = rationale_match.group(1).strip() if rationale_match else "Mutation based on failure analysis."

        # Default values
        target_file = "agent.py"
        old_str = ""
        new_str = ""

        # Extract old_str (marked by ### OLD ###)
        old_match = re.search(r'###\s*OLD\s*###[\s\S]*?(?=###\s*NEW\s*###|$)', text, re.IGNORECASE)
        if old_match:
            old_str = old_match.group(0).strip()

        # Extract new_str (marked by ### NEW ###)
        new_match = re.search(r'###\s*NEW\s*###[\s\S]*?(?=###|$)', text, re.IGNORECASE)
        if new_match:
            new_str = new_match.group(0).strip()

        return MutationRequest(operator=operator, target_file=target_file, old_str=old_str, new_str=new_str, rationale=rationale)

    def get_valid_mutation_rate(self) -> float:
        """Return the historical valid-mutation rate."""
        return self.valid_mutation_rate

    def reset_stats(self):
        """Reset tracking statistics."""
        self._valid_count = 0
        self._call_count = 0
        self.valid_mutation_rate = 0.0


# ──── UTILITY: Load Parent Code ───────────────────────────────────────────────
def load_parent_code(repo_path: str, target_file: str) -> str:
    """Load the content of a target file from the parent agent repository."""
    file_path = Path(repo_path) / target_file
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return ""
    return file_path.read_text(encoding="utf-8", errors="ignore")


# ──── UTILITY: Create Mutation Report ─────────────────────────────────────────
def create_mutation_report(results: List[MutationResult]) -> str:
    """Create a human-readable report of mutation attempts."""
    lines = [f"Mutation Attempts: {len(results)}"]
    lines.append(f"Valid Mutations: {sum(1 for r in results if r.success)}")
    lines.append(f"Valid-Mutation Rate: {0:.2f}" if not results else f"Valid-Mutation Rate: {0:.2f}")

    for i, r in enumerate(results[:5]):  # Show top 5
        status = "✓" if r.success else "✗"
        dup = " [DUPLICATE]" if r.duplicate else ""
        lines.append(f"{status} Attempt {i+1}: {r.request.operator} -> {r.request.target_file}")
        lines.append(f"  Rationale: {r.request.rationale[:100]}...")
        if r.preflight_checked:
            lines.append(f"  Preflight: {'PASS' if r.success else 'FAIL'}{dup}")

    return "\n".join(lines)
