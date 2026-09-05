"""Preflight AST check (§6, Phase A item 6).

Rejects agent code that directly imports openai, requests, or httpx.
This ensures cost accounting is unforgeable — all LLM calls must go through
the seds.llm shim which talks to the broker.
"""
from __future__ import annotations

import ast
import sys
from typing import List, Tuple

FORBIDDEN_IMPORTS = {"openai", "requests", "httpx"}
FORBIDDEN_FROM_IMPORTS = {"openai", "requests", "httpx"}


def check_agent_code(source: str) -> Tuple[bool, List[str]]:
    """Parse agent source and check for forbidden direct SDK imports.

    Returns:
        (passes, errors) — passes=True if no forbidden imports found.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]

    errors: List[str] = []

    for node in ast.walk(tree):
        # Check import statements: import openai, import requests, import httpx
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    errors.append(
                        f"Direct import of '{alias.name}' is forbidden. "
                        f"Use seds.llm.call() instead."
                    )

        # Check from imports: from openai import ..., from requests import ...
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in FORBIDDEN_FROM_IMPORTS:
                errors.append(
                    f"From-import of '{node.module}' is forbidden. "
                    f"Use seds.llm.call() instead."
                )

    return len(errors) == 0, errors


def preflight_check(filepath: str) -> Tuple[bool, List[str]]:
    """Read a Python file and run the preflight AST check."""
    with open(filepath) as f:
        source = f.read()
    return check_agent_code(source)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m seds.executor.preflight <agent_file.py>")
        sys.exit(1)

    passes, errors = preflight_check(sys.argv[1])
    if passes:
        print("PASSED: No forbidden imports found")
        sys.exit(0)
    else:
        print("FAILED: Forbidden imports detected:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
