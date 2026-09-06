"""Runtime LLM shim (§6, Phase A item 6).

Provides seds.llm.call(tier, messages, ...) over the Unix socket to the broker.
Generated agent code must use this shim — never import openai directly.
"""
from __future__ import annotations

import json
import socket
import os
import time
from typing import Any

SOCKET_PATH = "/tmp/seds/run/llm.sock"


def call(tier: str, messages: list[dict], tools: list[dict] | None = None,
         seed: int = 0, max_tokens: int = 2000,
         socket_path: str = SOCKET_PATH) -> dict:
    """Send an LLM request to the broker over the Unix socket.

    Args:
        tier: "deterministic" or "reasoner"
        messages: list of message dicts (role, content)
        tools: optional list of tool definitions
        seed: seed for deterministic tier
        max_tokens: max tokens to generate
        socket_path: path to the broker Unix socket

    Returns:
        Broker response dict with ok, cached, resolved_model, response,
        input_tokens, output_tokens, reasoning_tokens, cached_tokens, cost_usd
    """
    msg = {
        "node_id": "sandbox",
        "task_id": "sandbox",
        "tier": tier,
        "messages": messages,
        "tools": tools or [],
        "seed": seed,
        "max_tokens": max_tokens,
    }

    if not os.path.exists(socket_path):
        raise ConnectionError(f"Broker socket not found at {socket_path}")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)

    try:
        sock.sendall((json.dumps(msg) + "\n").encode())

        # Receive response
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in chunk:
                break

        result = json.loads(data.decode().strip())
        return result
    finally:
        sock.close()


# Convenience wrappers
def deterministic(messages: list[dict], tools: list[dict] | None = None,
                  seed: int = 0, max_tokens: int = 2000) -> dict:
    """Call the deterministic tier (glm-4-7-flash)."""
    return call("deterministic", messages, tools, seed=seed, max_tokens=max_tokens)


def reasoner(messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = 2000) -> dict:
    """Call the reasoner tier (gpt-5-nano)."""
    return call("reasoner", messages, tools, max_tokens=max_tokens)
