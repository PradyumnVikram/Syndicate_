"""Broker server implementation — the keystone of SEDS (§4).

Unix socket server with tier routing, param injection, replay cache,
rate limiting, spend metering, and full call logging.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, fields
from typing import Any

# Load .env file for API credentials
try:
    from dotenv import load_dotenv
    _env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    load_dotenv(_env_path)
except ImportError:
    _env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    if os.path.exists(_env_path):
        with open(_env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")

from openai import OpenAI

logger = logging.getLogger("seds.broker")

# ── Tier → model routing ──────────────────────────────────────────────
# §2.2: deterministic → glm-4-7-flash, reasoner → gpt-5-nano

TIER_ROUTES: dict[str, dict[str, Any]] = {
    "deterministic": {
        "model": "glm-4-7-flash",
        "base_url": os.environ.get("TENSORMUX_BASE_URL", ""),
        "api_key": os.environ.get("TENSORMUX_API_KEY", ""),
        "temperature": 0,
        "seed": None,
    },
    "reasoner": {
        "model": "gpt-5-nano",
        "base_url": os.environ.get("OPENAI_BASE_URL", ""),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "reasoning_effort": "minimal",
    },
}

SOCKET_PATH = "/tmp/seds/run/llm.sock"
DB_PATH = "data/broker.db"


@dataclass
class CallLogEntry:
    call_id: str
    node_id: str
    task_id: str
    tier: str
    resolved_model: str
    cache_key: str
    cached: bool
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    cost_usd: float
    response: dict[str, Any]
    error: str | None
    timestamp: float


class SpendMeter:
    """Per-node spend tracking with hard kill-switch."""

    def __init__(self, budget_per_node: float = 10.0):
        self.budget_per_node = budget_per_node
        self._spent: dict[str, float] = {}
        self._lock = threading.Lock()

    def record(self, node_id: str, cost: float) -> bool:
        with self._lock:
            self._spent[node_id] = self._spent.get(node_id, 0.0) + cost
            if self._spent[node_id] > self.budget_per_node:
                logger.warning(f"Node {node_id} exceeded budget: "
                               f"${self._spent[node_id]:.4f} > ${self.budget_per_node:.2f}")
                return False
            return True

    def remaining(self, node_id: str) -> float:
        with self._lock:
            return max(0.0, self.budget_per_node - self._spent.get(node_id, 0.0))

    def reset(self, node_id: str):
        with self._lock:
            self._spent[node_id] = 0.0


class RateLimiter:
    """Token-bucket rate limiter with exponential backoff + jitter."""

    def __init__(self, tokens_per_second: float = 10.0, burst_size: int = 20):
        self.tokens_per_second = tokens_per_second
        self.burst_size = burst_size
        self._tokens = burst_size
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.burst_size, self._tokens + elapsed * self.tokens_per_second)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def wait(self):
        while not self.acquire():
            delay = min(2.0, (1.0 - self._tokens / self.burst_size) * 0.5)
            jitter = delay * 0.3 * (2 * (time.time() % 1) - 1)
            delay = max(0.01, delay + jitter)
            time.sleep(delay)


class ReplayCache:
    """SHA-256 keyed replay cache for LLM responses."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                resolved_model TEXT NOT NULL,
                response TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                reasoning_tokens INTEGER NOT NULL,
                cached_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                stored_at REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _compute_key(self, resolved_model: str, messages: list, tools: list,
                     params: dict, seed: int) -> str:
        payload = json.dumps({
            "resolved_model": resolved_model,
            "messages": messages,
            "tools": tools,
            "params": params,
            "seed": seed,
        }, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, resolved_model: str, messages: list, tools: list,
            params: dict, seed: int) -> dict | None:
        key = self._compute_key(resolved_model, messages, tools, params, seed)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT response, input_tokens, output_tokens, reasoning_tokens, "
            "cached_tokens, cost_usd FROM cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "cache_key": key,
            "cached": True,
            "response": json.loads(row[0]),
            "input_tokens": row[1],
            "output_tokens": row[2],
            "reasoning_tokens": row[3],
            "cached_tokens": row[4],
            "cost_usd": row[5],
        }

    def store(self, resolved_model: str, messages: list, tools: list,
              params: dict, seed: int, response: dict, input_tokens: int,
              output_tokens: int, reasoning_tokens: int, cached_tokens: int,
              cost_usd: float):
        key = self._compute_key(resolved_model, messages, tools, params, seed)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO cache (cache_key, resolved_model, response, "
            "input_tokens, output_tokens, reasoning_tokens, cached_tokens, "
            "cost_usd, stored_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, resolved_model, json.dumps(response), input_tokens,
             output_tokens, reasoning_tokens, cached_tokens, cost_usd, time.time()),
        )
        conn.commit()
        conn.close()


def _dataclass_to_dict(obj) -> dict:
    if hasattr(obj, '__dataclass_fields__'):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    return obj.__dict__ if hasattr(obj, '__dict__') else {}


class Broker:
    """Unix socket server that routes LLM calls through a single credential-owning process.

    Features:
    - Tier→model routing with param injection
    - Replay cache (sha256 keyed)
    - Rate limiting with backoff + jitter
    - Cache-keyed dedup
    - Spend meter with per-node kill-switch
    - Full call log (SQLite)
    - --replay-only mode (serve exclusively from cache, error on miss)
    """

    def __init__(self, socket_path: str = SOCKET_PATH,
                 replay_only: bool = False,
                 budget_per_node: float = 10.0,
                 rate_limit_tps: float = 10.0):
        self.socket_path = socket_path
        self.replay_only = replay_only
        self.spend_meter = SpendMeter(budget_per_node=budget_per_node)
        self.rate_limiter = RateLimiter(tokens_per_second=rate_limit_tps)
        self.cache = ReplayCache(DB_PATH)
        self.call_log: list[CallLogEntry] = []
        self._lock = threading.Lock()
        self._running = False
        self._server_thread: threading.Thread | None = None
        self._init_call_log_db()

    def _init_call_log_db(self):
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect("data/broker.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS call_log (
                call_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                tier TEXT NOT NULL,
                resolved_model TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                cached BOOLEAN NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                reasoning_tokens INTEGER NOT NULL,
                cached_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                response TEXT NOT NULL,
                error TEXT,
                timestamp REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_call_log_node_task ON call_log(node_id, task_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_call_log_cache_key ON call_log(cache_key)")
        conn.commit()
        conn.close()

    def _resolve_params(self, tier: str, seed: int) -> tuple[str, dict]:
        route = TIER_ROUTES[tier]
        resolved_model = route["model"]
        params = {}

        if tier == "deterministic":
            params["temperature"] = 0
            params["seed"] = seed
        elif tier == "reasoner":
            params["reasoning_effort"] = "minimal"

        return resolved_model, params

    def _call_upstream(self, resolved_model: str, params: dict,
                       messages: list, tools: list) -> dict:
        route = next((r for t, r in TIER_ROUTES.items() if r["model"] == resolved_model), {})
        base_url = route.get("base_url", "")
        api_key = route.get("api_key", "")

        if not base_url or not api_key:
            raise ValueError(f"No credentials for model {resolved_model}")

        client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)

        kwargs = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": params.get("max_tokens", 2000),
        }

        if resolved_model == "glm-4-7-flash":
            kwargs["temperature"] = params["temperature"]
            kwargs["seed"] = params["seed"]
        elif resolved_model == "gpt-5-nano":
            kwargs["reasoning_effort"] = params["reasoning_effort"]
            kwargs["max_completion_tokens"] = params.get("max_tokens", 2000)

        if tools:
            kwargs["tools"] = tools  # tools are already in OpenAI format from the client

        response = client.chat.completions.create(**kwargs)
        usage = response.usage

        content = response.choices[0].message.content or ""
        reasoning = getattr(response.choices[0].message, "reasoning_content", "") or ""

        result = {
            "content": content,
            "reasoning": reasoning,
            "finish_reason": response.choices[0].finish_reason,
        }

        return {
            "response": result,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "reasoning_tokens": (
                usage.completion_tokens_details.reasoning_tokens
                if usage and hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details
                else 0
            ),
            "cached_tokens": (
                usage.completion_tokens_details.cached_tokens
                if usage and hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details
                else 0
            ),
        }

    def handle_call(self, msg: dict) -> dict:
        node_id = msg["node_id"]
        task_id = msg["task_id"]
        tier = msg["tier"]
        messages = msg["messages"]
        tools = msg.get("tools", [])
        seed = msg.get("seed", 0)
        max_tokens = msg.get("max_tokens", 2000)

        resolved_model, params = self._resolve_params(tier, seed)
        params["max_tokens"] = max_tokens

        cache_key = self.cache._compute_key(resolved_model, messages, tools, params, seed)

        # Check cache first
        cached = self.cache.get(resolved_model, messages, tools, params, seed)
        if cached:
            logger.info(f"Cache HIT for key={cache_key[:12]}...")
            entry = CallLogEntry(
                call_id=str(uuid.uuid4()),
                node_id=node_id, task_id=task_id, tier=tier,
                resolved_model=resolved_model, cache_key=cache_key,
                cached=True,
                input_tokens=cached["input_tokens"],
                output_tokens=cached["output_tokens"],
                reasoning_tokens=cached["reasoning_tokens"],
                cached_tokens=cached["cached_tokens"],
                cost_usd=0.0,
                response=cached["response"],
                error=None,
                timestamp=time.time(),
            )
            self._log_call(entry)
            result = {"ok": True, "cached": True, "cost_usd": 0.0}
            result.update(cached)
            result["cost_usd"] = 0.0
            return result

        # In replay-only mode, miss = error
        if self.replay_only:
            return {"ok": False, "error": "CACHE_MISS",
                    "detail": f"No cached response for key={cache_key[:12]}..."}

        # Rate limiting
        self.rate_limiter.wait()

        # Spend meter pre-check
        if not self.spend_meter.record(node_id, 0.0):
            return {"ok": False, "error": "BUDGET_EXCEEDED",
                    "detail": f"Node {node_id} budget exceeded"}

        # Make upstream call
        try:
            result = self._call_upstream(resolved_model, params, messages, tools)
        except Exception as e:
            logger.error(f"Upstream call failed: {e}")
            import traceback
            traceback.print_exc()
            return {"ok": False, "error": "UPSTREAM_ERROR", "detail": str(e)}

        # Compute cost
        cost_usd = self._compute_cost(resolved_model, result["input_tokens"],
                                       result["output_tokens"], result["reasoning_tokens"],
                                       result["cached_tokens"])

        # Spend meter check with actual cost
        if not self.spend_meter.record(node_id, cost_usd):
            return {"ok": False, "error": "BUDGET_EXCEEDED",
                    "detail": f"Node {node_id} budget exceeded after cost ${cost_usd:.4f}"}

        # Store in cache
        self.cache.store(
            resolved_model, messages, tools, params, seed,
            result["response"], result["input_tokens"],
            result["output_tokens"], result["reasoning_tokens"],
            result["cached_tokens"], cost_usd,
        )

        entry = CallLogEntry(
            call_id=str(uuid.uuid4()),
            node_id=node_id, task_id=task_id, tier=tier,
            resolved_model=resolved_model, cache_key=cache_key,
            cached=False,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            reasoning_tokens=result["reasoning_tokens"],
            cached_tokens=result["cached_tokens"],
            cost_usd=cost_usd,
            response=result["response"],
            error=None,
            timestamp=time.time(),
        )
        self._log_call(entry)

        return {
            "ok": True,
            "cached": False,
            "resolved_model": resolved_model,
            "cache_key": cache_key,
            "response": result["response"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "reasoning_tokens": result["reasoning_tokens"],
            "cached_tokens": result["cached_tokens"],
            "cost_usd": cost_usd,
        }

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int,
                      reasoning_tokens: int, cached_tokens: int) -> float:
        # Pricing per million tokens (in USD)
        if model == "glm-4-7-flash":
            input_price = 0.03  # $0.03 per 1M input tokens
            output_price = 0.03  # $0.03 per 1M output tokens
            return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        elif model == "gpt-5-nano":
            input_price = 0.10  # $0.10 per 1M input tokens
            output_price = 0.30  # $0.30 per 1M output tokens
            reasoning_price = 0.05  # $0.05 per 1M reasoning tokens
            return (input_tokens * input_price + output_tokens * output_price +
                    reasoning_tokens * reasoning_price) / 1_000_000
        return 0.001

    def _log_call(self, entry: CallLogEntry):
        with self._lock:
            self.call_log.append(entry)
            conn = sqlite3.connect("data/broker.db")
            conn.execute(
                "INSERT INTO call_log (call_id, node_id, task_id, tier, resolved_model, "
                "cache_key, cached, input_tokens, output_tokens, reasoning_tokens, "
                "cached_tokens, cost_usd, response, error, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry.call_id, entry.node_id, entry.task_id, entry.tier,
                 entry.resolved_model, entry.cache_key, entry.cached,
                 entry.input_tokens, entry.output_tokens, entry.reasoning_tokens,
                 entry.cached_tokens, entry.cost_usd, json.dumps(entry.response),
                 entry.error, entry.timestamp),
            )
            conn.commit()
            conn.close()

    def _handle_client(self, conn: socket.socket):
        try:
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in chunk:
                    break
            if not data:
                return

            msg = json.loads(data.decode().strip())

            if msg.get("cmd") == "shutdown":
                self.shutdown()
                conn.sendall(json.dumps({"ok": True}).encode())
                return

            result = self.handle_call(msg)
            conn.sendall((json.dumps(result) + "\n").encode())
        except Exception as e:
            logger.error(f"Client handler error: {e}")
            try:
                conn.sendall((json.dumps({"ok": False, "error": "INTERNAL_ERROR",
                                         "detail": str(e)}) + "\n").encode())
            except Exception:
                pass
        finally:
            conn.close()

    def start(self):
        self._running = True
        os.makedirs(os.path.dirname(self.socket_path) or ".", exist_ok=True)
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._server_thread.start()
        logger.info(f"Broker listening on {self.socket_path}")

    def _server_loop(self):
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        server.listen(10)
        while self._running:
            try:
                conn, _ = server.accept()
                thread = threading.Thread(target=self._handle_client,
                                          args=(conn,), daemon=True)
                thread.start()
            except Exception:
                break

    def shutdown(self):
        self._running = False
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        logger.info("Broker shut down")

    def get_call_log(self) -> list[dict]:
        return [_dataclass_to_dict(e) for e in self.call_log]
