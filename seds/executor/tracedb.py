"""Trace DB schema for SEDS (§5, Phase A item 5).

Creates and initializes the SQLite database at data/trace_db.sqlite
with tables: nodes, rollouts, spans, evaluations, llm_calls, tool_calls.
Indexed on (node_id, task_id) and cache_key.
"""
from __future__ import annotations

import os
import sqlite3

DB_PATH = "data/trace_db.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id TEXT,
    created_at REAL NOT NULL,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS rollouts (
    rollout_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    answer TEXT,
    score_correct BOOLEAN,
    score_partial REAL,
    score_detail TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    reasoning_tokens INTEGER,
    cost_usd REAL,
    wall_ms INTEGER,
    crashed BOOLEAN,
    timed_out BOOLEAN,
    error_log TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY(node_id) REFERENCES nodes(node_id)
);

CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    rollout_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    inputs TEXT,
    outputs TEXT,
    parent_span_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL,
    FOREIGN KEY(rollout_id) REFERENCES rollouts(rollout_id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    eval_id TEXT PRIMARY KEY,
    rollout_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    score_correct BOOLEAN,
    score_partial REAL,
    score_detail TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY(rollout_id) REFERENCES rollouts(rollout_id)
);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id TEXT PRIMARY KEY,
    rollout_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    cache_key TEXT,
    resolved_model TEXT NOT NULL,
    tier TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    reasoning_tokens INTEGER,
    cached_tokens INTEGER,
    cost_usd REAL,
    cached BOOLEAN,
    created_at REAL NOT NULL,
    FOREIGN KEY(rollout_id) REFERENCES rollouts(rollout_id)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    tool_call_id TEXT PRIMARY KEY,
    rollout_id TEXT NOT NULL,
    span_id TEXT,
    tool_name TEXT NOT NULL,
    arguments TEXT,
    result TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY(rollout_id) REFERENCES rollouts(rollout_id)
);

-- Indexes per §5 and §6
CREATE INDEX IF NOT EXISTS idx_rollouts_node_task ON rollouts(node_id, task_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_cache ON llm_calls(cache_key);
"""


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Initialize the trace DB. Returns the connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Get a connection to the trace DB, initializing if needed."""
    if not os.path.exists(db_path):
        return init_db(db_path)
    return sqlite3.connect(db_path)
