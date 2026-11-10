"""SQLite storage: metadata tables plus an append-only event log.

The event log (and the approvals table) are protected by triggers that reject UPDATE and DELETE,
so nothing - agent, human or toolkit bug - can rewrite history through SQL.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .util import canonical_json, json_hash, now

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  ts REAL NOT NULL,
  actor TEXT NOT NULL,          -- agent | human | system
  kind TEXT NOT NULL,           -- tool | forward | edit | eval | approval | system
  tool TEXT,
  args_json TEXT,
  result_summary TEXT,
  result_hash TEXT,
  ok INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS events_session ON events(session_id, id);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
  BEGIN SELECT RAISE(ABORT, 'events log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
  BEGIN SELECT RAISE(ABORT, 'events log is append-only'); END;

CREATE TABLE IF NOT EXISTS workspaces (
  id TEXT PRIMARY KEY,
  label TEXT,
  source TEXT NOT NULL,
  kind TEXT NOT NULL,
  base_hash TEXT NOT NULL,
  base_path TEXT NOT NULL,
  stack_json TEXT NOT NULL DEFAULT '[]',
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  actor TEXT NOT NULL,
  label TEXT,
  agent_model TEXT,
  access_level TEXT NOT NULL,
  budget_total INTEGER NOT NULL,
  budget_used INTEGER NOT NULL DEFAULT 0,
  policy_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL,
  closed_at REAL
);

CREATE TABLE IF NOT EXISTS patches (
  id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  base_hash TEXT NOT NULL,
  parent_id TEXT,
  status TEXT NOT NULL,         -- draft | tested | approved | rejected | superseded
  manifest_json TEXT NOT NULL,
  delta_path TEXT NOT NULL,
  flagged_json TEXT NOT NULL DEFAULT '[]',
  author TEXT,
  session_id TEXT,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patch_id TEXT NOT NULL,
  decision TEXT NOT NULL,       -- approved | rejected
  reviewer TEXT NOT NULL,
  note TEXT,
  acknowledged_flags INTEGER NOT NULL DEFAULT 0,
  ts REAL NOT NULL
);
CREATE TRIGGER IF NOT EXISTS approvals_no_update BEFORE UPDATE ON approvals
  BEGIN SELECT RAISE(ABORT, 'approvals are append-only'); END;
CREATE TRIGGER IF NOT EXISTS approvals_no_delete BEFORE DELETE ON approvals
  BEGIN SELECT RAISE(ABORT, 'approvals are append-only'); END;

CREATE TABLE IF NOT EXISTS results (
  id TEXT PRIMARY KEY,          -- content hash; used as a pageable handle
  created_at REAL NOT NULL,
  content_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS directions (
  id TEXT PRIMARY KEY,
  workspace_id TEXT,
  dim INTEGER NOT NULL,
  meta_json TEXT NOT NULL,
  vector_json TEXT NOT NULL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  session_id TEXT,
  patch_id TEXT,                -- NULL = base model
  model_hash TEXT NOT NULL,
  suite_id TEXT NOT NULL,
  split TEXT NOT NULL,          -- practice | heldout | regression
  kind TEXT NOT NULL,           -- practice | heldout | regression | baseline | test
  settings_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  details_json TEXT,            -- per-example results; never returned for held-out runs
  env_json TEXT NOT NULL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS registry (
  id TEXT PRIMARY KEY,
  model_hash TEXT NOT NULL,     -- '*' = any model
  kind TEXT NOT NULL,           -- tensor | direction | feature
  target TEXT NOT NULL,
  reason TEXT,
  created_by TEXT,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS zoo_nets (
  id TEXT PRIMARY KEY,
  split TEXT NOT NULL,
  family TEXT NOT NULL,
  card_json TEXT NOT NULL,      -- public metadata (no answers)
  path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS zoo_submissions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  submitter TEXT NOT NULL,
  net_id TEXT NOT NULL,
  task_type TEXT NOT NULL,
  answer_json TEXT NOT NULL,
  score REAL NOT NULL,
  ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_dicts (
  id TEXT PRIMARY KEY,
  model_hash TEXT NOT NULL,
  hook TEXT NOT NULL,
  size INTEGER NOT NULL,
  parent_dict TEXT,
  meta_json TEXT NOT NULL,
  path TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS features (
  id TEXT PRIMARY KEY,
  dict_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  label TEXT,
  fidelity REAL,
  sensitive INTEGER NOT NULL DEFAULT 0,
  record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS features_dict ON features(dict_id, idx);

CREATE TABLE IF NOT EXISTS challenge_runs (
  id TEXT PRIMARY KEY,
  challenge_id TEXT NOT NULL,
  agent_model TEXT NOT NULL,
  harness TEXT NOT NULL,         -- wsp-agent-runner | mcp (self-reported) | ...
  locked INTEGER NOT NULL,       -- 1 if the server was locked to this challenge's sources
  label TEXT,
  sessions_json TEXT NOT NULL,
  started_at REAL NOT NULL,
  submitted_at REAL,
  answer_json TEXT,
  score REAL,
  detail_json TEXT,
  usage_json TEXT
);

CREATE TABLE IF NOT EXISTS audits (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  base_workspace TEXT NOT NULL,
  tuned_workspace TEXT NOT NULL,
  report_json TEXT NOT NULL,
  markdown TEXT NOT NULL,
  created_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript(SCHEMA)
        for table, col, typ in [("challenge_runs", "seed", "INTEGER"), ("challenge_runs", "instance_hash", "TEXT")]:
            try:  # migrations for databases created by older versions
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass

    # ------------------------------------------------------------ basic access
    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    def one(self, sql: str, params: tuple | list = ()) -> dict | None:
        with self._lock:
            row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, params: tuple | list = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def transaction(self):
        return _Txn(self)

    # ------------------------------------------------------------ event log
    def log_event(
        self,
        *,
        session_id: str | None,
        actor: str,
        kind: str,
        tool: str | None,
        args: Any = None,
        result_summary: str | None = None,
        result: Any = None,
        ok: bool = True,
    ) -> int:
        rh = json_hash(result) if result is not None else None
        cur = self.execute(
            "INSERT INTO events(session_id, ts, actor, kind, tool, args_json, result_summary, result_hash, ok)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (session_id, now(), actor, kind, tool, canonical_json(args) if args is not None else None,
             (result_summary or "")[:2000], rh, 1 if ok else 0),
        )
        return int(cur.lastrowid)

    def events(self, session_id: str | None = None, after_id: int = 0, limit: int = 200) -> list[dict]:
        if session_id:
            return self.all("SELECT * FROM events WHERE session_id=? AND id>? ORDER BY id LIMIT ?",
                            (session_id, after_id, limit))
        return self.all("SELECT * FROM events WHERE id>? ORDER BY id LIMIT ?", (after_id, limit))

    # ------------------------------------------------------------ result handles
    def put_result(self, content: Any) -> str:
        cj = canonical_json(content)
        rid = "res_" + json_hash(content)[:20]
        self.execute("INSERT OR IGNORE INTO results(id, created_at, content_json) VALUES (?,?,?)",
                     (rid, now(), cj))
        return rid

    def get_result(self, rid: str) -> Any | None:
        row = self.one("SELECT content_json FROM results WHERE id=?", (rid,))
        return json.loads(row["content_json"]) if row else None


class _Txn:
    def __init__(self, store: Store):
        self.store = store

    def __enter__(self):
        self.store._lock.acquire()
        self.store.conn.execute("BEGIN IMMEDIATE")
        return self.store.conn

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.store.conn.execute("COMMIT")
            else:
                self.store.conn.execute("ROLLBACK")
        finally:
            self.store._lock.release()
        return False
