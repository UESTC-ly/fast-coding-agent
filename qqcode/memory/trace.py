"""Run trace store: persistent log of routing decisions and task outcomes.

Every call to run_task() can emit a TraceRecord. Traces accumulate in
.qqcode/trace.db (SQLite) inside the repository. This file is gitignored.

Schema is intentionally flat — one wide row per run. This makes ad-hoc
calibration queries readable without a custom query layer.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRACE_DB_FILENAME = ".qqcode/trace.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    task_hash TEXT NOT NULL,
    task_snippet TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    route_layer TEXT NOT NULL DEFAULT '',
    route_decision TEXT NOT NULL,
    l0_triggered INTEGER NOT NULL DEFAULT 0,
    l0_reason TEXT NOT NULL DEFAULT '',
    l1_decision TEXT NOT NULL DEFAULT '',
    l1_confidence REAL NOT NULL DEFAULT 0.0,
    l2_override INTEGER NOT NULL DEFAULT 0,
    l2_reason TEXT NOT NULL DEFAULT '',
    files_hint_count INTEGER NOT NULL DEFAULT 0,
    task_length INTEGER NOT NULL,
    fastpath_attempted INTEGER NOT NULL DEFAULT 0,
    fastpath_success INTEGER NOT NULL DEFAULT 0,
    fastpath_reason TEXT NOT NULL DEFAULT '',
    final_success INTEGER NOT NULL,
    mode_used TEXT NOT NULL,
    finish_reason TEXT NOT NULL,
    tokens_routing INTEGER NOT NULL DEFAULT 0,
    tokens_fastpath INTEGER NOT NULL DEFAULT 0,
    tokens_fullagent INTEGER NOT NULL DEFAULT 0,
    tokens_total INTEGER NOT NULL DEFAULT 0,
    skills_used_json TEXT NOT NULL DEFAULT '[]',
    turns_used INTEGER NOT NULL DEFAULT 0,
    finish_summary TEXT NOT NULL DEFAULT ''
)
"""

_INSERT = """
INSERT INTO traces (
    id, task_hash, task_snippet, timestamp, duration_ms,
    route_layer, route_decision, l0_triggered, l0_reason,
    l1_decision, l1_confidence, l2_override, l2_reason,
    files_hint_count, task_length,
    fastpath_attempted, fastpath_success, fastpath_reason,
    final_success, mode_used, finish_reason,
    tokens_routing, tokens_fastpath, tokens_fullagent, tokens_total,
    skills_used_json, turns_used, finish_summary
) VALUES (
    :id, :task_hash, :task_snippet, :timestamp, :duration_ms,
    :route_layer, :route_decision, :l0_triggered, :l0_reason,
    :l1_decision, :l1_confidence, :l2_override, :l2_reason,
    :files_hint_count, :task_length,
    :fastpath_attempted, :fastpath_success, :fastpath_reason,
    :final_success, :mode_used, :finish_reason,
    :tokens_routing, :tokens_fastpath, :tokens_fullagent, :tokens_total,
    :skills_used_json, :turns_used, :finish_summary
)
"""


@dataclass
class TraceRecord:
    """One recorded task run.

    Created by the orchestrator after each run_task() call and written to the
    trace store. All fields have defaults so callers can build records
    incrementally.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_hash: str = ""
    task_snippet: str = ""
    timestamp: str = ""
    duration_ms: int = 0

    # Routing metadata
    route_layer: str = ""        # "l0" | "l1_l2" | "fallback" | "mode_forced"
    route_decision: str = ""     # "fastpath" | "fullagent"
    l0_triggered: bool = False
    l0_reason: str = ""
    l1_decision: str = ""        # Raw L1 verdict; empty when L0 fired
    l1_confidence: float = 0.0   # Raw L1 confidence; 0.0 when L0 fired
    l2_override: bool = False
    l2_reason: str = ""
    files_hint_count: int = 0
    task_length: int = 0

    # FastPath outcome (only meaningful when route_decision == "fastpath")
    fastpath_attempted: bool = False
    fastpath_success: bool = False
    fastpath_reason: str = ""    # "ok" or the escalation_reason

    # Final outcome
    final_success: bool = False
    mode_used: str = ""          # "fastpath" | "fullagent"
    finish_reason: str = ""
    turns_used: int = 0          # Full Agent: N turns; FastPath: 0 (single-shot)
    finish_summary: str = ""     # Error text or agent's closing summary; "" when neither

    # Token costs
    tokens_routing: int = 0
    tokens_fastpath: int = 0
    tokens_fullagent: int = 0
    tokens_total: int = 0

    # Skills (list of skill names active during this run)
    skills_used: list[str] = field(default_factory=list)

    @classmethod
    def from_task(cls, task: str) -> TraceRecord:
        """Start a record for a task; fill outcome fields later."""
        return cls(
            task_hash=hashlib.sha256(task[:500].encode()).hexdigest()[:16],
            task_snippet=task[:200],
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            task_length=len(task),
        )

    def _as_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_hash": self.task_hash,
            "task_snippet": self.task_snippet,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "route_layer": self.route_layer,
            "route_decision": self.route_decision,
            "l0_triggered": int(self.l0_triggered),
            "l0_reason": self.l0_reason,
            "l1_decision": self.l1_decision,
            "l1_confidence": self.l1_confidence,
            "l2_override": int(self.l2_override),
            "l2_reason": self.l2_reason,
            "files_hint_count": self.files_hint_count,
            "task_length": self.task_length,
            "fastpath_attempted": int(self.fastpath_attempted),
            "fastpath_success": int(self.fastpath_success),
            "fastpath_reason": self.fastpath_reason,
            "final_success": int(self.final_success),
            "mode_used": self.mode_used,
            "finish_reason": self.finish_reason,
            "tokens_routing": self.tokens_routing,
            "tokens_fastpath": self.tokens_fastpath,
            "tokens_fullagent": self.tokens_fullagent,
            "tokens_total": self.tokens_total,
            "skills_used_json": json.dumps(self.skills_used),
            "turns_used": self.turns_used,
            "finish_summary": self.finish_summary,
        }

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> TraceRecord:
        d = dict(row)
        return cls(
            id=d["id"],
            task_hash=d["task_hash"],
            task_snippet=d["task_snippet"],
            timestamp=d["timestamp"],
            duration_ms=d["duration_ms"],
            route_layer=d["route_layer"],
            route_decision=d["route_decision"],
            l0_triggered=bool(d["l0_triggered"]),
            l0_reason=d["l0_reason"],
            l1_decision=d["l1_decision"],
            l1_confidence=float(d["l1_confidence"]),
            l2_override=bool(d["l2_override"]),
            l2_reason=d["l2_reason"],
            files_hint_count=d["files_hint_count"],
            task_length=d["task_length"],
            fastpath_attempted=bool(d["fastpath_attempted"]),
            fastpath_success=bool(d["fastpath_success"]),
            fastpath_reason=d["fastpath_reason"],
            final_success=bool(d["final_success"]),
            mode_used=d["mode_used"],
            finish_reason=d["finish_reason"],
            tokens_routing=d["tokens_routing"],
            tokens_fastpath=d["tokens_fastpath"],
            tokens_fullagent=d["tokens_fullagent"],
            tokens_total=d["tokens_total"],
            skills_used=json.loads(d["skills_used_json"]),
            turns_used=d.get("turns_used", 0),  # .get for forward compat with old DBs
            finish_summary=d.get("finish_summary", ""),
        )


class TraceStore:
    """SQLite-backed store for run traces.

    The database lives at <repo_root>/.qqcode/trace.db (gitignored). The
    parent directory is created on first write if it does not exist.
    """

    def __init__(self, db_path: Path):
        self._path = db_path
        self._conn: sqlite3.Connection | None = None

    @classmethod
    def for_repo(cls, repo: Path) -> TraceStore:
        """Open (or create) the trace store for a repository."""
        return cls(repo / TRACE_DB_FILENAME)

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE)
            # Migration: add columns to existing databases that predate them
            cols = {row[1] for row in conn.execute("PRAGMA table_info(traces)")}
            if "turns_used" not in cols:
                conn.execute("ALTER TABLE traces ADD COLUMN turns_used INTEGER NOT NULL DEFAULT 0")
            if "finish_summary" not in cols:
                conn.execute(
                    "ALTER TABLE traces ADD COLUMN finish_summary TEXT NOT NULL DEFAULT ''"
                )
            conn.commit()
            self._conn = conn
        return self._conn

    def write(self, record: TraceRecord) -> None:
        """Persist one trace record. Duplicate ids are silently ignored."""
        conn = self._connect()
        try:
            conn.execute(_INSERT, record._as_row())  # noqa: SLF001
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # Duplicate id — idempotent

    def all(self) -> list[TraceRecord]:
        """All traces ordered by timestamp ascending."""
        conn = self._connect()
        rows = conn.execute("SELECT * FROM traces ORDER BY timestamp ASC").fetchall()
        return [TraceRecord._from_row(r) for r in rows]  # noqa: SLF001

    def count(self) -> int:
        """Number of traces stored."""
        conn = self._connect()
        return int(conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0])

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> TraceStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
