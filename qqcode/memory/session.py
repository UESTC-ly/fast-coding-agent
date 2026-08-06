"""Session store: durable record of conversations, one row per session.

Kept separate from `trace.db` on purpose. A trace is one flat wide row per
*run*, scanned wholesale by ReplayEngine for threshold calibration. A session is
one row per *conversation*, holding an ordered turn log, and is looked up by id
or recency. Same directory, different lifetime and access pattern; sharing a
file would force one schema to serve both.

Turns are stored as JSON rather than a child table. The turn log is only ever
read and written as a whole — there is no query that asks about turns across
sessions — so a join buys nothing and costs a migration.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SESSION_DB_FILENAME = ".qqcode/sessions.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    base_commit TEXT NOT NULL DEFAULT '',
    turns_json TEXT NOT NULL DEFAULT '[]'
)
"""

_UPSERT = """
INSERT INTO sessions (id, repo, created_at, updated_at, base_commit, turns_json)
VALUES (:id, :repo, :created_at, :updated_at, :base_commit, :turns_json)
ON CONFLICT(id) DO UPDATE SET
    updated_at = excluded.updated_at,
    base_commit = excluded.base_commit,
    turns_json = excluded.turns_json
"""


def _now() -> str:
    """UTC timestamp, matching trace.py's format."""
    return datetime.datetime.now(datetime.UTC).isoformat()


@dataclass
class TurnRecord:
    """One completed exchange within a session."""

    task: str
    outcome: str          # "accepted" | "rejected" | "failed" | "interrupted"
    mode_used: str = ""
    changed_files: tuple[str, ...] = ()
    tokens: int = 0

    def _as_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["changed_files"] = list(self.changed_files)
        return d

    @classmethod
    def _from_json(cls, d: dict[str, Any]) -> TurnRecord:
        return cls(
            task=d["task"],
            outcome=d["outcome"],
            mode_used=d.get("mode_used", ""),
            changed_files=tuple(d.get("changed_files", ())),
            tokens=d.get("tokens", 0),
        )


@dataclass
class SessionRecord:
    """A conversation: its repository, its turns, and where it started."""

    repo: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    # HEAD at session start. This is the only viable undo anchor for a turn that
    # has already been finalized, which is why the REPL records it.
    base_commit: str = ""
    turns: list[TurnRecord] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return sum(1 for t in self.turns if t.outcome == "accepted")

    @property
    def total_tokens(self) -> int:
        return sum(t.tokens for t in self.turns)

    @property
    def short_id(self) -> str:
        """First segment of the uuid — enough to disambiguate when resuming."""
        return self.id.split("-")[0]

    def _as_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo": self.repo,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "base_commit": self.base_commit,
            "turns_json": json.dumps([t._as_json() for t in self.turns]),  # noqa: SLF001
        }

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> SessionRecord:
        d = dict(row)
        return cls(
            id=d["id"],
            repo=d["repo"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            base_commit=d["base_commit"],
            turns=[TurnRecord._from_json(t) for t in json.loads(d["turns_json"])],  # noqa: SLF001
        )


class SessionStore:
    """SQLite-backed store for conversation sessions.

    Lives at <repo_root>/.qqcode/sessions.db, alongside trace.db and covered by
    the same gitignore rule.
    """

    def __init__(self, db_path: Path):
        self._path = db_path
        self._conn: sqlite3.Connection | None = None

    @classmethod
    def for_repo(cls, repo: Path) -> SessionStore:
        """Open (or create) the session store for a repository."""
        return cls(repo / SESSION_DB_FILENAME)

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE)
            conn.commit()
            self._conn = conn
        return self._conn

    def save(self, record: SessionRecord) -> None:
        """Persist a session, overwriting any prior state for the same id.

        Called after every turn rather than at session end: a crash or a kill -9
        mid-session should not lose the turns that already completed.
        """
        record.updated_at = _now()
        conn = self._connect()
        conn.execute(_UPSERT, record._as_row())  # noqa: SLF001
        conn.commit()

    def load(self, session_id: str) -> SessionRecord | None:
        """Fetch by full id, or by the short prefix shown in listings.

        Returns None when nothing matches. A prefix matching several sessions
        raises rather than picking one arbitrarily — resuming the wrong
        conversation is worse than being asked to be specific.

        Raises:
            ValueError: The prefix is ambiguous.
        """
        conn = self._connect()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is not None:
            return SessionRecord._from_row(row)  # noqa: SLF001

        rows = conn.execute(
            "SELECT * FROM sessions WHERE id LIKE ? ORDER BY updated_at DESC",
            (f"{session_id}%",),
        ).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            ids = ", ".join(r["id"].split("-")[0] for r in rows)
            raise ValueError(f"session id {session_id!r} is ambiguous; matches: {ids}")
        return SessionRecord._from_row(rows[0])  # noqa: SLF001

    def latest(self, repo: Path | None = None) -> SessionRecord | None:
        """Most recently updated session, optionally scoped to one repository."""
        conn = self._connect()
        if repo is None:
            row = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM sessions WHERE repo = ? ORDER BY updated_at DESC LIMIT 1",
                (str(repo.resolve()),),
            ).fetchone()
        return SessionRecord._from_row(row) if row is not None else None  # noqa: SLF001

    def recent(self, limit: int = 20) -> list[SessionRecord]:
        """Sessions ordered by most recently updated."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [SessionRecord._from_row(r) for r in rows]  # noqa: SLF001

    def count(self) -> int:
        conn = self._connect()
        return int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
