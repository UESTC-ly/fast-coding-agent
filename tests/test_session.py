"""Session store tests: persistence, lookup, and resume semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from qqcode.memory.session import SessionRecord, SessionStore, TurnRecord


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions.db")


def _session(repo: str = "/r", turns: int = 0) -> SessionRecord:
    rec = SessionRecord(repo=repo)
    for i in range(turns):
        rec.turns.append(TurnRecord(task=f"task {i}", outcome="accepted", tokens=100))
    return rec


class TestRoundTrip:
    def test_save_then_load(self, store: SessionStore) -> None:
        rec = _session(turns=2)
        store.save(rec)

        loaded = store.load(rec.id)
        assert loaded is not None
        assert loaded.id == rec.id
        assert loaded.repo == rec.repo
        assert [t.task for t in loaded.turns] == ["task 0", "task 1"]

    def test_turn_fields_survive(self, store: SessionStore) -> None:
        rec = _session()
        rec.turns.append(
            TurnRecord(
                task="do it", outcome="rejected", mode_used="fullagent",
                changed_files=("a.py", "b/c.py"), tokens=4321,
            )
        )
        store.save(rec)

        loaded = store.load(rec.id)
        assert loaded is not None
        turn = loaded.turns[0]
        assert turn.task == "do it"
        assert turn.outcome == "rejected"
        assert turn.mode_used == "fullagent"
        assert turn.changed_files == ("a.py", "b/c.py")
        assert turn.tokens == 4321

    def test_base_commit_survives(self, store: SessionStore) -> None:
        rec = SessionRecord(repo="/r", base_commit="a" * 40)
        store.save(rec)
        loaded = store.load(rec.id)
        assert loaded is not None
        assert loaded.base_commit == "a" * 40

    def test_saving_twice_updates_rather_than_duplicates(self, store: SessionStore) -> None:
        rec = _session(turns=1)
        store.save(rec)
        rec.turns.append(TurnRecord(task="second", outcome="accepted"))
        store.save(rec)

        assert store.count() == 1
        loaded = store.load(rec.id)
        assert loaded is not None
        assert len(loaded.turns) == 2

    def test_unknown_id_returns_none(self, store: SessionStore) -> None:
        assert store.load("nope") is None


class TestLookup:
    def test_load_by_short_prefix(self, store: SessionStore) -> None:
        rec = _session()
        store.save(rec)
        loaded = store.load(rec.short_id)
        assert loaded is not None
        assert loaded.id == rec.id

    def test_ambiguous_prefix_raises(self, store: SessionStore) -> None:
        """Resuming the wrong conversation is worse than being asked to be specific."""
        a = SessionRecord(repo="/r", id="dup00000-1111-2222-3333-444444444444")
        b = SessionRecord(repo="/r", id="dup00000-9999-8888-7777-666666666666")
        store.save(a)
        store.save(b)

        with pytest.raises(ValueError, match="ambiguous"):
            store.load("dup00000")

    def test_latest_is_the_most_recently_updated(self, store: SessionStore) -> None:
        first, second = _session(), _session()
        store.save(first)
        store.save(second)
        store.save(first)  # touching `first` makes it newest

        loaded = store.latest()
        assert loaded is not None
        assert loaded.id == first.id

    def test_latest_scoped_to_repo(self, store: SessionStore, tmp_path: Path) -> None:
        here, elsewhere = tmp_path / "a", tmp_path / "b"
        here.mkdir()
        elsewhere.mkdir()
        mine = SessionRecord(repo=str(here.resolve()))
        theirs = SessionRecord(repo=str(elsewhere.resolve()))
        store.save(mine)
        store.save(theirs)  # newest overall, but a different repo

        loaded = store.latest(here)
        assert loaded is not None
        assert loaded.id == mine.id

    def test_latest_returns_none_when_empty(self, store: SessionStore) -> None:
        assert store.latest() is None

    def test_recent_is_newest_first(self, store: SessionStore) -> None:
        a, b, c = _session(), _session(), _session()
        for rec in (a, b, c):
            store.save(rec)

        assert [r.id for r in store.recent()] == [c.id, b.id, a.id]

    def test_recent_honors_limit(self, store: SessionStore) -> None:
        for _ in range(5):
            store.save(_session())
        assert len(store.recent(limit=2)) == 2


class TestAccounting:
    def test_counts_and_totals(self) -> None:
        rec = SessionRecord(repo="/r")
        rec.turns = [
            TurnRecord(task="a", outcome="accepted", tokens=100),
            TurnRecord(task="b", outcome="rejected", tokens=50),
            TurnRecord(task="c", outcome="accepted", tokens=25),
        ]
        assert rec.accepted_count == 2
        assert rec.total_tokens == 175

    def test_short_id_is_the_first_uuid_segment(self) -> None:
        rec = SessionRecord(repo="/r", id="abcd1234-5678-90ab-cdef-111111111111")
        assert rec.short_id == "abcd1234"


class TestDurability:
    def test_reopening_the_file_sees_prior_sessions(self, tmp_path: Path) -> None:
        """The point of persistence: a new process finds what the last one wrote."""
        path = tmp_path / "s.db"
        rec = _session(turns=3)
        with SessionStore(path) as first:
            first.save(rec)

        with SessionStore(path) as second:
            loaded = second.load(rec.id)
            assert loaded is not None
            assert len(loaded.turns) == 3

    def test_store_creates_its_parent_directory(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / ".qqcode" / "sessions.db")
        store.save(_session())
        assert (tmp_path / ".qqcode" / "sessions.db").is_file()
        store.close()
