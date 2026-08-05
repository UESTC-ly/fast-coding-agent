"""Tests for shadow workspace isolation and finalize semantics."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from qqcode.workspace.protocol import snapshot_directory
from qqcode.workspace.worktree import WorktreeWorkspace, _sanitized_env


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A plain (non-git) source directory."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("print('hello')\n")
    (root / "src").mkdir()
    (root / "src" / "util.py").write_text("def helper(): pass\n")
    return root


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A git repository with one commit."""
    root = tmp_path / "gitrepo"
    root.mkdir()
    (root / "main.py").write_text("print('hello')\n")

    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
    }
    for cmd in (["git", "init", "-q"], ["git", "add", "."], ["git", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=root, check=True, capture_output=True, env=env)
    return root


class TestIsolation:
    def test_shadow_root_is_outside_source(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws:
            assert ws.root != repo
            assert repo not in ws.root.parents

    def test_source_files_are_visible_in_shadow(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws:
            assert ws.read_file("main.py") == "print('hello')\n"
            assert ws.read_file("src/util.py") == "def helper(): pass\n"

    def test_writes_do_not_leak_to_source(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws:
            ws.write_file("main.py", "print('modified')\n")
            assert ws.read_file("main.py") == "print('modified')\n"

        assert (repo / "main.py").read_text() == "print('hello')\n"

    def test_new_files_do_not_leak_to_source(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws:
            ws.write_file("added.py", "x = 1\n")

        assert not (repo / "added.py").exists()

    def test_cleanup_removes_shadow_directory(self, repo: Path) -> None:
        ws = WorktreeWorkspace(repo, use_git=False)
        shadow = ws.root
        assert shadow.exists()
        ws.cleanup()
        assert not shadow.exists()


class TestGitWorktree:
    def test_uses_worktree_for_git_repo(self, git_repo: Path) -> None:
        with WorktreeWorkspace(git_repo) as ws:
            assert ws.is_worktree
            assert ws.read_file("main.py") == "print('hello')\n"

    def test_falls_back_to_copy_without_commits(self, repo: Path) -> None:
        with WorktreeWorkspace(repo) as ws:
            assert not ws.is_worktree
            assert ws.read_file("main.py") == "print('hello')\n"


class TestGuardIntegration:
    def test_read_rejects_escaping_path(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws, pytest.raises(PermissionError):
            ws.read_file("../outside.py")

    def test_write_rejects_escaping_path(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws, pytest.raises(PermissionError):
            ws.write_file("/tmp/evil.py", "x = 1")

    def test_run_command_rejects_denied_binary(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws, pytest.raises(PermissionError):
            ws.run_command(["curl", "https://example.com"])

    def test_run_command_executes_allowed_binary(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws:
            code, out, _ = ws.run_command(["echo", "ok"])
            assert code == 0
            assert out.strip() == "ok"

    def test_run_command_enforces_timeout(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws, pytest.raises(TimeoutError):
            ws.run_command(["python3", "-c", "import time; time.sleep(5)"], timeout=0.5)


class TestSnapshot:
    def test_snapshot_captures_all_files(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws:
            snap = ws.snapshot()
            assert set(snap.files) == {"main.py", "src/util.py"}

    def test_changed_files_detects_modification(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws:
            before = ws.snapshot()
            ws.write_file("main.py", "print('changed')\n")
            assert before.changed_files(ws.snapshot()) == {"main.py"}

    def test_changed_files_detects_addition(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws:
            before = ws.snapshot()
            ws.write_file("new.py", "x = 1\n")
            assert before.changed_files(ws.snapshot()) == {"new.py"}

    def test_changed_files_detects_deletion(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws:
            before = ws.snapshot()
            (ws.root / "main.py").unlink()
            assert before.changed_files(ws.snapshot()) == {"main.py"}

    def test_changed_files_empty_when_untouched(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws:
            assert ws.snapshot().changed_files(ws.snapshot()) == set()

    def test_snapshot_skips_ignored_directories(self, repo: Path) -> None:
        (repo / "__pycache__").mkdir()
        (repo / "__pycache__" / "x.pyc").write_bytes(b"junk")
        assert "__pycache__/x.pyc" not in snapshot_directory(repo).files


class TestFinalize:
    def test_finalize_applies_changes_to_target(self, repo: Path) -> None:
        ws = WorktreeWorkspace(repo, use_git=False)
        ws.write_file("main.py", "print('finalized')\n")
        ws.write_file("added.py", "y = 2\n")
        ws.finalize(repo)
        ws.cleanup()

        assert (repo / "main.py").read_text() == "print('finalized')\n"
        assert (repo / "added.py").read_text() == "y = 2\n"

    def test_finalize_preserves_target_git_directory(self, git_repo: Path) -> None:
        ws = WorktreeWorkspace(git_repo)
        ws.write_file("main.py", "print('finalized')\n")
        ws.finalize(git_repo)
        ws.cleanup()

        assert (git_repo / ".git").exists()
        assert (git_repo / "main.py").read_text() == "print('finalized')\n"

    def test_finalize_leaves_no_staging_artifacts(self, repo: Path) -> None:
        ws = WorktreeWorkspace(repo, use_git=False)
        ws.write_file("main.py", "print('x')\n")
        ws.finalize(repo)
        ws.cleanup()

        siblings = {p.name for p in repo.parent.iterdir()}
        assert not any(n.startswith(".qqcode_") for n in siblings)

    def test_target_untouched_without_finalize(self, repo: Path) -> None:
        with WorktreeWorkspace(repo, use_git=False) as ws:
            ws.write_file("main.py", "print('discarded')\n")
            ws.write_file("junk.py", "garbage\n")

        assert (repo / "main.py").read_text() == "print('hello')\n"
        assert not (repo / "junk.py").exists()


class TestEnvSanitization:
    def test_strips_prefixed_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
        env = _sanitized_env()
        assert "ANTHROPIC_API_KEY" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env

    def test_strips_substring_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_DB_PASSWORD", "hunter2")
        monkeypatch.setenv("SOME_TOKEN", "abc")
        env = _sanitized_env()
        assert "MY_DB_PASSWORD" not in env
        assert "SOME_TOKEN" not in env

    def test_keeps_benign_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", "/usr/bin")
        assert _sanitized_env()["PATH"] == "/usr/bin"

    def test_marks_environment_offline(self) -> None:
        assert _sanitized_env()["NO_NETWORK"] == "1"

    def test_overrides_win(self) -> None:
        assert _sanitized_env({"CUSTOM": "v"})["CUSTOM"] == "v"

    def test_secrets_never_reach_subprocess(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-leaked")
        with WorktreeWorkspace(repo, use_git=False) as ws:
            code, out, _ = ws.run_command(
                [
                    "python3",
                    "-c",
                    "import os; print(os.environ.get('ANTHROPIC_API_KEY', 'ABSENT'))",
                ]
            )
            assert code == 0
            assert out.strip() == "ABSENT"
