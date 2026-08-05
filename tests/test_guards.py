"""Tests for path, command, and write-quota guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from qqcode.safety.guards import CommandGuard, PathGuard, WriteQuota


class TestPathGuard:
    def test_allows_relative_path_inside_root(self, tmp_path: Path) -> None:
        guard = PathGuard(tmp_path)
        assert guard.validate("src/main.py") == (tmp_path / "src/main.py").resolve()

    def test_blocks_parent_traversal(self, tmp_path: Path) -> None:
        guard = PathGuard(tmp_path)
        with pytest.raises(PermissionError, match="deny pattern"):
            guard.validate("../outside.py")

    def test_blocks_absolute_path(self, tmp_path: Path) -> None:
        guard = PathGuard(tmp_path)
        with pytest.raises(PermissionError, match="deny pattern"):
            guard.validate("/etc/passwd")

    def test_blocks_git_internals(self, tmp_path: Path) -> None:
        guard = PathGuard(tmp_path)
        with pytest.raises(PermissionError, match="deny pattern"):
            guard.validate(".git/config")

    def test_blocks_env_file(self, tmp_path: Path) -> None:
        guard = PathGuard(tmp_path)
        with pytest.raises(PermissionError, match="deny pattern"):
            guard.validate(".env")

    def test_blocks_symlink_escape(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "link").symlink_to(outside)

        guard = PathGuard(root)
        with pytest.raises(PermissionError, match="escapes workspace"):
            guard.validate("link/secret.txt")

    def test_extra_deny_pattern(self, tmp_path: Path) -> None:
        guard = PathGuard(tmp_path, extra_deny=[r"migrations/"])
        with pytest.raises(PermissionError, match="deny pattern"):
            guard.validate("db/migrations/001.sql")


class TestCommandGuard:
    def test_allows_pytest(self) -> None:
        CommandGuard().validate(["pytest", "-q"])

    def test_rejects_empty_command(self) -> None:
        with pytest.raises(PermissionError, match="Empty command"):
            CommandGuard().validate([])

    def test_denies_network_binary(self) -> None:
        with pytest.raises(PermissionError, match="Command denied"):
            CommandGuard().validate(["curl", "https://example.com"])

    def test_denies_unlisted_binary(self) -> None:
        with pytest.raises(PermissionError, match="not in allow list"):
            CommandGuard().validate(["sudo", "ls"])

    def test_denies_recursive_force_remove(self) -> None:
        with pytest.raises(PermissionError, match="Dangerous flag"):
            CommandGuard().validate(["rm", "-rf", "/"])

    def test_denies_git_push(self) -> None:
        with pytest.raises(PermissionError, match="Dangerous flag"):
            CommandGuard().validate(["git", "push", "origin", "main"])

    def test_allows_git_status(self) -> None:
        CommandGuard().validate(["git", "status"])


class TestWriteQuota:
    def test_accepts_write_within_limits(self) -> None:
        quota = WriteQuota()
        quota.check("hello\n")
        quota.record("hello\n")
        assert quota.files_written == 1

    def test_rejects_file_over_line_limit(self) -> None:
        quota = WriteQuota(max_lines_per_file=10)
        with pytest.raises(PermissionError, match="line limit"):
            quota.check("x\n" * 20)

    def test_rejects_write_past_file_count(self) -> None:
        quota = WriteQuota(max_files=2)
        for _ in range(2):
            quota.check("a")
            quota.record("a")
        with pytest.raises(PermissionError, match="quota exhausted"):
            quota.check("a")

    def test_rejects_write_past_byte_budget(self) -> None:
        quota = WriteQuota(max_total_bytes=100)
        payload = "x" * 60
        quota.check(payload)
        quota.record(payload)
        with pytest.raises(PermissionError, match="Byte quota exceeded"):
            quota.check(payload)
