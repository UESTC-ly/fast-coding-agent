"""Path and command safety guards.

All workspace operations go through these guards before touching filesystem.
"""

from __future__ import annotations

import re
from pathlib import Path


class PathGuard:
    """Validate paths stay within workspace bounds."""

    # Blocked path components
    DENY_PATTERNS = [
        r"\.\.(/|$)",  # Parent directory traversal
        r"^/",  # Absolute paths
        r"~",  # Home expansion
        r"\.git/",  # Git internals
        r"\.ssh/",  # SSH keys
        r"\.aws/",  # Cloud credentials
        r"\.env",  # Environment files
    ]

    def __init__(self, root: Path, extra_deny: list[str] | None = None):
        """
        Args:
            root: Workspace root (resolved absolute path)
            extra_deny: Additional regex patterns to block
        """
        self.root = root.resolve()
        self.deny_re = re.compile("|".join(self.DENY_PATTERNS + (extra_deny or [])))

    def validate(self, path: str) -> Path:
        """Check if path is safe to access.

        Args:
            path: Relative path from workspace root

        Returns:
            Resolved absolute path

        Raises:
            PermissionError: Path escapes workspace or matches deny pattern
        """
        # Check deny patterns first (before resolving)
        if self.deny_re.search(path):
            raise PermissionError(f"Path blocked by deny pattern: {path}")

        # Resolve and check containment
        abs_path = (self.root / path).resolve()
        try:
            abs_path.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"Path escapes workspace: {path} -> {abs_path}") from exc

        return abs_path


class CommandGuard:
    """Validate commands before execution."""

    # Commands allowed without network
    ALLOW_LIST = {
        # Shell basics
        "echo",
        "cat",
        "ls",
        "pwd",
        "find",
        "grep",
        "head",
        "tail",
        "wc",
        # Build and test
        "python",
        "python3",
        "pytest",
        "pip",
        "uv",
        "npm",
        "node",
        "cargo",
        "go",
        "make",
        "cmake",
        # Version control (read-only)
        "git",
        # File operations
        "cp",
        "mv",
        "mkdir",
        "rm",
        "touch",
    }

    # Commands always denied
    DENY_LIST = {
        "curl",
        "wget",
        "ssh",
        "scp",
        "rsync",
        "docker",
        "kubectl",
        "nc",
        "netcat",
        "telnet",
        "ftp",
        "sftp",
    }

    # Dangerous flags
    DANGER_FLAGS = {
        "rm": {"-rf", "--recursive --force"},
        "git": {"push", "fetch", "pull", "clone"},
    }

    def validate(self, cmd: list[str]) -> None:
        """Check if command is allowed.

        Raises:
            PermissionError: Command denied
        """
        if not cmd:
            raise PermissionError("Empty command")

        binary = cmd[0]

        # Deny list takes precedence
        if binary in self.DENY_LIST:
            raise PermissionError(f"Command denied: {binary}")

        # Must be in allow list
        if binary not in self.ALLOW_LIST:
            raise PermissionError(f"Command not in allow list: {binary}")

        # Check dangerous flag combinations
        if binary in self.DANGER_FLAGS:
            cmd_str = " ".join(cmd)
            for danger in self.DANGER_FLAGS[binary]:
                if danger in cmd_str:
                    raise PermissionError(f"Dangerous flag blocked: {binary} {danger}")


class WriteQuota:
    """Track and limit write operations."""

    def __init__(
        self,
        max_files: int = 50,
        max_lines_per_file: int = 2000,
        max_total_bytes: int = 10_000_000,  # 10MB
    ):
        self.max_files = max_files
        self.max_lines_per_file = max_lines_per_file
        self.max_total_bytes = max_total_bytes

        self.files_written = 0
        self.total_bytes = 0

    def check(self, content: str) -> None:
        """Verify write is within quota.

        Raises:
            PermissionError: Quota exceeded
        """
        lines = content.count("\n") + 1
        size = len(content.encode("utf-8"))

        if lines > self.max_lines_per_file:
            raise PermissionError(
                f"File exceeds line limit: {lines} > {self.max_lines_per_file}"
            )

        if self.files_written >= self.max_files:
            raise PermissionError(f"Write quota exhausted: {self.files_written} files")

        if self.total_bytes + size > self.max_total_bytes:
            raise PermissionError(
                f"Byte quota exceeded: {self.total_bytes + size} > {self.max_total_bytes}"
            )

    def record(self, content: str) -> None:
        """Record a successful write."""
        self.files_written += 1
        self.total_bytes += len(content.encode("utf-8"))
