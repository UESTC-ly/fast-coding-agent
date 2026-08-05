"""Workspace protocol and implementations.

Workspace provides isolated read/write/exec with safety controls:
- Path allowlist (prevent escapes)
- Snapshot baseline (detect unexpected modifications)
- Shadow mode (apply changes in copy, verify before finalizing)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class FileSnapshot:
    """Fingerprint of a file at a point in time."""

    path: str  # Relative to repo root
    hash: str  # SHA-256
    size: int
    exists: bool = True


@dataclass
class WorkspaceSnapshot:
    """Baseline fingerprint of the workspace."""

    root: Path
    files: dict[str, FileSnapshot] = field(default_factory=dict)

    def changed_files(self, current: WorkspaceSnapshot) -> set[str]:
        """Return paths that differ from baseline."""
        changed = set()
        all_paths = set(self.files.keys()) | set(current.files.keys())
        for p in all_paths:
            baseline = self.files.get(p)
            now = current.files.get(p)
            if baseline != now:
                changed.add(p)
        return changed


class Workspace(Protocol):
    """Abstract workspace for code operations.

    Implementations:
    - WorktreeWorkspace: git worktree + file copy (M1)
    - SandboxWorkspace: Docker container (future)
    """

    @property
    def root(self) -> Path:
        """Workspace root directory."""
        ...

    def snapshot(self) -> WorkspaceSnapshot:
        """Capture current state of all tracked files."""
        ...

    def read_file(self, path: str) -> str:
        """Read file content (path relative to root).

        Raises:
            FileNotFoundError: Path doesn't exist
            PermissionError: Path outside allowlist
        """
        ...

    def write_file(self, path: str, content: str) -> None:
        """Write file (creates parent dirs if needed).

        Raises:
            PermissionError: Path outside allowlist or write budget exceeded
        """
        ...

    def list_files(self, pattern: str = "*") -> list[str]:
        """List files matching pattern (relative paths)."""
        ...

    def run_command(
        self,
        cmd: list[str],
        cwd: str | None = None,
        timeout: float = 30.0,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Execute command in workspace.

        Args:
            cmd: Command and arguments
            cwd: Working directory (relative to root)
            timeout: Max execution time
            env: Environment variables (replaces defaults, no secrets)

        Returns:
            (exit_code, stdout, stderr)

        Raises:
            TimeoutError: Command exceeded timeout
            PermissionError: Command not allowed
        """
        ...

    def finalize(self, target: Path) -> None:
        """Atomically replace target with workspace content.

        Only call after all three conditions pass:
        - Hidden acceptance tests pass
        - Agent reached valid finish state
        - Diff ⊆ expected file set
        """
        ...

    def cleanup(self) -> None:
        """Remove workspace (shadow copy or container)."""
        ...


def compute_file_hash(path: Path) -> str:
    """SHA-256 hash of file content."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def snapshot_directory(root: Path, patterns: list[str] | None = None) -> WorkspaceSnapshot:
    """Create snapshot of directory tree.

    Args:
        root: Directory to snapshot
        patterns: Glob patterns to include (default: all non-ignored files)

    Returns:
        Snapshot with file hashes
    """
    snap = WorkspaceSnapshot(root=root)

    # Default: track all files except common ignores
    ignore_patterns = {
        ".git",
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
        ".DS_Store",
    }

    def should_track(p: Path) -> bool:
        for part in p.parts:
            if any(part == pat or part.startswith(pat.rstrip("*")) for pat in ignore_patterns):
                return False
        return p.is_file()

    for path in root.rglob("*"):
        if not should_track(path):
            continue
        rel = str(path.relative_to(root))
        snap.files[rel] = FileSnapshot(
            path=rel,
            hash=compute_file_hash(path),
            size=path.stat().st_size,
        )

    return snap
