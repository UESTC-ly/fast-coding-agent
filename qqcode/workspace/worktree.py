"""Worktree-based workspace: git worktree or plain file copy.

Provides isolation via a shadow directory. All writes land in the shadow;
finalize() atomically swaps it into the real workspace only after the
three-condition gate passes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from qqcode.safety.guards import CommandGuard, PathGuard, WriteQuota
from qqcode.workspace.protocol import WorkspaceSnapshot, snapshot_directory

# Environment variables stripped before running any command (secret hygiene).
SECRET_ENV_PREFIXES = (
    "ANTHROPIC_",
    "OPENAI_",
    "AWS_",
    "GITHUB_",
    "GH_",
    "GOOGLE_",
    "AZURE_",
)
SECRET_ENV_SUBSTRINGS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL")


def _sanitized_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Build a command environment with secrets removed."""
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(SECRET_ENV_PREFIXES)
        and not any(s in k.upper() for s in SECRET_ENV_SUBSTRINGS)
    }
    # Force offline for tooling that respects these.
    clean["NO_NETWORK"] = "1"
    clean["PIP_NO_INPUT"] = "1"
    if overrides:
        clean.update(overrides)
    return clean


class WorktreeWorkspace:
    """Isolated workspace backed by a shadow directory copy.

    Implements the Workspace protocol. Uses `git worktree` when the source is a
    git repository with at least one commit, otherwise falls back to a plain
    recursive copy. Both paths yield a directory that can be mutated freely and
    discarded without touching the original.
    """

    def __init__(
        self,
        source: Path,
        *,
        use_git: bool = True,
        quota: WriteQuota | None = None,
    ):
        """
        Args:
            source: Real repository root to shadow.
            use_git: Attempt `git worktree add` before falling back to copy.
            quota: Write limits; a default quota is created when omitted.
        """
        self.source = source.resolve()
        self._shadow_parent = Path(tempfile.mkdtemp(prefix="qqcode_shadow_"))
        self._root = self._shadow_parent / self.source.name
        self._is_worktree = False

        if use_git and self._try_git_worktree():
            self._is_worktree = True
        else:
            shutil.copytree(
                self.source,
                self._root,
                ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", "*.pyc", ".pytest_cache", ".venv", "node_modules"
                ),
            )

        self.path_guard = PathGuard(self._root)
        self.command_guard = CommandGuard()
        self.quota = quota or WriteQuota()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def is_worktree(self) -> bool:
        """True when backed by `git worktree`, False when a plain copy."""
        return self._is_worktree

    def _try_git_worktree(self) -> bool:
        """Create a detached git worktree at the shadow root.

        Returns:
            True on success; False when the source is not a usable git repo.
        """
        try:
            head = subprocess.run(
                ["git", "-C", str(self.source), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if head.returncode != 0:
                return False

            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.source),
                    "worktree",
                    "add",
                    "--detach",
                    str(self._root),
                    head.stdout.strip(),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    def snapshot(self) -> WorkspaceSnapshot:
        return snapshot_directory(self._root)

    def read_file(self, path: str) -> str:
        abs_path = self.path_guard.validate(path)
        if not abs_path.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        return abs_path.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        abs_path = self.path_guard.validate(path)
        self.quota.check(content)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        self.quota.record(content)

    def list_files(self, pattern: str = "*") -> list[str]:
        return sorted(
            str(p.relative_to(self._root))
            for p in self._root.rglob(pattern)
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
        )

    def run_command(
        self,
        cmd: list[str],
        cwd: str | None = None,
        timeout: float = 30.0,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        self.command_guard.validate(cmd)
        work_dir = self.path_guard.validate(cwd) if cwd else self._root

        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_sanitized_env(env),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Command timed out after {timeout}s: {' '.join(cmd)}") from exc

        return result.returncode, result.stdout, result.stderr

    def finalize(self, target: Path) -> None:
        """Copy shadow content over target, preserving target's .git directory.

        Writes to a sibling staging directory first, then swaps directories so a
        crash mid-copy cannot leave the target half-updated.
        """
        target = target.resolve()
        staging = target.parent / f".qqcode_staging_{target.name}"
        backup = target.parent / f".qqcode_backup_{target.name}"

        for stale in (staging, backup):
            if stale.exists():
                shutil.rmtree(stale)

        shutil.copytree(
            self._root,
            staging,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache"),
        )

        # Carry over version control state from the original target.
        target_git = target / ".git"
        if target_git.exists():
            if target_git.is_dir():
                shutil.copytree(target_git, staging / ".git", symlinks=True)
            else:
                shutil.copy2(target_git, staging / ".git")

        target.rename(backup)
        try:
            staging.rename(target)
        except OSError:
            backup.rename(target)
            raise
        shutil.rmtree(backup)

    def cleanup(self) -> None:
        """Remove the shadow directory and any git worktree registration."""
        if self._is_worktree:
            subprocess.run(
                ["git", "-C", str(self.source), "worktree", "remove", "--force", str(self._root)],
                capture_output=True,
                timeout=30,
                check=False,
            )
        shutil.rmtree(self._shadow_parent, ignore_errors=True)

    def __enter__(self) -> WorktreeWorkspace:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()
