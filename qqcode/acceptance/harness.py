"""Hidden acceptance harness: the first of the three convergence conditions.

The acceptance test must be invisible to the agent. An agent that can read the
test can satisfy it without solving the task, which is exactly the failure mode
the three-condition gate exists to catch. Invisibility here means three things:

- **Absent during generation.** Test files are injected *after* the patch is
  written, never before, so no read tool can reach them.
- **Absent from the diff.** They are removed before the diff check runs, so a
  passing run cannot be spoiled by the harness's own files, and the agent cannot
  smuggle changes in under a harness path.
- **Off the agent's tool surface.** The command runs through this module, not
  through `Workspace.run_command`, so it never appears in the tool log the agent
  sees on the next turn.

That last point is why the harness deliberately does not go through
`CommandGuard`. `CommandGuard` is the leash on *agent-authored* commands; an
acceptance command comes from the task author, whose trust level is the same as
the person running QQCode. **Consequence worth stating plainly: whoever supplies
an acceptance test can execute arbitrary code on this machine.** Acceptance
suites are code, and must be trusted like code — never accept one from an
untrusted source. The harness still strips secrets from the environment and
enforces a timeout, because a trusted test can still be buggy or hang.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from qqcode.workspace.worktree import _sanitized_env

# Every injected file lives under this directory so cleanup is a single rmtree
# and the diff filter is a single prefix check. Nothing else may write here.
ACCEPTANCE_DIR = ".qqcode_acceptance"

# Artifacts test runners leave behind. Removed alongside the injected files so
# they never reach the diff check.
RUNNER_ARTIFACTS = ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache")


@dataclass(frozen=True)
class AcceptanceTest:
    """A hidden behavioral test, supplied externally by the task author.

    Args:
        name: Identifier used in diagnostics.
        files: Paths (relative to `ACCEPTANCE_DIR`) mapped to their content.
        command: Argv to run, with paths relative to the workspace root.
        timeout: Hard wall-clock limit in seconds.
    """

    name: str
    command: tuple[str, ...]
    files: dict[str, str] = field(default_factory=dict)
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AcceptanceTest.name must be non-empty")
        if not self.command:
            raise ValueError(f"{self.name}: command must be non-empty")
        if self.timeout <= 0:
            raise ValueError(f"{self.name}: timeout must be positive, got {self.timeout}")
        for rel in self.files:
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ValueError(
                    f"{self.name}: acceptance file path must be relative and contained: {rel!r}"
                )


@dataclass(frozen=True)
class AcceptanceResult:
    """Outcome of one acceptance run."""

    name: str
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def diagnostic(self, *, limit: int = 2000) -> dict[str, str]:
        """Failure context for the escalation payload.

        Truncated from the tail: a test runner's verdict and traceback land at
        the end of its output, so the head is the part worth dropping.
        """
        return {
            "acceptance_name": self.name,
            "acceptance_exit_code": str(self.exit_code),
            "acceptance_timed_out": str(self.timed_out).lower(),
            "acceptance_stdout": self.stdout[-limit:],
            "acceptance_stderr": self.stderr[-limit:],
        }


def is_acceptance_path(rel_path: str) -> bool:
    """Whether a repo-relative path belongs to the harness rather than the task."""
    parts = Path(rel_path).parts
    if not parts:
        return False
    return parts[0] == ACCEPTANCE_DIR or parts[0] in RUNNER_ARTIFACTS


def filter_acceptance_paths(paths: frozenset[str] | set[str]) -> frozenset[str]:
    """Drop harness-owned paths from a changed-file set."""
    return frozenset(p for p in paths if not is_acceptance_path(p))


class AcceptanceHarness:
    """Runs hidden acceptance tests against a shadow workspace.

    Writes test files, runs the command, then removes everything it created —
    including test-runner caches — so the workspace it hands back differs from
    the baseline only where the agent changed it.
    """

    def __init__(self, tests: list[AcceptanceTest]):
        """
        Args:
            tests: Suites to run in order. An empty list makes `run` a no-op
                that reports success, which is the correct reading of "no
                acceptance criteria were specified".

        Raises:
            ValueError: Two tests share a name.
        """
        names = [t.name for t in tests]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Duplicate acceptance test names: {sorted(duplicates)}")
        self._tests = list(tests)

    def __len__(self) -> int:
        return len(self._tests)

    @property
    def tests(self) -> list[AcceptanceTest]:
        return list(self._tests)

    def run(self, root: Path) -> list[AcceptanceResult]:
        """Run every test against the workspace at `root`.

        Stops at the first failure: later suites add no information once the
        gate is already closed, and each one costs wall-clock time.

        Returns:
            Results in execution order. Empty when no tests are configured.
        """
        results: list[AcceptanceResult] = []
        for test in self._tests:
            result = self._run_one(test, root)
            results.append(result)
            if not result.passed:
                break
        return results

    def _run_one(self, test: AcceptanceTest, root: Path) -> AcceptanceResult:
        """Inject, execute, clean up. Cleanup runs even when the command raises."""
        try:
            self._inject(test, root)
            return self._execute(test, root)
        finally:
            self._cleanup(root)

    def _inject(self, test: AcceptanceTest, root: Path) -> None:
        base = root / ACCEPTANCE_DIR
        base.mkdir(parents=True, exist_ok=True)
        for rel, content in test.files.items():
            target = (base / rel).resolve()
            # Defence in depth: __post_init__ already rejects traversal, but the
            # resolved path is what actually gets written.
            if not target.is_relative_to(base.resolve()):
                raise ValueError(f"{test.name}: acceptance file escapes harness dir: {rel!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def _execute(self, test: AcceptanceTest, root: Path) -> AcceptanceResult:
        """Run the command directly — no CommandGuard, by design (see module docstring)."""
        try:
            proc = subprocess.run(
                list(test.command),
                cwd=root,
                capture_output=True,
                text=True,
                timeout=test.timeout,
                env=_sanitized_env(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return AcceptanceResult(
                name=test.name,
                passed=False,
                exit_code=-1,
                stdout=_as_text(exc.stdout),
                stderr=_as_text(exc.stderr) or f"timed out after {test.timeout}s",
                timed_out=True,
            )
        except OSError as exc:
            # A missing interpreter is a harness misconfiguration, not a task failure.
            return AcceptanceResult(
                name=test.name,
                passed=False,
                exit_code=-1,
                stdout="",
                stderr=f"could not execute {test.command[0]!r}: {exc}",
            )

        return AcceptanceResult(
            name=test.name,
            passed=proc.returncode == 0,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    def _cleanup(self, root: Path) -> None:
        """Remove injected files and runner caches, recursively."""
        shutil.rmtree(root / ACCEPTANCE_DIR, ignore_errors=True)
        for artifact in RUNNER_ARTIFACTS:
            for path in root.rglob(artifact):
                shutil.rmtree(path, ignore_errors=True)


def _as_text(raw: str | bytes | None) -> str:
    """Normalize subprocess output, which may be bytes on timeout."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw


def all_passed(results: list[AcceptanceResult]) -> bool:
    """Whether the acceptance condition is satisfied.

    An empty result list passes: no criteria specified means this condition
    imposes nothing. The other two conditions still apply.
    """
    return all(r.passed for r in results)


def first_failure(results: list[AcceptanceResult]) -> AcceptanceResult | None:
    """The first failing result, or None when all passed."""
    return next((r for r in results if not r.passed), None)
