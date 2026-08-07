"""End-to-end escape verification: the whole-repo diff invariant.

The 31 existing guard invariants (`test_guards.py`, `test_workspace.py`) prove
that `PathGuard` and `Workspace` *reject* hostile paths when called directly.
They do not prove the guards sit on the path a model-authored patch actually
travels. That is the difference between "the mechanism exists" and "the
mechanism is wired", and the latter is the property the product sells.

Every test here drives a hostile or ordinary patch through `run_task` with
`dry_run=False` against a real git repo, then inspects the **real repo on disk**
rather than any in-process return value. No API calls: a scripted fake adapter
supplies the patch, so the whole file is deterministic and free to run.

The invariant under test is stronger than "nothing lands outside the repo":

    after run_task, the real repo's diff == result.changed_files, exactly.

`WorktreeWorkspace.finalize` is a whole-tree mirror plus directory swap, not a
per-file overlay. So a file present in the shadow and absent from
`changed_files` still reaches the real repo, and a file deleted from the shadow
disappears from it. Anything the caller was not told about is an escape, in the
sense that matters for the gate: the caller cannot review what it never saw.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from qqcode.acceptance.harness import ACCEPTANCE_DIR
from qqcode.models.billing import BilledClient, RetryPolicy
from qqcode.models.protocol import (
    Completion,
    CostLedger,
    Msg,
    ToolUseContent,
    Usage,
)

# ---------------------------------------------------------------------------
# Fake model plumbing
# ---------------------------------------------------------------------------

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
}


class FakeAdapter:
    def __init__(self, script: list[Completion]) -> None:
        self._script = list(script)
        self.calls = 0

    def invoke(self, messages: list[Msg], **kwargs: Any) -> Completion:
        self.calls += 1
        if not self._script:
            raise AssertionError("FakeAdapter script exhausted")
        return self._script.pop(0)


def _client(script: list[Completion]) -> BilledClient:
    return BilledClient(
        FakeAdapter(script),
        ledger=CostLedger(),
        retry_policy=RetryPolicy(max_attempts=1, sleep=lambda _: None),
    )


def submit_patch(*files: tuple[str, str], reasoning: str = "done") -> Completion:
    """A FastPath patch writing each (path, content) pair."""
    return Completion(
        content=[ToolUseContent(
            id="fp", name="submit_patch",
            input={
                "reasoning": reasoning,
                "files": [{"path": p, "content": c} for p, c in files],
            },
        )],
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=5), raw={},
    )


def fa_write(path: str, content: str) -> Completion:
    return Completion(
        content=[ToolUseContent(id="w", name="write_file",
                                input={"path": path, "content": content})],
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=5), raw={},
    )


def fa_finish(summary: str = "done") -> Completion:
    return Completion(
        content=[ToolUseContent(id="f", name="finish", input={"summary": summary})],
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=5), raw={},
    )


def l1(decision: str = "fastpath", files: tuple[str, ...] = ()) -> Completion:
    return Completion(
        content=[ToolUseContent(
            id="r", name="classify_task",
            input={"decision": decision, "confidence": 0.9,
                   "files": list(files), "reasoning": "x"},
        )],
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=5), raw={},
    )


# ---------------------------------------------------------------------------
# Real git repo, and the on-disk oracle
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A committed git repo. `use_git=True` is the production configuration."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("def greet(): pass\n")
    (root / "keep.py").write_text("UNTOUCHED = 1\n")
    (root / "pkg").mkdir()
    (root / "pkg" / "mod.py").write_text("X = 0\n")
    for cmd in (["git", "init", "-q"], ["git", "add", "."],
                ["git", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=root, check=True, capture_output=True, env=_GIT_ENV)
    return root


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A file outside the repo, in a directory that does not contain the repo.

    The containing directory matters. `test_symlink_escape_is_refused` links
    `repo/link` to `outside.parent`, so putting this file directly in
    `tmp_path` — which also holds `repo` — builds a symlink cycle
    (`repo/link` → `tmp_path` → `repo/link`). While the guard holds, the write
    is refused and `finalize` never runs, so the cycle stays invisible. Disarm
    the guard and `finalize`'s `copytree` (default `symlinks=False`, i.e. it
    dereferences) walks the loop until the disk fills.
    """
    outer = tmp_path / "elsewhere"
    outer.mkdir()
    p = outer / "outside.txt"
    p.write_text("SECRET\n")
    return p


def git_dirty(repo: Path) -> set[str]:
    """Paths the real repo reports as changed, straight from git.

    Deliberately git rather than `snapshot_directory`: reusing the product's own
    snapshot code would make the assertion agree with the implementation by
    construction, including agreeing with its bugs.
    """
    out = subprocess.run(
        ["git", "status", "--porcelain", "-uall"],
        cwd=repo, check=True, capture_output=True, text=True, env=_GIT_ENV,
    ).stdout
    paths: set[str] = set()
    for line in out.splitlines():
        if not line.strip():
            continue
        entry = line[3:]
        # Renames read "old -> new"; the new name is the changed path.
        paths.add(entry.split(" -> ")[-1].strip().strip('"'))
    return paths


def run(
    repo: Path,
    script: list[Completion],
    *,
    mode: str = "fast",
    task: str = "test task",
    dry_run: bool = False,
) -> Any:
    """Drive run_task for real (dry_run=False finalizes onto the repo)."""
    from qqcode.config import Config, ProviderConfig
    from qqcode.orchestrator import run_task

    client = _client(script)
    config = Config(anthropic=ProviderConfig(api_key="fake", base_url=None),
                    openai=None, default_provider="anthropic")
    with patch("qqcode.orchestrator.build_client",
               return_value=(client, client._ledger)):  # noqa: SLF001
        return run_task(task=task, repo=repo, config=config, mode=mode,
                        dry_run=dry_run, seed="worktree")


# ---------------------------------------------------------------------------
# 1. Containment: a hostile path must not escape the repo
# ---------------------------------------------------------------------------

class TestContainment:
    """Guards must be on the patch's path, not merely present in the codebase.

    mode="fast" so a refusal surfaces as `escalation_blocked` instead of being
    absorbed by a Full Agent retry — the point is to observe the refusal.
    """

    @pytest.mark.parametrize("hostile", [
        "../outside.txt",
        "../../outside.txt",
        ".git/config",
        ".env",
    ])
    def test_hostile_path_is_refused_and_repo_untouched(
        self, repo: Path, outside: Path, hostile: str
    ) -> None:
        before = git_dirty(repo)
        result = run(repo, [submit_patch((hostile, "PWNED\n"))])

        assert not result.success, f"{hostile} was accepted"
        assert git_dirty(repo) == before, f"{hostile} modified the real repo"
        assert outside.read_text() == "SECRET\n", f"{hostile} escaped the repo"

    def test_absolute_path_is_refused(self, repo: Path, outside: Path) -> None:
        """The `^/` deny rule, exercised against a path safe to actually hit.

        Deliberately not /etc/passwd: the mutation gate disarms both PathGuard
        layers to prove this test depends on them, and under that mutation the
        write is genuinely attempted. A test must not be able to damage the
        machine running it, so the absolute path points into the tmp tree.
        """
        result = run(repo, [submit_patch((str(outside), "PWNED\n"))])

        assert not result.success
        assert outside.read_text() == "SECRET\n", "absolute path escaped the repo"

    def test_symlink_escape_is_refused(self, repo: Path, outside: Path) -> None:
        """A committed symlink pointing out of the repo is seeded into the shadow.

        `PathGuard` resolves before comparing, so the write lands outside the
        root even though the literal path contains no "..".
        """
        (repo / "link").symlink_to(outside.parent)
        subprocess.run(["git", "add", "."], cwd=repo, check=True,
                       capture_output=True, env=_GIT_ENV)
        subprocess.run(["git", "commit", "-qm", "link"], cwd=repo, check=True,
                       capture_output=True, env=_GIT_ENV)

        result = run(repo, [submit_patch(("link/outside.txt", "PWNED\n"))])

        assert not result.success
        assert outside.read_text() == "SECRET\n", "symlink escaped the workspace"

    def test_acceptance_dir_write_is_refused_before_any_write(
        self, repo: Path
    ) -> None:
        """Condition 1's own directory is not a place the agent may write.

        A `conftest.py` there executes during the harness's collection, before
        the harness overwrites its own files. The check must precede the write,
        so a patch mixing a legal file with an illegal one lands neither.
        """
        before = git_dirty(repo)
        result = run(repo, [submit_patch(
            ("app.py", "def greet(): return 'hi'\n"),
            (f"{ACCEPTANCE_DIR}/conftest.py", "import os\n"),
        )])

        assert not result.success
        assert result.finish_reason == "escalation_blocked"
        assert git_dirty(repo) == before, "the legal half of the patch was applied"
        assert (repo / "app.py").read_text() == "def greet(): pass\n"


# ---------------------------------------------------------------------------
# 2. The full-diff invariant on the success path
# ---------------------------------------------------------------------------

class TestFullDiffMatchesReport:
    """What reached the repo must equal what the caller was told reached it.

    This is the end-to-end whole-diff check the ledger records as missing. A
    caller — human reviewer or benchmark — can only vet `changed_files`, so any
    divergence is unreviewable by construction.
    """

    def test_fastpath_success_diff_equals_changed_files(self, repo: Path) -> None:
        result = run(repo, [submit_patch(("app.py", "def greet(): return 'hi'\n"))])

        assert result.success
        assert git_dirty(repo) == set(result.changed_files) == {"app.py"}
        assert (repo / "app.py").read_text() == "def greet(): return 'hi'\n"

    def test_fullagent_success_diff_equals_changed_files(self, repo: Path) -> None:
        result = run(
            repo,
            [fa_write("pkg/new.py", "Y = 1\n"), fa_finish("added")],
            mode="full",
        )

        assert result.success
        assert git_dirty(repo) == set(result.changed_files) == {"pkg/new.py"}

    def test_untouched_files_are_byte_identical(self, repo: Path) -> None:
        """The whole-tree mirror must not perturb files nobody edited."""
        before = {
            p: (repo / p).read_bytes() for p in ("keep.py", "pkg/mod.py")
        }
        assert run(repo, [submit_patch(("app.py", "def greet(): return 1\n"))]).success

        for p, content in before.items():
            assert (repo / p).read_bytes() == content, f"{p} was perturbed"

    def test_git_directory_survives_finalize(self, repo: Path) -> None:
        """finalize swaps directories; losing .git would destroy the repo."""
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                              capture_output=True, text=True, env=_GIT_ENV).stdout
        assert run(repo, [submit_patch(("app.py", "def greet(): return 2\n"))]).success

        assert (repo / ".git").exists()
        after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                               capture_output=True, text=True, env=_GIT_ENV).stdout
        assert after == head

    def test_finalize_leaves_no_staging_or_backup_siblings(self, repo: Path) -> None:
        assert run(repo, [submit_patch(("app.py", "def greet(): return 3\n"))]).success

        siblings = {p.name for p in repo.parent.iterdir()}
        assert not {s for s in siblings if s.startswith(".qqcode_staging")}
        assert not {s for s in siblings if s.startswith(".qqcode_backup")}


# ---------------------------------------------------------------------------
# 3. Discarded work leaves no residue
# ---------------------------------------------------------------------------

class TestNoResidue:
    def test_dry_run_writes_nothing(self, repo: Path) -> None:
        before = git_dirty(repo)
        result = run(repo, [submit_patch(("app.py", "def greet(): return 4\n"))],
                     dry_run=True)

        assert result.success          # the patch was accepted …
        assert result.dry_run
        assert git_dirty(repo) == before   # … and still never reached the repo
        assert (repo / "app.py").read_text() == "def greet(): pass\n"

    def test_escalated_fastpath_patch_never_reaches_the_repo(
        self, repo: Path
    ) -> None:
        """FastPath's rejected shadow is discarded, not partially kept.

        L1 names app.py, so condition 3's contract is {app.py}; the patch also
        writes pkg/mod.py, so the attempt is refused as
        UNEXPECTED_MODIFICATIONS. Full Agent then restarts from a clean baseline
        and edits only keep.py. Neither file from the discarded attempt may
        appear, and app.py in particular must still hold its original text.
        """
        result = run(
            repo,
            [
                l1("fastpath", ("app.py",)),
                submit_patch(("app.py", "FP = 1\n"), ("pkg/mod.py", "FP = 2\n")),
                fa_write("keep.py", "UNTOUCHED = 2\n"),
                fa_finish("did it differently"),
            ],
            mode="auto",
            task="adjust the greeting wording",
        )

        assert result.success
        assert result.mode_used == "fullagent"
        assert git_dirty(repo) == set(result.changed_files) == {"keep.py"}
        assert (repo / "app.py").read_text() == "def greet(): pass\n"
        assert (repo / "pkg" / "mod.py").read_text() == "X = 0\n"


# ---------------------------------------------------------------------------
# 4. Whole-file writes to a file that was never read
# ---------------------------------------------------------------------------


class TestUnseenFileOverwrite:
    """The gate cannot vouch for a file the model was never shown.

    FastPath writes whole-file replacement content, so replacing a file whose
    text was never inlined means the content came from the model's memory of
    what such a file usually holds. On finalize the real file is destroyed.

    Nothing downstream catches it in the L0-no-hint shape:
      - condition 3 only compares when `files_hint` is non-empty, and this is
        precisely the branch where it is empty
      - condition 1 needs a harness, which batch callers do not pass
      - `finalize` is a whole-tree mirror, so the invented file lands

    `SYSTEM_PROMPT` now tells the model not to reconstruct an unread file and to
    decline instead, which is why real runs decline rather than fabricate. That
    is cooperation, not enforcement, and this test drives a stub that ignores the
    prompt entirely -- so it measures the mechanical gap the prompt cannot close.

    The invariant is narrower than "the patch is correct", which no test can
    decide: FastPath must not silently replace a file it was never shown.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "No mechanical guard yet, deliberately. Any check that rejects this "
            "shape also rejects ~10 existing tests that assert success for a "
            "patch touching an existing file the task never named -- measured, "
            "and not separable by narrowing: every one of them reaches the "
            "check with empty prefetch, exactly like this case. The guard "
            "belongs with the prefetch locator, which makes the resolved set "
            "reflect what the task actually needs; refusing the remainder is "
            "then a backstop rather than a narrowing of FastPath. Remove this "
            "marker when the locator lands -- strict=True fails on XPASS, so "
            "it cannot be forgotten."
        ),
    )
    def test_unseen_existing_file_is_not_silently_replaced(
        self, repo: Path
    ) -> None:
        original = (repo / "app.py").read_text()
        # Task names no path, no files_hint, no prefetch_hint -> zero prefetch.
        result = run(repo, [submit_patch(
            ("app.py", "def greet(): return 'invented from scratch'\n"),
            reasoning="wrote it from scratch since no contents were provided",
        )], task="make the greeting return a value")

        replaced = (repo / "app.py").read_text() != original
        assert not (result.success and replaced), (
            "FastPath replaced a file it never read: prefetch resolved nothing, "
            "condition 3 was unenforceable with an empty files_hint, condition 1 "
            "had no harness, and finalize mirrored the invented content onto the "
            "real file"
        )


# ---------------------------------------------------------------------------
# 5. Ignore-list asymmetry between the report and the mirror
# ---------------------------------------------------------------------------


class TestIgnoreListAsymmetry:
    """Three ignore lists disagree, and only one of them gates the mirror.

    - `snapshot_directory` (drives `changed_files`) ignores .git, __pycache__,
      *.pyc, .pytest_cache, node_modules, .venv, venv, .DS_Store
    - `finalize`'s copytree ignores only .git, __pycache__, *.pyc, .pytest_cache
    - `RUNNER_ARTIFACTS` (drives `filter_acceptance_paths`) covers __pycache__,
      .pytest_cache, .ruff_cache, .mypy_cache

    So a path in one list and not another can reach the real repo while being
    absent from `changed_files`. These tests pin the real behaviour; a failure
    here is a finding about the mirror, not about the test.
    """

    def test_ruff_cache_write_does_not_reach_the_repo_unreported(
        self, repo: Path
    ) -> None:
        """.ruff_cache is a RUNNER_ARTIFACT but neither ignore list excludes it.

        FastPath classifies it as an acceptance path and refuses outright, which
        is the behaviour that keeps the report honest. Asserted through
        `run_task` because the refusal only matters if it is on this path.
        """
        before = git_dirty(repo)
        result = run(repo, [submit_patch((".ruff_cache/evil.py", "BAD = 1\n"))])

        assert not result.success
        assert git_dirty(repo) == before
        assert not (repo / ".ruff_cache" / "evil.py").exists()

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN DEFECT, not a flaky test. Full Agent has no tampering check, "
            "so a write under .ruff_cache/ or .mypy_cache/ is stripped from "
            "changed_files by filter_acceptance_paths yet still mirrored into the "
            "real repo by finalize's copytree. Reachable without malice: `ruff` "
            "and `mypy` are absent from CommandGuard.ALLOW_LIST but `python` is, "
            "so `python -m ruff check .` passes the guard and creates the "
            "directory. Candidate fix is one line — add .ruff_cache and "
            ".mypy_cache to finalize's ignore_patterns, matching what it already "
            "does for __pycache__ and .pytest_cache. Deferred because it changes "
            "what QQCode deletes from a user's repo, which is the user's call. "
            "strict=True so fixing it fails here and forces this marker's removal."
        ),
    )
    def test_fullagent_artifact_write_is_reported_or_absent(
        self, repo: Path
    ) -> None:
        """Full Agent has no tampering check, so this is the exposed edge.

        `filter_acceptance_paths` strips .ruff_cache from the reported set while
        `finalize`'s copytree happily mirrors it. Either the file must not reach
        the repo, or it must appear in `changed_files` — silently arriving is
        the failure mode, because the caller cannot review what it never saw.
        """
        result = run(
            repo,
            [fa_write(".ruff_cache/evil.py", "BAD = 1\n"), fa_finish("done")],
            mode="full",
        )

        landed = (repo / ".ruff_cache" / "evil.py").exists()
        reported = ".ruff_cache/evil.py" in set(result.changed_files)
        assert not landed or reported, (
            "an unreported file reached the real repo: finalize mirrors "
            ".ruff_cache while filter_acceptance_paths hides it from the report"
        )
