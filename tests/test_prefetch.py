"""Tests for FastPath's prefetch path resolution.

L0 and the fallback route both send tasks to FastPath with `files_hint=()`,
because neither identifies *which* files a task touches. FastPath prefetched
by hint alone, so it prefetched nothing, and its prompt then told the model
that current file contents were provided. The model could not see the code it
was asked to change and took the documented exit (an empty patch), costing one
call before escalating.

These tests pin the fix: prefetch paths are resolved from the task text against
the real workspace, and that resolution is kept strictly separate from
`files_hint`'s second role as the condition-3 enforcement contract.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from qqcode.models.billing import BilledClient
from qqcode.models.protocol import Completion, TextContent, ToolUseContent, Usage
from qqcode.routing.fastpath import (
    _PATH_TOKEN,
    MAX_PREFETCH_FILE_CHARS,
    MAX_PREFETCH_RESOLVED_FILES,
    MAX_PREFETCH_SCAN_DEPTH,
    PATCH_TOOL_NAME,
    EscalationReason,
    FastPathInput,
    _is_file,
    execute_fastpath,
    resolve_prefetch_paths,
)
from qqcode.skills.index import SkillIndex
from qqcode.tools.builtins import default_registry
from qqcode.workspace.worktree import WorktreeWorkspace


@pytest.fixture
def workspace(tmp_path: Path) -> WorktreeWorkspace:
    """A workspace with a small, realistic file tree."""
    (tmp_path / "calc.py").write_text("def divide(a, b):\n    return a / b\n")
    (tmp_path / "utils.py").write_text("def helper():\n    pass\n")
    (tmp_path / "README.md").write_text("# docs\n")
    pkg = tmp_path / "src" / "app"
    pkg.mkdir(parents=True)
    (pkg / "config.py").write_text("DEBUG = False\n")
    (pkg / "models.py").write_text("class User:\n    pass\n")
    return WorktreeWorkspace(tmp_path, use_git=False)


def _patch_client(files: dict[str, str]) -> Mock:
    """A BilledClient stand-in that returns `files` as a submitted patch."""
    client = Mock(spec=BilledClient)
    client.invoke.return_value = Completion(
        content=[
            ToolUseContent(
                id="call_1",
                name=PATCH_TOOL_NAME,
                input={
                    "reasoning": "done",
                    "files": [{"path": p, "content": c} for p, c in files.items()],
                },
            )
        ],
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=5),
        raw={},
    )
    return client


def _prompt_text(client: Mock) -> str:
    """All text the client was asked to send, concatenated."""
    messages = client.invoke.call_args.kwargs["messages"]
    return "\n".join(
        block.text
        for msg in messages
        for block in msg.content
        if isinstance(block, TextContent)
    )


def _fastpath_input(task: str, ws: WorktreeWorkspace, hint: tuple[str, ...] = ()) -> FastPathInput:
    return FastPathInput(
        task=task,
        baseline=ws.snapshot(),
        skill_index=SkillIndex(),
        tool_registry=default_registry(),
        files_hint=hint,
    )


# ---------------------------------------------------------------------------
# Resolution from task text
# ---------------------------------------------------------------------------


class TestResolveFromTaskText:
    def test_bare_filename_resolves(self, workspace: WorktreeWorkspace) -> None:
        """The defect's exact case: a named file becomes a prefetch path."""
        got = resolve_prefetch_paths("Add a docstring to divide in calc.py", (), workspace)
        assert got == ("calc.py",)

    def test_nested_path_resolves_by_suffix(self, workspace: WorktreeWorkspace) -> None:
        """A bare filename matches a file nested under a package."""
        got = resolve_prefetch_paths("Set DEBUG in config.py", (), workspace)
        assert got == ("src/app/config.py",)

    def test_explicit_relative_path_resolves(self, workspace: WorktreeWorkspace) -> None:
        got = resolve_prefetch_paths("Edit src/app/models.py", (), workspace)
        assert got == ("src/app/models.py",)

    def test_multiple_files_all_resolve_sorted(self, workspace: WorktreeWorkspace) -> None:
        got = resolve_prefetch_paths("Move helper from utils.py into calc.py", (), workspace)
        assert got == ("calc.py", "utils.py")

    def test_nonexistent_file_is_not_resolved(self, workspace: WorktreeWorkspace) -> None:
        """A guessed name that isn't in the tree must not be prefetched.

        Prefetching a nonexistent path is worse than useless: `_prefetch_files`
        omits it silently, so it would cost resolution work for nothing.
        """
        got = resolve_prefetch_paths("Create brand_new_module.py", (), workspace)
        assert got == ()

    def test_no_filename_in_task_resolves_nothing(self, workspace: WorktreeWorkspace) -> None:
        """No invented paths when the task names no file."""
        got = resolve_prefetch_paths("Make division safe for zero denominators", (), workspace)
        assert got == ()

    def test_prose_word_with_dot_is_not_a_path(self, workspace: WorktreeWorkspace) -> None:
        """Sentence punctuation must not be read as a filename."""
        got = resolve_prefetch_paths("Fix the bug.Then add a test.", (), workspace)
        assert got == ()

    def test_extension_allowlist_rejects_prose_tokens(self) -> None:
        """The regex itself must reject prose, not lean on the existence check.

        Both layers filter `bug.Then`, so asserting only on resolver output
        passes even with a `\\S+\\.\\S+` regex. Pinning the token layer directly
        is what keeps the allowlist from being loosened later — a prose token
        that happened to collide with a real filename would otherwise be
        inlined as context.
        """
        assert _PATH_TOKEN.findall("Fix the bug.Then add a test.") == []
        assert _PATH_TOKEN.findall("Upgrade to version 2.0 today") == []
        assert _PATH_TOKEN.findall("Handle the e.g. case") == []
        # ...while still recognising real paths.
        assert _PATH_TOKEN.findall("edit calc.py and src/app/config.py") == [
            "calc.py",
            "src/app/config.py",
        ]

    def test_resolution_is_capped(self, tmp_path: Path) -> None:
        """A task naming many files cannot blow up the prompt."""
        names = [f"mod{i}.py" for i in range(MAX_PREFETCH_RESOLVED_FILES + 5)]
        for n in names:
            (tmp_path / n).write_text("x = 1\n")
        ws = WorktreeWorkspace(tmp_path, use_git=False)
        got = resolve_prefetch_paths("Update " + ", ".join(names), (), ws)
        assert len(got) == MAX_PREFETCH_RESOLVED_FILES

    def test_ambiguous_basename_is_skipped(self, tmp_path: Path) -> None:
        """When a bare name matches several files, guessing one is wrong.

        Prefetching the wrong `config.py` spends tokens on misleading context,
        which is worse than prefetching nothing.
        """
        for sub in ("a", "b"):
            d = tmp_path / sub
            d.mkdir()
            (d / "config.py").write_text("X = 1\n")
        ws = WorktreeWorkspace(tmp_path, use_git=False)
        assert resolve_prefetch_paths("edit config.py", (), ws) == ()

    def test_deep_file_beyond_scan_depth_is_not_resolved(self, tmp_path: Path) -> None:
        """Depth is bounded, so a deeply buried file gets no prefetch.

        Stated as a real limit rather than hidden: the task still runs, it just
        runs without inlined context, which is the pre-fix behaviour.
        """
        deep = tmp_path.joinpath(*[f"d{i}" for i in range(MAX_PREFETCH_SCAN_DEPTH + 2)])
        deep.mkdir(parents=True)
        (deep / "buried.py").write_text("x = 1\n")
        ws = WorktreeWorkspace(tmp_path, use_git=False)
        assert resolve_prefetch_paths("edit buried.py", (), ws) == ()


class TestResolutionStaysCheap:
    """FastPath's justification is being cheap, so resolution must not walk the repo.

    A git-worktree shadow holds only tracked files, so the walk is small there.
    The `copytree` fallback is not bounded that way: it retains anything not in
    its six ignore patterns, and a checkout carrying build output or a vendored
    `site-packages` measured >100k files, where a full `list_files()` rglob took
    3.3s. Resolution therefore searches a bounded neighbourhood instead.
    """

    def test_never_enumerates_the_whole_tree(self, workspace: WorktreeWorkspace) -> None:
        """The resolver must not call `list_files()`, whose cost is O(repo).

        Uses a nested file so resolution actually reaches the search step —
        a root-level name resolves by direct read and would pass either way.
        """

        def explode(pattern: str = "*") -> list[str]:
            raise AssertionError("resolve_prefetch_paths must not call list_files()")

        object.__setattr__(workspace, "list_files", explode)
        assert resolve_prefetch_paths("Set DEBUG in config.py", (), workspace) == (
            "src/app/config.py",
        )

    def test_vendored_directories_are_pruned(self, tmp_path: Path) -> None:
        """A basename colliding in build output must not make the name ambiguous.

        Both copies sit at the same depth, so depth ordering cannot break the
        tie — only pruning can. Without it the search sees two matches, calls
        the name ambiguous, and the real project file stops resolving.

        `build/` specifically: the shadow's `copytree` drops `.venv` and
        `node_modules` itself, so a test using those would pass with pruning
        removed. `build/`, `.tox/`, and `site-packages/` do reach the shadow.
        """
        real = tmp_path / "src"
        real.mkdir()
        (real / "config.py").write_text("REAL = True\n")
        vendored = tmp_path / "build"
        vendored.mkdir()
        (vendored / "config.py").write_text("VENDORED = True\n")
        ws = WorktreeWorkspace(tmp_path, use_git=False)
        assert resolve_prefetch_paths("edit config.py", (), ws) == ("src/config.py",)


class TestPathSafety:
    def test_traversal_outside_the_workspace_is_refused(self, tmp_path: Path) -> None:
        """A path escaping the workspace must not become prompt context.

        Resolution feeds the prompt, so anything it returns is sent to the
        model. The workspace's own guard decides what is reachable, rather
        than this module re-implementing containment.
        """
        root = tmp_path / "repo"
        root.mkdir()
        (root / "calc.py").write_text("x = 1\n")
        (tmp_path / "secret.py").write_text("API_KEY = 'leak'\n")
        ws = WorktreeWorkspace(root, use_git=False)

        # A sibling name that exists only outside the workspace must not
        # resolve, while the in-workspace file still does.
        got = resolve_prefetch_paths("Port secret.py logic into calc.py", (), ws)
        assert got == ("calc.py",)

    def test_absolute_path_is_not_resolved(self, tmp_path: Path) -> None:
        """An absolute path naming a real outside file stays unresolved.

        Checked at the `_is_file` layer as well as end to end: `Path(root) /
        "/abs"` discards the root and yields the absolute path, so a naive
        `is_file()` check would happily confirm a file outside the workspace.
        Containment has to come from the workspace guard.
        """
        root = tmp_path / "repo"
        root.mkdir()
        (root / "calc.py").write_text("x = 1\n")
        outside = tmp_path / "secret.py"
        outside.write_text("API_KEY = 'leak'\n")
        ws = WorktreeWorkspace(root, use_git=False)

        assert not _is_file(ws, str(outside))
        assert resolve_prefetch_paths(f"Read {outside}", (), ws) == ()


# ---------------------------------------------------------------------------
# files_hint precedence — the hint remains authoritative when present
# ---------------------------------------------------------------------------


class TestHintPrecedence:
    def test_hint_is_used_verbatim_when_present(self, workspace: WorktreeWorkspace) -> None:
        """An L1 hint is a real signal; text extraction must not second-guess it."""
        got = resolve_prefetch_paths("Add a docstring to calc.py", ("utils.py",), workspace)
        assert got == ("utils.py",)

    def test_hint_wins_even_for_nonexistent_paths(self, workspace: WorktreeWorkspace) -> None:
        """A hint naming a new file means 'create it'; that must survive.

        `_prefetch_files` already tolerates missing paths, and the hint also
        drives condition 3, so filtering it here would change enforcement.
        """
        got = resolve_prefetch_paths("Add a module", ("brand_new.py",), workspace)
        assert got == ("brand_new.py",)

    def test_empty_hint_triggers_resolution(self, workspace: WorktreeWorkspace) -> None:
        got = resolve_prefetch_paths("Fix calc.py", (), workspace)
        assert got == ("calc.py",)


# ---------------------------------------------------------------------------
# The separation that keeps this from regressing condition 3
# ---------------------------------------------------------------------------


class TestEnforcementIsUnaffected:
    def test_resolution_does_not_become_an_enforcement_contract(
        self, workspace: WorktreeWorkspace
    ) -> None:
        """Resolved paths must not be used as the condition-3 expected set.

        This is the regression this design exists to avoid. If resolution fed
        `files_hint`, a task saying "calc.py" whose correct patch also touches
        `utils.py` would fail as UNEXPECTED_MODIFICATIONS — trading a decline
        for a wrong rejection. With no hint, condition 3 stays unenforceable
        and the caller reviews the full changed set instead.
        """
        client = _patch_client({"calc.py": "# edited\n", "utils.py": "# also edited\n"})
        inp = _fastpath_input("Add a docstring to divide in calc.py", workspace)
        result = execute_fastpath(inp, workspace, client)

        assert result.success, (
            f"escalated as {result.escalation_reason}: resolution must not "
            "enforce the expected-file set"
        )
        assert result.escalation_reason != EscalationReason.UNEXPECTED_MODIFICATIONS
        assert {"calc.py", "utils.py"} <= set(result.changed_files)

    def test_real_hint_still_enforces_condition_three(
        self, workspace: WorktreeWorkspace
    ) -> None:
        """A supplied hint must keep rejecting out-of-scope edits."""
        client = _patch_client({"calc.py": "# edited\n", "utils.py": "# sneaky\n"})
        inp = _fastpath_input("Edit calc.py", workspace, hint=("calc.py",))
        result = execute_fastpath(inp, workspace, client)

        assert not result.success
        assert result.escalation_reason == EscalationReason.UNEXPECTED_MODIFICATIONS


# ---------------------------------------------------------------------------
# The end-to-end behaviour the defect describes
# ---------------------------------------------------------------------------


class TestPromptReceivesFileContents:
    def test_hintless_task_still_sees_the_file(self, workspace: WorktreeWorkspace) -> None:
        """The defect, end to end: no hint, yet the model sees calc.py."""
        client = _patch_client({"calc.py": "def divide(a, b):\n    return a / b\n"})
        execute_fastpath(
            _fastpath_input("Add a docstring to divide in calc.py", workspace),
            workspace,
            client,
        )

        sent = _prompt_text(client)
        assert "## Current file contents" in sent
        assert "### calc.py" in sent
        assert "return a / b" in sent

    def test_file_contents_are_truncated_to_budget(self, tmp_path: Path) -> None:
        """A large resolved file is capped, not inlined whole."""
        (tmp_path / "big.py").write_text("# pad\n" * MAX_PREFETCH_FILE_CHARS)
        ws = WorktreeWorkspace(tmp_path, use_git=False)
        client = _patch_client({"big.py": "x = 1\n"})
        execute_fastpath(_fastpath_input("Trim big.py", ws), ws, client)

        assert "... (truncated)" in _prompt_text(client)


# ---------------------------------------------------------------------------
# Wiring — the tests above call execute_fastpath directly, which cannot catch
# a resolver that is correct but never reached from run_task. Two defects in
# this repository were exactly that shape, so the real entry point gets its own
# test rather than being assumed.
# ---------------------------------------------------------------------------


class TestReachableFromRunTask:
    """The prompt built by a real `run_task` carries the resolved file."""

    def _run(self, repo: Path, task: str, mode: str) -> str:
        from qqcode.config import Config, ProviderConfig
        from qqcode.models.billing import RetryPolicy
        from qqcode.models.protocol import CostLedger
        from qqcode.orchestrator import run_task

        sent: list[str] = []

        class CapturingAdapter:
            def invoke(self, messages: list[object], **kwargs: object) -> Completion:
                sent.append(
                    "\n".join(
                        block.text
                        for msg in messages  # type: ignore[attr-defined]
                        for block in msg.content  # type: ignore[attr-defined]
                        if isinstance(block, TextContent)
                    )
                )
                return Completion(
                    content=[
                        ToolUseContent(
                            id="fp1",
                            name=PATCH_TOOL_NAME,
                            input={
                                "reasoning": "done",
                                "files": [{"path": "calc.py", "content": "# patched\n"}],
                            },
                        )
                    ],
                    stop_reason="tool_use",
                    usage=Usage(input_tokens=10, output_tokens=5),
                    raw={},
                )

        client = BilledClient(
            CapturingAdapter(),
            ledger=CostLedger(),
            retry_policy=RetryPolicy(max_attempts=1, sleep=lambda _: None),
        )
        config = Config(
            anthropic=ProviderConfig(api_key="fake", base_url=None),
            openai=None,
            default_provider="anthropic",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "qqcode.orchestrator.build_client",
                lambda *a, **k: (client, client._ledger),  # noqa: SLF001
            )
            run_task(task=task, repo=repo, config=config, mode=mode, dry_run=True)

        assert sent, "the model was never called"
        return sent[0]

    @pytest.fixture
    def repo(self, tmp_path: Path) -> Path:
        (tmp_path / "calc.py").write_text("def divide(a, b):\n    return a / b\n")
        return tmp_path

    def test_mode_fast_prompt_contains_the_resolved_file(self, repo: Path) -> None:
        """`mode="fast"` passes `files_hint=()` explicitly, so it must resolve."""
        sent = self._run(repo, "Add a docstring to divide in calc.py", mode="fast")

        assert "### calc.py" in sent
        assert "return a / b" in sent

    def test_l0_fast_hint_prompt_contains_the_resolved_file(self, repo: Path) -> None:
        """The exact defect path: L0's FAST skill hint routes with no hint.

        "docstring" matches the built-in python-docstrings skill, whose
        `routing_hint: fast` fires L0 and sends the task to FastPath with
        `files_hint=()` — the branch that produced every observed decline.
        """
        sent = self._run(repo, "Add a docstring to divide in calc.py", mode="auto")

        assert "### calc.py" in sent
        assert "return a / b" in sent


class TestAdvisoryHintReachesThePrompt:
    """The L0 → L1 recovery must reach the prompt, not just the router.

    Computing a value and failing to wire it is this repo's recurring defect
    shape, and a direct call to `resolve_prefetch_paths` cannot catch it. These
    run through `run_task`, the way production does.
    """

    def _run(self, repo: Path, task: str, l1_files: list[str]) -> str:
        """Answer L1 with a classification, then capture the FastPath prompt."""
        from qqcode.config import Config, ProviderConfig
        from qqcode.models.billing import RetryPolicy
        from qqcode.models.protocol import CostLedger
        from qqcode.orchestrator import run_task

        prompts: list[str] = []

        class ScriptedAdapter:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, messages: list[object], **kwargs: object) -> Completion:
                self.calls += 1
                text = "\n".join(
                    block.text
                    for msg in messages  # type: ignore[attr-defined]
                    for block in msg.content  # type: ignore[attr-defined]
                    if isinstance(block, TextContent)
                )
                # `BilledClient` consumes `phase`, so it never reaches the
                # adapter; the classifier's own system prompt identifies the call.
                if "task routing classifier" in text:
                    return Completion(
                        content=[ToolUseContent(
                            id="l1",
                            name="classify_task",
                            input={
                                "decision": "fastpath",
                                "confidence": 0.9,
                                "files": l1_files,
                                "reasoning": "simple",
                            },
                        )],
                        stop_reason="tool_use",
                        usage=Usage(input_tokens=10, output_tokens=5),
                        raw={},
                    )
                prompts.append(text)
                return Completion(
                    content=[ToolUseContent(
                        id="fp1",
                        name=PATCH_TOOL_NAME,
                        input={"reasoning": "done", "files": []},
                    )],
                    stop_reason="tool_use",
                    usage=Usage(input_tokens=10, output_tokens=5),
                    raw={},
                )

        client = BilledClient(
            ScriptedAdapter(),
            ledger=CostLedger(),
            retry_policy=RetryPolicy(max_attempts=1, sleep=lambda _: None),
        )
        config = Config(
            anthropic=ProviderConfig(api_key="fake", base_url=None),
            openai=None,
            default_provider="anthropic",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "qqcode.orchestrator.build_client",
                lambda *a, **k: (client, client._ledger),  # noqa: SLF001
            )
            run_task(task=task, repo=repo, config=config, mode="auto", dry_run=True)

        assert prompts, "FastPath was never reached"
        return prompts[0]

    @pytest.fixture
    def repo(self, tmp_path: Path) -> Path:
        (tmp_path / "saferepr.py").write_text("def saferepr(obj):\n    return repr(obj)\n")
        return tmp_path

    def test_statement_naming_no_file_still_gets_code_in_the_prompt(self, repo: Path) -> None:
        """The measured defect: a real issue report names no file at all.

        "Add a docstring" fires L0's FAST skill hint, and the statement offers no
        filename, so before this fix the prompt claimed file contents were
        provided while containing none.
        """
        sent = self._run(repo, "Add a docstring where repr is computed", ["saferepr.py"])

        assert "### saferepr.py" in sent
        assert "return repr(obj)" in sent

    def test_a_hallucinated_path_is_dropped_rather_than_inlined(self, repo: Path) -> None:
        """L1 never saw the repo, so its names are guesses that must be verified.

        Asserted on the resolved tuple, not the prompt: `_prefetch_files` already
        skips unreadable paths, so a prompt-level check passes either way and
        cannot tell a verified resolution from an unverified one.
        """
        from qqcode.routing.fastpath import resolve_prefetch_paths
        from qqcode.workspace.worktree import WorktreeWorkspace

        with WorktreeWorkspace(repo, use_git=False) as ws:
            resolved = resolve_prefetch_paths(
                "Add a docstring where repr is computed", (), ws,
                ("does_not_exist.py", "saferepr.py"),
            )

        assert resolved == ("saferepr.py",), (
            "an unverified guess must be dropped, not passed through"
        )
