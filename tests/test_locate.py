"""Tests for lexical file location.

The locator exists for one measured shape: a statement that names no path at
all. On the five derivable benchmark statements the path regex matched 0/5 and
the L1 classifier 1/4, so those runs reached the model with no code in the
prompt and declined at 23k-44k tokens each.

These tests pin the mechanisms that were measured to carry recall (IDF ranking,
filename stem affinity, test-file downweight), the cost bounds that keep the
walk affordable, and the boundary that matters most: locator output is advisory
and must never reach `files_hint`, which doubles as condition 3's enforcement
contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qqcode.config import Config, ProviderConfig
from qqcode.memory.trace import TraceRecord, TraceStore
from qqcode.models.billing import BilledClient, RetryPolicy
from qqcode.models.protocol import (
    Completion,
    CostLedger,
    TextContent,
    ToolUseContent,
    Usage,
)
from qqcode.orchestrator import run_task
from qqcode.routing.fastpath import (
    MAX_PREFETCH_RESOLVED_FILES,
    PATCH_TOOL_NAME,
    resolve_prefetch_paths,
)
from qqcode.routing.locate import (
    MAX_LOCATOR_FILE_BYTES,
    MAX_LOCATOR_FILES,
    MIN_STEM_PREFIX,
    STOP_WORDS,
    _is_test_path,
    locate_files,
    statement_tokens,
)
from qqcode.workspace.worktree import WorktreeWorkspace


def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A tree shaped like the real fixtures: source plus a mirroring test dir."""
    return _tree(
        tmp_path,
        {
            "src/pkg/saferepr.py": "def saferepr(obj):\n    return repr(obj)\n",
            "src/pkg/core.py": "def run():\n    pass\n",
            "src/pkg/utils.py": "def helper():\n    pass\n",
            "src/pkg/evaluate.py": "def evaluate(expr):\n    return eval(expr)\n",
            "testing/test_saferepr.py": (
                "from pkg.saferepr import saferepr\n"
                "def test_saferepr_handles_broken_repr():\n"
                "    saferepr(object())\n"
            ),
        },
    )


class TestTokenisation:
    def test_dunder_names_reach_their_bare_word(self) -> None:
        # `__repr__` in prose must reach code spelling it `repr`, or a statement
        # about a broken __repr__ cannot find the file implementing repr.
        #
        # Pins the outcome, not one mechanism: the underscore strip and the
        # CamelCase split both yield "repr" here, so disabling either alone
        # leaves this green. Verified by mutation. Keeping it means a change that
        # removes *both* is caught, which is the property that matters.
        assert "repr" in statement_tokens("saferepr crashes on a broken __repr__")

    def test_camel_case_is_split(self) -> None:
        tokens = statement_tokens("SafeRepr misbehaves")
        assert "safe" in tokens
        assert "repr" in tokens

    def test_short_and_closed_class_words_are_dropped(self) -> None:
        tokens = statement_tokens("Fix the bug when it is not used")
        # "fix"/"used"/"when"/"not" are stop words; "bug"/"it"/"is" are too short.
        assert tokens == ()

    def test_stop_list_carries_no_domain_vocabulary(self) -> None:
        # Leave-one-out showed a hand list containing defect words scored 60% at
        # @3 only because those words came from the measured statements. Pinning
        # their absence keeps the fitted list from creeping back in.
        fitted = {
            "regression", "missing", "twice", "respect", "incorrect",
            "caused", "whose", "across", "normal", "reference", "lets",
            "raises", "called", "error", "exception", "assertion", "test",
        }
        assert STOP_WORDS.isdisjoint(fitted)


class TestRanking:
    def test_filename_match_outranks_body_mentions(self, tmp_path: Path) -> None:
        # Both competitors are non-test files holding the token, so they earn
        # identical body IDF and the test-file downweight cannot decide this.
        # `dispatcher.py` also sorts first, so it wins every tie. Only the
        # filename bonus can put `saferepr.py` on top.
        root = _tree(
            tmp_path,
            {
                "src/pkg/saferepr.py": "def saferepr(obj):\n    return repr(obj)\n",
                "src/pkg/dispatcher.py": "from .saferepr import saferepr\n" * 50,
                "src/pkg/core.py": "def run():\n    pass\n",
                "src/pkg/utils.py": "def helper():\n    pass\n",
            },
        )
        assert locate_files("saferepr crashes on a broken repr", root, 1) == (
            "src/pkg/saferepr.py",
        )

    def test_morphological_filename_match(self, repo: Path) -> None:
        # "evaluation" never appears verbatim; `evaluate.py` shares 7 characters.
        # Substring matching alone cannot make this connection.
        found = locate_files("string condition evaluation is cached wrongly", repo, 3)
        assert "src/pkg/evaluate.py" in found

    def test_stem_floor_does_not_accept_a_four_char_overlap(self, repo: Path) -> None:
        # "skipif" vs `skipping.py` share only "skip". Accepting 4 would buy one
        # fixture at the cost of promoting a wrong file on every other statement.
        (repo / "src" / "pkg" / "skipping.py").write_text("def skip():\n    pass\n")
        assert MIN_STEM_PREFIX == 5
        found = locate_files("skipif marker is mishandled", repo, 3)
        assert "src/pkg/skipping.py" not in found

    def test_test_files_are_downweighted_not_excluded(self, repo: Path) -> None:
        # Downweight rather than drop: a test file can still be the best
        # available context. It just must not outrank real source.
        found = locate_files("saferepr broken repr", repo, 5)
        assert found[0] == "src/pkg/saferepr.py"
        assert "testing/test_saferepr.py" in found

    def test_ubiquitous_tokens_do_not_decide_the_ranking(self, tmp_path: Path) -> None:
        # Partial ubiquity is the case the cutoff exists for. A token in *every*
        # file already gets log(1) = 0 and is harmless either way; a token in 60%
        # keeps a small positive weight, and enough of those stacked on one file
        # outrank the single rare token that actually locates the task.
        #
        # The arithmetic is deliberate, because scoring is by presence and not by
        # frequency -- repeating a word in one file buys nothing. Ten files, five
        # shared words in six of them, one rare word in the target:
        #   cutoff on  -> shared words dropped (6 > 10 x 0.5); target scores
        #                 log(10/1) = 2.30 and is the only file scoring at all
        #   cutoff off -> each common file scores 5 x log(10/6) = 2.56 > 2.30,
        #                 so a decoy takes the slot the target should have
        common = "widget gadget wrapper adapter builder"
        files: dict[str, str] = {f"common{i}.py": f"# {common}\n" for i in range(6)}
        files["quiet_a.py"] = "def alpha():\n    pass\n"
        files["quiet_b.py"] = "def beta():\n    pass\n"
        files["quiet_c.py"] = "def gamma():\n    pass\n"
        # One token, not a compound: `unmistakable_symbol` would expand into
        # three tokens and score 3x, swamping the effect under measurement.
        files["target.py"] = "def flumdiddle():\n    pass\n"
        root = _tree(tmp_path, files)

        task = f"flumdiddle misbehaves around {common}"
        assert locate_files(task, root, 1) == ("target.py",)

    def test_ranking_is_deterministic(self, repo: Path) -> None:
        # A/B measurement is unreproducible if identical scores break ties by
        # dict order.
        first = locate_files("saferepr broken repr", repo, 3)
        assert first == locate_files("saferepr broken repr", repo, 3)


class TestCostBounds:
    def test_oversized_tree_returns_nothing(self, tmp_path: Path) -> None:
        # Ranking a truncated walk would present whichever subtree os.walk
        # reached first as if it were the whole repository. This repo's own tree
        # holds ~60k gitignored Python files, so this is not hypothetical.
        #
        # The files differ, and exactly one carries the task's token. Identical
        # files would put that token in 100% of the tree, where log(1) = 0 makes
        # everything score zero -- the assertion would then hold whether or not
        # the ceiling fired, which pins nothing.
        files = {f"gen/f{i}.py": f"def unrelated_{i}(): pass\n"
                 for i in range(MAX_LOCATOR_FILES + 2)}
        files["gen/f0.py"] = "def specific_handler(): pass\n"
        root = _tree(tmp_path, files)

        assert locate_files("specific_handler misbehaves", root, 3) == ()

    def test_excluded_directories_are_not_ranked(self, tmp_path: Path) -> None:
        root = _tree(
            tmp_path,
            {
                "app.py": "def handler():\n    pass\n",
                ".venv/lib/vendored.py": "def specific_handler():\n    pass\n" * 20,
                "node_modules/dep/index.js": "function specificHandler() {}\n",
            },
        )
        found = locate_files("specific_handler misbehaves", root, 5)
        assert all(".venv" not in p and "node_modules" not in p for p in found)

    def test_huge_file_is_read_but_truncated(self, tmp_path: Path) -> None:
        # Truncation is fine -- ranking needs which words appear, not all of
        # them -- but the file must still be considered.
        padding = "# pad\n" * (MAX_LOCATOR_FILE_BYTES // 6)
        root = _tree(
            tmp_path,
            {
                "big.py": "def specific_handler():\n    pass\n" + padding,
                "other.py": "def unrelated():\n    pass\n",
            },
        )
        assert locate_files("specific_handler misbehaves", root, 1) == ("big.py",)

    def test_unreadable_entry_does_not_abort_the_walk(self, tmp_path: Path) -> None:
        # Several decoy files, not one: IDF is a ratio over the tree, so at
        # n_total=1 every token sits in 100% of files, the ubiquity cutoff
        # discards all of them, and log(1/1) is 0 regardless. A one-file tree
        # would make this pass or fail for reasons unrelated to the symlink.
        root = _tree(
            tmp_path,
            {
                "good.py": "def specific_handler():\n    pass\n",
                "a.py": "def alpha():\n    pass\n",
                "b.py": "def beta():\n    pass\n",
                "c.py": "def gamma():\n    pass\n",
            },
        )
        (root / "broken.py").symlink_to(root / "does_not_exist.py")
        assert locate_files("specific_handler misbehaves", root, 1) == ("good.py",)


class TestNoUsableSignal:
    def test_task_without_tokens_returns_nothing(self, repo: Path) -> None:
        assert locate_files("fix the bug", repo, 3) == ()

    def test_zero_limit_skips_the_walk(self, repo: Path) -> None:
        assert locate_files("saferepr broken repr", repo, 0) == ()

    def test_empty_tree_returns_nothing(self, tmp_path: Path) -> None:
        assert locate_files("saferepr broken repr", tmp_path, 3) == ()


class TestTestPathDetection:
    @pytest.mark.parametrize(
        "rel",
        [
            "tests/test_x.py",
            "testing/io/test_saferepr.py",
            "src/pkg/x_test.py",
            "src/test/helper.py",
        ],
    )
    def test_recognised(self, rel: str) -> None:
        assert _is_test_path(rel)

    @pytest.mark.parametrize(
        "rel",
        ["src/pkg/saferepr.py", "src/pkg/latest.py", "contest/main.py"],
    )
    def test_not_recognised(self, rel: str) -> None:
        # "latest" and "contest" contain "test" as a substring. Matching on that
        # would downweight ordinary source files.
        assert not _is_test_path(rel)


class TestWiring:
    """The locator must actually be reachable from prefetch resolution.

    The recurring defect shape in this repository is a mechanism that is written
    and then not connected, so these assert the seam rather than the algorithm.
    """

    def test_pathless_task_now_resolves_files(self, repo: Path) -> None:
        ws = WorktreeWorkspace(repo, use_git=False)
        # No files_hint, no path token in the text, no prefetch_hint: precisely
        # the configuration measured to decline 4/4.
        resolved = resolve_prefetch_paths(
            "saferepr crashes on a broken __repr__", (), ws, ()
        )
        assert resolved, "locator not reached from resolve_prefetch_paths"
        assert "src/pkg/saferepr.py" in resolved

    def test_advisory_hint_wins_over_the_locator(self, repo: Path) -> None:
        # The hint came from a model that read the task; the locator counts
        # words. When the hint names a real file it is the better evidence.
        ws = WorktreeWorkspace(repo, use_git=False)
        resolved = resolve_prefetch_paths(
            "saferepr crashes on a broken __repr__", (), ws, ("src/pkg/core.py",)
        )
        assert resolved == ("src/pkg/core.py",)

    def test_files_hint_is_never_replaced_by_the_locator(self, repo: Path) -> None:
        # files_hint is condition 3's contract. Widening it here would reject a
        # correct patch instead of merely wasting tokens.
        ws = WorktreeWorkspace(repo, use_git=False)
        resolved = resolve_prefetch_paths(
            "saferepr crashes on a broken __repr__", ("src/pkg/core.py",), ws, ()
        )
        assert resolved == ("src/pkg/core.py",)

    def test_hint_naming_a_new_file_still_survives(self, repo: Path) -> None:
        # A hint naming a nonexistent path means "create it". The locator must
        # not displace it, or a creation task silently becomes an edit task.
        ws = WorktreeWorkspace(repo, use_git=False)
        resolved = resolve_prefetch_paths("add a module", ("src/pkg/new.py",), ws, ())
        assert resolved == ("src/pkg/new.py",)

    def test_result_respects_the_prefetch_cap(self, repo: Path) -> None:
        ws = WorktreeWorkspace(repo, use_git=False)
        resolved = resolve_prefetch_paths(
            "saferepr evaluate core utils helper broken repr", (), ws, ()
        )
        assert len(resolved) <= MAX_PREFETCH_RESOLVED_FILES

    def test_located_paths_are_readable_in_the_workspace(self, repo: Path) -> None:
        # Containment goes through the workspace guard like every other prefetch
        # path, so anything returned here must actually be readable.
        #
        # Honest limit: mutation testing showed dropping the `_is_file` filter in
        # `_resolve_advisory_or_locate` does not fail this test, and no fixture
        # can make it fail. The locator only walks paths under the workspace
        # root, so everything it returns is already inside the guard's allowlist.
        # The filter is defence-in-depth against a future locator that takes
        # paths from elsewhere; this test pins readability, not the filter.
        ws = WorktreeWorkspace(repo, use_git=False)
        resolved = resolve_prefetch_paths(
            "saferepr crashes on a broken __repr__", (), ws, ()
        )
        for rel in resolved:
            assert ws.read_file(rel)


class TestLocatorReachesThePromptFromRunTask:
    """The located file's *contents* must reach the model, via `run_task`.

    `TestWiring` above calls `resolve_prefetch_paths` directly, so it proves the
    resolver returns a path -- not that the path is read, inlined, and sent. Two
    defects in this repository were exactly that gap, and the locator's own
    measured value (60% @3 on the derivable statements) is worth nothing if the
    text never lands in the prompt.

    The shape reproduced here is the one the traces recorded for every observed
    decline: `route_layer="l0"`, `files_hint=()`, and an L1 that supplies no
    usable filename -- leaving the locator as the only possible source.
    """

    @staticmethod
    def _run(repo: Path, task: str, l1_files: list[str]) -> tuple[str, TraceRecord]:
        """Run `auto` end to end; return the FastPath prompt and its trace row.

        The trace row is returned so the test can prove *which* route it
        exercised. Asserting only on prompt text would pass if a future change
        sent this task down the L1 route with a real hint, and the locator seam
        would go untested while the test still looked green.
        """
        prompts: list[str] = []

        class ScriptedAdapter:
            def invoke(self, messages: list[object], **kwargs: object) -> Completion:
                text = "\n".join(
                    block.text
                    for msg in messages  # type: ignore[attr-defined]
                    for block in msg.content  # type: ignore[attr-defined]
                    if isinstance(block, TextContent)
                )
                # `BilledClient` consumes `phase`, so the classifier is
                # identified by its own system prompt, as in test_prefetch.py.
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
        with TraceStore(repo / ".qqcode" / "trace.db") as store:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(
                    "qqcode.orchestrator.build_client",
                    lambda *a, **k: (client, client._ledger),  # noqa: SLF001
                )
                run_task(
                    task=task,
                    repo=repo,
                    config=config,
                    mode="auto",
                    dry_run=True,
                    trace_store=store,
                )
            rows = store.all()

        assert prompts, "FastPath was never reached"
        assert len(rows) == 1, f"expected one trace row, got {len(rows)}"
        return prompts[0], rows[0]

    def test_located_file_contents_reach_the_prompt(self, repo: Path) -> None:
        """The measured decline shape, end to end, with the locator as sole source.

        "docstring" fires the built-in python-docstrings skill's
        `routing_hint: fast`, so L0 decides FastPath with `files_hint=()`. The
        statement names no path -- `_PATH_TOKEN` needs a known extension, and
        there is no dotted token here -- and L1 returns no filename, so
        `prefetch_hint` is empty too. Nothing but the locator can put code in
        this prompt.
        """
        sent, row = self._run(repo, "Add a docstring where a broken repr is handled", [])

        # Prove the route before trusting the prompt: this must be the hintless
        # L0 entry that the traces show declining 4/4, not some other path.
        assert row.route_layer == "l0"
        assert row.files_hint_count == 0
        assert row.prefetch_hint_count == 0

        # The seam under test: name, section header, and body all present.
        assert "### src/pkg/saferepr.py" in sent
        assert "return repr(obj)" in sent

    def test_locator_fills_in_when_the_advisory_hint_is_hallucinated(
        self, repo: Path
    ) -> None:
        """A hint naming a nonexistent file must not suppress the locator.

        L1 never saw the repository, so its guesses can name files that do not
        exist. `_resolve_advisory` drops those, and the fallthrough to the
        locator is what keeps the prompt from being empty -- the difference
        between a 60%-recall guess and no code at all.
        """
        sent, row = self._run(
            repo, "Add a docstring where a broken repr is handled", ["src/pkg/ghost.py"]
        )

        assert row.route_layer == "l0"
        assert "### src/pkg/ghost.py" not in sent, "an unverified guess was inlined"
        assert "### src/pkg/saferepr.py" in sent
        assert "return repr(obj)" in sent
