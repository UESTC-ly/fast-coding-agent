"""Measure FastPath vs Full Agent vs escalation on trivial, self-contained tasks.

Why this exists alongside `qqcode_benchmark.py`: the SWE-bench fixtures there
carry hidden tests that assert the *specific* upstream implementation, so a
correct-but-different fix scores zero and the resulting rate says as much about
fixture authoring as about the agent. These tasks are trivial and their hidden
tests assert only behaviour the statement actually states, which is what makes a
FastPath-vs-FullAgent token comparison meaningful.

Every task runs twice:
  full  — Full Agent forced. The baseline: what the suite costs with no routing.
  auto  — routing on. FastPath handles what it can; the rest escalate.

Savings are then reported per task and as a suite total, which is the figure that
answers "what does routing actually save".

A deliberately excluded fourth mode: `fast` with an empty `files_hint`. It looks
cheapest of all, but only because the shorter prompt fails to trigger the
model's extended thinking — a measurement artifact of a path no real user takes,
not a saving. Comparing `full` against `auto` keeps both sides on real code
paths, so thinking tokens are each path's true cost.

Two counters, because neither alone is comparable across modes:
  model_calls — every provider call (routing + FastPath + Full Agent turns).
      The only metric directly comparable between modes.
  turns_used  — Full Agent tool-loop iterations. FastPath is a single shot with
      no loop, so it reports 0; a FastPath run costs 1 model call and 0 turns.

Usage:
    python benchmarks/trivial_paths_benchmark.py --model gpt-5.6-terra
    python benchmarks/trivial_paths_benchmark.py --model gpt-5.6-terra --tasks fix-subtract
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qqcode import orchestrator  # noqa: E402
from qqcode.acceptance import AcceptanceHarness, AcceptanceTest  # noqa: E402
from qqcode.config import Config  # noqa: E402
from qqcode.orchestrator import run_task  # noqa: E402

# Captured before any patching so the sniffer wrapper can delegate to the real
# constructor rather than recursing into itself.
_real_build_client = orchestrator.build_client

REPO_ROOT = Path(__file__).resolve().parent.parent

# Order matters: `full` establishes the baseline before `auto` is measured
# against it.
MODES = ("full", "auto")


@dataclass(frozen=True)
class TrivialTask:
    """A self-contained task whose hidden test asserts only the stated behaviour."""

    task_id: str
    statement: str
    files: dict[str, str]
    test_name: str
    test_body: str


@dataclass
class PathRun:
    """One (model, task, mode) result."""

    task_id: str
    model: str
    mode: str
    success: bool
    behavioral_pass: bool
    mode_used: str
    finish_reason: str
    turns_used: int      # Full Agent tool-loop iterations; 0 for FastPath
    model_calls: int     # every provider call — comparable across modes
    tokens_total: int
    tokens_routing: int
    tokens_fastpath: int
    tokens_fullagent: int
    duration_s: float
    fastpath_attempted: bool = False
    fastpath_success: bool = False
    escalated: bool = False
    error: str | None = None
    acceptance_output: str = ""
    incident: str | None = None


def _git_init(repo: Path) -> None:
    """A git repo is required: the shadow workspace is a git worktree."""
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=bench@local", "-c", "user.name=bench",
         "commit", "-qm", "baseline"],
        cwd=repo, check=True,
    )


def _harness(task: TrivialTask, python: str) -> AcceptanceHarness:
    """Hidden test, run from the workspace root so imports resolve."""
    return AcceptanceHarness([
        AcceptanceTest(
            name=task.task_id,
            files={task.test_name: task.test_body},
            command=(python, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                     f".qqcode_acceptance/{task.test_name}"),
            timeout=120.0,
        )
    ])


# Provider/gateway faults say nothing about agent capability; excluded from rates.
_PROVIDER_FAULT_MARKERS = (
    "No tool output found for function call", "Bad gateway", "origin_bad_gateway",
    "Error code: 5", "overloaded", "rate limit", "Error code: 429",
    "service unavailable", "InternalServerError",
)

# Some gateways answer a normal request with a refusal in the ASSISTANT CONTENT
# instead of an HTTP error. It carries no tool call, so the agent loop counts it
# as an empty turn and terminates `stuck` — indistinguishable from a real agent
# failure unless we look at the text. Observed on the Anthropic-compatible
# gateway during this suite's development.
_INJECTION_MARKERS = (
    "access denied",
    "restricted to authorized use",
    "official claude code client",
)


def _looks_injected(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _INJECTION_MARKERS)


class _InjectionSniffer:
    """Adapter wrapper that flags gateway refusals delivered as normal content.

    Detection has to happen at the wire boundary: `RunResult` carries no
    transcript, so by the time the benchmark sees a `stuck` run the evidence is
    gone. Wrapping here keeps the workaround in the measurement layer rather
    than teaching the agent graph about one gateway's quirk.
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self.injected = False

    def model_for(self, tier: Any) -> str:
        return self._inner.model_for(tier)

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        completion = self._inner.invoke(messages, **kwargs)
        has_tool_call = any(
            type(b).__name__ == "ToolUseContent" for b in completion.content
        )
        if not has_tool_call:
            text = " ".join(
                getattr(b, "text", "") for b in completion.content
            )
            if _looks_injected(text):
                self.injected = True
        return completion


def _classify_incident(err: str | None) -> str | None:
    if not err:
        return None
    low = err.lower()
    if any(m.lower() in low for m in _PROVIDER_FAULT_MARKERS):
        return "provider"
    if any(m in low for m in ("could not resolve", "connection refused", "timed out")):
        return "network"
    return None


def _provider_for(model: str) -> str:
    """Anthropic model ids are the `claude-*` family; everything else is OpenAI."""
    return "anthropic" if model.startswith("claude") else "openai"


# ---------------------------------------------------------------------------
# Tasks
#
# Each statement fully determines the assertion. Where a name or message is
# checked, the statement gives it verbatim — the failure mode found in the
# SWE-bench fixtures was tests asserting strings the statement never mentioned.
# ---------------------------------------------------------------------------

TASKS: list[TrivialTask] = [
    TrivialTask(
        task_id="fix-subtract",
        statement="The add() function in calc.py subtracts instead of adding. Fix it.",
        files={"calc.py": "def add(a, b):\n    return a - b\n"},
        test_name="test_add.py",
        test_body=(
            "from calc import add\n\n\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n"
            "    assert add(-1, 1) == 0\n"
        ),
    ),
    TrivialTask(
        task_id="off-by-one",
        statement=(
            "last_index(items) in indexing.py returns an index one past the end. "
            "It must return the index of the final element, and -1 for an empty list."
        ),
        files={"indexing.py": "def last_index(items):\n    return len(items)\n"},
        test_name="test_indexing.py",
        test_body=(
            "from indexing import last_index\n\n\n"
            "def test_last_index():\n"
            "    assert last_index([1, 2, 3]) == 2\n"
            "    assert last_index(['a']) == 0\n"
            "    assert last_index([]) == -1\n"
        ),
    ),
    TrivialTask(
        task_id="guard-zero-division",
        statement=(
            "divide(a, b) in mathutil.py crashes when b is 0. It must raise "
            "ValueError with the message 'division by zero' instead."
        ),
        files={"mathutil.py": "def divide(a, b):\n    return a / b\n"},
        test_name="test_mathutil.py",
        test_body=(
            "import pytest\n\n"
            "from mathutil import divide\n\n\n"
            "def test_divide_ok():\n"
            "    assert divide(6, 3) == 2\n\n\n"
            "def test_divide_by_zero():\n"
            "    with pytest.raises(ValueError, match='division by zero'):\n"
            "        divide(1, 0)\n"
        ),
    ),
    TrivialTask(
        task_id="strip-whitespace",
        statement=(
            "normalize(name) in names.py must strip leading and trailing "
            "whitespace and collapse any run of internal whitespace to a single "
            "space. It currently only lowercases."
        ),
        files={"names.py": "def normalize(name):\n    return name.lower()\n"},
        test_name="test_names.py",
        test_body=(
            "from names import normalize\n\n\n"
            "def test_normalize():\n"
            "    assert normalize('  Ada   Lovelace  ') == 'ada lovelace'\n"
            "    assert normalize('Alan\\tTuring') == 'alan turing'\n"
            "    assert normalize('grace') == 'grace'\n"
        ),
    ),
    TrivialTask(
        task_id="empty-default",
        statement=(
            "count_words(text) in wordcount.py raises AttributeError when text "
            "is None. It must return 0 for None and for an empty string."
        ),
        files={"wordcount.py": "def count_words(text):\n    return len(text.split())\n"},
        test_name="test_wordcount.py",
        test_body=(
            "from wordcount import count_words\n\n\n"
            "def test_count_words():\n"
            "    assert count_words('one two three') == 3\n"
            "    assert count_words('') == 0\n"
            "    assert count_words(None) == 0\n"
        ),
    ),
    TrivialTask(
        task_id="inclusive-range",
        statement=(
            "in_range(value, low, high) in ranges.py excludes the endpoints. "
            "Make the bounds inclusive on both ends."
        ),
        files={"ranges.py": "def in_range(value, low, high):\n    return low < value < high\n"},
        test_name="test_ranges.py",
        test_body=(
            "from ranges import in_range\n\n\n"
            "def test_in_range():\n"
            "    assert in_range(5, 1, 10) is True\n"
            "    assert in_range(1, 1, 10) is True\n"
            "    assert in_range(10, 1, 10) is True\n"
            "    assert in_range(0, 1, 10) is False\n"
            "    assert in_range(11, 1, 10) is False\n"
        ),
    ),
    TrivialTask(
        task_id="dedupe-preserve-order",
        statement=(
            "dedupe(items) in dedupe.py uses set() and loses the original "
            "order. It must keep the first occurrence of each item in order."
        ),
        files={"dedupe.py": "def dedupe(items):\n    return list(set(items))\n"},
        test_name="test_dedupe.py",
        test_body=(
            "from dedupe import dedupe\n\n\n"
            "def test_dedupe():\n"
            "    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]\n"
            "    assert dedupe([]) == []\n"
            "    assert dedupe(['b', 'a', 'b']) == ['b', 'a']\n"
        ),
    ),
    TrivialTask(
        task_id="celsius-conversion",
        statement=(
            "to_fahrenheit(c) in temperature.py uses the wrong formula. The "
            "correct conversion is c * 9 / 5 + 32."
        ),
        files={"temperature.py": "def to_fahrenheit(c):\n    return c * 9 / 5\n"},
        test_name="test_temperature.py",
        test_body=(
            "from temperature import to_fahrenheit\n\n\n"
            "def test_to_fahrenheit():\n"
            "    assert to_fahrenheit(0) == 32\n"
            "    assert to_fahrenheit(100) == 212\n"
            "    assert to_fahrenheit(-40) == -40\n"
        ),
    ),
]

# Harder tasks: multi-file edits, cross-file consistency, and stateful logic.
# These exist to find where FastPath's single shot stops being enough — the
# escalation path cannot be measured on tasks FastPath always wins.
TASKS += [
    TrivialTask(
        task_id="multi-file-rename",
        statement=(
            "Rename the function fetch_user to get_user in user_api.py, and "
            "update every caller in handlers.py and report.py to match."
        ),
        files={
            "user_api.py": (
                "def fetch_user(uid):\n"
                '    """Return a user record."""\n'
                "    return {'id': uid, 'name': f'user{uid}'}\n"
            ),
            "handlers.py": (
                "from user_api import fetch_user\n\n\n"
                "def handle(uid):\n"
                "    return fetch_user(uid)['name']\n"
            ),
            "report.py": (
                "from user_api import fetch_user\n\n\n"
                "def summary(uids):\n"
                "    return [fetch_user(u)['id'] for u in uids]\n"
            ),
        },
        test_name="test_rename.py",
        test_body=(
            "import user_api\nimport handlers\nimport report\n\n\n"
            "def test_renamed():\n"
            "    assert hasattr(user_api, 'get_user')\n"
            "    assert not hasattr(user_api, 'fetch_user')\n\n\n"
            "def test_callers_still_work():\n"
            "    assert handlers.handle(7) == 'user7'\n"
            "    assert report.summary([1, 2]) == [1, 2]\n"
        ),
    ),
    TrivialTask(
        task_id="shared-constant",
        statement=(
            "MAX_RETRIES is duplicated as a literal 3 in client.py and worker.py. "
            "Move it into a new module config.py as MAX_RETRIES and import it in "
            "both, changing the value to 5."
        ),
        files={
            "client.py": (
                "def send(payload):\n"
                "    for attempt in range(3):\n"
                "        if payload:\n"
                "            return attempt\n"
                "    return -1\n"
            ),
            "worker.py": (
                "def process(job):\n"
                "    tries = 3\n"
                "    return tries if job else 0\n"
            ),
        },
        test_name="test_shared_constant.py",
        test_body=(
            "import config\nimport client\nimport worker\n\n\n"
            "def test_constant_centralised():\n"
            "    assert config.MAX_RETRIES == 5\n\n\n"
            "def test_both_modules_use_it():\n"
            "    import inspect\n"
            "    for mod in (client, worker):\n"
            "        src = inspect.getsource(mod)\n"
            "        assert 'MAX_RETRIES' in src\n"
            "        assert '3' not in src\n"
        ),
    ),
    TrivialTask(
        task_id="stateful-counter",
        statement=(
            "Counter in counter.py shares its count between instances because "
            "count is a class attribute. Make each instance independent, keeping "
            "the increment() and value() methods."
        ),
        files={
            "counter.py": (
                "class Counter:\n"
                "    count = 0\n\n"
                "    def increment(self):\n"
                "        Counter.count += 1\n\n"
                "    def value(self):\n"
                "        return Counter.count\n"
            )
        },
        test_name="test_counter.py",
        test_body=(
            "from counter import Counter\n\n\n"
            "def test_instances_are_independent():\n"
            "    a, b = Counter(), Counter()\n"
            "    a.increment()\n"
            "    a.increment()\n"
            "    b.increment()\n"
            "    assert a.value() == 2\n"
            "    assert b.value() == 1\n"
        ),
    ),
    TrivialTask(
        task_id="recursive-flatten",
        statement=(
            "flatten(items) in flatten.py only handles one level of nesting. It "
            "must flatten arbitrarily deep nested lists into a single flat list, "
            "preserving order."
        ),
        files={
            "flatten.py": (
                "def flatten(items):\n"
                "    out = []\n"
                "    for i in items:\n"
                "        if isinstance(i, list):\n"
                "            out.extend(i)\n"
                "        else:\n"
                "            out.append(i)\n"
                "    return out\n"
            )
        },
        test_name="test_flatten.py",
        test_body=(
            "from flatten import flatten\n\n\n"
            "def test_flatten_deep():\n"
            "    assert flatten([1, [2, [3, [4]]], 5]) == [1, 2, 3, 4, 5]\n"
            "    assert flatten([]) == []\n"
            "    assert flatten([[[['x']]]]) == ['x']\n"
            "    assert flatten([1, 2, 3]) == [1, 2, 3]\n"
        ),
    ),
    TrivialTask(
        task_id="preserve-docstring",
        statement=(
            "parse_port(text) in netconf.py must return an int instead of a str, "
            "and must raise ValueError with the message 'invalid port' when the "
            "text is not a number or is outside 1-65535. Keep its docstring."
        ),
        files={
            "netconf.py": (
                "def parse_port(text):\n"
                '    """Parse a port from configuration text."""\n'
                "    return text.strip()\n"
            )
        },
        test_name="test_netconf.py",
        test_body=(
            "import pytest\n\n"
            "from netconf import parse_port\n\n\n"
            "def test_parses_int():\n"
            "    assert parse_port(' 8080 ') == 8080\n\n\n"
            "def test_rejects_bad_input():\n"
            "    for bad in ('abc', '0', '70000'):\n"
            "        with pytest.raises(ValueError, match='invalid port'):\n"
            "            parse_port(bad)\n\n\n"
            "def test_docstring_kept():\n"
            "    assert parse_port.__doc__\n"
            "    assert 'port' in parse_port.__doc__.lower()\n"
        ),
    ),
    TrivialTask(
        task_id="sort-stability",
        statement=(
            "rank(records) in ranking.py must sort by score descending, and for "
            "equal scores keep the original input order. It currently sorts "
            "ascending and loses ties."
        ),
        files={
            "ranking.py": (
                "def rank(records):\n"
                "    return sorted(records, key=lambda r: r['score'])\n"
            )
        },
        test_name="test_ranking.py",
        test_body=(
            "from ranking import rank\n\n\n"
            "def test_rank_desc_stable():\n"
            "    recs = [\n"
            "        {'name': 'a', 'score': 5},\n"
            "        {'name': 'b', 'score': 9},\n"
            "        {'name': 'c', 'score': 5},\n"
            "        {'name': 'd', 'score': 9},\n"
            "    ]\n"
            "    assert [r['name'] for r in rank(recs)] == ['b', 'd', 'a', 'c']\n"
        ),
    ),
    TrivialTask(
        task_id="context-manager",
        statement=(
            "Tracker in tracker.py must work as a context manager: entering "
            "returns the instance and exiting sets its .closed attribute to True. "
            "It must still release on an exception."
        ),
        files={
            "tracker.py": (
                "class Tracker:\n"
                "    def __init__(self):\n"
                "        self.closed = False\n"
            )
        },
        test_name="test_tracker.py",
        test_body=(
            "import pytest\n\n"
            "from tracker import Tracker\n\n\n"
            "def test_context_manager():\n"
            "    with Tracker() as t:\n"
            "        assert t.closed is False\n"
            "    assert t.closed is True\n\n\n"
            "def test_closes_on_exception():\n"
            "    t = Tracker()\n"
            "    with pytest.raises(RuntimeError):\n"
            "        with t:\n"
            "            raise RuntimeError('boom')\n"
            "    assert t.closed is True\n"
        ),
    ),
    TrivialTask(
        task_id="merge-dicts-deep",
        statement=(
            "merge(a, b) in merging.py overwrites nested dicts wholesale. It must "
            "merge them recursively, with b winning on conflicting leaf values, "
            "and must not mutate either input."
        ),
        files={
            "merging.py": (
                "def merge(a, b):\n"
                "    out = dict(a)\n"
                "    out.update(b)\n"
                "    return out\n"
            )
        },
        test_name="test_merging.py",
        test_body=(
            "from merging import merge\n\n\n"
            "def test_deep_merge():\n"
            "    a = {'x': {'p': 1, 'q': 2}, 'y': 3}\n"
            "    b = {'x': {'q': 99, 'r': 4}}\n"
            "    assert merge(a, b) == {'x': {'p': 1, 'q': 99, 'r': 4}, 'y': 3}\n\n\n"
            "def test_inputs_not_mutated():\n"
            "    a = {'x': {'p': 1}}\n"
            "    b = {'x': {'p': 2}}\n"
            "    merge(a, b)\n"
            "    assert a == {'x': {'p': 1}}\n"
            "    assert b == {'x': {'p': 2}}\n"
        ),
    ),
]

# Idiom traps: a plausible-looking fix fails the hidden test. These probe whether
# the agent verifies its change rather than pattern-matching the statement.
TASKS += [
    TrivialTask(
        task_id="mutable-default",
        statement=(
            "add_item() shares one list across calls because of its default argument. "
            "Fix it so calling add_item('a') twice returns ['a'] each time, while "
            "passing an explicit list still appends to that list."
        ),
        files={"store.py": (
            '"""Item collection."""\n\n\n'
            "def add_item(item, bucket=[]):\n"
            "    bucket.append(item)\n"
            "    return bucket\n"
        )},
        test_name="test_store.py",
        test_body=(
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from store import add_item\n\n\n"
            "def test_default_is_fresh_each_call():\n"
            "    assert add_item('a') == ['a']\n"
            "    assert add_item('a') == ['a']\n\n\n"
            "def test_explicit_bucket_still_appends():\n"
            "    mine = ['x']\n"
            "    assert add_item('y', mine) == ['x', 'y']\n"
            "    assert mine == ['x', 'y']\n"
        ),
    ),
    TrivialTask(
        task_id="truncate-suffix",
        statement=(
            "truncate(text, limit) must return a string whose total length is at most "
            "limit. When it shortens the text it appends '...', and the '...' counts "
            "toward the limit. Text already within the limit is returned unchanged."
        ),
        files={"text_utils.py": (
            '"""Text helpers."""\n\n\n'
            "def truncate(text, limit):\n"
            "    if len(text) <= limit:\n"
            "        return text\n"
            "    return text[:limit] + '...'\n"
        )},
        test_name="test_text_utils.py",
        test_body=(
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from text_utils import truncate\n\n\n"
            "def test_total_length_respects_limit():\n"
            "    out = truncate('abcdefghij', 8)\n"
            "    assert len(out) <= 8\n"
            "    assert out.endswith('...')\n\n\n"
            "def test_short_text_untouched():\n"
            "    assert truncate('abc', 8) == 'abc'\n"
        ),
    ),
    TrivialTask(
        task_id="retry-with-backoff",
        statement=(
            "call_with_retry(fn, attempts) retries fn until it stops raising. It must "
            "not sleep after the final attempt, and it must re-raise the last "
            "exception when every attempt fails. Record each sleep by appending the "
            "delay to the module-level SLEEPS list instead of really sleeping."
        ),
        files={"retry.py": (
            '"""Retry helper."""\n\n'
            "SLEEPS = []\n\n\n"
            "def call_with_retry(fn, attempts):\n"
            "    last = None\n"
            "    for i in range(attempts):\n"
            "        try:\n"
            "            return fn()\n"
            "        except Exception as exc:\n"
            "            last = exc\n"
            "            SLEEPS.append(2 ** i)\n"
            "    return None\n"
        )},
        test_name="test_retry.py",
        test_body=(
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "import pytest\n"
            "import retry\n\n\n"
            "def test_no_sleep_after_final_attempt():\n"
            "    retry.SLEEPS.clear()\n\n"
            "    def boom():\n"
            "        raise ValueError('no')\n\n"
            "    with pytest.raises(ValueError):\n"
            "        retry.call_with_retry(boom, 3)\n"
            "    assert len(retry.SLEEPS) == 2\n\n\n"
            "def test_success_returns_value():\n"
            "    retry.SLEEPS.clear()\n"
            "    assert retry.call_with_retry(lambda: 7, 3) == 7\n"
            "    assert retry.SLEEPS == []\n"
        ),
    ),
    TrivialTask(
        task_id="cross-module-flag",
        statement=(
            "Add a module flags.py exposing FEATURE_FAST = True. Make both "
            "reader.py:load() and writer.py:store() import that flag and return the "
            "string 'fast' when it is true, 'slow' otherwise. Neither file may define "
            "its own copy of the flag."
        ),
        files={
            "reader.py": (
                '"""Read side."""\n\n\n'
                "def load():\n"
                "    return 'slow'\n"
            ),
            "writer.py": (
                '"""Write side."""\n\n\n'
                "def store():\n"
                "    return 'slow'\n"
            ),
        },
        test_name="test_flag.py",
        test_body=(
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "import flags\n"
            "from reader import load\n"
            "from writer import store\n\n\n"
            "def test_both_read_the_shared_flag():\n"
            "    assert flags.FEATURE_FAST is True\n"
            "    assert load() == 'fast'\n"
            "    assert store() == 'fast'\n\n\n"
            "def test_flipping_the_flag_changes_both():\n"
            "    flags.FEATURE_FAST = False\n"
            "    try:\n"
            "        assert load() == 'slow'\n"
            "        assert store() == 'slow'\n"
            "    finally:\n"
            "        flags.FEATURE_FAST = True\n"
        ),
    ),
    TrivialTask(
        task_id="validate-then-store",
        statement=(
            "Registry.add(name) must reject an empty or whitespace-only name by "
            "raising ValueError('name must not be blank'), and must leave the registry "
            "unchanged when it rejects. Valid names are stored stripped."
        ),
        files={"registry.py": (
            '"""Name registry."""\n\n\n'
            "class Registry:\n"
            "    def __init__(self):\n"
            "        self.names = []\n\n"
            "    def add(self, name):\n"
            "        self.names.append(name)\n"
            "        if not name:\n"
            "            raise ValueError('name must not be blank')\n"
            "        return name\n"
        )},
        test_name="test_registry.py",
        test_body=(
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "import pytest\n"
            "from registry import Registry\n\n\n"
            "def test_blank_leaves_registry_untouched():\n"
            "    r = Registry()\n"
            "    with pytest.raises(ValueError):\n"
            "        r.add('   ')\n"
            "    assert r.names == []\n\n\n"
            "def test_valid_name_stored_stripped():\n"
            "    r = Registry()\n"
            "    r.add('  ada  ')\n"
            "    assert r.names == ['ada']\n"
        ),
    ),
    TrivialTask(
        task_id="none-vs-falsy",
        statement=(
            "describe(value) must return 'missing' only when value is None. Zero, an "
            "empty string, and an empty list are present values and must return "
            "'present'."
        ),
        files={"describe.py": (
            '"""Value description."""\n\n\n'
            "def describe(value):\n"
            "    if not value:\n"
            "        return 'missing'\n"
            "    return 'present'\n"
        )},
        test_name="test_describe.py",
        test_body=(
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from describe import describe\n\n\n"
            "def test_none_is_missing():\n"
            "    assert describe(None) == 'missing'\n\n\n"
            "def test_falsy_values_are_present():\n"
            "    assert describe(0) == 'present'\n"
            "    assert describe('') == 'present'\n"
            "    assert describe([]) == 'present'\n"
        ),
    ),
    TrivialTask(
        task_id="case-insensitive-dedupe",
        statement=(
            "unique_tags(tags) must drop duplicates that differ only in case, keeping "
            "the first spelling it saw, in original order."
        ),
        files={"tags.py": (
            '"""Tag helpers."""\n\n\n'
            "def unique_tags(tags):\n"
            "    return sorted(set(t.lower() for t in tags))\n"
        )},
        test_name="test_tags.py",
        test_body=(
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from tags import unique_tags\n\n\n"
            "def test_keeps_first_spelling_in_order():\n"
            "    assert unique_tags(['Beta', 'beta', 'Alpha']) == ['Beta', 'Alpha']\n\n\n"
            "def test_no_duplicates_left():\n"
            "    assert unique_tags(['x', 'X', 'x']) == ['x']\n"
        ),
    ),
    TrivialTask(
        task_id="counter-reset-isolation",
        statement=(
            "Tally.reset() must clear only the instance it is called on. Two Tally "
            "objects must not share counts."
        ),
        files={"tally.py": (
            '"""Counting."""\n\n\n'
            "class Tally:\n"
            "    counts = {}\n\n"
            "    def bump(self, key):\n"
            "        self.counts[key] = self.counts.get(key, 0) + 1\n\n"
            "    def reset(self):\n"
            "        Tally.counts = {}\n"
        )},
        test_name="test_tally.py",
        test_body=(
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from tally import Tally\n\n\n"
            "def test_instances_do_not_share_counts():\n"
            "    a, b = Tally(), Tally()\n"
            "    a.bump('x')\n"
            "    assert b.counts.get('x', 0) == 0\n\n\n"
            "def test_reset_is_per_instance():\n"
            "    a, b = Tally(), Tally()\n"
            "    a.bump('x')\n"
            "    b.bump('y')\n"
            "    a.reset()\n"
            "    assert a.counts.get('x', 0) == 0\n"
            "    assert b.counts.get('y', 0) == 1\n"
        ),
    ),
]

TASKS_BY_ID = {t.task_id: t for t in TASKS}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _execute(
    task: TrivialTask,
    mode: str,
    config: Config,
    *,
    model: str,
    effort: str | None,
    python: str,
) -> PathRun:
    """Run one task in one mode against a throwaway repo."""
    t0 = time.monotonic()
    provider = _provider_for(model)
    sniffers: list[_InjectionSniffer] = []

    def _sniffing_build_client(*args: Any, **kwargs: Any) -> Any:
        client, ledger = _real_build_client(*args, **kwargs)
        sniffer = _InjectionSniffer(client._adapter)  # noqa: SLF001
        client._adapter = sniffer  # noqa: SLF001
        sniffers.append(sniffer)
        return client, ledger

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        for rel, content in task.files.items():
            (repo / rel).write_text(content, encoding="utf-8")
        _git_init(repo)

        harness = _harness(task, python)
        try:
            with patch.object(orchestrator, "build_client", _sniffing_build_client):
                result = run_task(
                    task.statement, repo, config,
                    mode=mode, provider=provider, model=model,
                    reasoning_effort=effort, harness=harness,
                )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            err = f"{type(exc).__name__}: {exc}"
            return PathRun(
                task_id=task.task_id, model=model, mode=mode, success=False,
                behavioral_pass=False, mode_used="error", finish_reason="error",
                turns_used=0, model_calls=0, tokens_total=0, tokens_routing=0,
                tokens_fastpath=0, tokens_fullagent=0,
                duration_s=time.monotonic() - t0, error=err,
                incident=_classify_incident(err),
            )

        summary = result.ledger.summary()
        by_phase = summary.get("by_phase", {})

        # Behaviour is re-verified on the finalized repo. The orchestrator
        # already gates on the harness, but checking the real files catches a
        # patch that passed in the shadow workspace yet never landed.
        acc = harness.run(repo)
        behavioral = bool(acc) and all(r.passed for r in acc)
        out = ""
        if not behavioral and acc:
            failed = next((r for r in acc if not r.passed), None)
            out = str(failed.diagnostic())[:600] if failed else ""

        err = result.error
        # A gateway refusal delivered as plain content leaves `error` empty and
        # the run looking `stuck`. Without this override it would be scored as an
        # agent failure, which is exactly the pollution the suite must avoid.
        incident = _classify_incident(err)
        if incident is None and any(s.injected for s in sniffers):
            incident = "provider"
            err = err or "gateway returned a refusal as assistant content"

        return PathRun(
            task_id=task.task_id, model=model, mode=mode, success=result.success,
            behavioral_pass=behavioral, mode_used=result.mode_used,
            finish_reason=result.finish_reason, turns_used=result.turns_used,
            model_calls=summary.get("calls", 0),
            tokens_total=summary.get("automatic_total", 0),
            tokens_routing=by_phase.get("routing", 0),
            tokens_fastpath=by_phase.get("fastpath", 0),
            tokens_fullagent=by_phase.get("fullagent", 0),
            duration_s=time.monotonic() - t0,
            # An escalated auto run bills both phases.
            escalated=(mode == "auto"
                       and by_phase.get("fastpath", 0) > 0
                       and by_phase.get("fullagent", 0) > 0),
            error=err, acceptance_output=out,
            incident=incident,
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _report(runs: list[PathRun]) -> str:
    """Full-Agent baseline vs auto routing: per-task, then suite totals."""
    lines: list[str] = []
    clean = [r for r in runs if r.incident is None]
    excluded = len(runs) - len(clean)
    lines.append(f"runs: {len(runs)}  clean: {len(clean)}  excluded (infra): {excluded}")

    full = {r.task_id: r for r in clean if r.mode == "full"}
    auto = {r.task_id: r for r in clean if r.mode == "auto"}

    # --- how auto resolved each task ---
    fp_done = [r for r in auto.values() if r.mode_used == "fastpath" and r.behavioral_pass]
    escalated = [r for r in auto.values() if r.escalated]
    lines.append("")
    lines.append("--- auto mode resolution ---")
    if auto:
        lines.append(f"FastPath completed  : {len(fp_done)}/{len(auto)}")
        lines.append(f"escalated to Full   : {len(escalated)}/{len(auto)}")
        lines.append(f"behavioral pass     : {sum(1 for r in auto.values() if r.behavioral_pass)}/{len(auto)}")
    if full:
        lines.append(f"full-mode behavioral: {sum(1 for r in full.values() if r.behavioral_pass)}/{len(full)}")

    # --- per task ---
    lines.append("")
    lines.append("--- per task (full baseline -> auto) ---")
    header = (f"{'task':<24}{'full calls':>11}{'full tok':>10}"
              f"{'auto calls':>11}{'auto tok':>10}{'tok saved':>11}  how")
    lines.append(header)
    for tid in sorted(set(full) & set(auto)):
        f, a = full[tid], auto[tid]
        saved = f.tokens_total - a.tokens_total
        how = a.mode_used if not a.escalated else "escalated"
        if not (f.behavioral_pass and a.behavioral_pass):
            how += f" (beh full={f.behavioral_pass} auto={a.behavioral_pass})"
        lines.append(
            f"{tid:<24}{f.model_calls:>11}{f.tokens_total:>10,}"
            f"{a.model_calls:>11}{a.tokens_total:>10,}{saved:>+11,}  {how}"
        )

    # --- suite totals ---
    # Restricted to tasks BOTH modes solved: a mode that gave up early looks
    # cheap, and counting that as a saving would overstate routing's value.
    paired = sorted(t for t in set(full) & set(auto)
                    if full[t].behavioral_pass and auto[t].behavioral_pass)
    lines.append("")
    lines.append("--- suite totals (tasks solved by BOTH modes) ---")
    if not paired:
        lines.append("no task was solved by both modes — no honest total available")
        return "\n".join(lines)

    ft = sum(full[t].tokens_total for t in paired)
    at = sum(auto[t].tokens_total for t in paired)
    fc = sum(full[t].model_calls for t in paired)
    ac = sum(auto[t].model_calls for t in paired)
    fl = sum(full[t].turns_used for t in paired)
    al = sum(auto[t].turns_used for t in paired)

    lines.append(f"tasks counted: {len(paired)}")
    lines.append(f"{'':<16}{'full':>12}{'auto':>12}{'saved':>12}{'':>3}")
    lines.append(f"{'tokens':<16}{ft:>12,}{at:>12,}{ft - at:>+12,}"
                 f"   {((ft - at) / ft * 100) if ft else 0:.1f}%")
    lines.append(f"{'model calls':<16}{fc:>12,}{ac:>12,}{fc - ac:>+12,}"
                 f"   {((fc - ac) / fc * 100) if fc else 0:.1f}%")
    lines.append(f"{'agent loops':<16}{fl:>12,}{al:>12,}{fl - al:>+12,}"
                 f"   {((fl - al) / fl * 100) if fl else 0:.1f}%")
    if at:
        lines.append(f"\ntoken reduction: {ft / at:.2f}x")

    # --- escalation overhead: the cost of guessing wrong ---
    if escalated:
        lines.append("")
        lines.append("--- escalation overhead (FastPath spend that bought nothing) ---")
        for r in escalated:
            lines.append(f"  {r.task_id}: wasted {r.tokens_fastpath:,} on FastPath, "
                         f"then {r.tokens_fullagent:,} on Full Agent")
        wasted = sum(r.tokens_fastpath for r in escalated)
        lines.append(f"  total wasted: {wasted:,} tokens across {len(escalated)} task(s)")
    return "\n".join(lines)


def _matrix_report(runs: list[PathRun]) -> str:
    """Cross-model summary: one row per model, so columns are comparable.

    Only tasks a model solved in BOTH modes contribute to its savings figure.
    Counting a task that auto failed would credit routing for skipping work it
    never completed.
    """
    models = sorted({r.model for r in runs})
    lines = ["", "=" * 78, "MATRIX SUMMARY", "=" * 78, ""]
    lines.append(f"{'model':<20}{'behav':>8}{'fp ok':>7}{'esc':>5}"
                 f"{'full tok':>11}{'auto tok':>11}{'saved':>8}{'x':>6}")

    for model in models:
        mine = [r for r in runs if r.model == model and r.incident is None]
        full = {r.task_id: r for r in mine if r.mode == "full"}
        auto = {r.task_id: r for r in mine if r.mode == "auto"}
        paired = [t for t in full if t in auto
                  and full[t].behavioral_pass and auto[t].behavioral_pass]

        ft = sum(full[t].tokens_total for t in paired)
        at = sum(auto[t].tokens_total for t in paired)
        autos = [r for r in mine if r.mode == "auto"]
        fp_ok = sum(1 for r in autos if r.mode_used == "fastpath")
        esc = sum(1 for r in autos if r.escalated)
        behav = sum(1 for r in mine if r.behavioral_pass)
        saved = f"{((ft - at) / ft * 100):.1f}%" if ft else "n/a"
        ratio = f"{ft / at:.2f}" if at else "n/a"
        lines.append(f"{model:<20}{behav:>5}/{len(mine):<2}{fp_ok:>6}/{len(autos):<1}"
                     f"{esc:>4}{ft:>11,}{at:>11,}{saved:>8}{ratio:>6}")

    incidents = [r for r in runs if r.incident]
    if incidents:
        lines.append("")
        lines.append("--- excluded (infrastructure, not agent capability) ---")
        by_model: dict[str, dict[str, int]] = {}
        for r in incidents:
            by_model.setdefault(r.model, {}).setdefault(r.incident or "?", 0)
            by_model[r.model][r.incident or "?"] += 1
        for model, kinds in sorted(by_model.items()):
            detail = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
            lines.append(f"  {model}: {detail}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="",
                    help="Comma-separated model ids; provider inferred per id")
    ap.add_argument("--model", default="", help="Single model id (shortcut for --models)")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high"],
                    help="OpenAI only; the Anthropic adapter has no such knob")
    ap.add_argument("--tasks", default="", help="Comma-separated task ids (default: all)")
    ap.add_argument("--modes", default=",".join(MODES), help="Comma-separated modes")
    ap.add_argument("--python", default=sys.executable, help="Interpreter for hidden tests")
    args = ap.parse_args()

    models = [m.strip() for m in (args.models or args.model).split(",") if m.strip()]
    if not models:
        print("need --models or --model", file=sys.stderr)
        return 2

    selected = (
        [TASKS_BY_ID[t] for t in args.tasks.split(",") if t]
        if args.tasks else TASKS
    )
    modes = [m for m in args.modes.split(",") if m]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        print(f"unknown mode(s): {unknown}", file=sys.stderr)
        return 2

    config = Config.from_env(REPO_ROOT / ".env")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / "benchmarks" / "results" / f"trivial-paths-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(models) * len(selected) * len(modes)
    print(f"models={models} effort={args.effort} tasks={len(selected)} "
          f"modes={modes} -> {total} runs")
    print(f"results: {out_dir}\n", flush=True)

    runs: list[PathRun] = []
    for model in models:
        print(f"\n{'=' * 60}\nMODEL {model} ({_provider_for(model)})\n{'=' * 60}",
              flush=True)
        for task in selected:
            for mode in modes:
                print(f"[{len(runs) + 1}/{total}] {model} {task.task_id} mode={mode}",
                      flush=True)
                run = _execute(task, mode, config, model=model,
                               effort=args.effort, python=args.python)
                runs.append(run)
                flag = "OK " if run.behavioral_pass else "   "
                print(f"  {flag} behavioral={run.behavioral_pass} used={run.mode_used} "
                      f"finish={run.finish_reason} calls={run.model_calls} "
                      f"turns={run.turns_used} tokens={run.tokens_total:,} "
                      f"{run.duration_s:.0f}s"
                      + ("  ESCALATED" if run.escalated else ""), flush=True)
                if run.incident:
                    print(f"      incident={run.incident}", flush=True)
                if run.error:
                    print(f"      error: {run.error[:200]}", flush=True)
                (out_dir / "runs.json").write_text(
                    json.dumps({"models": models, "effort": args.effort,
                                "runs": [asdict(r) for r in runs]}, indent=2),
                    encoding="utf-8",
                )

    parts = []
    for model in models:
        mine = [r for r in runs if r.model == model]
        if mine:
            parts.append(f"\n{'#' * 78}\n# {model}\n{'#' * 78}\n" + _report(mine))
    report = "\n".join(parts) + "\n" + _matrix_report(runs)
    print("\n" + report)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
