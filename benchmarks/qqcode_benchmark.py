"""QQCode AB/BA Benchmark Runner.

Runs real coding tasks from tasks/real_tasks_v2.json against QQCode in
automatic mode vs /full mode using AB/BA crossover design.

Usage:
    # Validate fixtures without spending tokens:
    python benchmarks/qqcode_benchmark.py --verify-fixtures

    # Full AB/BA experiment (all eligible tasks):
    python benchmarks/qqcode_benchmark.py --execute

    # Single task:
    python benchmarks/qqcode_benchmark.py --execute --task flask-config-file-mode

    # Specify model (required for publishable results):
    python benchmarks/qqcode_benchmark.py --execute --model claude-sonnet-5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qqcode.config import Config  # noqa: E402
from qqcode.memory.trace import TraceStore  # noqa: E402
from qqcode.orchestrator import run_task  # noqa: E402

TASKS_PATH = Path(__file__).parent / "tasks" / "real_tasks_v2.json"

# Derivability audit. Kept in its own file because TASKS_PATH is a symlink to
# another project's shared fixture definition — annotating it there would
# silently change what every run in both projects measures.
DERIVABILITY_PATH = Path(__file__).parent / "tasks" / "derivability.json"

# Only "derivable" fixtures can measure capability. The rest fail for reasons
# the agent cannot control, so averaging over them reports fixture-authoring
# artifacts as agent incapacity.
DERIVABILITY_VERDICTS = frozenset(
    {"derivable", "not_derivable", "whole_file_granularity", "unverified"}
)
MEASURABLE_VERDICT = "derivable"


def load_tasks(path: Path | None = None) -> list[dict[str, Any]]:
    """Task definitions from the (symlinked) shared fixture manifest."""
    return list(json.loads((path or TASKS_PATH).read_text())["tasks"])


def load_derivability(path: Path | None = None) -> dict[str, str]:
    """`{task_id: verdict}` from the audit file; empty when it is absent.

    An empty result means no exclusions are applied, so a missing audit file
    degrades to the previous all-fixtures behaviour rather than to zero
    measurable tasks — a silent zero would look like catastrophic incapacity.
    """
    target = path or DERIVABILITY_PATH
    if not target.exists():
        return {}
    raw = json.loads(target.read_text())
    return {task_id: entry["verdict"] for task_id, entry in raw["verdicts"].items()}
RESULTS_ROOT = Path(__file__).parent / "results"

# Shared caches live OUTSIDE any single run directory so repeated invocations
# reuse them. A per-run cache re-cloned ~20MB per repo every time, which is the
# main reason transient network faults kept burning tasks.
CACHE_ROOT = Path(__file__).parent / ".cache"
REPO_CACHE_DIR = CACHE_ROOT / "repos"
VENV_CACHE_DIR = CACHE_ROOT / "venvs"

_HF_ENDPOINT = "https://datasets-server.huggingface.co/rows"
_FETCH_ATTEMPTS = 3
_ACCEPT_TIMEOUT = 180  # seconds per acceptance command
_MODE_MAP = {"automatic": "auto", "full": "full"}

# Retry policy for network-bound git operations. Only transient network faults
# are retried; a bad commit hash or corrupt repo fails immediately.
_NET_ATTEMPTS = 4
_NET_BACKOFF = (1.0, 2.0, 4.0)  # seconds before attempts 2, 3, 4


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """Outcome of one agent run against one task."""

    task_id: str
    category: str
    mode: str              # "automatic" | "full"
    order: str             # "AB" | "BA" | "preflight"
    cycle: int
    # Agent outcome
    agent_success: bool
    mode_used: str         # "fastpath" | "fullagent" | "error"
    finish_reason: str
    fastpath_attempted: bool
    fastpath_success: bool
    turns_used: int
    tokens_total: int
    tokens_routing: int
    tokens_fastpath: int
    tokens_fullagent: int
    duration_ms: float
    # Evaluator outcome
    behavioral_complete: bool    # hidden acceptance passed
    acceptance_output: str       # truncated stdout+stderr
    # Errors
    error: str | None = None
    # Why FastPath escalated, straight from the trace DB ("" when it succeeded
    # or was never attempted).
    fastpath_reason: str = ""
    # Incident type: set when failure is NOT due to agent capability.
    # Runs with an incident are excluded from behavioral_rate and paired comparisons.
    # None=clean | "network"=HF/git/pip unreachable | "environment"=missing dep/venv
    # | "test_conflict"=agent edited a file the hidden test patch also touches,
    #   so the behavioral question was never asked
    incident_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcceptanceOutcome:
    """What the hidden acceptance run established, if anything.

    `conflict` is the case a plain bool could not express: the hidden test
    patch did not apply, so no behavioral claim was tested. Keeping it separate
    is what lets the run be excluded from the rate rather than counted as a
    failure the agent caused.
    """

    passed: bool
    output: str
    conflict: bool = False


@dataclass(frozen=True)
class FixtureMaterial:
    test_patch: str
    source_patch: str
    agent_statement: str        # effective prompt sent to QQCode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(text: str | None) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode()).hexdigest()


_NETWORK_RE = re.compile(
    r"Could not resolve|Connection refused|Network unreachable|"
    r"SSL|timed? out|Temporary failure|No route to host|"
    r"Max retries exceeded|NewConnectionError",
    re.I,
)


def _is_network_err(msg: str) -> bool:
    return bool(_NETWORK_RE.search(msg))


# Gateway/provider faults: the model API itself refused or broke the exchange.
# These say nothing about whether the agent *could* solve the task, so runs
# carrying them are excluded from behavioral_rate and paired comparisons.
#
# "No tool output found for function call fc_..." is included deliberately: the
# gateway rewrites our `call_<id>` tool-call ids to `fc_<id>` when translating
# Chat Completions -> Responses API and then fails to map them back. Captured
# request payloads confirm we send one tool result per tool call with matching
# ids, so the broken pairing originates upstream, not here.
_PROVIDER_FAULT_RE = re.compile(
    r"No tool output found for function call|"
    r"Bad gateway|origin_bad_gateway|Error code: 5\d\d|"
    r"\b50[0234]\b|overloaded_error|rate.?limit|Error code: 429|"
    r"service unavailable|upstream|InternalServerError",
    re.I,
)


def _is_provider_fault(msg: str) -> bool:
    return bool(_PROVIDER_FAULT_RE.search(msg))


# Positive evidence that pytest got far enough to execute and report on tests.
# Deliberately a whitelist of pytest's own report markers rather than a
# blacklist of exception names: strings like FileNotFoundError or Errno appear
# legitimately inside the traceback of a test that is *supposed* to fail, so
# scanning for them mislabels correct baselines as environment incidents.
_HARNESS_RAN_RE = re.compile(
    r"short test summary info|=+ (?:FAILURES|ERRORS) =+|"
    r"^FAILED \S|^ERROR \S|\d+ (?:passed|failed|error|skipped)|"
    r"^[.FEsxX]+\s*\[\s*\d+%\]|no tests ran|INTERNALERROR",
    re.M,
)


def _harness_ran(out: str) -> bool:
    """True when pytest produced test-level results.

    Used to separate "tests ran and failed" (a valid baseline) from "the runner
    never started" (infrastructure). Only the latter is an incident.
    """
    return bool(_HARNESS_RAN_RE.search(out))


def _git(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check,
        timeout=300,
    )


def _git_net(*args: str, what: str) -> None:
    """Run a network-bound git command, retrying only on transient faults.

    Raises RuntimeError with a "network error:" prefix once retries are
    exhausted, so the caller can label the run as a network incident rather
    than an agent capability failure. Deterministic failures (bad object,
    corrupt repo) are raised on the first attempt without burning retries.
    """
    last = ""
    for attempt in range(_NET_ATTEMPTS):
        r = _git(*args, check=False)
        if r.returncode == 0:
            return
        last = (r.stderr + r.stdout).strip()
        if not _is_network_err(last):
            raise RuntimeError(f"git {what} failed: {last[:300]}")
        if attempt + 1 < _NET_ATTEMPTS:
            delay = _NET_BACKOFF[min(attempt, len(_NET_BACKOFF) - 1)]
            print(
                f"    transient network fault ({what}), "
                f"retry {attempt + 2}/{_NET_ATTEMPTS} in {delay:.0f}s …",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"network error: {what} failed after {_NET_ATTEMPTS} attempts: {last[:300]}"
    )


def _clip(text: str, limit: int) -> str:
    """Truncate keeping BOTH ends.

    pytest prints its verdict last ("1 failed in 0.08s", "FAILED test::name",
    "short test summary info"), so head-only truncation throws away exactly the
    evidence baseline_failure_pattern needs to match. Keep a head for the
    collection/import phase and a tail for the summary.
    """
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    omitted = len(text) - limit
    return f"{text[:head]}\n… [{omitted} chars omitted] …\n{text[-tail:]}"


def _run_cmd(
    cmd: list[str], cwd: Path, timeout: int = _ACCEPT_TIMEOUT
) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, _clip(r.stdout + r.stderr, 6000)
    except subprocess.TimeoutExpired:
        return -1, f"[timed out after {timeout}s]"
    except Exception as exc:
        return -1, f"[error: {exc}]"


# ---------------------------------------------------------------------------
# Git provisioning: bare clone cache + disposable worktrees
# ---------------------------------------------------------------------------


class RepoCache:
    """Bare clone cache: fetch once, create disposable worktrees per run."""

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _bare_path(self, url: str) -> Path:
        slug = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self._dir / f"{slug}.git"

    def ensure(self, url: str, commit: str) -> Path:
        bare = self._bare_path(url)
        if not bare.exists():
            print(f"  cloning {url} …", flush=True)
            _git("init", "--bare", "--quiet", str(bare))
            _git("-C", str(bare), "remote", "add", "origin", url)
        # Check if commit is already present. On a persistent cache this is the
        # common case, so a whole run costs no network at all.
        r = _git("-C", str(bare), "cat-file", "-e", f"{commit}^{{commit}}", check=False)
        if r.returncode != 0:
            print(f"  fetching {commit[:12]} …", flush=True)
            _git_net(
                "-C", str(bare), "fetch", "--depth=1", "origin", commit,
                what=f"fetch {commit[:12]}",
            )
        return bare

    def checkout(self, url: str, commit: str, dest: Path) -> None:
        bare = self.ensure(url, commit)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # The cache outlives individual runs, so it accumulates worktree
        # registrations pointing at deleted directories. Prune them or
        # `worktree add` eventually fails on a stale entry.
        _git("-C", str(bare), "worktree", "prune", check=False)
        _git("-C", str(bare), "worktree", "add", "--detach", "--force", str(dest), commit)

    def remove_worktree(self, url: str, dest: Path) -> None:
        bare = self._bare_path(url)
        _git("-C", str(bare), "worktree", "remove", "--force", str(dest), check=False)
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)


INCIDENT_TEST_CONFLICT = "test_conflict"


def _apply_acceptance_result(record: RunRecord, outcome: AcceptanceOutcome) -> None:
    """Record an acceptance outcome, attributing a patch conflict as an incident.

    Kept separate from `_run_acceptance` so both call sites attribute
    identically: detecting the conflict is useless if a caller still files it as
    a clean behavioral failure, because the rate is computed from
    `incident_type`, not from the outcome.
    """
    record.behavioral_complete = outcome.passed
    # _clip keeps head AND tail: pytest's verdict line is at the end, so
    # head-only truncation hides the very reason a run was scored a failure.
    record.acceptance_output = _clip(outcome.output, 1200)
    if outcome.conflict and record.incident_type is None:
        record.incident_type = INCIDENT_TEST_CONFLICT


class TestPatchConflictError(RuntimeError):
    """The hidden test patch would not apply to the agent's workspace.

    Distinct from a patch failure on the *source* side: it means the agent
    edited a file the hidden test patch also touches, so the measurement never
    ran. That is an apparatus fault, not a verdict on the agent's fix.
    """


def _apply_patch(workspace: Path, patch_text: str, *, is_test_patch: bool = False) -> None:
    r = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        input=patch_text, cwd=workspace, capture_output=True, text=True,
    )
    if r.returncode != 0:
        detail = f"patch failed: {r.stderr[:400]}"
        raise TestPatchConflictError(detail) if is_test_patch else RuntimeError(detail)


# ---------------------------------------------------------------------------
# Fixture fetch: SWE-bench (HuggingFace) + upstream_commit (git diff)
# ---------------------------------------------------------------------------


def _fetch_upstream_commit(task: dict[str, Any], cache: RepoCache) -> FixtureMaterial:
    """For upstream_commit fixtures: clone repo, git diff base→fix for patches."""
    fx = task["fixture"]
    fix_commit = fx["fix_commit"]
    base_commit = task["base_commit"]
    test_paths: list[str] = fx.get("test_paths", [])

    # Ensure both base and fix commits are available in the bare repo
    bare = cache.ensure(task["repository_url"], base_commit)
    cache.ensure(task["repository_url"], fix_commit)  # fetch fix_commit too

    # test_patch: diff base→fix limited to test paths
    test_cmd = ["git", "-C", str(bare), "diff", "--binary", base_commit, fix_commit, "--", *test_paths]
    test_r = subprocess.run(test_cmd, capture_output=True, text=True, check=True)
    test_patch = test_r.stdout

    # source_patch: diff base→fix for all files
    src_cmd = ["git", "-C", str(bare), "diff", "--binary", base_commit, fix_commit]
    src_r = subprocess.run(src_cmd, capture_output=True, text=True, check=True)
    source_patch = src_r.stdout

    if _sha256(source_patch) != fx.get("source_patch_sha256", ""):
        raise RuntimeError(f"{task['id']}: source patch digest mismatch (upstream_commit)")
    if _sha256(test_patch) != fx.get("test_patch_sha256", ""):
        raise RuntimeError(f"{task['id']}: test patch digest mismatch (upstream_commit)")

    statement = fx.get("agent_statement") or task.get("source_statement", "")
    return FixtureMaterial(
        test_patch=test_patch,
        source_patch=source_patch,
        agent_statement=statement,
    )


def _fetch_fixture(task: dict[str, Any], cache: RepoCache | None = None) -> FixtureMaterial:
    fx_type = task["fixture"].get("type", "swebench_lite")
    if fx_type == "upstream_commit":
        if cache is None:
            raise ValueError("upstream_commit requires a RepoCache")
        return _fetch_upstream_commit(task, cache)
    return _fetch_swebench(task)


def _fetch_swebench(task: dict[str, Any]) -> FixtureMaterial:
    fx = task["fixture"]
    query = urllib.parse.urlencode({
        "dataset": fx["dataset_id"], "config": fx["config"],
        "split": fx["split"], "offset": fx["row_index"],
        "length": 1, "revision": fx["revision"],
    })
    url = f"{_HF_ENDPOINT}?{query}"
    payload = None
    for attempt in range(_FETCH_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except Exception:
            if attempt + 1 < _FETCH_ATTEMPTS:
                time.sleep(attempt + 1)
    if payload is None:
        raise RuntimeError(f"{task['id']}: HuggingFace fetch failed after {_FETCH_ATTEMPTS} attempts")

    rows = payload.get("rows", [])
    if len(rows) != 1:
        raise RuntimeError(f"{task['id']}: unexpected row count {len(rows)}")
    row = rows[0]["row"]

    source_patch = row.get("patch", "")
    test_patch = row.get("test_patch", "")
    problem_statement = row.get("problem_statement", "")

    if _sha256(source_patch) != fx.get("source_patch_sha256", ""):
        raise RuntimeError(f"{task['id']}: source patch digest mismatch")
    if _sha256(test_patch) != fx.get("test_patch_sha256", ""):
        raise RuntimeError(f"{task['id']}: test patch digest mismatch")

    statement = fx.get("agent_statement") or task.get("source_statement") or problem_statement
    return FixtureMaterial(
        test_patch=test_patch,
        source_patch=source_patch,
        agent_statement=statement,
    )


# ---------------------------------------------------------------------------
# Runtime venv: one cached venv per runtime_key
# ---------------------------------------------------------------------------


_PY_KEY_RE = re.compile(r"py(\d)(\d+)")


def _interpreter_version(exe: str) -> str | None:
    """Return the actual 'X.Y' of an interpreter, or None if it won't run."""
    try:
        r = subprocess.run(
            [exe, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def _required_python(fx: dict[str, Any]) -> str | None:
    """Requested interpreter 'X.Y' from runtime_python, else from runtime_key."""
    explicit = fx.get("runtime_python")
    if explicit:
        return str(explicit).strip()
    if m := _PY_KEY_RE.search(fx.get("runtime_key", "")):
        return f"{m.group(1)}.{m.group(2)}"
    return None


def _resolve_python(want: str | None) -> str:
    """Resolve an interpreter matching `want`, or raise.

    Never substitute a different minor version: runtime_packages are pinned to
    versions that are version-sensitive (setuptools 65.5.1 touches
    pkgutil.ImpImporter, removed in 3.12), so a silent downgrade surfaces later
    as a bogus environment incident instead of a clear setup failure.
    """
    if not want:
        return sys.executable
    for cand in (sys.executable, shutil.which(f"python{want}")):
        if cand and _interpreter_version(cand) == want:
            return cand
    uv = shutil.which("uv")
    if uv:
        for install_first in (False, True):
            if install_first:
                print(f"  uv python install {want} …", flush=True)
                subprocess.run(
                    [uv, "python", "install", want],
                    capture_output=True, text=True, timeout=900,
                )
            found = subprocess.run(
                [uv, "python", "find", want],
                capture_output=True, text=True, timeout=300,
            )
            cand = found.stdout.strip()
            if found.returncode == 0 and cand and _interpreter_version(cand) == want:
                return cand
    raise RuntimeError(
        f"python{want} unavailable but required by pinned runtime_packages; "
        f"install it (e.g. `uv python install {want}`). Refusing to substitute "
        f"python{_interpreter_version(sys.executable)}"
    )


def _ensure_venv(task: dict[str, Any], venv_cache: Path) -> Path:
    fx = task["fixture"]
    key = fx.get("runtime_key", "default")
    packages = fx.get("runtime_packages", [])
    python_exe = _resolve_python(_required_python(fx))

    venv = venv_cache / key
    stamp = venv / ".packages_stamp"
    # Interpreter identity is part of the cache key: a venv built with the wrong
    # minor version must be rebuilt rather than reused.
    pkg_hash = _sha256(json.dumps(
        {"packages": sorted(packages), "python": _interpreter_version(python_exe)},
        sort_keys=True,
    ))

    if not venv.exists() or not stamp.exists() or stamp.read_text().strip() != pkg_hash:
        if venv.exists():
            shutil.rmtree(venv)
        subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)
        pip = venv / "bin" / "pip"
        if packages:
            subprocess.run(
                [str(pip), "install", "-q", "--no-deps", *packages], check=True
            )
            # Re-install with deps to resolve transitive requirements
            subprocess.run([str(pip), "install", "-q", *packages], check=True)
        stamp.write_text(pkg_hash)

    return venv / "bin" / "python"


def _workspace_env(task: dict[str, Any], workspace: Path) -> dict[str, str]:
    """Environment for any command run against a workspace tree.

    PYTHONPATH points at the workspace (and src/ when present) on purpose:
    acceptance must exercise the source the agent edited, not an installed
    copy. That is also why setuptools-scm would otherwise try `git describe`
    on a shallow worktree and invent a 0.1.dev version — the manifest's
    runtime_environment (SETUPTOOLS_SCM_PRETEND_VERSION) pins it instead.
    """
    env = os.environ.copy()
    env.update(task["fixture"].get("runtime_environment", {}))

    python_paths = [str(workspace)]
    src_dir = workspace / "src"
    if src_dir.is_dir():
        python_paths.append(str(src_dir))
    if existing := env.get("PYTHONPATH"):
        python_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_prepare_commands(
    task: dict[str, Any], workspace: Path, venv_python: Path, env: dict[str, str]
) -> None:
    """Run manifest prepare_commands (e.g. `setup.py --version`).

    These exist to materialise build-time artefacts such as _version.py before
    any timed or evaluated command runs. A failure here is a setup failure, not
    a test result, so it raises.
    """
    for raw_cmd in task.get("prepare_commands", []):
        cmd = [str(venv_python) if c == "{python}" else c for c in raw_cmd]
        r = subprocess.run(
            cmd, cwd=workspace, capture_output=True, text=True,
            timeout=_ACCEPT_TIMEOUT, env=env,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"prepare command {cmd[1:]} failed: {(r.stderr + r.stdout)[:400]}"
            )


# ---------------------------------------------------------------------------
# Acceptance check
# ---------------------------------------------------------------------------


def _run_acceptance(
    task: dict[str, Any],
    workspace: Path,
    test_patch: str,
    venv_python: Path,
) -> AcceptanceOutcome:
    """Copy workspace, apply hidden test patch, run acceptance command.

    Returns an outcome that distinguishes "tests ran and failed" from "the
    hidden test patch could not be applied". Collapsing the two into one bool
    attributed apparatus faults to the agent: a run whose fix was correct but
    which also touched a test file scored exactly like a wrong fix.
    """
    eval_dir = workspace.parent / f"{workspace.name}-eval"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    shutil.copytree(workspace, eval_dir, symlinks=True)
    try:
        _apply_patch(eval_dir, test_patch, is_test_patch=True)

        env = _workspace_env(task, eval_dir)
        _run_prepare_commands(task, eval_dir, venv_python, env)

        cmd = [
            str(venv_python) if c == "{python}" else c
            for c in task["acceptance_command"]
        ]

        try:
            r = subprocess.run(
                cmd, cwd=eval_dir, capture_output=True, text=True,
                timeout=_ACCEPT_TIMEOUT, env=env,
            )
            code, out = r.returncode, _clip(r.stdout + r.stderr, 6000)
        except subprocess.TimeoutExpired:
            code, out = -1, f"[timed out after {_ACCEPT_TIMEOUT}s]"

        return AcceptanceOutcome(passed=code == 0, output=out)
    except TestPatchConflictError as exc:
        # The agent edited a file the hidden test patch also touches, so the
        # behavioral question was never asked. Report it instead of answering
        # "no" on the measurement's behalf.
        return AcceptanceOutcome(
            passed=False, output=f"[test patch conflict: {exc}]", conflict=True
        )
    except Exception as exc:
        return AcceptanceOutcome(passed=False, output=f"[acceptance setup error: {exc}]")
    finally:
        shutil.rmtree(eval_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# QQCode agent invocation
# ---------------------------------------------------------------------------


def _run_qqcode(
    statement: str,
    workspace: Path,
    mode: str,
    config: Config,
    results_dir: Path,
    *,
    model: str | None = None,
    provider: str | None = None,
    reasoning_effort: str | None = None,
) -> tuple[Any, float]:
    """Run QQCode and return (RunResult, duration_ms).

    model/provider/reasoning_effort are passed explicitly rather than via env
    vars: nothing reads a MODEL env var, so setting one silently left the run on
    default tier models while the report claimed otherwise.
    """
    store_path = results_dir / "trace.db"
    store = TraceStore(store_path)
    t0 = time.monotonic()
    try:
        result = run_task(
            task=statement,
            repo=workspace,
            config=config,
            mode=mode,
            dry_run=False,
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
            trace_store=store,
        )
    finally:
        store.close()
    duration_ms = (time.monotonic() - t0) * 1000
    return result, duration_ms


# ---------------------------------------------------------------------------
# Single run: provision → agent → evaluate
# ---------------------------------------------------------------------------


def _run_one(
    task: dict[str, Any],
    fixture: FixtureMaterial,
    mode_label: str,
    order: str,
    cycle: int,
    cache: RepoCache,
    venv_python: Path,
    run_dir: Path,
    config: Config,
    execute: bool,
    model: str | None = None,
    provider: str | None = None,
    reasoning_effort: str | None = None,
) -> RunRecord:
    workspace = run_dir / "workspace"
    base = RunRecord(
        task_id=task["id"], category=task.get("category", "?"),
        mode=mode_label, order=order, cycle=cycle,
        agent_success=False, mode_used="", finish_reason="",
        fastpath_attempted=False, fastpath_success=False,
        turns_used=0, tokens_total=0, tokens_routing=0,
        tokens_fastpath=0, tokens_fullagent=0, duration_ms=0.0,
        behavioral_complete=False, acceptance_output="",
    )

    try:
        # Provision a fresh worktree
        cache.checkout(task["repository_url"], task["base_commit"], workspace)
        # Deliberately NOT `pip install -e workspace`: that uninstalls the
        # pinned runtime_packages (e.g. pytest==6.2.5) and replaces them with
        # the workspace source, which is what produced the bogus
        # "minversion requires pytest-2.0, actual pytest-0.1.dev1" failures.
        # Importability comes from PYTHONPATH in _workspace_env instead.
        # Materialise build artefacts (e.g. _version.py) before the timed run
        _run_prepare_commands(
            task, workspace, venv_python, _workspace_env(task, workspace)
        )
    except Exception as exc:
        err = str(exc)
        base.error = f"provision failed: {err}"
        base.incident_type = "network" if _is_network_err(err) else "environment"
        return base

    if not execute:
        # verify-fixtures: check that baseline test correctly FAILS (pre-fix state).
        # Distinguish agent-irrelevant failures: environment crashes are incidents,
        # not fixture failures.
        try:
            outcome = _run_acceptance(task, workspace, fixture.test_patch, venv_python)
        except Exception as exc:
            base.error = f"acceptance setup: {exc}"
            base.incident_type = "environment"
            return base

        passed, out = outcome.passed, outcome.output
        base.acceptance_output = _clip(out, 1200)
        if outcome.conflict:
            # This workspace is pristine — no agent touched it — so a patch that
            # will not apply means the fixture's own test_patch is stale against
            # its pinned base_commit. Apparatus, not a derivability verdict.
            base.error = f"test_patch does not apply to base_commit: {out[:200]}"
            base.incident_type = "environment"
            return base
        if passed:
            base.behavioral_complete = False   # test passes pre-fix → fixture wrong
            return base

        pattern = task["fixture"].get("baseline_failure_pattern", "")
        if pattern and re.search(pattern, out):
            # The manifest is authoritative about what a correct baseline
            # failure looks like. Some patterns deliberately accept
            # ModuleNotFoundError / INTERNALERROR as the expected pre-fix
            # symptom, so no heuristic may veto a pattern match.
            base.behavioral_complete = True
        elif not _harness_ran(out):
            # No pytest result markers at all: the runner never got as far as
            # executing tests (missing interpreter, unimportable conftest), so
            # this is infrastructure, not an agent-relevant outcome.
            base.incident_type = "environment"
        elif not pattern:
            # No pattern to satisfy; a non-zero exit with real test output is
            # an acceptable baseline failure.
            base.behavioral_complete = True
        return base

    # --- timed agent run ---
    qqcode_mode = _MODE_MAP.get(mode_label, "auto")
    try:
        result, duration_ms = _run_qqcode(
            statement=fixture.agent_statement,
            workspace=workspace,
            mode=qqcode_mode,
            config=config,
            results_dir=run_dir,
            model=model,
            provider=provider,
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:
        base.error = f"agent error: {exc}"
        base.mode_used = "error"
        if _is_network_err(str(exc)):
            base.incident_type = "network"
        elif _is_provider_fault(str(exc)):
            base.incident_type = "provider"
        return base

    # Extract ledger metrics
    ledger = result.ledger
    s = ledger.summary()
    by_phase = s.get("by_phase", {})

    base.agent_success = result.success
    base.mode_used = result.mode_used
    base.finish_reason = result.finish_reason
    base.turns_used = result.turns_used
    if result.error:
        base.error = result.error
    base.tokens_total = s.get("automatic_total", 0)
    base.tokens_routing = by_phase.get("routing", 0)
    base.tokens_fastpath = by_phase.get("fastpath", 0)
    base.tokens_fullagent = by_phase.get("fullagent", 0)
    base.duration_ms = duration_ms

    # FastPath tracking from trace DB (if available)
    try:
        store = TraceStore(run_dir / "trace.db")
        records = store.all()
        store.close()
        if records:
            t = records[-1]
            base.fastpath_attempted = t.fastpath_attempted
            base.fastpath_success = t.fastpath_success
            # Surface why FastPath bailed and what killed the Full Agent. Without
            # this an `error` finish_reason arrives with no diagnosable cause.
            if t.fastpath_reason and t.fastpath_reason != "ok":
                base.fastpath_reason = t.fastpath_reason
            if not base.error and t.finish_summary:
                base.error = t.finish_summary
    except Exception:
        pass

    # A run that died inside the model API tells us nothing about agent
    # capability — classify it so it is excluded from behavioral_rate.
    if base.incident_type is None and not base.agent_success and base.error:
        if _is_network_err(base.error):
            base.incident_type = "network"
        elif _is_provider_fault(base.error):
            base.incident_type = "provider"

    # --- evaluate: apply hidden test + run acceptance ---
    try:
        outcome = _run_acceptance(task, workspace, fixture.test_patch, venv_python)
        _apply_acceptance_result(base, outcome)
    except Exception as exc:
        base.acceptance_output = f"[eval error: {exc}]"

    return base


# ---------------------------------------------------------------------------
# AB/BA scheduler
# ---------------------------------------------------------------------------


def _schedule(tasks: list[dict[str, Any]], cycles: int) -> list[tuple[dict, str, str, int]]:
    """Return (task, mode, order, cycle) in AB/BA order.

    full_only tasks run only in full mode (no AB/BA pairing).
    """
    runs: list[tuple[dict, str, str, int]] = []
    for task in tasks:
        cat = task.get("category", "")
        for cycle in range(1, cycles + 1):
            if cat == "full_only":
                runs.append((task, "full", "only", cycle))
            else:
                runs.append((task, "automatic", "AB", cycle))
                runs.append((task, "full",      "AB", cycle))
                runs.append((task, "full",      "BA", cycle))
                runs.append((task, "automatic", "BA", cycle))
    return runs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _build_report(
    records: list[RunRecord],
    config_meta: dict[str, Any],
) -> dict[str, Any]:
    auto_recs = [r for r in records if r.mode == "automatic"]
    full_recs  = [r for r in records if r.mode == "full"]
    # Exclude incident runs from capability rates (keep in attempted count)
    clean_auto = [r for r in auto_recs if r.incident_type is None]
    clean_full  = [r for r in full_recs  if r.incident_type is None]

    def _rate(recs: list[RunRecord], attr: str) -> float:
        return sum(1 for r in recs if getattr(r, attr)) / len(recs) if recs else 0.0

    def _mean(recs: list[RunRecord], attr: str) -> float:
        vals = [getattr(r, attr) for r in recs]
        return sum(vals) / len(vals) if vals else 0.0

    # Paired: both modes clean and behaviorally complete for same (task, cycle, order)
    auto_map = {(r.task_id, r.cycle, r.order): r
                for r in clean_auto if r.behavioral_complete}
    full_map  = {(r.task_id, r.cycle, r.order): r
                 for r in clean_full  if r.behavioral_complete}
    pairs = [(auto_map[k], full_map[k]) for k in auto_map if k in full_map]

    def _delta(attr: str) -> float:
        return (sum(getattr(f, attr) - getattr(a, attr) for a, f in pairs) / len(pairs)
                if pairs else 0.0)

    fp_attempted = sum(1 for r in clean_auto if r.fastpath_attempted)
    fp_completed = sum(1 for r in clean_auto if r.fastpath_success)
    incidents_by_type: dict[str, int] = {}
    for r in records:
        if r.incident_type:
            incidents_by_type[r.incident_type] = incidents_by_type.get(r.incident_type, 0) + 1

    # Derivability: a fixture whose hidden assertion the statement never implies
    # cannot measure capability. Averaging over those reports fixture-authoring
    # artifacts as agent incapacity, so they get their own, honest denominator.
    audit = load_derivability()
    excluded = {
        r.task_id: audit[r.task_id]
        for r in records
        if r.task_id in audit and audit[r.task_id] != MEASURABLE_VERDICT
    }
    measurable = [
        r for r in clean_auto
        if audit.get(r.task_id, MEASURABLE_VERDICT) == MEASURABLE_VERDICT
    ]
    measurable_full = [
        r for r in clean_full
        if audit.get(r.task_id, MEASURABLE_VERDICT) == MEASURABLE_VERDICT
    ]

    return {
        "config": config_meta,
        "derivability": {
            "measurable_tasks": len({r.task_id for r in measurable}),
            "excluded": excluded,
            "excluded_count": len(excluded),
            "behavioral_rate_measurable": _rate(measurable, "behavioral_complete"),
            "behavioral_rate_measurable_full": _rate(measurable_full, "behavioral_complete"),
            "note": (
                "behavioral_rate_measurable covers only fixtures whose hidden "
                "assertion is derivable from the statement the agent was shown. "
                "The all_runs rates below include every fixture and therefore "
                "understate capability; see tasks/derivability.json for the "
                "per-fixture verdict and reasoning."
            ),
        },
        "incidents": {
            "total": sum(incidents_by_type.values()),
            "by_type": incidents_by_type,
            "note": "Incident runs excluded from behavioral_rate and paired comparisons.",
        },
        "all_runs": {
            "automatic": {
                "attempted": len(auto_recs),
                "incidents": len(auto_recs) - len(clean_auto),
                "behavioral_complete": sum(1 for r in clean_auto if r.behavioral_complete),
                "behavioral_rate": _rate(clean_auto, "behavioral_complete"),
                "avg_tokens": _mean(clean_auto, "tokens_total"),
                "avg_turns": _mean(clean_auto, "turns_used"),
                "avg_duration_ms": _mean(clean_auto, "duration_ms"),
            },
            "full": {
                "attempted": len(full_recs),
                "incidents": len(full_recs) - len(clean_full),
                "behavioral_complete": sum(1 for r in clean_full if r.behavioral_complete),
                "behavioral_rate": _rate(clean_full, "behavioral_complete"),
                "avg_tokens": _mean(clean_full, "tokens_total"),
                "avg_turns": _mean(clean_full, "turns_used"),
                "avg_duration_ms": _mean(clean_full, "duration_ms"),
            },
        },
        "fastpath": {
            "attempted": fp_attempted,
            "completed": fp_completed,
            "upgrades": sum(1 for r in clean_auto if r.mode_used == "fullagent"),
            "success_rate": fp_completed / fp_attempted if fp_attempted else 0.0,
        },
        "paired": {
            "pairs": len(pairs),
            "token_delta_full_minus_auto": _delta("tokens_total"),
            "turn_delta_full_minus_auto": _delta("turns_used"),
            "duration_delta_ms": _delta("duration_ms"),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    a = report["all_runs"]["automatic"]
    f = report["all_runs"]["full"]
    fp = report["fastpath"]
    p  = report["paired"]
    d = report.get("derivability", {})
    lines = [
        "# QQCode Benchmark Report", "",
    ]
    if d:
        # Lead with the measurable rate: the all-fixtures rate below mixes in
        # fixtures a correct fix cannot pass, so reporting it first would
        # present a fixture-authoring artifact as a capability result.
        lines += [
            "## Measurable Capability", "",
            f"Fixtures whose hidden assertion is derivable from the statement: "
            f"**{d['measurable_tasks']}** "
            f"(excluded: {d['excluded_count']})",
            "",
            "| Metric | Automatic | /full |",
            "| --- | ---: | ---: |",
            f"| Behavioral rate (measurable only) | "
            f"{d['behavioral_rate_measurable']:.3f} | "
            f"{d['behavioral_rate_measurable_full']:.3f} |",
            "",
        ]
        if d["excluded"]:
            lines += ["Excluded fixtures and why:", ""]
            lines += [
                f"- `{task_id}` — {verdict}"
                for task_id, verdict in sorted(d["excluded"].items())
            ]
            lines += [
                "",
                "See `benchmarks/tasks/derivability.json` for the per-fixture reasoning.",
                "",
            ]
    lines += [
        "## All Runs (every fixture, understates capability)", "",
        "| Metric | Automatic | /full |",
        "| --- | ---: | ---: |",
        f"| Attempted | {a['attempted']} | {f['attempted']} |",
        f"| Behavioral completions | {a['behavioral_complete']} | {f['behavioral_complete']} |",
        f"| Behavioral rate | {a['behavioral_rate']:.3f} | {f['behavioral_rate']:.3f} |",
        f"| Avg tokens | {a['avg_tokens']:,.0f} | {f['avg_tokens']:,.0f} |",
        f"| Avg agent turns | {a['avg_turns']:.1f} | {f['avg_turns']:.1f} |",
        f"| Avg duration (ms) | {a['avg_duration_ms']:,.0f} | {f['avg_duration_ms']:,.0f} |",
        "", "## FastPath Routing", "",
        f"- Attempted: {fp['attempted']}",
        f"- Succeeded: {fp['completed']}",
        f"- Upgraded to Full Agent: {fp['upgrades']}",
        f"- Success rate: {fp['success_rate']:.3f}",
        "", "## Paired Comparison", "",
        f"- Both-complete pairs: {p['pairs']}",
        f"- Token delta (full − auto): {p['token_delta_full_minus_auto']:+,.0f}",
        f"- Turn delta (full − auto): {p['turn_delta_full_minus_auto']:+.1f}",
        f"- Duration delta (full − auto, ms): {p['duration_delta_ms']:+,.0f}",
        "",
        "> Positive delta = automatic mode used fewer resources than /full.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return list(data["tasks"])


def main() -> int:
    parser = argparse.ArgumentParser(description="QQCode AB/BA benchmark runner")
    parser.add_argument("--verify-fixtures", action="store_true",
                        help="Provision repos and verify baseline tests fail (no agent)")
    parser.add_argument("--execute", action="store_true",
                        help="Run full AB/BA experiment")
    parser.add_argument("--task", metavar="ID",
                        help="Run only this task id")
    parser.add_argument("--category", choices=["simple", "medium", "full_only"],
                        help="Run only tasks in this category")
    parser.add_argument("--only-measurable", action="store_true",
                        help="Run only fixtures whose hidden assertion is derivable from the "
                             "statement (see tasks/derivability.json). The other verdicts fail "
                             "for reasons the agent cannot control.")
    parser.add_argument("--model", default="",
                        help="Pin every tier to this model id (e.g. gpt-5.6-luna)")
    parser.add_argument("--provider", default="", choices=["", "openai", "anthropic"],
                        help="Override provider; defaults to DEFAULT_PROVIDER in .env")
    parser.add_argument("--effort", default="", choices=["", "low", "medium", "high"],
                        help="Reasoning effort (OpenAI path only)")
    parser.add_argument("--cycles", type=int, default=1,
                        help="AB/BA cycles per task (default 1)")
    parser.add_argument("--tasks-file", default=str(TASKS_PATH),
                        help="Path to real_tasks_v2.json")
    args = parser.parse_args()

    if not args.verify_fixtures and not args.execute:
        parser.print_help()
        return 0

    tasks = _load_tasks(Path(args.tasks_file))
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]
        if not tasks:
            print(f"Task not found: {args.task}")
            return 1
    if args.category:
        tasks = [t for t in tasks if t.get("category") == args.category]
    if args.only_measurable:
        # `load_derivability` degrades to `{}` when the audit file is missing, which
        # is right for reporting (no exclusions) but wrong here: silently running all
        # 15 fixtures under a flag that asked for 5 would report unmeasurable
        # fixtures as capability data. Refuse instead.
        audit = load_derivability()
        if not audit:
            print(f"--only-measurable requires the audit file: {DERIVABILITY_PATH}")
            return 1
        tasks = [t for t in tasks if audit.get(t["id"]) == MEASURABLE_VERDICT]
        if not tasks:
            print(f"No fixture has verdict {MEASURABLE_VERDICT!r} in {DERIVABILITY_PATH}")
            return 1

    print(f"Tasks: {len(tasks)}")
    for t in tasks:
        print(f"  [{t.get('category','?'):10}] {t['id']}")

    # Output directory
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tag = f"qqcode-{'verify' if args.verify_fixtures else 'abba'}-{ts}"
    out_dir = RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    # Shared, persistent caches: reused across invocations rather than rebuilt
    # per run. Repos are ~20MB each and venvs ~50MB, so re-creating them every
    # time was both slow and the main exposure to transient network faults.
    cache_dir = REPO_CACHE_DIR
    venv_cache = VENV_CACHE_DIR

    config = Config.from_env()
    cache = RepoCache(cache_dir)

    print(f"\nResults: {out_dir}")
    print(f"Repo cache: {cache_dir}")
    print(f"Venv cache: {venv_cache}")
    print("Fetching HuggingFace fixtures …")
    fixtures: dict[str, FixtureMaterial] = {}
    for task in tasks:
        try:
            fixtures[task["id"]] = _fetch_fixture(task, cache=cache)
            print(f"  ✓ {task['id']}")
        except Exception as exc:
            print(f"  ✗ {task['id']}: {exc}")
            return 1

    schedule = (
        [(t, "automatic", "preflight", 1) for t in tasks]
        if args.verify_fixtures
        else _schedule(tasks, args.cycles)
    )
    print(f"\nScheduled {len(schedule)} runs\n")

    cfg_meta = {
        "mode": "verify_fixtures" if args.verify_fixtures else "ab_ba",
        # Record only what was actually passed into run_task, so report
        # provenance cannot drift from the request that was really sent.
        "model": args.model,
        "provider": args.provider or config.default_provider,
        "reasoning_effort": args.effort,
        "cycles": args.cycles,
        "tasks": len(tasks),
        "timestamp": ts,
    }

    records: list[RunRecord] = []
    for seq, (task, mode_label, order, cycle) in enumerate(schedule, 1):
        tid = task["id"]
        print(f"[{seq:02d}/{len(schedule)}] {tid} mode={mode_label} order={order}", flush=True)

        venv_python = _ensure_venv(task, venv_cache)
        run_dir = out_dir / "runs" / f"{seq:03d}-{tid}-{mode_label}-{order}-c{cycle}"
        run_dir.mkdir(parents=True, exist_ok=True)

        rec = _run_one(
            task=task,
            fixture=fixtures[tid],
            mode_label=mode_label,
            order=order,
            cycle=cycle,
            cache=cache,
            venv_python=venv_python,
            run_dir=run_dir,
            config=config,
            execute=args.execute,
            model=args.model or None,
            provider=args.provider or None,
            reasoning_effort=args.effort or None,
        )
        records.append(rec)

        status = "✓" if (rec.behavioral_complete if args.verify_fixtures else rec.agent_success) else "✗"
        print(f"  {status} behavioral={rec.behavioral_complete} "
              f"agent_success={rec.agent_success} "
              f"mode_used={rec.mode_used} "
              f"tokens={rec.tokens_total:,} "
              f"turns={rec.turns_used} "
              f"dur={rec.duration_ms:.0f}ms")
        if rec.fastpath_reason:
            print(f"  fastpath_escalated: {rec.fastpath_reason}")
        if rec.error:
            print(f"  error: {rec.error}")

        # Flush runs incrementally
        runs_json = out_dir / "runs.json"
        runs_json.write_text(json.dumps({"runs": [r.to_dict() for r in records]}, indent=2))

    report = _build_report(records, cfg_meta)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "report.md").write_text(_markdown(report))

    print(f"\nReport: {out_dir / 'report.md'}")
    if args.verify_fixtures:
        ok = sum(1 for r in records if r.behavioral_complete)
        print(f"Fixtures verified: {ok}/{len(records)} (baseline correctly fails)")
    else:
        a = report["all_runs"]["automatic"]
        f = report["all_runs"]["full"]
        print(f"Automatic: {a['behavioral_complete']}/{a['attempted']} "
              f"({a['behavioral_rate']:.0%} behavioral)")
        print(f"Full:      {f['behavioral_complete']}/{f['attempted']} "
              f"({f['behavioral_rate']:.0%} behavioral)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
