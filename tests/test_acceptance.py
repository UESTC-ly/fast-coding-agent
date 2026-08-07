"""Tests for AcceptanceHarness: file injection, execution, cleanup."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from qqcode.acceptance import (
    ACCEPTANCE_DIR,
    RUNNER_ARTIFACTS,
    TRUST_WARNING,
    AcceptanceHarness,
    AcceptanceResult,
    AcceptanceTest,
    all_passed,
    filter_acceptance_paths,
    first_failure,
    is_acceptance_path,
    reset_trust_warning,
)


def test_single_test_passes_and_cleans_up() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test = AcceptanceTest(
            name="smoke",
            command=("sh", "-c", f"cat {ACCEPTANCE_DIR}/marker.txt | grep OK"),
            files={"marker.txt": "OK\n"},
        )
        harness = AcceptanceHarness([test])

        results = harness.run(root)

        assert len(results) == 1
        assert results[0].passed
        assert results[0].exit_code == 0
        assert not (root / ACCEPTANCE_DIR).exists()


def test_multiple_tests_stop_at_first_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        harness = AcceptanceHarness(
            [
                AcceptanceTest(name="pass", command=("true",)),
                AcceptanceTest(name="fail", command=("false",)),
                AcceptanceTest(name="never_run", command=("true",)),
            ]
        )

        results = harness.run(root)

        assert len(results) == 2
        assert results[0].passed
        assert not results[1].passed


def test_empty_harness_returns_empty_results() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        harness = AcceptanceHarness([])
        results = harness.run(Path(tmp))
        assert results == []
        assert all_passed(results)


def test_timeout_marks_result_as_timed_out() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        test = AcceptanceTest(
            name="hang", command=("sleep", "300"), timeout=0.1
        )
        harness = AcceptanceHarness([test])

        results = harness.run(Path(tmp))

        assert len(results) == 1
        assert not results[0].passed
        assert results[0].timed_out
        assert "timed out" in results[0].stderr.lower()


def test_nonexistent_command_is_recorded_as_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        test = AcceptanceTest(name="missing", command=("this-does-not-exist",))
        harness = AcceptanceHarness([test])

        results = harness.run(Path(tmp))

        assert len(results) == 1
        assert not results[0].passed
        assert "could not execute" in results[0].stderr


def test_injected_files_land_under_acceptance_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test = AcceptanceTest(
            name="file_check",
            command=("cat", f"{ACCEPTANCE_DIR}/nested/data.txt"),
            files={"nested/data.txt": "content\n"},
        )
        harness = AcceptanceHarness([test])

        results = harness.run(root)

        assert results[0].passed
        assert results[0].stdout == "content\n"


def test_cleanup_removes_runner_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "junk.pyc").write_text("x")
        (root / "subdir").mkdir()
        (root / "subdir" / ".pytest_cache").mkdir()

        # Cleanup needs at least one test to trigger
        harness = AcceptanceHarness([AcceptanceTest(name="noop", command=("true",))])
        harness.run(root)

        assert not (root / "__pycache__").exists()
        assert not (root / "subdir" / ".pytest_cache").exists()


def test_duplicate_test_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        AcceptanceHarness(
            [
                AcceptanceTest(name="dupe", command=("true",)),
                AcceptanceTest(name="dupe", command=("true",)),
            ]
        )


def test_acceptance_test_rejects_absolute_file_path() -> None:
    with pytest.raises(ValueError, match="relative"):
        AcceptanceTest(
            name="bad",
            command=("true",),
            files={"/etc/passwd": "nope"},
        )


def test_acceptance_test_rejects_dotdot_traversal() -> None:
    with pytest.raises(ValueError, match="relative"):
        AcceptanceTest(
            name="bad",
            command=("true",),
            files={"../escape.txt": "nope"},
        )


def test_acceptance_test_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AcceptanceTest(name="", command=("true",))


def test_acceptance_test_rejects_empty_command() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AcceptanceTest(name="bad", command=())


def test_acceptance_test_rejects_nonpositive_timeout() -> None:
    with pytest.raises(ValueError, match="positive"):
        AcceptanceTest(name="bad", command=("true",), timeout=0.0)


def test_is_acceptance_path_detects_harness_dir() -> None:
    assert is_acceptance_path(f"{ACCEPTANCE_DIR}/foo.txt")
    assert is_acceptance_path(f"{ACCEPTANCE_DIR}/nested/bar.txt")
    assert not is_acceptance_path("src/main.py")


def test_is_acceptance_path_detects_runner_artifacts() -> None:
    for artifact in RUNNER_ARTIFACTS:
        assert is_acceptance_path(artifact)
        assert is_acceptance_path(f"{artifact}/file.pyc")
        # Nested artifacts like subdir/__pycache__ are caught by cleanup's rglob,
        # but is_acceptance_path only checks top-level names for diff filtering.
        # That's correct: the diff never sees nested caches because cleanup runs
        # before the final snapshot.


def test_filter_acceptance_paths_drops_harness_files() -> None:
    paths = {
        "src/main.py",
        f"{ACCEPTANCE_DIR}/test.py",
        "__pycache__/file.pyc",
        "real.txt",
    }
    assert filter_acceptance_paths(paths) == {"src/main.py", "real.txt"}


def test_all_passed_with_empty_results() -> None:
    assert all_passed([])


def test_all_passed_with_mixed_results() -> None:
    results = [
        AcceptanceResult(name="a", passed=True, exit_code=0, stdout="", stderr=""),
        AcceptanceResult(name="b", passed=False, exit_code=1, stdout="", stderr=""),
    ]
    assert not all_passed(results)


def test_first_failure_returns_none_when_all_passed() -> None:
    results = [
        AcceptanceResult(name="a", passed=True, exit_code=0, stdout="", stderr=""),
    ]
    assert first_failure(results) is None


def test_first_failure_returns_the_failing_result() -> None:
    pass_result = AcceptanceResult(name="a", passed=True, exit_code=0, stdout="", stderr="")
    fail_result = AcceptanceResult(name="b", passed=False, exit_code=1, stdout="", stderr="")
    results = [pass_result, fail_result]
    assert first_failure(results) is fail_result


def test_acceptance_result_diagnostic_truncates_from_tail() -> None:
    result = AcceptanceResult(
        name="test",
        passed=False,
        exit_code=1,
        stdout="x" * 3000,
        stderr="y" * 3000,
    )
    diag = result.diagnostic(limit=100)
    assert len(diag["acceptance_stdout"]) == 100
    assert len(diag["acceptance_stderr"]) == 100
    assert diag["acceptance_stdout"] == "x" * 100
    assert diag["acceptance_stderr"] == "y" * 100


# --- trust-level warning (R10) -------------------------------------------
# The harness runs acceptance commands outside CommandGuard by design. That
# choice was only ever documented in a docstring, which the person at risk —
# whoever runs a suite they did not write — never reads. These tests pin the
# warning to the moment a suite actually executes.


def test_running_a_suite_warns_about_the_trust_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reset_trust_warning()
    harness = AcceptanceHarness([
        AcceptanceTest(name="t", command=("python3", "-c", "pass")),
    ])
    with tempfile.TemporaryDirectory() as tmp:
        harness.run(Path(tmp))
    assert TRUST_WARNING in capsys.readouterr().err


def test_trust_warning_is_emitted_only_once_per_process(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reset_trust_warning()
    harness = AcceptanceHarness([
        AcceptanceTest(name="t", command=("python3", "-c", "pass")),
    ])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        harness.run(root)
        capsys.readouterr()  # discard the first warning
        harness.run(root)
    assert TRUST_WARNING not in capsys.readouterr().err


def test_empty_harness_does_not_warn(capsys: pytest.CaptureFixture[str]) -> None:
    """No suite ran, so there is nothing to warn about — and a spurious warning
    would train users to ignore the real one."""
    reset_trust_warning()
    with tempfile.TemporaryDirectory() as tmp:
        AcceptanceHarness([]).run(Path(tmp))
    assert TRUST_WARNING not in capsys.readouterr().err
