"""Hidden acceptance layer: externally injected behavioral verification."""

from qqcode.acceptance.harness import (
    ACCEPTANCE_DIR,
    RUNNER_ARTIFACTS,
    AcceptanceHarness,
    AcceptanceResult,
    AcceptanceTest,
    all_passed,
    filter_acceptance_paths,
    first_failure,
    is_acceptance_path,
)

__all__ = [
    "ACCEPTANCE_DIR",
    "RUNNER_ARTIFACTS",
    "AcceptanceHarness",
    "AcceptanceResult",
    "AcceptanceTest",
    "all_passed",
    "filter_acceptance_paths",
    "first_failure",
    "is_acceptance_path",
]
