"""Hidden acceptance layer: externally injected behavioral verification."""

from qqcode.acceptance.harness import (
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

__all__ = [
    "ACCEPTANCE_DIR",
    "RUNNER_ARTIFACTS",
    "TRUST_WARNING",
    "AcceptanceHarness",
    "AcceptanceResult",
    "AcceptanceTest",
    "all_passed",
    "filter_acceptance_paths",
    "first_failure",
    "is_acceptance_path",
    "reset_trust_warning",
]
