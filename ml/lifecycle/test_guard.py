"""Explicit local guardrails around ReturnGuard's consumed official test."""

from __future__ import annotations

import os
from pathlib import Path

FINAL_TEST_ENVIRONMENT_FLAG = "RETURNGUARD_ALLOW_FINAL_TEST_RERUN"


def require_final_test_rerun_authorization(*, allow_flag: bool, reason: str | None) -> None:
    """Fail closed unless all documented exceptional rerun controls are present."""
    if not allow_flag:
        raise PermissionError("Final test rerun requires --allow-final-test-rerun")
    if os.environ.get(FINAL_TEST_ENVIRONMENT_FLAG) != "1":
        raise PermissionError(f"Final test rerun requires {FINAL_TEST_ENVIRONMENT_FLAG}=1")
    if not reason or not reason.strip():
        raise PermissionError("Final test rerun requires a documented --reason")


def assert_normal_lifecycle_path(path: Path) -> None:
    """Reject accidental final-test filenames in normal lifecycle helpers."""
    if "testing" in path.name or "final_test" in path.name:
        raise ValueError("Normal lifecycle commands must not access official test data")
