"""Tests for the explicit exceptional final-test rerun guard."""

from __future__ import annotations

import pytest

from ml.lifecycle.test_guard import require_final_test_rerun_authorization


def test_final_test_rerun_requires_cli_flag(monkeypatch):
    monkeypatch.setenv("RETURNGUARD_ALLOW_FINAL_TEST_RERUN", "1")
    with pytest.raises(PermissionError, match="--allow-final-test-rerun"):
        require_final_test_rerun_authorization(allow_flag=False, reason="defect")


def test_final_test_rerun_requires_environment_flag(monkeypatch):
    monkeypatch.delenv("RETURNGUARD_ALLOW_FINAL_TEST_RERUN", raising=False)
    with pytest.raises(PermissionError, match="RETURNGUARD_ALLOW_FINAL_TEST_RERUN"):
        require_final_test_rerun_authorization(allow_flag=True, reason="defect")


def test_final_test_rerun_requires_reason(monkeypatch):
    monkeypatch.setenv("RETURNGUARD_ALLOW_FINAL_TEST_RERUN", "1")
    with pytest.raises(PermissionError, match="--reason"):
        require_final_test_rerun_authorization(allow_flag=True, reason="")
