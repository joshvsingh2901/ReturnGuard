"""Small lifecycle checks that do not require ASOS raw data."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml.lifecycle.reproducibility import (
    compare_reference_predictions,
    create_reference_fixture,
    select_reference_positions,
    validation_metric_deltas,
)


def test_validation_tolerances_are_semantic_not_binary():
    expected = {"roc_auc": 0.65, "log_loss": 0.64, "brier": 0.23, "ece": 0.01}
    observed = {"roc_auc": 0.651, "log_loss": 0.641, "brier": 0.231, "ece": 0.011}
    deltas, passed = validation_metric_deltas(observed, expected)
    assert passed
    assert deltas["roc_auc"] == pytest.approx(0.001)


def test_reference_fixture_uses_positions_not_identifiers(synthetic_df_with_missing):
    positions = select_reference_positions(synthetic_df_with_missing)
    probabilities = np.linspace(0.2, 0.8, len(positions))
    fixture = create_reference_fixture(synthetic_df_with_missing, probabilities)
    reloaded = json.loads(json.dumps(fixture, sort_keys=True))
    diagnostics, passed = compare_reference_predictions(probabilities, reloaded)

    assert passed
    assert diagnostics["max_abs_delta"] == 0.0
    assert "hash(customerId)" not in fixture
    assert fixture["row_positions"] == positions
