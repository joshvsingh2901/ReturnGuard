"""Lightweight guards for the one-shot frozen-A3 final evaluation protocol."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data.schema import EVENT_CUST_COL, EVENT_PROD_COL, TARGET_COL
from ml.evaluation.final_protocol import (
    COLD_START_LABELS,
    FINAL_TRAINING_ROUNDS,
    final_xgb_params_from_freeze,
    predeclared_test_slices,
    transform_test_with_train_fitted_pipeline,
    validate_final_feature_manifest,
    validate_a3_freeze_spec,
)
from ml.explainability.grouping import FREQUENCY_FEATURE_NAMES
from ml.evaluation.metrics import compute_metrics


ROOT = Path(__file__).parent.parent


@pytest.fixture
def freeze_spec():
    return json.loads((ROOT / "reports" / "stage3_a3_freeze_spec.json").read_text())


def test_freeze_spec_is_a3_without_frequency_or_calibration(freeze_spec):
    validate_a3_freeze_spec(freeze_spec)
    assert freeze_spec["model_name"] == "A3"
    assert freeze_spec["calibration_policy"].startswith("none")
    assert freeze_spec["final_training_procedure"]["fixed_boosting_rounds"] == FINAL_TRAINING_ROUNDS


def test_final_training_params_are_fixed_rounds_not_test_early_stopping(freeze_spec):
    params = final_xgb_params_from_freeze(freeze_spec)
    assert params["n_estimators"] == FINAL_TRAINING_ROUNDS
    assert "early_stopping_rounds" not in params


def test_stage3_manifest_matches_freeze_and_has_no_frequency_features(freeze_spec):
    summary = json.loads((ROOT / "reports" / "stage3_summary.json").read_text())
    manifest = summary["feature_manifests"]["a3"]
    names = {entry["transformed_feature"] for entry in manifest["entries"]}
    assert manifest["manifest_hash"] == freeze_spec["feature_manifest_hash"]
    assert manifest["feature_count"] == 41
    assert not names & FREQUENCY_FEATURE_NAMES
    assert not any("hash(" in name or TARGET_COL in name for name in names)
    validate_final_feature_manifest(manifest, freeze_spec)


def test_freeze_validation_fails_closed_on_frequency_policy(freeze_spec):
    invalid = copy.deepcopy(freeze_spec)
    invalid["feature_policy"] = "A3 with frequency features"
    with pytest.raises(ValueError, match="prohibit frequency"):
        validate_a3_freeze_spec(invalid)


def test_test_transform_uses_only_a_train_fitted_transform_method():
    class TransformOnly:
        def __init__(self):
            self.seen_columns = None

        def fit(self, X):  # pragma: no cover - should never be invoked
            raise AssertionError("Test transform must not fit preprocessing")

        def transform(self, X):
            self.seen_columns = list(X.columns)
            return np.ones((len(X), 2), dtype=np.float32)

    pipeline = TransformOnly()
    test_df = pd.DataFrame({"feature": [1, 2], TARGET_COL: [0, 1]})
    transformed = transform_test_with_train_fitted_pipeline(pipeline, test_df)
    assert transformed.shape == (2, 2)
    assert pipeline.seen_columns == ["feature"]


def test_predeclared_slice_schema_and_counts_reconcile():
    test_df = pd.DataFrame(
        {
            EVENT_CUST_COL: [1, 2, 1, 3],
            EVENT_PROD_COL: [10, 10, 11, 12],
            TARGET_COL: [0, 1, 1, 0],
            "productType": ["A", "A", "__MISSING__", "B"],
            "isMale": [1.0, 1.0, np.nan, np.nan],
        }
    )
    report = predeclared_test_slices(
        test_df,
        np.array([0.1, 0.8, 0.7, 0.2]),
        train_customer_ids={1},
        train_product_ids={10},
    )
    assert tuple(report["cold_start"]) == COLD_START_LABELS
    assert report["reconciliation"] == {
        "test_row_count": 4,
        "cold_start_total": 4,
        "product_coverage_total": 4,
        "customer_coverage_total": 4,
    }
    for family in ("cold_start", "product_coverage", "customer_coverage"):
        for result in report[family].values():
            assert {"n", "prevalence", "roc_auc", "log_loss", "brier", "status"} <= set(result)


def test_predeclared_overall_metric_schema_is_complete():
    metrics = compute_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.4, 0.8]))
    assert {
        "n", "prevalence", "roc_auc", "log_loss", "brier", "pr_auc",
        "accuracy", "precision", "recall", "f1", "confusion_matrix",
    } <= set(metrics)
