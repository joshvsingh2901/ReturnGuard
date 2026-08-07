"""Fail-closed helpers for the predeclared frozen-A3 test evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ml.data.joins import has_product_node_mask
from ml.data.schema import EVENT_CUST_COL, EVENT_PROD_COL, TARGET_COL
from ml.evaluation.metrics import compute_metrics
from ml.explainability.grouping import FREQUENCY_FEATURE_NAMES
from ml.features.preprocessing import get_feature_names
from ml.governance.feature_manifest import build_feature_manifest
from ml.models.gbdt import DEFAULT_XGB_PARAMS, build_feature_pipeline


FINAL_MODEL_NAME = "A3"
FINAL_TRAINING_ROUNDS = 659
SMALL_SLICE_N = 5_000
COLD_START_LABELS = (
    "known_customer_known_product",
    "new_customer_known_product",
    "known_customer_new_product",
    "new_customer_new_product",
)


@dataclass
class FrozenA3Fit:
    """Fitted final-training components and their verified manifest."""

    feature_pipeline: object
    model: XGBClassifier
    feature_names: list[str]
    feature_manifest: dict
    fitted_xgb_params: dict


def validate_a3_freeze_spec(freeze_spec: dict) -> None:
    """Reject any freeze spec that is not the predeclared A3 protocol."""
    if freeze_spec.get("model_name") != FINAL_MODEL_NAME:
        raise ValueError("Final evaluation is restricted to frozen A3")
    if freeze_spec.get("model_status") != "MODEL_OF_RECORD_CONSERVATIVE_FREEZE_CANDIDATE":
        raise ValueError("Freeze spec is not the approved A3 model-of-record candidate")
    if freeze_spec.get("calibration_policy") != "none; raw scores remain dataset-conditional":
        raise ValueError("Frozen A3 must not fit or apply a calibrator")
    if freeze_spec.get("xgboost_params") != DEFAULT_XGB_PARAMS:
        raise ValueError("Frozen XGBoost parameters differ from the committed A3 configuration")
    procedure = freeze_spec.get("final_training_procedure", {})
    if procedure.get("fixed_boosting_rounds") != FINAL_TRAINING_ROUNDS:
        raise ValueError("Frozen A3 requires exactly 659 final boosting rounds")
    policy = freeze_spec.get("feature_policy", "")
    if "no frequency features" not in policy:
        raise ValueError("Frozen A3 feature policy must prohibit frequency features")


def final_xgb_params_from_freeze(freeze_spec: dict) -> dict:
    """Translate the early-stopped development config to fixed-round fitting."""
    validate_a3_freeze_spec(freeze_spec)
    params = dict(freeze_spec["xgboost_params"])
    # Early stopping selected round 659 during development. Final training
    # uses that predeclared count and no evaluation set, never the test set.
    params.pop("early_stopping_rounds")
    params["n_estimators"] = freeze_spec["final_training_procedure"]["fixed_boosting_rounds"]
    return params


def train_frozen_a3_on_training_events(train_df: pd.DataFrame, freeze_spec: dict) -> FrozenA3Fit:
    """Fit preprocessing and fixed-round A3 using official training rows only."""
    validate_a3_freeze_spec(freeze_spec)
    if TARGET_COL not in train_df:
        raise ValueError("Final training data must contain the training target")

    train_features = train_df.drop(columns=[TARGET_COL])
    pipeline = build_feature_pipeline(
        artifact_safe=True, include_derived=True, include_frequency=False
    )
    X_train = np.asarray(pipeline.fit_transform(train_features), dtype=np.float32)
    feature_names = get_feature_names(pipeline.named_steps["preprocess"])
    manifest = build_feature_manifest(feature_names, FINAL_MODEL_NAME)
    if manifest["feature_manifest_hash"] != freeze_spec["feature_manifest_hash"]:
        raise ValueError("Final-training feature manifest does not match the freeze spec")
    if manifest["feature_count"] != 41:
        raise ValueError("Frozen A3 must have exactly 41 approved features")
    if set(feature_names) & FREQUENCY_FEATURE_NAMES:
        raise ValueError("Frozen A3 must not contain product-frequency features")
    if any("hash(" in name or TARGET_COL in name for name in feature_names):
        raise ValueError("Raw identifier or target reached final A3 feature matrix")

    fitted_params = final_xgb_params_from_freeze(freeze_spec)
    model = XGBClassifier(**fitted_params)
    model.fit(X_train, train_df[TARGET_COL].to_numpy(), verbose=False)
    if model.get_booster().num_boosted_rounds() != FINAL_TRAINING_ROUNDS:
        raise AssertionError("Final A3 did not train the frozen 659 boosting rounds")
    return FrozenA3Fit(pipeline, model, feature_names, manifest, fitted_params)


def transform_test_with_train_fitted_pipeline(feature_pipeline, test_df: pd.DataFrame) -> np.ndarray:
    """Transform test features only; this function intentionally never fits."""
    if TARGET_COL in test_df:
        test_df = test_df.drop(columns=[TARGET_COL])
    return np.asarray(feature_pipeline.transform(test_df), dtype=np.float32)


def _safe_slice_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    """Return the predeclared slice metric subset, including empty slices."""
    n = len(y_true)
    if n == 0:
        return {
            "n": 0,
            "prevalence": None,
            "roc_auc": None,
            "log_loss": None,
            "brier": None,
            "status": "empty_slice",
        }
    metrics = compute_metrics(y_true, probability)
    return {
        "n": metrics["n"],
        "prevalence": metrics["prevalence"],
        "roc_auc": metrics["roc_auc"],
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier"],
        "status": "single_class" if metrics["roc_auc"] is None else "ok",
        "small_sample_warning": n < SMALL_SLICE_N,
    }


def predeclared_test_slices(
    test_df: pd.DataFrame,
    test_probability: np.ndarray,
    train_customer_ids: set,
    train_product_ids: set,
) -> dict:
    """Calculate exactly the cold-start and coverage slices in the protocol."""
    y_true = test_df[TARGET_COL].to_numpy()
    probability = np.asarray(test_probability, dtype=float)
    if len(test_df) != len(probability):
        raise ValueError("Test probabilities must align one-to-one with test rows")

    known_customer = test_df[EVENT_CUST_COL].isin(train_customer_ids)
    known_product = test_df[EVENT_PROD_COL].isin(train_product_ids)
    product_available = has_product_node_mask(test_df)
    # isMale is null only when no customer node is joined; yearOfBirth can be
    # null for a present node because the 1900 sentinel is cleaned to NaN.
    customer_available = test_df["isMale"].notna()

    masks = {
        "cold_start": {
            "known_customer_known_product": known_customer & known_product,
            "new_customer_known_product": ~known_customer & known_product,
            "known_customer_new_product": known_customer & ~known_product,
            "new_customer_new_product": ~known_customer & ~known_product,
        },
        "product_coverage": {
            "product_node_available": product_available,
            "product_node_missing": ~product_available,
        },
        "customer_coverage": {
            "customer_node_available": customer_available,
            "customer_node_missing": ~customer_available,
        },
    }
    results = {
        family: {
            label: _safe_slice_metrics(y_true[mask.to_numpy()], probability[mask.to_numpy()])
            for label, mask in family_masks.items()
        }
        for family, family_masks in masks.items()
    }
    for family, family_results in results.items():
        total = sum(entry["n"] for entry in family_results.values())
        if total != len(test_df):
            raise AssertionError(f"{family} slices do not reconcile to the test-row count")
    results["reconciliation"] = {
        "test_row_count": int(len(test_df)),
        "cold_start_total": sum(item["n"] for item in results["cold_start"].values()),
        "product_coverage_total": sum(item["n"] for item in results["product_coverage"].values()),
        "customer_coverage_total": sum(item["n"] for item in results["customer_coverage"].values()),
    }
    return results
