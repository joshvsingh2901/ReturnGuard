"""Frozen A3 lifecycle validation, reference fixtures, and composite artifact."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.schema import TARGET_COL
from ml.evaluation.final_protocol import FINAL_TRAINING_ROUNDS, validate_a3_freeze_spec
from ml.explainability.grouping import FREQUENCY_FEATURE_NAMES
from ml.features.preprocessing import get_feature_names
from ml.governance.feature_manifest import build_feature_manifest
from ml.lifecycle.hashing import sha256_payload

MODEL_VERSION = "returnguard-a3-v1"
RUN_TYPES = {
    "frozen_model_rebuild",
    "reproducibility_check",
    "historical_final_evaluation",
    "historical_development_import",
}
METRIC_TOLERANCES = {"roc_auc": 0.003, "log_loss": 0.005, "brier": 0.003, "ece": 0.01}
REFERENCE_MEAN_ABS_TOLERANCE = 1e-6
REFERENCE_MAX_ABS_TOLERANCE = 1e-5
ARTIFACT_RELOAD_MAX_ABS_TOLERANCE = 1e-7


@dataclass
class A3CompositeArtifact:
    """Serving-ready bundle: train-fitted preprocessing plus frozen A3 booster."""

    feature_pipeline: object
    model: object
    feature_names: list[str]
    model_version: str = MODEL_VERSION

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        features = frame.drop(columns=[TARGET_COL], errors="ignore")
        return np.asarray(self.feature_pipeline.transform(features), dtype=np.float32)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.transform(frame))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def freeze_spec_sha256(freeze_spec: dict) -> str:
    unsigned = dict(freeze_spec)
    declared = unsigned.pop("freeze_spec_hash", None)
    calculated = sha256_payload(unsigned)
    if declared != calculated:
        raise ValueError("Stage 3 freeze specification hash is invalid")
    return calculated


def validate_lifecycle_manifest(manifest: dict, freeze_spec: dict) -> None:
    """Fail closed if lifecycle metadata drifts from the immutable freeze."""
    validate_a3_freeze_spec(freeze_spec)
    if manifest.get("semantic_model_version") != MODEL_VERSION:
        raise ValueError("Lifecycle manifest does not name returnguard-a3-v1")
    if manifest.get("model_family") != freeze_spec.get("model_family"):
        raise ValueError("Lifecycle model family differs from Stage 3 freeze")
    if manifest.get("xgboost_params") != freeze_spec.get("xgboost_params"):
        raise ValueError("Lifecycle XGBoost parameters differ from Stage 3 freeze")
    if manifest.get("fixed_boosting_rounds") != FINAL_TRAINING_ROUNDS:
        raise ValueError("Lifecycle manifest must retain exactly 659 rounds")
    if manifest.get("feature_manifest_hash") != freeze_spec.get("feature_manifest_hash"):
        raise ValueError("Lifecycle feature manifest differs from Stage 3 freeze")
    if manifest.get("freeze_spec_hash") != freeze_spec_sha256(freeze_spec):
        raise ValueError("Lifecycle freeze-spec hash differs from Stage 3 freeze")
    preprocessing = manifest.get("preprocessing", {})
    expected = {
        "artifact_safe_donor_imputation": True,
        "derived_price_features": True,
        "product_frequency_features": False,
        "calibrator": False,
    }
    if preprocessing != expected:
        raise ValueError("Lifecycle preprocessing policy differs from frozen A3")
    if manifest.get("ordered_feature_count") != 41:
        raise ValueError("Frozen A3 requires 41 approved transformed features")
    if manifest.get("governance_status") != "frozen":
        raise ValueError("Lifecycle manifest must retain frozen governance status")


def validate_fitted_feature_manifest(feature_names: list[str], lifecycle_manifest: dict) -> dict:
    manifest = build_feature_manifest(feature_names, "A3")
    if manifest["manifest_hash"] != lifecycle_manifest["feature_manifest_hash"]:
        raise ValueError("Fitted transformed feature order does not match frozen A3")
    if manifest["feature_count"] != 41:
        raise ValueError("Fitted A3 does not have 41 features")
    names = set(feature_names)
    if names & FREQUENCY_FEATURE_NAMES:
        raise ValueError("Frozen A3 contains prohibited product-frequency features")
    return manifest


def validation_metric_deltas(observed: dict, expected: dict) -> tuple[dict, bool]:
    deltas = {}
    passed = True
    for metric, tolerance in METRIC_TOLERANCES.items():
        if metric not in observed or metric not in expected:
            raise ValueError(f"Missing required validation metric: {metric}")
        delta = abs(float(observed[metric]) - float(expected[metric]))
        deltas[metric] = delta
        passed = passed and delta <= tolerance
    return deltas, passed


def select_reference_positions(frame: pd.DataFrame) -> dict[str, int]:
    """Select representative positions without persisting customer or product IDs."""
    customer_present = frame["isMale"].notna()
    product_present = frame["productType"] != "__MISSING__"
    combinations = {
        "customer_and_product_present": customer_present & product_present,
        "customer_missing_product_present": ~customer_present & product_present,
        "customer_present_product_missing": customer_present & ~product_present,
        "customer_and_product_missing": ~customer_present & ~product_present,
    }
    positions = {}
    for label, mask in combinations.items():
        matches = np.flatnonzero(mask.to_numpy())
        if len(matches):
            positions[label] = int(matches[0])
    if len(positions) < 3:
        raise ValueError("Training data lacks enough preprocessing cases for reference fixture")
    return positions


def reference_frame(frame: pd.DataFrame, positions: dict[str, int]) -> pd.DataFrame:
    return frame.iloc[list(positions.values())].copy()


def validate_reference_fixture(reference: dict) -> None:
    if reference.get("model_version") != MODEL_VERSION:
        raise ValueError("Reference fixture has the wrong model version")
    positions = reference.get("row_positions", {})
    if not positions or any(not isinstance(value, int) or value < 0 for value in positions.values()):
        raise ValueError("Reference fixture must contain non-sensitive row positions")
    expected = reference.get("expected_probabilities")
    if expected is not None and len(expected) != len(positions):
        raise ValueError("Reference fixture probabilities do not align with row positions")


def create_reference_fixture(frame: pd.DataFrame, probabilities: np.ndarray) -> dict:
    positions = select_reference_positions(frame)
    selected = reference_frame(frame, positions)
    return {
        "model_version": MODEL_VERSION,
        "fixture_version": "a3-training-reference-v1",
        "selection_policy": "first deterministic row position for each available customer/product coverage combination; no raw IDs or targets persisted",
        "row_positions": positions,
        "expected_probabilities": [float(value) for value in probabilities],
        "expected_feature_count": 41,
        "fixture_hash": sha256_payload(
            {
                "row_positions": positions,
                "expected_probabilities": [float(value) for value in probabilities],
                "feature_count": 41,
            }
        ),
        "selected_case_count": int(len(selected)),
    }


def compare_reference_predictions(observed: np.ndarray, reference: dict) -> tuple[dict, bool]:
    validate_reference_fixture(reference)
    expected = reference.get("expected_probabilities")
    if expected is None:
        return {"mean_abs_delta": None, "max_abs_delta": None, "reference_bootstrapped": False}, False
    delta = np.abs(np.asarray(observed, dtype=float) - np.asarray(expected, dtype=float))
    diagnostics = {
        "mean_abs_delta": float(delta.mean()),
        "max_abs_delta": float(delta.max()),
        "reference_bootstrapped": False,
    }
    return diagnostics, (
        diagnostics["mean_abs_delta"] <= REFERENCE_MEAN_ABS_TOLERANCE
        and diagnostics["max_abs_delta"] <= REFERENCE_MAX_ABS_TOLERANCE
    )


def environment_details(package_versions: dict) -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions,
    }


def model_config_hash(lifecycle_manifest: dict) -> str:
    return sha256_payload(
        {
            "xgboost_params": lifecycle_manifest["xgboost_params"],
            "fixed_boosting_rounds": lifecycle_manifest["fixed_boosting_rounds"],
            "preprocessing": lifecycle_manifest["preprocessing"],
            "feature_manifest_hash": lifecycle_manifest["feature_manifest_hash"],
        }
    )


def transformed_feature_names(feature_pipeline: object) -> list[str]:
    return get_feature_names(feature_pipeline.named_steps["preprocess"])
