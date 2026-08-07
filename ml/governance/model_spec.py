"""Frozen model-of-record specifications for the post-Stage-3 handoff."""

from __future__ import annotations

import hashlib
import json


def build_model_freeze_spec(
    feature_manifest: dict,
    xgb_params: dict,
    best_iteration: int,
    code_version: str,
) -> dict:
    """Create the A3 model-of-record specification without fitting a model."""
    spec = {
        "model_status": "MODEL_OF_RECORD_CONSERVATIVE_FREEZE_CANDIDATE",
        "model_name": "A3",
        "model_family": "XGBoost binary logistic",
        "feature_policy": "A3 donor-imputed demographics, catalogue, suspicious price/discount, derived price; no frequency features",
        "preprocessing": {
            "customer_missingness": "stable customer-ID keyed joint donor imputation",
            "product_categoricals": "explicit __MISSING__ category",
            "product_numeric_missingness": "native XGBoost NaN routing",
            "year_of_birth_sentinel": "1900 converted to NaN",
        },
        "xgboost_params": xgb_params,
        "development_training_split": {
            "primary_validation": "customer-grouped deterministic MD5 buckets [0, 20)",
            "inner_train": "customer MD5 buckets [30, 100)",
            "early_stop": "customer MD5 buckets [20, 30)",
        },
        "final_training_procedure": {
            "official_test_not_run": True,
            "future_refit": "fit preprocessing and XGBoost on all training events using the frozen configuration and a predeclared fixed boosting-round count",
            "fixed_boosting_rounds": int(best_iteration) + 1,
        },
        "calibration_policy": "none; raw scores remain dataset-conditional",
        "explanation_policy": "Tree SHAP raw-margin explanations; user-facing return-risk factors are grouped and non-causal",
        "evaluation_metrics": ["ROC-AUC", "PR-AUC", "log loss", "Brier", "ECE", "calibration-in-the-large", "calibration slope"],
        "slice_definitions": [
            "product covered vs product missing",
            "known vs new product relative to fitted training fold",
            "original customer node present vs missing",
        ],
        "feature_manifest_hash": feature_manifest["manifest_hash"],
        "code_version": code_version,
    }
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    spec["freeze_spec_hash"] = hashlib.sha256(canonical).hexdigest()
    return spec
