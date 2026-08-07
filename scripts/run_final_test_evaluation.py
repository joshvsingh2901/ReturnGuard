"""Run the one predeclared official-test evaluation of frozen ReturnGuard A3.

This is evaluation-only. It verifies the committed A3 freeze before any test
prediction, trains A3 on official training features only for exactly 659
rounds, scores the official test features once, writes metrics, then stops.
It never tunes, selects models, fits a calibrator, or evaluates A4.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from ml.data.joins import build_joined_test_frame_for_final_evaluation, build_joined_training_frame
from ml.data.schema import EVENT_CUST_COL, EVENT_PROD_COL, TARGET_COL
from ml.evaluation.calibration import stage3_calibration_report
from ml.evaluation.final_protocol import (
    predeclared_test_slices,
    train_frozen_a3_on_training_events,
    transform_test_with_train_fitted_pipeline,
    validate_a3_freeze_spec,
)
from ml.evaluation.metrics import compute_metrics
from ml.features.preprocessing import clean_year_of_birth
from ml.lifecycle.test_guard import require_final_test_rerun_authorization

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
FREEZE_SPEC_PATH = REPORTS_DIR / "stage3_a3_freeze_spec.json"
STAGE3_SUMMARY_PATH = REPORTS_DIR / "stage3_summary.json"
EVALUATION_DESCRIPTION = "held-out performance evaluation with prior aggregate test inspection"


def _json_default(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _assert_freeze_revision_is_available(freeze_spec: dict) -> None:
    revision = freeze_spec.get("code_version")
    if not revision:
        raise ValueError("Freeze spec does not record a source revision")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("Freeze-spec source revision is not an ancestor of this checkout")


def _load_and_verify_freeze() -> tuple[dict, dict]:
    if not FREEZE_SPEC_PATH.exists() or not STAGE3_SUMMARY_PATH.exists():
        raise FileNotFoundError("Stage 3 freeze spec and summary are required")
    freeze_spec = json.loads(FREEZE_SPEC_PATH.read_text())
    stage3 = json.loads(STAGE3_SUMMARY_PATH.read_text())
    validate_a3_freeze_spec(freeze_spec)
    _assert_freeze_revision_is_available(freeze_spec)
    manifest = stage3["feature_manifests"]["a3"]
    if manifest["manifest_hash"] != freeze_spec["feature_manifest_hash"]:
        raise ValueError("Stage 3 A3 manifest does not match the freeze spec")
    if manifest["feature_count"] != 41:
        raise ValueError("Stage 3 A3 did not record 41 approved features")
    if stage3["freeze_spec"]["freeze_spec_hash"] != freeze_spec["freeze_spec_hash"]:
        raise ValueError("Stage 3 summary and standalone freeze spec differ")
    return freeze_spec, stage3


def _package_versions() -> dict:
    packages = ["numpy", "pandas", "scikit-learn", "xgboost", "shap"]
    return {package: importlib.metadata.version(package) for package in packages}


def _validation_comparison(stage3: dict, test_metrics: dict, test_calibration: dict) -> dict:
    development = stage3["models"]["a3"]
    development_calibration = development["calibration"]["overall"]
    values = {
        "roc_auc": (development["metrics"]["roc_auc"], test_metrics["roc_auc"]),
        "log_loss": (development["metrics"]["log_loss"], test_metrics["log_loss"]),
        "brier": (development["metrics"]["brier"], test_metrics["brier"]),
        "ece": (development_calibration["ece"], test_calibration["ece"]),
    }
    return {
        "development_primary_validation_is_not_pristine": True,
        "development_metrics": {name: pair[0] for name, pair in values.items()},
        "official_test_metrics": {name: pair[1] for name, pair in values.items()},
        "absolute_deltas": {name: abs(pair[1] - pair[0]) for name, pair in values.items()},
        "interpretation": "Descriptive comparison only; it does not reopen model selection.",
    }


def _attempt_report_paths() -> tuple[Path, Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    revision = _git_revision()[:8]
    attempt_dir = REPORTS_DIR / "final_test_attempts" / f"{timestamp}__{revision}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return (
        attempt_dir / "final_test_metrics.json",
        attempt_dir / "final_test_slices.json",
        attempt_dir / "final_test_calibration.json",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-final-test-rerun", action="store_true")
    parser.add_argument("--reason", help="Documented corrective reason for an exceptional rerun")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    require_final_test_rerun_authorization(
        allow_flag=args.allow_final_test_rerun,
        reason=args.reason,
    )
    metrics_path, slices_path, calibration_path = _attempt_report_paths()
    print("ReturnGuard — exceptional frozen A3 official test rerun")
    print(f"Reason: {args.reason}")
    print(f"Git revision: {_git_revision()}")
    print(f"Evaluation posture: {EVALUATION_DESCRIPTION}")
    freeze_spec, stage3 = _load_and_verify_freeze()
    print("Freeze verified before test prediction: A3 / 41 features / no frequency / 659 rounds / no calibrator")

    # Final training occurs before the official-test frame is loaded. All
    # learned preprocessing statistics and the donor pool come from training.
    print("Building official training frame and fitting frozen A3 on all training events...")
    train_df = clean_year_of_birth(build_joined_training_frame())
    fit = train_frozen_a3_on_training_events(train_df, freeze_spec)
    train_customer_ids = set(train_df[EVENT_CUST_COL])
    train_product_ids = set(train_df[EVENT_PROD_COL])
    print(f"  training rows={len(train_df):,}; boosted rounds={fit.model.get_booster().num_boosted_rounds()}")

    # Test labels are not read into an evaluation array until after the
    # training-fitted pipeline has transformed features and scored once.
    print("Building official test frame and scoring once with training-fitted preprocessing...")
    test_df = clean_year_of_birth(build_joined_test_frame_for_final_evaluation())
    X_test = transform_test_with_train_fitted_pipeline(fit.feature_pipeline, test_df)
    test_probability = fit.model.predict_proba(X_test)[:, 1]
    y_test = test_df[TARGET_COL].to_numpy()

    test_metrics = compute_metrics(y_test, test_probability)
    calibration = stage3_calibration_report(y_test, test_probability)
    slices = predeclared_test_slices(test_df, test_probability, train_customer_ids, train_product_ids)
    comparison = _validation_comparison(stage3, test_metrics, calibration)
    execution = {
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_description": EVALUATION_DESCRIPTION,
        "git_revision": _git_revision(),
        "freeze_source_revision": freeze_spec["code_version"],
        "freeze_spec_hash": freeze_spec["freeze_spec_hash"],
        "feature_manifest_hash": fit.feature_manifest["manifest_hash"],
        "package_versions": _package_versions(),
        "training_row_count": int(len(train_df)),
        "test_row_count": int(len(test_df)),
        "calibrator_fitted": False,
        "test_probability_generation_count": 1,
    }
    metrics_report = {
        "evaluation_type": EVALUATION_DESCRIPTION,
        "model_name": "A3",
        "model_status": freeze_spec["model_status"],
        "frozen_model_spec": freeze_spec,
        "final_training": {
            "preprocessing_steps": list(fit.feature_pipeline.named_steps),
            "feature_count": len(fit.feature_names),
            "feature_names": fit.feature_names,
            "fitted_xgboost_params": fit.fitted_xgb_params,
            "fixed_boosting_rounds": fit.model.get_booster().num_boosted_rounds(),
        },
        "execution": execution,
        "official_test_metrics": test_metrics,
        "validation_vs_test": comparison,
    }
    calibration_report = {
        "evaluation_type": EVALUATION_DESCRIPTION,
        "model_name": "A3",
        "execution": execution,
        "calibration": calibration,
        "policy": "No calibrator fitted or applied; scores remain dataset-conditional return-risk estimates.",
    }
    slices_report = {
        "evaluation_type": EVALUATION_DESCRIPTION,
        "model_name": "A3",
        "execution": execution,
        "slices": slices,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    _write_json(metrics_path, metrics_report)
    _write_json(slices_path, slices_report)
    _write_json(calibration_path, calibration_report)
    print(f"Official test ROC-AUC={test_metrics['roc_auc']:.6f} LogLoss={test_metrics['log_loss']:.6f} Brier={test_metrics['brier']:.6f}")
    print("Exceptional final evaluation reports written to a new attempt directory. Model development remains closed.")


if __name__ == "__main__":
    main()
