"""Rebuild frozen ReturnGuard A3 without accessing official test data."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.joins import build_joined_training_frame
from ml.data.loaders import RAW_DIR
from ml.data.schema import TARGET_COL
from ml.data.splits import inner_early_stopping_split
from ml.evaluation.calibration import stage3_calibration_report
from ml.evaluation.final_protocol import train_frozen_a3_on_training_events
from ml.evaluation.metrics import compute_metrics
from ml.features.preprocessing import clean_year_of_birth
from ml.lifecycle.hashing import (
    git_provenance,
    sha256_file,
    verify_training_files,
)
from ml.lifecycle.reproducibility import (
    ARTIFACT_RELOAD_MAX_ABS_TOLERANCE,
    MODEL_VERSION,
    A3CompositeArtifact,
    compare_reference_predictions,
    create_reference_fixture,
    environment_details,
    load_json,
    model_config_hash,
    reference_frame,
    select_reference_positions,
    validate_fitted_feature_manifest,
    validate_lifecycle_manifest,
    validate_reference_fixture,
    validation_metric_deltas,
    write_json,
)
from ml.lifecycle.tracking import log_rebuild_run
from ml.models.gbdt import fit_gbdt

ROOT = Path(__file__).parent.parent
MANIFEST_DIR = ROOT / "artifacts" / "manifests"
DATASET_MANIFEST_PATH = MANIFEST_DIR / "asos-graphreturns-osf-c793h-v1.json"
LIFECYCLE_MANIFEST_PATH = MANIFEST_DIR / "returnguard-a3-v1.json"
REFERENCE_PATH = ROOT / "artifacts" / "reference" / "a3-reference.json"
FREEZE_SPEC_PATH = ROOT / "reports" / "stage3_a3_freeze_spec.json"
MODEL_CARD_PATH = ROOT / "docs" / "model-card.md"
REPORT_DIR = ROOT / "reports" / "reproducibility"


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scikit-learn", "xgboost", "shap", "mlflow", "joblib"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _pip_freeze(path: Path) -> None:
    output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    path.write_text(output)


def _validation_fit(train_df):
    fold = inner_early_stopping_split(train_df)
    inner_train = train_df.loc[fold == "inner_train"].copy()
    early_stop = train_df.loc[fold == "early_stop"].copy()
    primary_val = train_df.loc[fold == "primary_val"].copy()
    result = fit_gbdt(
        inner_train,
        early_stop,
        primary_val,
        artifact_safe=True,
        include_derived=True,
        include_frequency=False,
    )
    metrics = compute_metrics(primary_val[TARGET_COL], result.proba_val)
    metrics["ece"] = stage3_calibration_report(primary_val[TARGET_COL], result.proba_val)["ece"]
    return result, metrics, {
        "inner_train": int(len(inner_train)),
        "early_stop": int(len(early_stop)),
        "primary_val": int(len(primary_val)),
    }


def _temporary_component_hashes(composite: A3CompositeArtifact) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="returnguard-a3-hashes-") as temporary:
        temporary_path = Path(temporary)
        bundle = temporary_path / "composite.joblib"
        pipeline = temporary_path / "feature_pipeline.joblib"
        model = temporary_path / "xgb_model.joblib"
        joblib.dump(composite, bundle)
        joblib.dump(composite.feature_pipeline, pipeline)
        joblib.dump(composite.model, model)
        return {
            "composite_joblib_sha256": sha256_file(bundle),
            "feature_pipeline_joblib_sha256": sha256_file(pipeline),
            "xgb_model_joblib_sha256": sha256_file(model),
        }


def run_full(*, raw_dir: Path, bootstrap_reference: bool, register: bool) -> tuple[dict, dict]:
    # This path calls only training loaders.  It never imports the final-test
    # evaluator or any official-test loader.
    dataset_manifest = load_json(DATASET_MANIFEST_PATH)
    lifecycle_manifest = load_json(LIFECYCLE_MANIFEST_PATH)
    freeze_spec = load_json(FREEZE_SPEC_PATH)
    reference = load_json(REFERENCE_PATH)
    validate_lifecycle_manifest(lifecycle_manifest, freeze_spec)
    training_hashes = verify_training_files(dataset_manifest, raw_dir)
    if reference["row_positions"]:
        validate_reference_fixture(reference)

    provenance = git_provenance(ROOT)
    train_df = clean_year_of_birth(build_joined_training_frame(raw_dir=raw_dir))
    validation_fit, validation_metrics, split_counts = _validation_fit(train_df)
    validation_manifest = validate_fitted_feature_manifest(
        validation_fit.feature_names, lifecycle_manifest
    )
    metric_deltas, metrics_pass = validation_metric_deltas(
        validation_metrics,
        lifecycle_manifest["expected_development_validation_metrics"],
    )

    final_fit = train_frozen_a3_on_training_events(train_df, freeze_spec)
    final_manifest = validate_fitted_feature_manifest(final_fit.feature_names, lifecycle_manifest)
    composite = A3CompositeArtifact(final_fit.feature_pipeline, final_fit.model, final_fit.feature_names)

    if not reference["row_positions"]:
        positions = select_reference_positions(train_df)
        fixture = reference_frame(train_df, positions)
        probability = composite.predict_proba(fixture)[:, 1]
        if not bootstrap_reference:
            raise ValueError("Reference fixture has not been bootstrapped; rerun once with --bootstrap-reference")
        reference = create_reference_fixture(train_df, probability)
        write_json(REFERENCE_PATH, reference)
        reference_diagnostics = {
            "mean_abs_delta": 0.0,
            "max_abs_delta": 0.0,
            "reference_bootstrapped": True,
        }
        reference_pass = True
    else:
        fixture = reference_frame(train_df, reference["row_positions"])
        probability = composite.predict_proba(fixture)[:, 1]
        reference_diagnostics, reference_pass = compare_reference_predictions(probability, reference)

    with tempfile.TemporaryDirectory(prefix="returnguard-a3-reload-") as temporary:
        artifact_path = Path(temporary) / "returnguard-a3-composite.joblib"
        joblib.dump(composite, artifact_path)
        reloaded = joblib.load(artifact_path)
        reloaded_probability = reloaded.predict_proba(fixture)[:, 1]
        reload_max_delta = float(np.max(np.abs(probability - reloaded_probability)))
    artifact_reload_pass = reload_max_delta <= ARTIFACT_RELOAD_MAX_ABS_TOLERANCE
    artifact_hashes = _temporary_component_hashes(composite)

    status = "PASS" if metrics_pass and reference_pass and artifact_reload_pass else "FAIL"
    report = {
        "model_version": MODEL_VERSION,
        **provenance,
        "run_type": "frozen_model_rebuild",
        "dataset_version": lifecycle_manifest["dataset_version"],
        "dataset_hashes": training_hashes,
        "feature_manifest_hash": final_manifest["manifest_hash"],
        "freeze_spec_hash": lifecycle_manifest["freeze_spec_hash"],
        "model_config_hash": model_config_hash(lifecycle_manifest),
        "artifact_hashes": artifact_hashes,
        "environment": environment_details(_package_versions()),
        "row_counts": {"training": int(len(train_df)), "validation_fit_rows": int(sum(split_counts.values()))},
        "split_counts": split_counts,
        "metrics": {metric: float(validation_metrics[metric]) for metric in lifecycle_manifest["expected_development_validation_metrics"]},
        "metric_deltas_vs_reference": metric_deltas,
        "reference_prediction_deltas": reference_diagnostics,
        "artifact_reload": {
            "max_abs_delta": reload_max_delta,
            "tolerance": ARTIFACT_RELOAD_MAX_ABS_TOLERANCE,
            "passed": artifact_reload_pass,
        },
        "model_config": {
            "xgboost_params": lifecycle_manifest["xgboost_params"],
            "fixed_boosting_rounds": lifecycle_manifest["fixed_boosting_rounds"],
            "preprocessing": lifecycle_manifest["preprocessing"],
        },
        "feature_manifests": {
            "development": validation_manifest,
            "final": final_manifest,
        },
        "test_data_accessed": False,
        "reproducibility_status": status,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{MODEL_VERSION}__{timestamp}.json"
    write_json(report_path, report)
    with tempfile.TemporaryDirectory(prefix="returnguard-a3-environment-") as temporary:
        freeze_path = Path(temporary) / "pip-freeze.txt"
        _pip_freeze(freeze_path)
        tracking = log_rebuild_run(
            root=ROOT,
            report=report,
            lifecycle_manifest_path=LIFECYCLE_MANIFEST_PATH,
            freeze_spec_path=FREEZE_SPEC_PATH,
            feature_manifest=final_manifest,
            reference_path=REFERENCE_PATH,
            model_card_path=MODEL_CARD_PATH,
            environment_freeze_path=freeze_path,
            composite=composite,
            register=register,
        )
    report["mlflow"] = tracking
    write_json(report_path, report)
    return report, {"report_path": str(report_path), **tracking}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["full"], default="full")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--bootstrap-reference", action="store_true", help="Create the one-time safe training-derived reference fixture")
    parser.add_argument("--register", action="store_true", help="Register a passing governed rebuild without promoting it")
    args = parser.parse_args()
    report, details = run_full(raw_dir=args.raw_dir, bootstrap_reference=args.bootstrap_reference, register=args.register)
    print(f"ReturnGuard {MODEL_VERSION}: {report['reproducibility_status']}")
    print(f"Validation ROC-AUC={report['metrics']['roc_auc']:.6f}; test data accessed={report['test_data_accessed']}")
    print(f"Report: {details['report_path']}")
    print(f"MLflow run: {details['run_id']}")
    if "registered_model_version" in details:
        print(f"Registered model: returnguard-a3 v{details['registered_model_version']}")
    if report["reproducibility_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
