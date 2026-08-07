"""Local MLflow tracking and registry support for the frozen A3 lifecycle."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import pandas as pd

from ml.lifecycle.reproducibility import MODEL_VERSION, A3CompositeArtifact

EXPERIMENT_NAME = "returnguard-frozen-lifecycle"
REGISTERED_MODEL_NAME = "returnguard-a3"


class A3PyfuncModel:
    """A small adapter defined lazily as an MLflow PythonModel at log time."""

    @staticmethod
    def build():
        import mlflow.pyfunc

        class _A3PyfuncModel(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                self._artifact = joblib.load(context.artifacts["composite"])

            def predict(self, context, model_input, params=None):
                frame = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
                return self._artifact.predict_proba(frame)[:, 1]

        return _A3PyfuncModel()


def local_mlflow_paths(root: Path) -> tuple[Path, Path]:
    base = root / ".mlflow"
    return base / "mlflow.db", base / "artifacts"


def configure_mlflow(root: Path):
    """Configure SQLite metadata and local-file artifacts; return mlflow module."""
    import mlflow

    database, artifact_root = local_mlflow_paths(root)
    database.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{database}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_root.as_uri())
    else:
        experiment_id = experiment.experiment_id
    return mlflow, client, experiment_id


def lifecycle_tags(*, run_type: str, git_commit: str, dataset_version: str, reproducibility_status: str, test_access: str) -> dict[str, str]:
    return {
        "project": "returnguard",
        "run_type": run_type,
        "semantic_model_version": MODEL_VERSION,
        "governance_status": "frozen",
        "model_development_closed": "true",
        "git_commit": git_commit,
        "dataset_version": dataset_version,
        "reproducibility_status": reproducibility_status,
        "test_access": test_access,
    }


def _flatten_numeric(prefix: str, value: object) -> dict[str, float]:
    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            flattened.update(_flatten_numeric(f"{prefix}.{key}" if prefix else key, child))
        return flattened
    if isinstance(value, bool):
        return {prefix: float(value)}
    if isinstance(value, (int, float)) and value is not None:
        return {prefix: float(value)}
    return {}


def log_rebuild_run(
    *,
    root: Path,
    report: dict,
    lifecycle_manifest_path: Path,
    freeze_spec_path: Path,
    feature_manifest: dict,
    reference_path: Path,
    model_card_path: Path,
    environment_freeze_path: Path,
    composite: A3CompositeArtifact,
    register: bool,
) -> dict:
    """Log a governed rebuild; register only after all local checks pass."""
    if report["reproducibility_status"] != "PASS":
        raise ValueError("Refusing to log a non-passing A3 rebuild as a governed artifact")
    mlflow, _, experiment_id = configure_mlflow(root)
    run_name = f"{MODEL_VERSION}__frozen_model_rebuild__{report['git_commit'][:8]}"
    tags = lifecycle_tags(
        run_type="frozen_model_rebuild",
        git_commit=report["git_commit"],
        dataset_version=report["dataset_version"],
        reproducibility_status=report["reproducibility_status"],
        test_access="none",
    )
    tags["repository_dirty"] = str(report["repository_dirty"]).lower()
    tags["mlflow.runName"] = run_name

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags=tags) as active_run:
        params = {
            "model_family": "XGBoost binary logistic",
            "fixed_boosting_rounds": 659,
            "feature_count": 41,
            "feature_manifest_hash": report["feature_manifest_hash"],
            "freeze_spec_hash": report["freeze_spec_hash"],
            "model_config_hash": report["model_config_hash"],
            "artifact_safe_donor_imputation": True,
            "derived_price_features": True,
            "product_frequency_features": False,
            "calibrator": False,
        }
        for key, value in report["model_config"]["xgboost_params"].items():
            params[f"xgboost.{key}"] = value
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(_flatten_numeric("validation", report["metrics"]))
        mlflow.log_metrics(_flatten_numeric("delta", report["metric_deltas_vs_reference"]))
        mlflow.log_metrics(_flatten_numeric("reference", report["reference_prediction_deltas"]))
        mlflow.log_dict(report, "reproducibility_report.json")
        mlflow.log_dict(feature_manifest, "feature_manifest.json")
        mlflow.log_artifact(str(lifecycle_manifest_path), "governance")
        mlflow.log_artifact(str(freeze_spec_path), "governance")
        mlflow.log_artifact(str(reference_path), "governance")
        mlflow.log_artifact(str(model_card_path), "documentation")
        mlflow.log_artifact(str(environment_freeze_path), "environment")

        with tempfile.TemporaryDirectory(prefix="returnguard-a3-") as temporary:
            composite_path = Path(temporary) / "returnguard-a3-composite.joblib"
            joblib.dump(composite, composite_path)
            # The only persisted copy is embedded in MLflow's pyfunc model.
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=A3PyfuncModel.build(),
                artifacts={"composite": str(composite_path)},
                metadata={
                    "semantic_model_version": MODEL_VERSION,
                    "feature_manifest_hash": report["feature_manifest_hash"],
                    "freeze_spec_hash": report["freeze_spec_hash"],
                },
            )
        run_id = active_run.info.run_id

    result = {"run_id": run_id, "model_uri": f"runs:/{run_id}/model"}
    if register:
        registered = mlflow.register_model(result["model_uri"], REGISTERED_MODEL_NAME)
        result["registered_model_name"] = REGISTERED_MODEL_NAME
        result["registered_model_version"] = str(registered.version)
    return result


def log_historical_final_evaluation(*, root: Path, metrics: dict, slices: dict, calibration: dict) -> str:
    """Import committed evidence without loading or scoring official test rows."""
    mlflow, _, experiment_id = configure_mlflow(root)
    execution = metrics["execution"]
    tags = lifecycle_tags(
        run_type="historical_final_evaluation",
        git_commit=execution["git_revision"],
        dataset_version="asos-graphreturns-osf-c793h-v1",
        reproducibility_status="historical_evidence",
        test_access="historical_report_only",
    )
    with mlflow.start_run(experiment_id=experiment_id, run_name="returnguard-a3-v1__historical_final_evaluation", tags=tags) as active_run:
        mlflow.log_params({
            "evaluation_source": "committed_json_reports_only",
            "test_probability_generation_count": str(execution["test_probability_generation_count"]),
            "fixed_boosting_rounds": "659",
            "calibrator_fitted": "false",
            "feature_manifest_hash": execution["feature_manifest_hash"],
            "freeze_spec_hash": execution["freeze_spec_hash"],
        })
        mlflow.log_metrics(_flatten_numeric("official_test", metrics["official_test_metrics"]))
        mlflow.log_metrics(_flatten_numeric("slice", slices["slices"]))
        mlflow.log_metrics(_flatten_numeric("calibration", calibration["calibration"]))
        mlflow.log_dict(metrics, "historical/final_test_metrics.json")
        mlflow.log_dict(slices, "historical/final_test_slices.json")
        mlflow.log_dict(calibration, "historical/final_test_calibration.json")
        return active_run.info.run_id


def promote_registered_model(*, root: Path, run_id: str, model_version: str, allow_dirty: bool) -> None:
    """Apply the alias only after a passing governed run has been reviewed."""
    _, client, _ = configure_mlflow(root)
    run = client.get_run(run_id)
    tags = run.data.tags
    required = {
        "project": "returnguard",
        "run_type": "frozen_model_rebuild",
        "semantic_model_version": MODEL_VERSION,
        "governance_status": "frozen",
        "reproducibility_status": "PASS",
        "test_access": "none",
    }
    for key, expected in required.items():
        if tags.get(key) != expected:
            raise ValueError(f"Run {run_id} does not satisfy promotion policy: {key}")
    if tags.get("repository_dirty") == "true" and not allow_dirty:
        raise ValueError("Refusing dirty-worktree model promotion without --allow-recorded-dirty-git")
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "model-of-record", model_version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version, "semantic_model_version", MODEL_VERSION)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version, "governance_status", "model_of_record")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version, "source_run_id", run_id)
