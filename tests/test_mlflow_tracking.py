"""Synthetic MLflow artifact logging and reload coverage."""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from ml.data.schema import TARGET_COL
from ml.features.preprocessing import get_feature_names
from ml.lifecycle.reproducibility import A3CompositeArtifact
from ml.lifecycle.tracking import configure_mlflow, log_rebuild_run
from ml.models.gbdt import build_feature_pipeline


def _synthetic_composite(synthetic_df_with_missing):
    train = synthetic_df_with_missing.copy()
    pipeline = build_feature_pipeline(artifact_safe=True, include_derived=True, include_frequency=False)
    matrix = pipeline.fit_transform(train.drop(columns=[TARGET_COL])).astype(np.float32)
    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cpu",
        n_estimators=3,
        max_depth=2,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1,
        eval_metric="logloss",
    )
    model.fit(matrix, train[TARGET_COL])
    return A3CompositeArtifact(pipeline, model, get_feature_names(pipeline.named_steps["preprocess"]))


def test_local_mlflow_logs_and_reloads_composite_model(tmp_path, synthetic_df_with_missing):
    lifecycle = tmp_path / "lifecycle.json"
    freeze = tmp_path / "freeze.json"
    reference = tmp_path / "reference.json"
    card = tmp_path / "model-card.md"
    environment = tmp_path / "pip-freeze.txt"
    lifecycle.write_text("{}")
    freeze.write_text("{}")
    reference.write_text("{}")
    card.write_text("model card")
    environment.write_text("package==1")
    report = {
        "reproducibility_status": "PASS",
        "git_commit": "abc12345",
        "repository_dirty": False,
        "dataset_version": "asos-graphreturns-osf-c793h-v1",
        "feature_manifest_hash": "feature-hash",
        "freeze_spec_hash": "freeze-hash",
        "model_config_hash": "config-hash",
        "model_config": {"xgboost_params": {"max_depth": 2}},
        "metrics": {"roc_auc": 0.6, "log_loss": 0.6, "brier": 0.2, "ece": 0.01},
        "metric_deltas_vs_reference": {"roc_auc": 0.0},
        "reference_prediction_deltas": {"mean_abs_delta": 0.0, "max_abs_delta": 0.0},
    }
    composite = _synthetic_composite(synthetic_df_with_missing)
    result = log_rebuild_run(
        root=tmp_path,
        report=report,
        lifecycle_manifest_path=lifecycle,
        freeze_spec_path=freeze,
        feature_manifest={"feature_count": len(composite.feature_names)},
        reference_path=reference,
        model_card_path=card,
        environment_freeze_path=environment,
        composite=composite,
        register=False,
    )

    mlflow, client, _ = configure_mlflow(tmp_path)
    loaded = mlflow.pyfunc.load_model(result["model_uri"])
    prediction = loaded.predict(synthetic_df_with_missing.drop(columns=[TARGET_COL]).head(3))
    assert len(prediction) == 3
    assert client.get_run(result["run_id"]).data.tags["test_access"] == "none"
