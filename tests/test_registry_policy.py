"""Registry promotion policy is explicit and independent of model metrics."""

from __future__ import annotations

import pytest

from ml.lifecycle.tracking import configure_mlflow, promote_registered_model


def test_promotion_rejects_non_governed_run(tmp_path):
    mlflow, client, experiment_id = configure_mlflow(tmp_path)
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
    with pytest.raises(ValueError, match="promotion policy"):
        promote_registered_model(root=tmp_path, run_id=run_id, model_version="1", allow_dirty=False)
    assert client.get_run(run_id)
