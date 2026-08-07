"""Opt-in smoke test of the ignored local MLflow model-of-record."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from ml.data import loaders
from ml.data.schema import EVENT_CUST_COL
from tests.api.conftest import MISSING_CUSTOMER_EVENT, VALID_EVENT


@pytest.mark.local_registry
def test_local_model_of_record_serves_safely_without_training_or_test_loading(monkeypatch):
    def raw_data_access_forbidden(*args, **kwargs):
        raise AssertionError("Stage 5 serving must not load raw training or official test data")

    monkeypatch.setattr(loaders, "_load_pickle", raw_data_access_forbidden)
    monkeypatch.setattr(loaders, "_load_final_evaluation_test_pickle", raw_data_access_forbidden)

    with TestClient(create_app()) as client:
        assert client.get("/ready").status_code == 200
        model = client.get("/model")
        assert model.status_code == 200
        assert model.json()["semantic_version"] == "returnguard-a3-v1"
        assert model.json()["registry_model"] == "returnguard-a3"
        assert model.json()["registry_version"]
        assert model.json()["feature_manifest_hash"]
        service = client.app.state.model_service
        assert EVENT_CUST_COL not in service._composite_artifact.feature_names

        first = client.post("/predict", json=VALID_EVENT)
        second = client.post("/predict", json=VALID_EVENT)
        assert first.status_code == second.status_code == 200
        assert first.json()["risk_score"] == second.json()["risk_score"]
        assert 0 <= first.json()["risk_score"] <= 100
        changed_complete_key = dict(VALID_EVENT, customer_context_key="different-complete-profile-key")
        changed_complete = client.post("/predict", json=changed_complete_key)
        assert changed_complete.status_code == 200
        assert changed_complete.json()["risk_score"] == first.json()["risk_score"]

        missing = client.post("/predict", json=MISSING_CUSTOMER_EVENT)
        batch = client.post("/predict/batch", json={"events": [VALID_EVENT, MISSING_CUSTOMER_EVENT]})
        reordered = client.post("/predict/batch", json={"events": [MISSING_CUSTOMER_EVENT, VALID_EVENT]})
        assert missing.status_code == batch.status_code == reordered.status_code == 200
        assert missing.json()["risk_score"] == batch.json()["results"][1]["risk_score"]
        assert missing.json()["risk_score"] == reordered.json()["results"][0]["risk_score"]
