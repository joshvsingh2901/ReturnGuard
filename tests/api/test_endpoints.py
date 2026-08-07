"""Data-free HTTP behaviour and safe-error tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.config import Settings
from backend.errors import ModelUnavailableError
from backend.main import create_app
from tests.api.conftest import MISSING_CUSTOMER_EVENT, VALID_EVENT, FakeService


def test_health_ready_model_and_prediction_endpoints(app_with_fake_service, fake_service):
    with TestClient(app_with_fake_service) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "code": None}

        assert client.get("/ready").json() == {"status": "ready", "code": None}
        model = client.get("/model")
        assert model.status_code == 200
        assert model.json()["semantic_version"] == "returnguard-a3-v1"
        assert "sqlite" not in model.text.lower()

        prediction = client.post("/predict", json=VALID_EVENT)
        assert prediction.status_code == 200
        body = prediction.json()
        assert 0 <= body["risk_score"] <= 100
        assert body["model"]["registry_model"] == "returnguard-a3"
        assert "raw_model_probability" not in body
        assert "opaque-customer-01" not in prediction.text
        assert prediction.headers["X-Request-ID"] == body["request_id"]
        assert fake_service.calls == 1

        openapi = client.get("/openapi.json")
        assert openapi.status_code == 200
        assert "/predict" in openapi.json()["paths"]
        assert "examples" in openapi.json()["components"]["schemas"]["PredictionRequest"]


def test_batch_is_one_service_call_and_preserves_result_order(app_with_fake_service, fake_service):
    with TestClient(app_with_fake_service) as client:
        response = client.post("/predict/batch", json={"events": [VALID_EVENT, MISSING_CUSTOMER_EVENT]})
        assert response.status_code == 200
        assert [row["risk_score"] for row in response.json()["results"]] == [67, 67]
        assert response.json()["results"][1]["data_context"]["customer_profile_imputed"] is True
        assert fake_service.calls == 1


def test_lifespan_loads_the_service_once_for_multiple_requests():
    starts = []

    def factory(_):
        starts.append("loaded")
        return FakeService()

    app = create_app(
        Settings("sqlite:////unused.db", "models:/returnguard-a3@model-of-record", "WARNING", 100, True),
        service_factory=factory,
    )
    with TestClient(app) as client:
        assert client.post("/predict", json=VALID_EVENT).status_code == 200
        assert client.post("/predict", json=VALID_EVENT).status_code == 200
    assert starts == ["loaded"]


def test_explain_contract_never_exposes_raw_shap_or_encoded_features(app_with_fake_service):
    with TestClient(app_with_fake_service) as client:
        response = client.post("/explain", json=VALID_EVENT)
    assert response.status_code == 200
    body = response.json()
    assert body["top_risk_factors"] == [{"display_name": "Product type: Jeans", "direction": "higher"}]
    assert "contribution" not in response.text
    assert "cat_prod__" not in response.text
    assert "raw_margin" not in response.text


def test_validation_and_sanitized_error_responses(app_with_fake_service):
    with TestClient(app_with_fake_service) as client:
        invalid = dict(MISSING_CUSTOMER_EVENT)
        invalid.pop("customer_context_key")
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422
        assert response.json()["code"] == "invalid_request"

    failing_service = FakeService(fail_prediction=True)
    app = create_app(
        Settings("sqlite:////unused.db", "models:/returnguard-a3@model-of-record", "WARNING", 100, True),
        service_factory=lambda _: failing_service,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json=VALID_EVENT)
    assert response.status_code == 500
    assert response.json()["code"] == "inference_failed"
    assert "secret filesystem path" not in response.text


def test_unavailable_model_leaves_health_live_but_not_ready():
    app = create_app(
        Settings("sqlite:////unused.db", "models:/returnguard-a3@model-of-record", "WARNING", 100, True),
        service_factory=lambda _: (_ for _ in ()).throw(ModelUnavailableError("private failure")),
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        response = client.post("/predict", json=VALID_EVENT)
    assert response.status_code == 503
    assert response.json()["code"] == "model_unavailable"


def test_explanations_can_be_explicitly_disabled():
    app = create_app(
        Settings("sqlite:////unused.db", "models:/returnguard-a3@model-of-record", "WARNING", 100, False),
        service_factory=lambda _: FakeService(explanations_enabled=False),
    )
    with TestClient(app) as client:
        response = client.post("/explain", json=VALID_EVENT)
    assert response.status_code == 503
    assert response.json()["code"] == "explanations_disabled"


def test_process_batch_limit_has_a_stable_422_code():
    app = create_app(
        Settings("sqlite:////unused.db", "models:/returnguard-a3@model-of-record", "WARNING", 1, True),
        service_factory=lambda _: FakeService(),
    )
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"events": [VALID_EVENT, VALID_EVENT]})
    assert response.status_code == 422
    assert response.json()["code"] == "batch_size_exceeded"
