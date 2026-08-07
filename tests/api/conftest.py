"""Shared fake service and payloads for API tests; no MLflow or ASOS data."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.errors import InferenceFailedError
from backend.model_service import ServingProvenance
from backend.schemas import DataContext

VALID_EVENT = {
    "customer_context_key": "opaque-customer-01",
    "customer_profile": {
        "yearOfBirth": 1988,
        "isMale": False,
        "shippingCountry": "Country_A",
        "premier": True,
    },
    "product_profile": {
        "productType": "Jeans",
        "brandDesc": "Brand_A",
        "avgGbpPrice": 54.99,
        "avgDiscountValue": 15.0,
    },
}

MISSING_CUSTOMER_EVENT = {
    "customer_context_key": "stable-missing-customer",
    "customer_profile": None,
    "product_profile": VALID_EVENT["product_profile"],
}


@dataclass
class FakeService:
    """A deliberate serving seam that returns only public-safe payloads."""

    fail_prediction: bool = False
    explanations_enabled: bool = True
    calls: int = 0

    def __post_init__(self) -> None:
        self.provenance = ServingProvenance(
            semantic_version="returnguard-a3-v1",
            registry_model="returnguard-a3",
            registry_version="1",
            mlflow_source_run_id="run-123",
            feature_manifest_hash="feature-hash",
            dataset_version="asos-graphreturns-osf-c793h-v1",
            governance_status="model_of_record",
        )
        self.methodology_note = (
            "This score is estimated from a returner-enriched ASOS research sample. "
            "It supports relative risk ranking but is not a merchant-wide probability of return. "
            "Model factors describe model behavior and are not causal."
        )

    def concise_provenance(self):
        return self.provenance.concise()

    def model_info(self):
        return self.provenance.public_model_info()

    def predict(self, requests):
        self.calls += 1
        if self.fail_prediction:
            raise InferenceFailedError("secret filesystem path should never be returned")
        results = []
        for event in requests:
            product = event.product_profile
            missing_product_fields = [] if product is not None else [
                "productType",
                "brandDesc",
                "avgGbpPrice",
                "avgDiscountValue",
            ]
            context = DataContext(
                customer_profile_imputed=event.customer_profile is None,
                customer_birth_year_imputed=(
                    event.customer_profile is not None and event.customer_profile.year_of_birth is None
                ),
                product_profile_available=product is not None,
                product_profile_complete=product is not None and not missing_product_fields,
                missing_product_fields=missing_product_fields,
                unknown_categories=[],
                outside_training_range_fields=[],
            )
            results.append(
                {
                    "risk_score": 67,
                    "risk_score_scale": "0-100",
                    "score_type": "dataset_conditional_return_risk",
                    "data_context": context,
                    "warnings": [],
                }
            )
        return results

    def explain(self, request):
        if not self.explanations_enabled:
            from backend.errors import ExplanationsDisabledError

            raise ExplanationsDisabledError("disabled")
        payload = self.predict([request])[0]
        payload.update(
            {
                "top_risk_factors": [{"display_name": "Product type: Jeans", "direction": "higher"}],
                "top_protective_factors": [{"display_name": "Price and discount", "direction": "lower"}],
                "explanation_method": "tree_shap_grouped_noncausal",
            }
        )
        return payload


@pytest.fixture
def fake_service() -> FakeService:
    return FakeService()


@pytest.fixture
def app_with_fake_service(fake_service: FakeService):
    from backend.config import Settings
    from backend.main import create_app

    settings = Settings(
        mlflow_tracking_uri="sqlite:////not-a-real-registry.db",
        model_uri="models:/returnguard-a3@model-of-record",
        log_level="WARNING",
        max_batch_size=100,
        explanations_enabled=True,
    )
    return create_app(settings, service_factory=lambda _: fake_service)
