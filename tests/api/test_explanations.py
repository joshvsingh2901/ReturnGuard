"""Public explanation contract tests without SHAP runtime or registry access."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from backend.model_service import ModelService, ServingProvenance
from backend.schemas import PredictionRequest
from tests.api.conftest import MISSING_CUSTOMER_EVENT, VALID_EVENT


class TinyComposite:
    feature_names = [
        "num_cust__yearOfBirth",
        "cat_prod__productType_Jeans",
        "num_prod__avgGbpPrice",
    ]
    model = object()

    @staticmethod
    def transform(frame):
        return np.ones((len(frame), 3), dtype=np.float32)


def explanation_service() -> ModelService:
    service = object.__new__(ModelService)
    service._composite_artifact = TinyComposite()
    service._category_values = {
        "shippingCountry": {"Country_A"},
        "productType": {"Jeans", "__MISSING__"},
        "brandDesc": {"Brand_A", "__MISSING__"},
    }
    service._explanations_enabled = True
    service._explainer = object()
    service.provenance = ServingProvenance(
        "returnguard-a3-v1", "returnguard-a3", "1", "run", "feature-hash", "dataset", "model_of_record"
    )
    return service


def test_public_explanations_hide_internal_mechanics_and_donor_demographics(monkeypatch):
    service = explanation_service()
    monkeypatch.setattr(
        "backend.model_service.compute_tree_shap",
        lambda *args, **kwargs: SimpleNamespace(
            probability=np.asarray([0.67]),
            raw_margin=np.asarray([0.7]),
            base_value=0.1,
            values=np.asarray([[0.3, 0.2, -0.1]]),
        ),
    )

    result = service.explain(PredictionRequest.model_validate(MISSING_CUSTOMER_EVENT))
    rendered = str(result)
    assert "raw_margin" not in rendered
    assert "contribution" not in rendered
    assert "num_cust__" not in rendered
    assert all("Customer" not in factor["display_name"] for factor in result["top_risk_factors"])
    assert result["top_risk_factors"] == [{"display_name": "Product type: Jeans", "direction": "higher"}]


def test_public_explanation_factor_direction_is_grouped_and_safe(monkeypatch):
    service = explanation_service()
    monkeypatch.setattr(
        "backend.model_service.compute_tree_shap",
        lambda *args, **kwargs: SimpleNamespace(
            probability=np.asarray([0.67]),
            raw_margin=np.asarray([0.7]),
            base_value=0.1,
            values=np.asarray([[0.3, 0.2, -0.1]]),
        ),
    )
    result = service.explain(PredictionRequest.model_validate(VALID_EVENT))
    assert result["top_protective_factors"] == [{"display_name": "Price and discount", "direction": "lower"}]
