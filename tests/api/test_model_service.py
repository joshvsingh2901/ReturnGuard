"""Unit tests for raw-frame construction without a registry or training data."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from backend.model_service import ModelService
from backend.schemas import PredictionRequest
from ml.data.schema import EVENT_CUST_COL
from tests.api.conftest import MISSING_CUSTOMER_EVENT, VALID_EVENT


class DeterministicPyfunc:
    def __init__(self):
        self.calls = 0
        self.frames = []

    def predict(self, frame):
        self.calls += 1
        self.frames.append(frame.copy())
        values = []
        for _, row in frame.iterrows():
            if np.isnan(row["isMale"]):
                stable = int.from_bytes(
                    hashlib.blake2b(str(row[EVENT_CUST_COL]).encode(), digest_size=2).digest(), "little"
                )
                values.append(0.2 + (stable % 40) / 100)
            else:
                values.append(0.67)
        return np.asarray(values)


def frame_service() -> ModelService:
    service = object.__new__(ModelService)
    service._category_values = {
        "shippingCountry": {"Country_A", "Country_B"},
        "productType": {"Jeans", "productType_A", "__MISSING__"},
        "brandDesc": {"Brand_A", "Brand_B", "__MISSING__"},
    }
    service._pyfunc_model = DeterministicPyfunc()
    return service


def test_only_model_of_record_alias_or_exact_resolved_version_is_allowed():
    ModelService._validate_model_uri("models:/returnguard-a3@model-of-record", "7")
    ModelService._validate_model_uri("models:/returnguard-a3/7", "7")

    with pytest.raises(ValueError, match="model-of-record"):
        ModelService._validate_model_uri("models:/returnguard-a3/6", "7")
    with pytest.raises(ValueError, match="model-of-record"):
        ModelService._validate_model_uri("models:/returnguard-a4@model-of-record", "7")


def test_raw_request_builds_only_the_artifact_inputs_and_preserves_null_policy():
    service = frame_service()
    request = PredictionRequest.model_validate(MISSING_CUSTOMER_EVENT)
    frame, contexts = service._frame_for_requests([request])

    assert list(frame.columns) == [
        EVENT_CUST_COL,
        "yearOfBirth",
        "isMale",
        "shippingCountry",
        "premier",
        "productType",
        "brandDesc",
        "avgGbpPrice",
        "avgDiscountValue",
    ]
    assert frame.loc[0, EVENT_CUST_COL] == "stable-missing-customer"
    assert np.isnan(frame.loc[0, "isMale"])
    assert contexts[0].customer_profile_imputed is True
    assert EVENT_CUST_COL not in {"yearOfBirth", "isMale", "shippingCountry", "premier"}


def test_complete_customer_prediction_is_independent_of_context_key():
    service = frame_service()
    first = PredictionRequest.model_validate(VALID_EVENT)
    second_payload = dict(VALID_EVENT, customer_context_key="different-opaque-token")
    second = PredictionRequest.model_validate(second_payload)

    results = service.predict([first, second])
    assert [result["risk_score"] for result in results] == [67, 67]
    assert service._pyfunc_model.calls == 1


def test_missing_customer_key_is_stable_across_repeat_batch_and_order():
    service = frame_service()
    missing = PredictionRequest.model_validate(MISSING_CUSTOMER_EVENT)
    known = PredictionRequest.model_validate(VALID_EVENT)

    repeated = service.predict([missing])[0]["risk_score"]
    batched = service.predict([known, missing])[1]["risk_score"]
    reordered = service.predict([missing, known])[0]["risk_score"]
    assert repeated == batched == reordered


def test_product_context_warnings_and_unknown_categories_are_public_field_names_only():
    service = frame_service()
    payload = dict(VALID_EVENT)
    payload["product_profile"] = {
        "productType": "merchant_new_type",
        "brandDesc": None,
        "avgGbpPrice": 900,
        "avgDiscountValue": None,
    }
    result = service.predict([PredictionRequest.model_validate(payload)])[0]

    assert result["data_context"].missing_product_fields == ["brandDesc", "avgDiscountValue"]
    assert result["data_context"].unknown_categories == ["productType"]
    assert result["data_context"].outside_training_range_fields == ["avgGbpPrice"]
    assert any("weaker evaluation performance" in warning for warning in result["warnings"])
