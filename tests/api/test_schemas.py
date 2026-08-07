"""Raw-contract validation tests for the frozen serving boundary."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from backend.schemas import BatchPredictionRequest, PredictionRequest
from tests.api.conftest import MISSING_CUSTOMER_EVENT, VALID_EVENT


def test_valid_complete_request_accepts_raw_inputs_only():
    request = PredictionRequest.model_validate(VALID_EVENT)

    assert request.customer_profile.shipping_country == "Country_A"
    assert request.product_profile.avg_gbp_price == 54.99


def test_missing_customer_requires_opaque_context_key():
    request = PredictionRequest.model_validate(MISSING_CUSTOMER_EVENT)
    assert request.customer_profile is None

    invalid = dict(MISSING_CUSTOMER_EVENT)
    invalid.pop("customer_context_key")
    with pytest.raises(ValidationError, match="customer_context_key"):
        PredictionRequest.model_validate(invalid)


def test_nullable_year_and_partial_customer_rejection():
    payload = dict(VALID_EVENT)
    payload["customer_profile"] = dict(VALID_EVENT["customer_profile"], yearOfBirth=None)
    assert PredictionRequest.model_validate(payload).customer_profile.year_of_birth is None

    payload["customer_profile"] = {"yearOfBirth": 1988, "isMale": True, "premier": False}
    with pytest.raises(ValidationError, match="shippingCountry"):
        PredictionRequest.model_validate(payload)


@pytest.mark.parametrize(
    "product_profile",
    [
        VALID_EVENT["product_profile"],
        {"productType": "Jeans", "brandDesc": None, "avgGbpPrice": None, "avgDiscountValue": 5.0},
        None,
    ],
)
def test_complete_partial_and_null_product_profiles_are_supported(product_profile):
    payload = dict(VALID_EVENT, product_profile=product_profile)
    assert PredictionRequest.model_validate(payload).product_profile == (
        None if product_profile is None else PredictionRequest.model_validate(payload).product_profile
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("avgGbpPrice", 0),
        ("avgGbpPrice", -1),
        ("avgDiscountValue", -0.1),
        ("avgDiscountValue", 100.1),
        ("avgGbpPrice", math.nan),
        ("avgDiscountValue", math.inf),
    ],
)
def test_invalid_numeric_product_values_are_rejected(field, value):
    payload = dict(VALID_EVENT)
    payload["product_profile"] = dict(VALID_EVENT["product_profile"], **{field: value})
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


@pytest.mark.parametrize("year", [1900, 1700, 2200])
def test_invalid_or_sentinel_birth_year_is_rejected(year):
    payload = dict(VALID_EVENT)
    payload["customer_profile"] = dict(VALID_EVENT["customer_profile"], yearOfBirth=year)
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


def test_reserved_missing_token_and_extra_features_are_rejected():
    payload = dict(VALID_EVENT)
    payload["product_profile"] = dict(VALID_EVENT["product_profile"], productType="__MISSING__")
    with pytest.raises(ValidationError, match="reserved"):
        PredictionRequest.model_validate(payload)

    payload = dict(VALID_EVENT, encoded_feature_1=1)
    with pytest.raises(ValidationError, match="Extra inputs"):
        PredictionRequest.model_validate(payload)


def test_unknown_categories_and_outside_support_remain_valid_requests():
    payload = dict(VALID_EVENT)
    payload["customer_profile"] = dict(VALID_EVENT["customer_profile"], yearOfBirth=1888)
    payload["product_profile"] = dict(
        VALID_EVENT["product_profile"], productType="merchant_new_type", avgGbpPrice=900, avgDiscountValue=70
    )
    request = PredictionRequest.model_validate(payload)
    assert request.product_profile.product_type == "merchant_new_type"


def test_batch_contract_enforces_nonempty_hard_limit():
    with pytest.raises(ValidationError):
        BatchPredictionRequest.model_validate({"events": []})
    with pytest.raises(ValidationError):
        BatchPredictionRequest.model_validate({"events": [VALID_EVENT] * 101})
