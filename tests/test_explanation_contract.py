"""Tests for the non-causal Stage 3 local explanation contract."""

import numpy as np
import pandas as pd

from ml.explainability.contracts import build_local_explanation, user_facing_explanation
from ml.explainability.grouping import feature_display_name, feature_indicator_display_name


def _row():
    return pd.Series(
        {
            "productType": "Jeans",
            "brandDesc": "Pull&Bear",
            "shippingCountry": "Country_A",
        }
    )


def test_local_explanation_has_valid_risk_and_protective_factors():
    names = ["num_prod__avgGbpPrice", "cat_prod__productType_Jeans"]
    explanation = build_local_explanation(
        raw_probability=0.67,
        raw_margin=0.7,
        base_value=0.2,
        shap_row=np.array([0.8, -0.3]),
        feature_names=names,
        row=_row(),
        customer_profile_imputed=False,
        product_profile_available=True,
        model_version="test",
    )
    assert explanation["risk_score"] == 67
    assert explanation["top_risk_factors"]
    assert explanation["top_protective_factors"]
    assert explanation["explanation_method"] == "tree_shap_raw_margin"


def test_donor_customer_demographics_are_suppressed_user_facing():
    names = ["num_cust__yearOfBirth", "bin__isMale", "num_prod__avgGbpPrice"]
    explanation = build_local_explanation(
        raw_probability=0.6,
        raw_margin=0.4,
        base_value=0.0,
        shap_row=np.array([1.0, 0.8, 0.2]),
        feature_names=names,
        row=_row(),
        customer_profile_imputed=True,
        product_profile_available=True,
        model_version="test",
    )
    public = user_facing_explanation(explanation)
    text = str(public)
    assert "birth" not in text.lower()
    assert "gender" not in text.lower()
    assert "raw_model_probability" not in public
    assert "contribution" not in text


def test_categorical_indicator_and_group_context_labels_do_not_conflict():
    """Internal one-hot factors name their own category; public groups name the row."""
    assert feature_display_name("cat_cust__shippingCountry_Country_E", _row()) == (
        "Shipping country: Country A"
    )
    assert feature_indicator_display_name("cat_cust__shippingCountry_Country_E") == (
        "Shipping country indicator: Country E"
    )
    explanation = build_local_explanation(
        raw_probability=0.6,
        raw_margin=0.4,
        base_value=0.0,
        shap_row=np.array([0.4]),
        feature_names=["cat_cust__shippingCountry_Country_E"],
        row=_row(),
        customer_profile_imputed=False,
        product_profile_available=True,
        model_version="test",
    )
    assert explanation["internal_ranked_factors"][0]["display_name"] == (
        "Shipping country indicator: Country E"
    )
    assert explanation["top_risk_factors"][0]["display_name"] == "Shipping country: Country A"
