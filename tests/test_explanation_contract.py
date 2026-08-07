"""Tests for the non-causal Stage 3 local explanation contract."""

import numpy as np
import pandas as pd

from ml.explainability.contracts import build_local_explanation, user_facing_explanation


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
