"""API-friendly, non-causal local explanation contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.explainability.grouping import feature_display_name, feature_family


METHODOLOGY_NOTE = (
    "This score is estimated from a returner-enriched ASOS research sample. "
    "It supports relative risk ranking but is not a merchant-wide probability of return. "
    "Model factors describe model behavior and are not causal."
)

_IMPUTED_CUSTOMER_FAMILIES = {
    "birth_year_customer_demographic",
    "customer_binary_attributes",
    "shipping_country",
}


def _factor_governance(feature_name: str) -> str:
    family = feature_family(feature_name)
    if family == "product_frequency_features":
        return "experimental_frequency_feature"
    if family in {"price_discount", "derived_price_features"}:
        return "documented_suspicious_feature"
    if family == "product_profile_missingness":
        return "approved_target_neutral_missingness"
    return "approved_feature"


def _factor_from_feature(feature_name: str, contribution: float, row: pd.Series) -> dict:
    return {
        "feature": feature_name,
        "display_name": feature_display_name(feature_name, row),
        "feature_family": feature_family(feature_name),
        "direction": "higher" if contribution > 0 else "lower",
        "contribution": float(contribution),
        "contribution_unit": "log_odds",
        "governance_status": _factor_governance(feature_name),
    }


def _grouped_user_factors(factors: list[dict], row: pd.Series) -> list[dict]:
    """Aggregate one-hot factors before presenting them to a human reader."""
    grouped = {}
    for factor in factors:
        family = factor["feature_family"]
        # Frequency features remain internal by default. Their experimental
        # status is explained in governance reports rather than customer text.
        if family == "product_frequency_features":
            continue
        if family not in grouped:
            grouped[family] = {
                "feature": f"group::{family}",
                "display_name": factor["display_name"],
                "feature_family": family,
                "contribution": 0.0,
                "contribution_unit": "log_odds",
                "governance_status": factor["governance_status"],
            }
        grouped[family]["contribution"] += factor["contribution"]

    outputs = []
    for factor in grouped.values():
        if factor["feature_family"] == "customer_binary_attributes":
            factor["display_name"] = "Customer account attributes"
        elif factor["feature_family"] == "price_discount":
            factor["display_name"] = "Price and discount"
        elif factor["feature_family"] == "derived_price_features":
            factor["display_name"] = "Derived price context"
        factor["direction"] = "higher" if factor["contribution"] > 0 else "lower"
        outputs.append(factor)
    return outputs


def build_local_explanation(
    *,
    raw_probability: float,
    raw_margin: float,
    base_value: float,
    shap_row: np.ndarray,
    feature_names: list[str],
    row: pd.Series,
    customer_profile_imputed: bool,
    product_profile_available: bool,
    model_version: str,
    top_k: int = 3,
) -> dict:
    """Build one internal explanation without presenting imputed facts as real.

    The internal factor list retains transformed features for auditability.
    User-facing factors suppress donor-sampled customer values whenever the
    original customer node was absent.
    """
    shap_row = np.asarray(shap_row, dtype=float)
    if shap_row.ndim != 1 or shap_row.size != len(feature_names):
        raise ValueError("Local SHAP row must align exactly to feature names")

    factors = [_factor_from_feature(name, value, row) for name, value in zip(feature_names, shap_row)]
    upward = sorted((factor for factor in factors if factor["contribution"] > 0), key=lambda f: f["contribution"], reverse=True)
    downward = sorted((factor for factor in factors if factor["contribution"] < 0), key=lambda f: f["contribution"])

    def user_visible(factor: dict) -> bool:
        return not (
            customer_profile_imputed
            and factor["feature_family"] in _IMPUTED_CUSTOMER_FAMILIES
        )

    grouped_user_factors = _grouped_user_factors(factors, row)
    visible_upward = sorted(
        (factor for factor in grouped_user_factors if factor["contribution"] > 0 and user_visible(factor)),
        key=lambda factor: factor["contribution"],
        reverse=True,
    )[:top_k]
    visible_downward = sorted(
        (factor for factor in grouped_user_factors if factor["contribution"] < 0 and user_visible(factor)),
        key=lambda factor: factor["contribution"],
    )[:top_k]

    return {
        "risk_score": int(np.clip(round(float(raw_probability) * 100), 0, 100)),
        "risk_score_scale": "0-100",
        "score_type": "dataset_conditional_return_risk",
        "raw_model_probability": float(raw_probability),
        "raw_model_margin": float(raw_margin),
        "shap_base_value": float(base_value),
        "shap_additivity_residual": float(
            raw_margin - (base_value + float(shap_row.sum()))
        ),
        "top_risk_factors": visible_upward,
        "top_protective_factors": visible_downward,
        "internal_ranked_factors": sorted(factors, key=lambda f: abs(f["contribution"]), reverse=True),
        "data_context": {
            "customer_profile_imputed": bool(customer_profile_imputed),
            "product_profile_available": bool(product_profile_available),
        },
        "model_version": model_version,
        "explanation_method": "tree_shap_raw_margin",
        "methodology_note": METHODOLOGY_NOTE,
    }


def user_facing_explanation(internal_explanation: dict) -> dict:
    """Remove raw SHAP mechanics and encoded identifiers from an explanation."""
    def public_factor(factor: dict) -> dict:
        return {
            "display_name": factor["display_name"],
            "direction": factor["direction"],
        }

    return {
        "risk_score": internal_explanation["risk_score"],
        "risk_score_scale": internal_explanation["risk_score_scale"],
        "score_type": internal_explanation["score_type"],
        "top_risk_factors": [
            public_factor(factor) for factor in internal_explanation["top_risk_factors"]
        ],
        "top_protective_factors": [
            public_factor(factor)
            for factor in internal_explanation["top_protective_factors"]
        ],
        "data_context": internal_explanation["data_context"],
        "model_version": internal_explanation["model_version"],
        "methodology_note": internal_explanation["methodology_note"],
    }
