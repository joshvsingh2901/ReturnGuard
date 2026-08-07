"""Feature-family grouping and display helpers for Stage 3 SHAP output."""

from __future__ import annotations

import numpy as np
import pandas as pd


FREQUENCY_FEATURE_NAMES = {
    "num_prod__variant_event_count",
    "num_prod__product_event_count",
    "num_prod__supplier_event_count",
}


def feature_family(feature_name: str) -> str:
    """Map one transformed model feature to its explainable family."""
    if feature_name == "num_cust__yearOfBirth":
        return "birth_year_customer_demographic"
    if feature_name in {"bin__isMale", "bin__premier"}:
        return "customer_binary_attributes"
    if feature_name.startswith("cat_cust__shippingCountry_"):
        return "shipping_country"
    if feature_name.startswith("cat_prod__productType___MISSING__") or feature_name.startswith(
        "cat_prod__brandDesc___MISSING__"
    ):
        return "product_profile_missingness"
    if feature_name.startswith("cat_prod__productType_"):
        return "product_type"
    if feature_name.startswith("cat_prod__brandDesc_"):
        return "brand"
    if feature_name in {"num_prod__avgGbpPrice", "num_prod__avgDiscountValue"}:
        return "price_discount"
    if feature_name in {
        "num_prod__discount_amount",
        "num_prod__price_rel_type_median",
        "num_prod__price_rel_brand_median",
    }:
        return "derived_price_features"
    if feature_name in FREQUENCY_FEATURE_NAMES:
        return "product_frequency_features"
    raise ValueError(f"No Stage 3 feature-family mapping for {feature_name!r}")


def feature_display_name(feature_name: str, row: pd.Series | None = None) -> str:
    """Turn an encoded feature or grouped factor into a readable label."""
    family = feature_family(feature_name)
    if family == "birth_year_customer_demographic":
        return "Customer birth year"
    if feature_name == "bin__isMale":
        return "Customer gender indicator"
    if feature_name == "bin__premier":
        return "Premier membership"
    if family == "shipping_country":
        value = row.get("shippingCountry") if row is not None else feature_name.split("shippingCountry_", 1)[1]
        return f"Shipping country: {str(value).replace('_', ' ')}"
    if family == "product_type":
        value = row.get("productType") if row is not None else feature_name.split("productType_", 1)[1]
        return f"Product type: {value}"
    if family == "brand":
        value = row.get("brandDesc") if row is not None else feature_name.split("brandDesc_", 1)[1]
        return f"Brand: {value}"
    if family == "product_profile_missingness":
        return "Product profile unavailable"
    if feature_name == "num_prod__avgGbpPrice":
        return "Product price"
    if feature_name == "num_prod__avgDiscountValue":
        return "Product discount value"
    if feature_name == "num_prod__discount_amount":
        return "Derived discount amount"
    if feature_name == "num_prod__price_rel_type_median":
        return "Price relative to product-type median"
    if feature_name == "num_prod__price_rel_brand_median":
        return "Price relative to brand median"
    if feature_name == "num_prod__variant_event_count":
        return "Variant exposure count (experimental)"
    if feature_name == "num_prod__product_event_count":
        return "Parent-product exposure count (experimental)"
    if feature_name == "num_prod__supplier_event_count":
        return "Supplier exposure count (experimental)"
    raise AssertionError(f"Unexpected display mapping for {feature_name!r}")


def feature_indicator_display_name(feature_name: str) -> str:
    """Name an encoded categorical column itself, rather than the row value.

    A one-hot SHAP contribution can be non-zero when its category is absent.
    Internal factor audits must therefore label the encoded indicator rather
    than imply the row belongs to that category. User-facing grouped factors
    continue to use :func:`feature_display_name` with the observed row.
    """
    family = feature_family(feature_name)
    if family == "shipping_country":
        value = feature_name.split("shippingCountry_", 1)[1]
        return f"Shipping country indicator: {value.replace('_', ' ')}"
    if family == "product_type":
        value = feature_name.split("productType_", 1)[1]
        return f"Product type indicator: {value}"
    if family == "brand":
        value = feature_name.split("brandDesc_", 1)[1]
        return f"Brand indicator: {value}"
    return feature_display_name(feature_name)


def aggregate_grouped_shap(shap_values: np.ndarray, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group signed SHAP values by family, then compute mean absolute value.

    Grouping before absolute value preserves SHAP additivity and avoids
    inflating one-hot categorical fields.
    """
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("SHAP array width must equal feature_names length")

    groups = [feature_family(name) for name in feature_names]
    grouped = pd.DataFrame(index=np.arange(values.shape[0]))
    for group in sorted(set(groups)):
        columns = [i for i, candidate in enumerate(groups) if candidate == group]
        grouped[group] = values[:, columns].sum(axis=1)

    ranking = (
        grouped.abs()
        .mean()
        .rename("mean_abs_shap")
        .sort_values(ascending=False)
        .rename_axis("feature_family")
        .reset_index()
    )
    total = float(ranking["mean_abs_shap"].sum())
    ranking["attribution_share"] = (
        ranking["mean_abs_shap"] / total if total else 0.0
    )
    return grouped, ranking


def encoded_shap_ranking(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Return encoded-feature mean absolute SHAP values in descending order."""
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("SHAP array width must equal feature_names length")
    result = pd.DataFrame(
        {
            "feature": feature_names,
            "display_name": [feature_display_name(name) for name in feature_names],
            "feature_family": [feature_family(name) for name in feature_names],
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }
    )
    return result.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
