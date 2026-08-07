"""Explicit Stage 3 feature manifests and fail-closed validation."""

from __future__ import annotations

import hashlib
import json

from ml.data.schema import (
    EXCLUDED_DIRECT_ID_COLUMNS,
    FREQUENCY_FEATURES,
    MODEL_EXCLUDED_COLUMNS,
    TARGET_COL,
)
from ml.explainability.grouping import feature_family


FREQUENCY_TRANSFORMED_NAMES = {
    "num_prod__variant_event_count",
    "num_prod__product_event_count",
    "num_prod__supplier_event_count",
}


def _provenance(feature_name: str) -> tuple[str, str]:
    family = feature_family(feature_name)
    if family == "product_frequency_features":
        return "SUSPICIOUS_TARGET_FREE_EXPOSURE", "experimental_a4_only"
    if family in {"price_discount", "derived_price_features"}:
        return "SUSPICIOUS_GLOBAL_PRICE_DISCOUNT", "documented_suspicious"
    if family == "product_profile_missingness":
        return "TARGET_NEUTRAL_PRODUCT_MISSINGNESS", "approved_with_monitoring"
    return "SAFE_PURCHASE_TIME_ATTRIBUTE", "approved"


def build_feature_manifest(feature_names: list[str], model_name: str) -> dict:
    """Build a serialisable manifest in exact transformed-column order."""
    if model_name not in {"A3", "A4"}:
        raise ValueError("model_name must be 'A3' or 'A4'")

    entries = []
    for order, feature_name in enumerate(feature_names):
        if TARGET_COL in feature_name or "hash(" in feature_name:
            raise ValueError(f"Forbidden target or raw identifier feature: {feature_name}")
        provenance, governance = _provenance(feature_name)
        entries.append(
            {
                "order": order,
                "transformed_feature": feature_name,
                "original_feature_family": feature_family(feature_name),
                "provenance_classification": provenance,
                "governance_classification": governance,
                "dtype": "float32",
            }
        )

    transformed_names = {entry["transformed_feature"] for entry in entries}
    present_frequency = transformed_names & FREQUENCY_TRANSFORMED_NAMES
    if model_name == "A3" and present_frequency:
        raise ValueError("A3 manifest must not contain frequency features")
    if model_name == "A4" and present_frequency != FREQUENCY_TRANSFORMED_NAMES:
        raise ValueError("A4 manifest must contain exactly the approved frequency features")

    # These values are raw source names, but the explicit check protects
    # against a future preprocessor accidentally forwarding a source column.
    forbidden_raw = set(MODEL_EXCLUDED_COLUMNS) | set(EXCLUDED_DIRECT_ID_COLUMNS) | {TARGET_COL}
    leaked = [name for name in transformed_names if name in forbidden_raw]
    if leaked:
        raise ValueError(f"Forbidden raw source columns reached model: {leaked}")

    canonical_entries = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "model_name": model_name,
        "feature_count": len(entries),
        "entries": entries,
        "manifest_hash": hashlib.sha256(canonical_entries).hexdigest(),
        "approved_frequency_features": list(FREQUENCY_FEATURES if model_name == "A4" else []),
    }


def validate_a3_a4_manifests(a3_manifest: dict, a4_manifest: dict) -> None:
    """Ensure A4 differs from A3 only by the three approved encodings."""
    a3_names = [entry["transformed_feature"] for entry in a3_manifest["entries"]]
    a4_names = [entry["transformed_feature"] for entry in a4_manifest["entries"]]
    a3_set, a4_set = set(a3_names), set(a4_names)
    if a3_set - a4_set:
        raise ValueError("A4 unexpectedly omits an A3 feature")
    if a4_set - a3_set != FREQUENCY_TRANSFORMED_NAMES:
        raise ValueError("A3/A4 feature difference is not exactly the approved frequency set")
    expected_a4_order = [name for name in a4_names if name not in FREQUENCY_TRANSFORMED_NAMES]
    if expected_a4_order != a3_names:
        raise ValueError("A4 changed the relative order of non-frequency A3 features")
