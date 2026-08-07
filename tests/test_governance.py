"""Tests for Stage 3 feature manifests and frozen model specifications."""

import pytest

from ml.governance.feature_manifest import build_feature_manifest, validate_a3_a4_manifests
from ml.governance.model_spec import build_model_freeze_spec


def _a3_names():
    return [
        "num_cust__yearOfBirth",
        "num_prod__avgGbpPrice",
        "num_prod__discount_amount",
        "bin__isMale",
        "cat_cust__shippingCountry_Country_A",
        "cat_prod__productType_Jeans",
        "cat_prod__brandDesc_Brand_A",
    ]


def test_a3_a4_manifests_differ_only_by_frequency_features():
    a3 = build_feature_manifest(_a3_names(), "A3")
    a4 = build_feature_manifest(
        _a3_names()
        + [
            "num_prod__variant_event_count",
            "num_prod__product_event_count",
            "num_prod__supplier_event_count",
        ],
        "A4",
    )
    validate_a3_a4_manifests(a3, a4)
    assert a3["feature_count"] + 3 == a4["feature_count"]


def test_manifest_fails_closed_for_raw_identifier_or_target():
    with pytest.raises(ValueError):
        build_feature_manifest(["hash(productID)"], "A3")
    with pytest.raises(ValueError):
        build_feature_manifest(["isReturned"], "A3")


def test_freeze_spec_is_reproducible_for_same_inputs():
    manifest = build_feature_manifest(_a3_names(), "A3")
    first = build_model_freeze_spec(manifest, {"max_depth": 6}, 100, "abc")
    second = build_model_freeze_spec(manifest, {"max_depth": 6}, 100, "abc")
    assert first["freeze_spec_hash"] == second["freeze_spec_hash"]
    assert first["final_training_procedure"]["fixed_boosting_rounds"] == 101
