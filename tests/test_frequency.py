"""Tests for ml.features.frequency.ProductFrequencyEncoder."""

import numpy as np
import pandas as pd
import pytest

from ml.data.schema import EVENT_CUST_COL
from ml.features.frequency import ID_COLS, OUT_COLS, ProductFrequencyEncoder


def test_variant_count_hand_calculated():
    train = pd.DataFrame({
        "hash(variantID)": [1, 1, 1, 2],
        "hash(productID)": [10, 10, 10, 20],
        "hash(supplierRef)": [100, 100, 100, 200],
    })
    enc = ProductFrequencyEncoder().fit(train)
    out = enc.transform(train)
    assert out["variant_event_count"].tolist() == [3, 3, 3, 1]


def test_product_count_hand_calculated():
    train = pd.DataFrame({
        "hash(variantID)": [1, 2, 3],
        "hash(productID)": [10, 10, 20],
        "hash(supplierRef)": [100, 100, 200],
    })
    enc = ProductFrequencyEncoder().fit(train)
    out = enc.transform(train)
    assert out["product_event_count"].tolist() == [2, 2, 1]
    assert out["supplier_event_count"].tolist() == [2, 2, 1]


def test_counts_fit_from_train_only():
    train = pd.DataFrame({
        "hash(variantID)": [1, 1],
        "hash(productID)": [10, 10],
        "hash(supplierRef)": [100, 100],
    })
    val = pd.DataFrame({
        "hash(variantID)": [1, 1, 1],  # 3 occurrences in val, only 2 in train
        "hash(productID)": [10, 10, 10],
        "hash(supplierRef)": [100, 100, 100],
    })
    enc = ProductFrequencyEncoder().fit(train)
    out = enc.transform(val)
    # count must reflect TRAIN frequency (2), not val's own frequency (3)
    assert (out["variant_event_count"] == 2).all()


def test_unseen_known_id_maps_to_zero():
    train = pd.DataFrame({
        "hash(variantID)": [1, 1],
        "hash(productID)": [10, 10],
        "hash(supplierRef)": [100, 100],
    })
    val = pd.DataFrame({
        "hash(variantID)": [999],       # never seen in train
        "hash(productID)": [888],
        "hash(supplierRef)": [777],
    })
    enc = ProductFrequencyEncoder().fit(train)
    out = enc.transform(val)
    assert out["variant_event_count"].iloc[0] == 0
    assert out["product_event_count"].iloc[0] == 0
    assert out["supplier_event_count"].iloc[0] == 0


def test_missing_product_id_gives_nan_not_zero():
    """An unknown (missing product node) ID gets NaN, distinct from a
    known ID that simply had zero train-fold events."""
    train = pd.DataFrame({
        "hash(variantID)": [1, 2],
        "hash(productID)": [10, np.nan],
        "hash(supplierRef)": [100, np.nan],
    })
    enc = ProductFrequencyEncoder().fit(train)
    out = enc.transform(train)
    assert out.loc[1, "product_event_count"] != out.loc[1, "product_event_count"]  # NaN
    assert pd.isna(out.loc[1, "product_event_count"])
    assert pd.isna(out.loc[1, "supplier_event_count"])
    # variant count is never NaN (variant ID is the event join key, always present)
    assert not pd.isna(out.loc[1, "variant_event_count"])


def test_transform_does_not_mutate_fitted_state():
    train = pd.DataFrame({
        "hash(variantID)": [1, 1],
        "hash(productID)": [10, 10],
        "hash(supplierRef)": [100, 100],
    })
    enc = ProductFrequencyEncoder().fit(train)
    counts_before = enc.variant_counts_.copy()

    other = pd.DataFrame({
        "hash(variantID)": [1, 1, 1, 1, 1],
        "hash(productID)": [10] * 5,
        "hash(supplierRef)": [100] * 5,
    })
    enc.transform(other)
    assert enc.variant_counts_.equals(counts_before)


def test_no_customer_id_frequency_feature_exists():
    """Regression guard: under the customer-grouped primary split, every
    validation customer has a train-fold count of exactly 0 by
    construction, which would be a textbook train/serve skew. No
    customer-side frequency feature should exist anywhere."""
    assert EVENT_CUST_COL not in ID_COLS
    assert not any("customer" in c.lower() for c in OUT_COLS)


def test_no_nan_on_fully_present_data(synthetic_df_with_ids):
    df = synthetic_df_with_ids
    present = df.dropna(subset=["hash(productID)", "hash(supplierRef)"])
    enc = ProductFrequencyEncoder().fit(present)
    out = enc.transform(present)
    assert out[OUT_COLS].isna().sum().sum() == 0
