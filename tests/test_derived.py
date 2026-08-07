"""Tests for ml.features.derived.DerivedPriceFeatures."""

import numpy as np
import pandas as pd
import pytest

from ml.features.derived import DerivedPriceFeatures


def test_discount_amount_hand_calculated():
    X = pd.DataFrame({
        "avgGbpPrice": [100.0, 50.0],
        "avgDiscountValue": [20.0, 10.0],
        "productType": ["A", "B"],
        "brandDesc": ["X", "Y"],
    })
    d = DerivedPriceFeatures().fit(X)
    out = d.transform(X)
    assert out["discount_amount"].tolist() == pytest.approx([20.0, 5.0])


def test_price_rel_type_median_hand_calculated():
    X = pd.DataFrame({
        "avgGbpPrice": [10.0, 20.0, 30.0, 100.0],
        "avgDiscountValue": [0.0, 0.0, 0.0, 0.0],
        "productType": ["A", "A", "A", "B"],
        "brandDesc": ["X", "X", "X", "X"],
    })
    d = DerivedPriceFeatures().fit(X)
    out = d.transform(X)
    # median of type A prices [10,20,30] = 20
    assert out.loc[0, "price_rel_type_median"] == pytest.approx(10.0 / 20.0)
    assert out.loc[1, "price_rel_type_median"] == pytest.approx(1.0)
    assert out.loc[2, "price_rel_type_median"] == pytest.approx(30.0 / 20.0)
    # type B has only itself -> median = 100 -> ratio = 1.0
    assert out.loc[3, "price_rel_type_median"] == pytest.approx(1.0)


def test_price_rel_brand_median_hand_calculated():
    X = pd.DataFrame({
        "avgGbpPrice": [10.0, 30.0],
        "avgDiscountValue": [0.0, 0.0],
        "productType": ["A", "A"],
        "brandDesc": ["X", "Y"],
    })
    d = DerivedPriceFeatures().fit(X)
    out = d.transform(X)
    # each brand has only one row -> ratio always 1.0
    assert out["price_rel_brand_median"].tolist() == pytest.approx([1.0, 1.0])


def test_statistics_learned_from_train_only():
    train = pd.DataFrame({
        "avgGbpPrice": [10.0, 20.0, 30.0],
        "avgDiscountValue": [0.0, 0.0, 0.0],
        "productType": ["A", "A", "A"],
        "brandDesc": ["X", "X", "X"],
    })
    d = DerivedPriceFeatures().fit(train)
    train_median = d.type_medians_["A"]
    assert train_median == pytest.approx(20.0)

    # transforming a different frame must not change the fitted stats
    val = pd.DataFrame({
        "avgGbpPrice": [1000.0],
        "avgDiscountValue": [0.0],
        "productType": ["A"],
        "brandDesc": ["X"],
    })
    d.transform(val)
    assert d.type_medians_["A"] == pytest.approx(20.0)


def test_fit_on_train_plus_val_gives_different_medians():
    """Meaningfulness guard: confirms the previous test isn't passing
    vacuously because train and train+val happen to share medians."""
    train = pd.DataFrame({
        "avgGbpPrice": [10.0, 20.0, 30.0],
        "avgDiscountValue": [0.0] * 3,
        "productType": ["A"] * 3,
        "brandDesc": ["X"] * 3,
    })
    train_plus_val = pd.concat([
        train,
        pd.DataFrame({
            "avgGbpPrice": [1000.0, 2000.0],
            "avgDiscountValue": [0.0, 0.0],
            "productType": ["A", "A"],
            "brandDesc": ["X", "X"],
        }),
    ], ignore_index=True)

    median_train_only = DerivedPriceFeatures().fit(train).type_medians_["A"]
    median_train_plus_val = DerivedPriceFeatures().fit(train_plus_val).type_medians_["A"]
    assert median_train_only != pytest.approx(median_train_plus_val)


def test_unseen_category_falls_back_to_global_median():
    train = pd.DataFrame({
        "avgGbpPrice": [10.0, 20.0, 30.0],
        "avgDiscountValue": [0.0] * 3,
        "productType": ["A", "A", "B"],
        "brandDesc": ["X", "X", "X"],
    })
    d = DerivedPriceFeatures().fit(train)
    global_median = d.global_median_

    val = pd.DataFrame({
        "avgGbpPrice": [50.0],
        "avgDiscountValue": [0.0],
        "productType": ["NEVER_SEEN"],
        "brandDesc": ["X"],
    })
    out = d.transform(val)
    assert out.loc[0, "price_rel_type_median"] == pytest.approx(50.0 / global_median)


def test_missing_price_propagates_nan():
    train = pd.DataFrame({
        "avgGbpPrice": [10.0, 20.0],
        "avgDiscountValue": [5.0, 10.0],
        "productType": ["A", "A"],
        "brandDesc": ["X", "X"],
    })
    d = DerivedPriceFeatures().fit(train)

    val = pd.DataFrame({
        "avgGbpPrice": [np.nan],
        "avgDiscountValue": [np.nan],
        "productType": ["__MISSING__"],
        "brandDesc": ["__MISSING__"],
    })
    out = d.transform(val)
    assert out["discount_amount"].isna().all()
    assert out["price_rel_type_median"].isna().all()
    assert out["price_rel_brand_median"].isna().all()


def test_no_crash_and_no_nan_on_clean_data(synthetic_df):
    d = DerivedPriceFeatures().fit(synthetic_df)
    out = d.transform(synthetic_df)
    # synthetic_df has no missing prices by default
    assert out["discount_amount"].isna().sum() == 0
