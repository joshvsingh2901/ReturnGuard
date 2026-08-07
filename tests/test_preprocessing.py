"""Tests for ml.features.preprocessing."""

import numpy as np
import pandas as pd

from ml.data.schema import TARGET_COL, YEAR_OF_BIRTH_SENTINEL
from ml.features.preprocessing import (
    build_preprocessor,
    clean_year_of_birth,
    get_feature_names,
)


def test_year_of_birth_sentinel_converted_to_nan():
    df = pd.DataFrame({"yearOfBirth": [1900, 1980, 1900, 1995]})
    cleaned = clean_year_of_birth(df)
    assert cleaned["yearOfBirth"].isna().sum() == 2
    assert cleaned.loc[1, "yearOfBirth"] == 1980
    assert cleaned.loc[3, "yearOfBirth"] == 1995


def test_year_of_birth_other_values_unchanged():
    df = pd.DataFrame({"yearOfBirth": [1930, 1960, 2010, 2020]})
    cleaned = clean_year_of_birth(df)
    assert cleaned["yearOfBirth"].tolist() == [1930, 1960, 2010, 2020]


def test_year_of_birth_sentinel_constant_matches_schema():
    assert YEAR_OF_BIRTH_SENTINEL == 1900


def test_preprocessor_fits_and_transforms(synthetic_df):
    pre = build_preprocessor(include_price=True)
    X = pre.fit_transform(synthetic_df)
    assert X.shape[0] == len(synthetic_df)
    assert X.shape[1] > 0


def test_preprocessor_train_val_dimensions_match(synthetic_df):
    train = synthetic_df.iloc[:1500]
    val = synthetic_df.iloc[1500:]

    pre = build_preprocessor(include_price=True)
    X_train = pre.fit_transform(train)
    X_val = pre.transform(val)

    assert X_train.shape[1] == X_val.shape[1]


def test_preprocessor_unseen_category_does_not_crash(synthetic_df):
    train = synthetic_df.iloc[:1500].copy()
    val = synthetic_df.iloc[1500:].copy()
    val["shippingCountry"] = "Country_NEVER_SEEN"
    val["brandDesc"] = "Brand_NEVER_SEEN"

    pre = build_preprocessor(include_price=True)
    pre.fit(train)
    X_val = pre.transform(val)  # must not raise

    assert X_val.shape[0] == len(val)
    assert np.isfinite(X_val.toarray() if hasattr(X_val, "toarray") else X_val).all()


def test_feature_names_exclude_target(synthetic_df):
    pre = build_preprocessor(include_price=True)
    pre.fit(synthetic_df)
    names = get_feature_names(pre)
    assert not any(TARGET_COL in n for n in names)


def test_preprocessor_handles_missing_customer_and_product_record(synthetic_df_with_missing):
    pre = build_preprocessor(include_price=True)
    X = pre.fit_transform(synthetic_df_with_missing)
    dense = X.toarray() if hasattr(X, "toarray") else X
    assert np.isfinite(dense).all()
    assert X.shape[0] == len(synthetic_df_with_missing)


def test_preprocessor_strict_excludes_price_columns(synthetic_df):
    pre = build_preprocessor(include_price=False)
    pre.fit(synthetic_df)
    names = get_feature_names(pre)
    assert not any("avgGbpPrice" in n for n in names)
    assert not any("avgDiscountValue" in n for n in names)


def test_preprocessor_with_price_includes_price_columns(synthetic_df):
    pre = build_preprocessor(include_price=True)
    pre.fit(synthetic_df)
    names = get_feature_names(pre)
    assert any("avgGbpPrice" in n for n in names)
    assert any("avgDiscountValue" in n for n in names)
