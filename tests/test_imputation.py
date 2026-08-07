"""Tests for ml.features.imputation.DonorImputer."""

import numpy as np
import pandas as pd
import pytest

from ml.features.imputation import CUSTOMER_COLS, DonorImputer


def test_fills_all_customer_fields_from_one_donor_row():
    """The four customer fields must come from a SINGLE donor row, not
    independent per-field sampling — verified by checking that every
    imputed row's combination actually exists somewhere in the donor
    pool (independent sampling would very likely produce combinations
    absent from the pool)."""
    n = 500
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "yearOfBirth": rng.integers(1950, 2005, size=n).astype(float),
        "isMale": rng.integers(0, 2, size=n).astype(float),
        "premier": rng.integers(0, 2, size=n).astype(float),
        "shippingCountry": rng.choice(["A", "B", "C", "D", "E"], size=n),
    })
    # complete donor pool
    train = X.copy()
    # target rows: fully missing
    target = pd.DataFrame({c: [np.nan] * 50 for c in CUSTOMER_COLS})
    target["shippingCountry"] = np.nan

    imp = DonorImputer(random_state=1)
    imp.fit(train)
    out = imp.transform(target)

    pool_tuples = set(map(tuple, train[CUSTOMER_COLS].to_numpy()))
    out_tuples = list(map(tuple, out[CUSTOMER_COLS].to_numpy()))
    assert all(t in pool_tuples for t in out_tuples), (
        "Imputed row combination not found in donor pool — suggests "
        "fields were sampled independently rather than jointly."
    )


def test_deterministic_with_fixed_seed():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        "yearOfBirth": rng.integers(1950, 2005, size=n).astype(float),
        "isMale": rng.integers(0, 2, size=n).astype(float),
        "premier": rng.integers(0, 2, size=n).astype(float),
        "shippingCountry": rng.choice(["A", "B", "C"], size=n),
    })
    train = X.iloc[:200]
    val = X.iloc[200:].copy()
    val.loc[:, CUSTOMER_COLS] = np.nan

    imp1 = DonorImputer(random_state=7)
    imp1.fit(train)
    out1 = imp1.transform(val)

    imp2 = DonorImputer(random_state=7)
    imp2.fit(train)
    out2 = imp2.transform(val)

    assert out1.equals(out2)


def test_donor_assignment_is_invariant_to_row_order_and_batching():
    """A missing customer's donor must not change with inference batching."""
    train = pd.DataFrame({
        "hash(customerId)": [10, 11, 12, 13],
        "yearOfBirth": [1980.0, 1990.0, 2000.0, 1970.0],
        "isMale": [1.0, 0.0, 1.0, 0.0],
        "premier": [0.0, 1.0, 0.0, 1.0],
        "shippingCountry": ["A", "B", "C", "D"],
    })
    target = pd.DataFrame({
        "hash(customerId)": [100, 101, 102, 103],
        "yearOfBirth": [np.nan] * 4,
        "isMale": [np.nan] * 4,
        "premier": [np.nan] * 4,
        "shippingCountry": [np.nan] * 4,
    })
    imp = DonorImputer(random_state=7).fit(train)
    whole = imp.transform(target)
    batched = pd.concat([imp.transform(target.iloc[:2]), imp.transform(target.iloc[2:])])
    reordered = imp.transform(target.iloc[::-1]).loc[target.index]

    assert whole[CUSTOMER_COLS].equals(batched[CUSTOMER_COLS])
    assert whole[CUSTOMER_COLS].equals(reordered[CUSTOMER_COLS])


def test_donor_pool_built_from_fit_data_only(synthetic_df_with_missing):
    """fit() must only see the fold passed to it — donor pool size and
    contents must not depend on any data outside that fold."""
    df = synthetic_df_with_missing
    train = df.iloc[: len(df) // 2]

    imp = DonorImputer(random_state=1)
    imp.fit(train[["yearOfBirth", "isMale", "premier", "shippingCountry"]])

    expected_pool_size = int(
        train[["yearOfBirth", "isMale", "premier", "shippingCountry"]]
        .notna().all(axis=1).sum()
    )
    assert len(imp.donor_pool_) == expected_pool_size


def test_resulting_rows_have_no_missing_customer_values(synthetic_df_with_missing):
    df = synthetic_df_with_missing
    train = df.iloc[: len(df) // 2]
    val = df.iloc[len(df) // 2:]

    imp = DonorImputer(random_state=42)
    train_out = imp.fit_transform(train[CUSTOMER_COLS])
    val_out = imp.transform(val[CUSTOMER_COLS])

    assert train_out[CUSTOMER_COLS].isna().sum().sum() == 0
    assert val_out[CUSTOMER_COLS].isna().sum().sum() == 0


def test_sentinel_only_rows_get_median_not_donor_draw():
    """A row with node present (isMale not null) but yearOfBirth missing
    is the sentinel case, not the node-missing artifact — it should get
    the plain train-fold median, not a joint donor draw."""
    train = pd.DataFrame({
        "yearOfBirth": [1980.0, 1990.0, 2000.0],
        "isMale": [1.0, 0.0, 1.0],
        "premier": [0.0, 1.0, 0.0],
        "shippingCountry": ["A", "B", "C"],
    })
    target = pd.DataFrame({
        "yearOfBirth": [np.nan],
        "isMale": [1.0],       # node present
        "premier": [1.0],
        "shippingCountry": ["A"],
    })

    imp = DonorImputer(random_state=1)
    imp.fit(train)
    out = imp.transform(target)

    assert out.loc[0, "yearOfBirth"] == imp.year_median_
    # other fields must be untouched (still the row's own real values)
    assert out.loc[0, "isMale"] == 1.0
    assert out.loc[0, "premier"] == 1.0
    assert out.loc[0, "shippingCountry"] == "A"


def test_raises_if_no_complete_rows_to_donate_from():
    train = pd.DataFrame({
        "yearOfBirth": [np.nan],
        "isMale": [np.nan],
        "premier": [np.nan],
        "shippingCountry": [np.nan],
    })
    imp = DonorImputer(random_state=1)
    with pytest.raises(ValueError):
        imp.fit(train)


def test_transform_without_fit_raises():
    target = pd.DataFrame({
        "yearOfBirth": [np.nan], "isMale": [np.nan],
        "premier": [np.nan], "shippingCountry": [np.nan],
    })
    imp = DonorImputer(random_state=1)
    with pytest.raises(AttributeError):
        imp.transform(target)
