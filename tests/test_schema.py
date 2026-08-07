"""
Tests that the leakage-safe feature schema is exactly what Stage 1
documentation claims — the single most important test in this stage,
since it is the last line of defense against accidental reintroduction
of a leaky column into the model.
"""

from ml.data.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CUSTOMER_SAFE_FEATURES,
    LEAKY_COLUMNS,
    LEAKY_CUSTOMER_COLUMNS,
    LEAKY_PRODUCT_COLUMNS,
    NUMERIC_FEATURES,
    NUMERIC_FEATURES_PRICE,
    PRODUCT_SAFE_FEATURES_STRICT,
    PRODUCT_SAFE_FEATURES_WITH_PRICE,
    TARGET_COL,
)


def test_expected_safe_customer_features():
    assert set(CUSTOMER_SAFE_FEATURES) == {
        "yearOfBirth", "isMale", "shippingCountry", "premier",
    }


def test_expected_safe_product_features_strict():
    assert set(PRODUCT_SAFE_FEATURES_STRICT) == {"productType", "brandDesc"}


def test_expected_safe_product_features_with_price():
    assert set(PRODUCT_SAFE_FEATURES_WITH_PRICE) == {
        "productType", "brandDesc", "avgGbpPrice", "avgDiscountValue",
    }


def test_safe_features_disjoint_from_leaky_columns():
    safe = set(CUSTOMER_SAFE_FEATURES) | set(PRODUCT_SAFE_FEATURES_WITH_PRICE)
    assert safe.isdisjoint(set(LEAKY_COLUMNS)), (
        f"Safe features overlap with leaky columns: {safe & set(LEAKY_COLUMNS)}"
    )


def test_target_not_in_any_feature_list():
    all_feature_lists = (
        CUSTOMER_SAFE_FEATURES
        + PRODUCT_SAFE_FEATURES_WITH_PRICE
        + NUMERIC_FEATURES
        + NUMERIC_FEATURES_PRICE
        + BINARY_FEATURES
        + CATEGORICAL_FEATURES
    )
    assert TARGET_COL not in all_feature_lists


def test_leaky_customer_columns_include_known_leakage():
    expected = {
        "returnsPerCustomer", "customerReturnRate", "salesPerCustomer",
    }
    assert expected.issubset(set(LEAKY_CUSTOMER_COLUMNS))


def test_leaky_product_columns_include_known_leakage():
    expected = {
        "returnsPerProduct", "productReturnRate", "salesPerProduct",
    }
    assert expected.issubset(set(LEAKY_PRODUCT_COLUMNS))


def test_all_return_code_columns_are_leaky():
    for suffix in "ABCDEFGHIJKL":
        assert f"customerId_level_return_code_{suffix}" in LEAKY_CUSTOMER_COLUMNS
        assert f"variantID_level_return_code_{suffix}" in LEAKY_PRODUCT_COLUMNS
