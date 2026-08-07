"""
sklearn preprocessing pipeline for the Stage 1 baseline.

A single ColumnTransformer bundled inside the model Pipeline, so fitting
happens exclusively on the training fold and inference always applies the
exact transform learned there. Two variants are built: strict (no price)
and with-price, matching LR-A / LR-B.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.data.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES_CUSTOMER,
    CATEGORICAL_FEATURES_PRODUCT,
    DERIVED_FEATURES,
    FREQUENCY_FEATURES,
    NUMERIC_FEATURES,
    NUMERIC_FEATURES_PRICE,
    YEAR_OF_BIRTH_SENTINEL,
)


def clean_year_of_birth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the confirmed missing-value sentinel yearOfBirth == 1900 to
    NaN. No other values are altered or clipped — see docs/leakage-audit.md
    and docs/stage1-baseline.md for the evidence this is a sentinel and
    not a real birth year (it is the single most common value by a wide
    margin, and forms an isolated spike far from the main distribution).

    This is deterministic data cleaning of a known sentinel, applied
    before the learned preprocessing pipeline (not fit on data).
    """
    df = df.copy()
    df["yearOfBirth"] = df["yearOfBirth"].replace(
        YEAR_OF_BIRTH_SENTINEL, np.nan
    )
    return df


def build_preprocessor(include_price: bool) -> ColumnTransformer:
    """
    Build the ColumnTransformer for LR-A (include_price=False) or LR-B
    (include_price=True).

    Numeric: median impute, then standard-scale.
    Binary (isMale, premier): most-frequent impute (already 0/1 encoded
        in the source data; not one-hot encoded).
    Categorical, customer-side (shippingCountry): most-frequent impute,
        THEN one-hot encode. shippingCountry can be NaN when the customer
        node is missing (~5.2% of events); without an imputer here,
        OneHotEncoder silently creates its own "nan" category, which would
        re-expose has-customer-node status as an implicit feature — exactly
        what ml.data.joins deliberately avoids for the other customer
        fields. Imputing first keeps this column consistent with
        isMale/premier.
    Categorical, product-side (productType, brandDesc): one-hot encode
        directly, no imputer. Missingness here is already handled upstream
        via an explicit "__MISSING__" token (ml.data.joins), so these
        columns are never NaN by the time they reach this pipeline.
    """
    numeric_cols = NUMERIC_FEATURES + (NUMERIC_FEATURES_PRICE if include_price else [])

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=False)),
        ("scale", StandardScaler()),
    ])

    binary_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=False)),
    ])

    categorical_customer_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=False)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    categorical_product_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("bin", binary_pipe, BINARY_FEATURES),
            ("cat_cust", categorical_customer_pipe, CATEGORICAL_FEATURES_CUSTOMER),
            ("cat_prod", categorical_product_pipe, CATEGORICAL_FEATURES_PRODUCT),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Return the output feature names of a fitted ColumnTransformer."""
    return list(preprocessor.get_feature_names_out())


def build_gbdt_preprocessor(
    artifact_safe: bool = True,
    include_derived: bool = False,
    include_frequency: bool = False,
) -> ColumnTransformer:
    """
    Stage 2 preprocessing for the XGBoost pipeline. Differs from
    build_preprocessor (Stage 1's Logistic Regression preprocessor) in
    ways specifically required for a tree model:

    - No StandardScaler: tree splits are scale-invariant, and scaling
      would force an imputer to run first (StandardScaler cannot handle
      NaN), which would defeat native missing-value routing below.
    - Product-side numeric columns (avgGbpPrice, avgDiscountValue, and
      the optional derived/frequency features) are never imputed — NaN is
      preserved so XGBoost's native missing-value routing can use it.
      Stage 0/1 confirmed product-node missingness is target-neutral
      (55.5% vs. 55.2% return rate), so this is safe.
    - Dense output (sparse_threshold=0): avoids any ambiguity about
      whether an explicit NaN survives a sparse-matrix conversion
      alongside one-hot blocks; at this column count (<= 45) the memory
      cost is small and the transparency is worth it for a leakage-
      sensitive pipeline.

    artifact_safe controls customer-side handling:
      - True (the honest A2+ rungs): assumes
        ml.features.imputation.DonorImputer has ALREADY run as an earlier
        pipeline step and guarantees zero NaN in yearOfBirth/isMale/
        premier/shippingCountry. This function then does NOT impute those
        columns again — passthrough for numeric/binary, direct one-hot
        for shippingCountry. If DonorImputer were skipped, NaN would
        reach the encoders and this preprocessor would surface it (raise
        or emit NaN) rather than silently falling back to a
        leakage-prone pattern; XGBoost's native NaN routing on these
        specific columns would otherwise let a tree exploit exactly the
        missing-node artifact this design exists to suppress.
      - False (A1 diagnostic ONLY): reproduces Stage 1's median/mode
        imputation exactly, so the leakage probe has a contaminated
        preprocessing path to detect. Never used for the honest baseline
        or for any reported headline metric.
    """
    numeric_prod_cols = list(NUMERIC_FEATURES_PRICE)
    if include_derived:
        numeric_prod_cols = numeric_prod_cols + DERIVED_FEATURES
    if include_frequency:
        numeric_prod_cols = numeric_prod_cols + FREQUENCY_FEATURES

    if artifact_safe:
        cust_numeric_pipe = "passthrough"
        binary_pipe = "passthrough"
        categorical_customer_pipe = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
    else:
        cust_numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=False)),
        ])
        binary_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", add_indicator=False)),
        ])
        categorical_customer_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", add_indicator=False)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

    categorical_product_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num_cust", cust_numeric_pipe, NUMERIC_FEATURES),
            ("num_prod", "passthrough", numeric_prod_cols),
            ("bin", binary_pipe, BINARY_FEATURES),
            ("cat_cust", categorical_customer_pipe, CATEGORICAL_FEATURES_CUSTOMER),
            ("cat_prod", categorical_product_pipe, CATEGORICAL_FEATURES_PRODUCT),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
