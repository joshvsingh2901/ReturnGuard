"""Shared fixtures for Stage 1 tests."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.schema import EVENT_CUST_COL, EVENT_PROD_COL, TARGET_COL


def make_synthetic_joined_df(
    n: int = 2000,
    n_customers: int = 200,
    n_products: int = 100,
    seed: int = 0,
    frac_missing_customer: float = 0.0,
    frac_missing_product: float = 0.0,
    include_sentinel_year: bool = True,
    include_product_ids: bool = False,
    n_parent_products: int = 40,
    n_suppliers: int = 15,
) -> pd.DataFrame:
    """
    Build a small synthetic frame with the same schema as
    build_joined_training_frame()'s output, for fast unit tests that
    should not depend on the real (multi-hundred-MB) raw pickle files.

    include_product_ids=True additionally adds hash(productID) and
    hash(supplierRef) (for Stage 2 frequency-encoding tests), each
    variant deterministically mapped to one parent product and one
    supplier, NaN wherever the product node itself is missing — matching
    the real join semantics in ml.data.joins / ml.features.frequency.
    """
    rng = np.random.default_rng(seed)

    cust_ids = rng.integers(-(2**31), 2**31, size=n_customers)
    prod_ids = rng.integers(-(2**63), 2**63, size=n_products, dtype=np.int64)

    row_cust = rng.choice(cust_ids, size=n)
    row_prod = rng.choice(prod_ids, size=n)

    if include_product_ids:
        parent_ids = rng.integers(-(2**63), 2**63, size=n_parent_products, dtype=np.int64)
        supplier_ids = rng.integers(-(2**31), 2**31, size=n_suppliers)
        # Deterministic many-to-one mapping: each distinct variant ID
        # gets exactly one parent product and one supplier.
        variant_to_parent = {v: parent_ids[i % n_parent_products] for i, v in enumerate(prod_ids)}
        variant_to_supplier = {v: supplier_ids[i % n_suppliers] for i, v in enumerate(prod_ids)}
        row_parent = np.array([variant_to_parent[v] for v in row_prod])
        row_supplier = np.array([variant_to_supplier[v] for v in row_prod])

    years = rng.integers(1950, 2005, size=n).astype(float)
    if include_sentinel_year:
        sentinel_mask = rng.random(n) < 0.05
        years[sentinel_mask] = 1900

    is_male = rng.integers(0, 2, size=n)
    premier = rng.integers(0, 2, size=n)
    countries = rng.choice(["Country_A", "Country_B", "Country_C"], size=n)
    product_types = rng.choice(["productType_A", "productType_B", "Jeans"], size=n)
    brands = rng.choice(["Brand_A", "Brand_B", "Pull&Bear"], size=n)
    price = rng.uniform(5, 100, size=n)
    discount = rng.uniform(0, 30, size=n)

    # target correlated weakly with price so LR has something to learn
    logit = 0.3 + 0.01 * (price - 50)
    prob = 1 / (1 + np.exp(-logit))
    target = (rng.random(n) < prob).astype(int)

    data = {
        EVENT_CUST_COL: row_cust,
        EVENT_PROD_COL: row_prod,
        TARGET_COL: target,
        "yearOfBirth": years,
        "isMale": is_male,
        "shippingCountry": countries,
        "premier": premier,
        "productType": product_types,
        "brandDesc": brands,
        "avgGbpPrice": price,
        "avgDiscountValue": discount,
    }
    if include_product_ids:
        data["hash(productID)"] = row_parent.astype(float)
        data["hash(supplierRef)"] = row_supplier.astype(float)
    df = pd.DataFrame(data)

    if frac_missing_customer > 0:
        mask = rng.random(n) < frac_missing_customer
        df.loc[mask, ["yearOfBirth", "isMale", "shippingCountry", "premier"]] = np.nan

    if frac_missing_product > 0:
        mask = rng.random(n) < frac_missing_product
        df.loc[mask, ["productType", "brandDesc"]] = "__MISSING__"
        df.loc[mask, ["avgGbpPrice", "avgDiscountValue"]] = np.nan
        if include_product_ids:
            df.loc[mask, ["hash(productID)", "hash(supplierRef)"]] = np.nan

    return df


@pytest.fixture
def synthetic_df():
    return make_synthetic_joined_df()


@pytest.fixture
def synthetic_df_with_missing():
    return make_synthetic_joined_df(frac_missing_customer=0.1, frac_missing_product=0.3)


@pytest.fixture
def synthetic_df_with_ids():
    """Stage 2: includes hash(productID)/hash(supplierRef) and realistic
    missingness on both the customer and product side."""
    return make_synthetic_joined_df(
        n=4000, frac_missing_customer=0.1, frac_missing_product=0.3,
        include_product_ids=True,
    )
