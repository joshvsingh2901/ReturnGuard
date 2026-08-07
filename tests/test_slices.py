"""Tests for ml.evaluation.slices cold-start quadrant logic."""

import numpy as np
import pandas as pd

from ml.data.schema import EVENT_CUST_COL, EVENT_PROD_COL
from ml.evaluation.slices import assign_cold_start_quadrant, evaluate_slices


def test_quadrant_assignment_all_four_cases():
    val_df = pd.DataFrame({
        EVENT_CUST_COL: [1, 1, 2, 2],
        EVENT_PROD_COL: [10, 20, 10, 20],
    })
    train_customers = {1}
    train_products = {10}

    quadrant = assign_cold_start_quadrant(val_df, train_customers, train_products)

    assert quadrant.iloc[0] == "known_customer_known_product"
    assert quadrant.iloc[1] == "known_customer_new_product"
    assert quadrant.iloc[2] == "new_customer_known_product"
    assert quadrant.iloc[3] == "new_customer_new_product"


def test_quadrant_assignment_all_new_when_train_sets_empty():
    val_df = pd.DataFrame({
        EVENT_CUST_COL: [1, 2],
        EVENT_PROD_COL: [10, 20],
    })
    quadrant = assign_cold_start_quadrant(val_df, set(), set())
    assert (quadrant == "new_customer_new_product").all()


def test_evaluate_slices_suppresses_small_slices():
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    y_proba = rng.uniform(size=n)
    quadrant = pd.Series(["small_slice"] * n)

    results = evaluate_slices(y_true, y_proba, quadrant, min_slice_size=1000)
    assert results["small_slice"]["suppressed"] is True
    assert results["small_slice"]["n"] == n


def test_evaluate_slices_computes_metrics_for_large_slices():
    rng = np.random.default_rng(0)
    n = 6000
    y_true = rng.integers(0, 2, size=n)
    y_proba = rng.uniform(size=n)
    quadrant = pd.Series(["big_slice"] * n)

    results = evaluate_slices(y_true, y_proba, quadrant, min_slice_size=1000)
    assert results["big_slice"]["suppressed"] is False
    assert results["big_slice"]["n"] == n
    assert results["big_slice"]["roc_auc"] is not None


def test_evaluate_slices_covers_all_rows():
    rng = np.random.default_rng(0)
    n = 10000
    y_true = rng.integers(0, 2, size=n)
    y_proba = rng.uniform(size=n)
    quadrant = pd.Series(rng.choice(["a", "b"], size=n))

    results = evaluate_slices(y_true, y_proba, quadrant, min_slice_size=1000)
    total_n = sum(r["n"] for r in results.values())
    assert total_n == n
