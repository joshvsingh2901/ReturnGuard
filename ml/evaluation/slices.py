"""
Cold-start slice evaluation.

"Known" customer/product is defined strictly relative to the TRAIN FOLD
of whichever split produced the predictions being evaluated — never the
full training set, and never the test set (which Stage 1 never loads).

For the primary customer-grouped split, every validation customer is by
construction unseen in the train fold, so only the two "new customer"
quadrants exist. For the secondary random split, customers and products
can appear on both sides, so all four quadrants are meaningful.
"""

import numpy as np
import pandas as pd

from ml.data.schema import EVENT_CUST_COL, EVENT_PROD_COL, TARGET_COL
from ml.evaluation.metrics import compute_metrics

MIN_SLICE_SIZE = 5000


def assign_cold_start_quadrant(
    val_df: pd.DataFrame,
    train_customer_ids: set,
    train_product_ids: set,
) -> pd.Series:
    """
    Label each validation row with one of four quadrant strings, based on
    whether its customer/product ID appeared in the train fold.
    """
    known_cust = val_df[EVENT_CUST_COL].isin(train_customer_ids)
    known_prod = val_df[EVENT_PROD_COL].isin(train_product_ids)

    quadrant = np.where(
        known_cust & known_prod, "known_customer_known_product",
        np.where(known_cust & ~known_prod, "known_customer_new_product",
        np.where(~known_cust & known_prod, "new_customer_known_product",
                 "new_customer_new_product")),
    )
    return pd.Series(quadrant, index=val_df.index, name="cold_start_quadrant")


def evaluate_slices(
    y_true: pd.Series,
    y_proba: np.ndarray,
    quadrant: pd.Series,
    min_slice_size: int = MIN_SLICE_SIZE,
) -> dict:
    """
    Compute metrics separately for each cold-start quadrant present in
    the data. Quadrants with fewer than min_slice_size rows have their
    metrics suppressed (reported as None) since AUC/log-loss estimates
    on very small slices are unstable and misleading.
    """
    results = {}
    y_proba = np.asarray(y_proba, dtype=float)
    y_true_arr = np.asarray(y_true)

    for q in sorted(quadrant.unique()):
        mask = (quadrant == q).to_numpy()
        n = int(mask.sum())
        if n < min_slice_size:
            results[q] = {"n": n, "suppressed": True, "reason": f"n < {min_slice_size}"}
            continue
        m = compute_metrics(y_true_arr[mask], y_proba[mask])
        m["suppressed"] = False
        results[q] = m

    return results
