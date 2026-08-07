"""
Validation splitting for Stage 1 model development.

Primary split — customer-grouped, deterministic:
    All events for a given customer are assigned to exactly one side
    (train or validation) based on a stable hash of the customer ID. This
    means validation customers are always unseen during training, matching
    the cold-start character of the real ASOS test split (71.9% new
    customers — see docs/dataset-audit.md). It also means the split stays
    valid for any future model that adds customer-history features,
    because no validation customer's events ever leak into the train fold.

    Uses hashlib.md5 rather than Python's builtin hash(), which is
    randomised per-process (PYTHONHASHSEED) and would NOT be reproducible
    across runs or machines.

Secondary split — random stratified, diagnostic only:
    A plain stratified event-level split. Included only to quantify how
    much "optimism" the customer-grouped split removes; never reported as
    the headline result (see docs/stage1-baseline.md).

Stage 2 addition — inner early-stopping split:
    XGBoost (Stage 2) needs an early-stopping fold that is NOT the primary
    validation fold, since using primary validation for early stopping
    would mean it influenced model selection and could no longer serve as
    an untouched final evaluation set. inner_early_stopping_split further
    subdivides the Stage 1 TRAIN fold (customer hash buckets 20-99) into
    an early-stopping slice (buckets 20-29) and an inner-training slice
    (buckets 30-99), using the exact same deterministic MD5 bucket scheme
    as the primary split — so "primary_val" rows under this function are
    byte-identical to customer_grouped_split's True rows (verified in
    tests/test_splits.py).
"""

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data.schema import EVENT_CUST_COL, TARGET_COL

VAL_FRACTION = 0.20
_HASH_BUCKETS = 100


def _stable_hash_bucket(values: pd.Series, n_buckets: int = _HASH_BUCKETS) -> np.ndarray:
    """
    Map each value to a deterministic integer bucket in [0, n_buckets) via
    MD5. Stable across processes, machines, and row order — unlike
    Python's built-in hash(), which is salted per-process.

    Hashes only the unique values then maps back, since group columns
    (customer IDs) typically repeat across many rows.
    """
    uniques = values.unique()
    bucket_of = {
        v: int(hashlib.md5(str(v).encode("utf-8")).hexdigest(), 16) % n_buckets
        for v in uniques
    }
    return values.map(bucket_of).to_numpy()


def compute_customer_hash_bucket(
    df: pd.DataFrame, customer_col: str = EVENT_CUST_COL
) -> np.ndarray:
    """
    Public accessor for the raw MD5 hash bucket (0-99) per row, keyed by
    customer_col. Exposed so downstream split functions (e.g.
    inner_early_stopping_split) can build additional partitions on the
    exact same deterministic bucket assignment customer_grouped_split
    uses, rather than re-deriving a parallel hash scheme that could drift
    out of sync.
    """
    return _stable_hash_bucket(df[customer_col])


def customer_grouped_split(
    df: pd.DataFrame,
    val_fraction: float = VAL_FRACTION,
    customer_col: str = EVENT_CUST_COL,
) -> pd.Series:
    """
    Return a boolean Series, aligned to df.index, True for validation rows.

    Deterministic: depends only on the customer ID values, not on row
    order, random seeds, or process state. Calling this twice on the same
    (possibly reordered) data yields identical assignments.
    """
    cutoff = int(val_fraction * _HASH_BUCKETS)
    buckets = compute_customer_hash_bucket(df, customer_col)
    is_val = buckets < cutoff
    return pd.Series(is_val, index=df.index, name="is_validation")


def inner_early_stopping_split(
    df: pd.DataFrame,
    customer_col: str = EVENT_CUST_COL,
    primary_val_fraction: float = VAL_FRACTION,
    early_stop_fraction: float = 0.10,
) -> pd.Series:
    """
    Three-way deterministic partition for Stage 2 XGBoost model
    development, built on the identical MD5 bucket scheme as
    customer_grouped_split:

      - "primary_val"  : buckets [0, 20)   — Stage 1's primary validation
                          fold, byte-identical to customer_grouped_split's
                          True rows. Reserved for a single final
                          evaluation; never used for early stopping or
                          ablation/hyperparameter selection.
      - "early_stop"    : buckets [20, 30) — held out from the Stage 1
                          train fold, used only to trigger XGBoost's
                          early stopping and to select between ablation
                          rungs / hyperparameter configurations.
      - "inner_train"   : buckets [30, 100) — the data models are
                          actually fit on.

    Calling code is responsible for filtering to the partition it needs;
    this function only assigns labels. Deterministic and row-order
    invariant for the same reasons as customer_grouped_split.
    """
    buckets = compute_customer_hash_bucket(df, customer_col)
    primary_cutoff = int(primary_val_fraction * _HASH_BUCKETS)
    early_stop_cutoff = primary_cutoff + int(early_stop_fraction * _HASH_BUCKETS)

    labels = np.where(
        buckets < primary_cutoff, "primary_val",
        np.where(buckets < early_stop_cutoff, "early_stop", "inner_train"),
    )
    return pd.Series(labels, index=df.index, name="stage2_fold")


def random_stratified_split(
    df: pd.DataFrame,
    val_fraction: float = VAL_FRACTION,
    target_col: str = TARGET_COL,
    random_state: int = 42,
) -> pd.Series:
    """
    Diagnostic-only secondary split: plain stratified random split at the
    event level (customers/products may appear on both sides). Returns a
    boolean Series aligned to df.index, True for validation rows.
    """
    _, val_idx = train_test_split(
        df.index,
        test_size=val_fraction,
        stratify=df[target_col],
        random_state=random_state,
    )
    is_val = pd.Series(False, index=df.index, name="is_validation")
    is_val.loc[val_idx] = True
    return is_val
