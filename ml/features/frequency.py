"""
Product-side frequency encoding for Stage 2.

Classified SUSPICIOUS/experimental (see docs/leakage-audit.md,
docs/stage2-gbdt.md). Counts are fold-local and never touch the target —
strictly safer than the banned salesPerProduct/salesPerCustomer, which
are global constants confirmed to span the test period. But without event
timestamps, "count of train-fold purchases sharing this ID" still carries
an unfalsifiable within-window-exposure assumption, the same caveat Stage
1 attached to avgGbpPrice/avgDiscountValue. Never used outside the A4
ablation rung.

No customer-side frequency feature exists here, deliberately. Under the
customer-grouped primary split, every validation customer is guaranteed
unseen in the train fold, so a customer-ID frequency feature would be
informative during training and constant-zero at validation — a train/
serve skew. See tests/test_frequency.py for a regression test asserting
this feature is absent.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml.data.loaders import RAW_DIR, PRODUCT_TRAIN_FILE, _load_pickle
from ml.data.schema import EVENT_PROD_COL, PRODUCT_PARENT_ID_COL, SUPPLIER_ID_COL

ID_COLS = [EVENT_PROD_COL, PRODUCT_PARENT_ID_COL, SUPPLIER_ID_COL]
OUT_COLS = ["variant_event_count", "product_event_count", "supplier_event_count"]


def load_product_ids_train(raw_dir=None) -> pd.DataFrame:
    """
    Load ONLY the identifier columns from the training product node
    table: hash(variantID) (join key), hash(productID) (parent product),
    hash(supplierRef) (supplier). No aggregate/leaky columns selected.
    """
    path = (raw_dir or RAW_DIR) / PRODUCT_TRAIN_FILE
    df = _load_pickle(path)
    return df[ID_COLS].copy()


def attach_frequency_id_columns(df: pd.DataFrame, raw_dir=None) -> pd.DataFrame:
    """
    Left-join hash(productID)/hash(supplierRef) onto the Stage 1 joined
    training frame via hash(variantID). Row count is asserted unchanged —
    the product node table has unique variant IDs (validated in Stage 1's
    ml.data.joins), so this join cannot multiply rows.
    """
    ids = load_product_ids_train(raw_dir=raw_dir)
    n_before = len(df)
    out = df.merge(ids, on=EVENT_PROD_COL, how="left")
    if len(out) != n_before:
        raise AssertionError(
            f"attach_frequency_id_columns changed row count: "
            f"{n_before} -> {len(out)}. Duplicate-key join bug."
        )
    return out


class ProductFrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Adds three fold-local exposure counts, all learned in fit(X) from
    whatever fold is passed (normally the training fold):

      variant_event_count = train-fold row count sharing hash(variantID)
      product_event_count = train-fold row count sharing hash(productID)
      supplier_event_count = train-fold row count sharing hash(supplierRef)

    hash(variantID) is the event join key and is never missing, so
    variant_event_count is 0 for a variant genuinely unseen in the
    training fold, never NaN.

    hash(productID)/hash(supplierRef) are NaN exactly when the product
    node itself is missing (same join-miss as productType/brandDesc/
    avgGbpPrice). In that case the corresponding count is NaN, not 0 — an
    unknown ID has an undefined count. 0 is reserved for a KNOWN ID that
    simply had zero train-fold events (a genuinely new variant, product,
    or supplier relative to the training fold).
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ProductFrequencyEncoder":
        self.variant_counts_ = X[EVENT_PROD_COL].value_counts()
        self.product_counts_ = X[PRODUCT_PARENT_ID_COL].value_counts()
        self.supplier_counts_ = X[SUPPLIER_ID_COL].value_counts()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["variant_event_count"] = (
            X[EVENT_PROD_COL].map(self.variant_counts_).fillna(0.0)
        )

        prod_known = X[PRODUCT_PARENT_ID_COL].notna()
        X["product_event_count"] = np.nan
        X.loc[prod_known, "product_event_count"] = (
            X.loc[prod_known, PRODUCT_PARENT_ID_COL]
            .map(self.product_counts_)
            .fillna(0.0)
        )

        sup_known = X[SUPPLIER_ID_COL].notna()
        X["supplier_event_count"] = np.nan
        X.loc[sup_known, "supplier_event_count"] = (
            X.loc[sup_known, SUPPLIER_ID_COL]
            .map(self.supplier_counts_)
            .fillna(0.0)
        )

        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return base + OUT_COLS
