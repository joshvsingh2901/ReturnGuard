"""
Join the training event table against customer and product node tables.

event LEFT JOIN customer_nodes LEFT JOIN product_nodes, both on training
files only. Node ID uniqueness is validated before joining so a left join
is guaranteed not to multiply event rows.

Missing-node handling (see docs/leakage-audit.md and docs/stage1-baseline.md
for the evidence behind these choices):

- Customer side (~5.2% of events have no customer node): the measured
  return-rate gap between joined and unjoined events (54.8% vs 64.6%) is
  believed to be a dataset-construction artifact, not a generalisable
  signal, so no missing-node indicator is exposed in the primary model.
  Numeric/binary customer features are median/mode imputed downstream.

- Product side (~34.7% of events have no product node): the return-rate
  gap is negligible (55.2% vs 55.5%), consistent with the missingness
  being target-neutral. Categorical product features are filled with an
  explicit "__MISSING__" token rather than mode-imputed, since a third of
  rows would otherwise be forced into the modal brand/type.
"""

import pandas as pd

from ml.data.loaders import (
    load_customer_train_safe,
    load_event_train,
    load_product_train_safe,
)
from ml.data.schema import (
    CUSTOMER_ID_COL,
    EVENT_CUST_COL,
    EVENT_PROD_COL,
    MISSING_CATEGORY_TOKEN,
    PRODUCT_ID_COL,
)


def _validate_unique_ids(df: pd.DataFrame, id_col: str, label: str) -> None:
    n_dupes = len(df) - df[id_col].nunique()
    if n_dupes > 0:
        raise ValueError(
            f"{label} node table has {n_dupes} duplicate {id_col!r} values; "
            "a left join would multiply event rows."
        )


def build_joined_training_frame(raw_dir=None) -> pd.DataFrame:
    """
    Return the joined training frame: one row per training event, with
    customer and product safe features attached where available.

    Row count is guaranteed equal to len(event_table_training.p).
    """
    kwargs = {} if raw_dir is None else {"raw_dir": raw_dir}

    events = load_event_train(**kwargs)
    customers = load_customer_train_safe(**kwargs)
    products = load_product_train_safe(**kwargs)

    _validate_unique_ids(customers, CUSTOMER_ID_COL, "Customer")
    _validate_unique_ids(products, PRODUCT_ID_COL, "Product")

    n_events_before = len(events)

    joined = events.merge(
        customers,
        left_on=EVENT_CUST_COL,
        right_on=CUSTOMER_ID_COL,
        how="left",
        suffixes=("", "_cust"),
    )
    joined = joined.merge(
        products,
        left_on=EVENT_PROD_COL,
        right_on=PRODUCT_ID_COL,
        how="left",
        suffixes=("", "_prod"),
    )

    if len(joined) != n_events_before:
        raise AssertionError(
            f"Join changed row count: {n_events_before} -> {len(joined)}. "
            "This indicates a duplicate-key join multiplication bug."
        )

    # CUSTOMER_ID_COL == EVENT_CUST_COL and PRODUCT_ID_COL == EVENT_PROD_COL
    # by construction (both "hash(customerId)" / "hash(variantID)"), so
    # pandas merges the join keys into a single column with no suffix or
    # duplication — no cleanup needed here.

    # Explicit missing-category handling for product categoricals.
    for col in ["productType", "brandDesc"]:
        joined[col] = joined[col].fillna(MISSING_CATEGORY_TOKEN)

    return joined


def has_customer_node_mask(joined: pd.DataFrame) -> pd.Series:
    """Boolean mask: True where the event's customer node was found."""
    return joined["yearOfBirth"].notna()


def has_product_node_mask(joined: pd.DataFrame) -> pd.Series:
    """Boolean mask: True where the event's product node was found."""
    return joined["productType"] != MISSING_CATEGORY_TOKEN
