"""Tests for ml.data.splits determinism and correctness."""

from ml.data.schema import EVENT_CUST_COL
from ml.data.splits import (
    customer_grouped_split,
    inner_early_stopping_split,
    random_stratified_split,
)


def test_customer_grouped_split_deterministic_across_calls(synthetic_df):
    v1 = customer_grouped_split(synthetic_df)
    v2 = customer_grouped_split(synthetic_df)
    assert (v1.values == v2.values).all()


def test_customer_grouped_split_invariant_to_row_order(synthetic_df):
    shuffled = synthetic_df.sample(frac=1.0, random_state=123)
    v_original = customer_grouped_split(synthetic_df)
    v_shuffled = customer_grouped_split(shuffled)
    # Re-align to compare the same rows regardless of order.
    assert (v_original.values == v_shuffled.reindex(synthetic_df.index).values).all()


def test_customer_grouped_split_customers_disjoint(synthetic_df):
    is_val = customer_grouped_split(synthetic_df)
    train_custs = set(synthetic_df.loc[~is_val, EVENT_CUST_COL])
    val_custs = set(synthetic_df.loc[is_val, EVENT_CUST_COL])
    assert train_custs.isdisjoint(val_custs)


def test_customer_grouped_split_all_events_for_customer_on_one_side(synthetic_df):
    is_val = customer_grouped_split(synthetic_df)
    per_customer_sides = synthetic_df.assign(is_val=is_val).groupby(EVENT_CUST_COL)["is_val"].nunique()
    assert (per_customer_sides == 1).all()


def test_customer_grouped_split_fraction_reasonable(synthetic_df):
    is_val = customer_grouped_split(synthetic_df)
    frac = is_val.mean()
    # With ~200 synthetic customers and 100 hash buckets, expect roughly
    # 20% but allow slack for small-N discretisation.
    assert 0.05 < frac < 0.40


def test_random_stratified_split_deterministic(synthetic_df):
    v1 = random_stratified_split(synthetic_df, random_state=42)
    v2 = random_stratified_split(synthetic_df, random_state=42)
    assert (v1.values == v2.values).all()


def test_random_stratified_split_fraction(synthetic_df):
    is_val = random_stratified_split(synthetic_df, val_fraction=0.2, random_state=42)
    assert abs(is_val.mean() - 0.2) < 0.02


def test_random_stratified_split_preserves_class_balance(synthetic_df):
    from ml.data.schema import TARGET_COL
    is_val = random_stratified_split(synthetic_df, random_state=42)
    train_prev = synthetic_df.loc[~is_val, TARGET_COL].mean()
    val_prev = synthetic_df.loc[is_val, TARGET_COL].mean()
    assert abs(train_prev - val_prev) < 0.05


# ---------------------------------------------------------------------------
# Stage 2: inner_early_stopping_split
# ---------------------------------------------------------------------------

def test_inner_split_primary_val_matches_customer_grouped_split(synthetic_df):
    """The three-way split's "primary_val" label must be byte-identical
    to customer_grouped_split's True rows — same bucket cutoff, same
    hash scheme. This is the guarantee that Stage 2's final evaluation
    fold is exactly Stage 1's, not a lookalike."""
    is_val_stage1 = customer_grouped_split(synthetic_df)
    labels = inner_early_stopping_split(synthetic_df)
    is_val_stage2 = labels == "primary_val"
    assert (is_val_stage1.values == is_val_stage2.values).all()


def test_inner_split_three_partitions_cover_all_rows(synthetic_df):
    labels = inner_early_stopping_split(synthetic_df)
    assert set(labels.unique()) <= {"primary_val", "early_stop", "inner_train"}
    assert len(labels) == len(synthetic_df)


def test_inner_split_early_stop_disjoint_from_primary_val_and_inner_train(synthetic_df):
    labels = inner_early_stopping_split(synthetic_df)
    primary_val_custs = set(synthetic_df.loc[labels == "primary_val", EVENT_CUST_COL])
    early_stop_custs = set(synthetic_df.loc[labels == "early_stop", EVENT_CUST_COL])
    inner_train_custs = set(synthetic_df.loc[labels == "inner_train", EVENT_CUST_COL])

    assert primary_val_custs.isdisjoint(early_stop_custs)
    assert primary_val_custs.isdisjoint(inner_train_custs)
    assert early_stop_custs.isdisjoint(inner_train_custs)


def test_inner_split_all_events_for_customer_on_one_partition(synthetic_df):
    labels = inner_early_stopping_split(synthetic_df)
    per_customer = (
        synthetic_df.assign(fold=labels).groupby(EVENT_CUST_COL)["fold"].nunique()
    )
    assert (per_customer == 1).all()


def test_inner_split_deterministic_and_order_invariant(synthetic_df):
    shuffled = synthetic_df.sample(frac=1.0, random_state=99)
    l1 = inner_early_stopping_split(synthetic_df)
    l2 = inner_early_stopping_split(shuffled)
    assert (l1.values == l2.reindex(synthetic_df.index).values).all()
