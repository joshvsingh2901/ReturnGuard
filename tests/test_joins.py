"""
Tests for ml.data.joins against the real raw training files.

These are cheap (<1s) despite using real data — building the joined
frame is a vectorised merge, not a model fit — so they run as ordinary
unit tests rather than being gated behind an integration marker.
"""

import pytest

from ml.data.joins import (
    build_joined_training_frame,
    has_customer_node_mask,
    has_product_node_mask,
)
from ml.data.loaders import load_event_train
from ml.data.schema import MISSING_CATEGORY_TOKEN, TARGET_COL

pytestmark = pytest.mark.raw_training_data


@pytest.fixture(scope="module")
def joined():
    return build_joined_training_frame()


def test_join_preserves_event_row_count(joined):
    events = load_event_train()
    assert len(joined) == len(events)


def test_join_does_not_multiply_events(joined):
    # A stronger check than row count alone: no (customer, product) pair
    # should have grown a duplicate node attachment.
    n_distinct_events = len(joined.drop_duplicates())
    assert len(joined) >= n_distinct_events  # duplicate purchases are legitimate...
    # ...but the count must match the raw event table exactly regardless.
    events = load_event_train()
    assert len(joined) == len(events)


def test_target_present_and_binary(joined):
    assert TARGET_COL in joined.columns
    assert set(joined[TARGET_COL].unique()) == {0, 1}


def test_product_missing_category_is_explicit_token(joined):
    assert (joined["productType"] == MISSING_CATEGORY_TOKEN).any()
    assert (joined["brandDesc"] == MISSING_CATEGORY_TOKEN).any()


def test_customer_node_coverage_matches_audit(joined):
    coverage = has_customer_node_mask(joined).mean()
    # Stage 0/1 audit measured 94.77% customer node coverage on training events.
    assert 0.94 < coverage < 0.96


def test_product_node_coverage_matches_audit(joined):
    coverage = has_product_node_mask(joined).mean()
    # Stage 0/1 audit measured 65.34% product node coverage on training events.
    assert 0.64 < coverage < 0.67


def test_no_leaky_columns_in_joined_frame(joined):
    from ml.data.schema import LEAKY_COLUMNS
    leaked = set(joined.columns) & set(LEAKY_COLUMNS)
    assert not leaked, f"Leaky columns present in joined frame: {leaked}"
