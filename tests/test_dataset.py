"""
Lightweight tests for dataset loading and structural invariants.

Run:
    pytest tests/test_dataset.py -v

These tests verify that the raw data files satisfy the structural
properties the rest of the pipeline depends on. They do NOT train
any model or mutate any data.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import ml.data.compat  # noqa: E402,F401 — must precede any pandas unpickling

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

EXPECTED_FILES = {
    "event_table_training.p",
    "event_table_testing.p",
    "customer_nodes_training.p",
    "customer_nodes_testing.p",
    "product_nodes_training.p",
    "product_nodes_testing.p",
}

EVENT_CUST_COL = "hash(customerId)"
EVENT_PROD_COL = "hash(variantID)"
TARGET_COL = "isReturned"
CUST_ID_COL = "hash(customerId)"
PROD_ID_COL = "hash(variantID)"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _load(filename: str) -> pd.DataFrame:
    import pickle
    with open(RAW_DIR / filename, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def event_train():
    return _load("event_table_training.p")


@pytest.fixture(scope="session")
def event_test():
    return _load("event_table_testing.p")


@pytest.fixture(scope="session")
def customer_train():
    return _load("customer_nodes_training.p")


@pytest.fixture(scope="session")
def customer_test():
    return _load("customer_nodes_testing.p")


@pytest.fixture(scope="session")
def product_train():
    return _load("product_nodes_training.p")


@pytest.fixture(scope="session")
def product_test():
    return _load("product_nodes_testing.p")


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------

def test_all_expected_files_present():
    actual = {p.name for p in RAW_DIR.glob("*.p")}
    missing = EXPECTED_FILES - actual
    assert not missing, f"Missing raw data files: {missing}"


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

def test_event_train_is_dataframe(event_train):
    assert isinstance(event_train, pd.DataFrame)


def test_event_test_is_dataframe(event_test):
    assert isinstance(event_test, pd.DataFrame)


def test_customer_train_is_dataframe(customer_train):
    assert isinstance(customer_train, pd.DataFrame)


def test_customer_test_is_dataframe(customer_test):
    assert isinstance(customer_test, pd.DataFrame)


def test_product_train_is_dataframe(product_train):
    assert isinstance(product_train, pd.DataFrame)


def test_product_test_is_dataframe(product_test):
    assert isinstance(product_test, pd.DataFrame)


# ---------------------------------------------------------------------------
# Required columns exist
# ---------------------------------------------------------------------------

def test_event_train_has_required_columns(event_train):
    for col in [EVENT_CUST_COL, EVENT_PROD_COL, TARGET_COL]:
        assert col in event_train.columns, f"Missing column: {col!r}"


def test_event_test_has_required_columns(event_test):
    for col in [EVENT_CUST_COL, EVENT_PROD_COL, TARGET_COL]:
        assert col in event_test.columns, f"Missing column: {col!r}"


def test_customer_train_has_id_column(customer_train):
    assert CUST_ID_COL in customer_train.columns


def test_customer_test_has_id_column(customer_test):
    assert CUST_ID_COL in customer_test.columns


def test_product_train_has_id_column(product_train):
    assert PROD_ID_COL in product_train.columns


def test_product_test_has_id_column(product_test):
    assert PROD_ID_COL in product_test.columns


# ---------------------------------------------------------------------------
# Target is binary
# ---------------------------------------------------------------------------

def test_target_train_is_binary(event_train):
    values = set(event_train[TARGET_COL].unique())
    assert values == {0, 1}, f"Expected {{0, 1}}, got {values}"


def test_target_test_is_binary(event_test):
    values = set(event_test[TARGET_COL].unique())
    assert values == {0, 1}, f"Expected {{0, 1}}, got {values}"


def test_target_train_has_no_nulls(event_train):
    assert event_train[TARGET_COL].isna().sum() == 0


def test_target_test_has_no_nulls(event_test):
    assert event_test[TARGET_COL].isna().sum() == 0


# ---------------------------------------------------------------------------
# Non-trivial sizes
# ---------------------------------------------------------------------------

def test_event_train_has_rows(event_train):
    assert len(event_train) > 1_000_000, "Training events unexpectedly small"


def test_event_test_has_rows(event_test):
    assert len(event_test) > 1_000_000, "Test events unexpectedly small"


def test_customer_train_has_rows(customer_train):
    assert len(customer_train) > 100_000


def test_product_train_has_rows(product_train):
    assert len(product_train) > 100_000


# ---------------------------------------------------------------------------
# Identifier columns are populated
# ---------------------------------------------------------------------------

def test_event_train_no_null_ids(event_train):
    assert event_train[EVENT_CUST_COL].isna().sum() == 0
    assert event_train[EVENT_PROD_COL].isna().sum() == 0


def test_event_test_no_null_ids(event_test):
    assert event_test[EVENT_CUST_COL].isna().sum() == 0
    assert event_test[EVENT_PROD_COL].isna().sum() == 0


# ---------------------------------------------------------------------------
# Schema consistency: event tables have same columns in both splits
# ---------------------------------------------------------------------------

def test_event_table_schema_consistency(event_train, event_test):
    assert list(event_train.columns) == list(event_test.columns), (
        "Event train and test column schemas differ"
    )


def test_customer_node_schema_consistency(customer_train, customer_test):
    assert list(customer_train.columns) == list(customer_test.columns), (
        "Customer train and test column schemas differ"
    )


def test_product_node_schema_consistency(product_train, product_test):
    assert list(product_train.columns) == list(product_test.columns), (
        "Product train and test column schemas differ"
    )


# ---------------------------------------------------------------------------
# Known data quality issues are documented (not blocking — warns)
# ---------------------------------------------------------------------------

def test_customer_duplicate_column_is_known_defect(customer_train):
    """The duplicate 'customerId_level_return_code_D' column is a known upstream defect."""
    col_list = list(customer_train.columns)
    duplicates = {c for c in col_list if col_list.count(c) > 1}
    assert duplicates == {"customerId_level_return_code_D"}, (
        f"Unexpected duplicate columns: {duplicates}. "
        "If this changed, update the leakage audit."
    )


def test_product_duplicate_column_is_known_defect(product_train):
    """The duplicate 'variantID_level_return_code_D' column is a known upstream defect."""
    col_list = list(product_train.columns)
    duplicates = {c for c in col_list if col_list.count(c) > 1}
    assert duplicates == {"variantID_level_return_code_D"}, (
        f"Unexpected duplicate columns: {duplicates}. "
        "If this changed, update the leakage audit."
    )
