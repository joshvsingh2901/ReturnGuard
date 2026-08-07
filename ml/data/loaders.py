"""
Raw pickle loading for the ASOS GraphReturns dataset.

Stage 1 uses TRAINING files only (event_table_training.p,
customer_nodes_training.p, product_nodes_training.p). The testing files
must not be imported by any Stage 1 training/evaluation code path — the
provided test split is reserved for final evaluation in a later stage.
"""

import pickle
from pathlib import Path

import ml.data.compat  # noqa: F401 — must precede any pandas unpickling
import pandas as pd

from ml.data.schema import (
    CUSTOMER_ID_COL,
    CUSTOMER_SAFE_FEATURES,
    EVENT_CUST_COL,
    EVENT_PROD_COL,
    PRODUCT_ID_COL,
    PRODUCT_SAFE_FEATURES_WITH_PRICE,
    TARGET_COL,
)

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

EVENT_TRAIN_FILE = "event_table_training.p"
CUSTOMER_TRAIN_FILE = "customer_nodes_training.p"
PRODUCT_TRAIN_FILE = "product_nodes_training.p"

# Filenames that Stage 1 must never load.
FORBIDDEN_TEST_FILES = {
    "event_table_testing.p",
    "customer_nodes_testing.p",
    "product_nodes_testing.p",
}


def _load_pickle(path: Path) -> pd.DataFrame:
    if path.name in FORBIDDEN_TEST_FILES:
        raise ValueError(
            f"Refusing to load {path.name!r}: the provided ASOS test split "
            "must not be accessed during Stage 1 development."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def load_event_train(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the raw training event table, unmodified (includes target)."""
    df = _load_pickle(raw_dir / EVENT_TRAIN_FILE)
    return df[[EVENT_CUST_COL, EVENT_PROD_COL, TARGET_COL]].copy()


def load_customer_train_safe(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Load training customer nodes, selecting only the ID column plus the
    leakage-safe feature columns. Selecting columns by name before any
    downstream processing means the duplicate 'customerId_level_return_code_D'
    source column (a data-quality defect — see docs/leakage-audit.md) never
    enters the pipeline, since it is not among the safe columns selected.
    """
    df = _load_pickle(raw_dir / CUSTOMER_TRAIN_FILE)
    cols = [CUSTOMER_ID_COL] + CUSTOMER_SAFE_FEATURES
    return df[cols].copy()


def load_product_train_safe(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Load training product nodes, selecting only the ID column plus the
    leakage-safe feature columns (including avgGbpPrice/avgDiscountValue,
    used by LR-B). Selecting by name drops the duplicate
    'variantID_level_return_code_D' source column before it can cause
    ambiguous column access downstream.
    """
    df = _load_pickle(raw_dir / PRODUCT_TRAIN_FILE)
    cols = [PRODUCT_ID_COL] + PRODUCT_SAFE_FEATURES_WITH_PRICE
    return df[cols].copy()
