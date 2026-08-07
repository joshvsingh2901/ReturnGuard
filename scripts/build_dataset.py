"""
Build the Stage 1 joined training dataset and validation splits.

Loads event_table_training.p, customer_nodes_training.p, and
product_nodes_training.p ONLY (never the test files), joins them,
cleans the yearOfBirth sentinel, computes both the primary
(customer-grouped) and secondary (random stratified) validation splits,
and persists the result to data/processed/ for reuse by
scripts/train_baseline.py.

Run:
    python scripts/build_dataset.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.joins import (
    build_joined_training_frame,
    has_customer_node_mask,
    has_product_node_mask,
)
from ml.data.splits import customer_grouped_split, random_stratified_split
from ml.features.preprocessing import clean_year_of_birth
from ml.data.schema import EVENT_CUST_COL, TARGET_COL

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "train_joined.pkl"


def main():
    print("=" * 70)
    print("  ReturnGuard — Stage 1 Dataset Build")
    print("=" * 70)

    t0 = time.time()
    print("\nLoading and joining training tables (event -> customer -> product)...")
    df = build_joined_training_frame()
    print(f"  Joined frame shape: {df.shape}  ({time.time()-t0:.1f}s)")

    n_events = len(df)
    cust_coverage = has_customer_node_mask(df).mean()
    prod_coverage = has_product_node_mask(df).mean()
    print(f"  Customer node coverage: {cust_coverage*100:.2f}%")
    print(f"  Product node coverage:  {prod_coverage*100:.2f}%")

    print("\nCleaning yearOfBirth sentinel (1900 -> NaN)...")
    n_sentinel = (df["yearOfBirth"] == 1900).sum()
    df = clean_year_of_birth(df)
    print(f"  Converted {n_sentinel:,} sentinel values ({n_sentinel/n_events*100:.2f}%)")

    print("\nComputing primary split (customer-grouped, deterministic)...")
    is_val_primary = customer_grouped_split(df)
    print(f"  Validation fraction: {is_val_primary.mean()*100:.2f}%")
    train_custs = set(df.loc[~is_val_primary, EVENT_CUST_COL])
    val_custs = set(df.loc[is_val_primary, EVENT_CUST_COL])
    overlap = train_custs & val_custs
    print(f"  Train customers: {len(train_custs):,}  Val customers: {len(val_custs):,}")
    print(f"  Customer overlap: {len(overlap)} (must be 0)")
    assert len(overlap) == 0, "Customer-grouped split leaked customers across folds!"

    print("\nComputing secondary split (random stratified, diagnostic only)...")
    is_val_secondary = random_stratified_split(df)
    print(f"  Validation fraction: {is_val_secondary.mean()*100:.2f}%")

    df["is_val_primary"] = is_val_primary
    df["is_val_secondary"] = is_val_secondary

    print(f"\nOverall target prevalence: {df[TARGET_COL].mean():.4f}")
    print(f"  Primary train-fold prevalence: {df.loc[~is_val_primary, TARGET_COL].mean():.4f}")
    print(f"  Primary val-fold prevalence:   {df.loc[is_val_primary, TARGET_COL].mean():.4f}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUTPUT_FILE)
    print(f"\nSaved joined dataset to {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size/1e6:.1f} MB)")
    print(f"Total build time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
