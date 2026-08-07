"""
Audit the ASOS GraphReturns dataset.

Run:
    python scripts/audit_dataset.py

Loads all six raw pickle files, computes summary statistics,
and prints a structured report. Does NOT mutate any data.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import ml.data.compat  # noqa: E402,F401 — must precede any pandas unpickling

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

FILES = {
    "event_train": "event_table_training.p",
    "event_test": "event_table_testing.p",
    "customer_train": "customer_nodes_training.p",
    "customer_test": "customer_nodes_testing.p",
    "product_train": "product_nodes_training.p",
    "product_test": "product_nodes_testing.p",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all(raw_dir: Path = RAW_DIR) -> dict:
    data = {}
    for key, fname in FILES.items():
        fpath = raw_dir / fname
        if not fpath.exists():
            print(f"[ERROR] Missing file: {fpath}")
            sys.exit(1)
        obj = load_pickle(fpath)
        data[key] = obj
    return data


# ---------------------------------------------------------------------------
# Generic inspection helpers
# ---------------------------------------------------------------------------

def _find_col(df: "pd.DataFrame", keywords: list) -> "str | None":
    """Return first column whose lowercased name contains any of the keywords."""
    for col in df.columns:
        lo = col.lower()
        if any(k in lo for k in keywords):
            return col
    return None


def _sep(title: str = "", width: int = 70) -> None:
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'='*2} {title} {'='*(pad)}")
    else:
        print("=" * width)


def inspect_object(key: str, obj) -> None:
    _sep(key)
    print(f"  type         : {type(obj).__name__}")

    if isinstance(obj, pd.DataFrame):
        print(f"  shape        : {obj.shape}")
        print(f"  memory       : {obj.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        print(f"  columns ({len(obj.columns)}) : {list(obj.columns)}")
        print()
        dup_cols = [c for c in obj.columns if list(obj.columns).count(c) > 1]
        if dup_cols:
            print(f"  [WARNING] Duplicate column names: {sorted(set(dup_cols))}")
        print("  dtypes:")
        for i, (col, dt) in enumerate(zip(obj.columns, obj.dtypes)):
            # Access by positional index to avoid duplicate-name ambiguity
            s = obj.iloc[:, i]
            null_n = int(s.isna().sum())
            null_pct = 100 * null_n / len(obj) if len(obj) else 0
            print(f"    {col:<45} {str(dt):<12}  nulls={null_n} ({null_pct:.1f}%)")
    elif isinstance(obj, dict):
        print(f"  keys ({len(obj)})    : {list(obj.keys())[:20]}")
    elif isinstance(obj, list):
        print(f"  length       : {len(obj)}")
        if obj:
            print(f"  first item   : {type(obj[0]).__name__}")
    else:
        print(f"  repr         : {repr(obj)[:200]}")


# ---------------------------------------------------------------------------
# Event table audit
# ---------------------------------------------------------------------------

def audit_events(ev_tr: pd.DataFrame, ev_te: pd.DataFrame) -> dict:
    _sep("EVENT TABLE AUDIT")

    results = {}
    for label, df in [("TRAINING", ev_tr), ("TESTING", ev_te)]:
        print(f"\n--- {label} ({len(df):,} rows) ---")

        # Detect target column
        possible_targets = [c for c in df.columns if "return" in c.lower()]
        print(f"  Return-related columns : {possible_targets}")

        # Try common target names
        target_col = None
        for candidate in ["returned", "return", "isReturn", "is_return", "label",
                          "target", "y", "Return", "RETURN", "Returned"]:
            if candidate in df.columns:
                target_col = candidate
                break
        if target_col is None and possible_targets:
            target_col = possible_targets[0]

        results[label.lower() + "_target_col"] = target_col

        if target_col:
            vc = df[target_col].value_counts(dropna=False).sort_index()
            print(f"  Target column          : {target_col!r}")
            print(f"  Unique target values   : {sorted(df[target_col].dropna().unique().tolist())}")
            for val, cnt in vc.items():
                print(f"    {val} : {cnt:>10,}  ({100*cnt/len(df):.2f}%)")
        else:
            print("  [WARNING] No obvious target column found")

        # Customer / product IDs
        cust_col = _find_col(df, ["customerid", "userid", "user_id"])
        prod_col = _find_col(df, ["variantid", "productid", "itemid"])

        results[label.lower() + "_cust_col"] = cust_col
        results[label.lower() + "_prod_col"] = prod_col

        if cust_col:
            print(f"  Customer col           : {cust_col!r}  unique={df[cust_col].nunique():,}")
        if prod_col:
            print(f"  Product col            : {prod_col!r}  unique={df[prod_col].nunique():,}")

        # Timestamp detection
        ts_cols = [c for c in df.columns if any(k in c.lower() for k in
                   ["date", "time", "ts", "timestamp", "created", "ordered"])]
        print(f"  Timestamp-like cols    : {ts_cols}")

        # Duplicate customer-product pairs
        if cust_col and prod_col:
            n_pairs = len(df[[cust_col, prod_col]].drop_duplicates())
            n_dup = len(df) - n_pairs
            print(f"  Dup cust-product pairs : {n_dup:,}")

        # Post-return / suspicious columns
        suspicious = [c for c in df.columns if any(k in c.lower() for k in
                      ["returnrate", "return_rate", "returncount", "return_count",
                       "returnreason", "return_reason", "nreturns", "num_return",
                       "returnsper", "returns_per"])]
        print(f"  Suspicious cols        : {suspicious}")

        results[label.lower() + "_df"] = df
        results[label.lower() + "_cust_col_name"] = cust_col
        results[label.lower() + "_prod_col_name"] = prod_col
        results[label.lower() + "_target_col_name"] = target_col

    return results


# ---------------------------------------------------------------------------
# Customer node audit
# ---------------------------------------------------------------------------

def audit_customers(cust_tr: pd.DataFrame, cust_te: pd.DataFrame) -> dict:
    _sep("CUSTOMER NODE AUDIT")
    results = {}

    id_col_tr = _find_col(cust_tr, ["customerid", "userid", "user_id"])
    id_col_te = _find_col(cust_te, ["customerid", "userid", "user_id"])

    for label, df, id_col in [("TRAINING", cust_tr, id_col_tr),
                               ("TESTING", cust_te, id_col_te)]:
        print(f"\n--- {label} ({len(df):,} rows) ---")
        print(f"  ID col            : {id_col!r}")
        if id_col:
            print(f"  Unique IDs        : {df[id_col].nunique():,}")
        print(f"  Columns           : {list(df.columns)}")

        # Categoricals cardinality
        print("\n  Column cardinalities / stats:")
        for i, col in enumerate(df.columns):
            s = df.iloc[:, i]
            null_n = int(s.isna().sum())
            if s.dtype == object or str(s.dtype) == "category":
                card = s.nunique()
                top = s.value_counts().index[:3].tolist()
                print(f"    {col:<45} object   card={card:<8}  top={top}  nulls={null_n}")
            else:
                try:
                    print(f"    {col:<45} {str(s.dtype):<8}  "
                          f"min={s.min():.3g}  max={s.max():.3g}  "
                          f"mean={s.mean():.3g}  nulls={null_n}")
                except Exception:
                    print(f"    {col:<45} {str(s.dtype):<8}  (non-numeric)  nulls={null_n}")

        # Suspicious aggregate columns
        suspicious = [c for c in df.columns if any(k in c.lower() for k in
                      ["returnrate", "return_rate", "returnsper", "returns_per",
                       "returncount", "return_count", "nreturn", "num_return",
                       "returnreason", "return_reason"])]
        print(f"\n  Suspicious (return-aggregate) cols : {suspicious}")

    # Train/test overlap
    if id_col_tr and id_col_te:
        ids_tr = set(cust_tr[id_col_tr].dropna())
        ids_te = set(cust_te[id_col_te].dropna())
        overlap = ids_tr & ids_te
        only_te = ids_te - ids_tr
        print(f"\n  Train customer IDs     : {len(ids_tr):,}")
        print(f"  Test  customer IDs     : {len(ids_te):,}")
        print(f"  Overlap (in both)      : {len(overlap):,}  ({100*len(overlap)/len(ids_te):.1f}% of test)")
        print(f"  Test-only (cold-start) : {len(only_te):,}  ({100*len(only_te)/len(ids_te):.1f}% of test)")
        results["cust_ids_tr"] = ids_tr
        results["cust_ids_te"] = ids_te
        results["cust_overlap"] = overlap
        results["cust_only_te"] = only_te

    results["id_col_tr"] = id_col_tr
    results["id_col_te"] = id_col_te
    return results


# ---------------------------------------------------------------------------
# Product node audit
# ---------------------------------------------------------------------------

def audit_products(prod_tr: pd.DataFrame, prod_te: pd.DataFrame) -> dict:
    _sep("PRODUCT NODE AUDIT")
    results = {}

    id_col_tr = _find_col(prod_tr, ["variantid", "productid", "itemid"])
    id_col_te = _find_col(prod_te, ["variantid", "productid", "itemid"])

    for label, df, id_col in [("TRAINING", prod_tr, id_col_tr),
                               ("TESTING", prod_te, id_col_te)]:
        print(f"\n--- {label} ({len(df):,} rows) ---")
        print(f"  ID col            : {id_col!r}")
        if id_col:
            print(f"  Unique IDs        : {df[id_col].nunique():,}")
        print(f"  Columns           : {list(df.columns)}")

        print("\n  Column cardinalities / stats:")
        for i, col in enumerate(df.columns):
            s = df.iloc[:, i]
            null_n = int(s.isna().sum())
            if s.dtype == object or str(s.dtype) == "category":
                card = s.nunique()
                top = s.value_counts().index[:3].tolist()
                print(f"    {col:<45} object   card={card:<8}  top={top}  nulls={null_n}")
            else:
                try:
                    print(f"    {col:<45} {str(s.dtype):<8}  "
                          f"min={s.min():.3g}  max={s.max():.3g}  "
                          f"mean={s.mean():.3g}  nulls={null_n}")
                except Exception:
                    print(f"    {col:<45} {str(s.dtype):<8}  (non-numeric)  nulls={null_n}")

        suspicious = [c for c in df.columns if any(k in c.lower() for k in
                      ["returnrate", "return_rate", "returnsper", "returns_per",
                       "returncount", "return_count", "nreturn", "num_return",
                       "returnreason", "return_reason"])]
        print(f"\n  Suspicious (return-aggregate) cols : {suspicious}")

    if id_col_tr and id_col_te:
        ids_tr = set(prod_tr[id_col_tr].dropna())
        ids_te = set(prod_te[id_col_te].dropna())
        overlap = ids_tr & ids_te
        only_te = ids_te - ids_tr
        print(f"\n  Train product IDs      : {len(ids_tr):,}")
        print(f"  Test  product IDs      : {len(ids_te):,}")
        print(f"  Overlap (in both)      : {len(overlap):,}  ({100*len(overlap)/len(ids_te):.1f}% of test)")
        print(f"  Test-only (cold-start) : {len(only_te):,}  ({100*len(only_te)/len(ids_te):.1f}% of test)")
        results["prod_ids_tr"] = ids_tr
        results["prod_ids_te"] = ids_te
        results["prod_overlap"] = overlap
        results["prod_only_te"] = only_te

    results["id_col_tr"] = id_col_tr
    results["id_col_te"] = id_col_te
    return results


# ---------------------------------------------------------------------------
# Cold-start analysis
# ---------------------------------------------------------------------------

def audit_cold_start(ev_tr, ev_te):
    _sep("COLD-START ANALYSIS")

    # Get training IDs from event table (most reliable)
    ev_cust_col    = _find_col(ev_tr, ["customerid", "userid", "user_id"])
    ev_prod_col    = _find_col(ev_tr, ["variantid", "productid", "itemid"])
    ev_cust_col_te = _find_col(ev_te, ["customerid", "userid", "user_id"])
    ev_prod_col_te = _find_col(ev_te, ["variantid", "productid", "itemid"])

    print(f"\n  Event train cust col   : {ev_cust_col!r}")
    print(f"  Event train prod col   : {ev_prod_col!r}")
    print(f"  Event test  cust col   : {ev_cust_col_te!r}")
    print(f"  Event test  prod col   : {ev_prod_col_te!r}")

    if not (ev_cust_col and ev_prod_col and ev_cust_col_te and ev_prod_col_te):
        print("  [SKIP] Cannot determine cold-start without customer and product columns.")
        return

    known_custs = set(ev_tr[ev_cust_col].dropna())
    known_prods = set(ev_tr[ev_prod_col].dropna())

    n_te = len(ev_te)
    te_c = ev_te[ev_cust_col_te]
    te_p = ev_te[ev_prod_col_te]

    known_c = te_c.isin(known_custs)
    known_p = te_p.isin(known_prods)

    kk = (known_c & known_p).sum()
    nk = (~known_c & known_p).sum()
    kn = (known_c & ~known_p).sum()
    nn = (~known_c & ~known_p).sum()

    print(f"\n  Test events total                    : {n_te:,}")
    print(f"  Known customer + Known product       : {kk:>10,}  ({100*kk/n_te:.2f}%)")
    print(f"  New   customer + Known product       : {nk:>10,}  ({100*nk/n_te:.2f}%)")
    print(f"  Known customer + New   product       : {kn:>10,}  ({100*kn/n_te:.2f}%)")
    print(f"  New   customer + New   product       : {nn:>10,}  ({100*nn/n_te:.2f}%)")

    pct_new_cust = 100 * (~known_c).sum() / n_te
    pct_new_prod = 100 * (~known_p).sum() / n_te
    print(f"\n  Test rows with new customer          : {(~known_c).sum():>10,}  ({pct_new_cust:.2f}%)")
    print(f"  Test rows with new product           : {(~known_p).sum():>10,}  ({pct_new_prod:.2f}%)")

    return {
        "known_cust_known_prod": kk,
        "new_cust_known_prod": nk,
        "known_cust_new_prod": kn,
        "new_cust_new_prod": nn,
        "n_test": n_te,
    }


# ---------------------------------------------------------------------------
# Timestamp / ordering analysis
# ---------------------------------------------------------------------------

def audit_ordering(ev_tr: pd.DataFrame, ev_te: pd.DataFrame) -> None:
    _sep("TEMPORAL / ORDERING ANALYSIS")

    for label, df in [("TRAINING", ev_tr), ("TESTING", ev_te)]:
        ts_cols = [c for c in df.columns if any(k in c.lower() for k in
                   ["date", "time", "ts", "timestamp", "created", "order", "purchase"])]
        print(f"\n--- {label} ---")
        if not ts_cols:
            print("  No timestamp columns found.")
            continue
        for col in ts_cols:
            s = df[col].dropna()
            print(f"  {col!r}: dtype={df[col].dtype}  nulls={df[col].isna().sum()}")
            if pd.api.types.is_numeric_dtype(s):
                print(f"    min={s.min():.3g}  max={s.max():.3g}  "
                      f"(possible epoch={pd.to_datetime(s, unit='s', errors='coerce').min()} "
                      f"to {pd.to_datetime(s, unit='s', errors='coerce').max()})")
            else:
                try:
                    parsed = pd.to_datetime(s, errors="coerce")
                    print(f"    parsed min={parsed.min()}  max={parsed.max()}")
                except Exception:
                    print(f"    sample={s.iloc[:3].tolist()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*70)
    print("  ReturnGuard — ASOS GraphReturns Dataset Audit")
    print("="*70)

    print("\nLoading files...")
    data = load_all()
    print("All files loaded.\n")

    # Raw object inspection
    for key, obj in data.items():
        inspect_object(key, obj)

    # Event audit
    audit_events(data["event_train"], data["event_test"])

    # Customer audit
    audit_customers(data["customer_train"], data["customer_test"])

    # Product audit
    audit_products(data["product_train"], data["product_test"])

    # Ordering
    audit_ordering(data["event_train"], data["event_test"])

    # Cold-start
    audit_cold_start(data["event_train"], data["event_test"])

    _sep("DONE")
    print()


if __name__ == "__main__":
    main()
