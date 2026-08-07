"""
Train and evaluate the Stage 1 leakage-safe baseline models.

Runs, in order, on the PRIMARY (customer-grouped) validation split:
  1. Prevalence baseline (constant train-fold prevalence)
  2. Majority-class baseline (always predicts train-fold majority class)
  3. LR-A: Logistic Regression on strict safe features (no price/discount)
  4. LR-B: Logistic Regression on safe features + avgGbpPrice/avgDiscountValue

Then, diagnostics:
  - Price/discount ablation: LR-A vs LR-B on identical data
  - Product-coverage diagnostic: LR-A/LR-B performance split by whether
    the event's product node was found
  - Cold-start quadrant table on the primary split (new-customer
    quadrants only, since the split guarantees no known customers)
  - A SEPARATE secondary-split training run (random stratified) used only
    to populate the full four-quadrant cold-start table for comparison

Never loads event_table_testing.p or any *_testing.p file.

Run:
    python scripts/build_dataset.py   # once, to materialize the joined data
    python scripts/train_baseline.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd

from ml.data.joins import has_product_node_mask
from ml.data.schema import EVENT_CUST_COL, EVENT_PROD_COL, TARGET_COL
from ml.evaluation.calibration import (
    expected_calibration_error,
    plot_reliability_curve,
    reliability_curve,
)
from ml.evaluation.metrics import compute_metrics, compute_skill_scores
from ml.evaluation.slices import assign_cold_start_quadrant, evaluate_slices
from ml.features.preprocessing import get_feature_names
from ml.models.baselines import build_majority_baseline, build_prevalence_baseline
from ml.models.logistic import build_logistic_pipeline, check_convergence

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
INPUT_FILE = PROCESSED_DIR / "train_joined.pkl"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

DUMMY_X_COL = [EVENT_CUST_COL]  # placeholder feature for DummyClassifier fit/predict


def _split(df: pd.DataFrame, is_val: pd.Series):
    return df.loc[~is_val].copy(), df.loc[is_val].copy()


def _fit_predict_dummy(builder, train, val):
    model = builder()
    model.fit(train[DUMMY_X_COL], train[TARGET_COL])
    proba = model.predict_proba(val[DUMMY_X_COL])[:, 1]
    return model, proba


def _fit_predict_lr(include_price: bool, train, val):
    model = build_logistic_pipeline(include_price=include_price)
    model.fit(train, train[TARGET_COL])
    check_convergence(model)
    proba = model.predict_proba(val)[:, 1]
    feature_names = get_feature_names(model.named_steps["preprocess"])
    return model, proba, feature_names


def run_primary_split(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("  PRIMARY SPLIT (customer-grouped) — headline results")
    print("=" * 70)

    is_val = df["is_val_primary"]
    train, val = _split(df, is_val)
    y_val = val[TARGET_COL]
    print(f"  Train: {len(train):,} rows   Val: {len(val):,} rows")
    print(f"  Train prevalence: {train[TARGET_COL].mean():.4f}   "
          f"Val prevalence: {y_val.mean():.4f}")

    results = {"n_train": len(train), "n_val": len(val)}

    # --- Trivial baselines ---
    print("\n-- Prevalence baseline --")
    _, proba_prev = _fit_predict_dummy(build_prevalence_baseline, train, val)
    m_prev = compute_metrics(y_val, proba_prev)
    print(f"  ROC-AUC={m_prev['roc_auc']}  LogLoss={m_prev['log_loss']:.4f}  "
          f"Brier={m_prev['brier']:.4f}")
    results["prevalence_baseline"] = m_prev

    print("\n-- Majority-class baseline --")
    _, proba_maj = _fit_predict_dummy(build_majority_baseline, train, val)
    m_maj = compute_metrics(y_val, proba_maj)
    print(f"  Accuracy={m_maj['accuracy']:.4f}  F1={m_maj['f1']:.4f}  "
          f"Precision={m_maj['precision']:.4f}  Recall={m_maj['recall']:.4f}")
    results["majority_baseline"] = m_maj

    # --- LR-A (strict, no price) ---
    print("\n-- LR-A (strict, no price/discount) --")
    t0 = time.time()
    lr_a, proba_a, feat_names_a = _fit_predict_lr(include_price=False, train=train, val=val)
    m_a = compute_metrics(y_val, proba_a)
    skill_a = compute_skill_scores(m_a, m_prev)
    print(f"  Fit time: {time.time()-t0:.1f}s  n_iter={lr_a.named_steps['clf'].n_iter_}")
    print(f"  ROC-AUC={m_a['roc_auc']:.4f}  LogLoss={m_a['log_loss']:.4f}  "
          f"Brier={m_a['brier']:.4f}  LogLossSkill={skill_a['log_loss_skill']:.4f}")
    results["lr_a"] = {**m_a, "skill_vs_prevalence": skill_a, "n_features": len(feat_names_a)}

    curve_a = reliability_curve(y_val, proba_a)
    ece_a = expected_calibration_error(y_val, proba_a)
    results["lr_a"]["calibration"] = {"ece": ece_a, "curve": curve_a}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_reliability_curve(curve_a, "LR-A Calibration (no price)", REPORTS_DIR / "calibration_lr_a.png")

    # --- LR-B (with price/discount) ---
    print("\n-- LR-B (with price/discount) --")
    t0 = time.time()
    lr_b, proba_b, feat_names_b = _fit_predict_lr(include_price=True, train=train, val=val)
    m_b = compute_metrics(y_val, proba_b)
    skill_b = compute_skill_scores(m_b, m_prev)
    print(f"  Fit time: {time.time()-t0:.1f}s  n_iter={lr_b.named_steps['clf'].n_iter_}")
    print(f"  ROC-AUC={m_b['roc_auc']:.4f}  LogLoss={m_b['log_loss']:.4f}  "
          f"Brier={m_b['brier']:.4f}  LogLossSkill={skill_b['log_loss_skill']:.4f}")
    results["lr_b"] = {**m_b, "skill_vs_prevalence": skill_b, "n_features": len(feat_names_b)}

    curve_b = reliability_curve(y_val, proba_b)
    ece_b = expected_calibration_error(y_val, proba_b)
    results["lr_b"]["calibration"] = {"ece": ece_b, "curve": curve_b}
    plot_reliability_curve(curve_b, "LR-B Calibration (with price)", REPORTS_DIR / "calibration_lr_b.png")

    # --- Price/discount ablation ---
    print("\n-- Price/discount ablation (LR-A vs LR-B) --")
    d_auc = m_b["roc_auc"] - m_a["roc_auc"]
    d_logloss = m_a["log_loss"] - m_b["log_loss"]  # positive = LR-B better (lower loss)
    d_brier = m_a["brier"] - m_b["brier"]
    d_ece = ece_a - ece_b
    print(f"  Delta ROC-AUC (B-A):  {d_auc:+.4f}")
    print(f"  Delta LogLoss (A-B):  {d_logloss:+.4f}  (positive = B better)")
    print(f"  Delta Brier (A-B):    {d_brier:+.4f}  (positive = B better)")
    print(f"  Delta ECE (A-B):      {d_ece:+.4f}  (positive = B better calibrated)")
    results["price_ablation"] = {
        "delta_roc_auc_b_minus_a": d_auc,
        "delta_log_loss_a_minus_b": d_logloss,
        "delta_brier_a_minus_b": d_brier,
        "delta_ece_a_minus_b": d_ece,
    }

    # --- Cold-start quadrants on primary split (new-customer only) ---
    print("\n-- Cold-start quadrants (primary split; customer always new) --")
    train_cust_ids = set(train[EVENT_CUST_COL])
    train_prod_ids = set(train[EVENT_PROD_COL])
    quadrant = assign_cold_start_quadrant(val, train_cust_ids, train_prod_ids)
    print("  Quadrant counts:")
    print(quadrant.value_counts().to_string())
    slices_b = evaluate_slices(y_val, proba_b, quadrant)
    for q, m in slices_b.items():
        if m.get("suppressed"):
            print(f"    {q}: n={m['n']} (suppressed, {m['reason']})")
        else:
            print(f"    {q}: n={m['n']}  prevalence={m['prevalence']:.4f}  "
                  f"ROC-AUC={m['roc_auc']:.4f}  LogLoss={m['log_loss']:.4f}")
    results["cold_start_primary_split_lr_b"] = slices_b

    # --- Product-coverage diagnostic (LR-A and LR-B) ---
    print("\n-- Product-coverage diagnostic (LR-B) --")
    has_prod = has_product_node_mask(val)
    for label, mask in [("with_product_node", has_prod), ("without_product_node", ~has_prod)]:
        n = int(mask.sum())
        if n < 1000:
            print(f"    {label}: n={n} (too small, skipped)")
            continue
        m = compute_metrics(y_val[mask], proba_b[mask])
        print(f"    {label}: n={n:,}  prevalence={m['prevalence']:.4f}  "
              f"ROC-AUC={m['roc_auc']:.4f}  LogLoss={m['log_loss']:.4f}")
        results.setdefault("product_coverage_diagnostic_lr_b", {})[label] = m

    return results, lr_a, lr_b, feat_names_a, feat_names_b


def run_secondary_split_cold_start(df: pd.DataFrame) -> dict:
    """
    Diagnostic-only: retrain LR-B on the random stratified split's train
    fold purely to populate the full four-quadrant cold-start table
    (known customers/products can only exist under this split, since the
    primary split is customer-disjoint by construction). Not used for any
    headline metric.
    """
    print("\n" + "=" * 70)
    print("  SECONDARY SPLIT (random stratified) — diagnostic cold-start only")
    print("=" * 70)

    is_val = df["is_val_secondary"]
    train, val = _split(df, is_val)
    y_val = val[TARGET_COL]
    print(f"  Train: {len(train):,} rows   Val: {len(val):,} rows")

    lr_b, proba_b, _ = _fit_predict_lr(include_price=True, train=train, val=val)
    m_b = compute_metrics(y_val, proba_b)
    print(f"  LR-B on secondary split: ROC-AUC={m_b['roc_auc']:.4f}  "
          f"LogLoss={m_b['log_loss']:.4f}  (diagnostic only)")

    train_cust_ids = set(train[EVENT_CUST_COL])
    train_prod_ids = set(train[EVENT_PROD_COL])
    quadrant = assign_cold_start_quadrant(val, train_cust_ids, train_prod_ids)
    print("  Quadrant counts:")
    print(quadrant.value_counts().to_string())
    slices_b = evaluate_slices(y_val, proba_b, quadrant)
    for q, m in slices_b.items():
        if m.get("suppressed"):
            print(f"    {q}: n={m['n']} (suppressed, {m['reason']})")
        else:
            print(f"    {q}: n={m['n']}  prevalence={m['prevalence']:.4f}  "
                  f"ROC-AUC={m['roc_auc']:.4f}  LogLoss={m['log_loss']:.4f}")

    return {
        "n_train": len(train),
        "n_val": len(val),
        "lr_b_overall": m_b,
        "cold_start_quadrants_lr_b": slices_b,
    }


def main():
    print("=" * 70)
    print("  ReturnGuard — Stage 1 Baseline Training")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print(f"\n[ERROR] {INPUT_FILE} not found. Run scripts/build_dataset.py first.")
        sys.exit(1)

    df = pd.read_pickle(INPUT_FILE)
    print(f"\nLoaded joined dataset: {df.shape}")

    primary_results, lr_a, lr_b, feat_names_a, feat_names_b = run_primary_split(df)
    secondary_results = run_secondary_split_cold_start(df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_a, REPORTS_DIR / "lr_a_pipeline.joblib")
    joblib.dump(lr_b, REPORTS_DIR / "lr_b_pipeline.joblib")
    print(f"\nSaved fitted pipelines to {REPORTS_DIR}/lr_a_pipeline.joblib, lr_b_pipeline.joblib")

    report = {
        "overall_prevalence": float(df[TARGET_COL].mean()),
        "primary_split": primary_results,
        "secondary_split_diagnostic": secondary_results,
        "lr_a_feature_names": feat_names_a,
        "lr_b_feature_names": feat_names_b,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "stage1_metrics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print(f"\nSaved metrics report to {out_path}")
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
