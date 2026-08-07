"""
Stage 2 ablation ladder: XGBoost on the Stage 1 harness.

Rungs, all evaluated on the IDENTICAL primary (customer-grouped) validation
fold used throughout Stage 1:

  A0  Stage 1 LR-B — read from reports/stage1_metrics.json, NOT recomputed.
  A1  XGBoost, Stage 1-style median/mode customer imputation.
      DIAGNOSTIC ONLY — the customer missing-node artifact is exploitable
      under this preprocessing (see the leakage probe below); never a
      headline number.
  A2  XGBoost, donor-imputed customer features. The honest tree baseline.
  A3  A2 + derived price features.
  A4  A3 + product-side frequency features.
  A5  Tuned re-fit of whichever of A2-A4 scores best on the EARLY-STOP
      fold (never primary validation) — a small fixed grid, no Optuna.

Fold discipline: every rung fits on inner_train, early-stops on
early_stop, and is evaluated ONCE on primary_val. Rung/hyperparameter
selection (for A5) uses early_stop only. Primary validation never
influences any decision — only final reporting.

Never loads event_table_testing.p or any *_testing.p file.

Run:
    python scripts/build_dataset.py   # once, if not already done
    python scripts/train_stage2.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.data.joins import has_product_node_mask
from ml.data.schema import EVENT_CUST_COL, EVENT_PROD_COL, TARGET_COL
from ml.data.splits import inner_early_stopping_split, random_stratified_split
from ml.evaluation.calibration import (
    expected_calibration_error,
    plot_reliability_curve,
    reliability_curve,
)
from ml.evaluation.metrics import compute_metrics, compute_skill_scores
from ml.evaluation.probe import leakage_probe_auc
from ml.evaluation.slices import assign_cold_start_quadrant, evaluate_slices
from ml.features.frequency import attach_frequency_id_columns
from ml.models.gbdt import DEFAULT_XGB_PARAMS, check_early_stopping_triggered, fit_gbdt

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
INPUT_FILE = PROCESSED_DIR / "train_joined.pkl"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
STAGE1_METRICS_FILE = REPORTS_DIR / "stage1_metrics.json"

PROBE_TARGET_AUC = 0.55
TUNING_GRID = [
    {"max_depth": d, "learning_rate": lr}
    for d in (4, 6, 8)
    for lr in (0.05, 0.1)
]


def _predict_with_result(result, df: pd.DataFrame) -> np.ndarray:
    """Reuse an already-fitted rung's pipeline + model to score a
    different fold (e.g. early_stop, for rung-selection purposes) without
    refitting anything."""
    X = result.feature_pipeline.transform(df).astype(np.float32)
    return result.model.predict_proba(X)[:, 1]


def load_stage1_a0() -> dict:
    if not STAGE1_METRICS_FILE.exists():
        print(f"[ERROR] {STAGE1_METRICS_FILE} not found. Run scripts/train_baseline.py first.")
        sys.exit(1)
    with open(STAGE1_METRICS_FILE) as f:
        stage1 = json.load(f)
    return stage1["primary_split"]["lr_b"]


def run_leakage_probe(inner_train: pd.DataFrame, early_stop: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("  LEAKAGE PROBE — can a tree detect has_customer_node?")
    print("=" * 70)

    contaminated = leakage_probe_auc(inner_train, early_stop, artifact_safe=False)
    print(f"  Median/mode (contaminated): probe_auc={contaminated['probe_auc']:.4f}")

    safe = leakage_probe_auc(inner_train, early_stop, artifact_safe=True)
    print(f"  Donor imputation (safe):    probe_auc={safe['probe_auc']:.4f}")

    if safe["probe_auc"] <= PROBE_TARGET_AUC:
        print(f"  -> Target met (<= {PROBE_TARGET_AUC}).")
    else:
        print(f"  -> ABOVE target ({PROBE_TARGET_AUC}). Documented honestly, not silently ignored.")

    return {"contaminated": contaminated, "artifact_safe": safe}


def run_rung(name, inner_train, early_stop, primary_val, **kwargs):
    print(f"\n-- {name} --")
    t0 = time.time()
    result = fit_gbdt(inner_train, early_stop, primary_val, **kwargs)
    check_early_stopping_triggered(result)
    fit_time = time.time() - t0

    m = compute_metrics(primary_val[TARGET_COL], result.proba_val)
    early_stop_proba = _predict_with_result(result, early_stop)
    early_stop_auc = roc_auc_score(early_stop[TARGET_COL], early_stop_proba)

    print(f"  fit_time={fit_time:.1f}s  best_iter={result.best_iteration}  "
          f"n_features={len(result.feature_names)}")
    print(f"  [selection metric] early_stop ROC-AUC={early_stop_auc:.4f}")
    print(f"  [reporting]        primary_val ROC-AUC={m['roc_auc']:.4f}  "
          f"LogLoss={m['log_loss']:.4f}  Brier={m['brier']:.4f}")

    return {
        "result": result,
        "metrics": m,
        "early_stop_auc": float(early_stop_auc),
        "fit_time_seconds": fit_time,
        "n_features": len(result.feature_names),
        "best_iteration": result.best_iteration,
    }


def main():
    print("=" * 70)
    print("  ReturnGuard — Stage 2 Ablation Ladder (XGBoost)")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print(f"\n[ERROR] {INPUT_FILE} not found. Run scripts/build_dataset.py first.")
        sys.exit(1)

    df = pd.read_pickle(INPUT_FILE)
    print(f"\nLoaded joined dataset: {df.shape}")
    df = attach_frequency_id_columns(df)
    print(f"Attached frequency ID columns: {df.shape}")

    fold = inner_early_stopping_split(df)
    assert (fold.values == "primary_val").tolist() == df["is_val_primary"].tolist(), (
        "Stage 2's primary_val fold does not match Stage 1's is_val_primary "
        "column — refusing to continue."
    )
    print("Verified: Stage 2 primary_val fold is byte-identical to Stage 1's.")

    inner_train = df[fold == "inner_train"].copy()
    early_stop = df[fold == "early_stop"].copy()
    primary_val = df[fold == "primary_val"].copy()
    print(f"inner_train={len(inner_train):,}  early_stop={len(early_stop):,}  "
          f"primary_val={len(primary_val):,}")

    # --- A0: read Stage 1 LR-B, never recomputed ---
    a0 = load_stage1_a0()
    print(f"\nA0 (Stage 1 LR-B, read from disk): ROC-AUC={a0['roc_auc']:.4f}  "
          f"LogLoss={a0['log_loss']:.4f}")

    # --- Leakage probe ---
    probe = run_leakage_probe(inner_train, early_stop)

    # --- A1: contaminated diagnostic ---
    a1 = run_rung("A1 (DIAGNOSTIC: median/mode imputation)",
                   inner_train, early_stop, primary_val,
                   artifact_safe=False, include_derived=False, include_frequency=False)

    # --- A2: honest baseline ---
    a2 = run_rung("A2 (honest: donor imputation)",
                   inner_train, early_stop, primary_val,
                   artifact_safe=True, include_derived=False, include_frequency=False)

    # --- A3: + derived features ---
    a3 = run_rung("A3 (A2 + derived price features)",
                   inner_train, early_stop, primary_val,
                   artifact_safe=True, include_derived=True, include_frequency=False)

    # --- A4: + frequency features ---
    a4 = run_rung("A4 (A3 + product frequency features)",
                   inner_train, early_stop, primary_val,
                   artifact_safe=True, include_derived=True, include_frequency=True)

    # --- Select best honest rung by EARLY-STOP AUC only ---
    honest_rungs = {"A2": a2, "A3": a3, "A4": a4}
    best_name = max(honest_rungs, key=lambda k: honest_rungs[k]["early_stop_auc"])
    best_rung = honest_rungs[best_name]
    print(f"\nBest honest rung by early-stop AUC: {best_name} "
          f"(early_stop_auc={best_rung['early_stop_auc']:.4f})")

    best_kwargs = {
        "A2": dict(artifact_safe=True, include_derived=False, include_frequency=False),
        "A3": dict(artifact_safe=True, include_derived=True, include_frequency=False),
        "A4": dict(artifact_safe=True, include_derived=True, include_frequency=True),
    }[best_name]

    # --- A5: small tuning grid on the winning feature configuration ---
    print("\n" + "=" * 70)
    print(f"  A5 — Tuning {best_name}'s feature configuration "
          f"({len(TUNING_GRID)} configs, selected by early_stop AUC)")
    print("=" * 70)

    tuning_results = []
    for cfg in TUNING_GRID:
        params = dict(DEFAULT_XGB_PARAMS)
        params.update(cfg)
        # Shallower trees / lower learning rates converge more slowly and
        # can need more boosting rounds to trigger early stopping than the
        # default rungs did (observed: max_depth=4, lr=0.05 exceeded the
        # default 2000-round cap in the first full run). Raise the cap
        # for the tuning grid specifically rather than for every rung.
        params["n_estimators"] = 6000
        t0 = time.time()
        result = fit_gbdt(inner_train, early_stop, primary_val,
                           xgb_params=params, **best_kwargs)
        check_early_stopping_triggered(result)
        early_stop_proba = _predict_with_result(result, early_stop)
        es_auc = roc_auc_score(early_stop[TARGET_COL], early_stop_proba)
        print(f"  max_depth={cfg['max_depth']} lr={cfg['learning_rate']}: "
              f"early_stop_auc={es_auc:.4f}  ({time.time()-t0:.1f}s)")
        tuning_results.append({"config": cfg, "early_stop_auc": float(es_auc), "result": result})

    best_tuned = max(tuning_results, key=lambda r: r["early_stop_auc"])
    a5_metrics = compute_metrics(primary_val[TARGET_COL], best_tuned["result"].proba_val)
    print(f"\nBest tuned config: {best_tuned['config']}  "
          f"primary_val ROC-AUC={a5_metrics['roc_auc']:.4f}")

    default_es_auc = best_rung["early_stop_auc"]
    tuning_gain = best_tuned["early_stop_auc"] - default_es_auc
    print(f"Tuning gain over default hyperparameters (early_stop AUC): {tuning_gain:+.4f}")
    if abs(tuning_gain) < 0.003:
        print("  -> Within noise (< 0.003); not narrated as a meaningful win.")

    # --- Final model: whichever of {best default rung, best tuned} wins on early_stop ---
    if best_tuned["early_stop_auc"] > best_rung["early_stop_auc"] + 0.003:
        final_name = f"A5 (tuned {best_name})"
        final_result = best_tuned["result"]
        final_metrics = a5_metrics
    else:
        final_name = best_name
        final_result = best_rung["result"]
        final_metrics = best_rung["metrics"]
    print(f"\nFinal chosen model: {final_name}")

    # --- Calibration for the final model ---
    proba_final = final_result.proba_val
    curve = reliability_curve(primary_val[TARGET_COL], proba_final)
    ece = expected_calibration_error(primary_val[TARGET_COL], proba_final)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_reliability_curve(curve, f"{final_name} Calibration",
                            REPORTS_DIR / "calibration_stage2_best.png")
    print(f"Final model ECE: {ece:.4f}")

    skill = compute_skill_scores(final_metrics, a0)  # vs Stage 1 LR-B, for context
    print(f"Skill vs Stage 1 LR-B (log_loss): {skill['log_loss_skill']:+.4f}")

    # --- Cold-start slices (primary split; customer always new) ---
    print("\n-- Cold-start quadrants (primary split; final model) --")
    train_cust_ids = set(inner_train[EVENT_CUST_COL]) | set(early_stop[EVENT_CUST_COL])
    train_prod_ids = set(inner_train[EVENT_PROD_COL]) | set(early_stop[EVENT_PROD_COL])
    quadrant = assign_cold_start_quadrant(primary_val, train_cust_ids, train_prod_ids)
    cold_start_primary = evaluate_slices(primary_val[TARGET_COL], proba_final, quadrant)
    for q, m in cold_start_primary.items():
        if m.get("suppressed"):
            print(f"    {q}: n={m['n']} (suppressed)")
        else:
            print(f"    {q}: n={m['n']}  ROC-AUC={m['roc_auc']:.4f}")

    # --- Product-coverage diagnostic ---
    has_prod = has_product_node_mask(primary_val)
    product_coverage = {}
    for label, mask in [("with_product_node", has_prod), ("without_product_node", ~has_prod)]:
        m = compute_metrics(primary_val.loc[mask, TARGET_COL], proba_final[mask.to_numpy()])
        product_coverage[label] = m
        print(f"    {label}: n={m['n']:,}  ROC-AUC={m['roc_auc']:.4f}")

    # --- Secondary split diagnostic (full 4-quadrant cold start) ---
    print("\n" + "=" * 70)
    print("  SECONDARY SPLIT — diagnostic cold-start only (final model config)")
    print("=" * 70)
    is_val_secondary = df["is_val_secondary"]
    secondary_train_full = df[~is_val_secondary].copy()
    secondary_val = df[is_val_secondary].copy()
    is_es2 = random_stratified_split(secondary_train_full, val_fraction=0.15, random_state=123)
    secondary_inner_train = secondary_train_full[~is_es2].copy()
    secondary_early_stop = secondary_train_full[is_es2].copy()
    print(f"  secondary_inner_train={len(secondary_inner_train):,}  "
          f"secondary_early_stop={len(secondary_early_stop):,}  "
          f"secondary_val={len(secondary_val):,}")

    secondary_result = fit_gbdt(secondary_inner_train, secondary_early_stop, secondary_val,
                                 **best_kwargs)
    check_early_stopping_triggered(secondary_result)
    secondary_metrics = compute_metrics(secondary_val[TARGET_COL], secondary_result.proba_val)
    print(f"  Secondary split ROC-AUC={secondary_metrics['roc_auc']:.4f}  (diagnostic only)")

    sec_train_cust = set(secondary_inner_train[EVENT_CUST_COL]) | set(secondary_early_stop[EVENT_CUST_COL])
    sec_train_prod = set(secondary_inner_train[EVENT_PROD_COL]) | set(secondary_early_stop[EVENT_PROD_COL])
    sec_quadrant = assign_cold_start_quadrant(secondary_val, sec_train_cust, sec_train_prod)
    cold_start_secondary = evaluate_slices(secondary_val[TARGET_COL], secondary_result.proba_val, sec_quadrant)
    for q, m in cold_start_secondary.items():
        if m.get("suppressed"):
            print(f"    {q}: n={m['n']} (suppressed)")
        else:
            print(f"    {q}: n={m['n']}  ROC-AUC={m['roc_auc']:.4f}")

    # --- Save everything ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_result.feature_pipeline, REPORTS_DIR / "stage2_feature_pipeline.joblib")
    joblib.dump(final_result.model, REPORTS_DIR / "stage2_xgb_model.joblib")

    def _strip(rung_dict):
        return {k: v for k, v in rung_dict.items() if k != "result"}

    report = {
        "a0_stage1_lr_b": a0,
        "leakage_probe": probe,
        "a1_diagnostic": _strip(a1),
        "a2_honest_baseline": _strip(a2),
        "a3_derived_features": _strip(a3),
        "a4_frequency_features": _strip(a4),
        "best_honest_rung": best_name,
        "a5_tuning": {
            "grid": TUNING_GRID,
            "results": [
                {"config": r["config"], "early_stop_auc": r["early_stop_auc"]}
                for r in tuning_results
            ],
            "best_config": best_tuned["config"],
            "best_primary_val_metrics": a5_metrics,
            "tuning_gain_early_stop_auc": tuning_gain,
        },
        "final_model": final_name,
        "final_metrics": final_metrics,
        "final_calibration_ece": ece,
        "final_skill_vs_lr_b": skill,
        "cold_start_primary_split": cold_start_primary,
        "product_coverage_diagnostic": product_coverage,
        "secondary_split_diagnostic": {
            "n_inner_train": len(secondary_inner_train),
            "n_early_stop": len(secondary_early_stop),
            "n_val": len(secondary_val),
            "metrics": secondary_metrics,
            "cold_start_quadrants": cold_start_secondary,
        },
        "final_feature_names": final_result.feature_names,
    }

    out_path = REPORTS_DIR / "stage2_metrics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2,
                   default=lambda o: float(o) if isinstance(o, np.floating) else
                   (int(o) if isinstance(o, np.integer) else str(o)))
    print(f"\nSaved metrics report to {out_path}")
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
