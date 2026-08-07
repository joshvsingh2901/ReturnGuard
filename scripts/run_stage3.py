"""Run Stage 3 understanding, governance, and calibration analysis.

This script deliberately never loads the official ASOS test files. It uses
the training-derived development folds only. Primary-validation findings are
reported as developmental evidence because Stage 2 repeatedly inspected that
fold across ablations.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd

from ml.data.joins import has_product_node_mask
from ml.data.schema import EVENT_CUST_COL, EVENT_PROD_COL, TARGET_COL
from ml.data.splits import inner_early_stopping_split
from ml.evaluation.calibration import plot_reliability_curve, stage3_calibration_report
from ml.evaluation.metrics import compute_metrics
from ml.explainability.contracts import build_local_explanation, user_facing_explanation
from ml.explainability.grouping import FREQUENCY_FEATURE_NAMES
from ml.explainability.sampling import (
    DEFAULT_SHAP_SAMPLE_SIZE,
    FALLBACK_SHAP_SAMPLE_SIZE,
    PILOT_SHAP_SAMPLE_SIZE,
    representative_validation_sample,
)
from ml.explainability.shap_analysis import (
    a3_a4_score_delta,
    compute_tree_shap,
    plot_beeswarm,
    plot_dependence,
    summarize_shap,
    transformed_matrix,
)
from ml.features.frequency import attach_frequency_id_columns
from ml.governance.feature_manifest import build_feature_manifest, validate_a3_a4_manifests
from ml.governance.model_spec import build_model_freeze_spec
from ml.models.gbdt import DEFAULT_XGB_PARAMS, fit_gbdt


ROOT = Path(__file__).parent.parent
PROCESSED_FILE = ROOT / "data" / "processed" / "train_joined.pkl"
REPORTS_DIR = ROOT / "reports"
STAGE2_REPORT = REPORTS_DIR / "stage2_metrics.json"
A4_PIPELINE_FILE = REPORTS_DIR / "stage2_feature_pipeline.joblib"
A4_MODEL_FILE = REPORTS_DIR / "stage2_xgb_model.joblib"

MAX_PROJECTED_SHAP_SECONDS = 15 * 60
REPRODUCTION_AUC_TOLERANCE = 0.003


def _json_default(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: dict | list) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _code_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _score(feature_pipeline, model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = transformed_matrix(feature_pipeline, df)
    return model.predict_proba(X)[:, 1], X


def _load_stage2_report() -> dict:
    with open(STAGE2_REPORT) as handle:
        return json.load(handle)


def _fit_fixed_model(name: str, inner_train: pd.DataFrame, early_stop: pd.DataFrame, primary_val: pd.DataFrame, *, include_frequency: bool):
    """Fit one fixed development configuration; no selection or tuning occurs."""
    print(f"\n-- Re-fitting fixed {name} configuration --")
    started = time.perf_counter()
    result = fit_gbdt(
        inner_train,
        early_stop,
        primary_val,
        artifact_safe=True,
        include_derived=True,
        include_frequency=include_frequency,
    )
    metrics = compute_metrics(primary_val[TARGET_COL], result.proba_val)
    print(
        f"  ROC-AUC={metrics['roc_auc']:.6f} LogLoss={metrics['log_loss']:.6f} "
        f"features={len(result.feature_names)} best_iteration={result.best_iteration} "
        f"({time.perf_counter() - started:.1f}s)"
    )
    return result, metrics


def _slice_masks(primary_val: pd.DataFrame, inner_train: pd.DataFrame) -> dict[str, pd.Series]:
    product_covered = has_product_node_mask(primary_val)
    known_product = primary_val[EVENT_PROD_COL].isin(set(inner_train[EVENT_PROD_COL]))
    customer_present = primary_val["isMale"].notna()
    return {
        "overall": pd.Series(True, index=primary_val.index),
        "product_covered": product_covered,
        "product_missing": ~product_covered,
        "known_product_inner_train": known_product,
        "new_product_inner_train": ~known_product,
        "customer_node_present": customer_present,
        "customer_node_missing": ~customer_present,
    }


def _calibration_by_slice(y_true: pd.Series, probability: np.ndarray, masks: dict[str, pd.Series]) -> dict:
    result = {}
    y_arr = np.asarray(y_true)
    probability = np.asarray(probability)
    for label, mask in masks.items():
        selected = mask.to_numpy()
        result[label] = stage3_calibration_report(y_arr[selected], probability[selected])
    return result


def _mean_abs_grouped_by_mask(grouped_values: pd.DataFrame, mask: pd.Series) -> dict:
    selected = grouped_values.loc[mask.to_numpy()]
    return {
        column: float(selected[column].abs().mean())
        for column in selected.columns
    }


def _donor_inference_audit(feature_pipeline, model, primary_val: pd.DataFrame) -> dict:
    """Verify stable donor assignment across batches, order, and alternative seeds."""
    audit_df = primary_val.iloc[: min(2_000, len(primary_val))].copy()
    direct_probability, _ = _score(feature_pipeline, model, audit_df)

    midpoint = len(audit_df) // 2
    first_probability, _ = _score(feature_pipeline, model, audit_df.iloc[:midpoint])
    second_probability, _ = _score(feature_pipeline, model, audit_df.iloc[midpoint:])
    batched_probability = np.concatenate([first_probability, second_probability])

    reversed_df = audit_df.iloc[::-1]
    reversed_probability, _ = _score(feature_pipeline, model, reversed_df)
    reordered_probability = pd.Series(reversed_probability, index=reversed_df.index).loc[audit_df.index].to_numpy()

    missing_df = audit_df[audit_df["isMale"].isna()].iloc[:500]
    alternative_seed_delta = {}
    if len(missing_df):
        baseline_probability, _ = _score(feature_pipeline, model, missing_df)
        for seed in (7, 99):
            alternative = copy.deepcopy(feature_pipeline)
            alternative.named_steps["donor_impute"].random_state = seed
            alternative_probability, _ = _score(alternative, model, missing_df)
            delta = np.abs(alternative_probability - baseline_probability)
            alternative_seed_delta[str(seed)] = {
                "n_missing_customer_rows": int(len(missing_df)),
                "mean_abs_probability_delta": float(delta.mean()),
                "max_abs_probability_delta": float(delta.max()),
            }

    return {
        "n_rows_checked": int(len(audit_df)),
        "batch_size_max_abs_probability_delta": float(
            np.max(np.abs(direct_probability - batched_probability))
        ),
        "row_order_max_abs_probability_delta": float(
            np.max(np.abs(direct_probability - reordered_probability))
        ),
        "alternative_donor_seed_sensitivity": alternative_seed_delta,
        "method": "customer-ID keyed donor selection; seed sensitivity is an audit, not serving behavior",
    }


def _frequency_governance(a4_summary: dict, a4_values: np.ndarray, a4_feature_names: list[str], a3_probability: np.ndarray, a4_probability: np.ndarray, X_a4: np.ndarray) -> dict:
    indexes = [a4_feature_names.index(name) for name in sorted(FREQUENCY_FEATURE_NAMES)]
    individual = {}
    for index in indexes:
        name = a4_feature_names[index]
        individual[name] = float(np.abs(a4_values[:, index]).mean())

    combined = a4_values[:, indexes].sum(axis=1)
    grouped_ranking = a4_summary["grouped_ranking"]
    frequency_frame = grouped_ranking.loc[
        grouped_ranking["feature_family"] == "product_frequency_features"
    ]
    frequency_row = frequency_frame.iloc[0]
    stable_families = {
        "shipping_country",
        "product_type",
        "brand",
        "customer_binary_attributes",
        "birth_year_customer_demographic",
    }
    stable = grouped_ranking[grouped_ranking["feature_family"].isin(stable_families)]
    strongest_stable = float(stable["mean_abs_shap"].max()) if len(stable) else 0.0
    frequency_importance = float(frequency_row["mean_abs_shap"])
    frequency_share = float(frequency_row["attribution_share"])
    frequency_rank = int(frequency_frame.index[0]) + 1

    top_three = np.argpartition(np.abs(a4_values), -3, axis=1)[:, -3:]
    frequency_in_top_three_rate = float(
        np.mean(np.isin(top_three, indexes).any(axis=1))
    )

    if frequency_share > 0.5 or (
        strongest_stable > 0 and frequency_importance > 2 * strongest_stable
    ):
        influence = "DOMINANT"
    elif frequency_rank == 1:
        influence = "STRONGLY_INFLUENTIAL"
    else:
        influence = "AUXILIARY"

    variant_index = a4_feature_names.index("num_prod__variant_event_count")
    delta = a3_a4_score_delta(a3_probability, a4_probability)
    variant_values = pd.Series(X_a4[:, variant_index])
    if variant_values.nunique() > 1:
        buckets = pd.qcut(variant_values, q=5, duplicates="drop")
        delta_by_frequency = (
            pd.DataFrame({"bucket": buckets, "delta": delta})
            .groupby("bucket", observed=True)["delta"]
            .agg(["size", "mean", "median"])
            .reset_index()
        )
        delta_by_frequency["bucket"] = delta_by_frequency["bucket"].astype(str)
        delta_by_frequency = delta_by_frequency.to_dict(orient="records")
    else:
        delta_by_frequency = []

    return {
        "individual_mean_abs_shap": individual,
        "combined_per_row_mean_abs_shap": float(np.abs(combined).mean()),
        "combined_per_row_mean_signed_shap": float(combined.mean()),
        "grouped_mean_abs_shap": frequency_importance,
        "grouped_attribution_share": frequency_share,
        "grouped_rank": frequency_rank,
        "strongest_stable_semantic_group_mean_abs_shap": strongest_stable,
        "frequency_in_local_top_three_rate": frequency_in_top_three_rate,
        "influence_classification": influence,
        "delta_by_variant_frequency_quintile": delta_by_frequency,
        "governance_verdict": (
            "Frequency features remain temporally unverifiable without timestamps, "
            "regardless of attribution magnitude. A4 remains experimental."
        ),
    }


def _select_local_examples(sample: pd.DataFrame, a3_probability: np.ndarray, a4_probability: np.ndarray, inner_train: pd.DataFrame) -> list[dict]:
    """Choose deterministic, overlapping-but-deduplicated local scenarios."""
    work = sample.copy()
    work["a3_probability"] = a3_probability
    work["a4_probability"] = a4_probability
    work["a4_minus_a3"] = a4_probability - a3_probability
    work["product_covered"] = has_product_node_mask(work)
    work["known_product"] = work[EVENT_PROD_COL].isin(set(inner_train[EVENT_PROD_COL]))
    work["customer_imputed"] = work["isMale"].isna()
    work["a3_pred"] = (work["a3_probability"] >= 0.5).astype(int)

    scenarios = [
        ("lowest_a3_risk", work["a3_probability"].idxmin()),
        ("median_a3_risk", (work["a3_probability"] - work["a3_probability"].median()).abs().idxmin()),
        ("highest_a3_risk", work["a3_probability"].idxmax()),
        ("correct_return", work[(work[TARGET_COL] == 1) & (work["a3_pred"] == 1)]["a3_probability"].idxmax()),
        ("correct_keep", work[(work[TARGET_COL] == 0) & (work["a3_pred"] == 0)]["a3_probability"].idxmin()),
        ("false_positive", work[(work[TARGET_COL] == 0) & (work["a3_pred"] == 1)]["a3_probability"].idxmax()),
        ("false_negative", work[(work[TARGET_COL] == 1) & (work["a3_pred"] == 0)]["a3_probability"].idxmin()),
        ("product_missing", work[~work["product_covered"]]["a3_probability"].idxmax()),
        ("product_covered", work[work["product_covered"]]["a3_probability"].idxmax()),
        ("new_product", work[~work["known_product"]]["a3_probability"].idxmax()),
        ("known_product", work[work["known_product"]]["a3_probability"].idxmax()),
        ("customer_profile_imputed", work[work["customer_imputed"]]["a3_probability"].idxmax()),
        ("customer_profile_present", work[~work["customer_imputed"]]["a3_probability"].idxmax()),
        ("largest_a4_uplift", work["a4_minus_a3"].idxmax()),
        ("largest_a4_reduction", work["a4_minus_a3"].idxmin()),
    ]
    return [{"scenario": scenario, "row_index": int(index)} for scenario, index in scenarios]


def _local_explanations(
    examples: list[dict], sample: pd.DataFrame, a3_shap, a4_shap, a3_feature_names: list[str], a4_feature_names: list[str]
) -> list[dict]:
    positions = {int(index): position for position, index in enumerate(sample.index)}
    outputs = []
    for example in examples:
        position = positions[example["row_index"]]
        row = sample.iloc[position]
        context = {
            "customer_profile_imputed": bool(pd.isna(row["isMale"])),
            "product_profile_available": bool(has_product_node_mask(sample.iloc[[position]]).iloc[0]),
        }
        a3_internal = build_local_explanation(
            raw_probability=a3_shap.probability[position],
            raw_margin=a3_shap.raw_margin[position],
            base_value=a3_shap.base_value,
            shap_row=a3_shap.values[position],
            feature_names=a3_feature_names,
            row=row,
            model_version="returnguard-a3-stage3-model-of-record-candidate",
            **context,
        )
        a4_internal = build_local_explanation(
            raw_probability=a4_shap.probability[position],
            raw_margin=a4_shap.raw_margin[position],
            base_value=a4_shap.base_value,
            shap_row=a4_shap.values[position],
            feature_names=a4_feature_names,
            row=row,
            model_version="returnguard-a4-stage3-experimental-challenger",
            **context,
        )
        outputs.append(
            {
                "scenario": example["scenario"],
                "row_index": example["row_index"],
                "target": int(row[TARGET_COL]),
                "a3_internal": a3_internal,
                "a3_user_facing": user_facing_explanation(a3_internal),
                "a4_internal": a4_internal,
                "a4_user_facing": user_facing_explanation(a4_internal),
            }
        )
    return outputs


def _shap_for_sample(a3_result, a4_result, sample: pd.DataFrame, train_product_ids: set):
    """Pilot Tree SHAP then use a 10k (or justified 5k) shared sample."""
    pilot, pilot_manifest = representative_validation_sample(
        sample, train_product_ids, n=min(PILOT_SHAP_SAMPLE_SIZE, len(sample))
    )
    pilot_x = transformed_matrix(a3_result.feature_pipeline, pilot)
    started = time.perf_counter()
    compute_tree_shap(a3_result.model, pilot_x)
    pilot_seconds = time.perf_counter() - started
    projected = pilot_seconds * (DEFAULT_SHAP_SAMPLE_SIZE / len(pilot)) * 2
    target_n = (
        FALLBACK_SHAP_SAMPLE_SIZE
        if projected > MAX_PROJECTED_SHAP_SECONDS
        else DEFAULT_SHAP_SAMPLE_SIZE
    )
    selected, manifest = representative_validation_sample(
        sample, train_product_ids, n=min(target_n, len(sample))
    )
    return selected, manifest, {
        "pilot_n": len(pilot),
        "pilot_seconds_a3": pilot_seconds,
        "projected_seconds_a3_a4": projected,
        "selected_n": len(selected),
        "fallback_used": target_n == FALLBACK_SHAP_SAMPLE_SIZE,
        "pilot_manifest_hash": pilot_manifest.manifest_hash,
    }


def main() -> None:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Missing {PROCESSED_FILE}; run scripts/build_dataset.py first")
    if not STAGE2_REPORT.exists():
        raise FileNotFoundError(f"Missing {STAGE2_REPORT}; Stage 2 report is required")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("ReturnGuard Stage 3 — understanding, governance, calibration")
    print("Official ASOS test model performance is deliberately not evaluated.\n")

    stage2_report = _load_stage2_report()
    df = pd.read_pickle(PROCESSED_FILE)
    df = attach_frequency_id_columns(df)
    fold = inner_early_stopping_split(df)
    if not (fold == "primary_val").equals(df["is_val_primary"]):
        raise AssertionError("Stage 3 primary validation does not match Stage 1 primary validation")
    inner_train = df[fold == "inner_train"].copy()
    early_stop = df[fold == "early_stop"].copy()
    primary_val = df[fold == "primary_val"].copy()

    # The original A4 artifact was trained with the old sequence-based donor
    # implementation. Stage 3 fixes a real inference-order issue, so both A3
    # and A4 are re-fit at frozen settings for internally consistent analysis.
    a3_result, a3_metrics = _fit_fixed_model(
        "A3", inner_train, early_stop, primary_val, include_frequency=False
    )
    a4_result, a4_metrics = _fit_fixed_model(
        "A4", inner_train, early_stop, primary_val, include_frequency=True
    )

    archived_a3_auc = stage2_report["a3_derived_features"]["metrics"]["roc_auc"]
    archived_a4_auc = stage2_report["a4_frequency_features"]["metrics"]["roc_auc"]
    reproduction = {
        "a3_archived_roc_auc": archived_a3_auc,
        "a3_refit_roc_auc": a3_metrics["roc_auc"],
        "a3_abs_auc_delta": abs(a3_metrics["roc_auc"] - archived_a3_auc),
        "a3_within_expected_tolerance": abs(a3_metrics["roc_auc"] - archived_a3_auc) <= REPRODUCTION_AUC_TOLERANCE,
        "a4_archived_roc_auc": archived_a4_auc,
        "a4_refit_roc_auc": a4_metrics["roc_auc"],
        "a4_abs_auc_delta": abs(a4_metrics["roc_auc"] - archived_a4_auc),
        "donor_imputation_change": "A3/A4 refit after customer-ID keyed donor-imputation fix; no hyperparameters or features were tuned.",
    }

    a3_manifest = build_feature_manifest(a3_result.feature_names, "A3")
    a4_manifest = build_feature_manifest(a4_result.feature_names, "A4")
    validate_a3_a4_manifests(a3_manifest, a4_manifest)

    joblib.dump(a3_result.feature_pipeline, REPORTS_DIR / "stage3_a3_feature_pipeline.joblib")
    joblib.dump(a3_result.model, REPORTS_DIR / "stage3_a3_xgb_model.joblib")
    joblib.dump(a4_result.feature_pipeline, REPORTS_DIR / "stage3_a4_feature_pipeline.joblib")
    joblib.dump(a4_result.model, REPORTS_DIR / "stage3_a4_xgb_model.joblib")

    train_product_ids = set(inner_train[EVENT_PROD_COL])
    sample, sample_manifest, shap_compute_plan = _shap_for_sample(
        a3_result, a4_result, primary_val, train_product_ids
    )
    _write_json(REPORTS_DIR / "stage3_shap_sample_manifest.json", sample_manifest.to_dict())
    _write_json(REPORTS_DIR / "stage3_shap_compute_plan.json", shap_compute_plan)

    print(f"\n-- Tree SHAP on {len(sample):,} shared primary-validation rows --")
    a3_x = transformed_matrix(a3_result.feature_pipeline, sample)
    a4_x = transformed_matrix(a4_result.feature_pipeline, sample)
    started = time.perf_counter()
    a3_shap = compute_tree_shap(a3_result.model, a3_x)
    a3_seconds = time.perf_counter() - started
    started = time.perf_counter()
    a4_shap = compute_tree_shap(a4_result.model, a4_x)
    a4_seconds = time.perf_counter() - started
    print(f"  A3={a3_seconds:.1f}s A4={a4_seconds:.1f}s")

    a3_summary = summarize_shap(a3_shap.values, a3_result.feature_names)
    a4_summary = summarize_shap(a4_shap.values, a4_result.feature_names)
    a3_summary["encoded_ranking"].to_csv(REPORTS_DIR / "stage3_a3_encoded_shap_ranking.csv", index=False)
    a3_summary["grouped_ranking"].to_csv(REPORTS_DIR / "stage3_a3_grouped_shap_ranking.csv", index=False)
    a4_summary["encoded_ranking"].to_csv(REPORTS_DIR / "stage3_a4_encoded_shap_ranking.csv", index=False)
    a4_summary["grouped_ranking"].to_csv(REPORTS_DIR / "stage3_a4_grouped_shap_ranking.csv", index=False)
    plot_beeswarm(a3_shap.values, a3_x, a3_result.feature_names, REPORTS_DIR / "stage3_a3_beeswarm.png")
    plot_beeswarm(a4_shap.values, a4_x, a4_result.feature_names, REPORTS_DIR / "stage3_a4_beeswarm.png")
    for name, values, matrix, feature_names in [
        ("a3", a3_shap.values, a3_x, a3_result.feature_names),
        ("a4", a4_shap.values, a4_x, a4_result.feature_names),
    ]:
        for feature in ("num_prod__avgGbpPrice", "num_prod__price_rel_type_median"):
            plot_dependence(values, matrix, feature_names, feature, REPORTS_DIR / f"stage3_{name}_{feature.split('__', 1)[1]}_dependence.png")
    for feature in sorted(FREQUENCY_FEATURE_NAMES):
        plot_dependence(a4_shap.values, a4_x, a4_result.feature_names, feature, REPORTS_DIR / f"stage3_a4_{feature.split('__', 1)[1]}_dependence.png")

    masks = _slice_masks(primary_val, inner_train)
    a3_calibration = _calibration_by_slice(primary_val[TARGET_COL], a3_result.proba_val, masks)
    a4_calibration = _calibration_by_slice(primary_val[TARGET_COL], a4_result.proba_val, masks)
    lr_a = joblib.load(REPORTS_DIR / "lr_a_pipeline.joblib")
    lr_b = joblib.load(REPORTS_DIR / "lr_b_pipeline.joblib")
    lr_a_probability = lr_a.predict_proba(primary_val)[:, 1]
    lr_b_probability = lr_b.predict_proba(primary_val)[:, 1]
    lr_a_calibration = _calibration_by_slice(primary_val[TARGET_COL], lr_a_probability, masks)
    lr_b_calibration = _calibration_by_slice(primary_val[TARGET_COL], lr_b_probability, masks)
    for label, report in {
        "lr_a": lr_a_calibration["overall"],
        "lr_b": lr_b_calibration["overall"],
        "a3": a3_calibration["overall"],
        "a4": a4_calibration["overall"],
    }.items():
        plot_reliability_curve(
            report["reliability_curve"],
            f"{label.upper()} Stage 3 Calibration (development validation)",
            REPORTS_DIR / f"stage3_calibration_{label}.png",
        )

    frequency = _frequency_governance(
        a4_summary,
        a4_shap.values,
        a4_result.feature_names,
        a3_shap.probability,
        a4_shap.probability,
        a4_x,
    )
    donor_a3 = _donor_inference_audit(a3_result.feature_pipeline, a3_result.model, primary_val)
    donor_a4 = _donor_inference_audit(a4_result.feature_pipeline, a4_result.model, primary_val)

    sample_masks = _slice_masks(sample, inner_train)
    product_missingness_audit = {
        "a3": {
            "performance": {
                label: compute_metrics(primary_val.loc[mask, TARGET_COL], a3_result.proba_val[mask.to_numpy()])
                for label, mask in masks.items()
                if label in {"product_covered", "product_missing"}
            },
            "grouped_mean_abs_shap_by_sample_slice": {
                label: _mean_abs_grouped_by_mask(a3_summary["grouped_values"], sample_masks[label])
                for label in ("product_covered", "product_missing")
            },
        },
        "a4": {
            "performance": {
                label: compute_metrics(primary_val.loc[mask, TARGET_COL], a4_result.proba_val[mask.to_numpy()])
                for label, mask in masks.items()
                if label in {"product_covered", "product_missing"}
            },
            "grouped_mean_abs_shap_by_sample_slice": {
                label: _mean_abs_grouped_by_mask(a4_summary["grouped_values"], sample_masks[label])
                for label in ("product_covered", "product_missing")
            },
        },
    }

    customer_artifact_audit = {
        "a3": {
            "donor_inference_invariance": donor_a3,
            "grouped_mean_abs_shap_by_customer_node": {
                label: _mean_abs_grouped_by_mask(a3_summary["grouped_values"], sample_masks[label])
                for label in ("customer_node_present", "customer_node_missing")
            },
        },
        "a4": {
            "donor_inference_invariance": donor_a4,
            "grouped_mean_abs_shap_by_customer_node": {
                label: _mean_abs_grouped_by_mask(a4_summary["grouped_values"], sample_masks[label])
                for label in ("customer_node_present", "customer_node_missing")
            },
        },
    }

    price_feature_names_a3 = [
        name for name in a3_result.feature_names if name.startswith("num_prod__")
        and name not in FREQUENCY_FEATURE_NAMES
    ]
    price_feature_names_a4 = [
        name for name in a4_result.feature_names if name.startswith("num_prod__")
        and name not in FREQUENCY_FEATURE_NAMES
    ]
    price_audit = {
        "a3_combined_raw_and_derived_mean_abs_shap": float(
            np.abs(a3_shap.values[:, [a3_result.feature_names.index(name) for name in price_feature_names_a3]].sum(axis=1)).mean()
        ),
        "a4_combined_raw_and_derived_mean_abs_shap": float(
            np.abs(a4_shap.values[:, [a4_result.feature_names.index(name) for name in price_feature_names_a4]].sum(axis=1)).mean()
        ),
        "provenance": "SUSPICIOUS_GLOBAL_PRICE_DISCOUNT",
    }

    local_examples = _select_local_examples(sample, a3_shap.probability, a4_shap.probability, inner_train)
    local_explanations = _local_explanations(
        local_examples,
        sample,
        a3_shap,
        a4_shap,
        a3_result.feature_names,
        a4_result.feature_names,
    )
    _write_json(REPORTS_DIR / "stage3_local_explanations.json", local_explanations)

    freeze_spec = build_model_freeze_spec(
        a3_manifest,
        a3_result.params,
        a3_result.best_iteration,
        _code_version(),
    )
    _write_json(REPORTS_DIR / "stage3_a3_freeze_spec.json", freeze_spec)

    governance_verdict = {
        "model_of_record": "A3",
        "model_of_record_status": "MODEL_OF_RECORD_CONSERVATIVE_FREEZE_CANDIDATE",
        "experimental_challenger": "A4",
        "experimental_challenger_status": "EXPERIMENTAL_CHALLENGER_TEMPORALLY_UNVERIFIABLE_FREQUENCY_FEATURES",
        "decision_rationale": [
            "A3 avoids product-frequency encodings while retaining the fixed donor-imputation defense.",
            "A4 may improve development discrimination, but target-free frequency counts remain temporally unverifiable without timestamps.",
            "No calibrator is fitted because scores are conditional on a returner-enriched research population.",
            "Official test model performance remains uncomputed pending frozen-model protocol completion.",
        ],
        "frequency_governance": frequency,
        "calibration_policy": "NO_CALIBRATOR",
        "test_evaluation_policy": "Run one predeclared held-out performance evaluation only after the frozen model spec and documentation are committed; do not use results for model selection.",
    }

    summary = {
        "stage": "Stage 3",
        "official_test_model_performance_evaluated": False,
        "evaluation_history": {
            "test_statistics_inspected": True,
            "validation_driven_development_occurred": True,
            "stage2_selection_fold": "inner early-stop",
            "primary_validation_pristine": False,
        },
        "reproduction": reproduction,
        "feature_manifests": {"a3": a3_manifest, "a4": a4_manifest},
        "shap_sample": sample_manifest.to_dict(),
        "shap_compute": {
            **shap_compute_plan,
            "a3_seconds": a3_seconds,
            "a4_seconds": a4_seconds,
            "a3_additivity_max_abs_error": a3_shap.additivity_max_abs_error,
            "a4_additivity_max_abs_error": a4_shap.additivity_max_abs_error,
            "a3_probability_max_abs_error": a3_shap.probability_max_abs_error,
            "a4_probability_max_abs_error": a4_shap.probability_max_abs_error,
        },
        "models": {
            "lr_a": {"calibration": lr_a_calibration},
            "lr_b": {"calibration": lr_b_calibration},
            "a3": {
                "metrics": a3_metrics,
                "best_iteration": a3_result.best_iteration,
                "feature_count": len(a3_result.feature_names),
                "encoded_shap_ranking": a3_summary["encoded_ranking"].to_dict(orient="records"),
                "grouped_shap_ranking": a3_summary["grouped_ranking"].to_dict(orient="records"),
                "calibration": a3_calibration,
            },
            "a4": {
                "metrics": a4_metrics,
                "best_iteration": a4_result.best_iteration,
                "feature_count": len(a4_result.feature_names),
                "encoded_shap_ranking": a4_summary["encoded_ranking"].to_dict(orient="records"),
                "grouped_shap_ranking": a4_summary["grouped_ranking"].to_dict(orient="records"),
                "calibration": a4_calibration,
            },
        },
        "frequency_feature_governance": frequency,
        "customer_artifact_audit": customer_artifact_audit,
        "product_missingness_audit": product_missingness_audit,
        "price_discount_audit": price_audit,
        "raw_identifier_audit": {
            "a3_raw_hash_identifier_features": [name for name in a3_result.feature_names if "hash(" in name],
            "a4_raw_hash_identifier_features": [name for name in a4_result.feature_names if "hash(" in name],
        },
        "local_explanations": {
            "count": len(local_explanations),
            "file": "reports/stage3_local_explanations.json",
        },
        "governance_verdict": governance_verdict,
        "freeze_spec": freeze_spec,
    }
    _write_json(REPORTS_DIR / "stage3_summary.json", summary)
    _write_json(REPORTS_DIR / "stage3_governance_verdict.json", governance_verdict)
    print("\nStage 3 report written to reports/stage3_summary.json")


if __name__ == "__main__":
    main()
