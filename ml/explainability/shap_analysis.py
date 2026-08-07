"""Tree SHAP computation, verification, and compact plot helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from xgboost import DMatrix

from ml.explainability.grouping import aggregate_grouped_shap, encoded_shap_ranking


@dataclass
class ShapResult:
    values: np.ndarray
    base_value: float
    raw_margin: np.ndarray
    probability: np.ndarray
    additivity_max_abs_error: float
    probability_max_abs_error: float


def transformed_matrix(feature_pipeline, df: pd.DataFrame) -> np.ndarray:
    """Transform a frame exactly as the fitted model expects."""
    return np.asarray(feature_pipeline.transform(df), dtype=np.float32)


def model_raw_margin(model, X: np.ndarray) -> np.ndarray:
    """Return margins from exactly the tree range used by ``predict_proba``.

    ``XGBClassifier.predict_proba`` automatically stops at ``best_iteration``
    after early stopping.  A direct ``Booster.predict`` does not, so use a
    sliced booster here (and in Tree SHAP) to keep the explanation target and
    the production scoring target identical.
    """
    return _prediction_booster(model).predict(DMatrix(X), output_margin=True)


def _prediction_booster(model):
    """Return the fitted scoring trees, respecting early stopping when set."""
    booster = model.get_booster()
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return booster
    return booster[: best_iteration + 1]


def compute_tree_shap(model, X: np.ndarray) -> ShapResult:
    """Compute raw-margin Tree SHAP and verify model-space additivity."""
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - dependency error is user-facing
        raise ImportError("Stage 3 requires the optional 'shap' dependency") from exc

    # Explain the same best-iteration tree range that the sklearn wrapper
    # scores.  Passing the unsliced booster would silently include all trees.
    explainer = shap.TreeExplainer(_prediction_booster(model), model_output="raw")
    values = np.asarray(explainer.shap_values(X), dtype=float)
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    raw_margin = np.asarray(model_raw_margin(model, X), dtype=float)
    probability = model.predict_proba(X)[:, 1]

    reconstructed_margin = base_value + values.sum(axis=1)
    additivity_error = float(np.max(np.abs(reconstructed_margin - raw_margin)))
    probability_error = float(np.max(np.abs(expit(raw_margin) - probability)))

    return ShapResult(
        values=values,
        base_value=base_value,
        raw_margin=raw_margin,
        probability=np.asarray(probability, dtype=float),
        additivity_max_abs_error=additivity_error,
        probability_max_abs_error=probability_error,
    )


def summarize_shap(values: np.ndarray, feature_names: list[str]) -> dict:
    """Return encoded and grouped rankings with no raw-row SHAP persistence."""
    grouped_values, grouped_ranking = aggregate_grouped_shap(values, feature_names)
    return {
        "encoded_ranking": encoded_shap_ranking(values, feature_names),
        "grouped_values": grouped_values,
        "grouped_ranking": grouped_ranking,
    }


def plot_beeswarm(values: np.ndarray, X: np.ndarray, feature_names: list[str], save_path, max_display: int = 20) -> None:
    """Save a compact SHAP beeswarm without exposing it to interactive state."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap.summary_plot(values, X, feature_names=feature_names, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def plot_dependence(values: np.ndarray, X: np.ndarray, feature_names: list[str], feature_name: str, save_path) -> None:
    """Save one dependence plot for an approved transformed feature."""
    if feature_name not in feature_names:
        raise ValueError(f"Cannot plot missing feature {feature_name!r}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap.dependence_plot(
        feature_names.index(feature_name),
        values,
        X,
        feature_names=feature_names,
        interaction_index=None,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def a3_a4_score_delta(a3_probability: np.ndarray, a4_probability: np.ndarray) -> pd.Series:
    """Return A4 minus A3 score change on the shared explanation sample."""
    a3_probability = np.asarray(a3_probability, dtype=float)
    a4_probability = np.asarray(a4_probability, dtype=float)
    if a3_probability.shape != a4_probability.shape:
        raise ValueError("A3 and A4 probability arrays must align")
    return pd.Series(a4_probability - a3_probability, name="a4_minus_a3_probability")
