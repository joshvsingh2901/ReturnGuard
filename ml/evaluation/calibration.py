"""
Calibration measurement for Stage 1.

Measurement only — no calibrator (Platt/isotonic) is fit here. That is
explicitly deferred; see docs/problem-definition.md on why the biased
returner-only sample means any calibrator fit now would not transfer to
the general population.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.evaluation.metrics import compute_metrics


def reliability_curve(y_true, y_proba, n_bins: int = 10) -> dict:
    """
    Quantile-binned reliability curve: for each bin, the mean predicted
    probability vs. the mean observed outcome, plus the bin's row count.

    Bin membership, predicted mean, and observed mean are all derived
    from a single pd.qcut assignment so counts always sum to n exactly.
    When y_proba has fewer distinct values than n_bins (common for LR on
    mostly-categorical features), pandas drops duplicate quantile edges
    and returns fewer, wider bins rather than misaligned ones.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if np.unique(y_proba).size <= 1:
        # qcut cannot form any bins from a constant array (e.g. the
        # prevalence baseline, which predicts one fixed probability for
        # every row) — it silently returns all-NaN bin membership. Treat
        # the whole array as a single bin instead.
        return {
            "n_bins": 1,
            "mean_predicted": [float(y_proba.mean())],
            "mean_observed": [float(y_true.mean())],
            "bin_counts": [int(len(y_true))],
        }

    bin_assignment = pd.qcut(y_proba, q=n_bins, duplicates="drop")
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba, "bin": bin_assignment})
    grouped = df.groupby("bin", observed=True)

    mean_predicted = grouped["y_proba"].mean().to_numpy()
    mean_observed = grouped["y_true"].mean().to_numpy()
    counts = grouped.size().to_numpy()

    return {
        "n_bins": len(mean_predicted),
        "mean_predicted": mean_predicted.tolist(),
        "mean_observed": mean_observed.tolist(),
        "bin_counts": counts.tolist(),
    }


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """
    ECE: count-weighted mean absolute gap between predicted and observed
    probability across quantile bins.
    """
    curve = reliability_curve(y_true, y_proba, n_bins=n_bins)
    predicted = np.array(curve["mean_predicted"])
    observed = np.array(curve["mean_observed"])
    counts = np.array(curve["bin_counts"], dtype=float)
    if counts.sum() == 0:
        return 0.0
    weights = counts / counts.sum()
    return float(np.sum(weights * np.abs(predicted - observed)))


def calibration_intercept_slope(y_true, y_proba) -> dict:
    """Estimate descriptive calibration-in-the-large and slope.

    This diagnostic regression is never retained or applied to scores; it
    is explicitly not a fitted probability calibrator. Values are None for
    degenerate slices where these quantities are not identifiable.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"calibration_intercept": None, "calibration_slope": None}

    clipped = np.clip(y_proba, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    if np.unique(logit).size < 2:
        return {"calibration_intercept": None, "calibration_slope": None}

    diagnostic = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    diagnostic.fit(logit.reshape(-1, 1), y_true)
    return {
        "calibration_intercept": float(diagnostic.intercept_[0]),
        "calibration_slope": float(diagnostic.coef_[0, 0]),
    }


def fixed_risk_bands(y_true, y_proba, width: float = 0.1) -> list[dict]:
    """Return fixed-width probability-band diagnostics, including empty bins."""
    if width <= 0 or not np.isclose(1 / width, round(1 / width)):
        raise ValueError("width must divide the unit interval exactly")

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    edges = np.arange(0, 1 + width / 2, width)
    clipped = np.clip(y_proba, 0, 1)
    # digitize gives bin indexes 0..n-1 and includes score == 1 in final bin.
    assignments = np.minimum(np.digitize(clipped, edges[1:], right=False), len(edges) - 2)
    bands = []
    for i in range(len(edges) - 1):
        mask = assignments == i
        n = int(mask.sum())
        bands.append(
            {
                "lower": float(edges[i]),
                "upper": float(edges[i + 1]),
                "n": n,
                "mean_prediction": float(y_proba[mask].mean()) if n else None,
                "observed_return_rate": float(y_true[mask].mean()) if n else None,
            }
        )
    return bands


def stage3_calibration_report(y_true, y_proba, n_bins: int = 10) -> dict:
    """Complete measurement-only calibration report used by Stage 3."""
    metrics = compute_metrics(y_true, y_proba)
    return {
        "n": metrics["n"],
        "prevalence": metrics["prevalence"],
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier"],
        "ece": expected_calibration_error(y_true, y_proba, n_bins=n_bins),
        "reliability_curve": reliability_curve(y_true, y_proba, n_bins=n_bins),
        "fixed_risk_bands": fixed_risk_bands(y_true, y_proba),
        **calibration_intercept_slope(y_true, y_proba),
    }


def plot_reliability_curve(curve: dict, title: str, save_path: str) -> None:
    """Save a reliability diagram (predicted vs. observed) to save_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(curve["mean_predicted"], curve["mean_observed"], marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
