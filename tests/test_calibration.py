"""
Tests for ml.evaluation.calibration.

Includes a regression test for a bug found during Stage 1 development:
reliability_curve's bin counts previously did not sum to n when quantile
edges collided (common when a Logistic Regression on mostly-categorical
features produces few distinct probability values), and silently
returned zero bins on a fully constant probability array (as produced by
the prevalence baseline).
"""

import numpy as np
import pytest

from ml.evaluation.calibration import expected_calibration_error, reliability_curve


def test_reliability_curve_bin_counts_sum_to_n():
    rng = np.random.default_rng(0)
    y_proba = rng.uniform(0.2, 0.8, size=5000)
    y_true = (rng.uniform(size=5000) < y_proba).astype(int)
    curve = reliability_curve(y_true, y_proba, n_bins=10)
    assert sum(curve["bin_counts"]) == 5000


def test_reliability_curve_handles_few_distinct_values():
    """Regression test: quantile edge collisions must not drop rows."""
    rng = np.random.default_rng(0)
    distinct = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    y_proba = rng.choice(distinct, size=10000)
    y_true = (rng.uniform(size=10000) < y_proba).astype(int)
    curve = reliability_curve(y_true, y_proba, n_bins=10)
    assert sum(curve["bin_counts"]) == 10000
    assert curve["n_bins"] <= 5


def test_reliability_curve_handles_constant_probability():
    """Regression test: prevalence-baseline-style constant input must
    produce one bin covering all rows, not zero bins."""
    y_proba = np.full(1000, 0.553)
    y_true = np.random.default_rng(0).integers(0, 2, size=1000)
    curve = reliability_curve(y_true, y_proba, n_bins=10)
    assert curve["n_bins"] == 1
    assert sum(curve["bin_counts"]) == 1000
    assert curve["mean_predicted"][0] == pytest.approx(0.553)


def test_ece_zero_for_perfectly_calibrated_predictions():
    # Large sample so within-bin observed frequency converges to predicted.
    rng = np.random.default_rng(0)
    n = 200_000
    y_proba = rng.uniform(0.1, 0.9, size=n)
    y_true = (rng.uniform(size=n) < y_proba).astype(int)
    ece = expected_calibration_error(y_true, y_proba, n_bins=10)
    assert ece < 0.02


def test_ece_high_for_badly_miscalibrated_predictions():
    y_true = np.zeros(1000)
    y_proba = np.full(1000, 0.9)  # always confident, always wrong
    ece = expected_calibration_error(y_true, y_proba, n_bins=10)
    assert ece == pytest.approx(0.9, abs=1e-6)
