"""Tests for ml.evaluation.metrics."""

import numpy as np
import pytest

from ml.evaluation.metrics import compute_metrics, compute_skill_scores


def test_prevalence_baseline_roc_auc_is_half():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=2000)
    p = y_true.mean()
    y_proba = np.full(2000, p)
    m = compute_metrics(y_true, y_proba)
    assert m["roc_auc"] == pytest.approx(0.5, abs=1e-9)


def test_prevalence_baseline_brier_closed_form():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=5000)
    p = y_true.mean()
    y_proba = np.full(5000, p)
    m = compute_metrics(y_true, y_proba)
    expected_brier = p * (1 - p)  # closed-form: E[(y - p)^2] when predicting constant p
    assert m["brier"] == pytest.approx(expected_brier, abs=1e-9)


def test_perfect_prediction_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    m = compute_metrics(y_true, y_proba)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["log_loss"] == pytest.approx(0.0, abs=1e-9)
    assert m["brier"] == pytest.approx(0.0, abs=1e-9)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_worst_case_prediction_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([1.0, 1.0, 0.0, 0.0])  # exactly backwards
    m = compute_metrics(y_true, y_proba)
    assert m["roc_auc"] == pytest.approx(0.0)
    assert m["accuracy"] == pytest.approx(0.0)


def test_single_class_returns_none_for_auc():
    y_true = np.array([1, 1, 1, 1])
    y_proba = np.array([0.6, 0.7, 0.55, 0.9])
    m = compute_metrics(y_true, y_proba)
    assert m["roc_auc"] is None
    assert m["pr_auc"] is None


def test_metrics_n_and_prevalence():
    y_true = np.array([0, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.8, 0.7, 0.2])
    m = compute_metrics(y_true, y_proba)
    assert m["n"] == 5
    assert m["prevalence"] == pytest.approx(0.6)


def test_confusion_matrix_shape_and_sum():
    y_true = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.6, 0.51])
    m = compute_metrics(y_true, y_proba)
    cm = m["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2
    assert sum(sum(row) for row in cm) == 5


def test_skill_score_zero_when_identical():
    m = {"log_loss": 0.5, "brier": 0.2}
    skill = compute_skill_scores(m, m)
    assert skill["log_loss_skill"] == pytest.approx(0.0)
    assert skill["brier_skill"] == pytest.approx(0.0)


def test_skill_score_positive_when_model_beats_baseline():
    model = {"log_loss": 0.4, "brier": 0.15}
    baseline = {"log_loss": 0.5, "brier": 0.2}
    skill = compute_skill_scores(model, baseline)
    assert skill["log_loss_skill"] > 0
    assert skill["brier_skill"] > 0
