"""
Core evaluation metrics for Stage 1.

`compute_metrics` is the single entry point used for every model
(trivial baselines and both Logistic Regression variants) so results are
directly comparable. The 0.5 threshold used for accuracy/precision/
recall/F1/confusion-matrix is a fixed reporting convention, not a
business-optimized decision threshold — see docs/stage1-baseline.md.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPORTING_THRESHOLD = 0.5


def compute_metrics(y_true, y_proba, threshold: float = REPORTING_THRESHOLD) -> dict:
    """
    Compute the full Stage 1 metrics set for one model's predictions on
    one evaluation slice.

    y_true: array-like of {0, 1}
    y_proba: array-like of P(y=1), same length as y_true

    Returns a JSON-serialisable dict. ROC-AUC and PR-AUC are set to None
    (not NaN, for clean JSON) when y_true contains only one class, since
    both are undefined in that case.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    n = len(y_true)
    y_pred = (y_proba >= threshold).astype(int)

    single_class = len(np.unique(y_true)) < 2

    metrics = {
        "n": int(n),
        "prevalence": float(y_true.mean()) if n > 0 else None,
        "threshold": threshold,
        "roc_auc": None if single_class else float(roc_auc_score(y_true, y_proba)),
        "pr_auc": None if single_class else float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    return metrics


def compute_skill_scores(model_metrics: dict, baseline_metrics: dict) -> dict:
    """
    Skill score = 1 - model_error / baseline_error, for log loss and
    Brier score. Positive means the model beats the prevalence baseline;
    0 means tied; negative means worse than predicting the constant rate.
    """
    def _skill(key):
        base = baseline_metrics[key]
        if base == 0:
            return None
        return 1.0 - (model_metrics[key] / base)

    return {
        "log_loss_skill": _skill("log_loss"),
        "brier_skill": _skill("brier"),
    }
