"""Regression tests for Tree SHAP scoring-space alignment."""

import numpy as np
from scipy.special import expit
from xgboost import XGBClassifier

from ml.explainability.shap_analysis import compute_tree_shap, model_raw_margin


def test_tree_shap_respects_early_stopping_prediction_tree_range():
    """SHAP must explain the same tree slice as XGBClassifier.predict_proba."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(120, 4)).astype(np.float32)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_early = rng.normal(size=(60, 4)).astype(np.float32)
    # Deliberately adversarial labels make early stopping deterministic and
    # leave post-best trees in the booster, which exposed the original bug.
    y_early = 1 - (X_early[:, 0] > 0).astype(int)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=30,
        max_depth=2,
        learning_rate=0.3,
        early_stopping_rounds=2,
        random_state=42,
        n_jobs=1,
    ).fit(X_train, y_train, eval_set=[(X_early, y_early)], verbose=False)

    assert model.best_iteration < model.get_booster().num_boosted_rounds() - 1
    X = X_early[:12]
    raw_margin = model_raw_margin(model, X)
    assert np.allclose(expit(raw_margin), model.predict_proba(X)[:, 1], atol=1e-6)

    result = compute_tree_shap(model, X)
    assert result.additivity_max_abs_error < 1e-5
    assert result.probability_max_abs_error < 1e-6
