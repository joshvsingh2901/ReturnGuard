"""Unit tests for Stage 3 SHAP grouping and additivity helpers."""

import numpy as np
from xgboost import XGBClassifier

from ml.explainability.grouping import aggregate_grouped_shap, feature_display_name
from ml.explainability.shap_analysis import compute_tree_shap


def test_grouped_shap_sums_signed_columns_before_absolute_value():
    names = [
        "cat_prod__productType_A",
        "cat_prod__productType_B",
        "num_prod__avgGbpPrice",
    ]
    values = np.array([[2.0, -1.0, 3.0], [4.0, -4.0, 1.0]])
    grouped, ranking = aggregate_grouped_shap(values, names)
    assert grouped["product_type"].tolist() == [1.0, 0.0]
    assert ranking.loc[ranking["feature_family"] == "product_type", "mean_abs_shap"].iloc[0] == 0.5


def test_one_hot_display_name_uses_observed_raw_category():
    import pandas as pd

    row = pd.Series({"productType": "Jeans"})
    assert feature_display_name("cat_prod__productType_productType_A", row) == "Product type: Jeans"


def test_tree_shap_width_and_raw_margin_additivity():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 4)).astype(np.float32)
    y = (X[:, 0] + 0.4 * X[:, 1] > 0).astype(int)
    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        n_estimators=20,
        max_depth=3,
        random_state=2,
        n_jobs=1,
    ).fit(X, y)
    result = compute_tree_shap(model, X[:20])
    assert result.values.shape == (20, 4)
    assert result.additivity_max_abs_error < 1e-4
    assert result.probability_max_abs_error < 1e-6
