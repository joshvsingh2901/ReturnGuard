"""
Tests for ml.models.gbdt.

Uses small synthetic data (a few thousand rows) so XGBoost fits stay in
the range of tens to hundreds of milliseconds — appropriate for ordinary
pytest, unlike the full 1.37M-row ablation ladder run in
scripts/train_stage2.py.
"""

import numpy as np
import pytest

from ml.data.schema import TARGET_COL
from ml.models.gbdt import (
    DEFAULT_XGB_PARAMS,
    check_early_stopping_triggered,
    fit_gbdt,
)

FAST_PARAMS = dict(DEFAULT_XGB_PARAMS)
FAST_PARAMS.update(n_estimators=50, early_stopping_rounds=10)


def _three_way_split(df, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n = len(df)
    a, b = int(n * 0.6), int(n * 0.8)
    return (
        df.iloc[idx[:a]].reset_index(drop=True),
        df.iloc[idx[a:b]].reset_index(drop=True),
        df.iloc[idx[b:]].reset_index(drop=True),
    )


def test_predict_proba_shape_and_validity(synthetic_df_with_missing):
    inner_train, early_stop, val = _three_way_split(synthetic_df_with_missing)
    result = fit_gbdt(
        inner_train, early_stop, val,
        artifact_safe=True, include_derived=False, include_frequency=False,
        xgb_params=FAST_PARAMS,
    )
    assert result.proba_val.shape == (len(val),)
    assert np.isfinite(result.proba_val).all()
    assert (result.proba_val >= 0).all() and (result.proba_val <= 1).all()


def test_deterministic_within_tolerance(synthetic_df_with_missing):
    inner_train, early_stop, val = _three_way_split(synthetic_df_with_missing)
    r1 = fit_gbdt(inner_train, early_stop, val, artifact_safe=True, xgb_params=FAST_PARAMS)
    r2 = fit_gbdt(inner_train, early_stop, val, artifact_safe=True, xgb_params=FAST_PARAMS)
    np.testing.assert_allclose(r1.proba_val, r2.proba_val, atol=1e-6)


def test_early_stopping_triggers_or_raises(synthetic_df_with_missing):
    inner_train, early_stop, val = _three_way_split(synthetic_df_with_missing)
    # Generous n_estimators so early stopping has room to trigger.
    params = dict(DEFAULT_XGB_PARAMS)
    params.update(n_estimators=500, early_stopping_rounds=10)
    result = fit_gbdt(inner_train, early_stop, val, artifact_safe=True, xgb_params=params)
    check_early_stopping_triggered(result)  # must not raise
    assert result.best_iteration < result.n_estimators_fit + 10


def test_artifact_safe_pipeline_has_donor_impute_step(synthetic_df_with_missing):
    inner_train, early_stop, val = _three_way_split(synthetic_df_with_missing)
    result = fit_gbdt(inner_train, early_stop, val, artifact_safe=True, xgb_params=FAST_PARAMS)
    assert "donor_impute" in result.feature_pipeline.named_steps


def test_diagnostic_pipeline_has_no_donor_impute_step(synthetic_df_with_missing):
    inner_train, early_stop, val = _three_way_split(synthetic_df_with_missing)
    result = fit_gbdt(inner_train, early_stop, val, artifact_safe=False, xgb_params=FAST_PARAMS)
    assert "donor_impute" not in result.feature_pipeline.named_steps


def test_include_derived_adds_features(synthetic_df_with_missing):
    inner_train, early_stop, val = _three_way_split(synthetic_df_with_missing)
    without = fit_gbdt(inner_train, early_stop, val, artifact_safe=True,
                        include_derived=False, xgb_params=FAST_PARAMS)
    with_derived = fit_gbdt(inner_train, early_stop, val, artifact_safe=True,
                             include_derived=True, xgb_params=FAST_PARAMS)
    assert len(with_derived.feature_names) > len(without.feature_names)
    assert any("discount_amount" in n for n in with_derived.feature_names)


def test_include_frequency_adds_features(synthetic_df_with_ids):
    inner_train, early_stop, val = _three_way_split(synthetic_df_with_ids)
    result = fit_gbdt(inner_train, early_stop, val, artifact_safe=True,
                       include_frequency=True, xgb_params=FAST_PARAMS)
    assert any("variant_event_count" in n for n in result.feature_names)
    assert not any("customer" in n.lower() and "count" in n.lower()
                   for n in result.feature_names)


def test_target_excluded_from_features(synthetic_df_with_missing):
    inner_train, early_stop, val = _three_way_split(synthetic_df_with_missing)
    result = fit_gbdt(inner_train, early_stop, val, artifact_safe=True, xgb_params=FAST_PARAMS)
    assert not any(TARGET_COL in n for n in result.feature_names)
