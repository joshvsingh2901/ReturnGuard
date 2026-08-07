"""Tests for ml.models.baselines and ml.models.logistic."""

import numpy as np

from ml.data.schema import EVENT_CUST_COL, TARGET_COL
from ml.models.baselines import build_majority_baseline, build_prevalence_baseline
from ml.models.logistic import build_logistic_pipeline, check_convergence


def test_prevalence_baseline_predicts_constant_probability(synthetic_df):
    model = build_prevalence_baseline()
    X = synthetic_df[[EVENT_CUST_COL]]
    y = synthetic_df[TARGET_COL]
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert np.allclose(proba, proba[0])
    assert np.isclose(proba[0], y.mean())


def test_majority_baseline_predicts_majority_class(synthetic_df):
    model = build_majority_baseline()
    X = synthetic_df[[EVENT_CUST_COL]]
    y = synthetic_df[TARGET_COL]
    model.fit(X, y)
    pred = model.predict(X)
    majority_class = int(y.mean() >= 0.5)
    assert (pred == majority_class).all()


def test_logistic_pipeline_predict_proba_shape(synthetic_df):
    pipe = build_logistic_pipeline(include_price=True)
    pipe.fit(synthetic_df, synthetic_df[TARGET_COL])
    proba = pipe.predict_proba(synthetic_df)
    assert proba.shape == (len(synthetic_df), 2)


def test_logistic_pipeline_probabilities_valid_range(synthetic_df):
    pipe = build_logistic_pipeline(include_price=True)
    pipe.fit(synthetic_df, synthetic_df[TARGET_COL])
    proba = pipe.predict_proba(synthetic_df)
    assert np.isfinite(proba).all()
    assert (proba >= 0).all() and (proba <= 1).all()


def test_logistic_pipeline_probabilities_sum_to_one(synthetic_df):
    pipe = build_logistic_pipeline(include_price=True)
    pipe.fit(synthetic_df, synthetic_df[TARGET_COL])
    proba = pipe.predict_proba(synthetic_df)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logistic_pipeline_converges(synthetic_df):
    pipe = build_logistic_pipeline(include_price=True)
    pipe.fit(synthetic_df, synthetic_df[TARGET_COL])
    check_convergence(pipe)  # must not raise


def test_logistic_pipeline_strict_variant_fits(synthetic_df):
    pipe = build_logistic_pipeline(include_price=False)
    pipe.fit(synthetic_df, synthetic_df[TARGET_COL])
    check_convergence(pipe)
    proba = pipe.predict_proba(synthetic_df)
    assert proba.shape == (len(synthetic_df), 2)


def test_logistic_pipeline_handles_missing_data(synthetic_df_with_missing):
    pipe = build_logistic_pipeline(include_price=True)
    pipe.fit(synthetic_df_with_missing, synthetic_df_with_missing[TARGET_COL])
    proba = pipe.predict_proba(synthetic_df_with_missing)
    assert np.isfinite(proba).all()
