"""
XGBoost pipeline construction and fitting for Stage 2.

XGBoost early stopping needs an eval_set that has already been through
the same preprocessing as the training data, which sklearn's Pipeline
does not support passing through automatically (fit_params are forwarded
to each step's own .fit(), not applied to extra arrays like eval_set).
Rather than fight that integration, this module keeps the preprocessing
steps as an explicit, fold-safe pipeline (fit on train, applied to
eval/validation) and fits XGBClassifier directly on the transformed
arrays. This is a standard, well-supported pattern for combining sklearn
preprocessing with XGBoost early stopping.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml.data.schema import TARGET_COL
from ml.features.derived import DerivedPriceFeatures
from ml.features.imputation import DonorImputer
from ml.features.preprocessing import build_gbdt_preprocessor, get_feature_names
from ml.features.frequency import ProductFrequencyEncoder

XGB_SEED = 42
XGB_N_JOBS = 4

DEFAULT_XGB_PARAMS = dict(
    objective="binary:logistic",
    tree_method="hist",
    device="cpu",
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.1,
    random_state=XGB_SEED,
    n_jobs=XGB_N_JOBS,
    eval_metric="logloss",
    early_stopping_rounds=50,
)


def build_feature_pipeline(
    artifact_safe: bool = True,
    include_derived: bool = False,
    include_frequency: bool = False,
) -> Pipeline:
    """
    The feature-engineering + preprocessing portion only (no classifier),
    so it can be fit once on inner_train and reused to transform
    early_stop / primary_val consistently before XGBoost fitting.
    """
    steps = []
    if artifact_safe:
        steps.append(("donor_impute", DonorImputer(random_state=XGB_SEED)))
    if include_derived:
        steps.append(("derived", DerivedPriceFeatures()))
    if include_frequency:
        steps.append(("frequency", ProductFrequencyEncoder()))
    steps.append((
        "preprocess",
        build_gbdt_preprocessor(
            artifact_safe=artifact_safe,
            include_derived=include_derived,
            include_frequency=include_frequency,
        ),
    ))
    return Pipeline(steps)


@dataclass
class GBDTFitResult:
    feature_pipeline: Pipeline
    model: XGBClassifier
    feature_names: list
    proba_val: np.ndarray
    best_iteration: int
    n_estimators_fit: int
    params: dict = field(default_factory=dict)


def fit_gbdt(
    inner_train: pd.DataFrame,
    early_stop: pd.DataFrame,
    eval_df: pd.DataFrame,
    artifact_safe: bool = True,
    include_derived: bool = False,
    include_frequency: bool = False,
    xgb_params: dict = None,
    target_col: str = TARGET_COL,
) -> GBDTFitResult:
    """
    Fit the feature pipeline on inner_train only, transform early_stop and
    eval_df with the fitted pipeline, then fit XGBClassifier with early
    stopping monitored on the transformed early_stop fold. Returns
    predictions on eval_df (never used for early stopping or any
    selection decision by this function itself — that is the caller's
    responsibility, per the "primary validation is evaluated exactly
    once" rule documented in scripts/train_stage2.py).
    """
    pipeline = build_feature_pipeline(
        artifact_safe=artifact_safe,
        include_derived=include_derived,
        include_frequency=include_frequency,
    )

    X_train = pipeline.fit_transform(inner_train)
    y_train = inner_train[target_col].to_numpy()

    X_early = pipeline.transform(early_stop)
    y_early = early_stop[target_col].to_numpy()

    X_eval = pipeline.transform(eval_df)

    params = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)

    clf = XGBClassifier(**params)
    clf.fit(
        X_train.astype(np.float32),
        y_train,
        eval_set=[(X_early.astype(np.float32), y_early)],
        verbose=False,
    )

    proba_val = clf.predict_proba(X_eval.astype(np.float32))[:, 1]
    feature_names = get_feature_names(pipeline.named_steps["preprocess"])

    return GBDTFitResult(
        feature_pipeline=pipeline,
        model=clf,
        feature_names=feature_names,
        proba_val=proba_val,
        best_iteration=int(clf.best_iteration),
        n_estimators_fit=int(clf.get_booster().num_boosted_rounds()),
        params=params,
    )


def check_early_stopping_triggered(result: GBDTFitResult) -> None:
    """
    Raise if early stopping never actually triggered before hitting the
    n_estimators cap — that would mean the model may be under-fit for its
    learning rate/depth, not that it converged cleanly.
    """
    if result.n_estimators_fit >= result.params["n_estimators"]:
        raise RuntimeError(
            f"XGBoost trained the full n_estimators={result.params['n_estimators']} "
            f"cap without early stopping triggering (best_iteration="
            f"{result.best_iteration}). Consider raising n_estimators."
        )
