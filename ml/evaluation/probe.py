"""
Leakage probe: measures whether the processed customer-side feature
matrix still encodes has_customer_node status.

This is the instrument that turns "we believe the artifact is
suppressed" into a measured number. It trains a small XGBoost to predict
has_customer_node (a label that is NEVER used as a model feature anywhere
else in this project) from the same customer-side columns the primary
model sees, under each of the two customer-side preprocessing regimes:

  - median/mode imputation (Stage 1-style, used only in A1): expected to
    be highly detectable, since a tree can trivially isolate the
    "yearOfBirth == median AND isMale == mode AND premier == mode AND
    shippingCountry == mode" conjunction that flags every missing-node
    row identically.
  - donor imputation (ml.features.imputation.DonorImputer, used in A2+):
    expected to move detectability toward chance, since every imputed row
    now carries a real, jointly-sampled customer profile indistinguishable
    from a genuine one by construction.

Target probe ROC-AUC <= 0.55 is a target, not a hard pass/fail gate — see
docs/stage2-gbdt.md for the measured result and how it was interpreted.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from ml.features.imputation import CUSTOMER_COLS, DonorImputer

PROBE_SEED = 42


def _has_customer_node(df: pd.DataFrame) -> pd.Series:
    """
    Cleaning-order-independent has-customer-node detector, via isMale
    (never NaN for any reason other than a missing customer node — Stage
    0/1 confirmed zero raw nulls in the customer node table).

    Deliberately does NOT use ml.data.joins.has_customer_node_mask, which
    is defined as yearOfBirth.notna() and is only correct when called on
    the RAW joined frame BEFORE ml.features.preprocessing.
    clean_year_of_birth runs. This project's Stage 2 pipeline always
    works with already-cleaned frames (the 1900 sentinel already
    converted to NaN), so calling the Stage 1 helper here would silently
    mislabel every sentinel-affected node-present row as node-missing —
    exactly the bug this docstring exists to prevent regressing.
    Discovered during Stage 2 development: doing this wrong inflated the
    donor-imputation probe's residual AUC from ~0.5 to ~0.615, entirely
    via a ~1.4%-of-node-present-rows label corruption, not any real
    signal in the donor-imputed features. See docs/stage2-gbdt.md.
    """
    return df["isMale"].notna()


def _median_mode_impute(train: pd.DataFrame, val: pd.DataFrame):
    """Standalone reproduction of Stage 1's customer-side imputation,
    used only to give the probe a contaminated preprocessing path to
    detect. Never used for any reported primary-model metric."""
    train_imp = train[CUSTOMER_COLS].copy()
    val_imp = val[CUSTOMER_COLS].copy()

    year_median = train_imp["yearOfBirth"].median()
    male_mode = train_imp["isMale"].mode(dropna=True).iloc[0]
    premier_mode = train_imp["premier"].mode(dropna=True).iloc[0]
    country_mode = train_imp["shippingCountry"].mode(dropna=True).iloc[0]

    fill_values = {
        "yearOfBirth": year_median,
        "isMale": male_mode,
        "premier": premier_mode,
        "shippingCountry": country_mode,
    }
    train_imp = train_imp.fillna(value=fill_values)
    val_imp = val_imp.fillna(value=fill_values)
    return train_imp, val_imp


def _donor_impute(train: pd.DataFrame, val: pd.DataFrame):
    imputer = DonorImputer(random_state=PROBE_SEED)
    train_imp = imputer.fit_transform(train[CUSTOMER_COLS])
    val_imp = imputer.transform(val[CUSTOMER_COLS])
    return train_imp, val_imp


def leakage_probe_auc(
    train: pd.DataFrame,
    val: pd.DataFrame,
    artifact_safe: bool,
    seed: int = PROBE_SEED,
) -> dict:
    """
    Fit a small XGBoost to predict has_customer_node from the processed
    customer-side columns only (product features are excluded — they
    carry no information about customer-node status by construction and
    would only dilute the signal this probe measures).

    has_customer_node labels are computed via _has_customer_node (isMale-
    based, robust regardless of whether yearOfBirth sentinel cleaning has
    already run) and used ONLY as this probe's target — never as a
    feature anywhere in the project.

    train must be a training-fold slice; this function fits both the
    imputer and the probe classifier on train only, then evaluates once
    on val.
    """
    y_train = _has_customer_node(train).astype(int).to_numpy()
    y_val = _has_customer_node(val).astype(int).to_numpy()

    if artifact_safe:
        train_imp, val_imp = _donor_impute(train, val)
    else:
        train_imp, val_imp = _median_mode_impute(train, val)

    ct = ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["yearOfBirth"]),
            ("bin", "passthrough", ["isMale", "premier"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["shippingCountry"]),
        ],
        sparse_threshold=0,
    )
    X_train = ct.fit_transform(train_imp).astype(np.float32)
    X_val = ct.transform(val_imp).astype(np.float32)

    clf = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cpu",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=4,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)[:, 1]

    auc = float(roc_auc_score(y_val, proba)) if len(np.unique(y_val)) > 1 else 0.5

    return {
        "probe_auc": auc,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_prevalence_has_node": float(y_train.mean()),
        "val_prevalence_has_node": float(y_val.mean()),
        "artifact_safe": artifact_safe,
    }
