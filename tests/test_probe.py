"""
Tests for ml.evaluation.probe — the leakage probe that measures whether
processed customer features still encode has_customer_node status.

Uses a synthetic fixture with an ARTIFICIAL target-correlated missingness
pattern shaped like the real dataset's confirmed artifact, so these tests
run in milliseconds rather than depending on the real raw pickle files.
"""

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.probe import leakage_probe_auc


def _make_probe_fixture(n=6000, frac_missing=0.15, seed=0):
    """
    Synthetic customer-side data where missing-node rows have ALL FOUR
    fields NaN simultaneously (matching real join-miss semantics), with
    enough rows and a low missingness rate so a median/mode-imputed
    conjunction is trivially separable by a tree, mirroring the real
    dataset's confirmed detectability.
    """
    rng = np.random.default_rng(seed)
    years = rng.integers(1950, 2005, size=n).astype(float)
    is_male = rng.integers(0, 2, size=n).astype(float)
    premier = rng.integers(0, 2, size=n).astype(float)
    country = rng.choice(["A", "B", "C", "D", "E"], size=n)

    df = pd.DataFrame({
        "yearOfBirth": years, "isMale": is_male,
        "premier": premier, "shippingCountry": country,
    })

    missing_mask = rng.random(n) < frac_missing
    df.loc[missing_mask, ["yearOfBirth", "isMale", "premier", "shippingCountry"]] = np.nan

    half = n // 2
    return df.iloc[:half].reset_index(drop=True), df.iloc[half:].reset_index(drop=True)


def test_median_mode_imputation_is_highly_detectable():
    train, val = _make_probe_fixture()
    result = leakage_probe_auc(train, val, artifact_safe=False)
    assert result["probe_auc"] > 0.9, (
        f"Expected median/mode imputation to be highly detectable, got "
        f"probe_auc={result['probe_auc']}"
    )


def test_donor_imputation_substantially_reduces_detectability():
    # Larger fixture than the default: at only a few thousand rows, the
    # probe's own AUC estimate is noisy even when the underlying signal
    # is near-absent, so a small fixture can land well above 0.5 by
    # chance alone. The REAL dataset run (900K+ rows, docs/stage2-gbdt.md)
    # targets <= 0.55; this test only checks the qualitative property
    # that donor imputation substantially reduces detectability relative
    # to median/mode, not an absolute bound tuned to real-data statistics.
    train, val = _make_probe_fixture(n=20000)
    contaminated = leakage_probe_auc(train, val, artifact_safe=False)
    safe = leakage_probe_auc(train, val, artifact_safe=True)
    assert safe["probe_auc"] < contaminated["probe_auc"] - 0.2, (
        f"Donor imputation did not substantially reduce detectability: "
        f"contaminated={contaminated['probe_auc']}, safe={safe['probe_auc']}"
    )
    assert safe["probe_auc"] < 0.8


def test_probe_never_uses_product_features():
    """The probe's ColumnTransformer must reference only the four
    customer columns — verified indirectly by confirming it runs on data
    with no product columns present at all."""
    train, val = _make_probe_fixture(n=2000)
    result = leakage_probe_auc(train, val, artifact_safe=True)
    assert 0.0 <= result["probe_auc"] <= 1.0


def test_probe_result_has_expected_keys():
    train, val = _make_probe_fixture(n=1000)
    result = leakage_probe_auc(train, val, artifact_safe=True)
    for key in ["probe_auc", "n_train", "n_val",
                "train_prevalence_has_node", "val_prevalence_has_node"]:
        assert key in result
