"""Tests for Stage 3 calibration diagnostics without fitting a calibrator."""

import numpy as np
import pytest

from ml.evaluation.calibration import fixed_risk_bands, stage3_calibration_report


def test_fixed_risk_bands_reconcile_to_all_rows_and_keep_empty_bins():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.01, 0.19, 0.81, 1.0])
    bands = fixed_risk_bands(y, p)
    assert len(bands) == 10
    assert sum(band["n"] for band in bands) == 4
    assert any(band["n"] == 0 and band["mean_prediction"] is None for band in bands)


def test_stage3_calibration_report_handles_single_class_slice():
    report = stage3_calibration_report(np.ones(10), np.full(10, 0.7))
    assert report["calibration_intercept"] is None
    assert report["calibration_slope"] is None
    assert sum(band["n"] for band in report["fixed_risk_bands"]) == 10


def test_stage3_ece_matches_hand_calculated_two_bin_fixture():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.2, 0.8, 0.8])
    report = stage3_calibration_report(y, p, n_bins=2)
    assert report["ece"] == pytest.approx(0.2)
