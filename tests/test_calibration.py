"""Calibration metric correctness and recalibration behaviour (spec section 13).

These lock down the math behind the headline result so a regression in ECE or a
broken temperature fit gets caught before it silently rewrites a reliability
diagram.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgespark.calibration import (
    PlattScaler,
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _bernoulli(rng, p):
    return (rng.random(p.shape) < p).astype(np.float64)


def test_perfectly_calibrated_data_has_low_ece():
    # If outcomes are Bernoulli(confidence), the head is calibrated by construction.
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, size=200_000)
    outcome = _bernoulli(rng, conf)
    assert expected_calibration_error(conf, outcome, n_bins=15) < 0.01


def test_miscalibrated_data_has_high_ece():
    # Report confidence c but actually accept with probability c**2 -> overconfident.
    rng = np.random.default_rng(1)
    conf = rng.uniform(0, 1, size=200_000)
    outcome = _bernoulli(rng, conf**2)
    ece = expected_calibration_error(conf, outcome, n_bins=15)
    assert ece > 0.1
    assert maximum_calibration_error(conf, outcome) >= ece


def test_brier_bounds():
    rng = np.random.default_rng(2)
    outcome = (rng.random(1000) < 0.5).astype(float)
    # Perfect hard predictions -> 0; worst-case flipped predictions -> 1.
    assert brier_score(outcome, outcome) == pytest.approx(0.0)
    assert brier_score(1 - outcome, outcome) == pytest.approx(1.0)
    # Constant 0.5 hedge -> 0.25 regardless of outcome.
    assert brier_score(np.full_like(outcome, 0.5), outcome) == pytest.approx(0.25, abs=1e-9)


def test_reliability_curve_partitions_data():
    rng = np.random.default_rng(3)
    conf = rng.uniform(0, 1, 5000)
    outcome = _bernoulli(rng, conf)
    curve = reliability_curve(conf, outcome, n_bins=10)
    assert curve.bin_count.sum() == conf.size
    assert np.all((curve.bin_accuracy >= 0) & (curve.bin_accuracy <= 1))


def test_temperature_scaling_recovers_overconfident_head():
    # Simulate an over-sharp (post-quantization) head: true logits scaled up by 2.5.
    rng = np.random.default_rng(4)
    true_logit = rng.standard_normal(100_000)
    outcome = _bernoulli(rng, _sigmoid(true_logit))
    sharp_conf = _sigmoid(2.5 * true_logit)  # miscalibrated confidence

    before = expected_calibration_error(sharp_conf, outcome)
    scaler = TemperatureScaler().fit(sharp_conf, outcome)
    after = expected_calibration_error(scaler.transform(sharp_conf), outcome)

    assert before > 0.05
    assert after < 0.5 * before  # recalibration at least halves ECE
    assert scaler.temperature > 1.0  # it cooled the over-sharp head down


def test_platt_scaling_handles_shift_and_scale():
    rng = np.random.default_rng(5)
    true_logit = rng.standard_normal(100_000)
    outcome = _bernoulli(rng, _sigmoid(true_logit))
    # Both a slope and a bias error.
    bad_conf = _sigmoid(1.8 * true_logit + 0.9)

    before = expected_calibration_error(bad_conf, outcome)
    platt = PlattScaler().fit(bad_conf, outcome)
    after = expected_calibration_error(platt.transform(bad_conf), outcome)
    assert after < 0.5 * before


def test_recalibration_is_monotone():
    # Scaling must preserve ranking: it rescales confidence, never reorders it.
    rng = np.random.default_rng(6)
    conf = np.sort(rng.uniform(0.01, 0.99, 500))
    outcome = _bernoulli(rng, conf)
    for scaler in (TemperatureScaler().fit(conf, outcome), PlattScaler().fit(conf, outcome)):
        mapped = scaler.transform(conf)
        assert np.all(np.diff(mapped) >= -1e-9)
