"""Verification-length policy behaviour and safety (spec section 13)."""

from __future__ import annotations

import numpy as np
import pytest

from edgespark.policy import (
    ThresholdPolicy,
    expected_accepted_length,
    optimal_length,
)


def test_expected_accepted_length_matches_formula():
    a = np.array([0.9, 0.8, 0.5])
    # 1 + a1 + a1a2 + a1a2a3
    manual = 1 + 0.9 + 0.9 * 0.8 + 0.9 * 0.8 * 0.5
    assert expected_accepted_length(a, 3) == pytest.approx(manual)
    assert expected_accepted_length(a, 0) == pytest.approx(1.0)


def test_expected_accepted_length_is_monotone_nondecreasing():
    a = np.array([0.95, 0.9, 0.7, 0.4, 0.2])
    vals = [expected_accepted_length(a, ell) for ell in range(len(a) + 1)]
    assert all(y >= x - 1e-12 for x, y in zip(vals, vals[1:], strict=False))


@pytest.mark.parametrize("theta", [0.05, 0.3, 0.6, 0.9, 0.999])
def test_choose_length_stays_in_bounds(theta):
    rng = np.random.default_rng(0)
    policy = ThresholdPolicy(theta=theta)
    for _ in range(200):
        block = rng.integers(1, 8)
        conf = rng.uniform(0, 1, block)
        ell = policy.choose_length(conf, recent_accept_ema=rng.random())
        assert 0 <= ell <= block


def test_higher_threshold_never_lengthens_verification():
    # On a fixed decreasing survival profile, raising theta can only shorten ell.
    conf = np.array([0.95, 0.9, 0.8, 0.6, 0.3])
    lengths = [
        ThresholdPolicy(theta=t).choose_length(conf, 0.5)
        for t in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    assert all(y <= x for x, y in zip(lengths, lengths[1:], strict=False))


def test_optimal_length_trades_verify_cost_against_acceptance():
    conf = np.array([0.9, 0.85, 0.8, 0.75, 0.7])

    # Cheap verification everywhere -> verify the whole block.
    cheap = optimal_length(conf, t_draft=1.0, t_verify=lambda ell: 0.01 * ell)
    assert cheap == len(conf)

    # A steep per-token verify cost -> stop well short of the full block.
    steep = optimal_length(conf, t_draft=1.0, t_verify=lambda ell: 2.0 * ell)
    assert 1 <= steep < len(conf)


def test_min_length_floor_respected():
    policy = ThresholdPolicy(theta=0.99, min_length=2)
    # Even with poor confidence the floor guarantees at least min_length.
    ell = policy.choose_length(np.array([0.1, 0.1, 0.1, 0.1]), 0.0)
    assert ell >= 2
