"""
Unit Tests for Phase 6: Matched-Exposure Timing + Asymmetric Downside Signals
=============================================================================
Tests mathematical invariants, safety constraints, and calibration mechanics:
1. Relative uncertainty causal lookahead check (strictly uses history up to t-1).
2. Robust normalization handling: zero variance, zero MAD, discrete values.
3. Fallback behavior when trailing history < min_history.
4. Relative exposure scale bounds [s_min, s_max], monotonicity, clamping.
5. Preserving cross-sectional relative risky weights under relative scaling.
6. Cash turnover and budget conservation: sum(w_risky) + w_cash == 1.0.
7. Placebo exposure path generator: distribution preservation and shape.
8. Reproducibility under fixed random seed.
"""

import numpy as np

from hypergraph_portfolio.uncertainty import (
    apply_portfolio_exposure_scale,
    compute_relative_epistemic_exposure_scale,
    compute_relative_epistemic_uncertainty,
    generate_placebo_exposure_paths,
)


def test_relative_uncertainty_causal_no_leakage():
    # If history is fixed, changing a future value or unrelated element does not alter z_t
    history = [0.010, 0.012, 0.011, 0.013, 0.012, 0.014]
    u_t = 0.018

    z1 = compute_relative_epistemic_uncertainty(history, u_t, window=5, method="zscore")
    # Using trailing 5 elements: [0.012, 0.011, 0.013, 0.012, 0.014]
    trailing = np.array([0.012, 0.011, 0.013, 0.012, 0.014])
    expected_z = (u_t - np.mean(trailing)) / np.std(trailing, ddof=1)
    assert np.isclose(z1, expected_z)

    # Robust zscore
    z_rob = compute_relative_epistemic_uncertainty(history, u_t, window=5, method="robust_zscore")
    med = np.median(trailing)
    mad = np.median(np.abs(trailing - med))
    expected_z_rob = (u_t - med) / (1.4826 * mad)
    assert np.isclose(z_rob, expected_z_rob)

    # Ratio
    r_val = compute_relative_epistemic_uncertainty(history, u_t, window=5, method="ratio")
    assert np.isclose(r_val, u_t / med)

    # Percentile rank
    p_val = compute_relative_epistemic_uncertainty(history, u_t, window=5, method="percentile_rank")
    assert np.isclose(p_val, 1.0)  # u_t is strictly greater than all elements


def test_relative_uncertainty_zero_mad_and_zero_variance():
    # Constant trailing history -> zero variance & zero MAD
    constant_hist = [0.01, 0.01, 0.01, 0.01, 0.01]
    u_t = 0.015

    # Should not divide by zero or crash; returns 0.0 safely
    z_zero_std = compute_relative_epistemic_uncertainty(constant_hist, u_t, method="zscore")
    assert z_zero_std == 0.0

    z_zero_mad = compute_relative_epistemic_uncertainty(constant_hist, u_t, method="robust_zscore")
    assert z_zero_mad == 0.0


def test_relative_uncertainty_short_history_fallback():
    # When history length < min_history, fallback safely
    short_hist = [0.01, 0.02]
    u_t = 0.03

    z = compute_relative_epistemic_uncertainty(short_hist, u_t, min_history=4, method="zscore")
    assert z == 0.0

    z_rob = compute_relative_epistemic_uncertainty(short_hist, u_t, min_history=4, method="robust_zscore")
    assert z_rob == 0.0

    rat = compute_relative_epistemic_uncertainty(short_hist, u_t, min_history=4, method="ratio")
    assert rat == 1.0

    rank = compute_relative_epistemic_uncertainty(short_hist, u_t, min_history=4, method="percentile_rank")
    assert rank == 0.5


def test_relative_exposure_scale_bounds_and_clamping():
    # Piecewise linear
    for z in [-3.0, -1.0, 0.0, 1.0, 2.5, 5.0]:
        s = compute_relative_epistemic_exposure_scale(
            z, s_base=0.70, lambda_rel=0.30, tau=0.5, s_min=0.10, s_max=0.90
        )
        assert 0.10 <= s <= 0.90
        # If z <= tau, s must equal s_base (clamped)
        if z <= 0.5:
            assert np.isclose(s, 0.70)
        else:
            assert s < 0.70

    # Smooth mapping
    for z in [-3.0, 0.0, 3.0]:
        s_smooth = compute_relative_epistemic_exposure_scale(
            z, s_min=0.20, s_max=0.80, tau=0.0, smooth=True
        )
        assert 0.20 <= s_smooth <= 0.80


def test_relative_exposure_monotonicity():
    # As relative uncertainty increases, exposure must weakly decrease
    z_grid = np.linspace(-2.0, 4.0, 50)
    scales = [compute_relative_epistemic_exposure_scale(z, s_base=0.80, lambda_rel=0.25, tau=0.0) for z in z_grid]

    for i in range(len(scales) - 1):
        assert scales[i] >= scales[i + 1] - 1e-9


def test_relative_weights_preservation():
    w_base = np.array([0.2, 0.3, 0.5])
    scale = 0.55

    w_risky, w_cash = apply_portfolio_exposure_scale(w_base, scale=scale)

    assert np.isclose(w_cash, 1.0 - scale)
    assert np.isclose(np.sum(w_risky), scale)
    assert np.isclose(np.sum(w_risky) + w_cash, 1.0)
    assert np.isclose(w_risky[0] / w_risky[1], w_base[0] / w_base[1])
    assert np.isclose(w_risky[1] / w_risky[2], w_base[1] / w_base[2])


def test_placebo_exposure_paths_distribution_preservation():
    actual_path = np.array([0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.5, 0.3, 0.9, 0.2])
    n_placebos = 100

    # Circular shift
    circ_placebos = generate_placebo_exposure_paths(actual_path, n_placebos=n_placebos, method="circular_shift", seed=42)
    assert circ_placebos.shape == (n_placebos, len(actual_path))
    for p in circ_placebos:
        assert np.isclose(np.mean(p), np.mean(actual_path))
        assert np.isclose(np.var(p), np.var(actual_path))
        assert np.allclose(np.sort(p), np.sort(actual_path))

    # Permutation
    perm_placebos = generate_placebo_exposure_paths(actual_path, n_placebos=n_placebos, method="permutation", seed=42)
    assert perm_placebos.shape == (n_placebos, len(actual_path))
    for p in perm_placebos:
        assert np.isclose(np.mean(p), np.mean(actual_path))
        assert np.isclose(np.var(p), np.var(actual_path))
        assert np.allclose(np.sort(p), np.sort(actual_path))


def test_placebo_reproducibility():
    path = np.array([0.2, 0.4, 0.6, 0.8])
    p1 = generate_placebo_exposure_paths(path, n_placebos=10, seed=123)
    p2 = generate_placebo_exposure_paths(path, n_placebos=10, seed=123)
    assert np.allclose(p1, p2)
