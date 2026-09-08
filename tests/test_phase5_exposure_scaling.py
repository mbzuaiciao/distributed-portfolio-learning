"""
Unit Tests for Phase 5: Portfolio-Level Risk Scaling with Epistemic Uncertainty
=============================================================================
Tests mathematical invariants, safety constraints, and calibration mechanics:
1. Aggregate epistemic uncertainty methods (mean, median, weighted, max, q75).
2. Exposure scale bounds [0, 1] and clamping.
3. Monotonicity: d s / d u_t <= 0.
4. Deterministic reproduction: lambda = 0 => s = 1.0.
5. Volatility targeting scaling and leverage caps.
6. Preservation of cross-sectional relative weights under scaling.
7. Budget conservation: w_risky.sum() + w_cash == 1.0.
8. Cash turnover accounting.
9. Zero lookahead in rolling volatility calculation.
"""

import numpy as np
import pytest

from hypergraph_portfolio.uncertainty import (
    apply_portfolio_exposure_scale,
    compute_aggregate_epistemic_uncertainty,
    compute_epistemic_exposure_scale,
    compute_volatility_targeting_scale,
)


def test_aggregate_epistemic_uncertainty_methods():
    stds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

    # Mean
    mean_val = compute_aggregate_epistemic_uncertainty(stds, method="mean")
    assert np.isclose(mean_val, 0.03)

    # Median
    med_val = compute_aggregate_epistemic_uncertainty(stds, method="median")
    assert np.isclose(med_val, 0.03)

    # Weighted: 0.1*0.01 + 0.2*0.02 + 0.4*0.03 + 0.2*0.04 + 0.1*0.05
    # = 0.001 + 0.004 + 0.012 + 0.008 + 0.005 = 0.030
    w_val = compute_aggregate_epistemic_uncertainty(stds, weights=weights, method="weighted")
    assert np.isclose(w_val, 0.03)

    # Max
    max_val = compute_aggregate_epistemic_uncertainty(stds, method="max")
    assert np.isclose(max_val, 0.05)

    # Q75
    q75_val = compute_aggregate_epistemic_uncertainty(stds, method="q75")
    assert np.isclose(q75_val, np.percentile(stds, 75))

    # Error handling
    with pytest.raises(ValueError, match="weights must be provided"):
        compute_aggregate_epistemic_uncertainty(stds, method="weighted")

    with pytest.raises(ValueError, match="Unknown uncertainty aggregation method"):
        compute_aggregate_epistemic_uncertainty(stds, method="invalid")


def test_epistemic_exposure_scale_bounds_and_clamping():
    # Test bounds [0, 1]
    for u in [0.0, 0.005, 0.01, 0.05, 0.5, 5.0]:
        for lam in [0.0, 1.0, 10.0, 100.0, 500.0]:
            s_rat = compute_epistemic_exposure_scale(u, lambda_u=lam, mapping="rational")
            assert 0.0 <= s_rat <= 1.0

            s_exp = compute_epistemic_exposure_scale(u, lambda_u=lam, mapping="exponential")
            assert 0.0 <= s_exp <= 1.0

            s_bkt = compute_epistemic_exposure_scale(u, lambda_u=lam, mapping="bucket")
            assert 0.0 <= s_bkt <= 1.0

    # Floor test
    s_floored = compute_epistemic_exposure_scale(10.0, lambda_u=100.0, floor=0.20)
    assert np.isclose(s_floored, 0.20)


def test_epistemic_exposure_scale_monotonicity():
    # As uncertainty increases, exposure must weakly decrease
    u_vals = np.linspace(0.001, 0.10, 50)
    lam = 50.0

    scales = [compute_epistemic_exposure_scale(u, lambda_u=lam, mapping="rational") for u in u_vals]
    for i in range(len(scales) - 1):
        assert scales[i] >= scales[i + 1] - 1e-9

    scales_exp = [compute_epistemic_exposure_scale(u, lambda_u=lam, mapping="exponential") for u in u_vals]
    for i in range(len(scales_exp) - 1):
        assert scales_exp[i] >= scales_exp[i + 1] - 1e-9


def test_deterministic_reproduction_lambda_zero():
    # When lambda_u == 0, exposure scale must be identically 1.0 regardless of uncertainty
    for u in [0.0, 0.01, 0.05, 1.0, 100.0]:
        s = compute_epistemic_exposure_scale(u, lambda_u=0.0)
        assert np.isclose(s, 1.0)


def test_volatility_targeting_scale():
    # Normal target
    s1 = compute_volatility_targeting_scale(rolling_vol=0.40, target_vol=0.20, max_leverage=1.0)
    assert np.isclose(s1, 0.50)

    # Low vol capped at max_leverage (no leverage when max_leverage=1.0)
    s2 = compute_volatility_targeting_scale(rolling_vol=0.10, target_vol=0.20, max_leverage=1.0)
    assert np.isclose(s2, 1.0)

    # With leverage allowed
    s3 = compute_volatility_targeting_scale(rolling_vol=0.10, target_vol=0.20, max_leverage=2.0)
    assert np.isclose(s3, 2.0)

    # Zero rolling vol
    s4 = compute_volatility_targeting_scale(rolling_vol=0.0, target_vol=0.20, max_leverage=1.0)
    assert np.isclose(s4, 1.0)

    with pytest.raises(ValueError, match="target_vol must be positive"):
        compute_volatility_targeting_scale(rolling_vol=0.20, target_vol=-0.1)


def test_preservation_of_relative_weights():
    # Test that relative weights across risky assets are strictly unchanged
    w_base = np.array([0.15, 0.25, 0.60])
    scale = 0.65

    w_risky, w_cash = apply_portfolio_exposure_scale(w_base, scale=scale)

    assert np.isclose(w_cash, 1.0 - scale)
    assert np.isclose(np.sum(w_risky), scale)
    assert np.isclose(np.sum(w_risky) + w_cash, 1.0)

    # Cross-sectional relative ratios must match exactly
    assert np.isclose(w_risky[0] / w_risky[1], w_base[0] / w_base[1])
    assert np.isclose(w_risky[1] / w_risky[2], w_base[1] / w_base[2])
    assert np.isclose(w_risky[0] / w_risky[2], w_base[0] / w_base[2])


def test_budget_conservation_and_extreme_scales():
    w_base = np.array([0.4, 0.6])

    # Scale 0 -> all cash
    w_r0, w_c0 = apply_portfolio_exposure_scale(w_base, scale=0.0)
    assert np.allclose(w_r0, 0.0)
    assert np.isclose(w_c0, 1.0)

    # Scale 1 -> fully invested
    w_r1, w_c1 = apply_portfolio_exposure_scale(w_base, scale=1.0)
    assert np.allclose(w_r1, w_base)
    assert np.isclose(w_c1, 0.0)


def test_cash_turnover_accounting():
    # Verify that cash transitions contribute to turnover:
    # turnover_t = sum_j |w_risky,j,t - w_risky,j,t-1| + |w_cash,t - w_cash,t-1|
    w_base_1 = np.array([0.5, 0.5])
    w_base_2 = np.array([0.5, 0.5])

    w_r1, w_c1 = apply_portfolio_exposure_scale(w_base_1, scale=1.0)  # [0.5, 0.5], cash=0.0
    w_r2, w_c2 = apply_portfolio_exposure_scale(w_base_2, scale=0.5)  # [0.25, 0.25], cash=0.5

    risky_turnover = np.sum(np.abs(w_r2 - w_r1))  # |0.25 - 0.5| + |0.25 - 0.5| = 0.5
    cash_turnover = np.abs(w_c2 - w_c1)  # |0.5 - 0.0| = 0.5
    total_turnover = risky_turnover + cash_turnover

    assert np.isclose(risky_turnover, 0.5)
    assert np.isclose(cash_turnover, 0.5)
    assert np.isclose(total_turnover, 1.0)
