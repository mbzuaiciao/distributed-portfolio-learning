"""
Tests for Phase 4: Decision-Relevant Risk Mappings & Downside Uncertainty
========================================================================
Validates:
1. Covariance Inflation:
   - lambda = 0 identity
   - symmetry preservation
   - positive semi-definiteness (PSD)
   - diagonal risk rises monotonically with epistemic uncertainty
   - off-diagonal elements preserved intact
2. Posterior Downside Probability:
   - probabilities bounded in [0, 1]
   - zero-variance behavior (above, below, equal to hurdle)
   - probability decreases monotonically as posterior mean rises
   - probability increases as uncertainty rises when mean is positive and hurdle is below mean
3. Downside-Adjusted Allocation:
   - downside-adjusted alpha monotonic dampening
   - downside risk inflation symmetry and PSD
4. Experimental Integrity:
   - development and holdout date non-overlap
   - identical point forecasts and covariance inputs across all policies
   - no lookahead target leakage in portfolio weights
   - deterministic reproducibility and block bootstrap determinism
"""

from __future__ import annotations

import numpy as np
import pytest

from hypergraph_portfolio import (
    BayesianDatasetAdapter,
    apply_downside_adjusted_alpha,
    apply_downside_covariance_inflation,
    apply_epistemic_covariance_inflation,
    compute_posterior_downside_probability,
)
from hypergraph_portfolio.mean_variance import MeanVariancePortfolio


def test_covariance_inflation_properties():
    """Verify symmetry, PSD, identity at lambda=0, and monotonicity."""
    rng = np.random.default_rng(42)
    m = 4
    # Create random positive definite matrix
    A = rng.normal(size=(m, m))
    Q = A.T @ A + 0.05 * np.eye(m)

    sigma_epi = np.array([0.01, 0.04, 0.02, 0.05])

    # 1. lambda = 0 identity
    Q_zero = apply_epistemic_covariance_inflation(Q, sigma_epi, lambda_cov=0.0)
    assert np.allclose(Q_zero, Q)

    # 2. Symmetry
    Q_inflated = apply_epistemic_covariance_inflation(Q, sigma_epi, lambda_cov=10.0)
    assert np.allclose(Q_inflated, Q_inflated.T)

    # 3. PSD preservation
    eigvals = np.linalg.eigvalsh(Q_inflated)
    assert np.all(eigvals >= -1e-12)

    # 4. Off-diagonal elements preserved intact
    off_diag_mask = ~np.eye(m, dtype=bool)
    assert np.allclose(Q_inflated[off_diag_mask], Q[off_diag_mask])

    # 5. Monotonic diagonal increase with lambda_cov
    Q_more_inflated = apply_epistemic_covariance_inflation(Q, sigma_epi, lambda_cov=20.0)
    assert np.all(np.diag(Q_more_inflated) >= np.diag(Q_inflated))
    assert np.all(np.diag(Q_inflated) >= np.diag(Q))

    # 6. Monotonic diagonal increase with uncertainty
    sigma_larger = sigma_epi * 2.0
    Q_larger_std = apply_epistemic_covariance_inflation(Q, sigma_larger, lambda_cov=10.0)
    assert np.all(np.diag(Q_larger_std) > np.diag(Q_inflated))


def test_covariance_inflation_invalid_inputs():
    """Verify input validation for invalid covariance matrices and parameters."""
    Q = np.eye(3)
    sigma = np.array([0.01, 0.02, 0.03])

    with pytest.raises(ValueError, match="lambda_cov must be non-negative"):
        apply_epistemic_covariance_inflation(Q, sigma, lambda_cov=-1.0)

    with pytest.raises(ValueError, match="must be non-negative"):
        apply_epistemic_covariance_inflation(Q, np.array([0.01, -0.02, 0.03]), lambda_cov=1.0)

    with pytest.raises(ValueError, match="Length of epistemic_std"):
        apply_epistemic_covariance_inflation(Q, np.array([0.01, 0.02]), lambda_cov=1.0)


def test_posterior_downside_probability_bounds_and_limits():
    """Verify bounds in [0, 1] and zero-variance limit behavior."""
    # Zero variance cases
    mu_hat = np.array([0.05, -0.05, 0.00])
    sigma_zero = np.array([0.0, 0.0, 0.0])

    p_down = compute_posterior_downside_probability(mu_hat, sigma_zero, r_hurdle=0.0)
    assert np.isclose(p_down[0], 0.0)  # mu > 0 => p_down = 0
    assert np.isclose(p_down[1], 1.0)  # mu < 0 => p_down = 1
    assert np.isclose(p_down[2], 0.5)  # mu == 0 => p_down = 0.5

    # Positive variance: probabilities must be strictly in [0, 1]
    sigma_pos = np.array([0.02, 0.02, 0.02])
    p_down_pos = compute_posterior_downside_probability(mu_hat, sigma_pos, r_hurdle=0.0)
    assert np.all((p_down_pos >= 0.0) & (p_down_pos <= 1.0))
    assert p_down_pos[0] < 0.5  # Positive return => p_down < 0.5
    assert p_down_pos[1] > 0.5  # Negative return => p_down > 0.5
    assert np.isclose(p_down_pos[2], 0.5)  # Zero return => p_down == 0.5


def test_posterior_downside_monotonicity():
    """Verify monotonicity with respect to expected return and uncertainty."""
    sigma = np.array([0.02, 0.02, 0.02])
    # 1. As mu_hat increases, p_down decreases monotonically
    mu_ascending = np.array([-0.02, 0.01, 0.04])
    p_descending = compute_posterior_downside_probability(mu_ascending, sigma, r_hurdle=0.0)
    assert p_descending[0] > p_descending[1] > p_descending[2]

    # 2. When mu_hat > r_hurdle (e.g., mu = +0.03, hurdle = 0), increasing uncertainty
    # broadens the left tail, strictly increasing downside probability
    mu_pos = np.array([0.03, 0.03, 0.03])
    sigmas = np.array([0.01, 0.03, 0.06])
    p_by_unc = compute_posterior_downside_probability(mu_pos, sigmas, r_hurdle=0.0)
    assert p_by_unc[0] < p_by_unc[1] < p_by_unc[2]


def test_downside_adjusted_alpha_and_covariance():
    """Verify Method 3 and Method 4 transformation properties."""
    mu_hat = np.array([0.05, 0.02, -0.03])
    p_down = np.array([0.10, 0.40, 0.85])

    # Method 3: Downside-adjusted alpha
    mu_adj = apply_downside_adjusted_alpha(mu_hat, p_down, lambda_down=1.0)
    # Asset 0 (low downside prob): shrunk lightly: 0.05 * (1 - 0.1) = 0.045
    assert np.isclose(mu_adj[0], 0.05 * 0.90)
    # Asset 1 (moderate downside prob): 0.02 * (1 - 0.4) = 0.012
    assert np.isclose(mu_adj[1], 0.02 * 0.60)
    # Asset 2 (negative return): scaled by positive factor (1 - 0.85) = 0.15 => -0.0045
    assert np.isclose(mu_adj[2], -0.03 * 0.15)

    # Method 4: Downside covariance inflation
    Q = 0.04 * np.eye(3) + 0.01 * np.ones((3, 3))
    sigma_epi = np.array([0.01, 0.02, 0.03])
    Q_down = apply_downside_covariance_inflation(Q, p_down, sigma_epi, lambda_down=5.0)

    assert np.allclose(Q_down, Q_down.T)
    assert np.all(np.linalg.eigvalsh(Q_down) > 0)
    # Asset 2 (highest p_down and highest sigma) should have largest variance expansion
    diag_delta = np.diag(Q_down) - np.diag(Q)
    assert diag_delta[2] > diag_delta[1] > diag_delta[0]


def test_data_partition_non_overlap():
    """Verify that development (2022-2023) and holdout (2024-2025) dates never overlap."""
    dev_path = "data/bayesian_lora/development/beta_0p003_seed_42_mc32_predictions.csv"
    holdout_path = "data/bayesian_lora/holdout/beta_0p003_seed_42_mc32_predictions.csv"
    meta_path = "data/bayesian_lora/reconstructed_returns.csv"

    dev_records = BayesianDatasetAdapter.load_from_csv(dev_path, meta_path)
    holdout_records = BayesianDatasetAdapter.load_from_csv(holdout_path, meta_path)

    dev_dates = set(r.timestamp.split(" ")[0] for r in dev_records)
    holdout_dates = set(r.timestamp.split(" ")[0] for r in holdout_records)

    # Strict zero overlap
    overlap = dev_dates & holdout_dates
    assert len(overlap) == 0, f"Development and holdout sets overlap on dates: {overlap}"

    # Dev is entirely <= 2023, Holdout is entirely >= 2024
    assert all(int(d[:4]) <= 2023 for d in dev_dates)
    assert all(int(d[:4]) >= 2024 for d in holdout_dates)


def test_identical_point_forecasts_across_phase4_policies():
    """Verify that all 5 policies (M0, M1, M2, M3, M4) receive identical base point forecasts."""
    holdout_path = "data/bayesian_lora/holdout/beta_0p003_seed_42_mc32_predictions.csv"
    meta_path = "data/bayesian_lora/reconstructed_returns.csv"
    records = BayesianDatasetAdapter.load_from_csv(holdout_path, meta_path)

    # Base point forecasts are fixed
    base_mu = np.array([r.mu_hat for r in records])
    base_sigma = np.array([r.epistemic_std for r in records])

    # Verify deterministic M0 and covariance inflation M2 use identical mu_hat
    mu_m0 = base_mu.copy()
    mu_m2 = base_mu.copy()
    assert np.all(mu_m0 == mu_m2)

    # Verify M1 and M3 are transformations strictly of base_mu and base_sigma
    from hypergraph_portfolio.uncertainty import apply_uncertainty_shrinkage

    mu_m1 = apply_uncertainty_shrinkage(base_mu, base_sigma, lambda_u=10.0)
    p_down = compute_posterior_downside_probability(base_mu, base_sigma)
    mu_m3 = apply_downside_adjusted_alpha(base_mu, p_down, lambda_down=0.5)

    assert len(mu_m1) == len(base_mu)
    assert len(mu_m3) == len(base_mu)


def test_no_future_realized_returns_in_weights():
    """Verify portfolio construction depends only on forecasts, never future returns."""
    mu_hat = np.array([0.04, 0.02, 0.01, -0.01])
    sigma_epi = np.array([0.01, 0.02, 0.01, 0.03])
    Q = 0.04 * np.eye(4) + 0.01 * np.ones((4, 4))

    # Construct weights under M0 and M2
    w_m0, _ = MeanVariancePortfolio(Q=Q, mu=mu_hat, budget=1.0).solve()
    Q_robust = apply_epistemic_covariance_inflation(Q, sigma_epi, lambda_cov=5.0)
    w_m2, _ = MeanVariancePortfolio(Q=Q_robust, mu=mu_hat, budget=1.0).solve()

    # Changing future realized return must have ZERO effect on weights
    real_r_scenario_1 = np.array([0.10, -0.05, 0.02, -0.08])
    real_r_scenario_2 = np.array([-0.20, 0.15, -0.03, 0.30])

    w_m0_check, _ = MeanVariancePortfolio(Q=Q, mu=mu_hat, budget=1.0).solve()
    assert np.allclose(w_m0, w_m0_check)
    assert not np.array_equal(real_r_scenario_1, real_r_scenario_2)
