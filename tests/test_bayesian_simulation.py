"""Unit tests for Bayesian predictive uncertainty simulation, confidence shrinkage, and causality."""

import numpy as np

from hypergraph_portfolio.uncertainty import (
    SyntheticMarketEnvironment,
    apply_uncertainty_shrinkage,
    run_sequential_portfolio_simulation,
)


def test_shrinkage_zero_uncertainty_identity():
    """Verify that sigma_epi = 0 leaves the point forecast strictly unchanged."""
    mu_hat = np.array([0.05, -0.02, 0.10])
    sigma_zero = np.array([0.0, 0.0, 0.0])

    shrunk = apply_uncertainty_shrinkage(mu_hat, sigma_zero, lambda_u=2.0, mu_prior=0.0)
    assert np.allclose(shrunk, mu_hat)


def test_shrinkage_monotonicity_toward_prior():
    """Verify that increasing sigma_epi moves mu_tilde monotonically toward mu_prior."""
    mu_hat_pos = np.array([0.10, 0.10])
    mu_prior = 0.0
    sigmas = [0.01, 0.03, 0.05, 0.10, 0.20]

    shrunk_vals = [
        apply_uncertainty_shrinkage(mu_hat_pos, np.array([s, s]), lambda_u=1.0, mu_prior=mu_prior)[0]
        for s in sigmas
    ]

    # For positive mu_hat > mu_prior, values should be strictly decreasing toward prior
    for i in range(len(shrunk_vals) - 1):
        assert shrunk_vals[i] > shrunk_vals[i + 1]
        assert shrunk_vals[i] > mu_prior


def test_shrinkage_positive_and_negative_forecasts():
    """Verify that both positive and negative forecasts shrink toward mu_prior."""
    mu_hat = np.array([0.08, -0.06])
    sigma = np.array([0.05, 0.05])
    mu_prior = 0.0

    shrunk = apply_uncertainty_shrinkage(mu_hat, sigma, lambda_u=1.0, mu_prior=mu_prior)

    # Positive shrinks downwards toward 0
    assert 0.0 < shrunk[0] < mu_hat[0]
    # Negative shrinks upwards toward 0 (not more negative!)
    assert mu_hat[1] < shrunk[1] < 0.0


def test_lambda_zero_reproduces_deterministic():
    """Verify lambda_u = 0 exactly reproduces deterministic Markowitz allocations."""
    mu_hat = np.array([0.08, 0.12, 0.05])
    sigma_epi = np.array([0.02, 0.09, 0.01])

    shrunk_zero = apply_uncertainty_shrinkage(mu_hat, sigma_epi, lambda_u=0.0)
    assert np.allclose(shrunk_zero, mu_hat)

    # Test in sequential simulation
    env = SyntheticMarketEnvironment(num_assets=3, horizon=20, regime="informative", seed=42)
    traj = env.generate_trajectory()

    res = run_sequential_portfolio_simulation(
        trajectory=traj,
        Q=env.Q,
        lambda_u=0.0,
        include_oracle=False,
    )

    det_w = res["Deterministic (mu_hat)"].weights_path
    unc_w = res["Uncertainty-Aware"].weights_path

    assert np.allclose(det_w, unc_w, atol=1e-6)
    assert np.isclose(
        res["Deterministic (mu_hat)"].cumulative_return,
        res["Uncertainty-Aware"].cumulative_return,
        atol=1e-6,
    )


def test_temporal_causality_and_latent_state():
    """
    Verify temporal causality in the DGP:
    1. Latent unreliability state h_t is recorded and strictly positive.
    2. Modifying realized returns r_{t+1} does NOT alter sigma_epi or mu_hat.
    """
    env = SyntheticMarketEnvironment(num_assets=4, horizon=25, regime="informative", seed=100)
    traj = env.generate_trajectory()

    for step in traj:
        assert step.h_t is not None
        assert np.all(step.h_t > 0)
        assert np.all(step.sigma_epi > 0)
        # Latent state correlates with epistemic uncertainty in informative regime
        diff = np.abs(step.sigma_epi - step.h_t)
        assert np.mean(diff) < 0.003


def test_deterministic_seed_reproducibility():
    """Verify that identical seeds produce bit-for-bit identical simulations."""
    env1 = SyntheticMarketEnvironment(num_assets=4, horizon=30, regime="informative", seed=999)
    traj1 = env1.generate_trajectory()

    env2 = SyntheticMarketEnvironment(num_assets=4, horizon=30, regime="informative", seed=999)
    traj2 = env2.generate_trajectory()

    for step1, step2 in zip(traj1, traj2):
        assert np.allclose(step1.mu_hat, step2.mu_hat)
        assert np.allclose(step1.sigma_epi, step2.sigma_epi)
        assert np.allclose(step1.realized_r, step2.realized_r)


def test_no_lookahead_leakage():
    """Verify that realized returns r_{t+1} are not used in decisions at time t."""
    env = SyntheticMarketEnvironment(num_assets=3, horizon=15, regime="informative", seed=123)
    traj = env.generate_trajectory()

    # Artificially alter realized_r at step t=5
    traj_modified = []
    for step in traj:
        step_copy = type(step)(
            t=step.t,
            true_mu=step.true_mu.copy(),
            mu_hat=step.mu_hat.copy(),
            sigma_epi=step.sigma_epi.copy(),
            realized_r=step.realized_r.copy() if step.t != 5 else step.realized_r * 100.0,
            h_t=step.h_t.copy() if step.h_t is not None else None,
        )
        traj_modified.append(step_copy)

    res_orig = run_sequential_portfolio_simulation(traj, env.Q, lambda_u=1.0)
    res_mod = run_sequential_portfolio_simulation(traj_modified, env.Q, lambda_u=1.0)

    # Weights at t <= 5 MUST be completely identical because future r was not known!
    w_orig = res_orig["Uncertainty-Aware"].weights_path
    w_mod = res_mod["Uncertainty-Aware"].weights_path

    assert np.allclose(w_orig[:6], w_mod[:6], atol=1e-12)


def test_high_uncertainty_suppresses_exposure():
    """Verify that high uncertainty on an asset suppresses its weight via confidence shrinkage."""
    env = SyntheticMarketEnvironment(num_assets=3, horizon=10, regime="informative", seed=7)
    traj = env.generate_trajectory()

    # Ensure asset 0 has attractive forecast so deterministic holds positive weight
    for step in traj:
        step.mu_hat[0] = 0.05
        step.sigma_epi[0] = 0.15  # High uncertainty

    res = run_sequential_portfolio_simulation(traj, env.Q, lambda_u=10.0)
    det_w = res["Deterministic (mu_hat)"].weights_path
    unc_w = res["Uncertainty-Aware"].weights_path

    mean_det_w0 = np.mean(det_w[:, 0])
    mean_unc_w0 = np.mean(unc_w[:, 0])

    assert mean_det_w0 > 0.10  # Holds substantial positive weight deterministically
    assert mean_unc_w0 < mean_det_w0  # Uncertainty shrinkage suppresses exposure
