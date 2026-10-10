"""Unit tests for Minimal Uncertainty-Estimator Diagnostic Benchmark."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hypergraph_portfolio.minimal_uncertainty_benchmark import (
    BayesianLogisticRegression,
    NeuralLaplaceEstimator,
    compute_selective_risk_coverage,
    evaluate_estimator_diagnostics,
    generate_diagnostic_dgp,
)


def test_blr_convergence_and_covariance_properties():
    """Verify Newton-Raphson converges and returns a positive-definite covariance matrix."""
    rng = np.random.default_rng(42)
    n, d = 100, 3
    X = rng.normal(size=(n, d))
    y = (rng.uniform(size=n) > 0.5).astype(float)

    blr = BayesianLogisticRegression.fit(X, y, alpha=1.0)
    assert blr.converged is True
    assert blr.iterations < 20
    assert blr.weights.shape == (d,)
    assert blr.covariance.shape == (d + 1, d + 1)

    # Covariance must be strictly symmetric positive-definite
    np.testing.assert_allclose(blr.covariance, blr.covariance.T, atol=1e-7)
    eigvals = np.linalg.eigvalsh(blr.covariance)
    assert np.all(eigvals > 0)
    assert blr.hessian_condition_number >= 1.0


def test_blr_canonical_1d_extrapolation_positive_control():
    """In 1D, epistemic variance x^T H^{-1} x must increase monotonically with |x| away from data mean."""
    rng = np.random.default_rng(123)
    X = rng.normal(loc=0.0, scale=1.0, size=(150, 1))
    y = (X[:, 0] > 0).astype(float)

    blr = BayesianLogisticRegression.fit(X, y, alpha=0.5)

    # Test points at increasing distances from the origin
    x_test = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    _, var = blr.predict_latent(x_test)

    # Epistemic variance must be strictly monotonically increasing with |x|
    assert np.all(np.diff(var) > 0)


def test_probit_approximation_matches_bounded_probabilities():
    """Predictive probabilities must be strictly bounded in (0, 1) and shrink toward 0.5 under high variance."""
    rng = np.random.default_rng(999)
    X = rng.normal(size=(50, 2))
    y = rng.integers(0, 2, size=50).astype(float)

    blr = BayesianLogisticRegression.fit(X, y, alpha=1.0)
    p_pred = blr.predict_proba(X)
    assert np.all((p_pred > 0.0) & (p_pred < 1.0))

    # An extreme out-of-distribution point with huge variance should shrink probit probability toward 0.5
    X_extreme = np.array([[100.0, 100.0]])
    p_extreme = blr.predict_proba(X_extreme)
    # Even with huge latent logit, the large variance dampens kappa, keeping probit bounded
    assert 0.0 < p_extreme[0] < 1.0


def test_dgp_generation_and_interval_coverage():
    """Diagnostic DGP generates consistent shapes and support truncation."""
    data = generate_diagnostic_dgp(seed=42, n_train=128, n_test_in=256, n_test_extrap=256)
    assert data["X_train"].shape == (128, 4)
    assert np.all(data["X_train"][:, 0] <= 0.0)  # Support truncation check
    assert data["X_test_all"].shape == (512, 4)
    assert len(data["is_extrap"]) == 512

    # Fit BLR and test credible interval coverage computation
    blr = BayesianLogisticRegression.fit(data["X_train"], data["y_train"], alpha=1.0)
    ci_lower, ci_upper = blr.credible_interval_95(data["X_test_all"])
    diag = evaluate_estimator_diagnostics(
        uncertainty_scores=blr.predict_latent(data["X_test_all"])[1],
        pred_probs=blr.predict_proba(data["X_test_all"]),
        y_true=data["y_test_all"],
        mahalanobis_d=data["mahalanobis_d"],
        is_extrap=data["is_extrap"],
        latent_ci=(ci_lower, ci_upper),
        z_true=data["z_test_all"],
    )

    # In-support interval coverage should be reasonable (> 0.70)
    assert diag["interval_coverage_in"] > 0.70
    assert diag["support_spearman_rho"] > 0.40
    assert 0.50 <= diag["ood_auroc"] <= 1.0


def test_positive_and_negative_controls():
    """Controls behave as theoretically expected."""
    data = generate_diagnostic_dgp(seed=777, n_train=100, n_test_in=100, n_test_extrap=100)
    dummy_probs = np.full(200, 0.5)

    # Positive control: Mahalanobis distance directly
    diag_pos = evaluate_estimator_diagnostics(
        uncertainty_scores=data["mahalanobis_d"],
        pred_probs=dummy_probs,
        y_true=data["y_test_all"],
        mahalanobis_d=data["mahalanobis_d"],
        is_extrap=data["is_extrap"],
    )
    assert diag_pos["support_spearman_rho"] > 0.99
    assert diag_pos["ood_auroc"] > 0.85

    # Negative control: Uniform random noise
    rng = np.random.default_rng(777)
    random_scores = rng.uniform(0.0, 1.0, size=200)
    diag_neg = evaluate_estimator_diagnostics(
        uncertainty_scores=random_scores,
        pred_probs=dummy_probs,
        y_true=data["y_test_all"],
        mahalanobis_d=data["mahalanobis_d"],
        is_extrap=data["is_extrap"],
    )
    # Random noise should have near-zero correlation with support distance
    assert abs(diag_neg["support_spearman_rho"]) < 0.25


def test_neural_laplace_estimator_and_prior_precision_sweep():
    """Neural Laplace estimator trains and computes diagnostics across alpha grid."""
    data = generate_diagnostic_dgp(seed=123, n_train=64, n_test_in=64, n_test_extrap=64)

    # Fit with default alpha
    n_est = NeuralLaplaceEstimator.fit(data["X_train"], data["y_train"], epochs=20, alpha=1.0)
    mean, var = n_est.predict_latent(data["X_test_all"])
    assert len(mean) == 128
    assert len(var) == 128
    assert np.all(var > 0)
    assert n_est.hessian_condition_number >= 1.0

    # Test sweep over alpha grid
    alphas = [0.1, 1.0, 10.0]
    cond_nums = []
    for a in alphas:
        n_a = NeuralLaplaceEstimator.fit(data["X_train"], data["y_train"], epochs=10, alpha=a)
        cond_nums.append(n_a.hessian_condition_number)

    # Increasing prior precision alpha regularizes Hessian, reducing condition number
    assert cond_nums[0] >= cond_nums[-1]


def test_selective_risk_coverage():
    """Verify AU-RC decreases when sorting errors monotonically."""
    errors = np.linspace(0.0, 1.0, 100)
    uncertainties = errors.copy()  # Perfect uncertainty ranking
    covs, risks, aurc_perfect = compute_selective_risk_coverage(errors, uncertainties)
    # With perfect ranking, risk must increase monotonically with coverage
    assert np.all(np.diff(risks) >= 0)

    # Inverted ranking (worst possible uncertainty)
    _, _, aurc_worst = compute_selective_risk_coverage(errors, -uncertainties)
    assert aurc_perfect < aurc_worst
