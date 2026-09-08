"""Unit tests for centralized mean-variance portfolio solver."""

import numpy as np

from hypergraph_portfolio.mean_variance import (
    MeanVariancePortfolio,
    project_to_simplex,
)


def test_project_to_simplex():
    # Vector already on simplex
    v = np.array([0.2, 0.5, 0.3])
    p = project_to_simplex(v, budget=1.0)
    assert np.allclose(p, v)
    assert np.isclose(np.sum(p), 1.0)

    # Negative and large entries
    v2 = np.array([-1.0, 2.0, 3.0])
    p2 = project_to_simplex(v2, budget=1.0)
    assert np.all(p2 >= 0)
    assert np.isclose(np.sum(p2), 1.0)
    assert p2[0] == 0.0  # Most negative entry should be zeroed


def test_three_asset_centralized_reproduction():
    """Verify exact match with paper Section 3.1."""
    Q = np.array([
        [4.1294, 1.4527, 0.9273],
        [1.4527, 4.4150, 1.1068],
        [0.9273, 1.1068, 3.3435],
    ])
    mu = np.array([0.3, 0.25, 0.1])
    c = 1.0

    model = MeanVariancePortfolio(Q=Q, mu=mu, c=c, budget=1.0)
    x_opt, f_opt = model.solve()

    # Paper reports: x* = (0.3217, 0.2567, 0.4217), f(x*) = 2.8408
    paper_x = np.array([0.3217, 0.2567, 0.4217])
    assert np.allclose(x_opt, paper_x, atol=1e-3)
    assert np.isclose(f_opt, 2.8408, atol=1e-3)
    assert np.isclose(np.sum(x_opt), 1.0)
    assert np.all(x_opt >= 0)


def test_cvxpy_solver_agreement():
    """Verify SciPy solver agrees with CVXPY benchmark."""
    Q = np.array([
        [2.0, 0.5],
        [0.5, 1.0],
    ])
    mu = np.array([0.1, 0.05])
    model = MeanVariancePortfolio(Q=Q, mu=mu, c=0.5, budget=1.0)

    x_scipy, f_scipy = model.solve()
    x_cvx, f_cvx = model.solve_cvxpy()

    assert np.allclose(x_scipy, x_cvx, atol=1e-5)
    assert np.isclose(f_scipy, f_cvx, atol=1e-5)
