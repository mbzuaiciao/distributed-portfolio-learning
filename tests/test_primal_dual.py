"""Unit tests for continuous-time primal-dual dynamics solver."""

import numpy as np
import pytest

from hypergraph_portfolio.hypergraph import HypergraphPortfolio
from hypergraph_portfolio.primal_dual import PrimalDualSolver


@pytest.fixture
def paper_three_asset_setup():
    Q = np.array([
        [4.1294, 1.4527, 0.9273],
        [1.4527, 4.4150, 1.1068],
        [0.9273, 1.1068, 3.3435],
    ])
    mu = np.array([0.3, 0.25, 0.1])
    sub_Q = [
        Q,
        Q[:2, :2],
        Q[[0, 2]][:, [0, 2]],
    ]
    sub_mu = [
        mu,
        mu[:2],
        mu[[0, 2]],
    ]
    sub_c = [1.0, 0.6, 0.4]
    sub_a = [0.33, 0.33, 0.33]

    hp = HypergraphPortfolio(
        asset_names=["x1", "x2", "x3"],
        subportfolios=[["x1", "x2", "x3"], ["x1", "x2"], ["x1", "x3"]],
    )
    return hp, sub_Q, sub_mu, sub_c, sub_a


def test_primal_dual_solver_convergence(paper_three_asset_setup):
    hp, sub_Q, sub_mu, sub_c, sub_a = paper_three_asset_setup
    solver = PrimalDualSolver(
        hypergraph=hp,
        sub_Q=sub_Q,
        sub_mu=sub_mu,
        sub_c=sub_c,
        sub_a=sub_a,
        budget=1.0,
        scale_gradient_by_weights=True,
    )

    x_opt, lam_opt, v_opt, history = solver.solve(
        dt=0.003,
        max_steps=20000,
        record_every=50,
    )

    # Check budget satisfaction
    assert np.isclose(np.sum(x_opt), 1.0, atol=1e-3)
    # Check non-negativity
    assert np.all(x_opt >= -1e-6)
    # Check consensus error norm
    consensus_err = np.linalg.norm(hp.L @ x_opt)
    assert consensus_err < 5e-4

    # Check Lyapunov function decreases overall
    v_start = history.lyapunov_trajectory[0]
    v_end = history.lyapunov_trajectory[-1]
    assert v_end < v_start
    assert v_end < 1e-3


def test_primal_dual_agrees_with_cvxpy(paper_three_asset_setup):
    hp, sub_Q, sub_mu, sub_c, sub_a = paper_three_asset_setup
    solver = PrimalDualSolver(
        hypergraph=hp,
        sub_Q=sub_Q,
        sub_mu=sub_mu,
        sub_c=sub_c,
        sub_a=sub_a,
        budget=1.0,
        scale_gradient_by_weights=True,
    )

    x_cvx, f_cvx, sub_f_cvx, _ = solver.solve_cvxpy()
    x_pd, _, _, _ = solver.solve(dt=0.003, max_steps=15000)

    assert np.allclose(x_pd, x_cvx, atol=2e-3)
    tot_f_pd, _ = solver.total_objective(x_pd)
    assert np.isclose(tot_f_pd, f_cvx, atol=2e-3)
