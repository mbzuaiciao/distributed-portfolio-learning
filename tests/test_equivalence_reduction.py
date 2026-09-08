"""Unit tests for Phase 2 mathematical equivalence reduction and audit tools."""

import cvxpy as cp
import numpy as np
import pytest

from hypergraph_portfolio.consensus import (
    build_true_lifted_system,
    check_consensus,
    compare_matrices_term_by_term,
    lift_centralized_to_distributed,
    reduce_distributed_objective,
)
from hypergraph_portfolio.hypergraph import HypergraphPortfolio
from hypergraph_portfolio.mean_variance import MeanVariancePortfolio


@pytest.fixture
def three_asset_setup():
    Q = np.array([
        [4.1294, 1.4527, 0.9273],
        [1.4527, 4.4150, 1.1068],
        [0.9273, 1.1068, 3.3435],
    ])
    mu = np.array([0.30, 0.25, 0.10])
    c = 1.0

    hp = HypergraphPortfolio(
        asset_names=["x1", "x2", "x3"],
        subportfolios=[["x1", "x2", "x3"], ["x1", "x2"], ["x1", "x3"]],
    )

    sub_Q = [Q, Q[:2, :2], Q[[0, 2]][:, [0, 2]]]
    sub_mu = [mu, mu[:2], mu[[0, 2]]]
    sub_c = [1.0, 0.6, 0.4]
    sub_a = [0.33, 0.33, 0.33]

    return hp, Q, mu, c, sub_Q, sub_mu, sub_c, sub_a


def test_consensus_characterization(three_asset_setup):
    """A1: Verify Lx = 0 iff x lies in Range(E)."""
    hp, Q, mu, c, sub_Q, sub_mu, sub_c, sub_a = three_asset_setup

    # Any vector of form x = E z must satisfy Lx = 0
    rng = np.random.default_rng(42)
    for _ in range(5):
        z_random = rng.standard_normal(3)
        x_cons = hp.E @ z_random
        is_cons, err = check_consensus(x_cons, hp)
        assert is_cons
        assert np.isclose(err, 0.0, atol=1e-12)

    # Any vector not in Range(E) must have Lx != 0
    for _ in range(5):
        x_rand = rng.standard_normal(7)
        # Project onto orthogonal complement of Range(E), which is Range(L)
        x_orth = hp.L @ x_rand
        if np.linalg.norm(x_orth) > 1e-4:
            is_cons, err = check_consensus(x_orth, hp)
            assert not is_cons
            assert err > 1e-4


def test_reduce_distributed_objective_matches_direct_eval(three_asset_setup):
    """A2: Verify F(Ez) = z^T Q_eff z - mu_eff^T z + c_eff exactly matches sum a_i f_i(x_i)."""
    hp, Q, mu, c, sub_Q, sub_mu, sub_c, sub_a = three_asset_setup

    Q_eff_z, mu_eff_z, c_eff = reduce_distributed_objective(
        hp, sub_Q, sub_mu, sub_c, sub_a, parameterization="local_copy"
    )

    z_test = np.array([0.15, 0.11, 0.16])
    x_test = hp.E @ z_test  # shape (7,)

    # Direct evaluation
    direct_val = 0.0
    for i in range(hp.num_subportfolios):
        sl = hp.get_subportfolio_slice(i)
        xi = x_test[sl]
        fi = xi @ sub_Q[i] @ xi - sub_mu[i] @ xi + sub_c[i]
        direct_val += sub_a[i] * fi

    # Reduced evaluation
    reduced_val = z_test @ Q_eff_z @ z_test - mu_eff_z @ z_test + c_eff

    assert np.isclose(direct_val, reduced_val, atol=1e-12)


def test_weight_parameterization_reduction(three_asset_setup):
    """Verify weight parameterization w = E^T x matches reduction."""
    hp, Q, mu, c, sub_Q, sub_mu, sub_c, sub_a = three_asset_setup

    Q_eff_w, mu_eff_w, c_eff = reduce_distributed_objective(
        hp, sub_Q, sub_mu, sub_c, sub_a, parameterization="weight"
    )

    w_test = np.array([0.45, 0.22, 0.33])
    x_test = lift_centralized_to_distributed(w_test, hp)

    direct_val = 0.0
    for i in range(hp.num_subportfolios):
        sl = hp.get_subportfolio_slice(i)
        xi = x_test[sl]
        fi = xi @ sub_Q[i] @ xi - sub_mu[i] @ xi + sub_c[i]
        direct_val += sub_a[i] * fi

    reduced_val = w_test @ Q_eff_w @ w_test - mu_eff_w @ w_test + c_eff
    assert np.isclose(direct_val, reduced_val, atol=1e-12)


def test_term_by_term_comparison_discrepancy(three_asset_setup):
    """A3: Confirm Q_eff != Q and measure Frobenius norm difference."""
    hp, Q, mu, c, sub_Q, sub_mu, sub_c, sub_a = three_asset_setup

    Q_eff_w, mu_eff_w, c_eff = reduce_distributed_objective(
        hp, sub_Q, sub_mu, sub_c, sub_a, parameterization="weight"
    )

    comp = compare_matrices_term_by_term(Q_eff_w, Q, mu_eff_w, mu, c_eff, c)

    # Q_eff is definitely not Q
    assert comp["frobenius_diff_Q"] > 1.0
    assert comp["relative_diff_Q"] > 0.5
    # mu_eff is roughly 1/3 of mu
    assert np.allclose(comp["entry_ratios_mu"], 0.33, atol=0.01)


def test_true_lifted_equivalence_to_centralized(three_asset_setup):
    """A4: Verify that true lifted problem has Q_eff == Q and achieves identical minimum."""
    hp, Q, mu, c, sub_Q, sub_mu, sub_c, sub_a = three_asset_setup

    # Centralized solution
    cent_model = MeanVariancePortfolio(Q=Q, mu=mu, c=c, budget=1.0)
    w_cent, f_cent = cent_model.solve()

    # True lifted formulation
    Q_lifted, mu_lifted, c_lifted = build_true_lifted_system(hp, Q, mu, c)

    # Solve true lifted QP in R^p
    x_var = cp.Variable(hp.total_nodes)
    z_var = hp.E.T @ x_var  # Total asset weight

    obj = cp.Minimize(cp.quad_form(x_var, Q_lifted) - mu_lifted @ x_var + c_lifted)
    constraints = [
        cp.sum(z_var) == 1.0,
        hp.L @ x_var == 0,
        x_var >= 0,
    ]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    w_lifted_opt = hp.aggregate_to_assets(x_var.value)
    f_lifted_opt = prob.value

    # Must be mathematically identical
    assert np.isclose(f_cent, f_lifted_opt, atol=1e-5)
    assert np.allclose(w_cent, w_lifted_opt, atol=1e-4)
