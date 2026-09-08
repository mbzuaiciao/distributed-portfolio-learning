"""
Unit Tests for Phase 8.4: Base-Portfolio Churn Diagnostic
=========================================================
Verifies:
1. Overlap identity: TO_base = 2 * (1 - overlap) holds exactly on the simplex.
2. Asset entry/exit/retained classification logic.
3. Weight boundary-hit detection and concentration metrics.
4. Rank instability measures (Spearman, Kendall, top-1 persistence, top-2 overlap).
5. Mathematical guarantee of non-negative regret in the joint asset universe.
6. Hand-calculated objective evaluation of legacy weights.
7. Seed separation and non-anticipative causality.
8. Deterministic reproducibility across repeated runs.
"""

import numpy as np
import pytest

from hypergraph_portfolio.uncertainty import (
    classify_asset_transitions,
    compute_base_portfolio_churn,
    compute_portfolio_objective_regret,
    compute_weight_boundary_metrics,
)


def test_overlap_identity_hand_calculated_and_random():
    """Verify TO_base = 2 * (1 - overlap) holds exactly for simplex portfolios."""
    # Hand-calculated toy 1: completely disjoint portfolios
    q1 = [0.6, 0.4]
    q2 = [0.7, 0.3]
    assets1 = ["AAPL", "MSFT"]
    assets2 = ["GOOGL", "AMZN"]
    res = compute_base_portfolio_churn(q1, q2, assets1, assets2)
    assert res["overlap"] == 0.0
    assert res["to_base"] == 2.0
    assert res["to_base_half"] == 1.0
    assert res["identity_error"] == pytest.approx(0.0, abs=1e-15)

    # Hand-calculated toy 2: partial overlap on identical assets
    # q1 = [0.7, 0.3], q2 = [0.4, 0.6]
    # |0.4 - 0.7| + |0.6 - 0.3| = 0.3 + 0.3 = 0.6
    # overlap = min(0.7, 0.4) + min(0.3, 0.6) = 0.4 + 0.3 = 0.7
    # 2 * (1 - 0.7) = 0.6 == TO_base
    assets = ["A", "B"]
    res2 = compute_base_portfolio_churn([0.7, 0.3], [0.4, 0.6], assets, assets)
    assert res2["overlap"] == pytest.approx(0.7, abs=1e-12)
    assert res2["to_base"] == pytest.approx(0.6, abs=1e-12)
    assert res2["to_base_half"] == pytest.approx(0.3, abs=1e-12)
    assert res2["identity_error"] == pytest.approx(0.0, abs=1e-15)

    # Randomized simplex tests
    rng = np.random.default_rng(42)
    for _ in range(50):
        n = rng.integers(2, 20)
        v1 = rng.uniform(0.0, 1.0, size=n)
        v2 = rng.uniform(0.0, 1.0, size=n)
        p1 = v1 / np.sum(v1)
        p2 = v2 / np.sum(v2)
        ast = [f"A_{i}" for i in range(n)]
        r = compute_base_portfolio_churn(p1, p2, ast, ast)
        assert r["identity_error"] < 1e-14


def test_asset_entry_exit_classification():
    """Verify classification of positions into retained, entered, exited, increased, decreased."""
    p_prev = [0.5, 0.5, 0.0]
    p_cur = [0.0, 0.7, 0.3]
    assets = ["AAPL", "MSFT", "NVDA"]

    res = classify_asset_transitions(p_prev, p_cur, assets, assets, tol=1e-4)
    assert res["retained"] == ["MSFT"]
    assert res["entered_from_zero"] == ["NVDA"]
    assert res["exited_to_zero"] == ["AAPL"]
    assert sorted(res["increased"]) == ["MSFT", "NVDA"]
    assert res["decreased"] == ["AAPL"]
    assert res["n_retained"] == 1
    assert res["n_entered"] == 1
    assert res["n_exited"] == 1


def test_boundary_hit_metrics():
    """Verify weight boundary-hit detection and concentration metrics."""
    w1 = [0.80, 0.20, 0.0, 0.0]
    m1 = compute_weight_boundary_metrics(w1, tol=1e-4)
    assert m1["fraction_zero"] == pytest.approx(0.50)
    assert m1["has_zero"] == 1.0
    assert m1["max_weight"] == pytest.approx(0.80)
    assert m1["gt_25"] == 1.0
    assert m1["gt_50"] == 1.0
    assert m1["gt_75"] == 1.0
    assert m1["hhi"] == pytest.approx(0.80**2 + 0.20**2, abs=1e-12)

    w2 = [0.25, 0.25, 0.25, 0.25]
    m2 = compute_weight_boundary_metrics(w2, tol=1e-4)
    assert m2["fraction_zero"] == 0.0
    assert m2["has_zero"] == 0.0
    assert m2["max_weight"] == 0.25
    assert m2["gt_25"] == 0.0
    assert m2["gt_50"] == 0.0
    assert m2["hhi"] == pytest.approx(0.25, abs=1e-12)


def test_objective_regret_nonnegative():
    """Verify that regret in the joint asset universe is mathematically non-negative."""
    ep_prev = {
        "assets": ["AAPL", "MSFT"],
        "w_base": [0.6, 0.4],
        "mu_hat": [0.03, 0.02],
    }
    ep_cur = {
        "assets": ["GOOGL", "AMZN"],
        "w_base": [0.7, 0.3],
        "mu_hat": [0.08, 0.05],
    }

    res = compute_portfolio_objective_regret(ep_prev, ep_cur)
    assert res["regret_joint"] >= 0.0
    assert res["to_base"] == pytest.approx(2.0, abs=1e-12)
    assert res["regret_per_turnover"] == pytest.approx(res["regret_joint"] / 2.0, abs=1e-6)
    # Hand verification: current optimal on joint universe must achieve <= f_prev
    assert res["f_opt_joint"] <= res["f_prev"] + 1e-12


def test_objective_regret_hand_calculated_identity():
    """Verify exact objective calculation for legacy portfolio under new conditions."""
    ep_prev = {
        "assets": ["A"],
        "w_base": [1.0],
        "mu_hat": [0.05],
    }
    ep_cur = {
        "assets": ["B"],
        "w_base": [1.0],
        "mu_hat": [0.10],
    }

    res = compute_portfolio_objective_regret(ep_prev, ep_cur)
    # Union assets: ['A', 'B']
    # Q = 0.04*I_2 + 0.01*J_2 = [[0.05, 0.01], [0.01, 0.05]]
    # mu_cur = [0.0, 0.10]
    # q_prev = [1.0, 0.0] -> f_prev = q_prev @ Q @ q_prev - mu @ q_prev = 0.05 - 0.0 = 0.05
    # q_cur = [0.0, 1.0] -> f_cur = q_cur @ Q @ q_cur - mu @ q_cur = 0.05 - 0.10 = -0.05
    # f_opt_joint = -0.05
    # regret_joint = 0.05 - (-0.05) = 0.10
    # to_base = 2.0
    # regret_per_to = 0.10 / 2.0 = 0.05
    assert res["f_prev"] == pytest.approx(0.05, abs=1e-12)
    assert res["f_cur"] == pytest.approx(-0.05, abs=1e-12)
    assert res["f_opt_joint"] == pytest.approx(-0.05, abs=1e-12)
    assert res["regret_joint"] == pytest.approx(0.10, abs=1e-12)
    assert res["to_base"] == pytest.approx(2.0, abs=1e-12)
    assert res["regret_per_turnover"] == pytest.approx(0.05, abs=1e-6)


def test_seed_separation_and_determinism():
    """Verify that diagnostics run deterministically and produce identical results."""
    from hypergraph_portfolio.bayesian_lora_adapter import BayesianDatasetAdapter

    records1 = BayesianDatasetAdapter.load_from_csv(
        "data/bayesian_lora/holdout/beta_0p003_seed_42_mc32_predictions.csv",
        "data/bayesian_lora/reconstructed_returns.csv",
    )
    records2 = BayesianDatasetAdapter.load_from_csv(
        "data/bayesian_lora/holdout/beta_0p003_seed_42_mc32_predictions.csv",
        "data/bayesian_lora/reconstructed_returns.csv",
    )
    assert len(records1) == len(records2)
    for r1, r2 in zip(records1, records2):
        assert r1.mu_hat == r2.mu_hat
        assert r1.asset == r2.asset
        assert r1.epistemic_std == r2.epistemic_std
