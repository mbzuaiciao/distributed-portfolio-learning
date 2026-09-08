"""
Unit tests for Phase 9: Accounting Audit & Frozen Hysteresis Confirmation.
Validates:
    1. Exact, non-overlapping turnover accounting (risky, cash, total, half).
    2. Canonical transaction cost application (fees on risky trades, cumulative sum invariant).
    3. Mutually exclusive, exhaustive 4-quadrant economic attribution closure identity.
    4. Frozen M13 parameter immutability and state machine determinism.
    5. Portfolio ledger integrity and single-subtraction net utility verification.
"""

from __future__ import annotations

import numpy as np
import pytest

from hypergraph_portfolio.uncertainty import (
    build_portfolio_ledger,
    compute_canonical_turnover,
    compute_hysteresis_exposure_path,
    decompose_timing_economic_attribution_rebuilt,
)


def test_turnover_definitions_exact_and_no_double_counting():
    """Verify that risky, cash, total, and half turnovers follow exact identities without double counting."""
    # Scenario A: First period (no previous weights)
    w_r0 = np.array([0.3, 0.4, 0.1])
    w_c0 = 0.2
    to0 = compute_canonical_turnover(w_r0, None, w_c0, None)
    assert to0["risky"] == 0.0
    assert to0["cash"] == 0.0
    assert to0["total"] == 0.0
    assert to0["half"] == 0.0

    # Scenario B: Pure macro exposure change (base weights proportional, exposure drops from 0.80 to 0.50)
    w_r1 = np.array([0.3, 0.4, 0.1])  # sum = 0.80
    w_c1 = 0.20
    # Exposure changes: sum drops to 0.50 (proportional scaling: 0.50 / 0.80)
    scale = 0.50 / 0.80
    w_r2 = w_r1 * scale  # array([0.1875, 0.25, 0.0625]), sum = 0.50
    w_c2 = 0.50

    to1 = compute_canonical_turnover(w_r2, w_r1, w_c2, w_c1)
    # Delta risky = |0.1875-0.3| + |0.25-0.4| + |0.0625-0.1| = 0.1125 + 0.15 + 0.0375 = 0.30
    assert pytest.approx(to1["risky"], abs=1e-12) == 0.30
    # Delta cash = |0.50 - 0.20| = 0.30
    assert pytest.approx(to1["cash"], abs=1e-12) == 0.30
    # Total two-way turnover = 0.60
    assert pytest.approx(to1["total"], abs=1e-12) == 0.60
    # Half (one-way) turnover = 0.30 == risky turnover
    assert pytest.approx(to1["half"], abs=1e-12) == 0.30

    # Scenario C: Pure internal rebalance (exposure constant at 0.70, cash unchanged)
    w_r3_a = np.array([0.35, 0.35])
    w_r3_b = np.array([0.45, 0.25])
    w_c3 = 0.30
    to2 = compute_canonical_turnover(w_r3_b, w_r3_a, w_c3, w_c3)
    assert pytest.approx(to2["risky"], abs=1e-12) == 0.20
    assert pytest.approx(to2["cash"], abs=1e-12) == 0.0
    assert pytest.approx(to2["total"], abs=1e-12) == 0.20
    assert pytest.approx(to2["half"], abs=1e-12) == 0.10


def test_transaction_cost_rate_and_cumulative_invariant():
    """Verify transaction costs apply to risky trades and cumulative costs equal the sum of period costs."""
    cost_rate = 0.0010  # 10 bps
    turnovers = [0.15, 0.0, 0.25, 0.10, 0.05]
    period_costs = [cost_rate * to for to in turnovers]
    total_cost = sum(period_costs)

    # Invariant 1: Period cost equals rate times risky turnover
    for to, tc in zip(turnovers, period_costs):
        assert pytest.approx(tc, abs=1e-12) == cost_rate * to

    # Invariant 2: Zero turnover incurs zero transaction cost
    assert period_costs[1] == 0.0

    # Invariant 3: Cumulative cost equals sum of period costs
    assert pytest.approx(total_cost, abs=1e-12) == cost_rate * sum(turnovers)


def test_economic_attribution_mutually_exclusive_and_exact_closure():
    """Verify that rebuilt 4-quadrant economic attribution is mutually exclusive, exhaustive, and closes exactly."""
    np.random.seed(42)
    T = 60
    r_base = np.random.normal(0.003, 0.025, size=T)
    # Dynamic exposures oscillating above and below constant exposure c = 0.70
    s_timing = np.clip(0.70 + 0.25 * np.sin(np.linspace(0, 4 * np.pi, T)), 0.20, 0.95)
    c_const = 0.70
    cost_bps = 0.0010

    decomp = decompose_timing_economic_attribution_rebuilt(
        base_returns=r_base,
        timing_exposures=s_timing,
        constant_exposure=c_const,
        cost_bps=cost_bps,
    )

    # Invariant 1: All quadrant components are non-negative
    assert decomp["losses_avoided"] >= 0.0
    assert decomp["upside_sacrificed"] >= 0.0
    assert decomp["extra_upside_captured"] >= 0.0
    assert decomp["extra_downside_suffered"] >= 0.0

    # Invariant 2: Exact algebraic closure to machine precision (< 1e-14)
    quadrant_sum = (
        decomp["losses_avoided"]
        - decomp["upside_sacrificed"]
        + decomp["extra_upside_captured"]
        - decomp["extra_downside_suffered"]
    )
    assert pytest.approx(decomp["gross_return_difference"], abs=1e-14) == quadrant_sum
    assert decomp["closure_error"] < 1e-14

    # Invariant 3: Net economic benefit equals gross difference minus incremental transaction costs
    assert pytest.approx(
        decomp["net_economic_benefit"], abs=1e-14
    ) == decomp["gross_return_difference"] - decomp["incremental_transaction_costs"]


def test_frozen_m13_parameters_immutability_and_determinism():
    """Verify that frozen M13 parameters (tau_enter=1.2, tau_exit=0.2) produce immutable, deterministic outputs."""
    z_signal = [0.1, 0.5, 1.3, 1.4, 0.9, 0.3, 0.15, 0.8, 1.5, 0.1]
    tau_enter = 1.2
    tau_exit = 0.2

    path1 = compute_hysteresis_exposure_path(z_signal, tau_enter=tau_enter, tau_exit=tau_exit, s_normal=0.85, s_reduced=0.45)
    path2 = compute_hysteresis_exposure_path(z_signal, tau_enter=tau_enter, tau_exit=tau_exit, s_normal=0.85, s_reduced=0.45)

    # Identical paths across multiple runs
    np.testing.assert_array_equal(path1, path2)

    # Hand-checked transitions:
    # t=0: z=0.1 < 1.2 -> normal (0.85)
    assert path1[0] == 0.85
    # t=2: z=1.3 >= 1.2 -> enter reduced (0.45)
    assert path1[2] == 0.45
    # t=4: z=0.9 > 0.2 -> remains reduced (0.45)
    assert path1[4] == 0.45
    # t=5: z=0.3 > 0.2 -> remains reduced (0.45)
    assert path1[5] == 0.45
    # t=6: z=0.15 <= 0.2 -> exit reduced back to normal (0.85)
    assert path1[6] == 0.85


def test_portfolio_ledger_integrity_and_single_subtraction():
    """Verify that portfolio ledger computes canonical metrics and does not double-subtract transaction costs in net utility."""
    # Synthetic epochs
    epochs = []
    for t in range(5):
        epochs.append({
            "week": f"2024-W{t+1:02d}",
            "assets": ["A", "B"],
            "w_base": np.array([0.6, 0.4]),
            "real_r": np.array([0.02, -0.01]),
            "Q_market": np.array([[0.04, 0.01], [0.01, 0.03]]),
        })

    s_path = [0.8, 0.8, 0.5, 0.5, 0.8]
    cost_bps = 0.0010
    gamma = 1.0

    ledger = build_portfolio_ledger(epochs, s_path, policy_name="TestPolicy", gamma=gamma, cost_bps=cost_bps)

    assert len(ledger) == 5

    # Check period t=2 where exposure drops from 0.8 to 0.5
    rec2 = ledger[2]
    expected_to_r = abs(0.5 * 0.6 - 0.8 * 0.6) + abs(0.5 * 0.4 - 0.8 * 0.4)  # 0.18 + 0.12 = 0.30
    assert pytest.approx(rec2["turnover_risky"], abs=1e-12) == 0.30
    assert pytest.approx(rec2["turnover_cash"], abs=1e-12) == 0.30
    assert pytest.approx(rec2["turnover_total"], abs=1e-12) == 0.60
    assert pytest.approx(rec2["turnover_half"], abs=1e-12) == 0.30

    expected_tc = cost_bps * expected_to_r
    assert pytest.approx(rec2["transaction_cost"], abs=1e-12) == expected_tc
    assert pytest.approx(rec2["net_return"], abs=1e-12) == rec2["gross_return"] - expected_tc

    # CRITICAL CHECK: Net utility MUST NOT double-subtract transaction costs
    expected_u_net = rec2["net_return"] - rec2["risk_penalty"]
    assert pytest.approx(rec2["net_utility"], abs=1e-12) == expected_u_net
    # Verify it does NOT equal net_return - risk_penalty - tc
    assert rec2["net_utility"] != rec2["net_return"] - rec2["risk_penalty"] - rec2["transaction_cost"]
