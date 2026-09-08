"""
Unit tests for Phase 8.1: Accounting Reconciliation & Frozen M13 Walk-Forward Audit.
Covers:
    1. Canonical turnover terms (risky_weight_turnover, cash_sleeve_turnover, total_weight_turnover, half_weight_turnover).
    2. Transaction cost arithmetic, period sums, and savings reconciliation.
    3. 4-quadrant economic attribution closure on deterministic toy sequences with hand-calculated values.
    4. Frozen M13 parameters immutability and state transitions.
    5. Period-level source-of-truth ledger consistency.
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


def test_canonical_turnover_definitions_hand_calculated():
    """Verify hand-calculated turnover for risky weight, cash sleeve, total, and half turnover."""
    # Toy scenario with 2 risky assets and cash
    # Period 0: w_r = [0.40, 0.30], w_c = 0.30 (s = 0.70)
    w_r0 = np.array([0.40, 0.30])
    w_c0 = 0.30

    # Period 1: pure exposure reduction to s = 0.40 (proportional: [0.40*4/7, 0.30*4/7] = [0.228571, 0.171429]), w_c = 0.60
    w_r1 = np.array([0.22857142857, 0.17142857143])
    w_c1 = 0.60

    to_dict1 = compute_canonical_turnover(w_r1, w_r0, w_c1, w_c0)
    # Delta risky = |0.228571 - 0.40| + |0.171429 - 0.30| = 0.171429 + 0.128571 = 0.30
    assert pytest.approx(to_dict1["risky"], abs=1e-6) == 0.30
    # Delta cash = |0.60 - 0.30| = 0.30
    assert pytest.approx(to_dict1["cash"], abs=1e-6) == 0.30
    # Total turnover = 0.30 + 0.30 = 0.60
    assert pytest.approx(to_dict1["total"], abs=1e-6) == 0.60
    # Half turnover = 0.5 * 0.60 = 0.30
    assert pytest.approx(to_dict1["half"], abs=1e-6) == 0.30

    # Period 2: pure cross-sectional rebalance (exposure constant at s = 0.40, w_c = 0.60)
    # w_r shifts to [0.30, 0.10], cash unchanged
    w_r2 = np.array([0.30, 0.10])
    w_c2 = 0.60

    to_dict2 = compute_canonical_turnover(w_r2, w_r1, w_c2, w_c1)
    # Delta risky = |0.30 - 0.228571| + |0.10 - 0.171429| = 0.071429 + 0.071429 = 0.142857
    assert pytest.approx(to_dict2["risky"], abs=1e-6) == 0.142857
    # Delta cash = 0.0
    assert pytest.approx(to_dict2["cash"], abs=1e-6) == 0.0
    # Total turnover = 0.142857
    assert pytest.approx(to_dict2["total"], abs=1e-6) == 0.142857
    # Half turnover = 0.071429
    assert pytest.approx(to_dict2["half"], abs=1e-6) == 0.071429


def test_transaction_cost_arithmetic_and_savings_reconciliation():
    """Verify transaction cost formula, cumulative sum invariant, and savings calculation."""
    cost_rate = 0.0010  # 10 bps
    # Toy turnovers over 4 periods
    to_m9_risky = [0.30, 0.20, 0.40, 0.10]
    to_m13_risky = [0.30, 0.00, 0.00, 0.10]  # Hysteresis suppresses periods 1 & 2

    tc_m9 = [cost_rate * to for to in to_m9_risky]
    tc_m13 = [cost_rate * to for to in to_m13_risky]

    sum_tc_m9 = sum(tc_m9)
    sum_tc_m13 = sum(tc_m13)

    # Invariant: Period sum equals rate * sum(turnover)
    assert pytest.approx(sum_tc_m9, abs=1e-12) == cost_rate * sum(to_m9_risky)
    assert pytest.approx(sum_tc_m13, abs=1e-12) == cost_rate * sum(to_m13_risky)

    # Cost savings:
    cost_saved = sum_tc_m9 - sum_tc_m13
    assert pytest.approx(cost_saved, abs=1e-12) == cost_rate * (sum(to_m9_risky) - sum(to_m13_risky))
    # Hand check: (1.00 - 0.40) * 0.0010 = 0.60 * 0.0010 = 0.00060 (6.0 bps)
    assert pytest.approx(cost_saved, abs=1e-12) == 0.00060


def test_economic_attribution_toy_hand_calculated():
    """Verify 4-quadrant economic attribution with exact hand-calculated toy sequence."""
    # Matched constant exposure c = 0.70
    c_const = 0.70
    cost_bps = 0.0010

    # 4 periods, each engineered to trigger exactly ONE quadrant:
    # Period 0: Case A (Loss avoided): s = 0.45 (< 0.70), r = -0.04 (< 0)
    #   diff_s = -0.25, loss_avoided = 0.25 * 0.04 = +0.0100
    # Period 1: Case B (Upside sacrificed): s = 0.45 (< 0.70), r = +0.06 (> 0)
    #   diff_s = -0.25, upside_sacrificed = 0.25 * 0.06 = +0.0150
    # Period 2: Case C (Extra upside captured): s = 0.85 (> 0.70), r = +0.05 (> 0)
    #   diff_s = +0.15, extra_upside = 0.15 * 0.05 = +0.0075
    # Period 3: Case D (Extra downside suffered): s = 0.85 (> 0.70), r = -0.02 (< 0)
    #   diff_s = +0.15, extra_downside = 0.15 * 0.02 = +0.0030

    base_rets = [ -0.04, +0.06, +0.05, -0.02 ]
    timing_exposures = [ 0.45, 0.45, 0.85, 0.85 ]

    # Turnovers for incremental TC:
    # Timing policy adjusts exposure in period 2 from 0.45 to 0.85 (|delta s| = 0.40)
    to_timing = [0.0, 0.0, 0.40, 0.0]
    to_const = [0.0, 0.0, 0.0, 0.0]

    decomp = decompose_timing_economic_attribution_rebuilt(
        base_returns=base_rets,
        timing_exposures=timing_exposures,
        constant_exposure=c_const,
        cost_bps=cost_bps,
        timing_turnovers=to_timing,
        constant_turnovers=to_const,
    )

    # 1. Verify exact hand-calculated values
    assert pytest.approx(decomp["losses_avoided"], abs=1e-12) == 0.0100
    assert pytest.approx(decomp["upside_sacrificed"], abs=1e-12) == 0.0150
    assert pytest.approx(decomp["extra_upside_captured"], abs=1e-12) == 0.0075
    assert pytest.approx(decomp["extra_downside_suffered"], abs=1e-12) == 0.0030

    # 2. Hand-calculated gross difference:
    # Period 0: -0.25 * (-0.04) = +0.0100
    # Period 1: -0.25 * (+0.06) = -0.0150
    # Period 2: +0.15 * (+0.05) = +0.0075
    # Period 3: +0.15 * (-0.02) = -0.0030
    # Sum = 0.0100 - 0.0150 + 0.0075 - 0.0030 = -0.0005
    expected_gross = -0.0005
    assert pytest.approx(decomp["gross_return_difference"], abs=1e-12) == expected_gross

    # 3. Exact closure: quadrant sum == gross difference
    quadrant_sum = (
        decomp["losses_avoided"]
        - decomp["upside_sacrificed"]
        + decomp["extra_upside_captured"]
        - decomp["extra_downside_suffered"]
    )
    assert pytest.approx(quadrant_sum, abs=1e-14) == expected_gross
    assert decomp["closure_error"] < 1e-14

    # 4. Incremental transaction cost: 0.40 * 0.0010 = 0.00040
    assert pytest.approx(decomp["incremental_transaction_costs"], abs=1e-12) == 0.00040

    # 5. Net economic benefit: -0.0005 - 0.00040 = -0.00090
    assert pytest.approx(decomp["net_economic_benefit"], abs=1e-12) == -0.00090


def test_frozen_m13_walkforward_immutability():
    """Verify that frozen M13 parameters (tau_enter=1.2, tau_exit=0.2) remain immutable across walkforward folds."""
    tau_enter = 1.2
    tau_exit = 0.2
    s_normal = 0.85
    s_reduced = 0.45

    # Test two different synthetic signal histories (representing Fold 1 and Fold 2)
    sig_f1 = [0.5, 1.3, 0.8, 0.1, 0.4]
    sig_f2 = [0.2, 0.7, 1.4, 1.1, 0.15]

    path_f1 = compute_hysteresis_exposure_path(sig_f1, tau_enter=tau_enter, tau_exit=tau_exit, s_normal=s_normal, s_reduced=s_reduced)
    path_f2 = compute_hysteresis_exposure_path(sig_f2, tau_enter=tau_enter, tau_exit=tau_exit, s_normal=s_normal, s_reduced=s_reduced)

    # In Fold 1: t=1 (1.3 >= 1.2) enters reduced; t=2 (0.8 > 0.2) stays reduced; t=3 (0.1 <= 0.2) exits to normal
    expected_f1 = [0.85, 0.45, 0.45, 0.85, 0.85]
    np.testing.assert_array_almost_equal(path_f1, expected_f1)

    # In Fold 2: t=2 (1.4 >= 1.2) enters reduced; t=3 (1.1 > 0.2) stays reduced; t=4 (0.15 <= 0.2) exits to normal
    expected_f2 = [0.85, 0.85, 0.45, 0.45, 0.85]
    np.testing.assert_array_almost_equal(path_f2, expected_f2)


def test_portfolio_ledger_consistency_with_summary_metrics():
    """Verify that portfolio ledger records aggregate directly to reported returns, costs, and utilities."""
    epochs = [
        {
            "week": "2024-W01",
            "assets": ["AAPL", "MSFT"],
            "w_base": np.array([0.5, 0.5]),
            "real_r": np.array([0.01, 0.02]),
            "Q_market": np.array([[0.04, 0.01], [0.01, 0.04]]),
        },
        {
            "week": "2024-W02",
            "assets": ["AAPL", "MSFT"],
            "w_base": np.array([0.6, 0.4]),
            "real_r": np.array([-0.02, 0.01]),
            "Q_market": np.array([[0.04, 0.01], [0.01, 0.04]]),
        },
    ]

    s_path = [0.70, 0.70]
    cost_bps = 0.0010
    ledger = build_portfolio_ledger(epochs, s_path, policy_name="M8_Constant", cost_bps=cost_bps)

    assert len(ledger) == 2

    # Check that net return equals gross return minus transaction cost in all periods
    for row in ledger:
        assert pytest.approx(row["net_return"], abs=1e-12) == row["gross_return"] - row["transaction_cost"]
        assert pytest.approx(row["net_utility"], abs=1e-12) == row["net_return"] - row["risk_penalty"]

    # Verify cumulative wealth equals product of (1 + net_r)
    expected_w1 = (1.0 + ledger[0]["net_return"]) * (1.0 + ledger[1]["net_return"])
    assert pytest.approx(ledger[1]["cumulative_wealth"], abs=1e-12) == expected_w1
