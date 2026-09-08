"""
Tests for Phase 8.2: Transaction Mechanics Audit for Exposure Scaling.

Verifies that transaction costs are applied to the actual risky-asset trades
created by changes in aggregate exposure w_j,t = s_t * q_j,t.
"""

import numpy as np
import pytest

from hypergraph_portfolio.uncertainty import (
    build_portfolio_ledger,
    compute_actual_traded_turnover,
    compute_hysteresis_exposure_path,
)


def test_one_asset_exposure_reduction_80_to_40():
    """
    PART B: Deterministic toy test for single asset exposure reduction (80% -> 40%).
    q_0 = [1.0], s_0 = 0.80 -> stock = 0.80, cash = 0.20
    q_1 = [1.0], s_1 = 0.40 -> stock = 0.40, cash = 0.60
    Expected:
        - risky-sleeve composition turnover: 0.00
        - actual risky security traded: 0.40
        - cash weight change: 0.40
        - costed_turnover: 0.40
        - transaction_cost at 10 bps: 0.0004 (4 bps of NAV)
    MUST fail if implementation reports zero broker cost!
    """
    res = compute_actual_traded_turnover(
        current_risky_composition=[1.0],
        previous_risky_composition=[1.0],
        current_exposure=0.40,
        previous_exposure=0.80,
    )

    assert res["risky_composition_turnover"] == pytest.approx(0.00, abs=1e-9)
    assert res["actual_risky_traded_turnover"] == pytest.approx(0.40, abs=1e-9)
    assert res["cash_weight_turnover"] == pytest.approx(0.40, abs=1e-9)
    assert res["costed_turnover"] == pytest.approx(0.40, abs=1e-9)

    cost_rate = 0.0010  # 10 bps
    tc = cost_rate * res["costed_turnover"]
    assert tc == pytest.approx(0.0004, abs=1e-9)
    assert tc > 0.0, "Transaction cost must NOT be zero for exposure reduction!"


def test_one_asset_exposure_increase_40_to_80():
    """
    PART E: Deterministic toy test for single asset pure exposure increase (40% -> 80%).
    q_0 = [1.0], s_0 = 0.40 -> stock = 0.40, cash = 0.60
    q_1 = [1.0], s_1 = 0.80 -> stock = 0.80, cash = 0.20
    Expected:
        - risky-sleeve composition turnover: 0.00
        - actual risky security traded: 0.40
        - cash weight change: 0.40
        - costed_turnover: 0.40
        - transaction_cost at 10 bps: 0.0004 (4 bps of NAV)
    """
    res = compute_actual_traded_turnover(
        current_risky_composition=[1.0],
        previous_risky_composition=[1.0],
        current_exposure=0.80,
        previous_exposure=0.40,
    )

    assert res["risky_composition_turnover"] == pytest.approx(0.00, abs=1e-9)
    assert res["actual_risky_traded_turnover"] == pytest.approx(0.40, abs=1e-9)
    assert res["cash_weight_turnover"] == pytest.approx(0.40, abs=1e-9)
    assert res["costed_turnover"] == pytest.approx(0.40, abs=1e-9)


def test_unchanged_exposure_changing_composition():
    """
    PART D: Test pure internal cross-sectional rebalancing at unchanged exposure.
    Time 0: q_0 = [0.60, 0.40], s_0 = 0.70 -> Actual: [0.42, 0.28]
    Time 1: q_1 = [0.40, 0.60], s_1 = 0.70 -> Actual: [0.28, 0.42]
    Expected:
        - composition turnover: |0.40 - 0.60| + |0.60 - 0.40| = 0.40
        - actual risky traded turnover: |0.28 - 0.42| + |0.42 - 0.28| = 0.28
        - cash weight turnover: |0.30 - 0.30| = 0.00
    """
    res = compute_actual_traded_turnover(
        current_risky_composition=[0.40, 0.60],
        previous_risky_composition=[0.60, 0.40],
        current_exposure=0.70,
        previous_exposure=0.70,
    )

    assert res["risky_composition_turnover"] == pytest.approx(0.40, abs=1e-9)
    assert res["actual_risky_traded_turnover"] == pytest.approx(0.28, abs=1e-9)
    assert res["cash_weight_turnover"] == pytest.approx(0.00, abs=1e-9)
    assert res["costed_turnover"] == pytest.approx(0.28, abs=1e-9)


def test_changing_exposure_changing_composition():
    """
    PART C: Test simultaneous composition and exposure change.
    Time 0: q_0 = [0.60, 0.40], s_0 = 0.80 -> Actual: [0.48, 0.32]
    Time 1: q_1 = [0.50, 0.50], s_1 = 0.40 -> Actual: [0.20, 0.20]
    Expected:
        - composition turnover: |0.50 - 0.60| + |0.50 - 0.40| = 0.20
        - actual risky trades: |0.20 - 0.48| + |0.20 - 0.32| = 0.28 + 0.12 = 0.40
        - cash weight change: |0.60 - 0.20| = 0.40
    Costable risky turnover must be 0.40, NOT 0.20!
    """
    res = compute_actual_traded_turnover(
        current_risky_composition=[0.50, 0.50],
        previous_risky_composition=[0.60, 0.40],
        current_exposure=0.40,
        previous_exposure=0.80,
    )

    assert res["risky_composition_turnover"] == pytest.approx(0.20, abs=1e-9)
    assert res["actual_risky_traded_turnover"] == pytest.approx(0.40, abs=1e-9)
    assert res["actual_risky_traded_turnover"] != pytest.approx(res["risky_composition_turnover"], abs=1e-4)
    assert res["costed_turnover"] == pytest.approx(0.40, abs=1e-9)
    assert res["cash_weight_turnover"] == pytest.approx(0.40, abs=1e-9)


def test_unchanged_composition_unchanged_exposure():
    """
    PART N.5: Unchanged composition and unchanged exposure must yield 0 turnover.
    """
    res = compute_actual_traded_turnover(
        current_risky_composition=[0.30, 0.70],
        previous_risky_composition=[0.30, 0.70],
        current_exposure=0.65,
        previous_exposure=0.65,
    )

    assert res["risky_composition_turnover"] == pytest.approx(0.0, abs=1e-9)
    assert res["actual_risky_traded_turnover"] == pytest.approx(0.0, abs=1e-9)
    assert res["cash_weight_turnover"] == pytest.approx(0.0, abs=1e-9)
    assert res["costed_turnover"] == pytest.approx(0.0, abs=1e-9)


def test_transaction_cost_rule_cost_rate_times_actual_risky():
    """
    PART G & N.6: Transaction cost is strictly cost_rate * actual_risky_traded_turnover.
    """
    res = compute_actual_traded_turnover(
        current_risky_composition=[0.25, 0.75],
        previous_risky_composition=[0.50, 0.50],
        current_exposure=0.60,
        previous_exposure=0.90,
    )
    # w_0 = [0.45, 0.45], w_1 = [0.15, 0.45]
    # Delta w = |0.15 - 0.45| + |0.45 - 0.45| = 0.30
    assert res["actual_risky_traded_turnover"] == pytest.approx(0.30, abs=1e-9)

    cost_rate = 0.0015  # 15 bps
    tc = cost_rate * res["actual_risky_traded_turnover"]
    assert tc == pytest.approx(0.00045, abs=1e-9)


def test_no_double_counting_of_cash():
    """
    PART N.7: Cash is frictionless; cash turnover is NOT charged in costed_turnover.
    """
    res = compute_actual_traded_turnover(
        current_risky_composition=[0.50, 0.50],
        previous_risky_composition=[0.50, 0.50],
        current_exposure=0.50,
        previous_exposure=0.80,
    )
    # Actual risky: 0.30, Cash: 0.30, Total L1: 0.60
    assert res["actual_risky_traded_turnover"] == pytest.approx(0.30, abs=1e-9)
    assert res["cash_weight_turnover"] == pytest.approx(0.30, abs=1e-9)
    assert res["total_l1_turnover"] == pytest.approx(0.60, abs=1e-9)
    # Broker cost is applied ONLY to risky trades (0.30), not total (0.60)
    assert res["costed_turnover"] == pytest.approx(0.30, abs=1e-9)
    assert res["costed_turnover"] < res["total_l1_turnover"]


def test_ledger_transaction_costs_equal_sum_of_actual_risky_trades():
    """
    PART N.8: Verify master ledger transaction costs match sum of actual risky trades.
    """
    mock_epochs = [
        {
            "week": "2024-W01",
            "assets": ["A", "B"],
            "w_base": np.array([0.60, 0.40]),
            "real_r": np.array([0.01, 0.02]),
            "Q_market": np.eye(2) * 0.04,
        },
        {
            "week": "2024-W02",
            "assets": ["A", "B"],
            "w_base": np.array([0.50, 0.50]),
            "real_r": np.array([-0.01, 0.01]),
            "Q_market": np.eye(2) * 0.04,
        },
        {
            "week": "2024-W03",
            "assets": ["A", "B"],
            "w_base": np.array([0.40, 0.60]),
            "real_r": np.array([0.03, -0.01]),
            "Q_market": np.eye(2) * 0.04,
        },
    ]
    exposures = [0.80, 0.40, 0.70]
    cost_rate = 0.0010

    ledger = build_portfolio_ledger(mock_epochs, exposures, cost_bps=cost_rate)

    # Period 0: t=0 -> turnover = 0
    assert ledger[0]["actual_risky_traded_turnover"] == 0.0
    assert ledger[0]["transaction_cost"] == 0.0

    # Period 1: t=1: w_0=[0.48, 0.32], w_1=[0.20, 0.20] -> actual risky = 0.40
    assert ledger[1]["actual_risky_traded_turnover"] == pytest.approx(0.40, abs=1e-9)
    assert ledger[1]["transaction_cost"] == pytest.approx(0.0004, abs=1e-9)

    # Period 2: t=2: w_1=[0.20, 0.20], w_2=[0.28, 0.42] -> actual risky = |0.28-0.20| + |0.42-0.20| = 0.08 + 0.22 = 0.30
    assert ledger[2]["actual_risky_traded_turnover"] == pytest.approx(0.30, abs=1e-9)
    assert ledger[2]["transaction_cost"] == pytest.approx(0.0003, abs=1e-9)

    total_tc = sum(r["transaction_cost"] for r in ledger)
    expected_total_tc = cost_rate * sum(r["actual_risky_traded_turnover"] for r in ledger)
    assert total_tc == pytest.approx(expected_total_tc, abs=1e-12)
    assert total_tc == pytest.approx(0.0007, abs=1e-9)


def test_frozen_m13_parameters_unchanged():
    """
    PART N.9: Verify frozen M13 parameters remain strictly immutable:
    tau_enter = 1.2, tau_exit = 0.2, s_normal = 0.85, s_reduced = 0.45.
    """
    z_scores = np.array([0.0, 0.5, 1.3, 1.0, 0.1, 0.0])
    s_path = compute_hysteresis_exposure_path(
        z_scores,
        s_normal=0.85,
        s_reduced=0.45,
        tau_enter=1.2,
        tau_exit=0.2,
    )
    # Expected behavior:
    # t=0: z=0.0 -> 0.85 (normal)
    # t=1: z=0.5 -> 0.85 (below 1.2)
    # t=2: z=1.3 -> 0.45 (enters defensive state: z >= 1.2)
    # t=3: z=1.0 -> 0.45 (stays defensive: z > 0.2)
    # t=4: z=0.1 -> 0.85 (exits defensive state: z <= 0.2)
    # t=5: z=0.0 -> 0.85 (normal)
    expected = [0.85, 0.85, 0.45, 0.45, 0.85, 0.85]
    np.testing.assert_allclose(s_path, expected, atol=1e-9)
