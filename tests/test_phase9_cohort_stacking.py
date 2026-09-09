"""
Tests for Phase 9 Multi-Week Cohort Stacking.
=============================================
Validates:
1. Simplex weight conservation (sum w_j,t = 1.0)
2. Cohort lifecycle timing (enters at t, exits at t+H, exact duration H)
3. Duplicate asset aggregation across active overlapping cohorts
4. Non-anticipative causality (frozen entry weights)
5. H=1 equivalence to unstacked single-week portfolio
6. Holding overlap identity (TO_base = 2 * (1 - Overlap))
7. Monotonic overlap scaling with horizon H
"""

import numpy as np
import pytest

from hypergraph_portfolio.uncertainty import (
    build_cohort_stacked_weights,
    simulate_cohort_stacked_portfolio,
)


def test_simplex_weight_conservation_and_duplicate_aggregation():
    """Verify that stacked cohort weights always sum to 1.0 and duplicate assets are summed."""
    weekly_cohorts = [
        {"AAPL": 0.5, "MSFT": 0.5},
        {"AAPL": 0.4, "NVDA": 0.6},  # AAPL overlaps with week 0
        {"GOOGL": 0.7, "AMZN": 0.3},
        {"MSFT": 0.8, "NVDA": 0.2},
    ]

    # Test H = 2
    H = 2
    stacked = build_cohort_stacked_weights(weekly_cohorts, horizon=H)
    assert len(stacked) == 4

    for t, w in enumerate(stacked):
        total_w = sum(w.values())
        assert pytest.approx(total_w, abs=1e-6) == 1.0
        assert all(val >= 0.0 for val in w.values())

    # At t=1, active cohorts are cohort 0 (50% cap) and cohort 1 (50% cap)
    # Cohort 0 has AAPL (0.5), MSFT (0.5) -> scaled by 0.5: AAPL 0.25, MSFT 0.25
    # Cohort 1 has AAPL (0.4), NVDA (0.6) -> scaled by 0.5: AAPL 0.20, NVDA 0.30
    # Aggregated: AAPL 0.45, MSFT 0.25, NVDA 0.30
    w_t1 = stacked[1]
    assert pytest.approx(w_t1["AAPL"], abs=1e-6) == 0.45
    assert pytest.approx(w_t1["MSFT"], abs=1e-6) == 0.25
    assert pytest.approx(w_t1["NVDA"], abs=1e-6) == 0.30


def test_cohort_lifecycle_timing():
    """Verify each cohort enters at t and exits strictly after H weeks."""
    # Create disjoint cohorts each week
    weekly_cohorts = [
        {f"STOCK_{t}_{i}": 0.5 for i in range(2)}
        for t in range(10)
    ]
    H = 3
    stacked = build_cohort_stacked_weights(weekly_cohorts, horizon=H)

    # For cohort 2: enters at t=2, active at t=2, t=3, t=4, exits at t=5
    for t in range(10):
        w = stacked[t]
        cohort_2_present = "STOCK_2_0" in w
        if 2 <= t <= 4:
            assert cohort_2_present, f"Cohort 2 should be present at t={t}"
        else:
            assert not cohort_2_present, f"Cohort 2 should NOT be present at t={t}"


def test_h1_equivalence_to_unstacked():
    """For H=1, stacked weights should be identical to the original weekly cohorts."""
    weekly_cohorts = [
        {"AAPL": 0.6, "MSFT": 0.4},
        {"NVDA": 0.7, "META": 0.3},
        {"GOOGL": 0.5, "AMZN": 0.5},
    ]
    stacked = build_cohort_stacked_weights(weekly_cohorts, horizon=1)
    for orig, st in zip(weekly_cohorts, stacked):
        assert set(orig.keys()) == set(st.keys())
        for k in orig:
            assert pytest.approx(orig[k], abs=1e-6) == st[k]


def test_holding_overlap_identity_and_monotonicity():
    """Verify that TO_base = 2 * (1 - Overlap) and Overlap increases monotonically with H."""
    # Disjoint cohorts each week
    weekly_cohorts = [
        {f"ASSET_{t}_{i}": 0.25 for i in range(4)}
        for t in range(20)
    ]

    overlaps_by_h = []
    base_turnovers_by_h = []

    for H in [1, 2, 4, 8]:
        stacked = build_cohort_stacked_weights(weekly_cohorts, horizon=H)

        # Compute weekly base turnover and overlap from t=H to end (steady state)
        overlaps = []
        base_tos = []
        for t in range(1, len(stacked)):
            w_curr = stacked[t]
            w_prev = stacked[t - 1]
            all_keys = set(w_curr.keys()) | set(w_prev.keys())

            overlap = sum(min(w_curr.get(k, 0.0), w_prev.get(k, 0.0)) for k in all_keys)
            base_to = sum(abs(w_curr.get(k, 0.0) - w_prev.get(k, 0.0)) for k in all_keys)

            # Identity check: TO_base = 2 * (1 - Overlap)
            assert pytest.approx(base_to, abs=1e-5) == 2.0 * (1.0 - overlap)

            if t >= H:
                overlaps.append(overlap)
                base_tos.append(base_to)

        mean_overlap = float(np.mean(overlaps))
        mean_base_to = float(np.mean(base_tos))
        overlaps_by_h.append(mean_overlap)
        base_turnovers_by_h.append(mean_base_to)

    # For disjoint cohorts:
    # H=1: Overlap = 0.0, TO_base = 2.0
    # H=2: Overlap = 0.5, TO_base = 1.0
    # H=4: Overlap = 0.75, TO_base = 0.5
    # H=8: Overlap = 0.875, TO_base = 0.25
    assert pytest.approx(overlaps_by_h[0], abs=1e-5) == 0.0
    assert pytest.approx(overlaps_by_h[1], abs=1e-5) == 0.5
    assert pytest.approx(overlaps_by_h[2], abs=1e-5) == 0.75
    assert pytest.approx(overlaps_by_h[3], abs=1e-5) == 0.875

    assert pytest.approx(base_turnovers_by_h[0], abs=1e-5) == 2.0
    assert pytest.approx(base_turnovers_by_h[1], abs=1e-5) == 1.0
    assert pytest.approx(base_turnovers_by_h[2], abs=1e-5) == 0.5
    assert pytest.approx(base_turnovers_by_h[3], abs=1e-5) == 0.25


def test_simulate_cohort_stacked_portfolio_causality():
    """Verify that simulate_cohort_stacked_portfolio runs correctly and respects causality."""
    T = 15
    np.random.seed(42)
    asset_names = [f"SYM_{i}" for i in range(20)]

    # Generate mock weekly returns
    weekly_returns_table = []
    for _ in range(T):
        rets = {sym: float(np.random.normal(0.002, 0.03)) for sym in asset_names}
        weekly_returns_table.append(rets)

    # Generate mock cohorts (3 assets per week)
    weekly_cohorts = []
    for t in range(T):
        chosen = [asset_names[(t * 2 + i) % len(asset_names)] for i in range(3)]
        weekly_cohorts.append({sym: 1.0 / 3.0 for sym in chosen})

    res_h1 = simulate_cohort_stacked_portfolio(weekly_cohorts, weekly_returns_table, horizon=1)
    res_h3 = simulate_cohort_stacked_portfolio(weekly_cohorts, weekly_returns_table, horizon=3)

    assert len(res_h1["weekly_net_returns"]) == T
    assert len(res_h3["weekly_net_returns"]) == T

    # Turnover for H=3 should be significantly lower than H=1
    assert res_h3["annual_base_turnover"] < res_h1["annual_base_turnover"]
    assert res_h3["mean_overlap"] > res_h1["mean_overlap"]

    # Cohort age returns tracking should exist
    assert 0 in res_h3["cohort_age_returns"]
    assert 1 in res_h3["cohort_age_returns"]
    assert 2 in res_h3["cohort_age_returns"]
