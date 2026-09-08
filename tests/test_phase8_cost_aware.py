"""
Unit tests for Phase 8: Cost-Aware Epistemic Timing with Persistence and Rebalancing Bands
========================================================================================
Validates:
- Hysteresis state machine & dead-band properties
- Persistence filtering (k-of-n logic)
- Inertia rebalancing bands with partial adjustment
- Lower-frequency rebalancing & emergency override
- Minimum holding duration enforcement
- Turnover attribution & alarm analysis accounting
"""

import numpy as np
import pytest

from hypergraph_portfolio.uncertainty import (
    analyze_uncertainty_alarms,
    compute_band_exposure_path,
    compute_hysteresis_exposure_path,
    compute_lower_frequency_exposure_path,
    compute_min_hold_exposure_path,
    compute_persistent_exposure_path,
    compute_turnover_attribution,
)


def test_hysteresis_state_machine():
    """Verify two-threshold hysteresis prevents rapid flip-flopping."""
    # tau_enter = 1.0, tau_exit = 0.2
    # Normal is 0.85, Reduced is 0.45
    signals = [0.0, 0.5, 1.2, 0.8, 0.4, 0.1, 0.5]
    # t=0: z=0.0 -> normal (0.85)
    # t=1: z=0.5 -> normal (0.85)
    # t=2: z=1.2 > 1.0 -> reduced (0.45)
    # t=3: z=0.8 (in dead band [0.2, 1.0]) -> remains reduced (0.45)
    # t=4: z=0.4 (in dead band [0.2, 1.0]) -> remains reduced (0.45)
    # t=5: z=0.1 < 0.2 -> exits to normal (0.85)
    # t=6: z=0.5 (in dead band) -> remains normal (0.85)
    exp = compute_hysteresis_exposure_path(signals, tau_enter=1.0, tau_exit=0.2, s_normal=0.85, s_reduced=0.45)
    expected = np.array([0.85, 0.85, 0.45, 0.45, 0.45, 0.85, 0.85])
    np.testing.assert_allclose(exp, expected)

    # Invalid thresholds must raise
    with pytest.raises(ValueError, match="strictly less"):
        compute_hysteresis_exposure_path(signals, tau_enter=0.5, tau_exit=0.5)
    with pytest.raises(ValueError, match="strictly less"):
        compute_hysteresis_exposure_path(signals, tau_enter=0.2, tau_exit=0.8)


def test_persistence_logic():
    """Verify k-of-n persistence ignores isolated 1-week spikes."""
    tau = 1.0
    # Case 1: Transient 1-week spike
    signals_transient = [0.0, 2.5, 0.1, 0.2]
    exp1 = compute_persistent_exposure_path(signals_transient, tau=tau, k=2, n=2, s_normal=0.85, s_reduced=0.45)
    # With k=2, n=2, single spike at t=1 does not satisfy condition (only 1 of 2 > 1.0)
    assert np.all(exp1 == 0.85)

    # Case 2: Persistent 2-week spike
    signals_persistent = [0.0, 1.5, 1.8, 0.1]
    exp2 = compute_persistent_exposure_path(signals_persistent, tau=tau, k=2, n=2, s_normal=0.85, s_reduced=0.45)
    # t=0: 0.0 -> normal (0.85)
    # t=1: [0.0, 1.5] (1 of 2 > 1.0) -> normal (0.85)
    # t=2: [1.5, 1.8] (2 of 2 > 1.0) -> reduced (0.45)
    # t=3: calm [0.1] -> re-risks to normal (0.85) if re_risk_k=1, re_risk_n=1
    assert exp2[0] == 0.85
    assert exp2[1] == 0.85
    assert exp2[2] == 0.45
    assert exp2[3] == 0.85


def test_rebalancing_band_and_partial_adjustment():
    """Verify no-rebalancing band suppresses small adjustments."""
    desired = [0.70, 0.72, 0.69, 0.85, 0.83]
    # With delta=0.05, changes of 0.02 and 0.03 relative to current should be ignored
    band_exp = compute_band_exposure_path(desired, delta=0.05, eta=1.0)
    # t=0: init 0.70
    # t=1: |0.72 - 0.70| = 0.02 < 0.05 -> keep 0.70
    # t=2: |0.69 - 0.70| = 0.01 < 0.05 -> keep 0.70
    # t=3: |0.85 - 0.70| = 0.15 >= 0.05 -> jumps to 0.85
    # t=4: |0.83 - 0.85| = 0.02 < 0.05 -> keep 0.85
    expected = np.array([0.70, 0.70, 0.70, 0.85, 0.85])
    np.testing.assert_allclose(band_exp, expected)

    # Partial adjustment eta=0.5
    partial_exp = compute_band_exposure_path(desired, delta=0.05, eta=0.5)
    # t=3: jump is 0.70 + 0.5*(0.85 - 0.70) = 0.775
    assert np.isclose(partial_exp[3], 0.775)


def test_lower_frequency_rebalancing_and_emergency():
    """Verify biweekly rebalancing with emergency override."""
    desired = [0.80, 0.60, 0.40, 0.70, 0.85]
    # Biweekly (frequency=2): changes allowed at t=0, 2, 4
    biweekly_exp = compute_lower_frequency_exposure_path(desired, frequency=2)
    # t=0: 0.80
    # t=1: keep 0.80
    # t=2: update to 0.40
    # t=3: keep 0.40
    # t=4: update to 0.85
    np.testing.assert_allclose(biweekly_exp, [0.80, 0.80, 0.40, 0.40, 0.85])

    # Emergency override at t=1
    signals = [0.5, 3.5, 0.2, 0.1, 0.4]
    emerg_exp = compute_lower_frequency_exposure_path(
        desired, frequency=2, severe_signal=signals, tau_severe=2.5, s_emergency=0.25
    )
    assert emerg_exp[1] == 0.25  # Emergency de-risking triggered off-schedule


def test_minimum_holding_period():
    """Verify minimum holding lock-in upon de-risking."""
    desired = [0.85, 0.45, 0.85, 0.85, 0.85]
    # min_hold_periods = 2
    # t=0: normal 0.85
    # t=1: desired 0.45 < 0.60 -> transitions to reduced (0.45), hold_counter=1
    # t=2: desired 0.85, but locked in -> must remain reduced (0.45)
    # t=3: lock expired, desired 0.85 -> returns to normal (0.85)
    held_exp = compute_min_hold_exposure_path(
        desired, min_hold_periods=2, threshold=0.60, s_reduced=0.45, s_normal=0.85
    )
    np.testing.assert_allclose(held_exp, [0.85, 0.45, 0.45, 0.85, 0.85])


def test_turnover_attribution_and_savings():
    """Verify turnover attribution and relative savings calculations."""
    base_rets = [0.01, -0.02, 0.03, -0.01]
    # Raw M9 with frequent adjustments
    m9_path = [0.85, 0.50, 0.85, 0.40]
    # Cost-aware policy with fewer adjustments
    cost_aware_path = [0.85, 0.50, 0.50, 0.50]

    attr = compute_turnover_attribution(
        exposure_path=cost_aware_path,
        base_returns=base_rets,
        cost_bps=0.0010,
        m9_exposure_path=m9_path,
    )

    assert attr["n_exposure_changes"] == 1.0  # Only changes once (0.85 -> 0.50)
    assert attr["cost_saved_vs_m9"] > 0.0  # Saves transaction costs relative to M9
    assert 0.0 <= attr["total_turnover"] < 2.0


def test_alarm_analysis_accounting():
    """Verify false alarm analysis correctly classifies hits vs false alarms."""
    signals = [0.1, 1.5, 2.0, 0.2]
    rets = [0.01, -0.03, 0.02, 0.01]
    # Exposure drops below median on periods 1 and 2
    exposures = [0.85, 0.45, 0.45, 0.85]

    alarms = analyze_uncertainty_alarms(signals, rets, exposures, tail_threshold=-0.02)
    assert alarms["n_alarms"] == 2.0
    # Period 1: -0.03 (< 0, < -0.02 tail hit)
    # Period 2: +0.02 (positive return = false positive)
    assert alarms["pct_negative_followed"] == 50.0
    assert alarms["pct_tail_loss_followed"] == 50.0
    assert alarms["false_positive_rate"] == 50.0
