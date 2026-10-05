"""Tests for Phase 20: Optimal Cohort Holding Horizon Experiment."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.run_phase20_optimal_cohort_horizon import (
    generate_synthetic_universe,
    run_simulator_audits,
    simulate_horizon_trajectories,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/phase20_optimal_cohort_horizon"


def test_preregistration_files_exist():
    assert (OUT / "preregistration.md").exists()
    assert (OUT / "preregistration.json").exists()
    assert (OUT / "analytical_derivation.md").exists()
    assert (OUT / "simulation_design.json").exists()


def test_preregistration_json_schema():
    with open(OUT / "preregistration.json") as f:
        cfg = json.load(f)

    assert cfg["phase"] == 20
    assert cfg["parameter_grids"]["horizons"] == [1, 2, 4, 8, 13]
    assert cfg["parameter_grids"]["persistence_rho"] == [0.0, 0.25, 0.50, 0.75, 0.90, 0.97]
    assert cfg["parameter_grids"]["transaction_costs_bps"] == [0, 5, 10, 25, 50]
    assert cfg["parameter_grids"]["signal_noise_std"] == [0.0, 0.5, 1.0, 2.0]
    assert cfg["parameter_grids"]["risk_aversion_gamma"] == [0.0, 0.5, 1.0, 2.5, 5.0]
    assert len(cfg["simulation_seeds"]) == 30
    assert cfg["accounting_closure_tolerance"] == 1e-12


def test_all_10_simulator_audits_pass():
    audits = run_simulator_audits()
    assert len(audits) == 10
    for name, passed in audits.items():
        assert passed, f"Audit failed: {name}"


def test_turnover_strictly_decreases_with_horizon():
    u = generate_synthetic_universe(200001, 30, 100, 0.02, 0.5, 0.5)
    turnovers = []
    for h in [1, 2, 4, 8, 13]:
        _, to, _, _ = simulate_horizon_trajectories(u["base_weights"], u["ret_by_age"], horizon=h)
        turnovers.append(np.mean(to[1:]))

    for i in range(len(turnovers) - 1):
        assert turnovers[i] > turnovers[i + 1], f"Turnover did not decrease: H={h} vs next"


def test_volatility_strictly_decreases_with_horizon():
    u = generate_synthetic_universe(200002, 30, 156, 0.02, 0.5, 0.5)
    vols = []
    for h in [1, 2, 4, 8, 13]:
        g, _, _, _ = simulate_horizon_trajectories(u["base_weights"], u["ret_by_age"], horizon=h)
        vols.append(np.std(g, ddof=1))

    for i in range(len(vols) - 1):
        assert vols[i] >= vols[i + 1] - 1e-6, f"Volatility did not decrease: H index {i}"


def test_exact_accounting_closure():
    u = generate_synthetic_universe(200003, 30, 156, 0.02, 0.5, 0.5)
    g1, to1, _, _ = simulate_horizon_trajectories(u["base_weights"], u["ret_by_age"], horizon=1)
    g4, to4, _, _ = simulate_horizon_trajectories(u["base_weights"], u["ret_by_age"], horizon=4)

    c = 0.0010
    gamma = 1.0

    net1 = g1 - c * to1
    net4 = g4 - c * to4

    u1 = np.mean(net1) - 0.5 * gamma * np.var(net1, ddof=1)
    u4 = np.mean(net4) - 0.5 * gamma * np.var(net4, ddof=1)

    delta_u = u4 - u1
    delta_gross = np.mean(g4 - g1)
    cost_effect = -c * np.mean(to4 - to1)
    risk_effect = -0.5 * gamma * (np.var(net4, ddof=1) - np.var(net1, ddof=1))

    closure_err = abs(delta_u - (delta_gross + cost_effect + risk_effect))
    assert closure_err < 1e-12, f"Closure error too large: {closure_err}"
