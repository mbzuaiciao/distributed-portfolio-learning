"""Targeted test suite for Phase 19 architecture signal stress test."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.run_phase19_architecture_signal_stress_test import (
    OUT,
    ROOT,
    load_freeze,
    sha256,
    signal_stat,
)


@pytest.fixture(scope="module")
def freeze_cfg():
    cfg, commit = load_freeze()
    return cfg, commit


def test_preregistration_committed_and_hashes_match(freeze_cfg):
    cfg, commit = freeze_cfg
    assert commit is not None and len(commit) > 0
    assert cfg["experiment_id"] == "phase19_architecture_signal_stress_test"
    assert cfg["portfolio_parameters"]["horizons"] == [1, 4]
    assert cfg["signal_regimes"]["synthetic_ladder"]["alpha_grid"] == [
        0.00,
        0.05,
        0.10,
        0.20,
        0.40,
    ]
    assert len(cfg["signal_regimes"]["random_ranking"]["seeds"]) == 50
    assert len(cfg["signal_regimes"]["synthetic_ladder"]["noise_seeds"]) == 20

    for path, digest in cfg["source_file_hashes"].items():
        full_path = ROOT / path
        assert full_path.exists(), f"Source file does not exist: {path}"
        assert sha256(full_path) == digest, f"Hash mismatch for {path}"


def test_mutated_preregistration_rejected(monkeypatch, tmp_path):
    real = OUT / "preregistration.json"
    altered = tmp_path / "preregistration.json"
    altered.write_text(real.read_text().replace("0.7073", "0.9999"))
    original = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda p: altered.read_text().encode() if p == real else original(p)
    )
    with pytest.raises(ValueError, match="differs from committed"):
        load_freeze()


def test_algebraic_utility_decomposition_closure():
    """Verify algebraic closure holds exactly to floating point tolerance on a toy test."""
    T = 20
    rng = np.random.default_rng(42)
    h1_net = rng.normal(0.001, 0.02, size=T)
    h4_net = rng.normal(0.001, 0.008, size=T)
    h1_gross = h1_net + 0.0015
    h4_gross = h4_net + 0.0004
    h1_to = np.full(T, 1.5)
    h4_to = np.full(T, 0.4)
    cost_rate = 0.001

    dg = float(np.mean(h4_gross - h1_gross))
    cost_saved = float(cost_rate * np.mean(h1_to - h4_to))
    var_gain = float(0.5 * (np.var(h1_net, ddof=1) - np.var(h4_net, ddof=1)))
    u1 = float(np.mean(h1_net) - 0.5 * np.var(h1_net, ddof=1))
    u4 = float(np.mean(h4_net) - 0.5 * np.var(h4_net, ddof=1))
    delta_u = u4 - u1

    error = abs(delta_u - (dg + cost_saved + var_gain))
    assert error < 1e-14


def test_signal_stat_constant_is_nan():
    score = np.full(100, 0.5)
    returns = np.random.randn(100)
    ids = np.arange(100)
    ic, spread, q = signal_stat(score, returns, ids)
    assert np.isnan(ic)
    assert np.isnan(spread)
    assert all(np.isnan(x) for x in q)


def test_signal_stat_perfect_correlation():
    score = np.linspace(-1, 1, 100)
    returns = np.linspace(-0.5, 0.5, 100)
    ids = np.arange(100)
    ic, spread, q = signal_stat(score, returns, ids)
    assert np.isclose(ic, 1.0, atol=1e-12)
    assert spread > 0
    assert q[4] > q[0]


def test_decomposition_algebraic_closure_across_all_runs():
    decomp_path = OUT / "decomposition_table.csv"
    assert decomp_path.exists(), "Decomposition table missing"
    df = pd.read_csv(decomp_path)
    assert len(df) >= 250, f"Expected >= 250 rows, got {len(df)}"

    reconstructed_delta = (
        df["delta_gross_weekly_return"]
        + df["weekly_cost_saving"]
        + df["sample_variance_utility_gain"]
    )
    max_err = np.max(np.abs(df["delta_net_utility"] - reconstructed_delta))
    assert max_err < 1e-12, f"Closure error exceeded 1e-12: max_err = {max_err}"
    assert np.all(df["closure_error"] < 1e-12)


def test_h1_mechanical_turnover_reduction_invariant():
    """H1: H=4 reduces traded turnover by at least 50% relative to H=1 across all empirical and null regimes."""
    decomp_path = OUT / "decomposition_table.csv"
    df = pd.read_csv(decomp_path)

    # Check clean LoRA, ticker risk, constant, equal weight, random ranking
    for reg in ["clean_lora", "ticker_risk", "constant_ranking", "equal_weight", "random_ranking"]:
        sub = df[df.regime == reg]
        assert len(sub) > 0, f"No entries for {reg}"
        min_to_red = sub["traded_turnover_reduction"].min()
        assert min_to_red >= 0.50, f"{reg} minimum turnover reduction {min_to_red} < 0.50"


def test_h2_diversification_volatility_reduction_invariant():
    """H2: H=4 reduces annualized volatility by at least 30% relative to H=1 across all empirical and null regimes."""
    decomp_path = OUT / "decomposition_table.csv"
    df = pd.read_csv(decomp_path)

    for reg in ["clean_lora", "ticker_risk", "constant_ranking", "equal_weight", "random_ranking"]:
        sub = df[df.regime == reg]
        assert len(sub) > 0, f"No entries for {reg}"
        min_vol_red = sub["volatility_reduction"].min()
        assert min_vol_red >= 0.30, f"{reg} minimum volatility reduction {min_vol_red} < 0.30"


def test_effective_holdings_ratio_expansion():
    """H=4 expands effective holdings by at least 2.0x relative to H=1."""
    decomp_path = OUT / "decomposition_table.csv"
    df = pd.read_csv(decomp_path)
    assert np.all(df["effective_holdings_ratio"] >= 2.0)
    assert df["effective_holdings_ratio"].mean() >= 3.0


def test_random_ranking_distribution_null_behavior():
    rand_path = OUT / "random_ranking_distribution.csv"
    assert rand_path.exists()
    df = pd.read_csv(rand_path)
    assert len(df) == 150  # 50 random seeds x 3 base model predictions

    # All random ranking runs must pass H1 and H2
    assert np.all(df["traded_turnover_reduction"] >= 0.50)
    assert np.all(df["volatility_reduction"] >= 0.30)
    # Mean delta utility must be positive due to cost + variance savings
    assert df["delta_net_utility"].mean() > 0.0
    assert df["weekly_cost_saving"].mean() > 0.0
    assert df["sample_variance_utility_gain"].mean() > 0.0


def test_synthetic_signal_ladder_monotonicity():
    synth_path = OUT / "synthetic_signal_ladder.csv"
    assert synth_path.exists()
    df = pd.read_csv(synth_path)
    assert len(df) == 5  # 5 alpha levels

    # Rank IC should be strictly monotonically increasing with alpha
    rank_ics = df["mean_rank_ic"].to_numpy()
    assert np.all(np.diff(rank_ics) > 0), f"Rank ICs not monotonic: {rank_ics}"

    # Turnover reduction must be invariant across all alphas
    assert np.allclose(df["to_reduction_mean"], 0.7397, atol=1e-3)
    assert np.all(df["vol_reduction_mean"] >= 0.60)


def test_outcome_classification_integrity():
    class_path = OUT / "outcome_classification.json"
    assert class_path.exists()
    c = json.loads(class_path.read_text())
    assert c["outcome"] == "D"
    assert c["h1_mechanical_invariance_pass"] is True
    assert c["h2_diversification_invariance_pass"] is True
    assert c["h3_utility_dependence_confirmed"] is False
    assert c["h3_direction_reversed"] is True
    assert c["h4_no_alpha_interpretation_enforced"] is True
    assert c["net_utility_all_positive"] is False
    assert c["algebraic_closure_pass"] is True
    assert c["max_closure_error"] < 1e-12


def test_all_deliverables_exist_and_nonempty():
    expected_files = [
        "preregistration.json",
        "preregistration.md",
        "experiment_matrix.csv",
        "signal_quality_table.csv",
        "h1_vs_h4_by_regime.csv",
        "random_ranking_distribution.csv",
        "synthetic_signal_ladder.csv",
        "decomposition_table.csv",
        "statistical_intervals.csv",
        "architecture_stress_test_figures.png",
        "architecture_stress_test_figures.pdf",
        "scientific_report.md",
        "outcome_classification.json",
    ]
    for f in expected_files:
        p = OUT / f
        assert p.exists(), f"Missing deliverable: {f}"
        assert p.stat().st_size > 0, f"Empty deliverable: {f}"

