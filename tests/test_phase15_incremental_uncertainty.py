"""Unit tests for Phase 15: Incremental Value of Bayesian Epistemic Uncertainty.

Audits preregistration constraints, data integrity, LORO split isolation,
dependence-aware bootstrap properties, and outcome artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def artifacts_dir(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "phase15_incremental_uncertainty"


def test_phase15_artifacts_exist(artifacts_dir: Path) -> None:
    """Verify all mandated Phase 15 artifacts were generated and are non-empty."""
    required_files = [
        "signal_comparison.csv",
        "regime_signal_comparison.csv",
        "seed_signal_comparison.csv",
        "loro_model_comparison.csv",
        "residual_information_test.csv",
        "quintile_calibration.csv",
        "block_bootstrap_intervals.csv",
        "phase15_summary.json",
        "phase15_report.md",
    ]
    for fname in required_files:
        fpath = artifacts_dir / fname
        assert fpath.exists(), f"Missing required artifact: {fname}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fname}"


def test_primary_clean_sample_constraints(artifacts_dir: Path) -> None:
    """Verify primary clean sample strictly contains 2022, 2023, 2024-2025 (N=12,300) and no 2020-2021."""
    summary_path = artifacts_dir / "phase15_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    scope = summary["dataset_scope"]
    assert sorted(scope["primary_clean_regimes"]) == ["2022", "2023", "2024-2025"]
    assert scope["total_clean_observations"] == 12300, "Clean observations must equal 4,100 events * 3 seeds = 12,300"
    assert sorted(scope["secondary_diagnostic_regimes"]) == ["2020", "2021"]
    assert scope["total_diagnostic_observations"] == 7167, "Diagnostic observations must equal 2,389 events * 3 seeds = 7,167"


def test_no_portfolio_strategies_introduced(repo_root: Path) -> None:
    """Audit codebase to ensure Phase 15 introduces NO new trading strategies (e.g. M17)."""
    phase15_script = repo_root / "experiments" / "run_phase15_incremental_uncertainty.py"
    with open(phase15_script) as f:
        content = f.read()

    assert "def run_portfolio" not in content, "Phase 15 prohibits portfolio simulation"
    assert "class M17" not in content, "Phase 15 strictly prohibits defining M17"
    assert "def optimize_portfolio" not in content, "Phase 15 prohibits portfolio optimization"
    assert "rebalance_portfolio" not in content, "Phase 15 prohibits portfolio rebalancing"


def test_loro_cross_validation_properties(artifacts_dir: Path) -> None:
    """Verify Leave-One-Regime-Out (LORO) model comparison properties."""
    loro_df = pd.read_csv(artifacts_dir / "loro_model_comparison.csv")

    # Verify each held-out clean regime is present
    held_out_regimes = set(loro_df["held_out_regime"].unique())
    assert {"2022", "2023", "2024-2025", "All_Clean_Regimes"}.issubset(held_out_regimes)

    # Verify models A, B, C, D are present for each split
    for regime in ["2022", "2023", "2024-2025", "All_Clean_Regimes"]:
        reg_df = loro_df[loro_df["held_out_regime"] == regime]
        models = set(reg_df["model_name"].unique())
        assert len(models) == 4
        assert "Model A (Confidence Only)" in models
        assert "Model B (Market Only)" in models
        assert "Model C (Confidence + Market)" in models
        assert "Model D (Confidence + Market + Epistemic)" in models

    # Check pooled delta metrics: delta_auroc should be within [-0.01, 0.01] (Outcome B)
    pooled_d = loro_df[(loro_df["evaluation_scope"] == "LORO_Pooled_Out_of_Sample") & (loro_df["model_name"].str.startswith("Model D"))].iloc[0]
    assert abs(pooled_d["delta_auroc_vs_model_c"]) <= 0.01
    assert abs(pooled_d["delta_mae_vs_model_c"]) <= 0.001


def test_residual_information_test_validity(artifacts_dir: Path) -> None:
    """Verify simple confidence explains >80% of epistemic variance and residuals are uninformative."""
    resid_df = pd.read_csv(artifacts_dir / "residual_information_test.csv")
    clean_pooled = resid_df[resid_df["sample_scope"] == "Clean_Pooled (2022-2025)"].iloc[0]

    assert clean_pooled["r2_explained_by_simple_confidence"] > 0.80, "Simple confidence must explain majority of epistemic variance"
    assert abs(clean_pooled["residual_epistemic_auroc"] - 0.50) < 0.02, "Residual epistemic AUROC should be approximately 0.50"


def test_block_bootstrap_confidence_intervals(artifacts_dir: Path) -> None:
    """Verify block bootstrap intervals span zero for incremental value (Model D vs Model C)."""
    boot_df = pd.read_csv(artifacts_dir / "block_bootstrap_intervals.csv")

    delta_auroc_row = boot_df[boot_df["comparison_metric"] == "Delta_AUROC_Incremental (Model D - Model C)"].iloc[0]
    # 95% CI should straddle zero
    assert delta_auroc_row["ci_95_lower"] <= 0.0 <= delta_auroc_row["ci_95_upper"]
    assert not delta_auroc_row["statistically_significant_at_05"]

    delta_mae_row = boot_df[boot_df["comparison_metric"] == "Delta_MAE_Incremental (Model C - Model D)"].iloc[0]
    # 95% CI should straddle zero
    assert delta_mae_row["ci_95_lower"] <= 0.0 <= delta_mae_row["ci_95_upper"]
    assert not delta_mae_row["statistically_significant_at_05"]


def test_scientific_adjudication_verdict(artifacts_dir: Path) -> None:
    """Verify the adjudicated verdict is Outcome B."""
    summary_path = artifacts_dir / "phase15_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    verdict = summary["scientific_adjudication"]["verdict"]
    assert "Outcome B" in verdict
