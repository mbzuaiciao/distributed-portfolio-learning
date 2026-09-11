"""Unit tests for Phase 13C: Frozen Out-of-Sample Regime Evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE13_DIR = REPO_ROOT / "artifacts" / "phase13_uncertainty_mapping"


def test_phase13c_artifacts_exist() -> None:
    """Verify all required Phase 13C artifacts were produced and are non-empty."""
    expected_files = [
        "evaluation_predictions.csv",
        "regime_performance.csv",
        "seed_regime_matrix.csv",
        "stability_summary.csv",
        "cost_sensitivity.csv",
        "uncertainty_quintile_allocation.csv",
        "forecast_error_allocation_audit.csv",
        "global_vs_local_comparison.csv",
        "phase13_summary.json",
        "phase13_report.md",
    ]
    for filename in expected_files:
        filepath = PHASE13_DIR / filename
        assert filepath.exists(), f"Missing required artifact: {filepath}"
        assert filepath.stat().st_size > 0, f"Artifact is empty: {filepath}"


def test_evaluation_predictions_integrity() -> None:
    """Verify evaluation_predictions.csv covers exactly 6,489 events x 3 seeds = 19,467 rows."""
    pred_path = PHASE13_DIR / "evaluation_predictions.csv"
    df = pd.read_csv(pred_path)
    assert len(df) == 19467, f"Expected 19,467 prediction rows, got {len(df)}"

    required_cols = [
        "event_id",
        "ticker",
        "date",
        "regime",
        "seed",
        "posterior_mean",
        "epistemic_uncertainty",
        "aleatoric_uncertainty",
        "total_uncertainty",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col}"

    assert df["event_id"].isna().sum() == 0
    assert set(df["seed"].unique()) == {42, 123, 777}
    assert (df["posterior_mean"] >= 0.0).all() and (df["posterior_mean"] <= 1.0).all()
    assert (df["epistemic_uncertainty"] >= 0.0).all()

    # Verify regime event counts
    regime_counts = df.groupby("regime")["event_id"].nunique()
    assert regime_counts["2020"] == 1198
    assert regime_counts["2021"] == 1191
    assert regime_counts["2022"] == 1191
    assert regime_counts["2023"] == 1181
    assert regime_counts["2024-2025"] == 1728


def test_clean_regimes_separation() -> None:
    """Verify primary clean regimes are strictly 2022, 2023, 2024-2025 (N=4,100 clean events)."""
    sum_path = PHASE13_DIR / "phase13_summary.json"
    with open(sum_path) as f:
        summary = json.load(f)

    clean_regs = summary["evaluation_scopes"]["primary_clean_regimes"]
    assert clean_regs == ["2022", "2023", "2024-2025"]
    assert summary["evaluation_scopes"]["total_clean_events"] == 4100
    assert summary["evaluation_scopes"]["total_clean_weeks"] == 168

    diag_regs = summary["evaluation_scopes"]["secondary_diagnostic_regimes"]
    assert diag_regs == ["2020", "2021"]
    assert summary["evaluation_scopes"]["total_diagnostic_events"] == 2389
    assert summary["evaluation_scopes"]["total_diagnostic_weeks"] == 99


def test_frozen_parameters_adherence() -> None:
    """Verify parameters match preregistered frozen values from commit 8ed3430ea5."""
    sum_path = PHASE13_DIR / "phase13_summary.json"
    with open(sum_path) as f:
        summary = json.load(f)

    params = summary["frozen_mapping_parameters"]
    assert params["lambda_M14"] == 0.25
    assert params["fraction_M15"] == 0.30
    assert params["lambda_M16"] == 0.10
    assert params["m13_parameters"]["tau_enter"] == 1.2
    assert params["m13_parameters"]["tau_exit"] == 0.2
    assert params["m13_parameters"]["s_normal"] == 0.85
    assert params["m13_parameters"]["s_reduced"] == 0.45
    assert params["horizon_H"] == 4
    assert params["transaction_cost_bps"] == 10.0
    assert params["gamma"] == 1.0


def test_stability_summary_and_verdict() -> None:
    """Verify stability summary values and final adjudicated verdict (Outcome F)."""
    stab_path = PHASE13_DIR / "stability_summary.csv"
    df = pd.read_csv(stab_path)
    assert set(df["strategy"]) == {"M13", "M14", "M15", "M16"}

    sum_path = PHASE13_DIR / "phase13_summary.json"
    with open(sum_path) as f:
        summary = json.load(f)

    verdict = summary["adjudicated_verdict"]["classification"]
    assert "Outcome F" in verdict

    # M16 has positive mean clean delta U, but only 1/3 positive clean regimes
    m16_row = df[df["strategy"] == "M16"].iloc[0]
    assert m16_row["mean_delta_u_clean"] > 0
    assert m16_row["positive_clean_regimes"] == 1

    # M13 has negative mean delta U across all clean regimes
    m13_row = df[df["strategy"] == "M13"].iloc[0]
    assert m13_row["mean_delta_u_clean"] < 0
    assert m13_row["positive_clean_regimes"] == 0


def test_cost_sensitivity_invariance() -> None:
    """Verify M14 and M16 utility deltas are invariant to cost levels due to identical turnover to M0."""
    cost_path = PHASE13_DIR / "cost_sensitivity.csv"
    df = pd.read_csv(cost_path)
    assert set(df["cost_bps"]) == {0, 10, 25}

    for strat in ["M14", "M16"]:
        sub = df[(df["strategy"] == strat) & (df["regime"] == "2023") & (df["seed"] == 42)]
        deltas = sub["delta_utility_vs_M0"].values
        assert np.allclose(deltas, deltas[0], atol=1e-6)


def test_mechanism_uncertainty_quintile_monotonicity() -> None:
    """Verify uncertainty quintiles show positive monotonic relationship with forecast error."""
    q_path = PHASE13_DIR / "uncertainty_quintile_allocation.csv"
    df = pd.read_csv(q_path)
    assert len(df) == 5

    # Check forecast error increases monotonically from Q1 to Q5
    errors = df["mean_forecast_error"].tolist()
    assert errors == sorted(errors), f"Forecast errors not monotonically non-decreasing: {errors}"
    assert errors[-1] > errors[0] * 1.5, f"Expected Q5 error to be > 1.5x Q1 error, got {errors[-1]} vs {errors[0]}"
