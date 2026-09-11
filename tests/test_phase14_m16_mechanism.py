"""Unit tests for Phase 14: Mechanism Decomposition of Uncertainty-Aware Risk Inflation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE14_DIR = REPO_ROOT / "artifacts" / "phase14_m16_mechanism"
PHASE13_DIR = REPO_ROOT / "artifacts" / "phase13_uncertainty_mapping"


def test_phase14_artifacts_exist() -> None:
    """Verify all 8 required Phase 14 artifacts were produced and are non-empty."""
    expected_files = [
        "phase14_weekly_mechanism.csv",
        "phase14_regime_comparison.csv",
        "phase14_bucket_analysis.csv",
        "phase14_correlations.csv",
        "phase14_weight_shift_by_uncertainty.csv",
        "phase14_error_alignment.csv",
        "phase14_explanatory_model.csv",
        "phase14_summary.json",
    ]
    for filename in expected_files:
        filepath = PHASE14_DIR / filename
        assert filepath.exists(), f"Missing required Phase 14 artifact: {filepath}"
        assert filepath.stat().st_size > 0, f"Artifact is empty: {filepath}"


def test_frozen_parameters_and_no_new_strategy() -> None:
    """Verify M16 lambda remains strictly 0.10 and no M17 or conditional strategies are created."""
    sum_path = PHASE14_DIR / "phase14_summary.json"
    with open(sum_path) as f:
        summary = json.load(f)

    frozen = summary["frozen_strategy"]
    assert frozen["strategy"] == "M16"
    assert frozen["lambda_u"] == 0.10
    assert frozen["status"] == "FROZEN_UNTOUCHED"
    assert frozen["horizon_H"] == 4
    assert frozen["transaction_cost_bps"] == 10.0
    assert frozen["gamma"] == 1.0

    # Ensure no M17 was introduced in any output
    weekly_path = PHASE14_DIR / "phase14_weekly_mechanism.csv"
    weekly_df = pd.read_csv(weekly_path)
    assert not any("m17" in col.lower() for col in weekly_df.columns)
    assert not any("conditional_m16" in col.lower() for col in weekly_df.columns)


def test_primary_clean_regimes_separation() -> None:
    """Verify primary clean regimes are strictly 2022, 2023, 2024-2025 and 2020-2021 are diagnostic."""
    sum_path = PHASE14_DIR / "phase14_summary.json"
    with open(sum_path) as f:
        summary = json.load(f)

    clean_regs = summary["universe"]["primary_clean_regimes"]
    assert clean_regs == ["2022", "2023", "2024-2025"]
    assert summary["universe"]["total_clean_weeks_observations"] == 504  # 168 weeks x 3 seeds

    diag_regs = summary["universe"]["secondary_diagnostic_regimes"]
    assert diag_regs == ["2020", "2021"]
    assert summary["universe"]["total_diagnostic_weeks_observations"] == 297  # 99 weeks x 3 seeds


def test_numerical_reconciliation_to_phase13c() -> None:
    """Verify week-level delta_u_m16 averages reconcile to Phase 13C regime performance."""
    weekly_path = PHASE14_DIR / "phase14_weekly_mechanism.csv"
    weekly_df = pd.read_csv(weekly_path)

    p13_path = PHASE13_DIR / "regime_performance.csv"
    p13_df = pd.read_csv(p13_path)

    for reg in ["2020", "2021", "2022", "2023", "2024-2025"]:
        for s in [42, 123, 777]:
            w_sub = weekly_df[(weekly_df["regime"] == reg) & (weekly_df["seed"] == s)]
            p13_sub = p13_df[(p13_df["regime"] == reg) & (p13_df["seed"] == s)]

            p13_u_m0 = float(p13_sub[p13_sub["strategy"] == "M0"]["mean_weekly_utility"].iloc[0])
            p13_u_m16 = float(p13_sub[p13_sub["strategy"] == "M16"]["mean_weekly_utility"].iloc[0])
            p13_delta_u = p13_u_m16 - p13_u_m0

            mean_w_delta_u = float(w_sub["delta_u_m16"].mean())
            assert np.isclose(mean_w_delta_u, p13_delta_u, atol=2e-5), (
                f"Reconciliation mismatch for {reg} seed {s}: {mean_w_delta_u} vs {p13_delta_u}"
            )


def test_ex_ante_vs_post_hoc_labeling() -> None:
    """Verify ex-ante state variables do not use future information and realized variables are labeled post-hoc."""
    corr_path = PHASE14_DIR / "phase14_correlations.csv"
    corr_df = pd.read_csv(corr_path)

    for _, row in corr_df.iterrows():
        col = row["variable"]
        vtype = row["variable_type"]
        if "post_hoc" in col or "error" in col or "realized" in col:
            assert vtype == "POST_HOC", f"Variable {col} contains realized info but labeled {vtype}"
        else:
            assert vtype == "EX_ANTE", f"Variable {col} is ex-ante but labeled {vtype}"


def test_error_alignment_diagnostic_validity() -> None:
    """Verify that M16 success is strongly correlated with capital reduction in high-error names."""
    align_path = PHASE14_DIR / "phase14_error_alignment.csv"
    df = pd.read_csv(align_path)
    clean = df[df["regime_type"] == "CLEAN_OOS"]

    mean_succ = clean[clean["m16_succeeded"] == True]["err_reduction_share"].mean()  # noqa: E712
    mean_fail = clean[clean["m16_succeeded"] == False]["err_reduction_share"].mean()  # noqa: E712

    assert mean_succ > mean_fail, f"Expected higher error reduction in success weeks: {mean_succ} vs {mean_fail}"
    assert mean_succ > 0.55
    assert mean_fail < 0.50


def test_seeds_remain_separate() -> None:
    """Verify all 3 seeds are present and analyzed separately in weekly dataset."""
    weekly_path = PHASE14_DIR / "phase14_weekly_mechanism.csv"
    df = pd.read_csv(weekly_path)
    assert set(df["seed"].unique()) == {42, 123, 777}
    for s in [42, 123, 777]:
        sub = df[df["seed"] == s]
        assert len(sub) == 267  # 168 clean + 99 diagnostic weeks
