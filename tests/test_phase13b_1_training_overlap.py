"""Tests for Phase 13B.1: Model-Training / Evaluation Overlap Audit.

Verifies:
1. Every valid Phase-13 event (N=8,692) receives a definite overlap status.
2. Post-2021 events (2022, 2023, 2024-2025) cannot be labeled training overlap.
3. Overlapping events (2018-2019, 2020, 2021) cannot be classified as clean out-of-sample.
4. Clean evaluation vs secondary diagnostic regimes are properly declared.
5. Frozen portfolio mapping parameters remain unchanged from commit 8ed3430.
6. M15 development selection status is explicitly recorded as failing the positive development criterion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

repo_root = Path(__file__).resolve().parent.parent


@pytest.fixture
def audit_df() -> pd.DataFrame:
    csv_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "model_training_overlap_audit.csv"
    if not csv_path.exists():
        pytest.skip("model_training_overlap_audit.csv does not exist.")
    return pd.read_csv(csv_path)


@pytest.fixture
def summary_data() -> dict:
    json_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "model_training_overlap_summary.json"
    if not json_path.exists():
        pytest.skip("model_training_overlap_summary.json does not exist.")
    with open(json_path) as f:
        return json.load(f)


def test_every_phase13_event_audited(audit_df: pd.DataFrame):
    """Verify that all 8,692 Phase-13 events receive an explicit overlap status."""
    assert len(audit_df) == 8692, f"Expected 8,692 audited events, got {len(audit_df)}"

    expected_cols = {
        "event_id",
        "ticker",
        "event_date",
        "regime",
        "training_overlap",
        "match_method",
        "training_row_id",
        "notes",
    }
    assert expected_cols.issubset(set(audit_df.columns))

    # Overlap must be strictly boolean (True/False) with 0 unknown
    overlap_vals = set(audit_df["training_overlap"].unique())
    assert overlap_vals.issubset({True, False, "True", "False"})
    assert "unknown" not in overlap_vals


def test_post_2021_events_have_zero_overlap(audit_df: pd.DataFrame):
    """Verify that post-2021 events cannot be labeled training overlap."""
    post_2021 = audit_df[audit_df["regime"].isin(["2022", "2023", "2024-2025"])].copy()
    assert len(post_2021) == 1191 + 1181 + 1728  # 4,100 events

    # None of these can have training_overlap == True
    assert not (post_2021["training_overlap"].astype(str) == "True").any(), (
        "Post-2021 event erroneously flagged as training overlap!"
    )
    assert (post_2021["training_overlap"].astype(str) == "False").all()


def test_overlapping_events_cannot_be_clean_oos(audit_df: pd.DataFrame, summary_data: dict):
    """Verify that overlapping regimes (2020, 2021) are excluded from clean OOS."""
    # 2020 and 2021 must be 100% overlap
    overlap_regimes = audit_df[audit_df["regime"].isin(["2018-2019", "2020", "2021"])].copy()
    assert (overlap_regimes["training_overlap"].astype(str) == "True").all()

    # Check protocol declaration in summary JSON
    clean_regimes = summary_data["evaluation_protocol_views"]["primary_clean_evaluation"]["regimes"]
    assert clean_regimes == ["2022", "2023", "2024-2025"]
    assert "2020" not in clean_regimes
    assert "2021" not in clean_regimes
    assert "2018-2019" not in clean_regimes

    diagnostic_regimes = summary_data["evaluation_protocol_views"]["secondary_diagnostic_evaluation"]["regimes"]
    assert set(diagnostic_regimes) == {"2020", "2021"}


def test_frozen_mapping_parameters_unchanged(summary_data: dict):
    """Verify that Phase-13 frozen parameters remain exactly as preregistered."""
    prereg_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "mapping_preregistration.json"
    with open(prereg_path) as f:
        prereg = json.load(f)

    frozen = prereg["selected_frozen_parameters"]
    assert frozen["lambda_M14"] == 0.25
    assert frozen["fraction_M15"] == 0.30
    assert frozen["lambda_M16"] == 0.10

    # Summary confirmation matches preregistration
    sum_params = summary_data["frozen_portfolio_mapping_parameters_confirmed_unchanged"]
    assert sum_params["lambda_M14"] == 0.25
    assert sum_params["fraction_M15"] == 0.30
    assert sum_params["lambda_M16"] == 0.10
    assert sum_params["horizon_H"] == 4
    assert sum_params["transaction_cost_bps"] == 10.0


def test_m15_failed_positive_criterion_recorded():
    """Verify that M15 development performance is documented as failing the positive-development criterion."""
    sum_csv = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "development_selection_summary.csv"
    assert sum_csv.exists()
    df = pd.read_csv(sum_csv)

    m15_rows = df[df["strategy"] == "M15"]
    assert len(m15_rows) == 3

    # All M15 candidates must have negative mean_delta_u
    assert (m15_rows["mean_delta_u"] < 0.0).all(), (
        "An M15 candidate had positive mean_delta_u; protocol requires acknowledging failure."
    )

    # In contrast, selected M14 and M16 must have positive mean_delta_u
    m14_sel = df[(df["strategy"] == "M14") & (df["parameter"] == 0.25)].iloc[0]
    m16_sel = df[(df["strategy"] == "M16") & (df["parameter"] == 0.10)].iloc[0]
    assert m14_sel["mean_delta_u"] > 0.0
    assert m16_sel["mean_delta_u"] > 0.0
