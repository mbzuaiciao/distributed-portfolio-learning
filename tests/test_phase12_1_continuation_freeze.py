"""
Tests for Phase 12.1: Prospective Interpretation Correction + Sequential Continuation Freeze
===========================================================================================
Verifies:
1. Exact 52-event funnel accounting closure with no unexplained exclusions.
2. Zero contrast between M13 and M8 exposure paths (informative timing weeks == 0).
3. Zero-contrast result is explicitly classified as Outcome F and non-informative bootstrap.
4. Model-level prospective sample size is strictly N=5 unique events (seeds do not inflate N).
5. Annualized short-sample metrics are properly quarantined to diagnostic appendix.
6. Prospective continuation milestone is fixed independently of future outcomes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def test_input_funnel_accounting_closure() -> None:
    base_dir = Path(__file__).parent.parent
    funnel_csv = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "phase12_input_funnel.csv"
    )
    assert funnel_csv.exists(), f"Missing funnel CSV: {funnel_csv}"

    with open(funnel_csv) as f:
        rows = list(csv.DictReader(f))

    # Exact 52 inputs accounting identity
    assert len(rows) == 52

    n_evaluated = sum(1 for r in rows if r["exclusion_reason"] == "NONE_EVALUATED")
    n_censored = sum(1 for r in rows if "RIGHT_CENSORED" in r["exclusion_reason"])
    n_pre_freeze = sum(1 for r in rows if "PRE_FREEZE" in r["exclusion_reason"])
    n_future = sum(1 for r in rows if "FUTURE_DATED" in r["exclusion_reason"])

    assert n_evaluated == 4
    assert n_censored == 1
    assert n_pre_freeze == 29
    assert n_future == 18

    # Identity closure: 4 + 1 + 29 + 18 = 52
    assert n_evaluated + n_censored + n_pre_freeze + n_future == 52


def test_outcome_f_and_non_informative_bootstrap() -> None:
    base_dir = Path(__file__).parent.parent
    summary_json = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "phase12_summary.json"
    )
    assert summary_json.exists()

    with open(summary_json) as f:
        data = json.load(f)

    # Corrected Outcome F verdict
    assert "Outcome F" in data["verdict"]
    assert "insufficient/uninformative" in data["verdict"]

    # Degenerate bootstrap classification
    assert "DEGENERATE / NON-INFORMATIVE BOOTSTRAP" in data["bootstrap_classification"]
    assert data["primary_endpoint_mean"] == 0.0

    # Informative timing weeks is zero
    assert data["informative_timing_weeks"] == 0


def test_unique_event_sample_size_accounting() -> None:
    base_dir = Path(__file__).parent.parent
    manifest_path = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "prospective_continuation_manifest.json"
    )
    assert manifest_path.exists()

    with open(manifest_path) as f:
        m = json.load(f)

    t1 = m["unblinded_tranche_1"]
    assert t1["unique_events_count"] == 5
    assert t1["complete_h4_events_count"] == 4
    assert t1["right_censored_events_count"] == 1
    assert t1["informative_timing_weeks_count"] == 0

    # Rule enforcing N=unique events, not unique events * seeds
    assert "strictly count UNIQUE events" in m["continuation_rules"]["model_uncertainty_sample_size_rule"]


def test_non_annualized_metrics_priority() -> None:
    base_dir = Path(__file__).parent.parent
    metrics_csv = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "short_sample_metrics_non_annualized.csv"
    )
    assert metrics_csv.exists()

    with open(metrics_csv) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 8  # 3 seeds x 2 policies + 1 mean x 2 policies

    mean_rows = {r["policy"]: r for r in rows if r["seed"] == "mean"}
    m8 = mean_rows["M8_Constant"]
    m13 = mean_rows["M13_Hysteresis"]

    # No contrast between M8 and M13
    assert m8["cumulative_net_return"] == m13["cumulative_net_return"]
    assert m8["mean_weekly_utility"] == m13["mean_weekly_utility"]
    assert float(m8["cumulative_net_return"]) > 0.10  # ~13.7% realized 5-week gain


def test_continuation_milestone_independence() -> None:
    base_dir = Path(__file__).parent.parent
    manifest_path = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "prospective_continuation_manifest.json"
    )
    with open(manifest_path) as f:
        m = json.load(f)

    rules = m["continuation_rules"]
    assert "20 UNIQUE events" in rules["milestone_criteria"]
    assert "December 31, 2026" in rules["milestone_criteria"]
    assert "No early stopping based on performance or whether M13 activates" in rules["stopping_rule_independence"]

    # Parameters locked
    params = m["frozen_strategy_parameters"]
    assert params["status"] == "LOCKED_UNCHANGED"
    assert params["holding_horizon_H"] == 4
    assert params["tau_enter"] == 1.2
    assert params["tau_exit"] == 0.2
