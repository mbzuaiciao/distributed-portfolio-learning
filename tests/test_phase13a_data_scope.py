"""Unit tests for Phase 13A: Historical Data Scope & Temporal Split.

Verifies:
1. Development period strictly precedes evaluation period.
2. No temporal overlap between development and evaluation regimes.
3. Each historical event belongs to exactly one temporal regime.
4. Incomplete H=4 events and missing return events are properly categorized and excluded from eligible sets.
5. Model source categories are completely explicit (no silent mixing of old and new models).
6. No synthetic perturbations or proxy files are counted as genuine inference.
7. Compute decision sizing matches measured local M2 MPS profiling and decision protocol.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def data_scope(repo_root: Path) -> dict:
    scope_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "data_scope.json"
    assert scope_path.exists(), f"Missing {scope_path}"
    with open(scope_path) as f:
        return json.load(f)


@pytest.fixture
def compute_decision(repo_root: Path) -> dict:
    decision_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "compute_decision.json"
    assert decision_path.exists(), f"Missing {decision_path}"
    with open(decision_path) as f:
        return json.load(f)


@pytest.fixture
def inventory_rows(repo_root: Path) -> list[dict[str, str]]:
    inv_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "historical_data_inventory.csv"
    assert inv_path.exists(), f"Missing {inv_path}"
    with open(inv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_development_period_precedes_evaluation(data_scope: dict):
    """Verify development period strictly precedes evaluation period."""
    dev = data_scope["development_split"]
    eval_split = data_scope["evaluation_split"]

    dev_end = dev["end_date"]
    eval_start = eval_split["start_date"]

    assert dev_end < eval_start, f"Development end ({dev_end}) must precede evaluation start ({eval_start})"
    assert dev["regime_id"] == "2018-2019"
    assert dev["n_events"] == 2203


def test_no_overlap_between_regimes(data_scope: dict):
    """Verify strictly non-overlapping temporal bounds across all 6 regimes."""
    dev = data_scope["development_split"]
    eval_regimes = data_scope["evaluation_split"]["regimes"]

    all_regimes = [
        {
            "id": dev["regime_id"],
            "start": dev["start_date"],
            "end": dev["end_date"],
        }
    ] + [
        {
            "id": r["regime_id"],
            "start": r["start_date"],
            "end": r["end_date"],
        }
        for r in eval_regimes
    ]

    expected_order = ["2018-2019", "2020", "2021", "2022", "2023", "2024-2025"]
    assert [r["id"] for r in all_regimes] == expected_order

    for i in range(len(all_regimes) - 1):
        curr_regime = all_regimes[i]
        next_regime = all_regimes[i + 1]
        assert curr_regime["end"] < next_regime["start"], (
            f"Regime {curr_regime['id']} end ({curr_regime['end']}) overlaps with "
            f"Regime {next_regime['id']} start ({next_regime['start']})"
        )


def test_each_event_belongs_to_exactly_one_regime(inventory_rows: list[dict[str, str]]):
    """Verify every event in the inventory has a single valid regime assignment."""
    valid_regimes = {
        "pre-2018_unscoped",
        "2018-2019",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024-2025",
    }
    assert len(inventory_rows) == 20000

    for row in inventory_rows:
        regime = row["regime"]
        assert regime in valid_regimes, f"Invalid regime {regime} for event {row['event_id']}"
        # Split assignment must match regime
        if regime == "pre-2018_unscoped":
            assert row["split"] == "unassigned_pre2018"
        elif regime == "2018-2019":
            assert row["split"] == "development"
        else:
            assert row["split"] == "evaluation"


def test_incomplete_h4_and_missing_returns_excluded(inventory_rows: list[dict[str, str]]):
    """Verify that events missing returns or incomplete H4 are excluded from Category A."""
    for row in inventory_rows:
        cat = row["coverage_category"]
        assert cat in {"A", "B", "C", "D", "E"}

        if cat == "A":
            # Must have valid 1d return, valid H4 return, and valid text
            assert row["has_return_1d"] == "True"
            assert row["has_return_h4"] == "True"
            assert row["has_text"] == "True"
            assert row["valid_model_input"] == "True"

        if cat == "D":
            # Missing 1d return
            assert row["has_return_1d"] == "False"

        if cat == "E":
            # Incomplete H4
            assert row["has_return_h4"] == "False"


def test_model_source_category_explicit(inventory_rows: list[dict[str, str]]):
    """Verify that model source categories are completely explicit and never conflated."""
    valid_statuses = {
        "phase11_checkpoint_inferred",
        "archived_old_model_exists_and_phase11_possible",
        "pending_phase11_inference",
        "pre2018_checkpoint_possible",
        "missing_return_data",
        "incomplete_h4_path",
        "missing_usable_text",
    }
    status_counts = {}
    for row in inventory_rows:
        status = row["model_source_status"]
        assert status in valid_statuses, f"Unknown status: {status}"
        status_counts[status] = status_counts.get(status, 0) + 1

    # 2024-2025 canonical holdout events verified
    assert status_counts["phase11_checkpoint_inferred"] == 1728
    # 2022-2023 dev events where archived old predictions exist
    assert status_counts["archived_old_model_exists_and_phase11_possible"] == 2372
    # Missing return data
    assert status_counts["missing_return_data"] == 1799


def test_no_synthetic_predictions_counted_as_genuine(repo_root: Path):
    """Verify that synthetic perturbations (e.g. hist_epi + noise) are not present in scope."""
    scope_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "data_scope.json"
    with open(scope_path) as f:
        content = f.read()

    forbidden_synthetic_terms = [
        "synthetic_perturbation",
        "hist_epi",
        "noise_epi",
        "repl_epi",
        "archived_epistemic_variance",
    ]
    for term in forbidden_synthetic_terms:
        assert term not in content, f"Synthetic prediction term found in data_scope.json: {term}"


def test_compute_decision_scale_and_rule(compute_decision: dict):
    """Verify compute scale sizing and adherence to 30-120 minute local preference rule."""
    scale = compute_decision["inference_workload_scale"]
    assert scale["total_scoped_events"] == 8692
    assert scale["seeds"] == [42, 123, 777]
    assert scale["mc_draws"] == 32
    assert scale["total_event_inferences"] == 26076  # 8692 * 3
    assert scale["total_forward_passes"] == 834432  # 26076 * 32

    # Check already computed and remaining
    assert scale["already_computed_2024_2025_events"] == 1728
    assert scale["remaining_events_to_infer"] == 6964

    # Check runtime projections
    proj = compute_decision["runtime_projections"]
    assert 30.0 <= proj["full_job_local_min"] <= 120.0
    assert compute_decision["decision"] == "prefer_local"
    assert "30-120 minutes" in compute_decision["decision_rule_window"]
