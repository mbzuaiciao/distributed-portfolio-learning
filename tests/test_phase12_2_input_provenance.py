"""Unit tests for Phase 12.2: Frozen 2026 Input Provenance Audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def test_every_frozen_row_has_provenance_category() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    provenance_path = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "frozen_input_temporal_provenance.csv"
    )
    assert provenance_path.exists(), f"Missing provenance CSV: {provenance_path}"

    with open(provenance_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 52, f"Expected 52 frozen rows, found {len(rows)}"

    valid_categories = {
        "RETROSPECTIVE-BUT-FROZEN",
        "GENUINELY PROSPECTIVE",
        "FUTURE PLACEHOLDER / NOT YET PREDICTABLE",
    }

    for r in rows:
        status = r["prospective_status"]
        assert (
            status in valid_categories
        ), f"Row {r['event_id']} has invalid prospective status: {status}"


def test_no_genuinely_prospective_row_uses_future_text() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    provenance_path = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "frozen_input_temporal_provenance.csv"
    )
    with open(provenance_path) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if r["prediction_row_exists"] == "True":
            doc_date = r["input_document_date"]
            pred_date = r["prediction_generation_date"]
            assert doc_date != "None" and pred_date != "None"
            assert (
                doc_date <= pred_date
            ), f"Causal violation: doc_date {doc_date} > pred_date {pred_date}"
            assert r["actual_text_available_at_prediction_time"] == "True"


def test_event_timestamp_and_cohort_date_are_separate_fields() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    provenance_path = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "frozen_input_temporal_provenance.csv"
    )
    with open(provenance_path) as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    required_separate_fields = [
        "event_timestamp",
        "filing_timestamp",
        "calendar_cohort_week",
        "first_tradable_date",
    ]
    for field in required_separate_fields:
        assert (
            field in fieldnames
        ), f"Field {field} missing from canonical field set"

    # Verify corporate events have differentiated values where expected
    googl = next(r for r in rows if r["event_id"] == "GOOGL_2026_Q2")
    assert googl["event_timestamp"] == "2026-07-23 16:00:00"
    assert googl["calendar_cohort_week"] == "2026-W30"
    assert googl["first_tradable_date"] == "2026-07-24"

    msft = next(r for r in rows if r["event_id"] == "MSFT_2026_Q4")
    assert msft["filing_timestamp"] == "2026-07-28 16:05:00"
    assert msft["first_tradable_date"] == "2026-07-29"

    aapl = next(r for r in rows if r["event_id"] == "AAPL_2026_Q3")
    assert aapl["filing_timestamp"] == "2026-07-31 16:30:00"
    assert aapl["first_tradable_date"] == "2026-08-03"

    amzn = next(r for r in rows if r["event_id"] == "AMZN_2026_Q2")
    assert amzn["filing_timestamp"] == "2026-08-01 16:00:00"
    assert amzn["first_tradable_date"] == "2026-08-03"


def test_prospective_milestone_counts_only_genuinely_prospective_events() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    continuation_path = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "prospective_continuation_manifest.json"
    )
    with open(continuation_path) as f:
        manifest = json.load(f)

    rules = manifest["continuation_rules"]
    assert "prospective_eligibility_rule" in rules
    assert "GENUINELY PROSPECTIVE" in rules["prospective_eligibility_rule"]
    assert rules["completed_genuinely_prospective_count_as_of_tranche_1"] == 0

    audit_summary = manifest["provenance_audit_summary"]
    assert audit_summary["eligible_prospective_events_toward_milestone"] == 0
    assert audit_summary["causal_input_violations_count"] == 0
    assert audit_summary["future_placeholder_rows"] == 18


def test_prediction_sha_remains_unchanged() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    pred_path = (
        base_dir
        / "artifacts"
        / "phase11_replication"
        / "frozen_2026_predictions.csv"
    )
    sha = hashlib.sha256()
    with open(pred_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)

    expected_sha = (
        "6542b33fa5810985beff8c835e71b598ffaf47bf943e88d43f7c50d7dcb30aeb"
    )
    assert (
        sha.hexdigest() == expected_sha
    ), f"Prediction SHA altered! Got {sha.hexdigest()}"
