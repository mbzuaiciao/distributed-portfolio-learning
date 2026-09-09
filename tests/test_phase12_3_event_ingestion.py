"""Unit tests for Phase 12.3: Genuine Prospective Event Ingestion Protocol."""

from __future__ import annotations

import csv
import datetime
import hashlib
import inspect
import json
import tempfile
from pathlib import Path

import pytest

from experiments.freeze_new_prospective_event import (
    FROZEN_CHECKPOINT_HASHES,
    MC_DRAWS,
    SEEDS,
    compute_first_eligible_trading_session,
    freeze_prospective_event,
)


def test_prediction_generation_never_accesses_outcome_data() -> None:
    """Verify freeze_new_prospective_event source code contains zero imports or references to price/market data."""
    import experiments.freeze_new_prospective_event as module

    source = inspect.getsource(module)
    forbidden_terms = [
        "yfinance",
        "yahoo",
        "raw_market_data",
        "reconstructed_returns",
        "portfolio_ledger",
        "ret_1d",
        "adj_close",
        "close_price",
        "realized_return",
    ]
    for term in forbidden_terms:
        assert (
            term not in source
        ), f"Forbidden market outcome term found in prediction ingestion module: {term}"


def test_prediction_timestamp_precedes_first_evaluated_market_observation() -> None:
    """Verify that a prediction frozen before the first session is genuinely prospective, but after is retrospective."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        doc_file = tmp_path / "filing.txt"
        doc_file.write_text("Quarterly earnings report filing text.")

        # Filing released 2026-10-15 16:30 (Thursday afternoon)
        # First regular session is Friday 2026-10-16 09:30
        filing_str = "2026-10-15 16:30:00"

        # Case 1: Frozen before first session (Thursday 17:00) -> GENUINELY_PROSPECTIVE
        t_valid = datetime.datetime.fromisoformat("2026-10-15T17:00:00+00:00")
        res_valid = freeze_prospective_event(
            ticker="MSFT",
            filing_path=doc_file,
            event_timestamp_str=filing_str,
            fiscal_period="2027_Q1",
            output_dir_base=tmp_path / "artifacts1",
            registry_path=tmp_path / "registry1.csv",
            prediction_time_override=t_valid,
        )
        assert res_valid["prospective_status"] == "GENUINELY_PROSPECTIVE"
        assert res_valid["eligible_for_milestone"] is True

        # Case 2: Frozen after session open (Friday 10:00) -> RETROSPECTIVE-BUT-FROZEN
        t_late = datetime.datetime.fromisoformat("2026-10-16T10:00:00+00:00")
        res_late = freeze_prospective_event(
            ticker="MSFT",
            filing_path=doc_file,
            event_timestamp_str=filing_str,
            fiscal_period="2027_Q1",
            output_dir_base=tmp_path / "artifacts2",
            registry_path=tmp_path / "registry2.csv",
            prediction_time_override=t_late,
        )
        assert res_late["prospective_status"] == "RETROSPECTIVE-BUT-FROZEN"
        assert res_late["eligible_for_milestone"] is False


def test_first_tradable_cutoff_rule_scenarios() -> None:
    """Verify conservative cutoff logic across weekend, intraday, pre-market, and after-market filings."""
    # Weekend Saturday 14:00 -> Monday 09:30
    sat_dt = datetime.datetime.fromisoformat("2026-10-17T14:00:00+00:00")
    assert compute_first_eligible_trading_session(
        sat_dt
    ) == datetime.datetime.fromisoformat("2026-10-19T09:30:00+00:00")

    # Pre-market Tuesday 07:45 -> Tuesday 09:30
    premkt_dt = datetime.datetime.fromisoformat("2026-10-20T07:45:00+00:00")
    assert compute_first_eligible_trading_session(
        premkt_dt
    ) == datetime.datetime.fromisoformat("2026-10-20T09:30:00+00:00")

    # Intraday Tuesday 11:30 -> Next regular session Wednesday 09:30 (Conservative rule)
    intraday_dt = datetime.datetime.fromisoformat("2026-10-20T11:30:00+00:00")
    assert compute_first_eligible_trading_session(
        intraday_dt
    ) == datetime.datetime.fromisoformat("2026-10-21T09:30:00+00:00")

    # After-market Friday 16:15 -> Next Monday 09:30
    fri_after_dt = datetime.datetime.fromisoformat("2026-10-23T16:15:00+00:00")
    assert compute_first_eligible_trading_session(
        fri_after_dt
    ) == datetime.datetime.fromisoformat("2026-10-26T09:30:00+00:00")


def test_placeholder_and_retrospective_rows_cannot_count_toward_milestone() -> None:
    """Verify prospective event registry and continuation manifest enforce 0 eligible events currently."""
    base_dir = Path(__file__).resolve().parent.parent
    reg_path = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "prospective_event_registry.csv"
    )
    assert reg_path.exists()
    with open(reg_path) as f:
        rows = list(csv.DictReader(f))

    # All initial 5 rows must be RETROSPECTIVE-BUT-FROZEN and NOT eligible for milestone
    assert len(rows) >= 5
    for r in rows:
        if r["event_id"] in {
            "GOOGL_2026_Q2",
            "MSFT_2026_Q4",
            "AAPL_2026_Q3",
            "AMZN_2026_Q2",
            "NVDA_2026_Q2",
        }:
            assert r["prospective_status"] == "RETROSPECTIVE-BUT-FROZEN"
            assert r["eligible_for_milestone"] == "False"

    # Continuation manifest check
    manifest_path = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "prospective_continuation_manifest.json"
    )
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert (
        manifest["continuation_rules"][
            "completed_genuinely_prospective_count_as_of_tranche_1"
        ]
        == 0
    )
    assert (
        manifest["unblinded_tranche_1"][
            "genuinely_prospective_completed_events_count"
        ]
        == 0
    )


def test_model_checkpoint_hashes_and_mc_draws_remain_fixed() -> None:
    """Verify MC draws is 32 and checkpoint hashes match freeze manifest."""
    assert MC_DRAWS == 32
    assert SEEDS == [42, 123, 777]

    base_dir = Path(__file__).resolve().parent.parent
    freeze_manifest_path = (
        base_dir
        / "artifacts"
        / "phase11_replication"
        / "prospective_freeze_manifest.json"
    )
    with open(freeze_manifest_path) as f:
        manifest = json.load(f)

    for seed, expected_sha in FROZEN_CHECKPOINT_HASHES.items():
        assert (
            manifest["frozen_model_checkpoints"][f"seed_{seed}"]["sha256"]
            == expected_sha
        )

    # Prediction file hash verification
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
    assert (
        sha.hexdigest()
        == "6542b33fa5810985beff8c835e71b598ffaf47bf943e88d43f7c50d7dcb30aeb"
    )


def test_malformed_event_timing_fails_loudly() -> None:
    """Verify missing file or empty file fails loudly with appropriate exceptions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        missing_doc = tmp_path / "nonexistent.txt"
        empty_doc = tmp_path / "empty.txt"
        empty_doc.write_text("")

        with pytest.raises(FileNotFoundError):
            freeze_prospective_event(
                ticker="AAPL",
                filing_path=missing_doc,
                event_timestamp_str="2026-10-22 16:30:00",
                fiscal_period="2026_Q4",
            )

        with pytest.raises(ValueError, match="empty"):
            freeze_prospective_event(
                ticker="AAPL",
                filing_path=empty_doc,
                event_timestamp_str="2026-10-22 16:30:00",
                fiscal_period="2026_Q4",
            )
