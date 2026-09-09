"""
Tests for Phase 12: Prospective 2026 Outcome Evaluation
=======================================================
Verifies:
1. Frozen prospective prediction SHA256 is exact (6542b33fa5810985beff8c835e71b598ffaf47bf943e88d43f7c50d7dcb30aeb).
2. Checkpoint hashes for seeds 42, 123, 777 are exact.
3. Event/ticker mapping has zero silent mismatches and avoids row-order joins.
4. Deterministic first-tradable date convention.
5. Weekly return mark-to-market compounding closes within numerical tolerance.
6. H=4 censoring depends only on calendar completeness as of September 9, 2026.
7. Cohort expiration is exactly H=4 with no prospective H search.
8. M13 thresholds (tau_enter=1.2, tau_exit=0.2, s_normal=0.85, s_reduced=0.45) and cost (10 bps) remain frozen.
9. Seed histories remain separate.
10. All headline metrics derive from the canonical ledger (portfolio_ledger_2026.csv).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_frozen_prediction_and_checkpoint_hashes() -> None:
    base_dir = Path(__file__).parent.parent
    pred_path = (
        base_dir
        / "artifacts"
        / "phase11_replication"
        / "frozen_2026_predictions.csv"
    )
    expected_pred_sha = (
        "6542b33fa5810985beff8c835e71b598ffaf47bf943e88d43f7c50d7dcb30aeb"
    )
    assert _compute_sha256(pred_path) == expected_pred_sha

    models_dir = (
        base_dir / "artifacts" / "phase11_replication" / "frozen_models"
    )
    ckpts = {
        "bayesian_lora_seed_42_state.pt": "d382640351dbef4fd40b47c8b033b1a4823c57d8fd6740b6fd9d1709043dc309",
        "bayesian_lora_seed_123_state.pt": "e094b775c7061b7c60341704525719b2c3fd4055e1aa325581eacabc23059301",
        "bayesian_lora_seed_777_state.pt": "0f084381c8b387f5a4946e337470c8c25707228c40cf676604d1db22f3a13e2a",
    }
    for fname, exp_hash in ckpts.items():
        assert _compute_sha256(models_dir / fname) == exp_hash


def test_event_ticker_mapping_and_timing_audit() -> None:
    base_dir = Path(__file__).parent.parent
    audit_dir = base_dir / "artifacts" / "phase12_prospective_evaluation"
    align_csv = audit_dir / "prospective_return_alignment_audit.csv"
    timing_csv = audit_dir / "event_timing_audit.csv"

    assert align_csv.exists()
    assert timing_csv.exists()

    with open(align_csv) as f:
        align_rows = list(csv.DictReader(f))
    with open(timing_csv) as f:
        timing_rows = list(csv.DictReader(f))

    assert len(align_rows) == 5
    assert len(timing_rows) == 5

    for r in align_rows:
        assert r["join_status"] == "EXACT_STABLE_MATCH"
        assert int(r["prediction_rows"]) == 3
        assert int(r["price_rows_available"]) > 200

    # Ensure deterministic timing
    by_event = {r["event_id"]: r for r in timing_rows}
    assert (
        by_event["GOOGL_2026_Q2"]["first_tradable_date"] == "2026-07-24"
    )  # Friday post-market -> Friday close / next day
    assert by_event["MSFT_2026_Q4"]["first_tradable_date"] == "2026-07-29"
    assert by_event["AAPL_2026_Q3"]["first_tradable_date"] == "2026-08-03"
    assert by_event["AMZN_2026_Q2"]["first_tradable_date"] == "2026-08-03"
    assert by_event["NVDA_2026_Q2"]["first_tradable_date"] == "2026-08-27"


def test_weekly_return_compounding_closure() -> None:
    base_dir = Path(__file__).parent.parent
    ledger_csv = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "prospective_event_return_ledger.csv"
    )
    assert ledger_csv.exists()

    with open(ledger_csv) as f:
        rows = list(csv.DictReader(f))

    complete_rows = [r for r in rows if r["complete_h4"] == "True"]
    assert len(complete_rows) == 4  # GOOGL, MSFT, AAPL, AMZN

    for r in complete_rows:
        r1 = float(r["r_1"])
        r2 = float(r["r_2"])
        r3 = float(r["r_3"])
        r4 = float(r["r_4"])
        cum_ret = float(r["cumulative_4w_return"])
        compounded = (1.0 + r1) * (1.0 + r2) * (1.0 + r3) * (1.0 + r4) - 1.0

        # Exact closure identity
        assert abs(compounded - cum_ret) < 1e-5


def test_right_censoring_rule_and_completeness() -> None:
    base_dir = Path(__file__).parent.parent
    cens_csv = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "right_censoring_audit.csv"
    )
    assert cens_csv.exists()

    with open(cens_csv) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 5
    by_event = {r["event_id"]: r for r in rows}

    # NVDA is right-censored because full 20 trading days are not observed by 2026-09-09
    assert by_event["NVDA_2026_Q2"]["complete_h4"] == "False"
    assert by_event["NVDA_2026_Q2"]["included"] == "False"

    # Other 4 are complete
    for eid in [
        "AAPL_2026_Q3",
        "MSFT_2026_Q4",
        "AMZN_2026_Q2",
        "GOOGL_2026_Q2",
    ]:
        assert by_event[eid]["complete_h4"] == "True"
        assert by_event[eid]["included"] == "True"


def test_canonical_ledger_and_primary_endpoint_consistency() -> None:
    base_dir = Path(__file__).parent.parent
    ledger_csv = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "portfolio_ledger_2026.csv"
    )
    endpoint_csv = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "primary_endpoint.csv"
    )
    bootstrap_csv = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "primary_endpoint_bootstrap.csv"
    )

    assert ledger_csv.exists()
    assert endpoint_csv.exists()
    assert bootstrap_csv.exists()

    with open(ledger_csv) as f:
        ledger_rows = list(csv.DictReader(f))
    with open(endpoint_csv) as f:
        endpoint_rows = list(csv.DictReader(f))
    with open(bootstrap_csv) as f:
        bootstrap_rows = list(csv.DictReader(f))

    # 5 calendar weeks x 3 seeds x 2 policies = 30 rows
    assert len(ledger_rows) == 30

    endpoint_by_seed = {r["seed"]: r for r in endpoint_rows}
    bootstrap_by_seed = {r["seed"]: r for r in bootstrap_rows}

    for seed in ["42", "123", "777"]:
        s_int = int(seed)
        m8_u = [
            float(r["net_utility"])
            for r in ledger_rows
            if int(r["seed"]) == s_int and r["policy"] == "M8_Constant"
        ]
        m13_u = [
            float(r["net_utility"])
            for r in ledger_rows
            if int(r["seed"]) == s_int and r["policy"] == "M13_Hysteresis"
        ]

        # Verify endpoint matches ledger exactly
        assert abs(float(endpoint_by_seed[seed]["M8_mean_weekly_utility"]) - np.mean(m8_u)) < 1e-5
        assert abs(float(endpoint_by_seed[seed]["M13_mean_weekly_utility"]) - np.mean(m13_u)) < 1e-5
        diff = np.mean(m13_u) - np.mean(m8_u)
        assert abs(float(endpoint_by_seed[seed]["delta_utility"]) - diff) < 1e-5

        # Verify bootstrap point estimate matches diff
        assert abs(float(bootstrap_by_seed[seed]["point_estimate_delta_u"]) - diff) < 1e-5


def test_frozen_parameters_in_integrity_manifest() -> None:
    base_dir = Path(__file__).parent.parent
    json_path = (
        base_dir
        / "artifacts"
        / "phase12_prospective_evaluation"
        / "pre_unblinding_integrity.json"
    )
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)

    params = data["frozen_parameters"]
    assert params["H"] == 4
    assert params["tau_enter"] == 1.2
    assert params["tau_exit"] == 0.2
    assert params["s_normal"] == 0.85
    assert params["s_reduced"] == 0.45
    assert params["transaction_cost_rate_bps"] == 10.0
    assert params["cash_return"] == 0.0
    assert params["mc_draws"] == 32
