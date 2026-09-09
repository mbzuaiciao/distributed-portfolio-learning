"""
Tests for Phase 11.2: Final Historical Endpoint Reconciliation Before Unblinding
================================================================================
Verifies:
1. Event-level diagnostic (+0.002519) is explicitly labeled non-canonical and not headline portfolio delta U.
2. Seed aggregation is the exact arithmetic mean of per-seed portfolio endpoints.
3. Old-model and new-model tables use identical portfolio metric definitions and values under H=4.
4. Prospective 2026 prediction file and SHA256 are unchanged byte-for-byte.
5. Prospective model checkpoint hashes are unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_event_diagnostic_not_labeled_canonical_delta_u() -> None:
    base_dir = Path(__file__).parent.parent
    manifest_path = base_dir / "artifacts" / "phase11_replication" / "prospective_freeze_manifest.json"
    with open(manifest_path) as f:
        m = json.load(f)

    recon = m["delta_utility_reconciliation"]
    assert "EVENT-LEVEL DIAGNOSTIC" in recon["event_level_classification"]
    assert "NOT CANONICAL PORTFOLIO ENDPOINT" in recon["event_level_classification"]
    assert recon["canonical_weekly_portfolio_classification"] == "CANONICAL HISTORICAL PORTFOLIO REPLICATION ENDPOINT"
    assert recon["canonical_weekly_portfolio_delta_u_mean"] == 0.000530
    assert recon["event_level_diagnostic_value"] != recon["canonical_weekly_portfolio_delta_u_mean"]


def test_seed_aggregation_is_arithmetic_mean() -> None:
    base_dir = Path(__file__).parent.parent
    recon_csv = base_dir / "artifacts" / "phase11_replication" / "historical_endpoint_reconciliation.csv"
    assert recon_csv.exists(), f"Missing reconciliation CSV: {recon_csv}"

    with open(recon_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by model_generation
    by_gen: dict[str, dict[str, float]] = {}
    for r in rows:
        gen = r["model_generation"]
        if gen not in by_gen:
            by_gen[gen] = {}
        by_gen[gen][r["seed"]] = float(r["delta_utility"])

    for gen, seeds_dict in by_gen.items():
        assert "42" in seeds_dict
        assert "123" in seeds_dict
        assert "777" in seeds_dict
        assert "mean" in seeds_dict

        expected_mean = (seeds_dict["42"] + seeds_dict["123"] + seeds_dict["777"]) / 3.0
        assert abs(seeds_dict["mean"] - expected_mean) < 1e-6, f"Mean mismatch in {gen}: {seeds_dict['mean']} != {expected_mean}"
        if gen in ("historical_lost_model", "synthetic_perturbation_diagnostic"):
            assert abs(seeds_dict["mean"] - 0.000530) < 1e-5
        elif gen == "actual_replicated_model":
            assert abs(seeds_dict["mean"] - 0.000202) < 1e-5


def test_old_and_new_models_identical_portfolio_definitions() -> None:
    base_dir = Path(__file__).parent.parent
    recon_csv = base_dir / "artifacts" / "phase11_replication" / "historical_endpoint_reconciliation.csv"

    with open(recon_csv) as f:
        rows = list(csv.DictReader(f))

    old_rows = {r["seed"]: r for r in rows if r["model_generation"] == "historical_lost_model"}
    synth_rows = {r["seed"]: r for r in rows if r["model_generation"] == "synthetic_perturbation_diagnostic"}
    repl_rows = {r["seed"]: r for r in rows if r["model_generation"] == "actual_replicated_model"}

    assert len(old_rows) == 4
    assert len(synth_rows) == 4
    assert len(repl_rows) == 4

    for s in ["42", "123", "777", "mean"]:
        old_r = old_rows[s]
        synth_r = synth_rows[s]
        repl_r = repl_rows[s]

        # Parameters match across all generations
        for r in [old_r, synth_r, repl_r]:
            assert r["H"] == "4"
            assert r["gamma"] == "1.0"
            assert r["cost_bps"] == "10.0"
            assert r["weeks"] == "69"
            assert r["date_start"] == "2024-01-04"
            assert r["date_end"] == "2025-05-15"

        # Diagnostic matches historical lost model
        assert float(old_r["M8_mean_weekly_utility"]) == float(synth_r["M8_mean_weekly_utility"])
        assert float(old_r["M13_mean_weekly_utility"]) == float(synth_r["M13_mean_weekly_utility"])
        assert float(old_r["delta_utility"]) == float(synth_r["delta_utility"])


def test_prospective_freeze_invariant_preserved() -> None:
    base_dir = Path(__file__).parent.parent
    pred_path = base_dir / "artifacts" / "phase11_replication" / "frozen_2026_predictions.csv"
    canonical_sha = "6542b33fa5810985beff8c835e71b598ffaf47bf943e88d43f7c50d7dcb30aeb"

    # Pred file SHA256 preserved byte-for-byte
    assert _compute_sha256(pred_path) == canonical_sha

    # Model checkpoint hashes unchanged
    ckpts = {
        "bayesian_lora_seed_42_state.pt": "d382640351dbef4fd40b47c8b033b1a4823c57d8fd6740b6fd9d1709043dc309",
        "bayesian_lora_seed_123_state.pt": "e094b775c7061b7c60341704525719b2c3fd4055e1aa325581eacabc23059301",
        "bayesian_lora_seed_777_state.pt": "0f084381c8b387f5a4946e337470c8c25707228c40cf676604d1db22f3a13e2a",
    }
    for filename, expected_sha in ckpts.items():
        p = base_dir / "artifacts" / "phase11_replication" / "frozen_models" / filename
        assert _compute_sha256(p) == expected_sha

    # MC draws == 32 in predictions
    with open(pred_path) as f:
        for r in csv.DictReader(f):
            assert int(r["mc_samples"]) == 32
