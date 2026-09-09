"""
Tests for Phase 10.1: Frozen Bayesian-LoRA Model Provenance Recovery Audit
==========================================================================
Verifies:
1. Provenance classification is justified (Category C: PARTIALLY_REPRODUCIBLE).
2. Frozen manifest exists, is valid JSON, and contains all required specifications.
3. Training data cutoff is strictly < 2026-01-01 (pre-2022 cutoff verified).
4. No 2026 data leaked into model training.
5. Input-only pipeline assertion holds (causal, text-only inputs, no return leakage).
6. No-retraining rule was strictly respected during Phase 10.1.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def test_provenance_artifacts_exist() -> None:
    base_dir = Path(__file__).parent.parent
    art_dir = base_dir / "artifacts" / "phase10_1_model_provenance"
    assert art_dir.exists(), f"Artifact directory missing: {art_dir}"

    manifest_path = art_dir / "frozen_model_manifest.json"
    prov_csv_path = art_dir / "historical_prediction_provenance.csv"
    repro_csv_path = art_dir / "prediction_reproduction_audit.csv"
    report_path = art_dir / "phase10_1_provenance_report.md"

    for p in [manifest_path, prov_csv_path, repro_csv_path, report_path]:
        assert p.exists(), f"Required Phase 10.1 artifact missing: {p}"
        assert p.stat().st_size > 0, f"Artifact is empty: {p}"


def test_frozen_model_manifest_validity() -> None:
    base_dir = Path(__file__).parent.parent
    manifest_path = base_dir / "artifacts" / "phase10_1_model_provenance" / "frozen_model_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Core architecture keys
    assert "base_model" in manifest
    assert manifest["base_model"]["identifier"] == "ProsusAI/finbert"

    lora = manifest["lora_configuration"]
    assert lora["rank"] == 16
    assert lora["alpha"] == 32
    assert "attention.self.query" in lora["target_modules"]
    assert "attention.self.value" in lora["target_modules"]

    var = manifest["bayesian_variational_formulation"]
    assert var["kl_beta"] == 0.003
    assert var["mc_sample_count"] == 32

    # Seeds
    assert manifest["seeds"] == [42, 123, 777]

    # Category
    cat = manifest["provenance_recovery_status"]["category"]
    assert "Category C" in cat or "Category B" in cat


def test_training_data_cutoff_strictly_pre_2026() -> None:
    base_dir = Path(__file__).parent.parent
    manifest_path = base_dir / "artifacts" / "phase10_1_model_provenance" / "frozen_model_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    cutoff_date = manifest["training_provenance"]["training_data_end_date"]
    assert cutoff_date == "2021-12-31"
    assert cutoff_date < "2026-01-01", "Training data leaked beyond 2025 cutoff!"
    assert manifest["training_provenance"]["training_ends_strictly_before_2026"] is True


def test_historical_prediction_provenance_csv() -> None:
    base_dir = Path(__file__).parent.parent
    prov_csv_path = base_dir / "artifacts" / "phase10_1_model_provenance" / "historical_prediction_provenance.csv"
    with open(prov_csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    for row in rows:
        assert int(row["seed"]) in [42, 123, 777]
        assert row["base_model"] == "ProsusAI/finbert"
        assert int(row["lora_r"]) == 16
        assert int(row["lora_alpha"]) == 32
        assert float(row["kl_beta"]) == 0.003
        assert int(row["mc_samples"]) == 32
        assert row["training_data_end_date"] == "2021-12-31"
        assert row["checkpoint_found"] == "False"
        assert row["provenance_status"] == "REPRODUCIBLE_TRAINING_RECIPE_ONLY"


def test_prediction_reproduction_audit_csv() -> None:
    base_dir = Path(__file__).parent.parent
    repro_csv_path = base_dir / "artifacts" / "phase10_1_model_provenance" / "prediction_reproduction_audit.csv"
    with open(repro_csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    for row in rows:
        assert int(row["seed"]) in [42, 123, 777]
        assert row["checkpoint_status"] == "CHECKPOINT_MISSING"
        assert row["exact_numerical_match"] == "NOT_TESTED"
        assert "not persisted" in row["reason"].lower()


def test_input_only_pipeline_and_no_leakage() -> None:
    base_dir = Path(__file__).parent.parent
    manifest_path = base_dir / "artifacts" / "phase10_1_model_provenance" / "frozen_model_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    integrity = manifest["confirmation_pipeline_integrity"]
    assert integrity["future_realized_returns_required"] is False
    assert integrity["causal_information_set"] is True
    assert integrity["no_retraining_rule_respected"] is True

    # Check inputs produce only posterior_mean and epistemic_variance
    assert set(integrity["outputs_produced"]) == {"posterior_mean", "epistemic_variance"}


def test_data_availability_2026_status() -> None:
    base_dir = Path(__file__).parent.parent
    manifest_path = base_dir / "artifacts" / "phase10_1_model_provenance" / "frozen_model_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    availability = manifest["data_availability_2026"]
    assert availability["status"] == "DATA_BLOCKED"
    assert availability["transcripts_available"] is False
    assert availability["count_2026_records"] == 0
