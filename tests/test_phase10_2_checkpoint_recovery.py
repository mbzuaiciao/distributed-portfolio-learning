"""
Tests for Phase 10.2: Original Checkpoint Recovery Search Audit
===============================================================
Verifies:
1. Candidate checkpoint search artifact exists with comprehensive audit entries.
2. Structural incompatibility of candidate adapter (r=8 deterministic vs r=16 Bayesian) is documented.
3. Checkpoint verification audit confirms Outcome D across all historical seeds (42, 123, 777).
4. Required explicit statement is recorded exactly:
   "Exact prospective continuation of the historical fitted Bayesian-LoRA models is unavailable."
5. No models were retrained during Phase 10.2.
"""

from __future__ import annotations

import csv
from pathlib import Path


def test_phase10_2_artifacts_exist() -> None:
    base_dir = Path(__file__).parent.parent
    art_dir = base_dir / "artifacts" / "phase10_2_checkpoint_recovery"
    assert art_dir.exists(), f"Artifact directory missing: {art_dir}"

    cand_csv = art_dir / "candidate_checkpoint_search.csv"
    verif_csv = art_dir / "checkpoint_verification_audit.csv"
    report_md = art_dir / "phase10_2_recovery_report.md"

    for p in [cand_csv, verif_csv, report_md]:
        assert p.exists(), f"Missing artifact: {p}"
        assert p.stat().st_size > 0, f"Empty artifact: {p}"


def test_candidate_search_audits_adapter() -> None:
    base_dir = Path(__file__).parent.parent
    cand_csv = base_dir / "artifacts" / "phase10_2_checkpoint_recovery" / "candidate_checkpoint_search.csv"
    with open(cand_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) >= 1
    # Check that adapter_model.safetensors is audited
    adapter_rows = [r for r in rows if r["filename"] == "adapter_model.safetensors"]
    assert len(adapter_rows) == 1
    row = adapter_rows[0]
    assert "r=8" in row["compatibility_status"]
    assert "r=16" in row["compatibility_status"]
    assert row["reproduction_test_result"] == "FAILED_STRUCTURAL_MISMATCH"


def test_checkpoint_verification_audit_outcome_d() -> None:
    base_dir = Path(__file__).parent.parent
    verif_csv = base_dir / "artifacts" / "phase10_2_checkpoint_recovery" / "checkpoint_verification_audit.csv"
    with open(verif_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    mandatory_statement = "Exact prospective continuation of the historical fitted Bayesian-LoRA models is unavailable."
    for r in rows:
        assert int(r["seed"]) in [42, 123, 777]
        assert r["search_status"] == "NOT_FOUND_IN_ANY_LOCATION"
        assert r["reproduction_status"] == "UNAVAILABLE"
        assert r["explicit_determination"] == mandatory_statement


def test_recovery_report_contains_mandatory_statement() -> None:
    base_dir = Path(__file__).parent.parent
    report_md = base_dir / "artifacts" / "phase10_2_checkpoint_recovery" / "phase10_2_recovery_report.md"
    content = report_md.read_text()

    mandatory_statement = "Exact prospective continuation of the historical fitted Bayesian-LoRA models is unavailable."
    assert mandatory_statement in content
    assert "Outcome D" in content


def test_no_retraining_in_phase10_2() -> None:
    base_dir = Path(__file__).parent.parent
    # Ensure no checkpoint .pt files were created in the repo
    pt_files = list(base_dir.glob("**/*.pt"))
    assert len(pt_files) == 0, f"Found unexpected model checkpoint file: {pt_files}"
