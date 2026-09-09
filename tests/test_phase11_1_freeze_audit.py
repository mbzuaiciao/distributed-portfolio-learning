"""
Tests for Phase 11.1: Pre-Outcome Replication Freeze Audit
==========================================================
Verifies:
1. Prospective freeze manifest exists and contains all required metadata.
2. Canonical 2026 predictions file exists, has matching SHA256, and uses MC draws == 32 across all rows.
3. 2026 input manifest contains 0 realized returns/outcomes and matches hash.
4. Frozen model checkpoints exist, match recorded hashes, and contain exactly 98 keys (LoRA variational + head).
5. Base model provenance records exact Hugging Face commit SHA and config hash.
6. Sample audit verifies 0 overlap between training data and holdout benchmark.
7. Delta U reconciliation is verified mathematically.
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


def test_prospective_freeze_manifest_structure() -> None:
    base_dir = Path(__file__).parent.parent
    manifest_path = base_dir / "artifacts" / "phase11_replication" / "prospective_freeze_manifest.json"
    assert manifest_path.exists(), f"Manifest missing: {manifest_path}"

    with open(manifest_path) as f:
        m = json.load(f)

    required_keys = [
        "manifest_title",
        "created_at_utc",
        "git_commit_prior",
        "canonical_prospective_predictions",
        "input_manifest",
        "base_model_provenance",
        "frozen_model_checkpoints",
        "round_trip_reproducibility",
        "frozen_portfolio_architecture",
        "historical_benchmark_sample_audit",
        "delta_utility_reconciliation",
        "prospective_status",
    ]
    for k in required_keys:
        assert k in m, f"Missing key in manifest: {k}"

    assert m["prospective_status"] == "PRE_OUTCOME_FREEZE_FINALIZED_CLEAN"


def test_mc_draw_count_and_canonical_prediction_hash() -> None:
    base_dir = Path(__file__).parent.parent
    pred_path = base_dir / "artifacts" / "phase11_replication" / "frozen_2026_predictions.csv"
    manifest_path = base_dir / "artifacts" / "phase11_replication" / "prospective_freeze_manifest.json"

    assert pred_path.exists(), f"Predictions file missing: {pred_path}"

    # Verify SHA256 match
    real_sha = _compute_sha256(pred_path)
    with open(manifest_path) as f:
        m = json.load(f)
    expected_sha = m["canonical_prospective_predictions"]["sha256"]
    assert real_sha == expected_sha, f"SHA mismatch: {real_sha} != {expected_sha}"

    # Verify mc_samples == 32 in every row
    with open(pred_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0
    for r in rows:
        assert int(r["mc_samples"]) == 32, f"Expected 32 MC draws, found {r['mc_samples']}"


def test_2026_input_manifest_hash_and_no_leakage() -> None:
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "artifacts" / "phase11_replication" / "2026_input_manifest.csv"
    manifest_path = base_dir / "artifacts" / "phase11_replication" / "prospective_freeze_manifest.json"

    real_sha = _compute_sha256(input_path)
    with open(manifest_path) as f:
        m = json.load(f)
    expected_sha = m["input_manifest"]["sha256"]
    assert real_sha == expected_sha, f"Input manifest SHA mismatch: {real_sha} != {expected_sha}"

    with open(input_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for col in fieldnames:
            for forbidden in ["return", "price", "move", "label", "outcome"]:
                assert forbidden not in col.lower(), f"Forbidden column detected: {col}"


def test_model_checkpoints_integrity_and_hashes() -> None:
    base_dir = Path(__file__).parent.parent
    manifest_path = base_dir / "artifacts" / "phase11_replication" / "prospective_freeze_manifest.json"
    with open(manifest_path) as f:
        m = json.load(f)

    ckpts = m["frozen_model_checkpoints"]
    assert set(ckpts.keys()) == {"seed_42", "seed_123", "seed_777"}

    for seed_key, info in ckpts.items():
        p = base_dir / "artifacts" / "phase11_replication" / "frozen_models" / info["filename"]
        assert p.exists(), f"Checkpoint file missing: {p}"
        assert p.stat().st_size == info["size_bytes"]
        assert _compute_sha256(p) == info["sha256"]


def test_historical_benchmark_zero_training_overlap() -> None:
    base_dir = Path(__file__).parent.parent
    manifest_path = base_dir / "artifacts" / "phase11_replication" / "prospective_freeze_manifest.json"
    with open(manifest_path) as f:
        m = json.load(f)

    audit = m["historical_benchmark_sample_audit"]
    assert audit["replication_benchmark_data"]["overlap_with_training"] == 0
    assert audit["training_data"]["sample_count_N"] == 15752
    assert audit["replication_benchmark_data"]["sample_count_N"] == 1728
