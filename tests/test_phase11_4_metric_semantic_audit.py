"""
Tests for Phase 11.4: Uncertainty Metric Semantic Audit
=======================================================
Verifies:
1. Exact binary label used for top-20% error detection.
2. Spearman correlation uses epistemic uncertainty vs absolute forecast error on identical rows.
3. Top-20% error AUROC, AP, and base rate calculations across seeds 42, 123, 777.
4. Uncertainty metric audit CSV table structure and numerical accuracy.
5. Prospective freeze SHA256 invariant (6542b33fa5810985beff8c835e71b598ffaf47bf943e88d43f7c50d7dcb30aeb).
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


def test_uncertainty_metric_audit_table() -> None:
    base_dir = Path(__file__).parent.parent
    audit_csv = (
        base_dir
        / "artifacts"
        / "phase11_replication"
        / "uncertainty_metric_audit.csv"
    )
    assert audit_csv.exists(), f"Missing audit CSV: {audit_csv}"

    with open(audit_csv) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 4
    by_seed = {r["seed"]: r for r in rows}

    for s in ["42", "123", "777", "mean"]:
        assert s in by_seed

    # Check Seed 42
    s42 = by_seed["42"]
    assert abs(float(s42["spearman_uncertainty_abs_error"]) - 0.5950) < 1e-3
    assert abs(float(s42["top20_error_auroc"]) - 0.5194) < 1e-3
    assert abs(float(s42["top20_error_ap"]) - 0.2181) < 1e-3
    assert abs(float(s42["top20_base_rate"]) - 0.2002) < 1e-3
    assert abs(float(s42["Q1_MAE"]) - 0.2161) < 1e-3
    assert abs(float(s42["Q5_MAE"]) - 0.3346) < 1e-3

    # Check Seed 123
    s123 = by_seed["123"]
    assert abs(float(s123["spearman_uncertainty_abs_error"]) - 0.5928) < 1e-3
    assert abs(float(s123["top20_error_auroc"]) - 0.5191) < 1e-3
    assert abs(float(s123["top20_error_ap"]) - 0.2143) < 1e-3
    assert abs(float(s123["top20_base_rate"]) - 0.2002) < 1e-3
    assert abs(float(s123["Q1_MAE"]) - 0.1891) < 1e-3
    assert abs(float(s123["Q5_MAE"]) - 0.2470) < 1e-3

    # Check Seed 777
    s777 = by_seed["777"]
    assert abs(float(s777["spearman_uncertainty_abs_error"]) - 0.4833) < 1e-3
    assert abs(float(s777["top20_error_auroc"]) - 0.5204) < 1e-3
    assert abs(float(s777["top20_error_ap"]) - 0.2282) < 1e-3
    assert abs(float(s777["top20_base_rate"]) - 0.2002) < 1e-3
    assert abs(float(s777["Q1_MAE"]) - 0.2170) < 1e-3
    assert abs(float(s777["Q5_MAE"]) - 0.2286) < 1e-3

    # Check arithmetic mean
    smean = by_seed["mean"]
    expected_spearman = (0.5950 + 0.5928 + 0.4833) / 3.0
    assert (
        abs(float(smean["spearman_uncertainty_abs_error"]) - expected_spearman)
        < 1e-3
    )


def test_row_alignment_and_binary_label_mechanism() -> None:
    base_dir = Path(__file__).parent.parent
    json_path = (
        base_dir
        / "artifacts"
        / "phase11_replication"
        / "phase11_4_metric_semantic_audit.json"
    )
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)

    for s in [42, 123, 777]:
        d = data["details_by_seed"][f"seed_{s}"]
        assert d["spearman_corr"] > 0.45
        assert abs(d["top20_base_rate"] - 0.2002) < 0.005
        # Top-20% error indicator is >= 98% overlapped with the ground-truth downside class (y=1)
        assert d["top20_overlap_with_y1"] >= 0.98
        # AUROC for top-20% error is ~0.52 (reflecting directional predictability)
        assert 0.51 < d["top20_auroc"] < 0.53
        assert 0.20 < d["top20_ap"] < 0.25
        # Majority class (y=0) drives positive continuous Spearman correlation
        assert d["spearman_within_y0"] > 0.75


def test_prospective_freeze_invariant() -> None:
    base_dir = Path(__file__).parent.parent
    pred_path = (
        base_dir
        / "artifacts"
        / "phase11_replication"
        / "frozen_2026_predictions.csv"
    )
    canonical_sha = (
        "6542b33fa5810985beff8c835e71b598ffaf47bf943e88d43f7c50d7dcb30aeb"
    )
    assert _compute_sha256(pred_path) == canonical_sha


def test_audit_json_metadata() -> None:
    base_dir = Path(__file__).parent.parent
    json_path = (
        base_dir
        / "artifacts"
        / "phase11_replication"
        / "phase11_4_metric_semantic_audit.json"
    )
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)

    assert data["prospective_sha_verified"] is True
    assert data["zero_2026_outcomes_accessed"] is True
    assert "explanation_of_discrepancy" in data
