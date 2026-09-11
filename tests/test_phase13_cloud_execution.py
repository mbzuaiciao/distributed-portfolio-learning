"""Unit tests for Phase 13 Cloud Execution Addendum.

Verifies:
1. compute_profile.json exists and contains complete local profiling data.
2. Sharded inference preemption-safety, manifest tracking, and atomic file creation.
3. Cloud cost estimation formulas for NVIDIA L4 Spot VM (g2-standard-4).
4. Cloud compute ledger format and data integrity.
5. Strict data minimality: GPU inference does not depend on or access 2026 market returns.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from experiments.cloud.manage_gcp_execution import (
    HOURLY_COST_L4_SPOT,
    estimate_job_cost,
    record_compute_ledger,
)
from experiments.cloud.sharded_inference import (
    ShardedInferenceRunner,
    compute_sha256,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_compute_profile_integrity(repo_root: Path):
    """Verify compute_profile.json has all 4 jobs profiled and valid metrics."""
    profile_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "compute_profile.json"
    assert profile_path.exists(), f"Missing {profile_path}"

    with open(profile_path) as f:
        data = json.load(f)

    # Check hardware specs
    assert "hardware" in data
    assert data["hardware"]["machine"]
    assert data["hardware"]["gpu"]

    # Check Jobs A, B, C, D
    assert "job_A_100_events_mc32" in data
    assert data["job_A_100_events_mc32"]["n_events"] == 100
    assert data["job_A_100_events_mc32"]["wall_time_sec"] > 0
    assert data["job_A_100_events_mc32"]["events_per_sec"] > 0

    assert "job_B_regime_576_events_mc32" in data
    assert data["job_B_regime_576_events_mc32"]["n_events"] == 576
    assert data["job_B_regime_576_events_mc32"]["wall_time_sec"] > 0

    assert "job_C_portfolio_regime" in data
    assert data["job_C_portfolio_regime"]["n_weeks"] == 69
    assert data["job_C_portfolio_regime"]["device"] == "cpu"

    assert "job_D_pytest_suite" in data
    assert data["job_D_pytest_suite"]["exit_code"] == 0
    assert data["job_D_pytest_suite"]["n_tests"] >= 170


def test_cloud_cost_estimation():
    """Verify pre-flight cost estimation logic for NVIDIA L4 Spot instances."""
    # 576 events at 120 events/sec = 4.8s + 300s startup = 304.8s (~0.0847 hr)
    est = estimate_job_cost(576, events_per_sec=120.0, hourly_rate=HOURLY_COST_L4_SPOT)
    assert est["n_events"] == 576
    assert est["est_inference_sec"] < 10.0
    assert est["total_runtime_sec"] > 300.0
    assert est["estimated_cost_usd"] < 0.05  # Pennies on L4 Spot


def test_sharded_manifest_and_idempotency():
    """Verify manifest loading, saving, and resume idempotency."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        runner = ShardedInferenceRunner(output_dir=tmp_path)

        # Initially empty
        assert len(runner.manifest["completed_shards"]) == 0
        assert not runner.is_shard_complete("shard_test_01")

        # Create dummy shard file and record in manifest
        shard_file = runner.shards_dir / "shard_test_01.csv"
        shard_file.write_text("event_id,seed,mean\n1,42,0.5\n")
        sha = compute_sha256(shard_file)

        runner.manifest["completed_shards"]["shard_test_01"] = {
            "filename": "shard_test_01.csv",
            "sha256": sha,
            "n_events": 1,
        }
        runner._save_manifest()

        # Reload new runner from same directory
        runner2 = ShardedInferenceRunner(output_dir=tmp_path)
        assert runner2.is_shard_complete("shard_test_01")
        assert len(runner2.manifest["completed_shards"]) == 1


def test_cloud_compute_ledger_recording():
    """Verify compute ledger logs runs accurately with all required columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_ledger = Path(tmpdir) / "test_ledger.csv"

        record_compute_ledger(
            run_id="test_run_01",
            machine_type="g2-standard-4",
            gpu="NVIDIA-L4",
            spot=True,
            start_time="2026-09-11T00:00:00Z",
            end_time="2026-09-11T00:05:00Z",
            runtime_hours=0.0833,
            estimated_cost=0.0237,
            task="Test batch inference",
            git_sha="test_sha_123",
            ledger_path=tmp_ledger,
        )

        assert tmp_ledger.exists()
        with open(tmp_ledger) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        r = rows[0]
        assert r["run_id"] == "test_run_01"
        assert r["machine_type"] == "g2-standard-4"
        assert r["gpu"] == "NVIDIA-L4"
        assert r["spot"] == "True"
        assert r["estimated_cost"] == "$0.0237"
        assert r["git_sha"] == "test_sha_123"


def test_data_minimality_cloud_inference(repo_root: Path):
    """Verify that cloud inference runner does not import or access price/return data."""
    shard_runner = repo_root / "experiments" / "cloud" / "sharded_inference.py"
    assert shard_runner.exists()

    with open(shard_runner) as f:
        source = f.read()

    forbidden_data_sources = [
        "yfinance",
        "raw_market_data",
        "reconstructed_returns",
        "portfolio_ledger",
        "adj_close",
        "close_price",
        "realized_return",
    ]
    for term in forbidden_data_sources:
        assert (
            term not in source
        ), f"Forbidden market outcome term found in cloud inference runner: {term}"
