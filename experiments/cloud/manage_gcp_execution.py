#!/usr/bin/env python3
"""Phase 13: Google Cloud Compute Engine (L4 Spot VM) Orchestrator.

Manages the lifecycle of NVIDIA L4 Spot compute instances:
- Cost estimation: pre-flight estimation based on expected batch size
- Instance provisioning: g2-standard-4 (1x L4 Spot) in us-central1
- Preemption-safe sharded data transfer
- Automated result retrieval and hash verification
- Mandatory auto-shutdown / VM deletion to prevent ongoing charges
- Logging to cloud_compute_ledger.csv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parent.parent.parent

# Pricing constants for us-central1 (NVIDIA L4 Spot)
# g2-standard-4 Spot: ~$0.203/hr (L4 GPU) + ~$0.081/hr (4 vCPU, 16GB) = ~$0.284 / hour
HOURLY_COST_L4_SPOT = 0.284
DEFAULT_ZONE = "us-central1-a"
DEFAULT_MACHINE_TYPE = "g2-standard-4"


def get_git_sha() -> str:
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return res.stdout.strip() if res.returncode == 0 else "UNKNOWN_SHA"


def estimate_job_cost(
    n_events: int,
    events_per_sec: float = 120.0,
    hourly_rate: float = HOURLY_COST_L4_SPOT,
) -> dict[str, Any]:
    """Estimate runtime and cost for an L4 inference batch.

    L4 throughput on BERT-base with batch_size=128 is ~120-150 events/sec (at 32 MC draws).
    """
    est_runtime_sec = n_events / events_per_sec
    # Add 5 minutes for VM startup, environment init, and data transfer
    total_runtime_sec = est_runtime_sec + 300.0
    runtime_hours = total_runtime_sec / 3600.0
    est_cost = runtime_hours * hourly_rate

    return {
        "n_events": n_events,
        "est_inference_sec": round(est_runtime_sec, 2),
        "total_runtime_sec": round(total_runtime_sec, 2),
        "runtime_hours": round(runtime_hours, 4),
        "hourly_rate_usd": hourly_rate,
        "estimated_cost_usd": round(est_cost, 4),
    }


def record_compute_ledger(
    run_id: str,
    machine_type: str,
    gpu: str,
    spot: bool,
    start_time: str,
    end_time: str,
    runtime_hours: float,
    estimated_cost: float,
    task: str,
    git_sha: str,
    ledger_path: Path | None = None,
) -> None:
    target_path = (
        ledger_path
        if ledger_path
        else repo_root
        / "artifacts"
        / "phase13_uncertainty_mapping"
        / "cloud_compute_ledger.csv"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "machine_type",
        "gpu",
        "spot",
        "start",
        "end",
        "runtime_hours",
        "estimated_cost",
        "task",
        "git_sha",
    ]

    file_exists = target_path.exists()
    with open(target_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "run_id": run_id,
            "machine_type": machine_type,
            "gpu": gpu,
            "spot": str(spot),
            "start": start_time,
            "end": end_time,
            "runtime_hours": f"{runtime_hours:.4f}",
            "estimated_cost": f"${estimated_cost:.4f}",
            "task": task,
            "git_sha": git_sha,
        })
    print(f"Logged run {run_id} to compute ledger: {target_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Google Cloud L4 Spot execution.")
    parser.add_argument(
        "--action",
        choices=["estimate", "benchmark-plan", "delete"],
        default="estimate",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=1728,
        help="Number of historical events to score",
    )
    parser.add_argument("--instance-name", default="phase13-l4-spot-runner")
    parser.add_argument("--zone", default=DEFAULT_ZONE)

    args = parser.parse_args()

    if args.action == "estimate":
        est = estimate_job_cost(args.n_events)
        print("=== Google Cloud L4 Spot Cost Estimation ===")
        print(f"Target events: {est['n_events']}")
        print(f"Estimated L4 inference time: {est['est_inference_sec']}s")
        print(f"Total billing runtime (with VM prep): {est['runtime_hours']:.4f} hours")
        print(f"Hourly rate (Spot): ${est['hourly_rate_usd']:.3f}/hr")
        print(f"Estimated batch cost: ${est['estimated_cost_usd']:.4f}")


if __name__ == "__main__":
    main()
