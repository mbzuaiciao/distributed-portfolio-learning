"""Collaborative LoRA Transferability Benchmark Runner.

Implements:
1. Stage generate: Synthesizes and saves positive and negative control datasets.
2. Stage benchmark: Executes a bounded 1-seed benchmark to measure CPU latency and memory.
3. Locked stages (train, evaluate_dev, evaluate_test) awaiting explicit user authorization.

Enforces CPU-only execution with PyTorch limited to two threads.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from hypergraph_portfolio.multitask_transfer_dgp import (
    CalibratedLogOddsFusion3,
    PooledLoRA,
    SharedSubspaceMultiTask,
    StructuralCoLoRA,
    TargetOnlyLogistic,
    TargetOnlyLoRA,
    compute_principal_angles,
    compute_subspace_alignment,
    compute_weight_cosine_similarity,
    generate_multitask_transfer_dgp,
)

# Host safeguard: strictly limit PyTorch CPU threads to 2
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts/multitask_transfer_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_peak_memory_mb() -> float:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return float(peak_rss / (1024 * 1024))
    return float(peak_rss / 1024)


def save_json(path: Path, obj: Any) -> None:
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def stage_generate() -> Dict[str, Any]:
    """Generate datasets for both Positive and Negative controls."""
    print("Generating Multi-Task LoRA Transferability Datasets...")
    t0 = time.perf_counter()

    # Generate Positive Control
    pos_data = generate_multitask_transfer_dgp(seed=42, regime="positive_control")
    np.savez_compressed(
        OUT_DIR / "dataset_transfer_positive_control.npz",
        U_true=pos_data["U_true"],
        beta_s1=pos_data["betas"]["source_1"],
        beta_s2=pos_data["betas"]["source_2"],
        beta_s3=pos_data["betas"]["source_3"],
        beta_target=pos_data["betas"]["target"],
        s1_X=pos_data["source_data"]["source_1"]["X"],
        s1_y=pos_data["source_data"]["source_1"]["y"],
        s2_X=pos_data["source_data"]["source_2"]["X"],
        s2_y=pos_data["source_data"]["source_2"]["y"],
        s3_X=pos_data["source_data"]["source_3"]["X"],
        s3_y=pos_data["source_data"]["source_3"]["y"],
        target_train_X=pos_data["target_train"]["X"],
        target_train_y=pos_data["target_train"]["y"],
        target_cal_X=pos_data["target_cal"]["X"],
        target_cal_y=pos_data["target_cal"]["y"],
        target_dev_X=pos_data["target_dev"]["X"],
        target_dev_y=pos_data["target_dev"]["y"],
        target_test_X=pos_data["target_test"]["X"],
        target_test_y=pos_data["target_test"]["y"],
    )

    # Generate Negative Control
    neg_data = generate_multitask_transfer_dgp(seed=42, regime="negative_control")
    np.savez_compressed(
        OUT_DIR / "dataset_transfer_negative_control.npz",
        beta_s1=neg_data["betas"]["source_1"],
        beta_s2=neg_data["betas"]["source_2"],
        beta_s3=neg_data["betas"]["source_3"],
        beta_target=neg_data["betas"]["target"],
        s1_X=neg_data["source_data"]["source_1"]["X"],
        s1_y=neg_data["source_data"]["source_1"]["y"],
        s2_X=neg_data["source_data"]["source_2"]["X"],
        s2_y=neg_data["source_data"]["source_2"]["y"],
        s3_X=neg_data["source_data"]["source_3"]["X"],
        s3_y=neg_data["source_data"]["source_3"]["y"],
        target_train_X=neg_data["target_train"]["X"],
        target_train_y=neg_data["target_train"]["y"],
        target_cal_X=neg_data["target_cal"]["X"],
        target_cal_y=neg_data["target_cal"]["y"],
        target_dev_X=neg_data["target_dev"]["X"],
        target_dev_y=neg_data["target_dev"]["y"],
        target_test_X=neg_data["target_test"]["X"],
        target_test_y=neg_data["target_test"]["y"],
    )

    summary = {
        "status": "DATASETS_GENERATED",
        "elapsed_seconds": time.perf_counter() - t0,
        "peak_memory_mb": get_peak_memory_mb(),
        "splits": {
            "source_1_train": 512,
            "source_2_train": 512,
            "source_3_train": 512,
            "target_train": 64,
            "target_cal": 128,
            "target_dev": 256,
            "target_test_quarantined": 512,
        },
    }
    save_json(OUT_DIR / "generation_summary.json", summary)
    print(f"Data generation complete in {summary['elapsed_seconds']:.2f} s. Peak RSS: {summary['peak_memory_mb']:.1f} MB.")
    return summary


def train_balanced_multitask_step(
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    model: nn.Module,
    s1_X: torch.Tensor,
    s1_y: torch.Tensor,
    s2_X: torch.Tensor,
    s2_y: torch.Tensor,
    s3_X: torch.Tensor,
    s3_y: torch.Tensor,
    target_X: torch.Tensor,
    target_y: torch.Tensor,
    batch_size: int = 16,
) -> float:
    """Execute a single balanced multi-task training step with equal sampling across 4 tasks."""
    optimizer.zero_grad()

    # Sample batch_size from each task
    idx_s1 = torch.randint(0, len(s1_y), (batch_size,))
    idx_s2 = torch.randint(0, len(s2_y), (batch_size,))
    idx_s3 = torch.randint(0, len(s3_y), (batch_size,))
    idx_tgt = torch.randint(0, len(target_y), (batch_size,))  # Sampled with replacement

    loss1 = criterion(model.forward_task(s1_X[idx_s1], 0), s1_y[idx_s1])
    loss2 = criterion(model.forward_task(s2_X[idx_s2], 1), s2_y[idx_s2])
    loss3 = criterion(model.forward_task(s3_X[idx_s3], 2), s3_y[idx_s3])
    loss_tgt = criterion(model.forward_task(target_X[idx_tgt], 3), target_y[idx_tgt])

    total_loss = 0.25 * (loss1 + loss2 + loss3 + loss_tgt)
    total_loss.backward()
    optimizer.step()
    return float(total_loss.item())


def run_bounded_benchmark() -> Dict[str, Any]:
    """Execute a single-seed benchmark of all 6 models across 500 gradient steps."""
    print("Starting Bounded 1-Seed Benchmark for Multi-Task LoRA Transferability...")
    t0 = time.perf_counter()

    data = generate_multitask_transfer_dgp(seed=42, regime="positive_control")
    target_train_X = torch.from_numpy(data["target_train"]["X"]).float()
    target_train_y = torch.from_numpy(data["target_train"]["y"]).float()
    s1_X = torch.from_numpy(data["source_data"]["source_1"]["X"]).float()
    s1_y = torch.from_numpy(data["source_data"]["source_1"]["y"]).float()
    s2_X = torch.from_numpy(data["source_data"]["source_2"]["X"]).float()
    s2_y = torch.from_numpy(data["source_data"]["source_2"]["y"]).float()
    s3_X = torch.from_numpy(data["source_data"]["source_3"]["X"]).float()
    s3_y = torch.from_numpy(data["source_data"]["source_3"]["y"]).float()

    criterion = nn.BCEWithLogitsLoss()
    timings = {}

    # 1. Target-Only Logistic (M0)
    t_start = time.perf_counter()
    m0 = TargetOnlyLogistic(in_features=32)
    opt0 = torch.optim.AdamW(m0.parameters(), lr=0.01, weight_decay=0.01)
    for _ in range(500):
        opt0.zero_grad()
        idx = torch.randint(0, len(target_train_y), (16,))
        loss = criterion(m0(target_train_X[idx]), target_train_y[idx])
        loss.backward()
        opt0.step()
    timings["target_only_logistic"] = time.perf_counter() - t_start

    # 2. Target-Only LoRA (MLoRA0)
    t_start = time.perf_counter()
    m_lora = TargetOnlyLoRA(in_features=32, rank=2)
    opt_lora = torch.optim.AdamW(m_lora.parameters(), lr=0.01, weight_decay=0.01)
    for _ in range(500):
        opt_lora.zero_grad()
        idx = torch.randint(0, len(target_train_y), (16,))
        loss = criterion(m_lora(target_train_X[idx]), target_train_y[idx])
        loss.backward()
        opt_lora.step()
    timings["target_only_lora"] = time.perf_counter() - t_start

    # 3. Pooled LoRA (M1)
    t_start = time.perf_counter()
    m_pooled = PooledLoRA(in_features=32, rank=2)
    opt_pooled = torch.optim.AdamW(m_pooled.parameters(), lr=0.01, weight_decay=0.01)
    all_X = torch.cat([s1_X, s2_X, s3_X, target_train_X], dim=0)
    all_y = torch.cat([s1_y, s2_y, s3_y, target_train_y], dim=0)
    for _ in range(500):
        opt_pooled.zero_grad()
        idx = torch.randint(0, len(all_y), (64,))
        loss = criterion(m_pooled(all_X[idx]), all_y[idx])
        loss.backward()
        opt_pooled.step()
    timings["pooled_lora"] = time.perf_counter() - t_start

    # 4. 3-Source Independent Specialists + Calibrated Fusion (M_fuse)
    t_start = time.perf_counter()
    source_models = []
    for s_X, s_y in [(s1_X, s1_y), (s2_X, s2_y), (s3_X, s3_y)]:
        m_s = TargetOnlyLoRA(in_features=32, rank=2)
        opt_s = torch.optim.AdamW(m_s.parameters(), lr=0.01, weight_decay=0.01)
        for _ in range(500):
            opt_s.zero_grad()
            idx = torch.randint(0, len(s_y), (16,))
            loss = criterion(m_s(s_X[idx]), s_y[idx])
            loss.backward()
            opt_s.step()
        source_models.append(m_s)
    # Fit fusion on target cal split
    cal_p1 = source_models[0].predict_proba(data["target_cal"]["X"])
    cal_p2 = source_models[1].predict_proba(data["target_cal"]["X"])
    cal_p3 = source_models[2].predict_proba(data["target_cal"]["X"])
    p_sources_cal = np.column_stack([cal_p1, cal_p2, cal_p3])
    m_fuse = CalibratedLogOddsFusion3.fit(p_sources_cal, data["target_cal"]["y"])
    timings["independent_specialists_and_fusion"] = time.perf_counter() - t_start

    # 5. Shared-Subspace Multi-Task (M_subspace)
    t_start = time.perf_counter()
    m_subspace = SharedSubspaceMultiTask(in_features=32, rank=2, n_tasks=4)
    opt_sub = torch.optim.AdamW(m_subspace.parameters(), lr=0.01, weight_decay=0.01)
    for _ in range(500):
        train_balanced_multitask_step(
            opt_sub, criterion, m_subspace, s1_X, s1_y, s2_X, s2_y, s3_X, s3_y, target_train_X, target_train_y
        )
    timings["shared_subspace_multitask"] = time.perf_counter() - t_start

    # 6. Structural CoLoRA (M_CoLoRA)
    t_start = time.perf_counter()
    m_colora = StructuralCoLoRA(in_features=32, rank=2, n_tasks=4)
    opt_col = torch.optim.AdamW(m_colora.parameters(), lr=0.01, weight_decay=0.01)
    for _ in range(500):
        train_balanced_multitask_step(
            opt_col, criterion, m_colora, s1_X, s1_y, s2_X, s2_y, s3_X, s3_y, target_train_X, target_train_y
        )
    timings["structural_colora"] = time.perf_counter() - t_start

    total_bench_seconds = time.perf_counter() - t0
    peak_mem = get_peak_memory_mb()

    # Projected runtime across 2 regimes (Positive & Negative) and 3 seeds (42, 123, 777)
    # Total runs = 2 regimes * 3 seeds = 6 full evaluations
    projected_full_seconds = 6 * total_bench_seconds
    projected_full_minutes = projected_full_seconds / 60.0

    benchmark_report = {
        "single_replication_all_models_seconds": total_bench_seconds,
        "timings_by_model": timings,
        "projected_full_experiment_seconds": projected_full_seconds,
        "projected_full_experiment_minutes": projected_full_minutes,
        "peak_memory_mb": peak_mem,
        "safe_to_proceed": bool(projected_full_minutes < 20.0),
        "host_platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
    }

    save_json(OUT_DIR / "bounded_benchmark_transfer.json", benchmark_report)

    print("\n================ BOUNDED BENCHMARK RESULTS ================")
    print(f"Single Replication (6 Models, 500 steps each): {total_bench_seconds:.2f} s")
    for m_name, t_sec in timings.items():
        print(f"  - {m_name}: {t_sec:.3f} s")
    print(f"Projected 6-run Full Experiment (2 Regimes x 3 Seeds): {projected_full_minutes:.2f} min (Ceiling: 20 min)")
    print(f"Peak Memory (RSS): {peak_mem:.1f} MB (Ceiling: 4000 MB)")
    print(f"Safe to proceed: {benchmark_report['safe_to_proceed']}")
    print("===========================================================\n")

    return benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Task LoRA Transferability Harness")
    parser.add_argument(
        "--stage",
        choices=["generate", "benchmark", "train", "evaluate_dev", "evaluate_test"],
        required=True,
    )
    args = parser.parse_args()

    if args.stage == "generate":
        stage_generate()
    elif args.stage == "benchmark":
        run_bounded_benchmark()
    elif args.stage in ["train", "evaluate_dev"]:
        print(f"Stage '{args.stage}' is prospectively defined but locked pending explicit user authorization.")
    elif args.stage == "evaluate_test":
        print("Stage 'evaluate_test' is strictly quarantined and locked.")


if __name__ == "__main__":
    main()
