"""Experiment 3: Bayesian Uncertainty-Aware Specialist Routing Harness.

Implements the prospective protocol defined in
docs/bayesian_collaboration_experiment3_plan.md and
docs/bayesian_collaboration_experiment3_amendment.md.

Enforces CPU-only execution with PyTorch strictly limited to two threads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from hypergraph_portfolio.bayesian_uncertainty import (
    BayesianPenultimateReadout,
    ConfidenceRouter,
    DensityRouter,
    InverseVarianceRouter,
    LinearMoERouter,
    MLPMoERouter,
    MatchedAblationRouter,
    select_strongest_nonbayes_baseline,
)
from hypergraph_portfolio.calibration_fusion import LogOddsFusion, PlattCalibrator
from hypergraph_portfolio.statistical_testing import (
    apply_holm_bonferroni,
    paired_contrast_bootstrap,
    paired_instance_bootstrap,
)
from hypergraph_portfolio.synthetic_models import CapacityMatchedMLP, SmallMLP
from hypergraph_portfolio.synthetic_multitask import DGPConfig, _sigmoid, generate_dgp
from hypergraph_portfolio.synthetic_multitask_exp3 import (
    Exp3SplitConfig,
    sample_exp3_dataset,
)

# Host safeguard: strictly limit PyTorch CPU threads to 2
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts/bayesian_collab_exp3"


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: Path, obj: Any) -> None:
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute calibration and discrimination metrics."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)

    ece = 0.0
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        mask = (p >= lo) & (p < hi)
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y_true[mask].mean()))

    brier = float(np.mean((p - y_true) ** 2))
    log_loss = float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    try:
        auroc = float(roc_auc_score(y_true, p))
    except Exception:
        auroc = 0.5

    try:
        ap = float(average_precision_score(y_true, p))
    except Exception:
        ap = float(y_true.mean())

    accuracy = float(np.mean((p >= 0.5) == y_true))

    return {
        "brier": brier,
        "log_loss": log_loss,
        "auroc": auroc,
        "average_precision": ap,
        "accuracy": accuracy,
        "ece": ece,
    }


def train_single_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.005,
    weight_decay: float = 0.01,
    seed: int = 42,
    model_cls: type = SmallMLP,
) -> nn.Module:
    torch.manual_seed(seed)
    in_features = x_train.shape[1]
    model = model_cls(in_features=in_features)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    criterion = nn.BCELoss()

    xt = torch.from_numpy(x_train).float()
    yt = torch.from_numpy(y_train).float()
    n = len(xt)
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            bx, by = xt[idx], yt[idx]
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    return model


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    xt = torch.from_numpy(x).float()
    with torch.no_grad():
        preds = model(xt).cpu().numpy()
    return preds


def stage_generate() -> None:
    """Stage 1: Generate synthetic datasets for Regime A and Regime B."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = DGPConfig(seed=20261014)
    spec = generate_dgp(cfg)
    split_cfg = Exp3SplitConfig(data_seed=20261015)

    save_json(OUT_DIR / "dgp_spec_exp3.json", spec.to_dict())

    for regime in ("regime_A", "regime_B"):
        splits = sample_exp3_dataset(spec, regime=regime, split_cfg=split_cfg)
        np.savez_compressed(
            OUT_DIR / f"splits_exp3_{regime}.npz",
            target_train_x=splits.target_train["x"],
            target_train_y=splits.target_train["y"],
            spec_A_train_x=splits.specialist_A_train["x"],
            spec_A_train_y=splits.specialist_A_train["y"],
            spec_B_train_x=splits.specialist_B_train["x"],
            spec_B_train_y=splits.specialist_B_train["y"],
            cal_x_full=splits.calibration["x_full"],
            cal_x_A=splits.calibration["x_A"],
            cal_x_B=splits.calibration["x_B"],
            cal_y=splits.calibration["y"],
            fus_x_full=splits.fusion_dev["x_full"],
            fus_x_A=splits.fusion_dev["x_A"],
            fus_x_B=splits.fusion_dev["x_B"],
            fus_y=splits.fusion_dev["y"],
            dev_x_full=splits.development["x_full"],
            dev_x_A=splits.development["x_A"],
            dev_x_B=splits.development["x_B"],
            dev_y=splits.development["y"],
            test_x_full=splits.test["x_full"],
            test_x_A=splits.test["x_A"],
            test_x_B=splits.test["x_B"],
            test_y=splits.test["y"],
        )

    config_record = {
        "seeds": [42, 123, 777],
        "dgp": asdict(cfg),
        "split_config": asdict(split_cfg),
        "regimes": ["regime_A", "regime_B"],
        "material_brier_gain": 0.010,
    }
    save_json(OUT_DIR / "config_exp3.json", config_record)
    print("Exp 3 datasets generated successfully for Regimes A and B.")


def stage_benchmark() -> Dict[str, Any]:
    """Stage Benchmark: Short bounded CPU benchmark to measure runtime and RAM."""
    if not (OUT_DIR / "splits_exp3_regime_A.npz").exists():
        stage_generate()

    start_time = time.perf_counter()
    data = np.load(OUT_DIR / "splits_exp3_regime_A.npz")

    # Run smoke training for 1 seed across 10 epochs
    print("Running bounded CPU benchmark (1 seed, 10 epochs)...")
    m0 = train_single_model(data["target_train_x"], data["target_train_y"], epochs=10, seed=42)
    m2 = train_single_model(data["spec_A_train_x"], data["spec_A_train_y"], epochs=10, seed=43)
    m3 = train_single_model(data["spec_B_train_x"], data["spec_B_train_y"], epochs=10, seed=44)

    # Fit Laplace readout
    readout_A = BayesianPenultimateReadout.fit(m2, data["spec_A_train_x"])
    readout_B = BayesianPenultimateReadout.fit(m3, data["spec_B_train_x"])

    # Predictions and router smoke fits
    pA_fus = predict(m2, data["fus_x_A"])
    pB_fus = predict(m3, data["fus_x_B"])
    xs_fus = data["fus_x_full"][:, :4]
    fus_y = data["fus_y"]

    moe_lin = LinearMoERouter.fit(xs_fus, pA_fus, pB_fus, fus_y, epochs=10, seed=42)
    moe_mlp = MLPMoERouter.fit(xs_fus, pA_fus, pB_fus, fus_y, epochs=10, seed=42)

    elapsed = time.perf_counter() - start_time
    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    # 10 epochs took `elapsed`. 100 epochs for 3 seeds and 2 regimes is ~ 60x smoke run
    projected_full_runtime_sec = elapsed * 10.0 * 6.0

    benchmark_results = {
        "benchmark_wall_seconds": elapsed,
        "peak_rss_mb": max_rss_mb,
        "projected_full_training_runtime_seconds": projected_full_runtime_sec,
        "within_20_minute_limit": bool(projected_full_runtime_sec < 1200.0),
        "host_platform": platform.platform(),
        "pytorch_threads": torch.get_num_threads(),
    }
    save_json(OUT_DIR / "benchmark_results_exp3.json", benchmark_results)

    print("\n================ EXPERIMENT 3 BENCHMARK PROFILE ================")
    print(f"Benchmark Execution Time:       {elapsed:.2f} s")
    print(f"Peak Resident Memory:           {max_rss_mb:.2f} MB")
    print(f"Projected Full Runtime:         {projected_full_runtime_sec:.1f} s (~{projected_full_runtime_sec/60:.2f} min)")
    print(f"Well Within 20-min Safety Cap:  {benchmark_results['within_20_minute_limit']}")
    print("================================================================")
    return benchmark_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3 Harness")
    parser.add_argument(
        "--stage",
        choices=["generate", "benchmark", "train", "evaluate_dev", "evaluate_test"],
        required=True,
    )
    args = parser.parse_args()

    if args.stage == "generate":
        stage_generate()
    elif args.stage == "benchmark":
        stage_benchmark()
    else:
        print(f"Stage {args.stage} is registered but halted pending user authorization.")


if __name__ == "__main__":
    main()
