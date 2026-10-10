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
from scipy import stats
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


def get_peak_memory_mb() -> float:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return float(peak_rss / (1024 * 1024))
    return float(peak_rss / 1024)


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Compute Pearson partial correlation between x and y partialling out z."""
    r_xy, _ = stats.pearsonr(x, y)
    r_xz, _ = stats.pearsonr(x, z)
    r_yz, _ = stats.pearsonr(y, z)
    denom = np.sqrt(max(1e-12, (1.0 - r_xz**2) * (1.0 - r_yz**2)))
    return float((r_xy - r_xz * r_yz) / denom)


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


def stage_train(smoke: bool = False) -> None:
    """Stage 2: Train base models, fit Laplace posteriors, fit calibrators, and fit all routers."""
    if not (OUT_DIR / "splits_exp3_regime_A.npz").exists():
        raise RuntimeError("Must run --stage generate first.")

    config = json.loads((OUT_DIR / "config_exp3.json").read_text())
    seeds = config["seeds"]
    epochs = 10 if smoke else 100

    frozen: Dict[str, Any] = {
        "timestamp": time.time(),
        "hardware": platform.platform(),
        "torch_version": torch.__version__,
        "smoke": smoke,
        "runs": {},
    }

    start_all = time.perf_counter()

    for regime in config["regimes"]:
        data = np.load(OUT_DIR / f"splits_exp3_{regime}.npz")
        cal_y = data["cal_y"]
        fus_y = data["fus_y"]
        xs_fus = data["fus_x_full"][:, :4]

        x_pooled = np.vstack([data["target_train_x"], data["spec_A_train_x"], data["spec_B_train_x"]])
        y_pooled = np.concatenate([data["target_train_y"], data["spec_A_train_y"], data["spec_B_train_y"]])

        frozen["runs"][regime] = {}

        for seed in seeds:
            print(f"Training Exp3 regime={regime} seed={seed}...")
            # 1. Base models
            m0 = train_single_model(data["target_train_x"], data["target_train_y"], epochs=epochs, seed=seed)
            m1_cap = train_single_model(x_pooled, y_pooled, epochs=epochs, seed=seed + 1, model_cls=CapacityMatchedMLP)
            m2 = train_single_model(data["spec_A_train_x"], data["spec_A_train_y"], epochs=epochs, seed=seed + 2)
            m3 = train_single_model(data["spec_B_train_x"], data["spec_B_train_y"], epochs=epochs, seed=seed + 3)

            # Checkpoint weights
            ckpt_paths = {}
            for name, m in [("m0_target", m0), ("m1_cap_pooled", m1_cap), ("m2_specA", m2), ("m3_specB", m3)]:
                p = OUT_DIR / f"ckpt_exp3_{regime}_seed_{seed}_{name}.pt"
                torch.save(m.state_dict(), p)
                ckpt_paths[name] = {"path": str(p), "sha256": digest(p)}

            # 2. Fit Laplace readouts on specialist training representations
            readout_A = BayesianPenultimateReadout.fit(m2, data["spec_A_train_x"], prior_precision=1.0)
            readout_B = BayesianPenultimateReadout.fit(m3, data["spec_B_train_x"], prior_precision=1.0)

            # 3. Fit Platt calibration on calibration split
            pA_cal_raw = predict(m2, data["cal_x_A"])
            pB_cal_raw = predict(m3, data["cal_x_B"])
            calibrator_A = PlattCalibrator.fit(pA_cal_raw, cal_y)
            calibrator_B = PlattCalibrator.fit(pB_cal_raw, cal_y)

            # 4. Predictions on fusion dev split
            pA_fus_raw = predict(m2, data["fus_x_A"])
            pB_fus_raw = predict(m3, data["fus_x_B"])
            pA_fus_cal = calibrator_A.predict(pA_fus_raw)
            pB_fus_cal = calibrator_B.predict(pB_fus_raw)

            _, s2_A_fus, _ = readout_A.predict(m2, data["fus_x_A"])
            _, s2_B_fus, _ = readout_B.predict(m3, data["fus_x_B"])

            # 5. Fit routers on fusion split
            # Static log-odds
            m6_logodds = LogOddsFusion.fit(pA_fus_cal, pB_fus_cal, fus_y)

            # Density router
            dens_router = DensityRouter.fit(data["spec_A_train_x"][:, :4], data["spec_B_train_x"][:, :4])

            # Linear MoE
            moe_lin = LinearMoERouter.fit(xs_fus, pA_fus_cal, pB_fus_cal, fus_y, epochs=epochs, seed=seed)
            p_moe_lin = OUT_DIR / f"ckpt_exp3_{regime}_seed_{seed}_moe_lin.pt"
            torch.save(moe_lin.state_dict(), p_moe_lin)
            ckpt_paths["m_moe_lin"] = {"path": str(p_moe_lin), "sha256": digest(p_moe_lin)}

            # MLP MoE
            moe_mlp = MLPMoERouter.fit(xs_fus, pA_fus_cal, pB_fus_cal, fus_y, epochs=epochs, seed=seed)
            p_moe_mlp = OUT_DIR / f"ckpt_exp3_{regime}_seed_{seed}_moe_mlp.pt"
            torch.save(moe_mlp.state_dict(), p_moe_mlp)
            ckpt_paths["m_moe_mlp"] = {"path": str(p_moe_mlp), "sha256": digest(p_moe_mlp)}

            # Inverse variance router
            inv_router = InverseVarianceRouter.fit(pA_fus_cal, pB_fus_cal, s2_A_fus, s2_B_fus, fus_y)

            # Matched Ablation Routers
            lA_fus = np.log(np.clip(pA_fus_cal, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(pA_fus_cal, 1e-6, 1.0 - 1e-6)))
            lB_fus = np.log(np.clip(pB_fus_cal, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(pB_fus_cal, 1e-6, 1.0 - 1e-6)))
            feat_no_uq = np.column_stack([lA_fus, lB_fus, xs_fus])
            feat_uq = np.column_stack([lA_fus, lB_fus, s2_A_fus, s2_B_fus, xs_fus])

            r_no_uq = MatchedAblationRouter.fit(feat_no_uq, fus_y, epochs=epochs, seed=seed)
            p_r_no_uq = OUT_DIR / f"ckpt_exp3_{regime}_seed_{seed}_router_no_uq.pt"
            torch.save(r_no_uq.state_dict(), p_r_no_uq)
            ckpt_paths["m_router_nouq"] = {"path": str(p_r_no_uq), "sha256": digest(p_r_no_uq)}

            r_uq = MatchedAblationRouter.fit(feat_uq, fus_y, epochs=epochs, seed=seed)
            p_r_uq = OUT_DIR / f"ckpt_exp3_{regime}_seed_{seed}_router_uq.pt"
            torch.save(r_uq.state_dict(), p_r_uq)
            ckpt_paths["m_router_uq"] = {"path": str(p_r_uq), "sha256": digest(p_r_uq)}

            frozen["runs"][regime][str(seed)] = {
                "checkpoints": ckpt_paths,
                "readout_A": readout_A.to_dict(),
                "readout_B": readout_B.to_dict(),
                "calibrator_A": calibrator_A.to_dict(),
                "calibrator_B": calibrator_B.to_dict(),
                "logodds": m6_logodds.to_dict(),
                "density_router": dens_router.to_dict(),
                "inverse_variance_router": inv_router.to_dict(),
            }

    frozen["total_training_seconds"] = time.perf_counter() - start_all
    frozen["peak_memory_mb"] = get_peak_memory_mb()
    save_json(OUT_DIR / "frozen_exp3.json", frozen)
    print(f"Stage train completed successfully in {frozen['total_training_seconds']:.2f} s (peak RSS: {frozen['peak_memory_mb']:.1f} MB).")


def stage_evaluate_dev() -> Dict[str, Any]:
    """Stage 3: Evaluate development gates U1/U2 and deterministically select comparator M*."""
    start_eval = time.perf_counter()
    if not (OUT_DIR / "frozen_exp3.json").exists():
        raise RuntimeError("Must run --stage train before --stage evaluate_dev.")

    config = json.loads((OUT_DIR / "config_exp3.json").read_text())
    frozen = json.loads((OUT_DIR / "frozen_exp3.json").read_text())
    dgp_data = json.loads((OUT_DIR / "dgp_spec_exp3.json").read_text())
    seeds = config["seeds"]

    csv_rows = []
    summary_results: Dict[str, Any] = {}
    gate_results: Dict[str, Any] = {}
    selected_comparators: Dict[str, str] = {}

    for regime in config["regimes"]:
        data = np.load(OUT_DIR / f"splits_exp3_{regime}.npz")
        dev_y = data["dev_y"]
        dev_x_full = data["dev_x_full"]
        dev_x_A = data["dev_x_A"]
        dev_x_B = data["dev_x_B"]
        xs_dev = dev_x_full[:, :4]
        n_dev = len(dev_y)

        seed_method_preds: Dict[int, Dict[str, np.ndarray]] = {}
        regime_u1_A, regime_u1_B = [], []
        regime_u2_A, regime_u2_B = [], []

        for seed in seeds:
            seed_info = frozen["runs"][regime][str(seed)]
            ckpts = seed_info["checkpoints"]

            # Load base models
            m0 = SmallMLP(in_features=40)
            m0.load_state_dict(torch.load(ckpts["m0_target"]["path"], weights_only=True))

            m1_cap = CapacityMatchedMLP(in_features=40)
            m1_cap.load_state_dict(torch.load(ckpts["m1_cap_pooled"]["path"], weights_only=True))

            m2 = SmallMLP(in_features=40)
            m2.load_state_dict(torch.load(ckpts["m2_specA"]["path"], weights_only=True))

            m3 = SmallMLP(in_features=40)
            m3.load_state_dict(torch.load(ckpts["m3_specB"]["path"], weights_only=True))

            # Load routers
            moe_lin = LinearMoERouter(in_features=4)
            moe_lin.load_state_dict(torch.load(ckpts["m_moe_lin"]["path"], weights_only=True))

            moe_mlp = MLPMoERouter(in_features=4, hidden_dim=16)
            moe_mlp.load_state_dict(torch.load(ckpts["m_moe_mlp"]["path"], weights_only=True))

            r_no_uq = MatchedAblationRouter(in_features=6, hidden_dim=16)
            r_no_uq.load_state_dict(torch.load(ckpts["m_router_nouq"]["path"], weights_only=True))

            r_uq = MatchedAblationRouter(in_features=8, hidden_dim=16)
            r_uq.load_state_dict(torch.load(ckpts["m_router_uq"]["path"], weights_only=True))

            # Reconstruct fitted objects
            readout_A = BayesianPenultimateReadout.from_dict(seed_info["readout_A"])
            readout_B = BayesianPenultimateReadout.from_dict(seed_info["readout_B"])
            cal_A = PlattCalibrator.from_dict(seed_info["calibrator_A"])
            cal_B = PlattCalibrator.from_dict(seed_info["calibrator_B"])
            logodds = LogOddsFusion.from_dict(seed_info["logodds"])
            dens_router = DensityRouter.from_dict(seed_info["density_router"])
            inv_router = InverseVarianceRouter.from_dict(seed_info["inverse_variance_router"])

            # Base predictions on development split
            p0 = predict(m0, dev_x_full)
            p1_cap = predict(m1_cap, dev_x_full)
            p2_raw = predict(m2, dev_x_A)
            p3_raw = predict(m3, dev_x_B)

            # Calibrated predictions
            p2_cal = cal_A.predict(p2_raw)
            p3_cal = cal_B.predict(p3_raw)

            # Bayesian uncertainty estimates on dev split
            _, s2_A_dev, p2_bayes = readout_A.predict(m2, dev_x_A)
            _, s2_B_dev, p3_bayes = readout_B.predict(m3, dev_x_B)

            # Router predictions
            p_m6_cal = logodds.predict(p2_cal, p3_cal)
            p_conf = ConfidenceRouter.predict(p2_cal, p3_cal)
            p_dens = dens_router.predict(xs_dev, p2_cal, p3_cal)
            p_moe_lin = moe_lin.predict(xs_dev, p2_cal, p3_cal)
            p_moe_mlp = moe_mlp.predict(xs_dev, p2_cal, p3_cal)
            p_bayes_inv = inv_router.predict(p2_cal, p3_cal, s2_A_dev, s2_B_dev)

            # Matched routers
            lA_dev = np.log(np.clip(p2_cal, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(p2_cal, 1e-6, 1.0 - 1e-6)))
            lB_dev = np.log(np.clip(p3_cal, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(p3_cal, 1e-6, 1.0 - 1e-6)))
            feat_no_uq = np.column_stack([lA_dev, lB_dev, xs_dev])
            feat_uq = np.column_stack([lA_dev, lB_dev, s2_A_dev, s2_B_dev, xs_dev])

            p_router_nouq = r_no_uq.predict(feat_no_uq)
            p_router_uq = r_uq.predict(feat_uq)

            # Prevalence null baseline
            p8_prev_null = np.full(n_dev, float(data["target_train_y"].mean()))

            # Ground truth joint Bayes oracle on dev features
            d_s = dgp_data["config"]["d_shared"]
            d_A = dgp_data["config"]["d_A"]
            d_B = dgp_data["config"]["d_B"]
            beta_s = np.array(dgp_data["beta_s"])
            beta_A = np.array(dgp_data["beta_A"])
            beta_B = np.array(dgp_data["beta_B"])
            intercept = dgp_data["intercept"]

            xs = dev_x_full[:, :d_s]
            xa = dev_x_full[:, d_s : d_s + d_A]
            xb = dev_x_full[:, d_s + d_A : d_s + d_A + d_B]
            p9_oracle = _sigmoid(xs @ beta_s + xa @ beta_A + xb @ beta_B + intercept)

            # Uncertainty Gates Evaluation on dev split
            # Gate U1: Support Sparsity Tracking (correlation between s2 and Mahalanobis distance d2)
            d2_A = np.sum(((xs_dev - dens_router.mu_A) @ dens_router.cov_inv_A) * (xs_dev - dens_router.mu_A), axis=1)
            d2_B = np.sum(((xs_dev - dens_router.mu_B) @ dens_router.cov_inv_B) * (xs_dev - dens_router.mu_B), axis=1)
            u1_corr_A, _ = stats.spearmanr(s2_A_dev, d2_A)
            u1_corr_B, _ = stats.spearmanr(s2_B_dev, d2_B)
            regime_u1_A.append(float(u1_corr_A))
            regime_u1_B.append(float(u1_corr_B))

            # Gate U2: Residual Error Partial Correlation controlling for distance to boundary
            resid2_A = (p2_cal - dev_y) ** 2
            resid2_B = (p3_cal - dev_y) ** 2
            dist_A = np.abs(p2_cal - 0.5)
            dist_B = np.abs(p3_cal - 0.5)
            u2_part_A = partial_corr(s2_A_dev, resid2_A, dist_A)
            u2_part_B = partial_corr(s2_B_dev, resid2_B, dist_B)
            regime_u2_A.append(float(u2_part_A))
            regime_u2_B.append(float(u2_part_B))

            preds = {
                "m0_target": p0,
                "m1_cap_pooled": p1_cap,
                "m2_specA_cal": p2_cal,
                "m3_specB_cal": p3_cal,
                "m6_logodds_cal": p_m6_cal,
                "m_conf": p_conf,
                "m_dens": p_dens,
                "m_moe_lin": p_moe_lin,
                "m_moe_mlp": p_moe_mlp,
                "m_router_nouq": p_router_nouq,
                "m_bayes_inv": p_bayes_inv,
                "m_bayes_learn": p_router_uq,
                "m8_prev_null": p8_prev_null,
                "m9_bayes_oracle": p9_oracle,
            }
            seed_method_preds[seed] = preds

            for meth_name, prob in preds.items():
                m = compute_metrics(dev_y, prob)
                csv_rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "method": meth_name,
                        **m,
                    }
                )

        # Aggregate metrics across seeds
        methods_list = list(seed_method_preds[seeds[0]].keys())
        agg_metrics = {}
        for m_name in methods_list:
            briers = [compute_metrics(dev_y, seed_method_preds[s][m_name])["brier"] for s in seeds]
            eces = [compute_metrics(dev_y, seed_method_preds[s][m_name])["ece"] for s in seeds]
            aurocs = [compute_metrics(dev_y, seed_method_preds[s][m_name])["auroc"] for s in seeds]
            agg_metrics[m_name] = {
                "mean_brier": float(np.mean(briers)),
                "std_brier": float(np.std(briers)),
                "mean_ece": float(np.mean(eces)),
                "mean_auroc": float(np.mean(aurocs)),
            }
        summary_results[regime] = agg_metrics

        # Determine strongest non-Bayesian comparator for this regime
        selected_comparator = select_strongest_nonbayes_baseline(agg_metrics)
        selected_comparators[regime] = selected_comparator

        # Gate verdicts
        mean_u1_A = float(np.mean(regime_u1_A))
        mean_u1_B = float(np.mean(regime_u1_B))
        mean_u2_A = float(np.mean(regime_u2_A))
        mean_u2_B = float(np.mean(regime_u2_B))

        gate_results[regime] = {
            "U1_support_sparsity_tracking": {
                "specialist_A_spearman": mean_u1_A,
                "specialist_B_spearman": mean_u1_B,
                "threshold": 0.40,
                "pass": bool(mean_u1_A >= 0.40 and mean_u1_B >= 0.40),
            },
            "U2_residual_error_partial_corr": {
                "specialist_A_partial_corr": mean_u2_A,
                "specialist_B_partial_corr": mean_u2_B,
                "threshold": 0.10,
                "pass": bool(mean_u2_A >= 0.10 and mean_u2_B >= 0.10),
            },
            "selected_strongest_nonbayes_comparator": selected_comparator,
            "brier_selected_comparator": agg_metrics[selected_comparator]["mean_brier"],
            "brier_bayes_learn": agg_metrics["m_bayes_learn"]["mean_brier"],
            "brier_bayes_inv": agg_metrics["m_bayes_inv"]["mean_brier"],
            "ablation_gain_brier": agg_metrics["m_router_nouq"]["mean_brier"] - agg_metrics["m_bayes_learn"]["mean_brier"],
        }

    # Save summary and CSV
    write_csv(OUT_DIR / "development_metrics_exp3.csv", csv_rows)
    save_json(OUT_DIR / "development_summary_exp3.json", summary_results)

    eval_report = {
        "status": "DEVELOPMENT_EVALUATION_COMPLETED",
        "evaluation_seconds": time.perf_counter() - start_eval,
        "peak_memory_mb": get_peak_memory_mb(),
        "gate_results": gate_results,
        "selected_comparators": selected_comparators,
        "held_out_test_status": "STRICTLY_LOCKED_AND_UNTOUCHED",
    }
    save_json(OUT_DIR / "development_evaluation_report_exp3.json", eval_report)

    print("\n================ EXPERIMENT 3 DEVELOPMENT EVALUATION ================")
    for regime, g in gate_results.items():
        print(f"\n--- {regime.upper()} ---")
        print(f"Gate U1 (Support Sparsity >= 0.40): Spec A={g['U1_support_sparsity_tracking']['specialist_A_spearman']:.3f}, Spec B={g['U1_support_sparsity_tracking']['specialist_B_spearman']:.3f} -> Pass={g['U1_support_sparsity_tracking']['pass']}")
        print(f"Gate U2 (Residual Partial >= 0.10): Spec A={g['U2_residual_error_partial_corr']['specialist_A_partial_corr']:.3f}, Spec B={g['U2_residual_error_partial_corr']['specialist_B_partial_corr']:.3f} -> Pass={g['U2_residual_error_partial_corr']['pass']}")
        print(f"Selected Non-Bayesian Comparator:   {g['selected_strongest_nonbayes_comparator']} (Brier={g['brier_selected_comparator']:.4f})")
        print(f"Bayes Learned Router:               m_bayes_learn (Brier={g['brier_bayes_learn']:.4f})")
        print(f"Uncertainty Feature Ablation Gain:  {g['ablation_gain_brier']:+.4f} Brier")
    print("\n======================================================================")

    return eval_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3 Harness")
    parser.add_argument(
        "--stage",
        choices=["generate", "benchmark", "train", "evaluate_dev", "evaluate_test"],
        required=True,
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.stage == "generate":
        stage_generate()
    elif args.stage == "benchmark":
        stage_benchmark()
    elif args.stage == "train":
        stage_train(smoke=args.smoke)
    elif args.stage == "evaluate_dev":
        stage_evaluate_dev()
    elif args.stage == "evaluate_test":
        print("Stage evaluate_test is locked pending pre-confirmatory audit and review.")


if __name__ == "__main__":
    main()

