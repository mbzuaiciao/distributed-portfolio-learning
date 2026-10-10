"""Experiment 2: Specialist Calibration & Evidence Fusion Research Harness.

Implements the 5-split prospective protocol defined in
docs/bayesian_collaboration_experiment2_protocol.md:
1. Target, specialist A, specialist B model training.
2. Dedicated calibration split for Platt scaling.
3. Dedicated fusion split for simplex & log-odds evidence combination.
4. Dedicated development split for prospective hypothesis testing.
5. Strictly isolated held-out test split (untouched until final audit).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from hypergraph_portfolio.calibration_fusion import (
    LogOddsFusion,
    PlattCalibrator,
    SimplexFusion,
)
from hypergraph_portfolio.synthetic_models import (
    CapacityMatchedMLP,
    CoLoRAMLP,
    SmallMLP,
)
from hypergraph_portfolio.synthetic_multitask import (
    DGPConfig,
    _sigmoid,
    generate_dgp,
)
from hypergraph_portfolio.synthetic_multitask_exp2 import (
    Exp2SplitConfig,
    sample_exp2_dataset,
)
from hypergraph_portfolio.statistical_testing import (
    apply_holm_bonferroni,
    paired_contrast_bootstrap,
    paired_instance_bootstrap,
)


# Host safeguard: strictly limit PyTorch CPU threads to 2
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts/bayesian_collab_exp2"


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
    """Train a SmallMLP or CapacityMatchedMLP model."""
    torch.manual_seed(seed)
    model = model_cls(in_features=x_train.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    criterion = nn.BCELoss()

    x_t = torch.from_numpy(x_train).float()
    y_t = torch.from_numpy(y_train).float()
    n = len(x_train)

    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch_size):
            b_idx = idx[i : i + batch_size]
            bx = x_t[b_idx]
            by = y_t[b_idx]

            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def train_colora_model(
    datasets: List[Tuple[np.ndarray, np.ndarray, int]],
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.005,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> CoLoRAMLP:
    """Train a multi-task CoLoRAMLP model with task routing."""
    torch.manual_seed(seed)
    in_features = datasets[0][0].shape[1]
    model = CoLoRAMLP(
        in_features=in_features, rank=4, alpha=8.0, tasks=len(datasets)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    criterion = nn.BCELoss()

    rng = np.random.default_rng(seed)
    model.train()

    tensors = [
        (torch.from_numpy(x).float(), torch.from_numpy(y).float(), tid)
        for x, y, tid in datasets
    ]

    for _ in range(epochs):
        batches = []
        for xt, yt, tid in tensors:
            n = len(xt)
            idx = rng.permutation(n)
            for i in range(0, n, batch_size):
                b_idx = idx[i : i + batch_size]
                batches.append((xt[b_idx], yt[b_idx], tid))

        rng.shuffle(batches)

        for bx, by, tid in batches:
            optimizer.zero_grad()
            preds = model(bx, task_id=tid)
            loss = criterion(preds, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def predict(
    model: nn.Module,
    x: np.ndarray,
    task_id: int = 0,
) -> np.ndarray:
    """Generate probabilistic predictions."""
    model.eval()
    xt = torch.from_numpy(x).float()
    with torch.no_grad():
        if isinstance(model, CoLoRAMLP):
            preds = model(xt, task_id=task_id).cpu().numpy()
        else:
            preds = model(xt).cpu().numpy()
    return preds


def stage_generate() -> None:
    """Stage 1: Generate synthetic DGP and Exp2 datasets (held-out test split isolated)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = DGPConfig(seed=20261009)
    spec = generate_dgp(cfg)
    split_cfg = Exp2SplitConfig(data_seed=20261011)

    save_json(OUT_DIR / "dgp_spec_exp2.json", spec.to_dict())

    # Generate positive control
    pos_splits = sample_exp2_dataset(
        spec, regime="positive_control", split_cfg=split_cfg
    )
    np.savez_compressed(
        OUT_DIR / "splits_exp2_positive.npz",
        target_train_x=pos_splits.target_train["x"],
        target_train_y=pos_splits.target_train["y"],
        spec_A_train_x=pos_splits.specialist_A_train["x"],
        spec_A_train_y=pos_splits.specialist_A_train["y"],
        spec_B_train_x=pos_splits.specialist_B_train["x"],
        spec_B_train_y=pos_splits.specialist_B_train["y"],
        cal_x_full=pos_splits.calibration["x_full"],
        cal_x_A=pos_splits.calibration["x_A"],
        cal_x_B=pos_splits.calibration["x_B"],
        cal_y=pos_splits.calibration["y"],
        fus_x_full=pos_splits.fusion_dev["x_full"],
        fus_x_A=pos_splits.fusion_dev["x_A"],
        fus_x_B=pos_splits.fusion_dev["x_B"],
        fus_y=pos_splits.fusion_dev["y"],
        dev_x_full=pos_splits.development["x_full"],
        dev_x_A=pos_splits.development["x_A"],
        dev_x_B=pos_splits.development["x_B"],
        dev_y=pos_splits.development["y"],
        test_x_full=pos_splits.test["x_full"],
        test_x_A=pos_splits.test["x_A"],
        test_x_B=pos_splits.test["x_B"],
        test_y=pos_splits.test["y"],
    )

    # Generate negative control (redundant specialists)
    neg_splits = sample_exp2_dataset(
        spec, regime="negative_control", split_cfg=split_cfg
    )
    np.savez_compressed(
        OUT_DIR / "splits_exp2_negative.npz",
        target_train_x=neg_splits.target_train["x"],
        target_train_y=neg_splits.target_train["y"],
        spec_A_train_x=neg_splits.specialist_A_train["x"],
        spec_A_train_y=neg_splits.specialist_A_train["y"],
        spec_B_train_x=neg_splits.specialist_B_train["x"],
        spec_B_train_y=neg_splits.specialist_B_train["y"],
        cal_x_full=neg_splits.calibration["x_full"],
        cal_x_A=neg_splits.calibration["x_A"],
        cal_x_B=neg_splits.calibration["x_B"],
        cal_y=neg_splits.calibration["y"],
        fus_x_full=neg_splits.fusion_dev["x_full"],
        fus_x_A=neg_splits.fusion_dev["x_A"],
        fus_x_B=neg_splits.fusion_dev["x_B"],
        fus_y=neg_splits.fusion_dev["y"],
        dev_x_full=neg_splits.development["x_full"],
        dev_x_A=neg_splits.development["x_A"],
        dev_x_B=neg_splits.development["x_B"],
        dev_y=neg_splits.development["y"],
        test_x_full=neg_splits.test["x_full"],
        test_x_A=neg_splits.test["x_A"],
        test_x_B=neg_splits.test["x_B"],
        test_y=neg_splits.test["y"],
    )

    config_record = {
        "seeds": [42, 123, 777],
        "dgp": asdict(cfg),
        "split_config": asdict(split_cfg),
        "analytical_oracle_gap": spec.analytical_oracle_gap,
        "material_brier_gain": 0.010,
        "regimes": ["positive_control", "negative_control"],
    }
    save_json(OUT_DIR / "config_exp2.json", config_record)
    print("Stage generate completed successfully.")
    print(f"Analytical oracle gap: {spec.analytical_oracle_gap:.6f}")


def stage_train(smoke: bool = False) -> None:
    """Stage 2: Train models, fit calibration scalers on cal split, and fit fusion on fusion split."""
    if not (OUT_DIR / "splits_exp2_positive.npz").exists():
        raise RuntimeError("Must run --stage generate first.")

    config = json.loads((OUT_DIR / "config_exp2.json").read_text())
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
        data = np.load(OUT_DIR / f"splits_exp2_{regime.split('_')[0]}.npz")
        cal_y = data["cal_y"]
        fus_y = data["fus_y"]

        frozen["runs"][regime] = {}

        for seed in seeds:
            print(f"Training Exp2 regime={regime} seed={seed}...")
            # 1. Train base models
            m0 = train_single_model(
                data["target_train_x"],
                data["target_train_y"],
                epochs=epochs,
                seed=seed,
            )
            m2 = train_single_model(
                data["spec_A_train_x"],
                data["spec_A_train_y"],
                epochs=epochs,
                seed=seed + 1,
            )
            m3 = train_single_model(
                data["spec_B_train_x"],
                data["spec_B_train_y"],
                epochs=epochs,
                seed=seed + 2,
            )

            x_pooled = np.vstack(
                [
                    data["target_train_x"],
                    data["spec_A_train_x"],
                    data["spec_B_train_x"],
                ]
            )
            y_pooled = np.concatenate(
                [
                    data["target_train_y"],
                    data["spec_A_train_y"],
                    data["spec_B_train_y"],
                ]
            )
            m1 = train_single_model(
                x_pooled, y_pooled, epochs=epochs, seed=seed + 3
            )
            m1_cap = train_single_model(
                x_pooled,
                y_pooled,
                epochs=epochs,
                seed=seed + 3,
                model_cls=CapacityMatchedMLP,
            )
            m7 = train_colora_model(
                [
                    (data["target_train_x"], data["target_train_y"], 0),
                    (data["spec_A_train_x"], data["spec_A_train_y"], 1),
                    (data["spec_B_train_x"], data["spec_B_train_y"], 2),
                ],
                epochs=epochs,
                seed=seed + 4,
            )

            # Checkpoint weights
            ckpt_paths = {}
            for name, m in [
                ("m0_target", m0),
                ("m1_pooled", m1),
                ("m1_cap_pooled", m1_cap),
                ("m2_specA", m2),
                ("m3_specB", m3),
                ("m7_colora", m7),
            ]:
                p = OUT_DIR / f"ckpt_exp2_{regime}_seed_{seed}_{name}.pt"
                torch.save(m.state_dict(), p)
                ckpt_paths[name] = {"path": str(p), "sha256": digest(p)}

            # 2. Fit calibration models strictly on dedicated calibration split
            pA_cal_split = predict(m2, data["cal_x_A"])
            pB_cal_split = predict(m3, data["cal_x_B"])

            calibrator_A = PlattCalibrator.fit(pA_cal_split, cal_y)
            calibrator_B = PlattCalibrator.fit(pB_cal_split, cal_y)

            # 3. Fit fusion models strictly on dedicated fusion development split
            pA_fus_raw = predict(m2, data["fus_x_A"])
            pB_fus_raw = predict(m3, data["fus_x_B"])

            pA_fus_cal = calibrator_A.predict(pA_fus_raw)
            pB_fus_cal = calibrator_B.predict(pB_fus_raw)

            # Fit simplex linear mixtures
            simplex_raw = SimplexFusion.fit(pA_fus_raw, pB_fus_raw, fus_y)
            simplex_cal = SimplexFusion.fit(pA_fus_cal, pB_fus_cal, fus_y)

            # Fit log-odds fusion models
            logodds_raw = LogOddsFusion.fit(pA_fus_raw, pB_fus_raw, fus_y)
            logodds_cal = LogOddsFusion.fit(pA_fus_cal, pB_fus_cal, fus_y)

            frozen["runs"][regime][str(seed)] = {
                "checkpoints": ckpt_paths,
                "calibrator_A": calibrator_A.to_dict(),
                "calibrator_B": calibrator_B.to_dict(),
                "simplex_raw": simplex_raw.to_dict(),
                "simplex_cal": simplex_cal.to_dict(),
                "logodds_raw": logodds_raw.to_dict(),
                "logodds_cal": logodds_cal.to_dict(),
            }

    frozen["total_training_seconds"] = time.perf_counter() - start_all
    save_json(OUT_DIR / "frozen_exp2.json", frozen)
    print("Stage train completed successfully.")


def stage_evaluate_dev() -> Dict[str, Any]:
    """Stage 3: Evaluate all 16 methods on unseen development split (test split strictly untouched)."""
    if not (OUT_DIR / "frozen_exp2.json").exists():
        raise RuntimeError("Must run --stage train before --stage evaluate_dev.")

    config = json.loads((OUT_DIR / "config_exp2.json").read_text())
    frozen = json.loads((OUT_DIR / "frozen_exp2.json").read_text())
    dgp_data = json.loads((OUT_DIR / "dgp_spec_exp2.json").read_text())
    seeds = config["seeds"]

    csv_rows = []
    summary_results: Dict[str, Any] = {}

    for regime in config["regimes"]:
        data = np.load(OUT_DIR / f"splits_exp2_{regime.split('_')[0]}.npz")
        dev_y = data["dev_y"]
        dev_x_full = data["dev_x_full"]
        dev_x_A = data["dev_x_A"]
        dev_x_B = data["dev_x_B"]
        n_dev = len(dev_y)

        seed_method_preds: Dict[int, Dict[str, np.ndarray]] = {}

        for seed in seeds:
            seed_info = frozen["runs"][regime][str(seed)]
            ckpts = seed_info["checkpoints"]

            # Load models
            m0 = SmallMLP(in_features=40)
            m0.load_state_dict(
                torch.load(ckpts["m0_target"]["path"], weights_only=True)
            )

            m1 = SmallMLP(in_features=40)
            m1.load_state_dict(
                torch.load(ckpts["m1_pooled"]["path"], weights_only=True)
            )

            m1_cap = CapacityMatchedMLP(in_features=40)
            m1_cap.load_state_dict(
                torch.load(ckpts["m1_cap_pooled"]["path"], weights_only=True)
            )

            m2 = SmallMLP(in_features=40)
            m2.load_state_dict(
                torch.load(ckpts["m2_specA"]["path"], weights_only=True)
            )

            m3 = SmallMLP(in_features=40)
            m3.load_state_dict(
                torch.load(ckpts["m3_specB"]["path"], weights_only=True)
            )

            m7 = CoLoRAMLP(in_features=40)
            m7.load_state_dict(
                torch.load(ckpts["m7_colora"]["path"], weights_only=True)
            )

            # Reconstruct fitted post-hoc transforms
            cal_A = PlattCalibrator.from_dict(seed_info["calibrator_A"])
            cal_B = PlattCalibrator.from_dict(seed_info["calibrator_B"])
            simplex_raw = SimplexFusion.from_dict(seed_info["simplex_raw"])
            simplex_cal = SimplexFusion.from_dict(seed_info["simplex_cal"])
            logodds_raw = LogOddsFusion.from_dict(seed_info["logodds_raw"])
            logodds_cal = LogOddsFusion.from_dict(seed_info["logodds_cal"])

            # Base predictions on development split
            p0 = predict(m0, dev_x_full)
            p1 = predict(m1, dev_x_full)
            p1_cap = predict(m1_cap, dev_x_full)
            p2_raw = predict(m2, dev_x_A)
            p3_raw = predict(m3, dev_x_B)
            p7_colora = predict(m7, dev_x_full, task_id=0)

            # Calibrated predictions
            p2_cal = cal_A.predict(p2_raw)
            p3_cal = cal_B.predict(p3_raw)

            # Combination predictions
            p4_equal_raw = 0.5 * (p2_raw + p3_raw)
            p4_equal_cal = 0.5 * (p2_cal + p3_cal)

            p5_simplex_raw = simplex_raw.predict(p2_raw, p3_raw)
            p5_simplex_cal = simplex_cal.predict(p2_cal, p3_cal)

            p6_logodds_raw = logodds_raw.predict(p2_raw, p3_raw)
            p6_logodds_cal = logodds_cal.predict(p2_cal, p3_cal)

            # Baselines and oracles
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
            p9_oracle = _sigmoid(
                xs @ beta_s + xa @ beta_A + xb @ beta_B + intercept
            )

            preds = {
                "m0_target": p0,
                "m1_pooled": p1,
                "m1_cap_pooled": p1_cap,
                "m2_specA_raw": p2_raw,
                "m2_specA_cal": p2_cal,
                "m3_specB_raw": p3_raw,
                "m3_specB_cal": p3_cal,
                "m4_equal_raw": p4_equal_raw,
                "m4_equal_cal": p4_equal_cal,
                "m5_simplex_raw": p5_simplex_raw,
                "m5_simplex_cal": p5_simplex_cal,
                "m6_logodds_raw": p6_logodds_raw,
                "m6_logodds_cal": p6_logodds_cal,
                "m7_colora": p7_colora,
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

        # Aggregate mean across seeds
        methods_list = list(seed_method_preds[seeds[0]].keys())
        agg_metrics = {}
        for m_name in methods_list:
            briers = [
                compute_metrics(dev_y, seed_method_preds[s][m_name])["brier"]
                for s in seeds
            ]
            eces = [
                compute_metrics(dev_y, seed_method_preds[s][m_name])["ece"]
                for s in seeds
            ]
            aurocs = [
                compute_metrics(dev_y, seed_method_preds[s][m_name])["auroc"]
                for s in seeds
            ]
            agg_metrics[m_name] = {
                "mean_brier": float(np.mean(briers)),
                "std_brier": float(np.std(briers)),
                "mean_ece": float(np.mean(eces)),
                "mean_auroc": float(np.mean(aurocs)),
            }

        summary_results[regime] = agg_metrics

    # Save summary and CSV
    write_csv(OUT_DIR / "development_metrics_exp2.csv", csv_rows)
    save_json(OUT_DIR / "development_summary_exp2.json", summary_results)

    # Hypothesis evaluations on development data
    pos = summary_results["positive_control"]
    neg = summary_results["negative_control"]

    # H2.1: Calibration improvement on specialists
    h2_1_specA = pos["m2_specA_cal"]["mean_brier"] < pos["m2_specA_raw"]["mean_brier"]
    h2_1_specB = pos["m3_specB_cal"]["mean_brier"] < pos["m3_specB_raw"]["mean_brier"]
    h2_1_ece = (
        pos["m2_specA_cal"]["mean_ece"] <= 0.15
        and pos["m3_specB_cal"]["mean_ece"] <= 0.15
    )
    h2_1_pass = h2_1_specA and h2_1_specB and h2_1_ece

    # H2.2: Log-odds fusion vs linear averaging
    h2_2_pass = (
        pos["m6_logodds_cal"]["mean_brier"] < pos["m5_simplex_cal"]["mean_brier"]
        and pos["m6_logodds_cal"]["mean_brier"]
        < pos["m4_equal_cal"]["mean_brier"]
    )

    # H2.3: Calibrated log-odds fusion vs Target-only
    gain_vs_m0 = (
        pos["m0_target"]["mean_brier"] - pos["m6_logodds_cal"]["mean_brier"]
    )
    h2_3_pass = gain_vs_m0 >= 0.010

    # H2.4: Attribution to complementarity (contrast vs redundant control)
    gain_pos = (
        pos["m0_target"]["mean_brier"] - pos["m6_logodds_cal"]["mean_brier"]
    )
    gain_neg = (
        neg["m0_target"]["mean_brier"] - neg["m6_logodds_cal"]["mean_brier"]
    )
    h2_4_contrast = gain_pos - gain_neg
    h2_4_pass = h2_4_contrast > 0.0

    eval_report = {
        "H2.1_specialist_calibration_pass": bool(h2_1_pass),
        "H2.2_log_odds_fusion_pass": bool(h2_2_pass),
        "H2.3_system_vs_target_pass": bool(h2_3_pass),
        "H2.4_complementarity_attribution_pass": bool(h2_4_pass),
        "metrics_summary": {
            "gain_vs_target_only": float(gain_vs_m0),
            "matched_complementarity_contrast": float(h2_4_contrast),
            "brier_target_only": pos["m0_target"]["mean_brier"],
            "brier_capacity_matched_pooled": pos["m1_cap_pooled"]["mean_brier"],
            "brier_calibrated_log_odds": pos["m6_logodds_cal"]["mean_brier"],
            "brier_calibrated_simplex": pos["m5_simplex_cal"]["mean_brier"],
            "brier_calibrated_equal": pos["m4_equal_cal"]["mean_brier"],
            "brier_bayes_oracle": pos["m9_bayes_oracle"]["mean_brier"],
        },
    }
    save_json(OUT_DIR / "development_evaluation_report_exp2.json", eval_report)

    print("\n================ EXPERIMENT 2 DEVELOPMENT EVALUATION ================")
    print(f"H2.1 Specialist Calibration Pass:     {h2_1_pass}")
    print(f"H2.2 Log-Odds Fusion Superiority:     {h2_2_pass}")
    print(f"H2.3 System vs. Target-Only (>=0.010):{h2_3_pass} (Gain: {gain_vs_m0:.4f})")
    print(f"H2.4 Complementarity Contrast (>0):   {h2_4_pass} (Contrast: {h2_4_contrast:.4f})")
    print("======================================================================")

    return eval_report


def stage_evaluate_test() -> Dict[str, Any]:
    """Stage 4: Confirmatory Evaluation on Quarantined Held-Out Test Split (N=512).

    Executes the frozen Confirmatory Statistical Analysis Plan (SAP) using paired
    cluster bootstrap over test instances preserving method and seed pairing.
    """
    if not (OUT_DIR / "frozen_exp2.json").exists():
        raise RuntimeError("Must run --stage train before running confirmatory evaluation.")

    config = json.loads((OUT_DIR / "config_exp2.json").read_text())
    frozen = json.loads((OUT_DIR / "frozen_exp2.json").read_text())
    dgp_data = json.loads((OUT_DIR / "dgp_spec_exp2.json").read_text())
    seeds = config["seeds"]

    csv_rows = []
    summary_results: Dict[str, Any] = {}
    regime_seed_method_preds: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {
        "positive_control": {},
        "negative_control": {},
    }
    regime_test_y: Dict[str, np.ndarray] = {}

    for regime in config["regimes"]:
        data = np.load(OUT_DIR / f"splits_exp2_{regime.split('_')[0]}.npz")
        test_y = data["test_y"]
        test_x_full = data["test_x_full"]
        test_x_A = data["test_x_A"]
        test_x_B = data["test_x_B"]
        regime_test_y[regime] = test_y
        n_test = len(test_y)

        seed_method_preds: Dict[int, Dict[str, np.ndarray]] = {}

        for seed in seeds:
            seed_info = frozen["runs"][regime][str(seed)]
            ckpts = seed_info["checkpoints"]

            # Load models
            m0 = SmallMLP(in_features=40)
            m0.load_state_dict(torch.load(ckpts["m0_target"]["path"], weights_only=True))

            m1 = SmallMLP(in_features=40)
            m1.load_state_dict(torch.load(ckpts["m1_pooled"]["path"], weights_only=True))

            m1_cap = CapacityMatchedMLP(in_features=40)
            m1_cap.load_state_dict(torch.load(ckpts["m1_cap_pooled"]["path"], weights_only=True))

            m2 = SmallMLP(in_features=40)
            m2.load_state_dict(torch.load(ckpts["m2_specA"]["path"], weights_only=True))

            m3 = SmallMLP(in_features=40)
            m3.load_state_dict(torch.load(ckpts["m3_specB"]["path"], weights_only=True))

            m7 = CoLoRAMLP(in_features=40)
            m7.load_state_dict(torch.load(ckpts["m7_colora"]["path"], weights_only=True))

            # Reconstruct fitted post-hoc transforms
            cal_A = PlattCalibrator.from_dict(seed_info["calibrator_A"])
            cal_B = PlattCalibrator.from_dict(seed_info["calibrator_B"])
            simplex_raw = SimplexFusion.from_dict(seed_info["simplex_raw"])
            simplex_cal = SimplexFusion.from_dict(seed_info["simplex_cal"])
            logodds_raw = LogOddsFusion.from_dict(seed_info["logodds_raw"])
            logodds_cal = LogOddsFusion.from_dict(seed_info["logodds_cal"])

            # Base predictions on held-out test split
            p0 = predict(m0, test_x_full)
            p1 = predict(m1, test_x_full)
            p1_cap = predict(m1_cap, test_x_full)
            p2_raw = predict(m2, test_x_A)
            p3_raw = predict(m3, test_x_B)
            p7_colora = predict(m7, test_x_full, task_id=0)

            # Calibrated predictions
            p2_cal = cal_A.predict(p2_raw)
            p3_cal = cal_B.predict(p3_raw)

            # Combination predictions
            p4_equal_raw = 0.5 * (p2_raw + p3_raw)
            p4_equal_cal = 0.5 * (p2_cal + p3_cal)

            p5_simplex_raw = simplex_raw.predict(p2_raw, p3_raw)
            p5_simplex_cal = simplex_cal.predict(p2_cal, p3_cal)

            p6_logodds_raw = logodds_raw.predict(p2_raw, p3_raw)
            p6_logodds_cal = logodds_cal.predict(p2_cal, p3_cal)

            # Prevalence null baseline
            p8_prev_null = np.full(n_test, float(data["target_train_y"].mean()))

            # Ground truth joint Bayes oracle on test features
            d_s = dgp_data["config"]["d_shared"]
            d_A = dgp_data["config"]["d_A"]
            d_B = dgp_data["config"]["d_B"]
            beta_s = np.array(dgp_data["beta_s"])
            beta_A = np.array(dgp_data["beta_A"])
            beta_B = np.array(dgp_data["beta_B"])
            intercept = dgp_data["intercept"]

            xs = test_x_full[:, :d_s]
            xa = test_x_full[:, d_s : d_s + d_A]
            xb = test_x_full[:, d_s + d_A : d_s + d_A + d_B]
            p9_oracle = _sigmoid(xs @ beta_s + xa @ beta_A + xb @ beta_B + intercept)

            preds = {
                "m0_target": p0,
                "m1_pooled": p1,
                "m1_cap_pooled": p1_cap,
                "m2_specA_raw": p2_raw,
                "m2_specA_cal": p2_cal,
                "m3_specB_raw": p3_raw,
                "m3_specB_cal": p3_cal,
                "m4_equal_raw": p4_equal_raw,
                "m4_equal_cal": p4_equal_cal,
                "m5_simplex_raw": p5_simplex_raw,
                "m5_simplex_cal": p5_simplex_cal,
                "m6_logodds_raw": p6_logodds_raw,
                "m6_logodds_cal": p6_logodds_cal,
                "m7_colora": p7_colora,
                "m8_prev_null": p8_prev_null,
                "m9_bayes_oracle": p9_oracle,
            }
            seed_method_preds[seed] = preds

            for meth_name, prob in preds.items():
                m = compute_metrics(test_y, prob)
                csv_rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "method": meth_name,
                        **m,
                    }
                )

        # Re-index for paired bootstrap: method -> {seed -> preds}
        meth_keys = list(seed_method_preds[seeds[0]].keys())
        for m_name in meth_keys:
            regime_seed_method_preds[regime][m_name] = {
                s: seed_method_preds[s][m_name] for s in seeds
            }

        # Aggregate mean across seeds
        agg_metrics = {}
        for m_name in meth_keys:
            briers = [
                compute_metrics(test_y, seed_method_preds[s][m_name])["brier"]
                for s in seeds
            ]
            eces = [
                compute_metrics(test_y, seed_method_preds[s][m_name])["ece"]
                for s in seeds
            ]
            aurocs = [
                compute_metrics(test_y, seed_method_preds[s][m_name])["auroc"]
                for s in seeds
            ]
            agg_metrics[m_name] = {
                "mean_brier": float(np.mean(briers)),
                "std_brier": float(np.std(briers)),
                "mean_ece": float(np.mean(eces)),
                "mean_auroc": float(np.mean(aurocs)),
            }
        summary_results[regime] = agg_metrics

    # Save summary and CSV
    write_csv(OUT_DIR / "confirmatory_metrics_exp2.csv", csv_rows)
    save_json(OUT_DIR / "confirmatory_summary_exp2.json", summary_results)

    # ------------------ CONFIRMATORY STATISTICAL ANALYSIS ------------------
    pos_preds = regime_seed_method_preds["positive_control"]
    neg_preds = regime_seed_method_preds["negative_control"]
    y_pos = regime_test_y["positive_control"]
    y_neg = regime_test_y["negative_control"]

    # 1. Primary Confirmatory Hypothesis: M6_cal vs M0
    primary_res = paired_instance_bootstrap(
        y_true=y_pos,
        preds_A=pos_preds["m6_logodds_cal"],
        preds_B=pos_preds["m0_target"],
        n_bootstraps=2000,
        seed=20261012,
        effect_size_threshold=0.010,
    )

    # 2. Secondary Confirmatory Family:
    # C1: M6_cal vs M4_cal (Equal-weight)
    c1 = paired_instance_bootstrap(
        y_true=y_pos,
        preds_A=pos_preds["m6_logodds_cal"],
        preds_B=pos_preds["m4_equal_cal"],
        n_bootstraps=2000,
        seed=20261012,
        effect_size_threshold=0.0,
    )
    # C2: M6_cal vs M5_cal (Simplex)
    c2 = paired_instance_bootstrap(
        y_true=y_pos,
        preds_A=pos_preds["m6_logodds_cal"],
        preds_B=pos_preds["m5_simplex_cal"],
        n_bootstraps=2000,
        seed=20261012,
        effect_size_threshold=0.0,
    )
    # C3: M6_cal vs M1_cap (Capacity-matched pooled)
    c3 = paired_instance_bootstrap(
        y_true=y_pos,
        preds_A=pos_preds["m6_logodds_cal"],
        preds_B=pos_preds["m1_cap_pooled"],
        n_bootstraps=2000,
        seed=20261012,
        effect_size_threshold=0.0,
    )
    # C4: Matched Complementarity Diagnostic Contrast
    c4 = paired_contrast_bootstrap(
        y_true_pos=y_pos,
        preds_A_pos=pos_preds["m6_logodds_cal"],
        preds_B_pos=pos_preds["m0_target"],
        y_true_neg=y_neg,
        preds_A_neg=neg_preds["m6_logodds_cal"],
        preds_B_neg=neg_preds["m0_target"],
        n_bootstraps=2000,
        seed=20261012,
        effect_size_threshold=0.0,
    )

    secondary_adjusted = apply_holm_bonferroni([c1, c2, c3, c4])

    confirmatory_report = {
        "status": "COMPLETED",
        "primary_hypothesis_H2_3": primary_res.to_dict(),
        "secondary_family_holm": [r.to_dict() for r in secondary_adjusted],
        "overall_confirmatory_success": bool(
            primary_res.confirmatory_pass
            and any(r.confirmatory_pass for r in secondary_adjusted)
        ),
    }

    save_json(OUT_DIR / "confirmatory_evaluation_report_exp2.json", confirmatory_report)

    print("\n================ EXPERIMENT 2 CONFIRMATORY EVALUATION ================")
    print(f"Primary Contrast (M6_cal vs M0):")
    print(f"  Point Gain:            {primary_res.point_estimate:.4f} (Threshold >= {primary_res.effect_size_threshold:.4f})")
    print(f"  95% Bootstrap CI:      [{primary_res.ci_lower:.4f}, {primary_res.ci_upper:.4f}]")
    print(f"  Bootstrap p-value:     {primary_res.p_value_raw:.4e}")
    print(f"  Effect Size Pass:      {primary_res.effect_size_pass}")
    print(f"  Statistical Sig Pass:  {primary_res.statistical_significance_pass}")
    print(f"  CONFIRMATORY PASS:     {primary_res.confirmatory_pass}")
    print("----------------------------------------------------------------------")
    print("Secondary Confirmatory Family (Holm Adjusted):")
    for r in secondary_adjusted:
        print(f"  {r.contrast_name}: Gain={r.point_estimate:.4f}, 95% CI=[{r.ci_lower:.4f}, {r.ci_upper:.4f}], p_holm={r.p_value_holm:.4e}, ConfPass={r.confirmatory_pass}")
    print("======================================================================")

    return confirmatory_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2 Harness")
    parser.add_argument(
        "--stage",
        choices=["generate", "train", "evaluate_dev", "evaluate_test"],
        required=True,
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.stage == "generate":
        stage_generate()
    elif args.stage == "train":
        stage_train(smoke=args.smoke)
    elif args.stage == "evaluate_dev":
        stage_evaluate_dev()
    elif args.stage == "evaluate_test":
        stage_evaluate_test()


if __name__ == "__main__":
    main()

