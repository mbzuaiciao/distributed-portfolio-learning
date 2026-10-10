"""Exploratory diagnostic follow-up runner for Experiment 1.

Investigates four specific questions on fresh diagnostic splits:
Q1: Specialist B underperformance & calibration under partial observability.
Q2: Redundant vs. complementary ensemble dynamics and Brier decomposition.
Q3: CoLoRA vs. capacity-matched pooled gradient flow and rank sensitivity.
Q4: Exploitation of complementarity: linear averaging vs. log-odds vs. Bayes oracle.

Execution: CPU-only, torch threads=2, lightweight bounded execution (< 30s).
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from hypergraph_portfolio.collaborative_lora import combine_predictions, fit_simplex_weights
from hypergraph_portfolio.diagnostics_utils import (
    apply_platt_scaling,
    brier_score_decomposition,
    compute_task_gradient_cosine,
    fit_log_odds_combiner,
    fit_platt_scaling,
    predict_log_odds_combiner,
)
from hypergraph_portfolio.synthetic_models import (
    CoLoRAMLP,
    SmallMLP,
)
from hypergraph_portfolio.synthetic_multitask import (
    DGPConfig,
    generate_dgp,
    sample_dataset,
)

# Host safeguard: cap PyTorch threads strictly to 2
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIR = ROOT / "artifacts/bayesian_collab_positive_control"
DIAG_OUT = FROZEN_DIR / "diagnostics_followup_results.json"


def compute_metrics_dict(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1.0 - 1e-7)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    ece = 0.0
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        mask = (p >= lo) & (p < hi)
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    from sklearn.metrics import roc_auc_score

    try:
        auroc = float(roc_auc_score(y, p))
    except Exception:
        auroc = 0.5
    return {
        "brier": brier,
        "log_loss": log_loss,
        "auroc": auroc,
        "ece": ece,
    }


def predict_model(model: nn.Module, x: np.ndarray, task_id: int = 0) -> np.ndarray:
    model.eval()
    xt = torch.from_numpy(x).float()
    with torch.no_grad():
        if isinstance(model, CoLoRAMLP):
            preds = model(xt, task_id=task_id).cpu().numpy()
        else:
            preds = model(xt).cpu().numpy()
    return preds


def run_diagnostics() -> Dict[str, Any]:
    t0 = time.perf_counter()
    results: Dict[str, Any] = {
        "timestamp": time.time(),
        "q1_specialist_calibration": {},
        "q2_ensemble_dynamics": {},
        "q3_colora_gradient_analysis": {},
        "q4_complementarity_exploitation": {},
    }

    # 1. Generate fresh diagnostic split with independent data_seed
    spec = generate_dgp(DGPConfig(seed=20261009))
    diag_pos = sample_dataset(spec, "positive_control", data_seed=20261009 + 999)
    diag_neg = sample_dataset(spec, "negative_control", data_seed=20261009 + 999)

    dev_x_A = diag_pos.development["x_A"]
    dev_x_B = diag_pos.development["x_B"]
    dev_y = diag_pos.development["y"]

    eval_x_full = diag_pos.test["x_full"]  # fresh evaluation split (N=512)
    eval_x_A = diag_pos.test["x_A"]
    eval_x_B = diag_pos.test["x_B"]
    eval_y = diag_pos.test["y"]

    # -------------------------------------------------------------
    # Q1: Specialist B Underperformance & Calibration Analysis
    # -------------------------------------------------------------
    q1_results = {}
    for seed in [42, 123, 777]:
        m2_path = FROZEN_DIR / f"ckpt_positive_control_seed_{seed}_m2_specA.pt"
        m3_path = FROZEN_DIR / f"ckpt_positive_control_seed_{seed}_m3_specB.pt"

        m2 = SmallMLP(in_features=40)
        m2.load_state_dict(torch.load(m2_path, weights_only=True))

        m3 = SmallMLP(in_features=40)
        m3.load_state_dict(torch.load(m3_path, weights_only=True))

        # Raw predictions on dev and eval
        pA_dev = predict_model(m2, dev_x_A)
        pB_dev = predict_model(m3, dev_x_B)

        pA_eval = predict_model(m2, eval_x_A)
        pB_eval = predict_model(m3, eval_x_B)

        # Decompositions
        decomp_A = brier_score_decomposition(eval_y, pA_eval)
        decomp_B = brier_score_decomposition(eval_y, pB_eval)

        # Platt scaling on dev, applied to eval
        clf_A = fit_platt_scaling(pA_dev, dev_y)
        clf_B = fit_platt_scaling(pB_dev, dev_y)

        pA_eval_cal = apply_platt_scaling(clf_A, pA_eval)
        pB_eval_cal = apply_platt_scaling(clf_B, pB_eval)

        metrics_A_raw = compute_metrics_dict(eval_y, pA_eval)
        metrics_A_cal = compute_metrics_dict(eval_y, pA_eval_cal)
        metrics_B_raw = compute_metrics_dict(eval_y, pB_eval)
        metrics_B_cal = compute_metrics_dict(eval_y, pB_eval_cal)

        q1_results[str(seed)] = {
            "spec_A": {
                "raw": metrics_A_raw,
                "calibrated": metrics_A_cal,
                "decomposition": decomp_A,
                "pred_stats": {
                    "mean": float(pA_eval.mean()),
                    "std": float(pA_eval.std()),
                    "min": float(pA_eval.min()),
                    "max": float(pA_eval.max()),
                },
            },
            "spec_B": {
                "raw": metrics_B_raw,
                "calibrated": metrics_B_cal,
                "decomposition": decomp_B,
                "pred_stats": {
                    "mean": float(pB_eval.mean()),
                    "std": float(pB_eval.std()),
                    "min": float(pB_eval.min()),
                    "max": float(pB_eval.max()),
                },
            },
        }
    results["q1_specialist_calibration"] = q1_results

    # -------------------------------------------------------------
    # Q2: Redundant vs. Complementary Ensemble Dynamics
    # -------------------------------------------------------------
    q2_results = {}
    for seed in [42, 123, 777]:
        # Positive control models
        m2_p = SmallMLP(in_features=40)
        m2_p.load_state_dict(
            torch.load(
                FROZEN_DIR / f"ckpt_positive_control_seed_{seed}_m2_specA.pt",
                weights_only=True,
            )
        )
        m3_p = SmallMLP(in_features=40)
        m3_p.load_state_dict(
            torch.load(
                FROZEN_DIR / f"ckpt_positive_control_seed_{seed}_m3_specB.pt",
                weights_only=True,
            )
        )

        pA_pos = predict_model(m2_p, eval_x_A)
        pB_pos = predict_model(m3_p, eval_x_B)
        p_ens_pos = 0.5 * (pA_pos + pB_pos)

        # Negative control models
        m2_n = SmallMLP(in_features=40)
        m2_n.load_state_dict(
            torch.load(
                FROZEN_DIR / f"ckpt_negative_control_seed_{seed}_m2_specA.pt",
                weights_only=True,
            )
        )
        m3_n = SmallMLP(in_features=40)
        m3_n.load_state_dict(
            torch.load(
                FROZEN_DIR / f"ckpt_negative_control_seed_{seed}_m3_specB.pt",
                weights_only=True,
            )
        )

        pA_neg = predict_model(m2_n, diag_neg.test["x_A"])
        pB_neg = predict_model(m3_n, diag_neg.test["x_B"])
        p_ens_neg = 0.5 * (pA_neg + pB_neg)

        q2_results[str(seed)] = {
            "pos_ensemble": {
                "metrics": compute_metrics_dict(eval_y, p_ens_pos),
                "decomposition": brier_score_decomposition(eval_y, p_ens_pos),
                "dynamic_range": float(p_ens_pos.max() - p_ens_pos.min()),
                "pred_std": float(p_ens_pos.std()),
            },
            "neg_ensemble": {
                "metrics": compute_metrics_dict(diag_neg.test["y"], p_ens_neg),
                "decomposition": brier_score_decomposition(
                    diag_neg.test["y"], p_ens_neg
                ),
                "dynamic_range": float(p_ens_neg.max() - p_ens_neg.min()),
                "pred_std": float(p_ens_neg.std()),
            },
        }
    results["q2_ensemble_dynamics"] = q2_results

    # -------------------------------------------------------------
    # Q3: CoLoRA vs. Capacity-Matched Pooled Gradient Flow & Bottleneck
    # -------------------------------------------------------------
    # Train bounded mini-experiment (25 epochs) tracking gradient norms and task cosines
    torch.manual_seed(42)
    # Datasets for 3 tasks
    t1_x = torch.from_numpy(diag_pos.target_train["x"]).float()
    t1_y = torch.from_numpy(diag_pos.target_train["y"]).float()

    t2_x = torch.from_numpy(diag_pos.specialist_A_train["x"]).float()
    t2_y = torch.from_numpy(diag_pos.specialist_A_train["y"]).float()

    t3_x = torch.from_numpy(diag_pos.specialist_B_train["x"]).float()
    t3_y = torch.from_numpy(diag_pos.specialist_B_train["y"]).float()

    criterion = nn.BCELoss()

    # Compare gradient alignment in CoLoRA base weights
    colora_m = CoLoRAMLP(in_features=40, rank=4, alpha=8.0, tasks=3)
    opt_col = torch.optim.AdamW(colora_m.parameters(), lr=0.005)

    grad_cosines = []
    base_norms = []
    lora_norms = []

    for ep in range(25):
        colora_m.train()
        # Compute gradient on Task 1 (Spec A: block B masked)
        opt_col.zero_grad()
        loss2 = criterion(colora_m(t2_x, task_id=1), t2_y)
        loss2.backward()
        g_taskA = (
            colora_m.colora_fc1.base_weight.grad.detach().cpu().numpy().copy()
        )

        # Compute gradient on Task 2 (Spec B: block A masked)
        opt_col.zero_grad()
        loss3 = criterion(colora_m(t3_x, task_id=2), t3_y)
        loss3.backward()
        g_taskB = (
            colora_m.colora_fc1.base_weight.grad.detach().cpu().numpy().copy()
        )

        cos_sim = compute_task_gradient_cosine(g_taskA, g_taskB)
        grad_cosines.append(cos_sim)

        # Standard step
        opt_col.zero_grad()
        l_all = (
            criterion(colora_m(t1_x, task_id=0), t1_y)
            + criterion(colora_m(t2_x, task_id=1), t2_y)
            + criterion(colora_m(t3_x, task_id=2), t3_y)
        )
        l_all.backward()
        opt_col.step()

        w_base = colora_m.colora_fc1.base_weight.detach()
        w_delta = (8.0 / 4.0) * (
            colora_m.colora_fc1.lora_b
            @ colora_m.colora_fc1.lambdas[0]
            @ colora_m.colora_fc1.lora_a
        ).detach()
        base_norms.append(float(torch.norm(w_base)))
        lora_norms.append(float(torch.norm(w_delta)))

    results["q3_colora_gradient_analysis"] = {
        "mean_taskA_taskB_gradient_cosine": float(np.mean(grad_cosines)),
        "min_taskA_taskB_gradient_cosine": float(np.min(grad_cosines)),
        "final_base_weight_norm": base_norms[-1],
        "final_lora_delta_norm": lora_norms[-1],
        "lora_to_base_norm_ratio": float(lora_norms[-1] / base_norms[-1]),
    }

    # -------------------------------------------------------------
    # Q4: Complementarity Exploitation: Linear vs. Log-Odds vs. Bayes
    # -------------------------------------------------------------
    # Compare combination rules on fresh diagnostic split
    q4_methods = {}
    for seed in [42, 123, 777]:
        m0 = SmallMLP(in_features=40)
        m0.load_state_dict(
            torch.load(
                FROZEN_DIR / f"ckpt_positive_control_seed_{seed}_m0_target.pt",
                weights_only=True,
            )
        )

        m2 = SmallMLP(in_features=40)
        m2.load_state_dict(
            torch.load(
                FROZEN_DIR / f"ckpt_positive_control_seed_{seed}_m2_specA.pt",
                weights_only=True,
            )
        )

        m3 = SmallMLP(in_features=40)
        m3.load_state_dict(
            torch.load(
                FROZEN_DIR / f"ckpt_positive_control_seed_{seed}_m3_specB.pt",
                weights_only=True,
            )
        )

        p0 = predict_model(m0, eval_x_full)
        pA_eval = predict_model(m2, eval_x_A)
        pB_eval = predict_model(m3, eval_x_B)

        pA_dev = predict_model(m2, dev_x_A)
        pB_dev = predict_model(m3, dev_x_B)

        # 1. Equal linear mixture
        p_equal = 0.5 * (pA_eval + pB_eval)

        # 2. Simplex learned linear mixture (weights fit on fresh dev)
        p_dev_mat = np.column_stack([pA_dev, pB_dev])
        w = fit_simplex_weights(p_dev_mat, dev_y)
        p_learned = combine_predictions(
            np.column_stack([pA_eval, pB_eval]), w
        )

        # 3. Log-odds combination rule (fit on fresh dev)
        clf_lo = fit_log_odds_combiner(pA_dev, pB_dev, dev_y)
        p_log_odds = predict_log_odds_combiner(clf_lo, pA_eval, pB_eval)

        # 4. Analytical joint Bayes oracle
        # P(y=1|x) = sigmoid(beta_s^T xs + beta_A^T xa + beta_B^T xb + b)
        from hypergraph_portfolio.synthetic_multitask import _sigmoid

        xs = eval_x_full[:, : spec.config.d_shared]
        xa = eval_x_full[
            :,
            spec.config.d_shared : spec.config.d_shared + spec.config.d_A,
        ]
        xb = eval_x_full[
            :,
            spec.config.d_shared
            + spec.config.d_A : spec.config.d_shared
            + spec.config.d_A
            + spec.config.d_B,
        ]
        z_bayes = (
            xs @ spec.beta_s
            + xa @ spec.beta_A
            + xb @ spec.beta_B
            + spec.intercept
        )
        p_bayes_full = _sigmoid(z_bayes)

        q4_methods[str(seed)] = {
            "m0_target": compute_metrics_dict(eval_y, p0),
            "m2_specA": compute_metrics_dict(eval_y, pA_eval),
            "m3_specB": compute_metrics_dict(eval_y, pB_eval),
            "equal_linear": compute_metrics_dict(eval_y, p_equal),
            "learned_simplex": compute_metrics_dict(eval_y, p_learned),
            "log_odds_combiner": compute_metrics_dict(eval_y, p_log_odds),
            "bayes_full_oracle": compute_metrics_dict(eval_y, p_bayes_full),
            "combiner_weights": {
                "simplex": w.tolist(),
                "log_odds_coefs": clf_lo.coef_[0].tolist(),
                "log_odds_intercept": float(clf_lo.intercept_[0]),
            },
        }
    results["q4_complementarity_exploitation"] = q4_methods

    elapsed = time.perf_counter() - t0
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024 * 1024
    )
    results["execution"] = {
        "elapsed_seconds": elapsed,
        "peak_rss_mb": max_rss,
        "threads": torch.get_num_threads(),
    }

    # Save outputs
    DIAG_OUT.write_text(json.dumps(results, indent=2))
    print(f"Diagnostics completed in {elapsed:.2f} s. Peak RSS: {max_rss:.2f} MB")
    return results


if __name__ == "__main__":
    run_diagnostics()
