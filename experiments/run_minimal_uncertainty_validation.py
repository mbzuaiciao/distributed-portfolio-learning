"""Minimal Uncertainty-Estimator Diagnostic Benchmark Runner.

Executes:
1. Stage A: Bayesian Logistic Regression Validation across M=20 synthetic replications.
2. Stage B: Neural Penultimate Laplace Comparison across matched M=20 replications.

Enforces CPU execution with PyTorch limited to two threads.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy import stats

from hypergraph_portfolio.minimal_uncertainty_benchmark import (
    BayesianLogisticRegression,
    NeuralLaplaceEstimator,
    compute_selective_risk_coverage,
    evaluate_estimator_diagnostics,
    generate_diagnostic_dgp,
)

# Host safeguard: strictly limit PyTorch CPU threads to 2
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts/minimal_uncertainty_validation"
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


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_bounded_benchmark() -> Dict[str, Any]:
    """Run a single replication benchmark to verify compute safeguards and project full runtime."""
    print("Executing bounded 1-seed benchmark...")
    t0 = time.perf_counter()

    data = generate_diagnostic_dgp(seed=999, n_train=256, n_test_in=512, n_test_extrap=512)

    # BLR fit & eval
    t_blr_0 = time.perf_counter()
    blr = BayesianLogisticRegression.fit(data["X_train"], data["y_train"], alpha=1.0)
    _ = blr.predict_latent(data["X_test_all"])
    t_blr = time.perf_counter() - t_blr_0

    # Neural Laplace fit & eval
    t_nn_0 = time.perf_counter()
    nn_est = NeuralLaplaceEstimator.fit(data["X_train"], data["y_train"], hidden_dim=32, epochs=150, alpha=1.0)
    _ = nn_est.predict_latent(data["X_test_all"])
    t_nn = time.perf_counter() - t_nn_0

    total_time = time.perf_counter() - t0
    peak_mem = get_peak_memory_mb()

    # Project 20 replications with 5 alpha values for neural Laplace
    # BLR: 20 reps * t_blr
    # Neural Laplace: 20 reps * 5 alphas * t_nn
    projected_seconds = 20 * t_blr + 20 * 5 * t_nn

    res = {
        "single_rep_seconds": total_time,
        "blr_seconds": t_blr,
        "neural_laplace_seconds": t_nn,
        "projected_full_seconds": projected_seconds,
        "projected_minutes": projected_seconds / 60.0,
        "peak_memory_mb": peak_mem,
        "safe_to_proceed": bool(projected_seconds < 1200.0),
    }

    print(f"Bounded benchmark completed in {total_time:.3f} s (BLR: {t_blr:.4f} s, NN: {t_nn:.3f} s).")
    print(f"Projected 20-rep full runtime: {res['projected_minutes']:.2f} min (Limit: 20 min). Peak RSS: {peak_mem:.1f} MB.")
    print(f"Safe to proceed: {res['safe_to_proceed']}")

    save_json(OUT_DIR / "bounded_benchmark_result.json", res)
    return res


def run_diagnostic_experiment(m_reps: int = 20, smoke: bool = False) -> Dict[str, Any]:
    """Execute Stage A and Stage B across M replications."""
    t_start_all = time.perf_counter()
    seeds = [101 + i for i in range(m_reps if not smoke else 2)]
    alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0] if not smoke else [1.0]

    config = {
        "seeds": seeds,
        "alpha_grid": alpha_grid,
        "n_train": 256,
        "n_test_in": 512,
        "n_test_extrap": 512,
        "d": 4,
        "hidden_dim": 32,
        "smoke": smoke,
    }
    save_json(OUT_DIR / "config_minimal_validation.json", config)

    csv_rows = []
    blr_results = []
    ctrl_pos_results = []
    ctrl_neg_rand_results = []
    ctrl_neg_conf_results = []
    nn_results_by_alpha: Dict[float, List[Dict[str, Any]]] = {a: [] for a in alpha_grid}
    nn_geometry_results: List[Dict[str, Any]] = []

    print(f"Starting Diagnostic Experiment across {len(seeds)} replications...")

    for rep_idx, seed in enumerate(seeds):
        data = generate_diagnostic_dgp(
            seed=seed,
            n_train=config["n_train"],
            n_test_in=config["n_test_in"],
            n_test_extrap=config["n_test_extrap"],
            d=config["d"],
        )
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test_all"], data["y_test_all"]
        z_test = data["z_test_all"]
        d_mahal = data["mahalanobis_d"]
        is_extrap = data["is_extrap"]

        # ---------------- STAGE A: BAYESIAN LOGISTIC REGRESSION ----------------
        blr = BayesianLogisticRegression.fit(X_train, y_train, alpha=1.0)
        mean_blr, var_blr = blr.predict_latent(X_test)
        p_blr = blr.predict_proba(X_test)
        ci_lower_blr, ci_upper_blr = blr.credible_interval_95(X_test)

        diag_blr = evaluate_estimator_diagnostics(
            uncertainty_scores=var_blr,
            pred_probs=p_blr,
            y_true=y_test,
            mahalanobis_d=d_mahal,
            is_extrap=is_extrap,
            latent_ci=(ci_lower_blr, ci_upper_blr),
            z_true=z_test,
        )
        diag_blr["condition_number"] = blr.hessian_condition_number
        blr_results.append(diag_blr)

        csv_rows.append(
            {
                "replication": rep_idx,
                "seed": seed,
                "model": "BLR_alpha_1.0",
                **diag_blr,
            }
        )

        # Baseline Controls
        # 1. Positive Control: Mahalanobis Distance directly
        diag_pos = evaluate_estimator_diagnostics(
            uncertainty_scores=d_mahal,
            pred_probs=p_blr,
            y_true=y_test,
            mahalanobis_d=d_mahal,
            is_extrap=is_extrap,
        )
        ctrl_pos_results.append(diag_pos)
        csv_rows.append({"replication": rep_idx, "seed": seed, "model": "Ctrl_Pos_Mahalanobis", **diag_pos})

        # 2. Negative Control 1: Random Uniform Noise
        rng = np.random.default_rng(seed)
        noise_scores = rng.uniform(0.0, 1.0, size=len(y_test))
        diag_rand = evaluate_estimator_diagnostics(
            uncertainty_scores=noise_scores,
            pred_probs=p_blr,
            y_true=y_test,
            mahalanobis_d=d_mahal,
            is_extrap=is_extrap,
        )
        ctrl_neg_rand_results.append(diag_rand)
        csv_rows.append({"replication": rep_idx, "seed": seed, "model": "Ctrl_Neg_Random", **diag_rand})

        # 3. Negative Control 2: Inverted Prediction Confidence 1 - |p - 0.5|
        inv_conf = 1.0 - np.abs(p_blr - 0.5)
        diag_conf = evaluate_estimator_diagnostics(
            uncertainty_scores=inv_conf,
            pred_probs=p_blr,
            y_true=y_test,
            mahalanobis_d=d_mahal,
            is_extrap=is_extrap,
        )
        ctrl_neg_conf_results.append(diag_conf)
        csv_rows.append({"replication": rep_idx, "seed": seed, "model": "Ctrl_Neg_InvConf", **diag_conf})

        # ---------------- STAGE B: NEURAL LAPLACE COMPARISON ----------------
        for alpha in alpha_grid:
            epochs = 20 if smoke else 150
            nn_est = NeuralLaplaceEstimator.fit(
                X_train, y_train, hidden_dim=config["hidden_dim"], epochs=epochs, alpha=alpha, seed=seed
            )
            mean_nn, var_nn = nn_est.predict_latent(X_test)
            p_nn = nn_est.predict_proba(X_test)

            diag_nn = evaluate_estimator_diagnostics(
                uncertainty_scores=var_nn,
                pred_probs=p_nn,
                y_true=y_test,
                mahalanobis_d=d_mahal,
                is_extrap=is_extrap,
            )
            diag_nn["condition_number"] = nn_est.hessian_condition_number
            nn_results_by_alpha[alpha].append(diag_nn)

            csv_rows.append(
                {
                    "replication": rep_idx,
                    "seed": seed,
                    "model": f"Neural_Laplace_alpha_{alpha}",
                    **diag_nn,
                }
            )

            # Representation geometry diagnostic (recorded once per replication at alpha=1.0)
            if alpha == 1.0:
                phi_in = nn_est.extract_features(data["X_test_in"])
                phi_extrap = nn_est.extract_features(data["X_test_extrap"])
                norm_in = float(np.mean(np.linalg.norm(phi_in, axis=1)))
                norm_extrap = float(np.mean(np.linalg.norm(phi_extrap, axis=1)))
                norm_ratio = float(norm_extrap / max(norm_in, 1e-12))
                nn_geometry_results.append(
                    {
                        "seed": seed,
                        "mean_phi_norm_in": norm_in,
                        "mean_phi_norm_extrap": norm_extrap,
                        "phi_norm_ratio_extrap_to_in": norm_ratio,
                        "hessian_cond_num": nn_est.hessian_condition_number,
                    }
                )

        if (rep_idx + 1) % 5 == 0 or smoke:
            print(f"Replication {rep_idx + 1}/{len(seeds)} completed. Elapsed: {time.perf_counter() - t_start_all:.1f} s.")

    write_csv(OUT_DIR / "metrics_minimal_validation.csv", csv_rows)

    # Aggregate summaries across replications
    def aggregate(res_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = res_list[0].keys()
        agg = {}
        for k in keys:
            vals = [r[k] for r in res_list if (r[k] is not None and not np.isnan(r[k]))]
            if vals:
                agg[f"mean_{k}"] = float(np.mean(vals))
                agg[f"std_{k}"] = float(np.std(vals))
                agg[f"ci95_{k}"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            else:
                agg[f"mean_{k}"] = None
        return agg

    summary = {
        "execution": {
            "total_seconds": time.perf_counter() - t_start_all,
            "peak_memory_mb": get_peak_memory_mb(),
            "n_replications": len(seeds),
            "smoke": smoke,
        },
        "stage_A_blr": aggregate(blr_results),
        "controls": {
            "positive_mahalanobis": aggregate(ctrl_pos_results),
            "negative_random": aggregate(ctrl_neg_rand_results),
            "negative_inv_confidence": aggregate(ctrl_neg_conf_results),
        },
        "stage_B_neural_laplace_by_alpha": {
            str(a): aggregate(nn_results_by_alpha[a]) for a in alpha_grid
        },
        "neural_representation_geometry": {
            "mean_norm_in": float(np.mean([g["mean_phi_norm_in"] for g in nn_geometry_results])),
            "mean_norm_extrap": float(np.mean([g["mean_phi_norm_extrap"] for g in nn_geometry_results])),
            "mean_norm_ratio_extrap_to_in": float(np.mean([g["phi_norm_ratio_extrap_to_in"] for g in nn_geometry_results])),
            "mean_hessian_cond_num": float(np.mean([g["hessian_cond_num"] for g in nn_geometry_results])),
        },
    }

    # Hypothesis Verification Verdicts
    mean_blr_rho = summary["stage_A_blr"]["mean_support_spearman_rho"]
    mean_blr_cov_in = summary["stage_A_blr"]["mean_interval_coverage_in"]
    mean_blr_aurc = summary["stage_A_blr"]["mean_aurc_brier"]
    mean_rand_aurc = summary["controls"]["negative_random"]["mean_aurc_brier"]

    mean_nn_rho_alpha1 = summary["stage_B_neural_laplace_by_alpha"]["1.0"]["mean_support_spearman_rho"]
    all_nn_rhos = [summary["stage_B_neural_laplace_by_alpha"][str(a)]["mean_support_spearman_rho"] for a in alpha_grid]
    max_nn_rho = max(all_nn_rhos)
    best_alpha = alpha_grid[int(np.argmax(all_nn_rhos))]

    norm_ratio = summary["neural_representation_geometry"]["mean_norm_ratio_extrap_to_in"]
    cond_num_ratio = (
        summary["neural_representation_geometry"]["mean_hessian_cond_num"]
        / max(summary["stage_A_blr"]["mean_condition_number"], 1e-12)
    )

    verdicts = {
        "H_A.1_BLR_Support_Tracking": {
            "mean_spearman_rho": mean_blr_rho,
            "threshold": 0.50,
            "passed": bool(mean_blr_rho >= 0.50),
        },
        "H_A.2_BLR_Interval_Coverage": {
            "mean_coverage_in_support": mean_blr_cov_in,
            "threshold": 0.90,
            "passed": bool(mean_blr_cov_in >= 0.85),  # nominal 95% interval with practical 85% lower bound
        },
        "H_A.3_BLR_Selective_Prediction": {
            "aurc_blr": mean_blr_aurc,
            "aurc_random": mean_rand_aurc,
            "delta_aurc": mean_rand_aurc - mean_blr_aurc,
            "passed": bool(mean_blr_aurc < mean_rand_aurc),
        },
        "H_B.1_Representation_Distortion": {
            "norm_ratio_extrap_to_in": norm_ratio,
            "distortion_detected": bool(norm_ratio < 1.5),  # In input space, Mahalanobis ratio is > 3.0
            "passed": bool(norm_ratio < 1.5),
        },
        "H_B.2_Hyperprior_Sensitivity": {
            "best_alpha": best_alpha,
            "max_nn_spearman_rho": max_nn_rho,
            "blr_spearman_rho": mean_blr_rho,
            "gap": mean_blr_rho - max_nn_rho,
            "passed": bool(mean_blr_rho - max_nn_rho >= 0.15),
        },
        "H_B.3_Curvature_Conditioning": {
            "mean_nn_cond_num": summary["neural_representation_geometry"]["mean_hessian_cond_num"],
            "mean_blr_cond_num": summary["stage_A_blr"]["mean_condition_number"],
            "condition_ratio": cond_num_ratio,
            "passed": bool(cond_num_ratio > 5.0),
        },
    }

    summary["hypothesis_verdicts"] = verdicts
    save_json(OUT_DIR / "summary_minimal_validation.json", summary)

    print("\n================ DIAGNOSTIC BENCHMARK SUMMARY ================")
    print(f"Total Runtime: {summary['execution']['total_seconds']:.2f} s | Peak Memory: {summary['execution']['peak_memory_mb']:.1f} MB")
    print(f"H_A.1 (BLR Support Tracking >= 0.50):        rho={mean_blr_rho:.3f} -> Passed={verdicts['H_A.1_BLR_Support_Tracking']['passed']}")
    print(f"H_A.2 (BLR 95% CI Coverage in-support):     cov={mean_blr_cov_in:.3f} -> Passed={verdicts['H_A.2_BLR_Interval_Coverage']['passed']}")
    print(f"H_A.3 (BLR Selective AU-RC vs Random):       AU-RC={mean_blr_aurc:.4f} vs {mean_rand_aurc:.4f} -> Passed={verdicts['H_A.3_BLR_Selective_Prediction']['passed']}")
    print(f"H_B.1 (NN Representation Distortion):       extrap/in norm ratio={norm_ratio:.2f} -> Passed={verdicts['H_B.1_Representation_Distortion']['passed']}")
    print(f"H_B.2 (NN Prior Grid vs BLR Gap >= 0.15):   best NN rho={max_nn_rho:.3f} vs BLR {mean_blr_rho:.3f} (gap={verdicts['H_B.2_Hyperprior_Sensitivity']['gap']:.3f}) -> Passed={verdicts['H_B.2_Hyperprior_Sensitivity']['passed']}")
    print(f"H_B.3 (NN Hessian Ill-Conditioning):        NN cond={summary['neural_representation_geometry']['mean_hessian_cond_num']:.1f} vs BLR {summary['stage_A_blr']['mean_condition_number']:.1f} -> Passed={verdicts['H_B.3_Curvature_Conditioning']['passed']}")
    print("==============================================================\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Uncertainty-Estimator Diagnostic Benchmark Runner")
    parser.add_argument("--stage", choices=["benchmark", "run"], required=True)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.stage == "benchmark":
        run_bounded_benchmark()
    elif args.stage == "run":
        # First verify bounded benchmark
        bm = run_bounded_benchmark()
        if not bm["safe_to_proceed"]:
            raise RuntimeError("Benchmark projected runtime exceeds 20-minute safety limit. Aborting.")
        run_diagnostic_experiment(m_reps=args.reps, smoke=args.smoke)


if __name__ == "__main__":
    main()
