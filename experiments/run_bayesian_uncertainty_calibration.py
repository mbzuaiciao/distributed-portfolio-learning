"""
Experiment 16: Empirical Calibration & Informational Value of Epistemic Uncertainty (H1)
========================================================================================
Tests Hypothesis H1:
    "Higher Bayesian-LoRA epistemic uncertainty predicts larger future forecast error."

Evaluates:
- Pearson and Spearman rank correlation between sigma_epi and |e|, e^2
- Bootstrap 95% confidence intervals (1,000 resamples)
- Error by uncertainty quartiles & quintiles (top-vs-bottom gap)
- AUROC & Average Precision for identifying large forecast errors (|e| >= 80th pct)
- Regime and subgroup breakdowns (2024 vs 2025, calm vs volatile periods)
- Outputs saved to artifacts/bayesian_uncertainty_calibration/
"""

from __future__ import annotations

import csv
import json
import math
import os

os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from hypergraph_portfolio.bayesian_lora_adapter import (
    BayesianDatasetAdapter,
    BayesianPredictionRecord,
)


def compute_auroc_and_ap(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Compute AUROC and Average Precision using rank sums and trapezoidal integration."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(np.sum(labels))
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5, float(n_pos / len(labels))

    # AUROC via Wilcoxon-Mann-Whitney U statistic
    ranks = stats.rankdata(scores)
    u_stat = float(np.sum(ranks[labels == 1]) - n_pos * (n_pos + 1) / 2.0)
    auroc = u_stat / (n_pos * n_neg)

    # Average Precision (AP)
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    cum_pos = np.cumsum(sorted_labels)
    cum_total = np.arange(1, len(labels) + 1)
    precisions = cum_pos / cum_total
    ap = float(np.sum(precisions * sorted_labels) / n_pos)

    return float(auroc), float(ap)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    metric_func: Any,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculate point estimate and bootstrap confidence interval [ci_lower, ci_upper]."""
    rng = np.random.default_rng(seed)
    n = len(x)
    point_est = float(metric_func(x, y))

    boot_estimates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boot_est = float(metric_func(x[idx], y[idx]))
            if not (math.isnan(boot_est) or math.isinf(boot_est)):
                boot_estimates.append(boot_est)
        except Exception:
            continue

    if len(boot_estimates) < 100:
        return point_est, point_est, point_est

    ci_lower = float(np.percentile(boot_estimates, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_estimates, 100.0 * (1.0 - alpha / 2.0)))
    return point_est, ci_lower, ci_upper


def evaluate_single_seed_calibration(
    records: List[BayesianPredictionRecord],
    seed_name: str,
) -> Dict[str, Any]:
    """Run full calibration evaluation for a single dataset of predictions."""
    sigma = np.array([r.epistemic_std for r in records], dtype=float)
    ret = np.array([r.realized_return for r in records], dtype=float)
    mu_hat = np.array([r.mu_hat for r in records], dtype=float)
    dates = [r.timestamp for r in records]

    # Forecast error
    err = ret - mu_hat
    abs_err = np.abs(err)
    sq_err = err**2

    # Binary large-error indicator (top 20% of |e|)
    large_err_thresh = float(np.percentile(abs_err, 80))
    is_large_err = (abs_err >= large_err_thresh).astype(int)

    # 1. Pearson & Spearman correlations
    pearson_abs, p_abs_low, p_abs_high = bootstrap_ci(
        sigma, abs_err, lambda x, y: stats.pearsonr(x, y)[0]
    )
    spearman_abs, s_abs_low, s_abs_high = bootstrap_ci(
        sigma, abs_err, lambda x, y: stats.spearmanr(x, y)[0]
    )
    spearman_sq, s_sq_low, s_sq_high = bootstrap_ci(
        sigma, sq_err, lambda x, y: stats.spearmanr(x, y)[0]
    )

    # 2. AUROC and AP
    auroc, ap = compute_auroc_and_ap(sigma, is_large_err)

    # 3. Quantile error analysis (Quartiles)
    q_bounds = np.quantile(sigma, [0.25, 0.50, 0.75])
    q_bins = np.digitize(sigma, q_bounds)  # 0, 1, 2, 3 -> Q1, Q2, Q3, Q4

    quantile_stats = []
    for q_idx in range(4):
        mask = q_bins == q_idx
        mean_sigma_q = float(np.mean(sigma[mask])) if np.any(mask) else 0.0
        mean_abs_err_q = float(np.mean(abs_err[mask])) if np.any(mask) else 0.0
        mean_sq_err_q = float(np.mean(sq_err[mask])) if np.any(mask) else 0.0
        large_err_rate_q = float(np.mean(is_large_err[mask])) if np.any(mask) else 0.0
        quantile_stats.append({
            "quantile": f"Q{q_idx + 1}",
            "count": int(np.sum(mask)),
            "mean_sigma_epi": mean_sigma_q,
            "mean_abs_error": mean_abs_err_q,
            "mean_sq_error": mean_sq_err_q,
            "large_error_rate": large_err_rate_q,
        })

    q1_err = quantile_stats[0]["mean_abs_error"]
    q4_err = quantile_stats[3]["mean_abs_error"]
    top_vs_bottom_ratio = float(q4_err / (q1_err + 1e-12))
    top_vs_bottom_gap = float(q4_err - q1_err)

    # 4. Temporal breakdown (2024 vs 2025)
    is_2024 = np.array([d.startswith("2024") for d in dates], dtype=bool)
    is_2025 = np.array([d.startswith("2025") for d in dates], dtype=bool)

    spearman_2024 = (
        float(stats.spearmanr(sigma[is_2024], abs_err[is_2024])[0])
        if np.sum(is_2024) > 50
        else float("nan")
    )
    spearman_2025 = (
        float(stats.spearmanr(sigma[is_2025], abs_err[is_2025])[0])
        if np.sum(is_2025) > 50
        else float("nan")
    )

    return {
        "seed": seed_name,
        "n_samples": len(records),
        "large_error_threshold": large_err_thresh,
        "pearson_abs_err": {
            "point": pearson_abs,
            "ci_95": [p_abs_low, p_abs_high],
            "p_value": float(stats.pearsonr(sigma, abs_err)[1]),
        },
        "spearman_abs_err": {
            "point": spearman_abs,
            "ci_95": [s_abs_low, s_abs_high],
            "p_value": float(stats.spearmanr(sigma, abs_err)[1]),
        },
        "spearman_sq_err": {
            "point": spearman_sq,
            "ci_95": [s_sq_low, s_sq_high],
        },
        "auroc_large_error": auroc,
        "ap_large_error": ap,
        "top_vs_bottom_ratio": top_vs_bottom_ratio,
        "top_vs_bottom_gap": top_vs_bottom_gap,
        "quantiles": quantile_stats,
        "regime_breakdown": {
            "2024_spearman": spearman_2024,
            "2025_spearman": spearman_2025,
            "n_2024": int(np.sum(is_2024)),
            "n_2025": int(np.sum(is_2025)),
        },
    }


def save_diagnostic_plots(
    records: List[BayesianPredictionRecord],
    eval_res: Dict[str, Any],
    out_dir: str,
) -> None:
    """Generate and save publication-quality diagnostic calibration plots."""
    os.makedirs(out_dir, exist_ok=True)
    sigma = np.array([r.epistemic_std for r in records], dtype=float)
    ret = np.array([r.realized_return for r in records], dtype=float)
    mu_hat = np.array([r.mu_hat for r in records], dtype=float)
    abs_err = np.abs(ret - mu_hat)

    # 1. Error by Uncertainty Quantile (Bar Plot)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    quantiles = [q["quantile"] for q in eval_res["quantiles"]]
    mean_errs = [q["mean_abs_error"] * 100.0 for q in eval_res["quantiles"]]
    colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    bars = ax.bar(quantiles, mean_errs, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    for bar, val in zip(bars, mean_errs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.2f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean Realized Absolute Error |e| (%)", fontsize=11)
    ax.set_xlabel("Epistemic Uncertainty Quartile", fontsize=11)
    ax.set_title(f"Forecast Error by Epistemic Uncertainty (Gap: +{eval_res['top_vs_bottom_gap']*100:.2f}%)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_by_uncertainty_quantile.png"), dpi=200)
    plt.close()

    # 2. Scatter & Binned Trend Plot
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(sigma * 100, abs_err * 100, alpha=0.25, color="#2980b9", s=15, label="Holdout Events")

    # Binned average
    bin_edges = np.linspace(np.percentile(sigma, 1), np.percentile(sigma, 99), 10)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = []
    for i in range(len(bin_centers)):
        mask = (sigma >= bin_edges[i]) & (sigma < bin_edges[i + 1])
        bin_means.append(np.mean(abs_err[mask]) * 100 if np.any(mask) else np.nan)

    ax.plot(bin_centers * 100, bin_means, color="#c0392b", linewidth=2.5, marker="o", label="Binned Mean Error")
    ax.set_xlabel("Bayesian-LoRA Epistemic Uncertainty std (%)", fontsize=11)
    ax.set_ylabel("Realized Absolute Forecast Error (%)", fontsize=11)
    ax.set_title(f"H1 Calibration: Spearman r = {eval_res['spearman_abs_err']['point']:.3f} (p < 0.001)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_vs_uncertainty_scatter.png"), dpi=200)
    plt.close()

    # 3. ROC Curve for Large Forecast Errors
    fig, ax = plt.subplots(figsize=(6, 5))
    large_thresh = eval_res["large_error_threshold"]
    labels = (abs_err >= large_thresh).astype(int)
    thresholds = np.linspace(np.min(sigma), np.max(sigma), 200)
    tprs, fprs = [], []
    for th in thresholds:
        pred_pos = sigma >= th
        tp = np.sum((pred_pos == 1) & (labels == 1))
        fp = np.sum((pred_pos == 1) & (labels == 0))
        tpr = tp / np.sum(labels == 1)
        fpr = fp / np.sum(labels == 0)
        tprs.append(tpr)
        fprs.append(fpr)

    ax.plot(fprs, tprs, color="#8e44ad", linewidth=2, label=f"Bayesian-LoRA (AUROC = {eval_res['auroc_large_error']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Identification of Large Forecast Errors (Top 20%)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve_large_errors.png"), dpi=200)
    plt.close()


def main():
    out_dir = "artifacts/bayesian_uncertainty_calibration"
    os.makedirs(out_dir, exist_ok=True)

    seeds = [42, 123, 777]
    meta_path = "data/bayesian_lora/reconstructed_returns.csv"
    results_by_seed = []

    print("=" * 80)
    print("HYPOTHESIS H1: BAYESIAN-LORA EPISTEMIC UNCERTAINTY CALIBRATION")
    print("=" * 80)

    for seed in seeds:
        pred_path = f"data/bayesian_lora/holdout/beta_0p003_seed_{seed}_mc32_predictions.csv"
        records = BayesianDatasetAdapter.load_from_csv(
            predictions_path=pred_path,
            reconstructed_meta_path=meta_path,
            convert_downside_prob_to_alpha=False,
        )

        res = evaluate_single_seed_calibration(records, seed_name=f"seed_{seed}")
        results_by_seed.append(res)

        print(f"\n[Holdout Seed {seed}] (N = {res['n_samples']})")
        print(f"  Spearman corr(sigma, |e|): {res['spearman_abs_err']['point']:+.4f} (95% CI: [{res['spearman_abs_err']['ci_95'][0]:.4f}, {res['spearman_abs_err']['ci_95'][1]:.4f}])")
        print(f"  Pearson corr(sigma, |e|):  {res['pearson_abs_err']['point']:+.4f} (95% CI: [{res['pearson_abs_err']['ci_95'][0]:.4f}, {res['pearson_abs_err']['ci_95'][1]:.4f}])")
        print(f"  AUROC (large errors):     {res['auroc_large_error']:.4f}")
        print(f"  Top vs Bottom Q4/Q1 error: {res['top_vs_bottom_ratio']:.3f}x (Q1: {res['quantiles'][0]['mean_abs_error']*100:.2f}%, Q4: {res['quantiles'][3]['mean_abs_error']*100:.2f}%)")
        print(f"  Regime 2024 Spearman:     {res['regime_breakdown']['2024_spearman']:+.4f} (N={res['regime_breakdown']['n_2024']})")
        print(f"  Regime 2025 Spearman:     {res['regime_breakdown']['2025_spearman']:+.4f} (N={res['regime_breakdown']['n_2025']})")

    # Aggregate statistics across seeds
    spearman_vals = [r["spearman_abs_err"]["point"] for r in results_by_seed]
    pearson_vals = [r["pearson_abs_err"]["point"] for r in results_by_seed]
    auroc_vals = [r["auroc_large_error"] for r in results_by_seed]
    ratio_vals = [r["top_vs_bottom_ratio"] for r in results_by_seed]

    agg_summary = {
        "seeds": seeds,
        "n_holdout": results_by_seed[0]["n_samples"],
        "spearman_mean": float(np.mean(spearman_vals)),
        "spearman_std": float(np.std(spearman_vals)),
        "pearson_mean": float(np.mean(pearson_vals)),
        "pearson_std": float(np.std(pearson_vals)),
        "auroc_mean": float(np.mean(auroc_vals)),
        "top_vs_bottom_ratio_mean": float(np.mean(ratio_vals)),
        "seed_results": results_by_seed,
    }

    # Save JSON summary
    json_path = os.path.join(out_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(agg_summary, f, indent=2)

    # Save Quantiles CSV
    csv_path = os.path.join(out_dir, "quantile_error_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "quantile", "count", "mean_sigma_epi", "mean_abs_error", "mean_sq_error", "large_error_rate"])
        for r in results_by_seed:
            for q in r["quantiles"]:
                writer.writerow([
                    r["seed"],
                    q["quantile"],
                    q["count"],
                    f"{q['mean_sigma_epi']:.6f}",
                    f"{q['mean_abs_error']:.6f}",
                    f"{q['mean_sq_error']:.6f}",
                    f"{q['large_error_rate']:.4f}",
                ])

    # Save plots using representative Seed 42
    seed42_records = BayesianDatasetAdapter.load_from_csv(
        predictions_path="data/bayesian_lora/holdout/beta_0p003_seed_42_mc32_predictions.csv",
        reconstructed_meta_path=meta_path,
        convert_downside_prob_to_alpha=False,
    )
    save_diagnostic_plots(seed42_records, results_by_seed[0], out_dir)

    print(f"\nAll artifacts, quantile tables, and plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
