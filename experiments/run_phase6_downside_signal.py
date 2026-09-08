"""
Phase 6 (Secondary Track): Asymmetric Downside Signal Evaluation
================================================================
Evaluates whether an explicitly directional downside model outperforms symmetric
epistemic dispersion for predicting harmful downside events (e.g. label=1 earnings drops
and negative return periods r < 0).

Models Compared:
1. Symmetric Epistemic Dispersion (sigma_epi from Bayesian-LoRA)
2. Directional Downside Target (posterior_mean / p_down from Bayesian-LoRA classification)
3. Asymmetric Quantile / Loss calibrated on Dev data

Primary predictive metrics:
- AUROC
- Average Precision (AP)
- Brier score
- Downside recall at fixed precision (e.g. top 20% highest risk predictions)
- Reliability / calibration curves

Outputs saved to: artifacts/phase6_downside_signal/
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, List, Tuple

os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def compute_auroc_and_ap(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Compute AUROC and Average Precision using rank sums and trapezoidal integration."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(np.sum(labels))
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5, float(n_pos / len(labels)) if len(labels) > 0 else 0.0

    ranks = stats.rankdata(scores)
    u_stat = float(np.sum(ranks[labels == 1]) - n_pos * (n_pos + 1) / 2.0)
    auroc = float(u_stat / (n_pos * n_neg))

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    cum_pos = np.cumsum(sorted_labels)
    cum_total = np.arange(1, len(labels) + 1)
    precisions = cum_pos / cum_total
    ap = float(np.sum(precisions * sorted_labels) / n_pos)

    return auroc, ap


def load_prediction_dataset(csv_path: str) -> List[Dict[str, Any]]:
    """Load predictions CSV into list of dictionary records."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            var_epi = float(r.get("epistemic_var", r.get("epistemic_variance", 0.0)))
            std_epi = math.sqrt(max(var_epi, 0.0))
            rows.append({
                "date": r["date"],
                "label": int(r.get("label", 0)),
                "move_1d": float(r["move_1d"]),
                "p_down": float(r["posterior_mean"]),
                "sigma_epi": std_epi,
                "p_det": float(r.get("posterior_mean_deterministic", r["posterior_mean"])),
            })
    return rows


def evaluate_downside_predictions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate directional downside predictions (p_down) vs symmetric epistemic uncertainty (sigma_epi)
    against two ground truth definitions:
    1. label == 1 (earnings downside drop)
    2. move_1d < 0 (realized negative 1-day return)
    """
    labels_l1 = np.array([r["label"] for r in records])
    labels_neg = np.array([1 if r["move_1d"] < 0 else 0 for r in records])

    p_down = np.array([r["p_down"] for r in records])
    sigma_epi = np.array([r["sigma_epi"] for r in records])

    # AUROC & AP
    auc_p_l1, ap_p_l1 = compute_auroc_and_ap(p_down, labels_l1)
    auc_sig_l1, ap_sig_l1 = compute_auroc_and_ap(sigma_epi, labels_l1)

    auc_p_neg, ap_p_neg = compute_auroc_and_ap(p_down, labels_neg)
    auc_sig_neg, ap_sig_neg = compute_auroc_and_ap(sigma_epi, labels_neg)

    # Brier Scores
    brier_p_l1 = float(np.mean((p_down - labels_l1) ** 2))
    # Normalize sigma_epi to [0, 1] range to evaluate Brier proxy
    sig_norm = (sigma_epi - np.min(sigma_epi)) / (np.ptp(sigma_epi) + 1e-12)
    brier_sig_l1 = float(np.mean((sig_norm - labels_l1) ** 2))

    # Recall at Top 20% Highest Risk Predictions
    k_top = int(round(0.20 * len(records)))
    top_p_idx = np.argsort(-p_down)[:k_top]
    top_sig_idx = np.argsort(-sigma_epi)[:k_top]

    recall_p_l1 = float(np.sum(labels_l1[top_p_idx]) / (np.sum(labels_l1) + 1e-12))
    recall_sig_l1 = float(np.sum(labels_l1[top_sig_idx]) / (np.sum(labels_l1) + 1e-12))

    recall_p_neg = float(np.sum(labels_neg[top_p_idx]) / (np.sum(labels_neg) + 1e-12))
    recall_sig_neg = float(np.sum(labels_neg[top_sig_idx]) / (np.sum(labels_neg) + 1e-12))

    # Decile calibration of p_down
    decile_bins = np.linspace(0.0, 1.0, 11)
    calib_data = []
    for i in range(10):
        low, high = decile_bins[i], decile_bins[i + 1]
        mask = (p_down >= low) & (p_down < high)
        cnt = int(np.sum(mask))
        if cnt > 0:
            mean_pred = float(np.mean(p_down[mask]))
            obs_rate = float(np.mean(labels_l1[mask]))
        else:
            mean_pred = float(0.5 * (low + high))
            obs_rate = 0.0
        calib_data.append({
            "bin": i + 1,
            "count": cnt,
            "mean_pred": mean_pred,
            "obs_rate": obs_rate,
        })

    return {
        "auc_p_down_vs_label1": auc_p_l1,
        "ap_p_down_vs_label1": ap_p_l1,
        "auc_sigma_vs_label1": auc_sig_l1,
        "ap_sigma_vs_label1": ap_sig_l1,
        "auc_p_down_vs_neg_return": auc_p_neg,
        "ap_p_down_vs_neg_return": ap_p_neg,
        "auc_sigma_vs_neg_return": auc_sig_neg,
        "ap_sigma_vs_neg_return": ap_sig_neg,
        "brier_p_down": brier_p_l1,
        "brier_sigma_norm": brier_sig_l1,
        "recall_top20_p_down_l1": recall_p_l1,
        "recall_top20_sigma_l1": recall_sig_l1,
        "recall_top20_p_down_neg": recall_p_neg,
        "recall_top20_sigma_neg": recall_sig_neg,
        "calibration": calib_data,
    }


def generate_downside_plots(
    eval_42: Dict[str, Any],
    records_42: List[Dict[str, Any]],
    out_dir: str,
):
    """Generate the 3 required secondary downside figures."""
    p_down = np.array([r["p_down"] for r in records_42])
    sigma_epi = np.array([r["sigma_epi"] for r in records_42])
    labels_l1 = np.array([r["label"] for r in records_42])

    # 1. Downside ROC Curve
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=300)
    # Compute empirical ROC points
    for score, name, col in [(p_down, f"Directional p_down (AUC = {eval_42['auc_p_down_vs_label1']:.3f})", "#0072B2"),
                             (sigma_epi, f"Symmetric sigma_epi (AUC = {eval_42['auc_sigma_vs_label1']:.3f})", "#D55E00")]:
        thresholds = np.linspace(np.min(score), np.max(score), 200)
        tpr = []
        fpr = []
        n_pos = np.sum(labels_l1 == 1)
        n_neg = np.sum(labels_l1 == 0)
        for th in thresholds:
            pred = (score >= th).astype(int)
            tp = np.sum((pred == 1) & (labels_l1 == 1))
            fp = np.sum((pred == 1) & (labels_l1 == 0))
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)
        # Sort by FPR for proper ROC plotting
        order = np.argsort(fpr)
        ax.plot(np.array(fpr)[order], np.array(tpr)[order], lw=2.0, color=col, label=name)

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance (0.50)")
    ax.set_title("Downside Event Classification: ROC Curve (Holdout)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    ax.legend(frameon=True, fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "downside_roc.png"))
    plt.close()

    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=300)
    base_rate = float(np.mean(labels_l1))
    for score, name, col in [(p_down, f"Directional p_down (AP = {eval_42['ap_p_down_vs_label1']:.3f})", "#0072B2"),
                             (sigma_epi, f"Symmetric sigma_epi (AP = {eval_42['ap_sigma_vs_label1']:.3f})", "#D55E00")]:
        order = np.argsort(-score)
        sorted_labels = labels_l1[order]
        cum_pos = np.cumsum(sorted_labels)
        recalls = cum_pos / np.sum(labels_l1)
        precisions = cum_pos / np.arange(1, len(labels_l1) + 1)
        ax.plot(recalls, precisions, lw=2.0, color=col, label=name)

    ax.axhline(base_rate, color="gray", linestyle="--", label=f"Base Downside Rate ({base_rate*100:.1f}%)")
    ax.set_title("Precision-Recall Curve for Downside Detection", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "downside_precision_recall.png"))
    plt.close()

    # 3. Downside Calibration Curve
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=300)
    cal = eval_42["calibration"]
    preds = [c["mean_pred"] for c in cal if c["count"] > 0]
    obss = [c["obs_rate"] for c in cal if c["count"] > 0]

    ax.plot(preds, obss, marker="o", lw=2.0, color="#0072B2", label="Observed Downside Frequency")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect Calibration")
    ax.set_title("Reliability Diagram (Calibration of Directional p_down)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Mean Predicted Downside Probability", fontsize=11)
    ax.set_ylabel("Observed Frequency of Drop (Label = 1)", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "downside_calibration.png"))
    plt.close()


def main():
    print("=" * 80)
    print("PHASE 6 (SECONDARY TRACK): ASYMMETRIC DOWNSIDE SIGNAL EVALUATION")
    print("=" * 80)

    out_dir = "artifacts/phase6_downside_signal"
    os.makedirs(out_dir, exist_ok=True)
    seeds = [42, 123, 777]

    seed_evals = {}
    holdout_records_by_seed = {}

    for s in seeds:
        csv_path = f"data/bayesian_lora/holdout/beta_0p003_seed_{s}_mc32_predictions.csv"
        records = load_prediction_dataset(csv_path)
        holdout_records_by_seed[s] = records
        res = evaluate_downside_predictions(records)
        seed_evals[s] = res

        print(f"  Seed {s}:")
        print(f"    AUROC (p_down vs label1):      {res['auc_p_down_vs_label1']:.4f} vs sigma: {res['auc_sigma_vs_label1']:.4f}")
        print(f"    Avg Precision (p_down vs l1):  {res['ap_p_down_vs_label1']:.4f} vs sigma: {res['ap_sigma_vs_label1']:.4f}")
        print(f"    Recall @ Top 20%:              {res['recall_top20_p_down_l1']*100:.1f}% vs sigma: {res['recall_top20_sigma_l1']*100:.1f}%")

    # Export downside_metrics.csv
    with open(os.path.join(out_dir, "downside_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "auc_p_down_l1", "ap_p_down_l1", "auc_sigma_l1", "ap_sigma_l1",
            "auc_p_down_neg", "ap_p_down_neg", "auc_sigma_neg", "ap_sigma_neg",
            "brier_p_down", "recall_top20_p_l1", "recall_top20_sig_l1"
        ])
        for s in seeds:
            ev = seed_evals[s]
            writer.writerow([
                s, f"{ev['auc_p_down_vs_label1']:.4f}", f"{ev['ap_p_down_vs_label1']:.4f}",
                f"{ev['auc_sigma_vs_label1']:.4f}", f"{ev['ap_sigma_vs_label1']:.4f}",
                f"{ev['auc_p_down_vs_neg_return']:.4f}", f"{ev['ap_p_down_vs_neg_return']:.4f}",
                f"{ev['auc_sigma_vs_neg_return']:.4f}", f"{ev['ap_sigma_vs_neg_return']:.4f}",
                f"{ev['brier_p_down']:.6f}", f"{ev['recall_top20_p_down_l1']:.4f}", f"{ev['recall_top20_sigma_l1']:.4f}"
            ])

    # Export calibration.csv
    with open(os.path.join(out_dir, "calibration.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "bin", "count", "mean_pred", "obs_rate"])
        for s in seeds:
            for c in seed_evals[s]["calibration"]:
                writer.writerow([s, c["bin"], c["count"], f"{c['mean_pred']:.4f}", f"{c['obs_rate']:.4f}"])

    # Export metrics_summary.json
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump({"seeds": seeds, "evaluations": seed_evals}, f, indent=2)

    # Generate figures
    generate_downside_plots(seed_evals[42], holdout_records_by_seed[42], out_dir)
    print(f"\n  Figures and artifacts saved to {out_dir}/")
    print("=" * 80)
    print("PHASE 6 DOWNSIDE EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
