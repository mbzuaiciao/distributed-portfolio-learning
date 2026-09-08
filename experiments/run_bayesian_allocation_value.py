"""
Experiment 14: Empirical Value of Bayesian Predictive Uncertainty in Portfolio Allocation
========================================================================================
Rigorous controlled study testing whether and when epistemic uncertainty adds economic
value conditional on the SAME point forecast mu_hat.

Key Experimental Controls:
- Point forecast mu_hat is identical across methods (isolating uncertainty from forecast skill).
- 5 Calibration Regimes:
    1. Well-Calibrated: Uncertainty strongly ranks forecast errors.
    2. Weakly-Calibrated: Noisy correlation with forecast errors.
    3. Uninformative: Pure random noise independent of errors.
    4. Miscalibrated: Anti-correlated (high confidence on erroneous forecasts).
    5. Regime Shift: Sudden economic break with spike in epistemic uncertainty.
- 30 Random Seeds with paired comparisons and empirical standard errors.
- Comprehensive Metrics: Sharpe, Sortino, Max Drawdown, Turnover, Cost-Adjusted Return,
  Realized Utility, Regret vs Oracle, Tail Risk.
- Saves 6 figures and JSON/CSV summary to artifacts/bayesian_allocation_value/.
"""

import csv
import json
import os

os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from hypergraph_portfolio.uncertainty import (
    SyntheticMarketEnvironment,
    run_sequential_portfolio_simulation,
)


def run_single_seed_experiment(seed: int, regime: str, lambda_u: float = 50.0):
    env = SyntheticMarketEnvironment(num_assets=5, horizon=120, regime=regime, seed=seed)
    traj = env.generate_trajectory()
    sim_res = run_sequential_portfolio_simulation(
        trajectory=traj,
        Q=env.Q,
        lambda_u=lambda_u,
        gamma=2.0,
        transaction_cost=0.0010,  # 10 bps
        kappa_turnover=0.0005,
        include_oracle=True,
    )
    return env, traj, sim_res


def run_multiseed_regime_study(
    num_seeds: int = 30,
    regimes: Optional[List[str]] = None,
    lambda_u: float = 50.0,
):
    if regimes is None:
        regimes = [
            "well_calibrated",
            "weakly_calibrated",
            "uninformative",
            "miscalibrated",
            "regime_shift",
        ]

    summary_records = []
    regime_results = {}

    print("=" * 85)
    print(f"RUNNING BAYESIAN UNCERTAINTY EXPERIMENT ({num_seeds} SEEDS PER REGIME, lambda_u={lambda_u})")
    print("=" * 85)

    for regime in regimes:
        regime_results[regime] = {
            "Deterministic": [],
            "Uncertainty-Aware": [],
            "Oracle": [],
            "Paired_Sharpe_Diff": [],
            "Paired_Utility_Diff": [],
            "Paired_Drawdown_Diff": [],
        }

        for s in range(num_seeds):
            seed = 1000 + s
            env, traj, res = run_single_seed_experiment(seed=seed, regime=regime, lambda_u=lambda_u)

            det = res["Deterministic (mu_hat)"]
            unc = res["Uncertainty-Aware"]
            ora = res["Oracle (mu_true)"]

            regime_results[regime]["Deterministic"].append(det)
            regime_results[regime]["Uncertainty-Aware"].append(unc)
            regime_results[regime]["Oracle"].append(ora)

            # Paired differences (Uncertainty-Aware minus Deterministic)
            regime_results[regime]["Paired_Sharpe_Diff"].append(unc.sharpe_ratio - det.sharpe_ratio)
            regime_results[regime]["Paired_Utility_Diff"].append(unc.mean_utility - det.mean_utility)
            regime_results[regime]["Paired_Drawdown_Diff"].append(unc.max_drawdown - det.max_drawdown)

        # Aggregate statistics
        p_sharpe = np.array(regime_results[regime]["Paired_Sharpe_Diff"])
        p_util = np.array(regime_results[regime]["Paired_Utility_Diff"])
        p_dd = np.array(regime_results[regime]["Paired_Drawdown_Diff"])

        det_sharpes = [r.sharpe_ratio for r in regime_results[regime]["Deterministic"]]
        unc_sharpes = [r.sharpe_ratio for r in regime_results[regime]["Uncertainty-Aware"]]
        ora_sharpes = [r.sharpe_ratio for r in regime_results[regime]["Oracle"]]

        det_utils = [r.mean_utility for r in regime_results[regime]["Deterministic"]]
        unc_utils = [r.mean_utility for r in regime_results[regime]["Uncertainty-Aware"]]

        det_dds = [r.max_drawdown for r in regime_results[regime]["Deterministic"]]
        unc_dds = [r.max_drawdown for r in regime_results[regime]["Uncertainty-Aware"]]

        rec = {
            "regime": regime,
            "lambda_u": lambda_u,
            "det_sharpe_mean": float(np.mean(det_sharpes)),
            "det_sharpe_std": float(np.std(det_sharpes)),
            "unc_sharpe_mean": float(np.mean(unc_sharpes)),
            "unc_sharpe_std": float(np.std(unc_sharpes)),
            "ora_sharpe_mean": float(np.mean(ora_sharpes)),
            "paired_sharpe_diff_mean": float(np.mean(p_sharpe)),
            "paired_sharpe_diff_se": float(np.std(p_sharpe) / np.sqrt(num_seeds)),
            "det_utility_mean": float(np.mean(det_utils)),
            "unc_utility_mean": float(np.mean(unc_utils)),
            "paired_utility_diff_mean": float(np.mean(p_util)),
            "paired_utility_diff_se": float(np.std(p_util) / np.sqrt(num_seeds)),
            "det_drawdown_mean": float(np.mean(det_dds)),
            "unc_drawdown_mean": float(np.mean(unc_dds)),
            "paired_drawdown_diff_mean": float(np.mean(p_dd)),
        }
        summary_records.append(rec)

        print(f"\nRegime: [{regime.upper()}]")
        print(f"  Deterministic Sharpe:     {rec['det_sharpe_mean']:.3f} +/- {rec['det_sharpe_std']:.3f}")
        print(f"  Uncertainty-Aware Sharpe: {rec['unc_sharpe_mean']:.3f} +/- {rec['unc_sharpe_std']:.3f}")
        print(f"  Paired Sharpe Delta:      {rec['paired_sharpe_diff_mean']:+.3f} (SE: {rec['paired_sharpe_diff_se']:.3f})")
        print(f"  Paired Utility Delta:     {rec['paired_utility_diff_mean']:+.5f} (SE: {rec['paired_utility_diff_se']:.5f})")
        print(f"  Max Drawdown (Det vs Unc): {rec['det_drawdown_mean']*100:.1f}% vs {rec['unc_drawdown_mean']*100:.1f}%")

    return summary_records, regime_results


def run_lambda_sensitivity_study(num_seeds: int = 15):
    """Test utility across varying shrinkage intensities lambda_u."""
    lambda_vals = [0.0, 10.0, 25.0, 50.0, 75.0, 100.0]
    results_by_lambda = {l_val: [] for l_val in lambda_vals}

    for l_val in lambda_vals:
        for s in range(num_seeds):
            seed = 2000 + s
            _, _, res = run_single_seed_experiment(seed=seed, regime="well_calibrated", lambda_u=l_val)
            det = res["Deterministic (mu_hat)"]
            unc = res["Uncertainty-Aware"]
            results_by_lambda[l_val].append(unc.mean_utility - det.mean_utility)

    mean_diffs = [float(np.mean(results_by_lambda[lam])) for lam in lambda_vals]
    se_diffs = [float(np.std(results_by_lambda[lam]) / np.sqrt(num_seeds)) for lam in lambda_vals]
    return lambda_vals, mean_diffs, se_diffs


def save_artifacts_and_plots(summary_records, regime_results, lambda_study_data):
    out_dir = "artifacts/bayesian_allocation_value"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Save JSON summary
    json_path = os.path.join(out_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_records, f, indent=2)

    # 2. Save CSV summary
    csv_path = os.path.join(out_dir, "metrics_summary.csv")
    if summary_records:
        keys = list(summary_records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_records)

    # 3. Figure 1: Cumulative Wealth Trajectory (Sample Seed)
    sample_env, _, sample_res = run_single_seed_experiment(seed=42, regime="well_calibrated", lambda_u=1.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, m in sample_res.items():
        color = "#2980b9" if "Deterministic" in name else ("#27ae60" if "Uncertainty" in name else "#8e44ad")
        lw = 2.0 if "Uncertainty" in name else 1.5
        ax.plot(m.wealth_path, label=name, color=color, linewidth=lw)
    ax.set_title("1. Cumulative Wealth Trajectory (Well-Calibrated Uncertainty, Seed 42)")
    ax.set_xlabel("Time Steps (Months)")
    ax.set_ylabel("Wealth ($)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "cumulative_wealth.png"), dpi=200)
    plt.close()

    # 4. Figure 2: Drawdown Profile
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, m in sample_res.items():
        peak = np.maximum.accumulate(m.wealth_path)
        dd = (m.wealth_path - peak) / peak
        color = "#2980b9" if "Deterministic" in name else ("#27ae60" if "Uncertainty" in name else "#8e44ad")
        ax.plot(dd * 100, label=name, color=color)
    ax.set_title("2. Drawdown Profile (Well-Calibrated Regime)")
    ax.set_xlabel("Time Steps (Months)")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "drawdown_profile.png"), dpi=200)
    plt.close()

    # 5. Figure 3: Uncertainty vs Forecast Error Calibration Scatter
    traj = sample_env.generate_trajectory()
    errors = []
    uncs = []
    for step in traj:
        err = np.abs(step.mu_hat - step.true_mu)
        errors.extend(err)
        uncs.extend(step.sigma_epi)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(uncs, errors, alpha=0.4, color="#34495e", edgecolors="none")
    z = np.polyfit(uncs, errors, 1)
    p = np.poly1d(z)
    ax.plot(np.sort(uncs), p(np.sort(uncs)), "r--", linewidth=2, label=f"Trend (Slope: {z[0]:.2f})")
    ax.set_xlabel("Epistemic Uncertainty sigma_epi")
    ax.set_ylabel("Actual Forecast Error |mu_hat - mu_true|")
    ax.set_title("3. Calibration Quality: Uncertainty vs Realized Error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "uncertainty_vs_error.png"), dpi=200)
    plt.close()

    # 6. Figure 4: Portfolio Exposure to Most Uncertain Asset
    fig, ax = plt.subplots(figsize=(10, 4))
    det_w = sample_res["Deterministic (mu_hat)"].weights_path[:, 0]
    unc_w = sample_res["Uncertainty-Aware"].weights_path[:, 0]
    ax.plot(det_w, label="Deterministic Weight on Asset 0", color="#e74c3c", alpha=0.8)
    ax.plot(unc_w, label="Uncertainty-Aware Weight on Asset 0", color="#27ae60", alpha=0.8)
    ax.set_title("4. Weight Allocation to High-Uncertainty Asset Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Portfolio Weight")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "uncertain_asset_exposure.png"), dpi=200)
    plt.close()

    # 7. Figure 5: Utility Difference Across Lambda_u
    l_vals, u_diffs, u_ses = lambda_study_data
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(l_vals, u_diffs, yerr=u_ses, fmt="o-", color="#2980b9", capsize=4, linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Uncertainty Penalty Coefficient lambda_u")
    ax.set_ylabel("Mean Realized Utility Delta (Uncertainty - Deterministic)")
    ax.set_title("5. Utility Delta vs Shrinkage Intensity lambda_u")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "utility_across_lambda.png"), dpi=200)
    plt.close()

    # 8. Figure 6: Paired Sharpe Delta Across 5 Regimes
    reg_names = [r["regime"].replace("_", "\n") for r in summary_records]
    sharpe_deltas = [r["paired_sharpe_diff_mean"] for r in summary_records]
    sharpe_ses = [r["paired_sharpe_diff_se"] for r in summary_records]
    colors = ["#27ae60" if d > 0 else "#e74c3c" for d in sharpe_deltas]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(reg_names, sharpe_deltas, yerr=sharpe_ses, capsize=4, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Paired Sharpe Ratio Delta (Uncertainty - Deterministic)")
    ax.set_title("6. When Does Epistemic Uncertainty Add Economic Value?")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "regime_comparison.png"), dpi=200)
    plt.close()

    print(f"\nAll artifacts and 6 plots successfully saved to {out_dir}/")


if __name__ == "__main__":
    records, res_by_regime = run_multiseed_regime_study(num_seeds=30, lambda_u=50.0)
    l_data = run_lambda_sensitivity_study(num_seeds=15)
    save_artifacts_and_plots(records, res_by_regime, l_data)
