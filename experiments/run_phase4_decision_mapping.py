"""
Phase 4: Convert Epistemic Uncertainty into Decision-Relevant Risk
==================================================================
Determines whether Bayesian-LoRA epistemic uncertainty can be converted
into economic portfolio value using decision mappings other than symmetric mean shrinkage:

Policies Evaluated:
- M0: Deterministic Point Forecast (Baseline)
- M1: Symmetric Mean Shrinkage (Phase 3 Benchmark)
- M2: Bayesian Covariance Inflation (Q + lambda_cov * diag(sigma_epi^2))
- M3: Downside-Adjusted Alpha (mu_hat * max(0, 1 - lambda_down * p_down))
- M4: Downside Risk Inflation (Q + lambda_down * diag(p_down * sigma_epi^2))

Protocol:
1. Tune all hyperparameters strictly on 2022-2023 development set (zero holdout leakage).
2. Freeze hyperparameters before evaluating holdout.
3. Evaluate M0 through M4 on 2024-2025 frozen holdout across Seeds 42, 123, 777.
4. Perform Selectivity Analysis (Bad/Good capital allocation ratio).
5. Conduct High-Alpha / High-Uncertainty 4-bin analysis.
6. Calibrate downside risk directly (predicting r < 0 and label == 1).
7. Compute paired block bootstrap (B = 4 weeks, 1,000 resamples) confidence intervals.
8. Save machine-readable artifacts and publication figures.
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import os
from collections import defaultdict

os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from hypergraph_portfolio.bayesian_lora_adapter import (
    BayesianDatasetAdapter,
    BayesianPredictionRecord,
)
from hypergraph_portfolio.mean_variance import MeanVariancePortfolio
from hypergraph_portfolio.uncertainty import (
    apply_downside_adjusted_alpha,
    apply_downside_covariance_inflation,
    apply_epistemic_covariance_inflation,
    apply_uncertainty_shrinkage,
    compute_posterior_downside_probability,
)


def group_by_weekly_rebalance(records: List[BayesianPredictionRecord]) -> Dict[str, List[BayesianPredictionRecord]]:
    """Group records into weekly sequential rebalancing decision epochs."""
    buckets = defaultdict(list)
    for r in records:
        dt = datetime.datetime.fromisoformat(r.timestamp.replace(" ", "T"))
        w_key = f"{dt.year}-W{dt.isocalendar()[1]:02d}"
        buckets[w_key].append(r)
    return dict(sorted(buckets.items()))


def compute_cross_sectional_turnover(
    current_map: Dict[str, float],
    previous_map: Optional[Dict[str, float]],
) -> float:
    """Compute rebalancing turnover L1 distance across varying asset cross-sections."""
    if previous_map is None:
        return 0.0
    all_assets = set(current_map.keys()) | set(previous_map.keys())
    return float(sum(abs(current_map.get(a, 0.0) - previous_map.get(a, 0.0)) for a in all_assets))


def compute_auroc_and_ap(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Compute AUROC and Average Precision using rank sums and trapezoidal integration."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(np.sum(labels))
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5, float(n_pos / len(labels))

    ranks = stats.rankdata(scores)
    u_stat = float(np.sum(ranks[labels == 1]) - n_pos * (n_pos + 1) / 2.0)
    auroc = u_stat / (n_pos * n_neg)

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    cum_pos = np.cumsum(sorted_labels)
    cum_total = np.arange(1, len(labels) + 1)
    precisions = cum_pos / cum_total
    ap = float(np.sum(precisions * sorted_labels) / n_pos)

    return float(auroc), float(ap)


# --------------------------------------------------------------------------
# Development Hyperparameter Tuning (Zero Holdout Leakage)
# --------------------------------------------------------------------------

def tune_hyperparameters_on_dev(
    dev_records: List[BayesianPredictionRecord],
    gamma: float = 1.0,
    cost_bps: float = 0.0010,
) -> Dict[str, Any]:
    """
    Tune hyperparameters for M1, M2, M3, M4 strictly on 2022-2023 development set.
    Returns optimal values and parameter sweeps.
    """
    buckets = group_by_weekly_rebalance(dev_records)

    # Grids
    grids = {
        "M1_shrinkage": [0.0, 5.0, 15.0, 30.0, 50.0, 100.0],
        "M2_cov_inflation": [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        "M3_downside_alpha": [0.0, 0.2, 0.5, 1.0, 1.5, 2.0],
        "M4_downside_cov": [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    }

    results = {}

    for method, param_list in grids.items():
        util_map = {}
        for param in param_list:
            utilities = []
            prev_w_map = None

            for w_key, basket in buckets.items():
                m = len(basket)
                assets = [r.asset for r in basket]
                mu_hat = np.array([r.mu_hat for r in basket])
                sigma = np.array([r.epistemic_std for r in basket])
                real_r = np.array([r.realized_return for r in basket])
                Q = 0.04 * np.eye(m) + 0.01 * np.ones((m, m))

                # Apply policy
                if method == "M1_shrinkage":
                    mu_used = apply_uncertainty_shrinkage(mu_hat, sigma, lambda_u=param)
                    Q_used = Q
                elif method == "M2_cov_inflation":
                    mu_used = mu_hat
                    Q_used = apply_epistemic_covariance_inflation(Q, sigma, lambda_cov=param)
                elif method == "M3_downside_alpha":
                    p_down = compute_posterior_downside_probability(mu_hat, sigma, r_hurdle=0.0)
                    mu_used = apply_downside_adjusted_alpha(mu_hat, p_down, lambda_down=param)
                    Q_used = Q
                elif method == "M4_downside_cov":
                    p_down = compute_posterior_downside_probability(mu_hat, sigma, r_hurdle=0.0)
                    mu_used = mu_hat
                    Q_used = apply_downside_covariance_inflation(Q, p_down, sigma, lambda_down=param)
                else:
                    mu_used = mu_hat
                    Q_used = Q

                try:
                    w, _ = MeanVariancePortfolio(Q=Q_used, mu=mu_used, budget=1.0).solve()
                except Exception:
                    w = np.ones(m) / float(m)

                cur_w_map = {assets[i]: float(w[i]) for i in range(m)}
                to = compute_cross_sectional_turnover(cur_w_map, prev_w_map)
                port_r = float(w @ real_r)
                port_var = float(w @ Q @ w)  # Evaluated under true market risk
                net_u = port_r - 0.5 * gamma * port_var - cost_bps * to

                utilities.append(net_u)
                prev_w_map = cur_w_map

            util_map[param] = float(np.mean(utilities))

        best_param = max(util_map.keys(), key=lambda k: util_map[k])
        results[method] = {
            "grid_results": util_map,
            "optimal_param": best_param,
            "optimal_dev_utility": util_map[best_param],
        }

    return results


# --------------------------------------------------------------------------
# Out-of-Sample Holdout Simulation
# --------------------------------------------------------------------------

def run_phase4_holdout_simulation(
    holdout_records: List[BayesianPredictionRecord],
    hyperparams: Dict[str, float],
    gamma: float = 1.0,
    cost_bps: float = 0.0010,
) -> Dict[str, Any]:
    """
    Run backtest of M0, M1, M2, M3, M4 on 2024-2025 holdout with frozen hyperparameters.
    """
    buckets = group_by_weekly_rebalance(holdout_records)
    periods = list(buckets.keys())
    T = len(periods)

    policies = ["M0_Deterministic", "M1_SymmetricShrink", "M2_CovInflation", "M3_DownsideAlpha", "M4_DownsideCov"]

    history = {
        p: {
            "returns": [],
            "utilities": [],
            "turnovers": [],
            "hhi": [],
            "wealth": [1.0],
            "weights": [],
        }
        for p in policies
    }

    # Tracking per-asset decision records for selectivity & binning
    asset_eval_records = []
    prev_weights = {p: None for p in policies}

    for p_idx, w_key in enumerate(periods):
        basket = buckets[w_key]
        m = len(basket)
        assets = [r.asset for r in basket]
        mu_hat = np.array([r.mu_hat for r in basket])
        sigma = np.array([r.epistemic_std for r in basket])
        real_r = np.array([r.realized_return for r in basket])
        labels = np.array([r.metadata.get("label", 0) for r in basket])
        raw_probs = np.array([r.metadata.get("raw_probability", 0.5) for r in basket])

        Q_market = 0.04 * np.eye(m) + 0.01 * np.ones((m, m))

        # Posterior downside probability
        p_down = compute_posterior_downside_probability(mu_hat, sigma, r_hurdle=0.0)

        # Build policy inputs
        policy_inputs = {
            "M0_Deterministic": (mu_hat, Q_market),
            "M1_SymmetricShrink": (
                apply_uncertainty_shrinkage(mu_hat, sigma, lambda_u=hyperparams["M1_shrinkage"]),
                Q_market,
            ),
            "M2_CovInflation": (
                mu_hat,
                apply_epistemic_covariance_inflation(Q_market, sigma, lambda_cov=hyperparams["M2_cov_inflation"]),
            ),
            "M3_DownsideAlpha": (
                apply_downside_adjusted_alpha(mu_hat, p_down, lambda_down=hyperparams["M3_downside_alpha"]),
                Q_market,
            ),
            "M4_DownsideCov": (
                mu_hat,
                apply_downside_covariance_inflation(Q_market, p_down, sigma, lambda_down=hyperparams["M4_downside_cov"]),
            ),
        }

        weights = {}
        for p in policies:
            mu_p, Q_p = policy_inputs[p]
            try:
                w, _ = MeanVariancePortfolio(Q=Q_p, mu=mu_p, budget=1.0).solve()
            except Exception:
                w = np.ones(m) / float(m)
            weights[p] = w

        # Record asset level statistics for selectivity analysis
        for i in range(m):
            err_i = real_r[i] - mu_hat[i]
            asset_eval_records.append({
                "period": w_key,
                "asset": assets[i],
                "mu_hat": mu_hat[i],
                "sigma_epi": sigma[i],
                "p_down": p_down[i],
                "raw_prob": raw_probs[i],
                "label": labels[i],
                "realized_return": real_r[i],
                "forecast_error": err_i,
                "abs_error": abs(err_i),
                "w_M0": weights["M0_Deterministic"][i],
                "w_M1": weights["M1_SymmetricShrink"][i],
                "w_M2": weights["M2_CovInflation"][i],
                "w_M3": weights["M3_DownsideAlpha"][i],
                "w_M4": weights["M4_DownsideCov"][i],
            })

        # Update portfolio time series
        for p in policies:
            w = weights[p]
            cur_map = {assets[i]: float(w[i]) for i in range(m)}
            to = compute_cross_sectional_turnover(cur_map, prev_weights[p])
            port_r = float(w @ real_r)
            tc = cost_bps * to
            net_r = port_r - tc
            port_var = float(w @ Q_market @ w)
            u = net_r - 0.5 * gamma * port_var - cost_bps * to

            history[p]["returns"].append(net_r)
            history[p]["utilities"].append(u)
            history[p]["turnovers"].append(to)
            history[p]["hhi"].append(float(np.sum(w**2)))
            history[p]["wealth"].append(history[p]["wealth"][-1] * (1.0 + net_r))
            history[p]["weights"].append(w.copy())
            prev_weights[p] = cur_map

    # Compute holdout performance metrics
    ann_factor = 52.0
    metrics = {}

    for p in policies:
        r_arr = np.array(history[p]["returns"])
        u_arr = np.array(history[p]["utilities"])
        w_arr = np.array(history[p]["wealth"])

        ann_ret = float(np.mean(r_arr) * ann_factor)
        ann_vol = float(np.std(r_arr) * np.sqrt(ann_factor))
        sharpe = float(ann_ret / (ann_vol + 1e-12))

        # Peak and Max Drawdown
        peak = np.maximum.accumulate(w_arr)
        drawdowns = (w_arr - peak) / peak
        max_dd = float(np.min(drawdowns))

        # Downside deviation & Sortino
        neg_r = r_arr[r_arr < 0]
        downside_vol = float(np.std(neg_r) * np.sqrt(ann_factor)) if len(neg_r) > 0 else 1e-6
        sortino = float(ann_ret / (downside_vol + 1e-12))

        # Worst 5% return
        worst_5pct = float(np.percentile(r_arr, 5))

        metrics[p] = {
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "worst_5pct_return": worst_5pct,
            "downside_deviation": downside_vol,
            "mean_turnover": float(np.mean(history[p]["turnovers"])),
            "mean_hhi": float(np.mean(history[p]["hhi"])),
            "mean_utility": float(np.mean(u_arr)),
            "cumulative_return": float(w_arr[-1] - 1.0),
            "cumulative_utility": float(np.sum(u_arr)),
        }

    return {
        "metrics": metrics,
        "history": history,
        "asset_eval_records": asset_eval_records,
        "n_periods": T,
    }


# --------------------------------------------------------------------------
# Selectivity & High-Alpha Analysis
# --------------------------------------------------------------------------

def compute_selectivity_analysis(asset_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ex-post classify forecasts into Best 20%, Middle 60%, Worst 20%.
    Measure pre-allocated weights under each policy.
    """
    # A forecast outcome is good if realized return is high or forecast error is small
    # Here, following standard attribution, we evaluate based on realized return outcomes
    returns = np.array([r["realized_return"] for r in asset_records])
    p20 = np.percentile(returns, 20)
    p80 = np.percentile(returns, 80)

    policies = ["M0_Deterministic", "M1_SymmetricShrink", "M2_CovInflation", "M3_DownsideAlpha", "M4_DownsideCov"]
    prefix_map = {
        "M0_Deterministic": "w_M0",
        "M1_SymmetricShrink": "w_M1",
        "M2_CovInflation": "w_M2",
        "M3_DownsideAlpha": "w_M3",
        "M4_DownsideCov": "w_M4",
    }

    res = {}
    for p in policies:
        col = prefix_map[p]
        w_all = np.array([r[col] for r in asset_records])
        w_best = w_all[returns >= p80]
        w_mid = w_all[(returns > p20) & (returns < p80)]
        w_worst = w_all[returns <= p20]

        mean_best = float(np.mean(w_best)) if len(w_best) > 0 else 0.0
        mean_worst = float(np.mean(w_worst)) if len(w_worst) > 0 else 0.0
        bad_good_ratio = float(mean_worst / (mean_best + 1e-12))

        res[p] = {
            "mean_weight_best20": mean_best,
            "mean_weight_middle60": float(np.mean(w_mid)) if len(w_mid) > 0 else 0.0,
            "mean_weight_worst20": mean_worst,
            "bad_to_good_ratio": bad_good_ratio,
        }

    return res


def compute_alpha_uncertainty_bins(asset_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split holdout into 4 bins:
    1. low alpha / low uncertainty
    2. low alpha / high uncertainty
    3. high alpha / low uncertainty
    4. high alpha / high uncertainty
    """
    abs_alphas = np.array([abs(r["mu_hat"]) for r in asset_records])
    sigmas = np.array([r["sigma_epi"] for r in asset_records])

    med_alpha = float(np.median(abs_alphas))
    med_sigma = float(np.median(sigmas))

    bin_defs = [
        ("Low Alpha / Low Uncertainty", abs_alphas < med_alpha, sigmas < med_sigma),
        ("Low Alpha / High Uncertainty", abs_alphas < med_alpha, sigmas >= med_sigma),
        ("High Alpha / Low Uncertainty", abs_alphas >= med_alpha, sigmas < med_sigma),
        ("High Alpha / High Uncertainty", abs_alphas >= med_alpha, sigmas >= med_sigma),
    ]

    bin_results = []
    for bin_name, alpha_mask, sigma_mask in bin_defs:
        mask = alpha_mask & sigma_mask
        sub = [asset_records[i] for i in range(len(asset_records)) if mask[i]]
        count = len(sub)
        if count == 0:
            continue

        real_rets = np.array([r["realized_return"] for r in sub])
        abs_errs = np.array([r["abs_error"] for r in sub])
        labels = np.array([r["label"] for r in sub])
        is_neg = (real_rets < 0).astype(float)

        bin_results.append({
            "bin": bin_name,
            "count": count,
            "mean_realized_return": float(np.mean(real_rets)),
            "mean_abs_error": float(np.mean(abs_errs)),
            "downside_freq_neg_ret": float(np.mean(is_neg)),
            "downside_freq_label1": float(np.mean(labels)),
            "mean_w_M0": float(np.mean([r["w_M0"] for r in sub])),
            "mean_w_M1": float(np.mean([r["w_M1"] for r in sub])),
            "mean_w_M2": float(np.mean([r["w_M2"] for r in sub])),
            "mean_w_M3": float(np.mean([r["w_M3"] for r in sub])),
            "mean_w_M4": float(np.mean([r["w_M4"] for r in sub])),
        })

    return bin_results


def compute_downside_calibration(asset_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calibrate epistemic uncertainty and p_down directly against realized downside:
    1. Realized negative return (r < 0)
    2. Large downside earnings event (label == 1)
    """
    sigmas = np.array([r["sigma_epi"] for r in asset_records])
    p_downs = np.array([r["p_down"] for r in asset_records])
    rets = np.array([r["realized_return"] for r in asset_records])
    labels = np.array([r["label"] for r in asset_records], dtype=int)
    is_neg = (rets < 0).astype(int)

    # AUROC & AP
    auroc_neg_sigma, ap_neg_sigma = compute_auroc_and_ap(sigmas, is_neg)
    auroc_neg_pdown, ap_neg_pdown = compute_auroc_and_ap(p_downs, is_neg)
    auroc_l1_sigma, ap_l1_sigma = compute_auroc_and_ap(sigmas, labels)
    auroc_l1_pdown, ap_l1_pdown = compute_auroc_and_ap(p_downs, labels)

    # Quintiles of p_down
    p_quintiles = np.quantile(p_downs, [0.2, 0.4, 0.6, 0.8])
    p_bins = np.digitize(p_downs, p_quintiles)

    quintile_breakdown = []
    for q in range(5):
        mask = p_bins == q
        if np.sum(mask) == 0:
            continue
        quintile_breakdown.append({
            "quintile": f"Q{q+1}",
            "count": int(np.sum(mask)),
            "mean_p_down": float(np.mean(p_downs[mask])),
            "realized_neg_rate": float(np.mean(is_neg[mask])),
            "realized_label1_rate": float(np.mean(labels[mask])),
            "mean_realized_return": float(np.mean(rets[mask])),
        })

    # Brier score for p_down vs is_neg
    brier_neg = float(np.mean((p_downs - is_neg)**2))

    return {
        "auroc_neg_return_sigma": auroc_neg_sigma,
        "ap_neg_return_sigma": ap_neg_sigma,
        "auroc_neg_return_pdown": auroc_neg_pdown,
        "ap_neg_return_pdown": ap_neg_pdown,
        "auroc_label1_sigma": auroc_l1_sigma,
        "ap_label1_sigma": ap_l1_sigma,
        "auroc_label1_pdown": auroc_l1_pdown,
        "ap_label1_pdown": ap_l1_pdown,
        "brier_score_pdown_vs_neg": brier_neg,
        "quintiles": quintile_breakdown,
    }


# --------------------------------------------------------------------------
# Paired Block Bootstrap
# --------------------------------------------------------------------------

def compute_paired_block_bootstrap(
    history: Dict[str, Dict[str, List[float]]],
    block_size: int = 4,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform paired circular block bootstrap over weekly rebalancing epochs.
    Calculates 95% confidence intervals for Delta Utility and Delta Return vs M0.
    """
    rng = np.random.default_rng(seed)
    T = len(history["M0_Deterministic"]["returns"])
    policies = ["M1_SymmetricShrink", "M2_CovInflation", "M3_DownsideAlpha", "M4_DownsideCov"]

    u_m0 = np.array(history["M0_Deterministic"]["utilities"])
    r_m0 = np.array(history["M0_Deterministic"]["returns"])

    results = {}

    for p in policies:
        u_p = np.array(history[p]["utilities"])
        r_p = np.array(history[p]["returns"])

        delta_u = u_p - u_m0
        delta_r = r_p - r_m0

        point_delta_u = float(np.mean(delta_u))
        point_delta_r = float(np.mean(delta_r))

        boot_du = []
        boot_dr = []

        n_blocks = int(math.ceil(T / block_size))
        for _ in range(n_bootstrap):
            # Sample block start indices
            start_indices = rng.integers(0, T, size=n_blocks)
            sampled_idx = []
            for s in start_indices:
                for b in range(block_size):
                    sampled_idx.append((s + b) % T)
            sampled_idx = np.array(sampled_idx[:T])

            boot_du.append(float(np.mean(delta_u[sampled_idx])))
            boot_dr.append(float(np.mean(delta_r[sampled_idx])))

        ci_du = [float(np.percentile(boot_du, 2.5)), float(np.percentile(boot_du, 97.5))]
        ci_dr = [float(np.percentile(boot_dr, 2.5)), float(np.percentile(boot_dr, 97.5))]

        # Block bootstrap t-statistic and pseudo p-value
        se_du = float(np.std(boot_du))
        t_stat_du = float(point_delta_u / (se_du + 1e-12))
        p_val_du = float(2.0 * min(np.mean(np.array(boot_du) <= 0), np.mean(np.array(boot_du) >= 0)))

        results[p] = {
            "point_delta_utility": point_delta_u,
            "ci95_delta_utility": ci_du,
            "se_delta_utility": se_du,
            "t_stat_utility": t_stat_du,
            "p_val_utility": p_val_du,
            "point_delta_return": point_delta_r,
            "ci95_delta_return": ci_dr,
        }

    return results


# --------------------------------------------------------------------------
# Plotting Functions
# --------------------------------------------------------------------------

def generate_phase4_plots(
    sim_result: Dict[str, Any],
    selectivity: Dict[str, Any],
    alpha_bins: List[Dict[str, Any]],
    downside_cal: Dict[str, Any],
    out_dir: str,
):
    """Generate the 6 publication-ready diagnostic charts."""
    history = sim_result["history"]

    policy_colors = {
        "M0_Deterministic": "#333333",
        "M1_SymmetricShrink": "#E69F00",
        "M2_CovInflation": "#0072B2",
        "M3_DownsideAlpha": "#009E73",
        "M4_DownsideCov": "#D55E00",
    }
    policy_labels = {
        "M0_Deterministic": "M0: Deterministic",
        "M1_SymmetricShrink": "M1: Symmetric Shrink",
        "M2_CovInflation": "M2: Cov Inflation",
        "M3_DownsideAlpha": "M3: Downside Alpha",
        "M4_DownsideCov": "M4: Downside Cov",
    }

    # 1. Cumulative Wealth
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    for p, c in policy_colors.items():
        wealth = history[p]["wealth"]
        ax.plot(wealth, label=policy_labels[p], color=c, lw=1.8)
    ax.set_title("Phase 4: Cumulative Wealth on Holdout (2024–2025)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Rebalancing Epoch (Weeks)")
    ax.set_ylabel("Wealth ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "cumulative_wealth.png"), dpi=200)
    plt.close(fig)

    # 2. Drawdowns
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    for p, c in policy_colors.items():
        w_arr = np.array(history[p]["wealth"])
        peak = np.maximum.accumulate(w_arr)
        dd = (w_arr - peak) / peak
        ax.plot(dd * 100, label=policy_labels[p], color=c, lw=1.5)
    ax.set_title("Phase 4: Holdout Drawdowns Across Policies (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Rebalancing Epoch (Weeks)")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "drawdowns.png"), dpi=200)
    plt.close(fig)

    # 3. Policy Weight Comparison (Average HHI and Turnover)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=200)
    p_names = list(policy_colors.keys())
    short_labels = ["M0", "M1", "M2", "M3", "M4"]
    hhis = [float(np.mean(history[p]["hhi"])) for p in p_names]
    turnovers = [float(np.mean(history[p]["turnovers"])) for p in p_names]
    colors = [policy_colors[p] for p in p_names]

    ax1.bar(short_labels, hhis, color=colors, alpha=0.85, edgecolor="black")
    ax1.set_title("Mean Concentration (HHI)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Herfindahl-Hirschman Index")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(short_labels, turnovers, color=colors, alpha=0.85, edgecolor="black")
    ax2.set_title("Mean Weekly Turnover", fontsize=10, fontweight="bold")
    ax2.set_ylabel("L1 Turnover Distance")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "policy_weight_comparison.png"), dpi=200)
    plt.close(fig)

    # 4. Selectivity Bad vs Good
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    x = np.arange(len(p_names))
    width = 0.35
    w_best = [selectivity[p]["mean_weight_best20"] * 100 for p in p_names]
    w_worst = [selectivity[p]["mean_weight_worst20"] * 100 for p in p_names]

    ax.bar(x - width/2, w_best, width, label="Best 20% Forecasts", color="#2ca02c", alpha=0.85, edgecolor="black")
    ax.bar(x + width/2, w_worst, width, label="Worst 20% Forecasts", color="#d62728", alpha=0.85, edgecolor="black")

    ax.set_ylabel("Pre-Allocated Capital Weight (%)")
    ax.set_title("Selectivity Analysis: Pre-Allocated Weight on Outcomes", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "selectivity_bad_vs_good.png"), dpi=200)
    plt.close(fig)

    # 5. High-Alpha / High-Uncertainty Bins
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    bin_names = [b["bin"].replace(" / ", "\n") for b in alpha_bins]
    m0_w = [b["mean_w_M0"] * 100 for b in alpha_bins]
    m1_w = [b["mean_w_M1"] * 100 for b in alpha_bins]
    m2_w = [b["mean_w_M2"] * 100 for b in alpha_bins]
    m3_w = [b["mean_w_M3"] * 100 for b in alpha_bins]
    m4_w = [b["mean_w_M4"] * 100 for b in alpha_bins]

    bx = np.arange(len(bin_names))
    b_width = 0.15
    ax.bar(bx - 2*b_width, m0_w, b_width, label="M0 Det", color="#333333", alpha=0.8)
    ax.bar(bx - b_width, m1_w, b_width, label="M1 Shrink", color="#E69F00", alpha=0.8)
    ax.bar(bx, m2_w, b_width, label="M2 CovInf", color="#0072B2", alpha=0.8)
    ax.bar(bx + b_width, m3_w, b_width, label="M3 DownAlpha", color="#009E73", alpha=0.8)
    ax.bar(bx + 2*b_width, m4_w, b_width, label="M4 DownCov", color="#D55E00", alpha=0.8)

    ax.set_title("Capital Allocation Across Alpha/Uncertainty Bins (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Pre-Allocated Weight (%)")
    ax.set_xticks(bx)
    ax.set_xticklabels(bin_names, fontsize=9)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "alpha_uncertainty_bins.png"), dpi=200)
    plt.close(fig)

    # 6. Downside Probability Calibration
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    quants = downside_cal["quintiles"]
    pred_p = [q["mean_p_down"] for q in quants]
    real_neg = [q["realized_neg_rate"] for q in quants]

    ax.plot(pred_p, real_neg, marker="o", color="#0072B2", lw=2, label="Empirical Downside Rate")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Perfect Calibration")
    ax.set_title("Downside Probability Calibration (Predicted vs Realized Negative)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Predicted Posterior Downside Probability P(mu < 0)")
    ax.set_ylabel("Realized Negative Return Rate P(r < 0)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "downside_probability_calibration.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main Execution Pipeline
# --------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PHASE 4: CONVERT EPISTEMIC UNCERTAINTY INTO DECISION-RELEVANT RISK")
    print("=" * 80)

    out_dir = "artifacts/phase4_decision_mapping"
    os.makedirs(out_dir, exist_ok=True)

    meta_path = "data/bayesian_lora/reconstructed_returns.csv"
    dev_path = "data/bayesian_lora/development/beta_0p003_seed_42_mc32_predictions.csv"
    seeds = [42, 123, 777]

    # 1. Hyperparameter Tuning strictly on Development Data (2022-2023)
    print("\n[Step 1] Tuning hyperparameters strictly on 2022-2023 development data...")
    dev_records = BayesianDatasetAdapter.load_from_csv(dev_path, meta_path)
    tuning_results = tune_hyperparameters_on_dev(dev_records)

    frozen_hyperparams = {
        "M1_shrinkage": tuning_results["M1_shrinkage"]["optimal_param"],
        "M2_cov_inflation": tuning_results["M2_cov_inflation"]["optimal_param"],
        "M3_downside_alpha": tuning_results["M3_downside_alpha"]["optimal_param"],
        "M4_downside_cov": tuning_results["M4_downside_cov"]["optimal_param"],
    }

    print("  Selected Frozen Hyperparameters (strictly from Dev):")
    for k, v in frozen_hyperparams.items():
        print(f"    {k}: {v}")

    with open(os.path.join(out_dir, "selected_hyperparameters.json"), "w") as f:
        json.dump(
            {
                "frozen_hyperparameters": frozen_hyperparams,
                "tuning_details": tuning_results,
            },
            f,
            indent=2,
        )

    # 2. Holdout Evaluation across all 3 seeds
    print("\n[Step 2] Evaluating M0-M4 on 2024-2025 Holdout across Seeds 42, 123, 777...")
    seed_sims = {}
    seed_selectivity = {}
    seed_alpha_bins = {}
    seed_downside_cal = {}
    seed_bootstraps = {}

    for s in seeds:
        h_path = f"data/bayesian_lora/holdout/beta_0p003_seed_{s}_mc32_predictions.csv"
        records = BayesianDatasetAdapter.load_from_csv(h_path, meta_path)

        # Run simulation
        sim_res = run_phase4_holdout_simulation(records, frozen_hyperparams)
        seed_sims[s] = sim_res

        # Selectivity
        sel = compute_selectivity_analysis(sim_res["asset_eval_records"])
        seed_selectivity[s] = sel

        # Alpha/Uncertainty Bins
        abins = compute_alpha_uncertainty_bins(sim_res["asset_eval_records"])
        seed_alpha_bins[s] = abins

        # Downside Calibration
        dcal = compute_downside_calibration(sim_res["asset_eval_records"])
        seed_downside_cal[s] = dcal

        # Block Bootstrap
        boot = compute_paired_block_bootstrap(sim_res["history"], seed=s)
        seed_bootstraps[s] = boot

        print(f"\n--- Seed {s} Holdout Results ---")
        for p, m in sim_res["metrics"].items():
            print(f"  {p:20s} | Ret: {m['annualized_return']*100:+6.2f}% | Vol: {m['annualized_volatility']*100:5.2f}% | Sharpe: {m['sharpe_ratio']:+5.3f} | Utility: {m['mean_utility']:+.5f} | MaxDD: {m['max_drawdown']*100:5.1f}% | Bad/Good: {sel[p]['bad_to_good_ratio']:.3f}")

    # 3. Save CSV Artifacts
    # metrics_by_seed.csv
    with open(os.path.join(out_dir, "metrics_by_seed.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "policy", "annualized_return", "annualized_volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "worst_5pct_return", "downside_deviation",
            "mean_turnover", "mean_hhi", "mean_utility", "bad_to_good_ratio"
        ])
        for s in seeds:
            for p, m in seed_sims[s]["metrics"].items():
                b_g = seed_selectivity[s][p]["bad_to_good_ratio"]
                writer.writerow([
                    s, p, f"{m['annualized_return']:.6f}", f"{m['annualized_volatility']:.6f}",
                    f"{m['sharpe_ratio']:.6f}", f"{m['sortino_ratio']:.6f}", f"{m['max_drawdown']:.6f}",
                    f"{m['worst_5pct_return']:.6f}", f"{m['downside_deviation']:.6f}",
                    f"{m['mean_turnover']:.6f}", f"{m['mean_hhi']:.6f}", f"{m['mean_utility']:.6f}",
                    f"{b_g:.6f}"
                ])

    # aggregate_metrics.csv
    policies = list(seed_sims[42]["metrics"].keys())
    with open(os.path.join(out_dir, "aggregate_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy", "ann_return_mean", "ann_return_std", "ann_vol_mean", "ann_vol_std",
            "sharpe_mean", "sharpe_std", "sortino_mean", "sortino_std", "max_dd_mean",
            "mean_utility_mean", "mean_utility_std", "bad_to_good_mean", "bad_to_good_std"
        ])
        for p in policies:
            rets = [seed_sims[s]["metrics"][p]["annualized_return"] for s in seeds]
            vols = [seed_sims[s]["metrics"][p]["annualized_volatility"] for s in seeds]
            sharpes = [seed_sims[s]["metrics"][p]["sharpe_ratio"] for s in seeds]
            sortinos = [seed_sims[s]["metrics"][p]["sortino_ratio"] for s in seeds]
            dds = [seed_sims[s]["metrics"][p]["max_drawdown"] for s in seeds]
            utils = [seed_sims[s]["metrics"][p]["mean_utility"] for s in seeds]
            b_gs = [seed_selectivity[s][p]["bad_to_good_ratio"] for s in seeds]

            writer.writerow([
                p,
                f"{np.mean(rets):.6f}", f"{np.std(rets):.6f}",
                f"{np.mean(vols):.6f}", f"{np.std(vols):.6f}",
                f"{np.mean(sharpes):.6f}", f"{np.std(sharpes):.6f}",
                f"{np.mean(sortinos):.6f}", f"{np.std(sortinos):.6f}",
                f"{np.mean(dds):.6f}",
                f"{np.mean(utils):.6f}", f"{np.std(utils):.6f}",
                f"{np.mean(b_gs):.6f}", f"{np.std(b_gs):.6f}",
            ])

    # selectivity_analysis.csv
    with open(os.path.join(out_dir, "selectivity_analysis.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "policy", "mean_weight_best20", "mean_weight_middle60", "mean_weight_worst20", "bad_to_good_ratio"])
        for s in seeds:
            for p in policies:
                sel = seed_selectivity[s][p]
                writer.writerow([
                    s, p, f"{sel['mean_weight_best20']:.6f}", f"{sel['mean_weight_middle60']:.6f}",
                    f"{sel['mean_weight_worst20']:.6f}", f"{sel['bad_to_good_ratio']:.6f}"
                ])

    # alpha_uncertainty_bins.csv
    with open(os.path.join(out_dir, "alpha_uncertainty_bins.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "bin", "count", "mean_realized_return", "mean_abs_error", "downside_freq_neg", "downside_freq_l1", "w_M0", "w_M1", "w_M2", "w_M3", "w_M4"])
        for s in seeds:
            for b in seed_alpha_bins[s]:
                writer.writerow([
                    s, b["bin"], b["count"], f"{b['mean_realized_return']:.6f}", f"{b['mean_abs_error']:.6f}",
                    f"{b['downside_freq_neg_ret']:.6f}", f"{b['downside_freq_label1']:.6f}",
                    f"{b['mean_w_M0']:.6f}", f"{b['mean_w_M1']:.6f}", f"{b['mean_w_M2']:.6f}",
                    f"{b['mean_w_M3']:.6f}", f"{b['mean_w_M4']:.6f}"
                ])

    # downside_calibration.csv
    with open(os.path.join(out_dir, "downside_calibration.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "auroc_neg_sigma", "auroc_neg_pdown", "auroc_label1_sigma", "auroc_label1_pdown", "brier_score"])
        for s in seeds:
            dc = seed_downside_cal[s]
            writer.writerow([
                s, f"{dc['auroc_neg_return_sigma']:.6f}", f"{dc['auroc_neg_return_pdown']:.6f}",
                f"{dc['auroc_label1_sigma']:.6f}", f"{dc['auroc_label1_pdown']:.6f}",
                f"{dc['brier_score_pdown_vs_neg']:.6f}"
            ])

    # bootstrap_intervals.csv
    with open(os.path.join(out_dir, "bootstrap_intervals.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "policy", "point_delta_u", "ci95_u_lower", "ci95_u_upper", "t_stat_u", "p_val_u", "point_delta_r", "ci95_r_lower", "ci95_r_upper"])
        for s in seeds:
            for p in ["M1_SymmetricShrink", "M2_CovInflation", "M3_DownsideAlpha", "M4_DownsideCov"]:
                b = seed_bootstraps[s][p]
                writer.writerow([
                    s, p, f"{b['point_delta_utility']:.6f}", f"{b['ci95_delta_utility'][0]:.6f}",
                    f"{b['ci95_delta_utility'][1]:.6f}", f"{b['t_stat_utility']:.3f}", f"{b['p_val_utility']:.4f}",
                    f"{b['point_delta_return']:.6f}", f"{b['ci95_delta_return'][0]:.6f}", f"{b['ci95_delta_return'][1]:.6f}"
                ])

    # metrics_summary.json
    summary_json = {
        "frozen_hyperparameters": frozen_hyperparams,
        "seeds": seeds,
        "seed_evaluations": {
            s: {
                "metrics": seed_sims[s]["metrics"],
                "selectivity": seed_selectivity[s],
                "downside_calibration": seed_downside_cal[s],
                "bootstrap": seed_bootstraps[s],
            }
            for s in seeds
        }
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    # 4. Generate Figures
    print("\n[Step 3] Generating publication figures...")
    generate_phase4_plots(seed_sims[42], seed_selectivity[42], seed_alpha_bins[42], seed_downside_cal[42], out_dir)
    print(f"  Figures and artifacts saved to {out_dir}/")
    print("=" * 80)
    print("PHASE 4 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
