"""
Phase 5: Portfolio-Level Risk Scaling with Bayesian Epistemic Uncertainty
========================================================================
Tests the hypothesis:
"Epistemic uncertainty may be poor at choosing WHICH assets to own (cross-sectional selection),
but useful for deciding HOW MUCH aggregate portfolio risk to take (portfolio-level exposure timing)."

Evaluates 4 exposure timing strategies on top of identical base risky weights:
- M0: Deterministic base risky portfolio (s_t = 1.0)
- M5: Epistemic exposure scaling (s_t = g(u_t))
- M6: Realized-volatility targeting (s_t = min(1.0, sigma_target / sigma_est))
- M7: Combined scaling (s_t = min(s_epi,t, s_vol,t))

Protocol:
1. Dev hyperparameter tuning (2022-2023):
   - Evaluate aggregate uncertainty metrics: mean, median, weighted, max, q75.
   - Tune lambda_epi in [0, 1, 3, 10, 30, 100, 300].
   - Tune sigma_target in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50].
   - Freeze optimal hyperparameters before holdout (zero data leakage).
2. Out-of-sample holdout simulation (2024-2025 across seeds 42, 123, 777):
   - Evaluate M0, M5, M6, M7 with zero cash return and exact cash turnover tracking.
3. Predictive Information Value of Aggregate Epistemic Uncertainty (Part J):
   - Correlation / AUROC / quintile breakdown of u_t vs next-period error, negative return, tail loss.
4. Incremental Value Over Realized Volatility (Part K):
   - Bivariate regression: future_loss ~ hist_vol + agg_uncertainty.
   - Correlation between u_t and hist_vol.
5. Macro-Regime & Exposure Timing Diagnostics (Part L & M):
   - 2x2 regime matrix (Low/High Uncertainty x Low/High Volatility).
   - Exposure attribution before top 10% best vs worst weeks.
6. Statistical Significance:
   - Paired circular block bootstrap (B = 4 weeks, 1,000 resamples) for Delta Sharpe and Delta Utility.
7. Artifacts & 8 Publication-Quality Figures:
   - Saved to artifacts/phase5_exposure_scaling/
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import os
from collections import defaultdict

os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from hypergraph_portfolio.bayesian_lora_adapter import (
    BayesianDatasetAdapter,
    BayesianPredictionRecord,
)
from hypergraph_portfolio.mean_variance import MeanVariancePortfolio
from hypergraph_portfolio.uncertainty import (
    apply_portfolio_exposure_scale,
    compute_aggregate_epistemic_uncertainty,
    compute_epistemic_exposure_scale,
    compute_volatility_targeting_scale,
)


def group_by_weekly_rebalance(records: List[BayesianPredictionRecord]) -> Dict[str, List[BayesianPredictionRecord]]:
    """Group records into weekly sequential rebalancing decision epochs."""
    buckets = defaultdict(list)
    for r in records:
        dt = datetime.datetime.fromisoformat(r.timestamp.replace(" ", "T"))
        w_key = f"{dt.year}-W{dt.isocalendar()[1]:02d}"
        buckets[w_key].append(r)
    return dict(sorted(buckets.items()))


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC using rank sums."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(np.sum(labels))
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = stats.rankdata(scores)
    u_stat = float(np.sum(ranks[labels == 1]) - n_pos * (n_pos + 1) / 2.0)
    return float(u_stat / (n_pos * n_neg))


# --------------------------------------------------------------------------
# Base Portfolio Builder & Rolling Volatility Estimator
# --------------------------------------------------------------------------

def build_base_portfolio_series(
    buckets: Dict[str, List[BayesianPredictionRecord]],
) -> Dict[str, Any]:
    """
    Construct the baseline mean-variance portfolio at each weekly decision epoch.
    Returns:
        epochs: List of epoch dicts containing:
            - period: str
            - assets: List[str]
            - mu_hat: np.ndarray
            - sigma_epi: np.ndarray
            - real_r: np.ndarray
            - Q_market: np.ndarray
            - w_base: np.ndarray (normalized risky weights)
            - base_return: float (w_base @ real_r)
    """
    epochs = []
    for w_key, basket in buckets.items():
        m = len(basket)
        assets = [r.asset for r in basket]
        mu_hat = np.array([r.mu_hat for r in basket])
        sigma_epi = np.array([r.epistemic_std for r in basket])
        real_r = np.array([r.realized_return for r in basket])

        Q_market = 0.04 * np.eye(m) + 0.01 * np.ones((m, m))

        try:
            w_base, _ = MeanVariancePortfolio(Q=Q_market, mu=mu_hat, budget=1.0).solve()
        except Exception:
            w_base = np.ones(m) / float(m)

        # Normalize strictly to sum to 1.0
        w_sum = np.sum(w_base)
        if w_sum > 0:
            w_base = w_base / w_sum
        else:
            w_base = np.ones(m) / float(m)

        base_r = float(w_base @ real_r)

        epochs.append({
            "period": w_key,
            "assets": assets,
            "mu_hat": mu_hat,
            "sigma_epi": sigma_epi,
            "real_r": real_r,
            "Q_market": Q_market,
            "w_base": w_base,
            "base_return": base_r,
        })
    return {"epochs": epochs}


def compute_rolling_historical_volatilities(
    returns: List[float],
    lookback: int = 12,
    ann_factor: float = 52.0,
) -> List[float]:
    """
    Compute annualized rolling historical volatility with ZERO lookahead.
    For step t, uses historical returns up to t-1:
        rolling_vol_t = std(returns[max(0, t - lookback) : t]) * sqrt(ann_factor)
    At t=0, falls back to an expanding window or default baseline.
    """
    vols = []
    for t in range(len(returns)):
        if t == 0:
            # Prior knowledge default or expanding from first available
            vols.append(0.30)  # Moderate default prior vol
        else:
            start_idx = max(0, t - lookback)
            hist = returns[start_idx:t]
            if len(hist) < 2:
                vols.append(0.30)
            else:
                v = float(np.std(hist, ddof=1) * np.sqrt(ann_factor))
                vols.append(max(v, 1e-4))
    return vols


# --------------------------------------------------------------------------
# Development Hyperparameter Tuning (Zero Holdout Leakage)
# --------------------------------------------------------------------------

def tune_phase5_hyperparameters_on_dev(
    dev_records: List[BayesianPredictionRecord],
    gamma: float = 1.0,
    cost_bps: float = 0.0010,
) -> Dict[str, Any]:
    """
    Tune Phase 5 hyperparameters strictly on 2022-2023 development set:
    1. Select best aggregate uncertainty metric: mean, median, weighted, max, q75.
    2. Tune lambda_epi in [0, 1, 3, 10, 30, 100, 300].
    3. Tune sigma_target in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50].
    """
    buckets = group_by_weekly_rebalance(dev_records)
    base_data = build_base_portfolio_series(buckets)
    epochs = base_data["epochs"]
    base_returns = [e["base_return"] for e in epochs]
    rolling_vols = compute_rolling_historical_volatilities(base_returns, lookback=12)

    agg_methods = ["mean", "median", "weighted", "max", "q75"]
    lambda_grid = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    target_vol_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    # 1. Sweep Aggregate Uncertainty Metric & Lambda_epi
    m5_results = {}
    best_m5_metric = "weighted"
    best_m5_lambda = 0.0
    best_m5_utility = -1e9

    for method in agg_methods:
        m5_results[method] = {}
        for lam in lambda_grid:
            utils = []
            prev_w_risky = None
            prev_w_cash = 0.0
            prev_assets = []

            for t, ep in enumerate(epochs):
                u_t = compute_aggregate_epistemic_uncertainty(
                    ep["sigma_epi"], weights=ep["w_base"], method=method
                )
                s_epi = compute_epistemic_exposure_scale(u_t, lambda_u=lam, mapping="rational")
                w_risky, w_cash = apply_portfolio_exposure_scale(ep["w_base"], scale=s_epi)

                # Turnover accounting
                if prev_w_risky is None:
                    turnover = 0.0
                else:
                    # Match dimensions by asset keys
                    all_assets = sorted(set(ep["assets"]) | set(prev_assets))
                    cur_map = {ep["assets"][i]: w_risky[i] for i in range(len(ep["assets"]))}
                    prev_map = {prev_assets[i]: prev_w_risky[i] for i in range(len(prev_assets))}
                    risky_to = sum(abs(cur_map.get(a, 0.0) - prev_map.get(a, 0.0)) for a in all_assets)
                    cash_to = abs(w_cash - prev_w_cash)
                    turnover = risky_to + cash_to

                # Return and Utility
                port_r = float(w_risky @ ep["real_r"])  # Cash return is 0.0
                net_r = port_r - cost_bps * turnover
                port_var = float(w_risky @ ep["Q_market"] @ w_risky)
                net_u = net_r - 0.5 * gamma * port_var - cost_bps * turnover
                utils.append(net_u)

                prev_w_risky = w_risky
                prev_w_cash = w_cash
                prev_assets = ep["assets"]

            mean_u = float(np.mean(utils))
            m5_results[method][lam] = mean_u
            if mean_u > best_m5_utility:
                best_m5_utility = mean_u
                best_m5_metric = method
                best_m5_lambda = lam

    # 2. Sweep Target Volatility for M6
    m6_results = {}
    best_m6_target_vol = 0.25
    best_m6_utility = -1e9

    for tv in target_vol_grid:
        utils = []
        prev_w_risky = None
        prev_w_cash = 0.0
        prev_assets = []

        for t, ep in enumerate(epochs):
            vol_t = rolling_vols[t]
            s_vol = compute_volatility_targeting_scale(vol_t, target_vol=tv, max_leverage=1.0)
            w_risky, w_cash = apply_portfolio_exposure_scale(ep["w_base"], scale=s_vol)

            if prev_w_risky is None:
                turnover = 0.0
            else:
                all_assets = sorted(set(ep["assets"]) | set(prev_assets))
                cur_map = {ep["assets"][i]: w_risky[i] for i in range(len(ep["assets"]))}
                prev_map = {prev_assets[i]: prev_w_risky[i] for i in range(len(prev_assets))}
                risky_to = sum(abs(cur_map.get(a, 0.0) - prev_map.get(a, 0.0)) for a in all_assets)
                cash_to = abs(w_cash - prev_w_cash)
                turnover = risky_to + cash_to

            port_r = float(w_risky @ ep["real_r"])
            net_r = port_r - cost_bps * turnover
            port_var = float(w_risky @ ep["Q_market"] @ w_risky)
            net_u = net_r - 0.5 * gamma * port_var - cost_bps * turnover
            utils.append(net_u)

            prev_w_risky = w_risky
            prev_w_cash = w_cash
            prev_assets = ep["assets"]

        mean_u = float(np.mean(utils))
        m6_results[tv] = mean_u
        if mean_u > best_m6_utility:
            best_m6_utility = mean_u
            best_m6_target_vol = tv

    return {
        "best_m5_metric": best_m5_metric,
        "best_m5_lambda": best_m5_lambda,
        "best_m5_utility": best_m5_utility,
        "m5_sweep": m5_results,
        "best_m6_target_vol": best_m6_target_vol,
        "best_m6_utility": best_m6_utility,
        "m6_sweep": m6_results,
    }


# --------------------------------------------------------------------------
# Out-of-Sample Holdout Simulation
# --------------------------------------------------------------------------

def run_phase5_holdout_simulation(
    holdout_records: List[BayesianPredictionRecord],
    agg_metric: str,
    lambda_epi: float,
    target_vol: float,
    gamma: float = 1.0,
    cost_bps: float = 0.0010,
) -> Dict[str, Any]:
    """
    Run backtest of M0, M5, M6, M7 on 2024-2025 holdout with frozen hyperparameters.
    """
    buckets = group_by_weekly_rebalance(holdout_records)
    base_data = build_base_portfolio_series(buckets)
    epochs = base_data["epochs"]

    base_returns = [e["base_return"] for e in epochs]
    rolling_vols = compute_rolling_historical_volatilities(base_returns, lookback=12)

    policies = ["M0_Deterministic", "M5_EpistemicScaling", "M6_VolTargeting", "M7_CombinedScaling"]

    history = {
        p: {
            "returns": [],
            "utilities": [],
            "turnovers": [],
            "exposures": [],
            "cash_weights": [],
            "wealth": [1.0],
            "risky_weights": [],
        }
        for p in policies
    }

    # Tracking metrics per epoch
    epoch_diagnostics = []
    prev_weights = {p: (None, 0.0) for p in policies}
    prev_assets = {p: [] for p in policies}

    for t, ep in enumerate(epochs):
        u_t = compute_aggregate_epistemic_uncertainty(
            ep["sigma_epi"], weights=ep["w_base"], method=agg_metric
        )
        vol_t = rolling_vols[t]

        # Calculate scales
        s_m0 = 1.0
        s_m5 = compute_epistemic_exposure_scale(u_t, lambda_u=lambda_epi, mapping="rational")
        s_m6 = compute_volatility_targeting_scale(vol_t, target_vol=target_vol, max_leverage=1.0)
        s_m7 = min(s_m5, s_m6)

        scales = {
            "M0_Deterministic": s_m0,
            "M5_EpistemicScaling": s_m5,
            "M6_VolTargeting": s_m6,
            "M7_CombinedScaling": s_m7,
        }

        # Calculate base error magnitude: mean absolute forecast error
        mean_abs_err = float(np.mean(np.abs(ep["real_r"] - ep["mu_hat"])))
        weighted_abs_err = float(np.sum(ep["w_base"] * np.abs(ep["real_r"] - ep["mu_hat"])))

        epoch_diagnostics.append({
            "period": ep["period"],
            "u_t": u_t,
            "vol_t": vol_t,
            "base_return": ep["base_return"],
            "mean_abs_err": mean_abs_err,
            "weighted_abs_err": weighted_abs_err,
            "s_m0": s_m0,
            "s_m5": s_m5,
            "s_m6": s_m6,
            "s_m7": s_m7,
        })

        for p in policies:
            s = scales[p]
            w_risky, w_cash = apply_portfolio_exposure_scale(ep["w_base"], scale=s)

            # Turnover
            prev_w_r, prev_w_c = prev_weights[p]
            if prev_w_r is None:
                turnover = 0.0
            else:
                p_assets = prev_assets[p]
                all_assets = sorted(set(ep["assets"]) | set(p_assets))
                cur_map = {ep["assets"][i]: w_risky[i] for i in range(len(ep["assets"]))}
                prev_map = {p_assets[i]: prev_w_r[i] for i in range(len(p_assets))}
                risky_to = sum(abs(cur_map.get(a, 0.0) - prev_map.get(a, 0.0)) for a in all_assets)
                cash_to = abs(w_cash - prev_w_c)
                turnover = risky_to + cash_to

            # Returns and utility
            port_r = float(w_risky @ ep["real_r"])
            tc = cost_bps * turnover
            net_r = port_r - tc
            port_var = float(w_risky @ ep["Q_market"] @ w_risky)
            u = net_r - 0.5 * gamma * port_var - cost_bps * turnover

            history[p]["returns"].append(net_r)
            history[p]["utilities"].append(u)
            history[p]["turnovers"].append(turnover)
            history[p]["exposures"].append(s)
            history[p]["cash_weights"].append(w_cash)
            history[p]["wealth"].append(history[p]["wealth"][-1] * (1.0 + net_r))
            history[p]["risky_weights"].append(w_risky.copy())

            prev_weights[p] = (w_risky, w_cash)
            prev_assets[p] = ep["assets"]

    # Summary performance metrics
    ann_factor = 52.0
    metrics = {}

    for p in policies:
        r_arr = np.array(history[p]["returns"])
        u_arr = np.array(history[p]["utilities"])
        w_arr = np.array(history[p]["wealth"])
        to_arr = np.array(history[p]["turnovers"])
        s_arr = np.array(history[p]["exposures"])

        ann_ret = float(np.mean(r_arr) * ann_factor)
        ann_vol = float(np.std(r_arr, ddof=1) * np.sqrt(ann_factor))
        sharpe = float(ann_ret / (ann_vol + 1e-12))

        # Peak and Max Drawdown
        peak = np.maximum.accumulate(w_arr)
        drawdowns = (w_arr - peak) / peak
        max_dd = float(np.min(drawdowns))

        # Downside Vol & Sortino
        neg_returns = r_arr[r_arr < 0]
        downside_std = float(np.std(neg_returns, ddof=1) * np.sqrt(ann_factor)) if len(neg_returns) > 1 else 1e-6
        sortino = float(ann_ret / (downside_std + 1e-12))

        # Calmar Ratio
        calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0

        metrics[p] = {
            "cumulative_return": float(w_arr[-1] - 1.0),
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "mean_turnover": float(np.mean(to_arr)),
            "mean_exposure": float(np.mean(s_arr)),
            "min_exposure": float(np.min(s_arr)),
            "max_exposure": float(np.max(s_arr)),
            "mean_utility": float(np.mean(u_arr)),
            "cumulative_utility": float(np.sum(u_arr)),
        }

    return {
        "metrics": metrics,
        "history": history,
        "epoch_diagnostics": epoch_diagnostics,
    }


# --------------------------------------------------------------------------
# Predictive Analysis (Part J) & Incremental Value (Part K)
# --------------------------------------------------------------------------

def compute_predictive_information_value(epoch_diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate whether aggregate epistemic uncertainty predicts future portfolio risk:
    1. Spearman corr with next-period forecast error magnitude
    2. Spearman corr with next-period base portfolio return
    3. AUROC for detecting negative return weeks (r < 0)
    4. AUROC for detecting tail loss weeks (r < -2%)
    5. Quintile breakdown of returns, errors, and loss rates by u_t
    """
    u_vals = np.array([d["u_t"] for d in epoch_diagnostics])
    v_vals = np.array([d["vol_t"] for d in epoch_diagnostics])
    r_vals = np.array([d["base_return"] for d in epoch_diagnostics])
    err_vals = np.array([d["mean_abs_err"] for d in epoch_diagnostics])
    w_err_vals = np.array([d["weighted_abs_err"] for d in epoch_diagnostics])

    neg_ret_labels = (r_vals < 0.0).astype(int)
    tail_loss_labels = (r_vals < -0.02).astype(int)

    # Correlations
    spearman_u_err, p_u_err = stats.spearmanr(u_vals, err_vals)
    spearman_u_werr, p_u_werr = stats.spearmanr(u_vals, w_err_vals)
    spearman_u_ret, p_u_ret = stats.spearmanr(u_vals, r_vals)
    spearman_u_vol, p_u_vol = stats.spearmanr(u_vals, v_vals)

    # AUROCs
    auroc_neg_ret = compute_auroc(u_vals, neg_ret_labels)
    auroc_tail_loss = compute_auroc(u_vals, tail_loss_labels)

    auroc_vol_neg_ret = compute_auroc(v_vals, neg_ret_labels)
    auroc_vol_tail_loss = compute_auroc(v_vals, tail_loss_labels)

    # Quintile analysis
    quintile_labels = np.zeros(len(u_vals), dtype=int)
    quantiles = np.percentile(u_vals, [20, 40, 60, 80])
    for i, u in enumerate(u_vals):
        if u <= quantiles[0]:
            quintile_labels[i] = 1
        elif u <= quantiles[1]:
            quintile_labels[i] = 2
        elif u <= quantiles[2]:
            quintile_labels[i] = 3
        elif u <= quantiles[3]:
            quintile_labels[i] = 4
        else:
            quintile_labels[i] = 5

    quintile_breakdown = []
    for q in range(1, 6):
        mask = quintile_labels == q
        q_rets = r_vals[mask]
        q_errs = err_vals[mask]
        quintile_breakdown.append({
            "quintile": q,
            "count": int(np.sum(mask)),
            "mean_uncertainty": float(np.mean(u_vals[mask])),
            "mean_base_return": float(np.mean(q_rets)) if len(q_rets) > 0 else 0.0,
            "mean_abs_error": float(np.mean(q_errs)) if len(q_errs) > 0 else 0.0,
            "pct_negative_return": float(np.mean(q_rets < 0.0)) if len(q_rets) > 0 else 0.0,
            "pct_tail_loss": float(np.mean(q_rets < -0.02)) if len(q_rets) > 0 else 0.0,
        })

    # Part K: Bivariate regression of future absolute error and loss on (vol_t, u_t)
    # loss_proxy = max(0, -r_t)
    loss_vals = np.maximum(0.0, -r_vals)

    # Standardize regressors
    u_std = (u_vals - np.mean(u_vals)) / (np.std(u_vals) + 1e-12)
    v_std = (v_vals - np.mean(v_vals)) / (np.std(v_vals) + 1e-12)
    X = np.column_stack([np.ones_like(u_std), v_std, u_std])

    # OLS for future loss
    beta_loss, _, _, _ = np.linalg.lstsq(X, loss_vals, rcond=None)
    # OLS for future forecast error
    beta_err, _, _, _ = np.linalg.lstsq(X, err_vals, rcond=None)

    return {
        "spearman_u_vs_abs_error": float(spearman_u_err),
        "p_val_u_vs_abs_error": float(p_u_err),
        "spearman_u_vs_weighted_error": float(spearman_u_werr),
        "spearman_u_vs_base_return": float(spearman_u_ret),
        "p_val_u_vs_base_return": float(p_u_ret),
        "spearman_u_vs_hist_vol": float(spearman_u_vol),
        "auroc_u_vs_neg_return": float(auroc_neg_ret),
        "auroc_u_vs_tail_loss": float(auroc_tail_loss),
        "auroc_vol_vs_neg_return": float(auroc_vol_neg_ret),
        "auroc_vol_vs_tail_loss": float(auroc_vol_tail_loss),
        "quintile_breakdown": quintile_breakdown,
        "regression_loss_coefs": {
            "intercept": float(beta_loss[0]),
            "beta_vol": float(beta_loss[1]),
            "beta_uncertainty": float(beta_loss[2]),
        },
        "regression_err_coefs": {
            "intercept": float(beta_err[0]),
            "beta_vol": float(beta_err[1]),
            "beta_uncertainty": float(beta_err[2]),
        },
    }


# --------------------------------------------------------------------------
# Macro-Regime & Exposure Attribution (Part L & M)
# --------------------------------------------------------------------------

def compute_regime_and_attribution(epoch_diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    1. 2x2 Regime Analysis: Low/High Uncertainty x Low/High Historical Volatility.
    2. Exposure Attribution: Mean exposure of M5, M6, M7 before best 10% vs worst 10% weeks.
    """
    u_vals = np.array([d["u_t"] for d in epoch_diagnostics])
    v_vals = np.array([d["vol_t"] for d in epoch_diagnostics])
    r_vals = np.array([d["base_return"] for d in epoch_diagnostics])
    s_m5 = np.array([d["s_m5"] for d in epoch_diagnostics])
    s_m6 = np.array([d["s_m6"] for d in epoch_diagnostics])
    s_m7 = np.array([d["s_m7"] for d in epoch_diagnostics])

    med_u = np.median(u_vals)
    med_v = np.median(v_vals)

    regimes = {
        "Low_U_Low_Vol": (u_vals <= med_u) & (v_vals <= med_v),
        "Low_U_High_Vol": (u_vals <= med_u) & (v_vals > med_v),
        "High_U_Low_Vol": (u_vals > med_u) & (v_vals <= med_v),
        "High_U_High_Vol": (u_vals > med_u) & (v_vals > med_v),
    }

    regime_results = {}
    for name, mask in regimes.items():
        sub_r = r_vals[mask]
        sub_s5 = s_m5[mask]
        sub_s6 = s_m6[mask]
        sub_s7 = s_m7[mask]
        regime_results[name] = {
            "count": int(np.sum(mask)),
            "mean_base_return": float(np.mean(sub_r)) if len(sub_r) > 0 else 0.0,
            "mean_exposure_M5": float(np.mean(sub_s5)) if len(sub_s5) > 0 else 1.0,
            "mean_exposure_M6": float(np.mean(sub_s6)) if len(sub_s6) > 0 else 1.0,
            "mean_exposure_M7": float(np.mean(sub_s7)) if len(sub_s7) > 0 else 1.0,
        }

    # Exposure attribution: best 10% vs worst 10% weeks
    n_weeks = len(r_vals)
    k_top = max(1, int(round(0.10 * n_weeks)))

    sort_idx = np.argsort(r_vals)
    worst_idx = sort_idx[:k_top]
    best_idx = sort_idx[-k_top:]

    attribution = {
        "worst_10pct": {
            "count": len(worst_idx),
            "mean_base_return": float(np.mean(r_vals[worst_idx])),
            "mean_exposure_M0": 1.0,
            "mean_exposure_M5": float(np.mean(s_m5[worst_idx])),
            "mean_exposure_M6": float(np.mean(s_m6[worst_idx])),
            "mean_exposure_M7": float(np.mean(s_m7[worst_idx])),
        },
        "best_10pct": {
            "count": len(best_idx),
            "mean_base_return": float(np.mean(r_vals[best_idx])),
            "mean_exposure_M0": 1.0,
            "mean_exposure_M5": float(np.mean(s_m5[best_idx])),
            "mean_exposure_M6": float(np.mean(s_m6[best_idx])),
            "mean_exposure_M7": float(np.mean(s_m7[best_idx])),
        },
        "middle_80pct": {
            "count": n_weeks - 2 * k_top,
            "mean_exposure_M0": 1.0,
            "mean_exposure_M5": float(np.mean(s_m5[sort_idx[k_top:-k_top]])),
            "mean_exposure_M6": float(np.mean(s_m6[sort_idx[k_top:-k_top]])),
            "mean_exposure_M7": float(np.mean(s_m7[sort_idx[k_top:-k_top]])),
        }
    }

    return {
        "regimes": regime_results,
        "attribution": attribution,
    }


# --------------------------------------------------------------------------
# Paired Circular Block Bootstrap (Part O)
# --------------------------------------------------------------------------

def compute_phase5_block_bootstrap(
    history: Dict[str, Dict[str, List[float]]],
    block_size: int = 4,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform paired circular block bootstrap over weekly epochs.
    Calculates 95% confidence intervals for:
        - Delta Sharpe vs M0
        - Delta Utility vs M0
        - Delta Return vs M0
    """
    rng = np.random.default_rng(seed)
    T = len(history["M0_Deterministic"]["returns"])
    policies = ["M5_EpistemicScaling", "M6_VolTargeting", "M7_CombinedScaling"]

    r_m0 = np.array(history["M0_Deterministic"]["returns"])
    u_m0 = np.array(history["M0_Deterministic"]["utilities"])
    ann_factor = 52.0

    results = {}

    for p in policies:
        r_p = np.array(history[p]["returns"])
        u_p = np.array(history[p]["utilities"])

        delta_u = u_p - u_m0
        delta_r = r_p - r_m0

        point_delta_u = float(np.mean(delta_u))
        point_delta_r = float(np.mean(delta_r))

        # Base point Sharpe ratio
        sharpe_m0 = (np.mean(r_m0) * ann_factor) / (np.std(r_m0, ddof=1) * np.sqrt(ann_factor) + 1e-12)
        sharpe_p = (np.mean(r_p) * ann_factor) / (np.std(r_p, ddof=1) * np.sqrt(ann_factor) + 1e-12)
        point_delta_sharpe = float(sharpe_p - sharpe_m0)

        boot_du = []
        boot_dr = []
        boot_ds = []

        n_blocks = int(math.ceil(T / block_size))
        for _ in range(n_bootstrap):
            start_indices = rng.integers(0, T, size=n_blocks)
            sampled_idx = []
            for s in start_indices:
                for b in range(block_size):
                    sampled_idx.append((s + b) % T)
            sampled_idx = np.array(sampled_idx[:T])

            # Sampled returns
            r_m0_b = r_m0[sampled_idx]
            r_p_b = r_p[sampled_idx]
            u_m0_b = u_m0[sampled_idx]
            u_p_b = u_p[sampled_idx]

            boot_du.append(float(np.mean(u_p_b - u_m0_b)))
            boot_dr.append(float(np.mean(r_p_b - r_m0_b)))

            s_m0_b = (np.mean(r_m0_b) * ann_factor) / (np.std(r_m0_b, ddof=1) * np.sqrt(ann_factor) + 1e-12)
            s_p_b = (np.mean(r_p_b) * ann_factor) / (np.std(r_p_b, ddof=1) * np.sqrt(ann_factor) + 1e-12)
            boot_ds.append(float(s_p_b - s_m0_b))

        ci_du = [float(np.percentile(boot_du, 2.5)), float(np.percentile(boot_du, 97.5))]
        ci_dr = [float(np.percentile(boot_dr, 2.5)), float(np.percentile(boot_dr, 97.5))]
        ci_ds = [float(np.percentile(boot_ds, 2.5)), float(np.percentile(boot_ds, 97.5))]

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
            "point_delta_sharpe": point_delta_sharpe,
            "ci95_delta_sharpe": ci_ds,
        }

    return results


# --------------------------------------------------------------------------
# Publication Figures (8 Required Figures)
# --------------------------------------------------------------------------

def generate_phase5_plots(
    sim_result: Dict[str, Any],
    pred_val: Dict[str, Any],
    regime_val: Dict[str, Any],
    tuning_res: Dict[str, Any],
    out_dir: str,
):
    """Generate the 8 publication-ready diagnostic charts."""
    history = sim_result["history"]
    diag = sim_result["epoch_diagnostics"]
    T = len(diag)
    step_indices = np.arange(T)

    colors = {
        "M0_Deterministic": "#333333",
        "M5_EpistemicScaling": "#0072B2",
        "M6_VolTargeting": "#E69F00",
        "M7_CombinedScaling": "#D55E00",
    }
    labels = {
        "M0_Deterministic": "M0: Deterministic (s=1.0)",
        "M5_EpistemicScaling": "M5: Epistemic Scaling",
        "M6_VolTargeting": "M6: Vol Targeting",
        "M7_CombinedScaling": "M7: Combined Scaling",
    }

    # 1. Cumulative Wealth Trajectories
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    for p in ["M0_Deterministic", "M5_EpistemicScaling", "M6_VolTargeting", "M7_CombinedScaling"]:
        ax.plot(history[p]["wealth"], label=labels[p], color=colors[p], lw=2.0)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Out-of-Sample Cumulative Wealth Trajectories (2024-2025 Holdout)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Weekly Decision Epoch", fontsize=11)
    ax.set_ylabel("Portfolio Wealth (Base = 1.0)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_cumulative_wealth.png"))
    plt.close()

    # 2. Exposure Trajectories Over Time
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.plot(step_indices, history["M5_EpistemicScaling"]["exposures"], label="M5: Epistemic Exposure", color=colors["M5_EpistemicScaling"], lw=1.8)
    ax.plot(step_indices, history["M6_VolTargeting"]["exposures"], label="M6: Vol Target Exposure", color=colors["M6_VolTargeting"], lw=1.8, linestyle="--")
    ax.plot(step_indices, history["M7_CombinedScaling"]["exposures"], label="M7: Combined Exposure", color=colors["M7_CombinedScaling"], lw=2.0)
    ax.axhline(1.0, color="black", linestyle=":", label="M0: Fixed Full Exposure")
    ax.set_title("Portfolio Risky Exposure Trajectory (s_t) Over Time", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Weekly Decision Epoch", fontsize=11)
    ax.set_ylabel("Risky Asset Exposure s_t in [0, 1]", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_exposure_trajectories.png"))
    plt.close()

    # 3. Drawdown Underwater Curves
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    for p in ["M0_Deterministic", "M5_EpistemicScaling", "M6_VolTargeting", "M7_CombinedScaling"]:
        w_arr = np.array(history[p]["wealth"])
        peak = np.maximum.accumulate(w_arr)
        dd = (w_arr - peak) / peak
        ax.plot(dd * 100, label=labels[p], color=colors[p], lw=1.8)
    ax.set_title("Drawdown Underwater Curves Across Policies", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Weekly Decision Epoch", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_drawdown_curves.png"))
    plt.close()

    # 4. Epistemic Uncertainty vs Realized Volatility Scatter
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=300)
    u_vals = [d["u_t"] for d in diag]
    v_vals = [d["vol_t"] for d in diag]
    r_vals = [d["base_return"] for d in diag]
    scatter = ax.scatter(v_vals, u_vals, c=r_vals, cmap="coolwarm_r", s=50, alpha=0.85, edgecolors="none")
    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label("Next-Period Base Return", fontsize=10)
    ax.set_title("Aggregate Epistemic Uncertainty vs Historical Volatility", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Rolling Historical Volatility (12-week annualized)", fontsize=11)
    ax.set_ylabel("Aggregate Epistemic Uncertainty u_t", fontsize=11)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_uncertainty_vs_volatility_scatter.png"))
    plt.close()

    # 5. Risk-Return / Sharpe Tradeoff Comparison
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=300)
    for p in ["M0_Deterministic", "M5_EpistemicScaling", "M6_VolTargeting", "M7_CombinedScaling"]:
        m = sim_result["metrics"][p]
        ax.scatter(m["annualized_volatility"] * 100, m["annualized_return"] * 100, color=colors[p], s=140, label=f"{labels[p]} (SR={m['sharpe_ratio']:.2f})", zorder=5)
    ax.set_title("Risk-Return Profile Across Policies (Holdout)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Annualized Volatility (%)", fontsize=11)
    ax.set_ylabel("Annualized Net Return (%)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_risk_return_tradeoff.png"))
    plt.close()

    # 6. Quintile Breakdown of Downside Rate by Aggregate Uncertainty
    q_data = pred_val["quintile_breakdown"]
    fig, ax1 = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    q_nums = [q["quintile"] for q in q_data]
    neg_rates = [q["pct_negative_return"] * 100 for q in q_data]
    mean_rets = [q["mean_base_return"] * 100 for q in q_data]

    w = 0.35
    ax1.bar(np.array(q_nums) - w/2, neg_rates, width=w, color="#D55E00", alpha=0.85, label="% Negative Return Weeks")
    ax1.set_xlabel("Aggregate Uncertainty Quintile (Q1=Lowest, Q5=Highest)", fontsize=11)
    ax1.set_ylabel("% Negative Return Weeks", color="#D55E00", fontsize=11)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(q_nums, mean_rets, color="#0072B2", marker="o", lw=2.2, label="Mean Base Return (%)")
    ax2.set_ylabel("Mean Base Weekly Return (%)", color="#0072B2", fontsize=11)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)

    plt.title("Downside Frequency and Mean Return Across Uncertainty Quintiles", fontsize=12, pad=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_quintile_downside_analysis.png"))
    plt.close()

    # 7. Macro-Regime Performance Heatmap (2x2 Matrix)
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=300)
    regimes = regime_val["regimes"]
    reg_matrix = np.array([
        [regimes["Low_U_Low_Vol"]["mean_base_return"] * 52 * 100, regimes["Low_U_High_Vol"]["mean_base_return"] * 52 * 100],
        [regimes["High_U_Low_Vol"]["mean_base_return"] * 52 * 100, regimes["High_U_High_Vol"]["mean_base_return"] * 52 * 100],
    ])
    cax = ax.matshow(reg_matrix, cmap="RdYlGn", alpha=0.85)
    for i in range(2):
        for j in range(2):
            val = reg_matrix[i, j]
            ax.text(j, i, f"{val:.1f}%\nann. return", ha="center", va="center", color="black", fontsize=11, fontweight="bold")
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low Hist Vol", "High Hist Vol"], fontsize=11)
    ax.set_yticklabels(["Low Uncertainty", "High Uncertainty"], fontsize=11)
    ax.set_title("Annualized Base Portfolio Return in 2x2 Macro Regimes", fontsize=12, pad=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_macro_regime_matrix.png"))
    plt.close()

    # 8. Exposure Attribution Before Tail Loss vs Peak Weeks
    attrib = regime_val["attribution"]
    fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=300)
    cats = ["Worst 10% Weeks\n(Downside Defense)", "Best 10% Weeks\n(Upside Capture)"]

    exp_worst = [attrib["worst_10pct"]["mean_exposure_M0"], attrib["worst_10pct"]["mean_exposure_M5"], attrib["worst_10pct"]["mean_exposure_M6"], attrib["worst_10pct"]["mean_exposure_M7"]]
    exp_best = [attrib["best_10pct"]["mean_exposure_M0"], attrib["best_10pct"]["mean_exposure_M5"], attrib["best_10pct"]["mean_exposure_M6"], attrib["best_10pct"]["mean_exposure_M7"]]

    x = np.arange(len(cats))
    width = 0.18

    ax.bar(x - 1.5*width, [exp_worst[0], exp_best[0]], width, label="M0 (Full)", color=colors["M0_Deterministic"])
    ax.bar(x - 0.5*width, [exp_worst[1], exp_best[1]], width, label="M5 (Epistemic)", color=colors["M5_EpistemicScaling"])
    ax.bar(x + 0.5*width, [exp_worst[2], exp_best[2]], width, label="M6 (Vol Target)", color=colors["M6_VolTargeting"])
    ax.bar(x + 1.5*width, [exp_worst[3], exp_best[3]], width, label="M7 (Combined)", color=colors["M7_CombinedScaling"])

    ax.set_ylabel("Average Portfolio Risky Exposure (s_t)", fontsize=11)
    ax.set_title("Exposure Attribution: Downside Defense vs Upside Capture", fontsize=12, pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig8_exposure_attribution.png"))
    plt.close()


# --------------------------------------------------------------------------
# Main Orchestration
# --------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PHASE 5: PORTFOLIO-LEVEL RISK SCALING WITH BAYESIAN EPISTEMIC UNCERTAINTY")
    print("=" * 80)

    out_dir = "artifacts/phase5_exposure_scaling"
    os.makedirs(out_dir, exist_ok=True)

    meta_path = "data/bayesian_lora/reconstructed_returns.csv"
    dev_path = "data/bayesian_lora/development/beta_0p003_seed_42_mc32_predictions.csv"
    seeds = [42, 123, 777]

    # 1. Dev Hyperparameter Tuning
    print("\n[Step 1] Tuning hyperparameters strictly on 2022-2023 development data...")
    dev_records = BayesianDatasetAdapter.load_from_csv(dev_path, meta_path)
    tuning_res = tune_phase5_hyperparameters_on_dev(dev_records)

    best_metric = tuning_res["best_m5_metric"]
    best_lambda = tuning_res["best_m5_lambda"]
    best_target_vol = tuning_res["best_m6_target_vol"]

    print(f"  Optimal Aggregate Uncertainty Metric: {best_metric}")
    print(f"  Optimal Epistemic Shrinkage Lambda:  {best_lambda}")
    print(f"  Optimal Volatility Target:           {best_target_vol:.2f}")

    with open(os.path.join(out_dir, "hyperparameter_tuning.json"), "w") as f:
        json.dump(tuning_res, f, indent=2)

    # 2. Holdout Evaluation across all 3 seeds
    print("\n[Step 2] Evaluating M0, M5, M6, M7 on 2024-2025 Holdout across Seeds 42, 123, 777...")
    seed_sims = {}
    seed_preds = {}
    seed_regimes = {}
    seed_bootstraps = {}

    for s in seeds:
        h_path = f"data/bayesian_lora/holdout/beta_0p003_seed_{s}_mc32_predictions.csv"
        records = BayesianDatasetAdapter.load_from_csv(h_path, meta_path)

        sim_res = run_phase5_holdout_simulation(
            records,
            agg_metric=best_metric,
            lambda_epi=best_lambda,
            target_vol=best_target_vol,
        )
        seed_sims[s] = sim_res

        pred_val = compute_predictive_information_value(sim_res["epoch_diagnostics"])
        seed_preds[s] = pred_val

        regime_val = compute_regime_and_attribution(sim_res["epoch_diagnostics"])
        seed_regimes[s] = regime_val

        boot = compute_phase5_block_bootstrap(sim_res["history"], block_size=4, n_bootstrap=1000, seed=s)
        seed_bootstraps[s] = boot

        print(f"  Seed {s} complete. Sharpe: M0={sim_res['metrics']['M0_Deterministic']['sharpe_ratio']:.2f}, "
              f"M5={sim_res['metrics']['M5_EpistemicScaling']['sharpe_ratio']:.2f}, "
              f"M6={sim_res['metrics']['M6_VolTargeting']['sharpe_ratio']:.2f}, "
              f"M7={sim_res['metrics']['M7_CombinedScaling']['sharpe_ratio']:.2f}")

    # 3. Export Machine-Readable CSV & JSON Artifacts
    print("\n[Step 3] Exporting machine-readable artifacts...")

    # exposure_simulation_results.csv
    policies = ["M0_Deterministic", "M5_EpistemicScaling", "M6_VolTargeting", "M7_CombinedScaling"]
    with open(os.path.join(out_dir, "exposure_simulation_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "policy", "cumulative_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown", "mean_turnover",
            "mean_exposure", "min_exposure", "max_exposure", "mean_utility", "cumulative_utility"
        ])
        for s in seeds:
            for p in policies:
                m = seed_sims[s]["metrics"][p]
                writer.writerow([
                    s, p, f"{m['cumulative_return']:.6f}", f"{m['annualized_return']:.6f}",
                    f"{m['annualized_volatility']:.6f}", f"{m['sharpe_ratio']:.4f}",
                    f"{m['sortino_ratio']:.4f}", f"{m['calmar_ratio']:.4f}",
                    f"{m['max_drawdown']:.6f}", f"{m['mean_turnover']:.6f}",
                    f"{m['mean_exposure']:.4f}", f"{m['min_exposure']:.4f}",
                    f"{m['max_exposure']:.4f}", f"{m['mean_utility']:.6f}",
                    f"{m['cumulative_utility']:.6f}"
                ])

    # aggregate_metrics.csv
    with open(os.path.join(out_dir, "aggregate_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy", "ann_return_mean", "ann_return_std", "ann_vol_mean", "ann_vol_std",
            "sharpe_mean", "sharpe_std", "sortino_mean", "sortino_std", "calmar_mean", "calmar_std",
            "max_dd_mean", "mean_exposure_mean", "mean_utility_mean", "mean_utility_std"
        ])
        for p in policies:
            rets = [seed_sims[s]["metrics"][p]["annualized_return"] for s in seeds]
            vols = [seed_sims[s]["metrics"][p]["annualized_volatility"] for s in seeds]
            sharpes = [seed_sims[s]["metrics"][p]["sharpe_ratio"] for s in seeds]
            sortinos = [seed_sims[s]["metrics"][p]["sortino_ratio"] for s in seeds]
            calmars = [seed_sims[s]["metrics"][p]["calmar_ratio"] for s in seeds]
            dds = [seed_sims[s]["metrics"][p]["max_drawdown"] for s in seeds]
            exps = [seed_sims[s]["metrics"][p]["mean_exposure"] for s in seeds]
            utils = [seed_sims[s]["metrics"][p]["mean_utility"] for s in seeds]

            writer.writerow([
                p,
                f"{np.mean(rets):.6f}", f"{np.std(rets):.6f}",
                f"{np.mean(vols):.6f}", f"{np.std(vols):.6f}",
                f"{np.mean(sharpes):.4f}", f"{np.std(sharpes):.4f}",
                f"{np.mean(sortinos):.4f}", f"{np.std(sortinos):.4f}",
                f"{np.mean(calmars):.4f}", f"{np.std(calmars):.4f}",
                f"{np.mean(dds):.6f}",
                f"{np.mean(exps):.4f}",
                f"{np.mean(utils):.6f}", f"{np.std(utils):.6f}"
            ])

    # predictive_information_value.csv
    with open(os.path.join(out_dir, "predictive_information_value.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "spearman_u_err", "p_val_err", "spearman_u_ret", "p_val_ret",
            "spearman_u_vol", "auroc_u_neg_ret", "auroc_u_tail_loss",
            "auroc_vol_neg_ret", "auroc_vol_tail_loss",
            "beta_vol_loss", "beta_uncertainty_loss"
        ])
        for s in seeds:
            pv = seed_preds[s]
            writer.writerow([
                s, f"{pv['spearman_u_vs_abs_error']:.4f}", f"{pv['p_val_u_vs_abs_error']:.4e}",
                f"{pv['spearman_u_vs_base_return']:.4f}", f"{pv['p_val_u_vs_base_return']:.4e}",
                f"{pv['spearman_u_vs_hist_vol']:.4f}", f"{pv['auroc_u_vs_neg_return']:.4f}",
                f"{pv['auroc_u_vs_tail_loss']:.4f}", f"{pv['auroc_vol_vs_neg_return']:.4f}",
                f"{pv['auroc_vol_vs_tail_loss']:.4f}",
                f"{pv['regression_loss_coefs']['beta_vol']:.6f}",
                f"{pv['regression_loss_coefs']['beta_uncertainty']:.6f}"
            ])

    # quintile_analysis.csv
    with open(os.path.join(out_dir, "quintile_analysis.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "quintile", "count", "mean_uncertainty", "mean_base_return", "mean_abs_error", "pct_negative_return", "pct_tail_loss"])
        for s in seeds:
            for q in seed_preds[s]["quintile_breakdown"]:
                writer.writerow([
                    s, q["quintile"], q["count"], f"{q['mean_uncertainty']:.6f}",
                    f"{q['mean_base_return']:.6f}", f"{q['mean_abs_error']:.6f}",
                    f"{q['pct_negative_return']:.4f}", f"{q['pct_tail_loss']:.4f}"
                ])

    # macro_regimes.csv
    with open(os.path.join(out_dir, "macro_regimes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "regime", "count", "mean_base_return", "mean_exposure_M5", "mean_exposure_M6", "mean_exposure_M7"])
        for s in seeds:
            for rname, rdata in seed_regimes[s]["regimes"].items():
                writer.writerow([
                    s, rname, rdata["count"], f"{rdata['mean_base_return']:.6f}",
                    f"{rdata['mean_exposure_M5']:.4f}", f"{rdata['mean_exposure_M6']:.4f}",
                    f"{rdata['mean_exposure_M7']:.4f}"
                ])

    # exposure_attribution.csv
    with open(os.path.join(out_dir, "exposure_attribution.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "segment", "count", "mean_base_return", "exp_M0", "exp_M5", "exp_M6", "exp_M7"])
        for s in seeds:
            att = seed_regimes[s]["attribution"]
            for seg in ["worst_10pct", "best_10pct", "middle_80pct"]:
                writer.writerow([
                    s, seg, att[seg]["count"],
                    f"{att[seg].get('mean_base_return', 0.0):.6f}",
                    f"{att[seg]['mean_exposure_M0']:.4f}",
                    f"{att[seg]['mean_exposure_M5']:.4f}",
                    f"{att[seg]['mean_exposure_M6']:.4f}",
                    f"{att[seg]['mean_exposure_M7']:.4f}"
                ])

    # bootstrap_intervals.csv
    with open(os.path.join(out_dir, "bootstrap_intervals.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "policy", "point_delta_u", "ci95_u_lower", "ci95_u_upper", "t_stat_u", "p_val_u",
            "point_delta_sharpe", "ci95_sharpe_lower", "ci95_sharpe_upper",
            "point_delta_r", "ci95_r_lower", "ci95_r_upper"
        ])
        for s in seeds:
            for p in ["M5_EpistemicScaling", "M6_VolTargeting", "M7_CombinedScaling"]:
                b = seed_bootstraps[s][p]
                writer.writerow([
                    s, p, f"{b['point_delta_utility']:.6f}", f"{b['ci95_delta_utility'][0]:.6f}",
                    f"{b['ci95_delta_utility'][1]:.6f}", f"{b['t_stat_utility']:.3f}", f"{b['p_val_utility']:.4f}",
                    f"{b['point_delta_sharpe']:.4f}", f"{b['ci95_delta_sharpe'][0]:.4f}", f"{b['ci95_delta_sharpe'][1]:.4f}",
                    f"{b['point_delta_return']:.6f}", f"{b['ci95_delta_return'][0]:.6f}", f"{b['ci95_delta_return'][1]:.6f}"
                ])

    # summary_results.json
    summary_json = {
        "frozen_hyperparameters": {
            "aggregate_uncertainty_metric": best_metric,
            "lambda_epi": best_lambda,
            "target_vol": best_target_vol,
        },
        "seeds": seeds,
        "seed_evaluations": {
            s: {
                "metrics": seed_sims[s]["metrics"],
                "predictive_value": seed_preds[s],
                "regimes_and_attribution": seed_regimes[s],
                "bootstrap": seed_bootstraps[s],
            }
            for s in seeds
        }
    }
    with open(os.path.join(out_dir, "summary_results.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    # 4. Generate Figures
    print("\n[Step 4] Generating publication figures...")
    generate_phase5_plots(seed_sims[42], seed_preds[42], seed_regimes[42], tuning_res, out_dir)
    print(f"  Figures and artifacts successfully saved to {out_dir}/")
    print("=" * 80)
    print("PHASE 5 EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
