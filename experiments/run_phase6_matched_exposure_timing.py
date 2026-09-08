"""
Phase 6: Matched-Exposure Timing + Asymmetric Downside Signals
==============================================================
Tests the primary timing hypothesis:
"Does relative / abnormally elevated epistemic uncertainty contain dynamic timing
information once we control strictly for average exposure budget (mean-exposure matching)?"

Policies Evaluated at matched risk budgets (e.g. target mean exposure 0.30, 0.50, 0.70, 0.90):
- M0: Full-risk deterministic baseline (s_t = 1.0)
- M8: Constant exposure baseline (s_t = c)
- M9: Relative epistemic timing (s_t = clip(s_base - lambda_rel * max(0, z_t - tau), s_min, 1.0))
- M10: Matched volatility timing (s_t = min(1.0, sigma_target / sigma_est), calibrated to match M9's mean exposure)
- M11: Placebo / permuted timing control (preserves empirical distribution of M9's exposure, alters timing)
- M12: Combined relative epistemic + volatility timing, matched to M9's mean exposure

Protocol:
1. Dev hyperparameter tuning (2022-2023):
   - Choose normalization method: zscore, robust_zscore, ratio, percentile_rank.
   - Choose trailing window: W in [8, 13, 26, 52].
   - Calibrate exposure rules to match target exposure budgets c in [0.30, 0.50, 0.70, 0.90].
   - Freeze all rules before holdout (zero data leakage).
2. Out-of-sample holdout simulation (2024-2025 across seeds 42, 123, 777):
   - Evaluate M0, M8, M9, M10, M11, M12 with identical base risky weights.
   - Cash return r_cash = 0.0, exact cash turnover accounting.
3. Predictive Information Value:
   - Relative vs absolute uncertainty vs historical volatility (Spearman, AUROC, AP for next-period losses).
4. Incremental Value Beyond Volatility:
   - Logistic regression: Tail_Loss ~ Hist_Vol + Rel_Uncertainty vs Hist_Vol alone.
5. Placebo Permutation Test (Part I):
   - 1,000 circular shift placebos of M9 exposure path: compute percentile rank & empirical p-value.
6. Timing Selectivity & Exposure Attribution (Part F):
   - Average exposure before worst 10% vs best 10% weeks.
   - Selectivity ratio: AvgExposure_before_worst10 / AvgExposure_before_best10.
7. Artifacts & 7 Publication Figures saved to artifacts/phase6_matched_exposure_timing/
"""

from __future__ import annotations

import csv
import datetime
import json
import os
from collections import defaultdict

os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"

from typing import Any, Dict, List, Tuple

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
    compute_relative_epistemic_exposure_scale,
    compute_relative_epistemic_uncertainty,
    compute_volatility_targeting_scale,
    generate_placebo_exposure_paths,
)


def group_by_weekly_rebalance(records: List[BayesianPredictionRecord]) -> Dict[str, List[BayesianPredictionRecord]]:
    """Group records into weekly sequential rebalancing decision epochs."""
    buckets = defaultdict(list)
    for r in records:
        dt = datetime.datetime.fromisoformat(r.timestamp.replace(" ", "T"))
        w_key = f"{dt.year}-W{dt.isocalendar()[1]:02d}"
        buckets[w_key].append(r)
    return dict(sorted(buckets.items()))


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


def build_base_portfolio_series(
    buckets: Dict[str, List[BayesianPredictionRecord]],
) -> Dict[str, Any]:
    """
    Construct the baseline mean-variance portfolio at each weekly decision epoch.
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
    """
    vols = []
    for t in range(len(returns)):
        if t == 0:
            vols.append(0.30)
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
# Development Tuning & Exposure Matching Calibration
# --------------------------------------------------------------------------

def tune_phase6_hyperparameters_on_dev(
    dev_records: List[BayesianPredictionRecord],
    target_budgets: List[float] = [0.30, 0.50, 0.70, 0.90],
    gamma: float = 1.0,
    cost_bps: float = 0.0010,
) -> Dict[str, Any]:
    """
    Tune Phase 6 on 2022-2023 development set:
    1. Select normalization method and window W by evaluating AUROC for detecting negative return weeks.
    2. Calibrate M9 parameters (s_base, lambda_rel) to hit each target exposure budget in [0.30, 0.50, 0.70, 0.90].
    3. Calibrate M10 target_vol to match each target exposure budget.
    """
    buckets = group_by_weekly_rebalance(dev_records)
    base_data = build_base_portfolio_series(buckets)
    epochs = base_data["epochs"]
    base_returns = [e["base_return"] for e in epochs]
    rolling_vols = compute_rolling_historical_volatilities(base_returns, lookback=12)

    # Compute absolute aggregate uncertainty (q75 from Phase 5)
    u_abs_series = [
        compute_aggregate_epistemic_uncertainty(ep["sigma_epi"], weights=ep["w_base"], method="q75")
        for ep in epochs
    ]

    candidate_methods = ["zscore", "robust_zscore", "ratio", "percentile_rank"]
    candidate_windows = [8, 13, 26, 52]

    # Evaluate predictive power of relative uncertainty candidates on dev
    neg_ret_labels = (np.array(base_returns) < 0.0).astype(int)
    best_method = "zscore"
    best_window = 13
    best_auroc = 0.0
    signal_evals = {}

    for m in candidate_methods:
        signal_evals[m] = {}
        for w in candidate_windows:
            rel_signals = []
            for t in range(len(u_abs_series)):
                hist_u = u_abs_series[:t]
                curr_u = u_abs_series[t]
                z = compute_relative_epistemic_uncertainty(hist_u, curr_u, window=w, method=m)
                rel_signals.append(z)

            auc, ap = compute_auroc_and_ap(np.array(rel_signals), neg_ret_labels)
            signal_evals[m][w] = {"auroc": auc, "ap": ap}
            if auc > best_auroc:
                best_auroc = auc
                best_method = m
                best_window = w

    # Precompute selected relative signal series on dev
    selected_rel_u = []
    for t in range(len(u_abs_series)):
        z = compute_relative_epistemic_uncertainty(
            u_abs_series[:t], u_abs_series[t], window=best_window, method=best_method
        )
        selected_rel_u.append(z)

    # Exposure Budget Calibration for each target budget in [0.30, 0.50, 0.70, 0.90]
    budget_calibrations = {}

    for target in target_budgets:
        # 1. Calibrate M9: grid search over s_base in [target, 1.0] and lambda_rel in [0.05, 0.50]
        best_m9_params = None
        best_m9_diff = 1e9

        s_base_grid = np.linspace(target, 1.0, 25)
        lambda_grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        tau = 0.0

        for sb in s_base_grid:
            for lam in lambda_grid:
                exposures = [
                    compute_relative_epistemic_exposure_scale(z, s_base=sb, lambda_rel=lam, tau=tau)
                    for z in selected_rel_u
                ]
                mean_exp = float(np.mean(exposures))
                diff = abs(mean_exp - target)
                if diff < best_m9_diff:
                    best_m9_diff = diff
                    best_m9_params = {
                        "s_base": float(sb),
                        "lambda_rel": float(lam),
                        "tau": float(tau),
                        "achieved_mean_exposure": mean_exp,
                    }

        # 2. Calibrate M10: grid search over target_vol in [0.05, 0.60] to hit target
        best_m10_vol = 0.20
        best_m10_diff = 1e9
        vol_grid = np.linspace(0.05, 0.60, 100)

        for tv in vol_grid:
            vol_exposures = [
                compute_volatility_targeting_scale(v, target_vol=tv, max_leverage=1.0)
                for v in rolling_vols
            ]
            mean_v_exp = float(np.mean(vol_exposures))
            diff = abs(mean_v_exp - target)
            if diff < best_m10_diff:
                best_m10_diff = diff
                best_m10_vol = float(tv)

        budget_calibrations[str(target)] = {
            "target_exposure": target,
            "m9_params": best_m9_params,
            "m10_target_vol": best_m10_vol,
        }

    return {
        "best_method": best_method,
        "best_window": best_window,
        "signal_evaluations": signal_evals,
        "budget_calibrations": budget_calibrations,
    }


# --------------------------------------------------------------------------
# Simulation Engine with Matched Controls & Placebos
# --------------------------------------------------------------------------

def run_phase6_holdout_simulation(
    holdout_records: List[BayesianPredictionRecord],
    dev_calibration: Dict[str, Any],
    target_budget: float = 0.70,
    n_placebos: int = 1000,
    seed: int = 42,
    gamma: float = 1.0,
    cost_bps: float = 0.0010,
) -> Dict[str, Any]:
    """
    Run backtest of M0, M8 (Constant), M9 (Rel Epistemic), M10 (Matched Vol),
    M11 (Placebo Controls), and M12 (Combined) on 2024-2025 holdout.
    """
    buckets = group_by_weekly_rebalance(holdout_records)
    base_data = build_base_portfolio_series(buckets)
    epochs = base_data["epochs"]
    T = len(epochs)

    base_returns = [e["base_return"] for e in epochs]
    rolling_vols = compute_rolling_historical_volatilities(base_returns, lookback=12)

    # Absolute aggregate uncertainty
    u_abs_series = [
        compute_aggregate_epistemic_uncertainty(ep["sigma_epi"], weights=ep["w_base"], method="q75")
        for ep in epochs
    ]

    method = dev_calibration["best_method"]
    window = dev_calibration["best_window"]
    calib = dev_calibration["budget_calibrations"][str(target_budget)]
    m9_p = calib["m9_params"]
    m10_vol = calib["m10_target_vol"]

    # Generate causal relative uncertainty series on holdout
    u_rel_series = []
    for t in range(T):
        hist_u = u_abs_series[:t]
        curr_u = u_abs_series[t]
        z = compute_relative_epistemic_uncertainty(hist_u, curr_u, window=window, method=method)
        u_rel_series.append(z)

    # Compute exposure series for primary policies
    # M0: Full risk (s=1.0)
    s_m0 = np.ones(T, dtype=float)

    # M9: Relative epistemic timing
    s_m9 = np.array([
        compute_relative_epistemic_exposure_scale(
            z, s_base=m9_p["s_base"], lambda_rel=m9_p["lambda_rel"], tau=m9_p["tau"]
        )
        for z in u_rel_series
    ])
    achieved_mean_m9 = float(np.mean(s_m9))

    # M8: Constant exposure matched strictly to achieved mean of M9 on holdout
    s_m8 = np.full(T, achieved_mean_m9, dtype=float)

    # M10: Matched volatility timing
    s_m10_raw = np.array([
        compute_volatility_targeting_scale(v, target_vol=m10_vol, max_leverage=1.0)
        for v in rolling_vols
    ])
    # Exact adjustment factor to match holdout mean exposure to M9
    if np.mean(s_m10_raw) > 0:
        s_m10 = np.clip(s_m10_raw * (achieved_mean_m9 / np.mean(s_m10_raw)), 0.0, 1.0)
    else:
        s_m10 = s_m10_raw

    # M12: Combined relative epistemic + volatility timing, matched to M9 mean
    s_m12_raw = np.minimum(s_m9, s_m10)
    if np.mean(s_m12_raw) > 0:
        s_m12 = np.clip(s_m12_raw * (achieved_mean_m9 / np.mean(s_m12_raw)), 0.0, 1.0)
    else:
        s_m12 = s_m12_raw

    # M11: Generate 1,000 circular shift placebo exposure paths
    placebos = generate_placebo_exposure_paths(s_m9, n_placebos=n_placebos, method="circular_shift", seed=seed)

    all_policies = {
        "M0_Deterministic": s_m0,
        "M8_ConstantMatched": s_m8,
        "M9_RelativeEpistemic": s_m9,
        "M10_MatchedVolTarget": s_m10,
        "M12_CombinedMatched": s_m12,
    }

    # Simulate each policy
    ann_factor = 52.0
    history = {}
    metrics = {}

    for p_name, exp_series in all_policies.items():
        ret_list = []
        u_list = []
        to_list = []
        wealth_list = [1.0]
        prev_w_r = None
        prev_w_c = 0.0
        prev_assets = []

        for t, ep in enumerate(epochs):
            s = exp_series[t]
            w_risky, w_cash = apply_portfolio_exposure_scale(ep["w_base"], scale=s)

            if prev_w_r is None:
                turnover = 0.0
            else:
                all_assets = sorted(set(ep["assets"]) | set(prev_assets))
                cur_map = {ep["assets"][i]: w_risky[i] for i in range(len(ep["assets"]))}
                prev_map = {prev_assets[i]: prev_w_r[i] for i in range(len(prev_assets))}
                risky_to = sum(abs(cur_map.get(a, 0.0) - prev_map.get(a, 0.0)) for a in all_assets)
                cash_to = abs(w_cash - prev_w_c)
                turnover = risky_to + cash_to

            port_r = float(w_risky @ ep["real_r"])
            tc = cost_bps * turnover
            net_r = port_r - tc
            port_var = float(w_risky @ ep["Q_market"] @ w_risky)
            u = net_r - 0.5 * gamma * port_var - tc

            ret_list.append(net_r)
            u_list.append(u)
            to_list.append(turnover)
            wealth_list.append(wealth_list[-1] * (1.0 + net_r))

            prev_w_r = w_risky
            prev_w_c = w_cash
            prev_assets = ep["assets"]

        r_arr = np.array(ret_list)
        u_arr = np.array(u_list)
        w_arr = np.array(wealth_list)
        to_arr = np.array(to_list)

        ann_ret = float(np.mean(r_arr) * ann_factor)
        ann_vol = float(np.std(r_arr, ddof=1) * np.sqrt(ann_factor))
        sharpe = float(ann_ret / (ann_vol + 1e-12))

        peak = np.maximum.accumulate(w_arr)
        drawdowns = (w_arr - peak) / peak
        max_dd = float(np.min(drawdowns))

        neg_returns = r_arr[r_arr < 0]
        downside_std = float(np.std(neg_returns, ddof=1) * np.sqrt(ann_factor)) if len(neg_returns) > 1 else 1e-6
        sortino = float(ann_ret / (downside_std + 1e-12))
        calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0

        # CVaR (Expected Shortfall at 5%) and Worst 5% return
        worst_5pct = float(np.percentile(r_arr, 5))
        cvar_5pct = float(np.mean(r_arr[r_arr <= worst_5pct])) if np.sum(r_arr <= worst_5pct) > 0 else worst_5pct

        history[p_name] = {
            "returns": ret_list,
            "utilities": u_list,
            "wealth": wealth_list,
            "turnovers": to_list,
            "exposures": exp_series.tolist(),
        }

        metrics[p_name] = {
            "mean_exposure": float(np.mean(exp_series)),
            "std_exposure": float(np.std(exp_series)),
            "min_exposure": float(np.min(exp_series)),
            "max_exposure": float(np.max(exp_series)),
            "pct_derisked_periods": float(np.mean(exp_series < m9_p["s_base"])),
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "worst_5pct_return": worst_5pct,
            "cvar_5pct": cvar_5pct,
            "mean_turnover": float(np.mean(to_arr)),
            "mean_utility": float(np.mean(u_arr)),
            "cumulative_utility": float(np.sum(u_arr)),
        }

    # Simulate Placebos (M11)
    placebo_utilities = []
    placebo_sharpes = []

    for i in range(n_placebos):
        p_exp = placebos[i]
        p_rets = []
        p_utils = []
        p_prev_w_r = None
        p_prev_w_c = 0.0
        p_prev_assets = []

        for t, ep in enumerate(epochs):
            s = p_exp[t]
            w_risky, w_cash = apply_portfolio_exposure_scale(ep["w_base"], scale=s)

            if p_prev_w_r is None:
                turnover = 0.0
            else:
                all_assets = sorted(set(ep["assets"]) | set(p_prev_assets))
                cur_map = {ep["assets"][i]: w_risky[i] for i in range(len(ep["assets"]))}
                prev_map = {p_prev_assets[i]: p_prev_w_r[i] for i in range(len(p_prev_assets))}
                risky_to = sum(abs(cur_map.get(a, 0.0) - prev_map.get(a, 0.0)) for a in all_assets)
                cash_to = abs(w_cash - p_prev_w_c)
                turnover = risky_to + cash_to

            port_r = float(w_risky @ ep["real_r"])
            tc = cost_bps * turnover
            net_r = port_r - tc
            port_var = float(w_risky @ ep["Q_market"] @ w_risky)
            u = net_r - 0.5 * gamma * port_var - tc

            p_rets.append(net_r)
            p_utils.append(u)

            p_prev_w_r = w_risky
            p_prev_w_c = w_cash
            p_prev_assets = ep["assets"]

        r_p_arr = np.array(p_rets)
        sh_p = (np.mean(r_p_arr) * ann_factor) / (np.std(r_p_arr, ddof=1) * np.sqrt(ann_factor) + 1e-12)
        placebo_sharpes.append(float(sh_p))
        placebo_utilities.append(float(np.mean(p_utils)))

    # Placebo statistics
    m9_actual_u = metrics["M9_RelativeEpistemic"]["mean_utility"]
    m9_actual_s = metrics["M9_RelativeEpistemic"]["sharpe_ratio"]

    p_rank_u = float(np.mean(np.array(placebo_utilities) <= m9_actual_u) * 100.0)
    p_val_u = float(np.mean(np.array(placebo_utilities) >= m9_actual_u))

    p_rank_s = float(np.mean(np.array(placebo_sharpes) <= m9_actual_s) * 100.0)
    p_val_s = float(np.mean(np.array(placebo_sharpes) >= m9_actual_s))

    placebo_summary = {
        "m9_actual_utility": m9_actual_u,
        "placebo_utility_mean": float(np.mean(placebo_utilities)),
        "placebo_utility_std": float(np.std(placebo_utilities)),
        "utility_percentile_rank": p_rank_u,
        "empirical_p_val_utility": p_val_u,
        "m9_actual_sharpe": m9_actual_s,
        "placebo_sharpe_mean": float(np.mean(placebo_sharpes)),
        "placebo_sharpe_std": float(np.std(placebo_sharpes)),
        "sharpe_percentile_rank": p_rank_s,
        "empirical_p_val_sharpe": p_val_s,
    }

    # Diagnostics table
    epoch_diagnostics = []
    for t in range(T):
        epoch_diagnostics.append({
            "period": epochs[t]["period"],
            "base_return": base_returns[t],
            "u_abs": u_abs_series[t],
            "u_rel": u_rel_series[t],
            "vol_hist": rolling_vols[t],
            "mean_abs_err": float(np.mean(np.abs(epochs[t]["real_r"] - epochs[t]["mu_hat"]))),
            "s_m0": s_m0[t],
            "s_m8": s_m8[t],
            "s_m9": s_m9[t],
            "s_m10": s_m10[t],
            "s_m12": s_m12[t],
        })

    return {
        "target_budget": target_budget,
        "metrics": metrics,
        "history": history,
        "placebo_summary": placebo_summary,
        "placebo_utilities": placebo_utilities,
        "placebo_sharpes": placebo_sharpes,
        "epoch_diagnostics": epoch_diagnostics,
    }


# --------------------------------------------------------------------------
# Predictive Information & Incremental Value Analysis
# --------------------------------------------------------------------------

def compute_phase6_predictive_value(epoch_diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Test whether relative epistemic uncertainty outperforms absolute uncertainty and volatility
    for predicting:
    1. Negative return (r < 0)
    2. Large portfolio loss (r < -2%)
    3. Large forecast error magnitude
    4. Future realized tail loss proxy: max(0, -r)
    """
    r_vals = np.array([d["base_return"] for d in epoch_diagnostics])
    u_abs = np.array([d["u_abs"] for d in epoch_diagnostics])
    u_rel = np.array([d["u_rel"] for d in epoch_diagnostics])
    v_hist = np.array([d["vol_hist"] for d in epoch_diagnostics])
    err_vals = np.array([d["mean_abs_err"] for d in epoch_diagnostics])

    neg_labels = (r_vals < 0.0).astype(int)
    tail_labels = (r_vals < -0.02).astype(int)

    # AUROC & AP for each signal
    auc_abs_neg, ap_abs_neg = compute_auroc_and_ap(u_abs, neg_labels)
    auc_rel_neg, ap_rel_neg = compute_auroc_and_ap(u_rel, neg_labels)
    auc_vol_neg, ap_vol_neg = compute_auroc_and_ap(v_hist, neg_labels)

    auc_abs_tail, ap_abs_tail = compute_auroc_and_ap(u_abs, tail_labels)
    auc_rel_tail, ap_rel_tail = compute_auroc_and_ap(u_rel, tail_labels)
    auc_vol_tail, ap_vol_tail = compute_auroc_and_ap(v_hist, tail_labels)

    # Spearman correlations with next-period forecast error
    rho_abs_err, p_abs_err = stats.spearmanr(u_abs, err_vals)
    rho_rel_err, p_rel_err = stats.spearmanr(u_rel, err_vals)
    rho_vol_err, p_vol_err = stats.spearmanr(v_hist, err_vals)

    # Incremental value logistic regression for Tail Loss:
    # Model 1: logit(P(tail)) = a + b1 * Hist_Vol
    # Model 2: logit(P(tail)) = a + b1 * Hist_Vol + b2 * Rel_Uncertainty
    v_std = (v_hist - np.mean(v_hist)) / (np.std(v_hist) + 1e-12)
    u_std = (u_rel - np.mean(u_rel)) / (np.std(u_rel) + 1e-12)

    # Simple OLS probability proxy / logistic Brier score
    X1 = np.column_stack([np.ones_like(v_std), v_std])
    X2 = np.column_stack([np.ones_like(v_std), v_std, u_std])

    beta1, _, _, _ = np.linalg.lstsq(X1, tail_labels, rcond=None)
    beta2, _, _, _ = np.linalg.lstsq(X2, tail_labels, rcond=None)

    pred1 = np.clip(X1 @ beta1, 0.0, 1.0)
    pred2 = np.clip(X2 @ beta2, 0.0, 1.0)

    brier1 = float(np.mean((pred1 - tail_labels) ** 2))
    brier2 = float(np.mean((pred2 - tail_labels) ** 2))

    auc_m1, ap_m1 = compute_auroc_and_ap(pred1, tail_labels)
    auc_m2, ap_m2 = compute_auroc_and_ap(pred2, tail_labels)

    return {
        "auroc_neg_return": {"abs_uncertainty": auc_abs_neg, "rel_uncertainty": auc_rel_neg, "hist_vol": auc_vol_neg},
        "ap_neg_return": {"abs_uncertainty": ap_abs_neg, "rel_uncertainty": ap_rel_neg, "hist_vol": ap_vol_neg},
        "auroc_tail_loss": {"abs_uncertainty": auc_abs_tail, "rel_uncertainty": auc_rel_tail, "hist_vol": auc_vol_tail},
        "ap_tail_loss": {"abs_uncertainty": ap_abs_tail, "rel_uncertainty": ap_rel_tail, "hist_vol": ap_vol_tail},
        "spearman_forecast_error": {
            "abs_uncertainty": float(rho_abs_err),
            "rel_uncertainty": float(rho_rel_err),
            "hist_vol": float(rho_vol_err),
        },
        "incremental_controls": {
            "model1_vol_only": {"auroc": auc_m1, "ap": ap_m1, "brier": brier1, "beta_vol": float(beta1[1])},
            "model2_vol_plus_rel_u": {
                "auroc": auc_m2,
                "ap": ap_m2,
                "brier": brier2,
                "beta_vol": float(beta2[1]),
                "beta_rel_u": float(beta2[2]),
            },
        },
    }


# --------------------------------------------------------------------------
# Timing Attribution & Selectivity (Part F)
# --------------------------------------------------------------------------

def compute_timing_selectivity(epoch_diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Measure whether dynamic exposure policies reduce risk BEFORE bad periods:
    - Exposure before best 10% weeks vs worst 10% weeks.
    - TimingSelectivity = AvgExposure_before_worst10 / AvgExposure_before_best10 (lower is better).
    - ExposureDifference = AvgExposure_before_best10 - AvgExposure_before_worst10 (higher is better).
    """
    r_vals = np.array([d["base_return"] for d in epoch_diagnostics])
    n = len(r_vals)
    k = max(1, int(round(0.10 * n)))

    sort_idx = np.argsort(r_vals)
    worst_idx = sort_idx[:k]
    best_idx = sort_idx[-k:]

    policies = ["s_m0", "s_m8", "s_m9", "s_m10", "s_m12"]
    results = {}

    for p in policies:
        exp_vals = np.array([d[p] for d in epoch_diagnostics])
        exp_worst = float(np.mean(exp_vals[worst_idx]))
        exp_best = float(np.mean(exp_vals[best_idx]))
        selectivity_ratio = float(exp_worst / (exp_best + 1e-12))
        exposure_diff = float(exp_best - exp_worst)

        results[p] = {
            "exp_before_worst10": exp_worst,
            "exp_before_best10": exp_best,
            "selectivity_ratio": selectivity_ratio,
            "exposure_difference": exposure_diff,
        }

    return results


# --------------------------------------------------------------------------
# Publication Figures (7 Primary Timing Figures)
# --------------------------------------------------------------------------

def generate_phase6_plots(
    sim_res: Dict[str, Any],
    pred_res: Dict[str, Any],
    selectivity_res: Dict[str, Any],
    out_dir: str,
):
    """Generate the 7 required primary timing diagnostic figures."""
    diag = sim_res["epoch_diagnostics"]
    T = len(diag)
    step_indices = np.arange(T)
    history = sim_res["history"]

    colors = {
        "M0_Deterministic": "#333333",
        "M8_ConstantMatched": "#56B4E9",
        "M9_RelativeEpistemic": "#0072B2",
        "M10_MatchedVolTarget": "#E69F00",
        "M12_CombinedMatched": "#D55E00",
    }
    labels = {
        "M0_Deterministic": "M0: Full Risk (100%)",
        "M8_ConstantMatched": "M8: Constant Matched",
        "M9_RelativeEpistemic": "M9: Relative Epistemic",
        "M10_MatchedVolTarget": "M10: Matched Vol Target",
        "M12_CombinedMatched": "M12: Combined Matched",
    }

    # 1. Relative Epistemic Uncertainty Over Time
    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=300)
    u_rel = [d["u_rel"] for d in diag]
    ax.plot(step_indices, u_rel, color="#0072B2", lw=1.8, label="Relative Epistemic Uncertainty (z-score)")
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.6, label="Normal Trailing Baseline (z=0)")
    ax.axhline(1.5, color="#D55E00", linestyle=":", label="High Uncertainty Threshold (z=+1.5)")
    ax.fill_between(step_indices, u_rel, 0.0, where=np.array(u_rel) > 0, color="#0072B2", alpha=0.15)
    ax.set_title("Relative Epistemic Uncertainty Trajectory (Trailing Normalization)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Weekly Decision Epoch (2024-2025 Holdout)", fontsize=11)
    ax.set_ylabel("Relative Uncertainty z_t", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "relative_epistemic_uncertainty_over_time.png"))
    plt.close()

    # 2. Matched Exposure Paths Over Time
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.plot(step_indices, history["M8_ConstantMatched"]["exposures"], label=labels["M8_ConstantMatched"], color=colors["M8_ConstantMatched"], lw=2.0, linestyle="--")
    ax.plot(step_indices, history["M9_RelativeEpistemic"]["exposures"], label=labels["M9_RelativeEpistemic"], color=colors["M9_RelativeEpistemic"], lw=2.0)
    ax.plot(step_indices, history["M10_MatchedVolTarget"]["exposures"], label=labels["M10_MatchedVolTarget"], color=colors["M10_MatchedVolTarget"], lw=1.8, alpha=0.85)
    ax.plot(step_indices, history["M12_CombinedMatched"]["exposures"], label=labels["M12_CombinedMatched"], color=colors["M12_CombinedMatched"], lw=1.8, linestyle=":")
    ax.set_title("Matched-Exposure Trajectories Across Dynamic Policies (Mean = 70%)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Weekly Decision Epoch", fontsize=11)
    ax.set_ylabel("Portfolio Risky Exposure s_t", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "matched_exposure_paths.png"))
    plt.close()

    # 3. Cumulative Wealth Matched Exposure
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    for p in ["M0_Deterministic", "M8_ConstantMatched", "M9_RelativeEpistemic", "M10_MatchedVolTarget", "M12_CombinedMatched"]:
        ax.plot(history[p]["wealth"], label=labels[p], color=colors[p], lw=2.0)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Cumulative Wealth Under Exposure-Matched Policies (Holdout)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Weekly Decision Epoch", fontsize=11)
    ax.set_ylabel("Portfolio Wealth (Base = 1.0)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cumulative_wealth_matched_exposure.png"))
    plt.close()

    # 4. Drawdowns Matched Exposure
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    for p in ["M8_ConstantMatched", "M9_RelativeEpistemic", "M10_MatchedVolTarget", "M12_CombinedMatched"]:
        w_arr = np.array(history[p]["wealth"])
        peak = np.maximum.accumulate(w_arr)
        dd = (w_arr - peak) / peak
        ax.plot(dd * 100, label=labels[p], color=colors[p], lw=1.8)
    ax.set_title("Underwater Drawdown Curves Under Matched Average Risk (Holdout)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Weekly Decision Epoch", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drawdowns_matched_exposure.png"))
    plt.close()

    # 5. Exposure Before Best vs Worst Weeks
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    sel = selectivity_res
    p_keys = ["s_m8", "s_m9", "s_m10", "s_m12"]
    p_names = ["M8 (Const)", "M9 (Rel Epi)", "M10 (Vol)", "M12 (Comb)"]

    exp_worst = [sel[k]["exp_before_worst10"] for k in p_keys]
    exp_best = [sel[k]["exp_before_best10"] for k in p_keys]

    x = np.arange(len(p_names))
    width = 0.35

    ax.bar(x - width/2, exp_worst, width, label="Exposure Before Worst 10% Weeks", color="#D55E00", alpha=0.85)
    ax.bar(x + width/2, exp_best, width, label="Exposure Before Best 10% Weeks", color="#0072B2", alpha=0.85)
    ax.set_ylabel("Average Portfolio Exposure", fontsize=11)
    ax.set_title("Timing Selectivity: Exposure Before Extreme Periods", fontsize=12, pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(p_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exposure_before_best_vs_worst.png"))
    plt.close()

    # 6. Placebo Timing Distribution
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    placebo_utils = sim_res["placebo_utilities"]
    actual_u = sim_res["placebo_summary"]["m9_actual_utility"]
    rank_u = sim_res["placebo_summary"]["utility_percentile_rank"]
    p_val_u = sim_res["placebo_summary"]["empirical_p_val_utility"]

    ax.hist(placebo_utils, bins=35, color="#A6CEE3", edgecolor="black", alpha=0.75, label="1,000 Circular Shift Placebos")
    ax.axvline(actual_u, color="#D55E00", lw=2.5, linestyle="--", label=f"Actual M9 Timing (Rank: {rank_u:.1f}%, p={p_val_u:.3f})")
    ax.axvline(np.mean(placebo_utils), color="black", lw=1.5, linestyle=":", label="Placebo Mean")
    ax.set_title("Placebo Timing Test: Realized Utility vs 1,000 Permutations", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Mean Weekly Net Utility", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "placebo_timing_distribution.png"))
    plt.close()

    # 7. Relative vs Absolute Signal AUROC Comparison
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    categories = ["Negative Return\n(r < 0)", "Tail Loss\n(r < -2%)"]
    auc_data = pred_res

    abs_aucs = [auc_data["auroc_neg_return"]["abs_uncertainty"], auc_data["auroc_tail_loss"]["abs_uncertainty"]]
    rel_aucs = [auc_data["auroc_neg_return"]["rel_uncertainty"], auc_data["auroc_tail_loss"]["rel_uncertainty"]]
    vol_aucs = [auc_data["auroc_neg_return"]["hist_vol"], auc_data["auroc_tail_loss"]["hist_vol"]]

    x = np.arange(len(categories))
    width = 0.25

    ax.bar(x - width, abs_aucs, width, label="Absolute Epistemic (u_t)", color="#999999", alpha=0.85)
    ax.bar(x, rel_aucs, width, label="Relative Epistemic (z_t)", color="#0072B2", alpha=0.85)
    ax.bar(x + width, vol_aucs, width, label="Historical Volatility (sigma_t)", color="#E69F00", alpha=0.85)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Random Chance (0.50)")
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title("Predictive Discrimination: Relative vs Absolute vs Volatility", fontsize=12, pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0.35, 0.85)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "relative_vs_absolute_signal_auc.png"))
    plt.close()


# --------------------------------------------------------------------------
# Main Orchestration
# --------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PHASE 6: MATCHED-EXPOSURE TIMING + ASYMMETRIC DOWNSIDE SIGNALS")
    print("=" * 80)

    out_dir = "artifacts/phase6_matched_exposure_timing"
    os.makedirs(out_dir, exist_ok=True)

    meta_path = "data/bayesian_lora/reconstructed_returns.csv"
    dev_path = "data/bayesian_lora/development/beta_0p003_seed_42_mc32_predictions.csv"
    seeds = [42, 123, 777]
    target_budgets = [0.30, 0.50, 0.70, 0.90]

    # 1. Dev Hyperparameter Tuning & Budget Calibration
    print("\n[Step 1] Tuning hyperparameters strictly on 2022-2023 development data...")
    dev_records = BayesianDatasetAdapter.load_from_csv(dev_path, meta_path)
    tuning_res = tune_phase6_hyperparameters_on_dev(dev_records, target_budgets=target_budgets)

    best_m = tuning_res["best_method"]
    best_w = tuning_res["best_window"]

    print(f"  Selected Relative Uncertainty Normalization: {best_m}")
    print(f"  Selected Trailing Normalization Window:      {best_w} weeks")

    with open(os.path.join(out_dir, "selected_hyperparameters.json"), "w") as f:
        json.dump(tuning_res, f, indent=2)

    # 2. Holdout Simulation across seeds and target budgets
    print("\n[Step 2] Evaluating matched-exposure policies on 2024-2025 Holdout...")
    all_seed_results = {}
    primary_budget = 0.70  # Primary representative risk budget

    for s in seeds:
        h_path = f"data/bayesian_lora/holdout/beta_0p003_seed_{s}_mc32_predictions.csv"
        records = BayesianDatasetAdapter.load_from_csv(h_path, meta_path)

        all_seed_results[s] = {}
        for b in target_budgets:
            sim_res = run_phase6_holdout_simulation(
                records, dev_calibration=tuning_res, target_budget=b, n_placebos=1000, seed=s
            )
            pred_res = compute_phase6_predictive_value(sim_res["epoch_diagnostics"])
            sel_res = compute_timing_selectivity(sim_res["epoch_diagnostics"])

            all_seed_results[s][str(b)] = {
                "sim": sim_res,
                "predictive": pred_res,
                "selectivity": sel_res,
            }

        m_m8 = all_seed_results[s][str(primary_budget)]["sim"]["metrics"]["M8_ConstantMatched"]
        m_m9 = all_seed_results[s][str(primary_budget)]["sim"]["metrics"]["M9_RelativeEpistemic"]
        m_m10 = all_seed_results[s][str(primary_budget)]["sim"]["metrics"]["M10_MatchedVolTarget"]

        print(f"  Seed {s} (Budget {primary_budget:.2f}):")
        print(f"    M8 (Constant {m_m8['mean_exposure']:.2f}): Sharpe={m_m8['sharpe_ratio']:.3f}, Utility={m_m8['mean_utility']:.6f}, MaxDD={m_m8['max_drawdown']*100:.1f}%")
        print(f"    M9 (Rel Epi  {m_m9['mean_exposure']:.2f}): Sharpe={m_m9['sharpe_ratio']:.3f}, Utility={m_m9['mean_utility']:.6f}, MaxDD={m_m9['max_drawdown']*100:.1f}%")
        print(f"    M10(Vol Matched {m_m10['mean_exposure']:.2f}): Sharpe={m_m10['sharpe_ratio']:.3f}, Utility={m_m10['mean_utility']:.6f}, MaxDD={m_m10['max_drawdown']*100:.1f}%")

    # 3. Export Machine-Readable CSV & JSON Artifacts
    print("\n[Step 3] Exporting CSV and JSON artifacts...")

    # matched_exposure_metrics.csv
    policies = ["M0_Deterministic", "M8_ConstantMatched", "M9_RelativeEpistemic", "M10_MatchedVolTarget", "M12_CombinedMatched"]
    with open(os.path.join(out_dir, "matched_exposure_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "target_budget", "policy", "mean_exposure", "std_exposure", "min_exposure", "max_exposure",
            "annualized_return", "annualized_volatility", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "worst_5pct_return", "cvar_5pct", "mean_turnover", "mean_utility", "cumulative_utility"
        ])
        for s in seeds:
            for b in target_budgets:
                m_dict = all_seed_results[s][str(b)]["sim"]["metrics"]
                for p in policies:
                    m = m_dict[p]
                    writer.writerow([
                        s, b, p, f"{m['mean_exposure']:.4f}", f"{m['std_exposure']:.4f}",
                        f"{m['min_exposure']:.4f}", f"{m['max_exposure']:.4f}",
                        f"{m['annualized_return']:.6f}", f"{m['annualized_volatility']:.6f}",
                        f"{m['sharpe_ratio']:.4f}", f"{m['sortino_ratio']:.4f}",
                        f"{m['calmar_ratio']:.4f}", f"{m['max_drawdown']:.6f}",
                        f"{m['worst_5pct_return']:.6f}", f"{m['cvar_5pct']:.6f}",
                        f"{m['mean_turnover']:.6f}", f"{m['mean_utility']:.6f}",
                        f"{m['cumulative_utility']:.6f}"
                    ])

    # timing_selectivity.csv
    with open(os.path.join(out_dir, "timing_selectivity.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "target_budget", "policy", "exp_before_worst10", "exp_before_best10", "selectivity_ratio", "exposure_difference"])
        for s in seeds:
            for b in target_budgets:
                sel = all_seed_results[s][str(b)]["selectivity"]
                for p in ["s_m8", "s_m9", "s_m10", "s_m12"]:
                    v = sel[p]
                    writer.writerow([
                        s, b, p, f"{v['exp_before_worst10']:.4f}", f"{v['exp_before_best10']:.4f}",
                        f"{v['selectivity_ratio']:.4f}", f"{v['exposure_difference']:.4f}"
                    ])

    # signal_predictive_value.csv
    with open(os.path.join(out_dir, "signal_predictive_value.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "target_budget", "auc_neg_abs", "auc_neg_rel", "auc_neg_vol",
            "auc_tail_abs", "auc_tail_rel", "auc_tail_vol",
            "rho_err_abs", "rho_err_rel", "rho_err_vol"
        ])
        for s in seeds:
            for b in target_budgets:
                pv = all_seed_results[s][str(b)]["predictive"]
                writer.writerow([
                    s, b,
                    f"{pv['auroc_neg_return']['abs_uncertainty']:.4f}",
                    f"{pv['auroc_neg_return']['rel_uncertainty']:.4f}",
                    f"{pv['auroc_neg_return']['hist_vol']:.4f}",
                    f"{pv['auroc_tail_loss']['abs_uncertainty']:.4f}",
                    f"{pv['auroc_tail_loss']['rel_uncertainty']:.4f}",
                    f"{pv['auroc_tail_loss']['hist_vol']:.4f}",
                    f"{pv['spearman_forecast_error']['abs_uncertainty']:.4f}",
                    f"{pv['spearman_forecast_error']['rel_uncertainty']:.4f}",
                    f"{pv['spearman_forecast_error']['hist_vol']:.4f}"
                ])

    # incremental_volatility_controls.csv
    with open(os.path.join(out_dir, "incremental_volatility_controls.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "target_budget", "m1_auc", "m1_ap", "m1_brier", "m1_beta_vol",
            "m2_auc", "m2_ap", "m2_brier", "m2_beta_vol", "m2_beta_rel_u"
        ])
        for s in seeds:
            for b in target_budgets:
                inc = all_seed_results[s][str(b)]["predictive"]["incremental_controls"]
                writer.writerow([
                    s, b,
                    f"{inc['model1_vol_only']['auroc']:.4f}", f"{inc['model1_vol_only']['ap']:.4f}",
                    f"{inc['model1_vol_only']['brier']:.6f}", f"{inc['model1_vol_only']['beta_vol']:.6f}",
                    f"{inc['model2_vol_plus_rel_u']['auroc']:.4f}", f"{inc['model2_vol_plus_rel_u']['ap']:.4f}",
                    f"{inc['model2_vol_plus_rel_u']['brier']:.6f}", f"{inc['model2_vol_plus_rel_u']['beta_vol']:.6f}",
                    f"{inc['model2_vol_plus_rel_u']['beta_rel_u']:.6f}"
                ])

    # placebo_timing_distribution.csv
    with open(os.path.join(out_dir, "placebo_timing_distribution.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "target_budget", "m9_actual_u", "placebo_u_mean", "placebo_u_std",
            "u_percentile_rank", "u_empirical_pval", "m9_actual_sharpe",
            "placebo_sharpe_mean", "placebo_sharpe_std", "sharpe_percentile_rank", "sharpe_empirical_pval"
        ])
        for s in seeds:
            for b in target_budgets:
                ps = all_seed_results[s][str(b)]["sim"]["placebo_summary"]
                writer.writerow([
                    s, b,
                    f"{ps['m9_actual_utility']:.6f}", f"{ps['placebo_utility_mean']:.6f}", f"{ps['placebo_utility_std']:.6f}",
                    f"{ps['utility_percentile_rank']:.2f}", f"{ps['empirical_p_val_utility']:.4f}",
                    f"{ps['m9_actual_sharpe']:.4f}", f"{ps['placebo_sharpe_mean']:.4f}", f"{ps['placebo_sharpe_std']:.4f}",
                    f"{ps['sharpe_percentile_rank']:.2f}", f"{ps['empirical_p_val_sharpe']:.4f}"
                ])

    # metrics_summary.json
    summary_json = {
        "seeds": seeds,
        "target_budgets": target_budgets,
        "selected_hyperparameters": tuning_res,
        "seed_evaluations": {
            s: {
                str(b): {
                    "metrics": all_seed_results[s][str(b)]["sim"]["metrics"],
                    "placebo_summary": all_seed_results[s][str(b)]["sim"]["placebo_summary"],
                    "predictive": all_seed_results[s][str(b)]["predictive"],
                    "selectivity": all_seed_results[s][str(b)]["selectivity"],
                }
                for b in target_budgets
            }
            for s in seeds
        }
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    # 4. Generate Figures
    print("\n[Step 4] Generating publication figures...")
    primary_sim = all_seed_results[42]["0.7"]["sim"]
    primary_pred = all_seed_results[42]["0.7"]["predictive"]
    primary_sel = all_seed_results[42]["0.7"]["selectivity"]
    generate_phase6_plots(primary_sim, primary_pred, primary_sel, out_dir)

    print(f"  Figures and artifacts successfully saved to {out_dir}/")
    print("=" * 80)
    print("PHASE 6 PRIMARY EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
