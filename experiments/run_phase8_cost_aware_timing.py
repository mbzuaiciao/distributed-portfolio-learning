"""
Phase 8: Cost-Aware Epistemic Timing with Persistence and Rebalancing Bands
===========================================================================
Core Objective:
Determine whether execution rules—hysteresis, persistence, rebalancing bands,
and lower-frequency exposure adjustments—can preserve the downside avoidance benefit
of relative epistemic timing while drastically slashing turnover and transaction costs.

Evaluated Policies:
- M8:  Constant Matched Exposure
- M9:  Raw Relative Epistemic Timing (high-turnover benchmark)
- M13: Hysteresis Timing (two-threshold state machine)
- M14: Persistence Timing (k-of-n confirmation before de-risking)
- M15: Rebalancing-Band Timing (inactivity band with partial adjustment)
- M16: Lower-Frequency Rebalancing (biweekly rebalancing with emergency override)
- M17: Best Simple Cost-Aware Combination (Hysteresis + Rebalancing Band)

Artifacts saved to: artifacts/phase8_cost_aware_timing/
"""

from __future__ import annotations

import csv
import datetime
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence

os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"

import matplotlib.pyplot as plt
import numpy as np

from hypergraph_portfolio.bayesian_lora_adapter import (
    BayesianDatasetAdapter,
    BayesianPredictionRecord,
)
from hypergraph_portfolio.mean_variance import MeanVariancePortfolio
from hypergraph_portfolio.uncertainty import (
    analyze_uncertainty_alarms,
    apply_portfolio_exposure_scale,
    compute_aggregate_epistemic_uncertainty,
    compute_band_exposure_path,
    compute_hysteresis_exposure_path,
    compute_lower_frequency_exposure_path,
    compute_persistent_exposure_path,
    compute_relative_epistemic_exposure_scale,
    compute_relative_epistemic_uncertainty,
    compute_turnover_attribution,
    decompose_timing_economic_benefit,
    generate_block_placebo_paths,
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


def build_base_portfolio_series(
    buckets: Dict[str, List[BayesianPredictionRecord]],
) -> Dict[str, Any]:
    """Construct centralized Markowitz base portfolio series."""
    epochs = []
    for w_key, basket in buckets.items():
        if not basket:
            continue

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

        w_sum = np.sum(w_base)
        if w_sum > 0:
            w_base = w_base / w_sum
        else:
            w_base = np.ones(m) / float(m)

        base_r = float(w_base @ real_r)

        epochs.append({
            "week": w_key,
            "assets": assets,
            "mu_hat": mu_hat,
            "sigma_epi": sigma_epi,
            "real_r": real_r,
            "Q_market": Q_market,
            "w_base": w_base,
            "base_return": base_r,
        })

    return {"epochs": epochs, "n_epochs": len(epochs)}


def simulate_portfolio_strategy(
    epochs: List[Dict[str, Any]],
    exposure_path: Sequence[float],
    gamma: float = 1.0,
    cost_bps: float = 0.0010,
    r_cash_annual: float = 0.0,
    ann_factor: float = 52.0,
) -> Dict[str, Any]:
    """Simulate a given exposure trajectory against epoch data."""
    T = len(epochs)
    r_cash_weekly = r_cash_annual / ann_factor

    ret_list = []
    u_list = []
    to_list = []
    wealth_list = [1.0]

    prev_w_r = None
    prev_w_c = 0.0
    prev_assets = []

    for t in range(T):
        ep = epochs[t]
        s = float(exposure_path[t])
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

        port_r = float(w_risky @ ep["real_r"]) + w_cash * r_cash_weekly
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
    ann_vol = float(np.std(r_arr, ddof=1) * np.sqrt(ann_factor)) if len(r_arr) > 1 else 1e-6
    sharpe = float(ann_ret / (ann_vol + 1e-12))

    peak = np.maximum.accumulate(w_arr)
    drawdowns = (w_arr - peak) / peak
    max_dd = float(np.min(drawdowns))

    neg_returns = r_arr[r_arr < 0]
    downside_std = float(np.std(neg_returns, ddof=1) * np.sqrt(ann_factor)) if len(neg_returns) > 1 else 1e-6
    sortino = float(ann_ret / (downside_std + 1e-12))
    calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0

    return {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "mean_turnover": float(np.mean(to_arr)),
        "mean_utility": float(np.mean(u_arr)),
        "cumulative_utility": float(np.sum(u_arr)),
        "mean_exposure": float(np.mean(exposure_path)),
        "total_tc_drag": float(cost_bps * np.sum(to_arr)),
        "returns": ret_list,
        "utilities": u_list,
        "wealth": wealth_list,
        "drawdowns": drawdowns.tolist(),
    }


def main():
    print("=" * 80)
    print("PHASE 8: COST-AWARE EPISTEMIC TIMING & EXECUTION RULES")
    print("=" * 80)

    out_dir = "artifacts/phase8_cost_aware_timing"
    os.makedirs(out_dir, exist_ok=True)

    meta_path = "data/bayesian_lora/reconstructed_returns.csv"
    seeds = [42, 123, 777]

    # Load holdout and dev records
    holdout_records = {
        s: BayesianDatasetAdapter.load_from_csv(
            f"data/bayesian_lora/holdout/beta_0p003_seed_{s}_mc32_predictions.csv", meta_path
        )
        for s in seeds
    }
    dev_records = {
        s: BayesianDatasetAdapter.load_from_csv(
            f"data/bayesian_lora/development/beta_0p003_seed_{s}_mc32_predictions.csv", meta_path
        )
        for s in seeds
    }

    # ----------------------------------------------------------------------
    # STEP 1: Parameter Tuning on Development Data strictly
    # ----------------------------------------------------------------------
    print("\n[Step 1] Tuning Execution Rules on Development Data strictly...")
    dev_b_42 = group_by_weekly_rebalance(dev_records[42])
    dev_base_42 = build_base_portfolio_series(dev_b_42)
    dev_epochs_42 = dev_base_42["epochs"]

    dev_u_abs = [compute_aggregate_epistemic_uncertainty(ep["sigma_epi"], weights=ep["w_base"], method="q75") for ep in dev_epochs_42]
    dev_z = [compute_relative_epistemic_uncertainty(dev_u_abs[:t], dev_u_abs[t], window=8, method="robust_zscore") for t in range(len(dev_epochs_42))]

    dev_s_m9 = [compute_relative_epistemic_exposure_scale(z, s_base=0.85, lambda_rel=0.20, tau=0.0) for z in dev_z]
    dev_mean_exp = float(np.mean(dev_s_m9))

    # Grid search M13 Hysteresis on Dev Net Utility
    best_hys_params = (1.0, 0.2)
    best_hys_u = -999.0
    for tau_ent in [0.8, 1.0, 1.2, 1.5]:
        for tau_ex in [0.0, 0.2, 0.4, 0.6]:
            if tau_ex >= tau_ent:
                continue
            path = compute_hysteresis_exposure_path(dev_z, tau_enter=tau_ent, tau_exit=tau_ex, s_normal=0.85, s_reduced=0.45)
            # Scale to match mean exposure
            if np.mean(path) > 0:
                path = np.clip(path * (dev_mean_exp / np.mean(path)), 0.0, 1.0)
            res = simulate_portfolio_strategy(dev_epochs_42, path, cost_bps=0.0010)
            if res["mean_utility"] > best_hys_u:
                best_hys_u = res["mean_utility"]
                best_hys_params = (tau_ent, tau_ex)

    # Grid search M14 Persistence on Dev Net Utility
    best_pers_params = (2, 2, 1.0)
    best_pers_u = -999.0
    for (k, n) in [(2, 2), (2, 3), (3, 3)]:
        for tau_val in [0.8, 1.0, 1.2]:
            path = compute_persistent_exposure_path(dev_z, tau=tau_val, k=k, n=n, s_normal=0.85, s_reduced=0.45)
            if np.mean(path) > 0:
                path = np.clip(path * (dev_mean_exp / np.mean(path)), 0.0, 1.0)
            res = simulate_portfolio_strategy(dev_epochs_42, path, cost_bps=0.0010)
            if res["mean_utility"] > best_pers_u:
                best_pers_u = res["mean_utility"]
                best_pers_params = (k, n, tau_val)

    # Grid search M15 Rebalancing Band on Dev Net Utility
    best_band_params = (0.10, 1.0)
    best_band_u = -999.0
    for delta in [0.05, 0.10, 0.15, 0.20]:
        for eta in [0.5, 1.0]:
            path = compute_band_exposure_path(dev_s_m9, delta=delta, eta=eta)
            if np.mean(path) > 0:
                path = np.clip(path * (dev_mean_exp / np.mean(path)), 0.0, 1.0)
            res = simulate_portfolio_strategy(dev_epochs_42, path, cost_bps=0.0010)
            if res["mean_utility"] > best_band_u:
                best_band_u = res["mean_utility"]
                best_band_params = (delta, eta)

    # Combination (M17): Hysteresis + Rebalancing Band
    # Apply optimal band to optimal hysteresis
    raw_hys = compute_hysteresis_exposure_path(dev_z, tau_enter=best_hys_params[0], tau_exit=best_hys_params[1], s_normal=0.85, s_reduced=0.45)
    _ = compute_band_exposure_path(raw_hys, delta=best_band_params[0], eta=best_band_params[1])

    selected_params = {
        "hysteresis": {"tau_enter": best_hys_params[0], "tau_exit": best_hys_params[1], "s_normal": 0.85, "s_reduced": 0.45},
        "persistence": {"k": best_pers_params[0], "n": best_pers_params[1], "tau": best_pers_params[2]},
        "band": {"delta": best_band_params[0], "eta": best_band_params[1]},
        "frequency": {"interval_weeks": 2, "emergency_tau": 2.0},
        "combination_m17": {"type": "hysteresis_plus_band", "hys_tau_enter": best_hys_params[0], "hys_tau_exit": best_hys_params[1], "band_delta": best_band_params[0]},
    }
    with open(os.path.join(out_dir, "selected_parameters.json"), "w") as f:
        json.dump(selected_params, f, indent=2)
    print(f"  Selected parameters on Dev: Hysteresis {best_hys_params}, Persistence {best_pers_params}, Band {best_band_params}")

    # ----------------------------------------------------------------------
    # STEP 2: Evaluate All Policies on Holdout Across Seeds (Primary Target 0.70)
    # ----------------------------------------------------------------------
    print("\n[Step 2] Evaluating M8 through M17 on 2024-2025 Frozen Holdout (target 0.70)...")
    target_budget = 0.70

    # We evaluate for each seed and average
    policy_names = ["M8_Constant", "M9_RawTiming", "M13_Hysteresis", "M14_Persistence", "M15_Band", "M16_Biweekly", "M17_Combined"]
    holdout_sim_results = {p: [] for p in policy_names}
    holdout_exposures = {s: {} for s in seeds}
    epochs_by_seed = {}

    for s in seeds:
        h_buckets = group_by_weekly_rebalance(holdout_records[s])
        base_d = build_base_portfolio_series(h_buckets)
        epochs = base_d["epochs"]
        epochs_by_seed[s] = epochs

        # Aggregate epistemic uncertainty & relative z-score
        u_abs = [compute_aggregate_epistemic_uncertainty(ep["sigma_epi"], weights=ep["w_base"], method="q75") for ep in epochs]
        z_scores = [compute_relative_epistemic_uncertainty(u_abs[:t], u_abs[t], window=8, method="robust_zscore") for t in range(len(epochs))]

        # M9 Raw
        s_m9 = np.array([compute_relative_epistemic_exposure_scale(z, s_base=0.85, lambda_rel=0.20, tau=0.0) for z in z_scores])
        # Scale to match target_budget
        scale_factor = target_budget / np.mean(s_m9)
        s_m9 = np.clip(s_m9 * scale_factor, 0.0, 1.0)
        mean_exp = float(np.mean(s_m9))

        # M8 Constant
        s_m8 = np.array([mean_exp] * len(epochs))

        # M13 Hysteresis
        s_m13 = compute_hysteresis_exposure_path(z_scores, tau_enter=best_hys_params[0], tau_exit=best_hys_params[1], s_normal=0.85, s_reduced=0.45)
        s_m13 = np.clip(s_m13 * (mean_exp / np.mean(s_m13)), 0.0, 1.0)

        # M14 Persistence
        s_m14 = compute_persistent_exposure_path(z_scores, tau=best_pers_params[2], k=best_pers_params[0], n=best_pers_params[1], s_normal=0.85, s_reduced=0.45)
        s_m14 = np.clip(s_m14 * (mean_exp / np.mean(s_m14)), 0.0, 1.0)

        # M15 Rebalancing Band
        s_m15 = compute_band_exposure_path(s_m9, delta=best_band_params[0], eta=best_band_params[1])
        s_m15 = np.clip(s_m15 * (mean_exp / np.mean(s_m15)), 0.0, 1.0)

        # M16 Lower Frequency (Biweekly with emergency threshold)
        s_m16 = compute_lower_frequency_exposure_path(s_m9, frequency=2, severe_signal=z_scores, tau_severe=2.0, s_emergency=0.35)
        s_m16 = np.clip(s_m16 * (mean_exp / np.mean(s_m16)), 0.0, 1.0)

        # M17 Best Combination (Hysteresis + Rebalancing Band)
        s_m17 = compute_band_exposure_path(s_m13, delta=best_band_params[0], eta=best_band_params[1])
        s_m17 = np.clip(s_m17 * (mean_exp / np.mean(s_m17)), 0.0, 1.0)

        holdout_exposures[s] = {
            "M8_Constant": s_m8,
            "M9_RawTiming": s_m9,
            "M13_Hysteresis": s_m13,
            "M14_Persistence": s_m14,
            "M15_Band": s_m15,
            "M16_Biweekly": s_m16,
            "M17_Combined": s_m17,
        }

        for pname in policy_names:
            sim_res = simulate_portfolio_strategy(epochs, holdout_exposures[s][pname], cost_bps=0.0010)
            holdout_sim_results[pname].append(sim_res)

    # Write policy_metrics.csv
    with open(os.path.join(out_dir, "policy_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "annual_return", "annual_vol", "sharpe_ratio", "sortino_ratio", "max_drawdown", "mean_turnover", "annual_turnover", "mean_utility", "cost_drag_bps"])
        for pname in policy_names:
            arr_res = holdout_sim_results[pname]
            ann_ret = np.mean([r["annualized_return"] for r in arr_res])
            ann_vol = np.mean([r["annualized_volatility"] for r in arr_res])
            sh = np.mean([r["sharpe_ratio"] for r in arr_res])
            so = np.mean([r["sortino_ratio"] for r in arr_res])
            dd = np.mean([r["max_drawdown"] for r in arr_res])
            to = np.mean([r["mean_turnover"] for r in arr_res])
            ann_to = to * 52.0
            u = np.mean([r["mean_utility"] for r in arr_res])
            drag = np.mean([r["total_tc_drag"] for r in arr_res]) * 10000.0
            writer.writerow([pname, f"{ann_ret*100:.2f}%", f"{ann_vol*100:.2f}%", f"{sh:.3f}", f"{so:.3f}", f"{dd*100:.2f}%", f"{to:.4f}", f"{ann_to:.2f}", f"{u:.6f}", f"{drag:.1f}"])

    # ----------------------------------------------------------------------
    # STEP 3: Turnover Attribution & Savings vs M9
    # ----------------------------------------------------------------------
    print("\n[Step 3] Computing Turnover Attribution & Savings vs M9...")
    with open(os.path.join(out_dir, "turnover_attribution.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "annual_turnover", "turnover_reduction_pct", "exposure_changes_per_year", "cost_saved_bps", "gross_benefit_retained_pct"])

        epochs_42 = epochs_by_seed[42]
        r_base_42 = [ep["base_return"] for ep in epochs_42]
        s_m9_42 = holdout_exposures[42]["M9_RawTiming"]
        to_m9_attr = compute_turnover_attribution(s_m9_42, r_base_42, cost_bps=0.0010)

        for pname in policy_names:
            s_pol = holdout_exposures[42][pname]
            attr = compute_turnover_attribution(s_pol, r_base_42, cost_bps=0.0010, m9_exposure_path=s_m9_42)
            to_red = (1.0 - attr["annual_turnover"] / to_m9_attr["annual_turnover"]) * 100.0 if to_m9_attr["annual_turnover"] > 0 else 0.0
            changes_yr = attr["n_exposure_changes"] * (52.0 / len(s_pol))
            cost_saved_bps = attr.get("cost_saved_vs_m9", 0.0) * 10000.0
            retained_pct = attr.get("gross_benefit_retained", 1.0) * 100.0
            writer.writerow([pname, f"{attr['annual_turnover']:.2f}", f"{to_red:.1f}%", f"{changes_yr:.1f}", f"{cost_saved_bps:.1f}", f"{retained_pct:.1f}%"])

    # ----------------------------------------------------------------------
    # STEP 4: Economic Attribution Decomposition
    # ----------------------------------------------------------------------
    print("\n[Step 4] Decomposing Economic Benefit (Avoided Losses vs Sacrificed Upside)...")
    with open(os.path.join(out_dir, "economic_attribution.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "avoided_losses_pct", "sacrificed_upside_pct", "bonus_upside_pct", "excess_tc_pct", "net_economic_benefit_pct"])

        mean_exp_42 = float(np.mean(s_m9_42))
        for pname in policy_names:
            s_pol = holdout_exposures[42][pname]
            decomp = decompose_timing_economic_benefit(r_base_42, s_pol, constant_exposure=mean_exp_42, cost_bps=0.0010)
            writer.writerow([pname, f"{decomp['avoided_losses']*100:.2f}%", f"{decomp['sacrificed_upside']*100:.2f}%",
                             f"{decomp['bonus_upside']*100:.2f}%", f"{decomp['excess_transaction_costs']*100:.2f}%",
                             f"{decomp['net_economic_benefit']*100:.2f}%"])

    # ----------------------------------------------------------------------
    # STEP 5: False Alarm Analysis
    # ----------------------------------------------------------------------
    print("\n[Step 5] Analyzing Uncertainty False Alarms...")
    u_abs_42 = [compute_aggregate_epistemic_uncertainty(ep["sigma_epi"], weights=ep["w_base"], method="q75") for ep in epochs_42]
    z_42 = [compute_relative_epistemic_uncertainty(u_abs_42[:t], u_abs_42[t], window=8, method="robust_zscore") for t in range(len(epochs_42))]

    with open(os.path.join(out_dir, "false_alarm_analysis.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "alarm_count", "alarm_frequency_pct", "pct_negative_followed", "pct_tail_loss_followed", "mean_return_alarm", "false_positive_rate_pct"])

        for pname in policy_names:
            s_pol = holdout_exposures[42][pname]
            alm = analyze_uncertainty_alarms(z_42, r_base_42, s_pol, tail_threshold=-0.02)
            writer.writerow([pname, int(alm["n_alarms"]), f"{alm['alarm_frequency']:.1f}%", f"{alm['pct_negative_followed']:.1f}%",
                             f"{alm['pct_tail_loss_followed']:.1f}%", f"{alm['mean_return_during_alarm']*100:.2f}%", f"{alm['false_positive_rate']:.1f}%"])

    # ----------------------------------------------------------------------
    # STEP 6: Uncertainty Episode Analysis (Summer 2024 Tech Correction)
    # ----------------------------------------------------------------------
    print("\n[Step 6] Running Episode Analysis for Summer 2024 Correction...")
    # Find summer 2024 drawdown dates (approx weeks 2024-W28 to 2024-W34)
    weeks = [ep["week"] for ep in epochs_42]
    with open(os.path.join(out_dir, "episode_analysis.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "start_week", "end_week", "policy", "mean_exposure", "local_return", "local_max_dd", "turnover_generated"])

        ep_mask = [(w >= "2024-W26" and w <= "2024-W35") for w in weeks]
        ep_idx = [i for i, m in enumerate(ep_mask) if m]

        if ep_idx:
            s_w = weeks[ep_idx[0]]
            e_w = weeks[ep_idx[-1]]
            sub_epochs = [epochs_42[i] for i in ep_idx]

            for pname in policy_names:
                sub_s = holdout_exposures[42][pname][ep_idx]
                sim_ep = simulate_portfolio_strategy(sub_epochs, sub_s, cost_bps=0.0010)
                writer.writerow(["Summer_2024_Correction", s_w, e_w, pname, f"{np.mean(sub_s):.3f}",
                                 f"{sim_ep['annualized_return']*100:.2f}%", f"{sim_ep['max_drawdown']*100:.2f}%",
                                 f"{np.sum(np.abs(np.diff(sub_s)))*2.0:.3f}"])

    # ----------------------------------------------------------------------
    # STEP 7: Transaction Cost Sensitivity Surface (0 to 50 bps)
    # ----------------------------------------------------------------------
    print("\n[Step 7] Evaluating Transaction Cost Sensitivity Curve (0, 5, 10, 25, 50 bps)...")
    cost_levels = [0.0, 0.0005, 0.0010, 0.0025, 0.0050]
    cost_records = {p: [] for p in policy_names}

    with open(os.path.join(out_dir, "cost_sensitivity.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cost_bps"] + [f"{p}_Sharpe" for p in policy_names] + [f"{p}_NetUtility" for p in policy_names])

        for c in cost_levels:
            c_bps = int(c * 10000)
            sh_row = []
            u_row = []
            for pname in policy_names:
                # Average across seeds
                shs = [simulate_portfolio_strategy(epochs_by_seed[s], holdout_exposures[s][pname], cost_bps=c)["sharpe_ratio"] for s in seeds]
                utils = [simulate_portfolio_strategy(epochs_by_seed[s], holdout_exposures[s][pname], cost_bps=c)["mean_utility"] for s in seeds]
                sh_row.append(f"{np.mean(shs):.3f}")
                u_row.append(f"{np.mean(utils):.6f}")
                cost_records[pname].append({"cost_bps": c_bps, "sharpe": np.mean(shs), "utility": np.mean(utils)})
            writer.writerow([c_bps] + sh_row + u_row)

    # ----------------------------------------------------------------------
    # STEP 8: Exposure Matching Verification Across Budgets (0.50, 0.70, 0.90)
    # ----------------------------------------------------------------------
    print("\n[Step 8] Verifying Exposure Matching Across Budgets (0.50, 0.70, 0.90)...")
    target_budgets = [0.50, 0.70, 0.90]
    with open(os.path.join(out_dir, "exposure_matching.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_budget", "policy", "achieved_mean_exposure", "sharpe_ratio", "delta_sharpe_vs_M8", "net_utility", "delta_utility_vs_M8"])

        for tb in target_budgets:
            for pname in policy_names:
                d_sh_list = []
                d_u_list = []
                achieved_list = []
                sh_list = []
                u_list = []
                for s in seeds:
                    epochs = epochs_by_seed[s]
                    raw_s = holdout_exposures[s][pname]
                    # Scale to target
                    s_scaled = np.clip(raw_s * (tb / np.mean(raw_s)), 0.0, 1.0)
                    s_m8_b = np.array([tb] * len(epochs))

                    res_pol = simulate_portfolio_strategy(epochs, s_scaled, cost_bps=0.0010)
                    res_m8 = simulate_portfolio_strategy(epochs, s_m8_b, cost_bps=0.0010)

                    achieved_list.append(np.mean(s_scaled))
                    sh_list.append(res_pol["sharpe_ratio"])
                    u_list.append(res_pol["mean_utility"])
                    d_sh_list.append(res_pol["sharpe_ratio"] - res_m8["sharpe_ratio"])
                    d_u_list.append(res_pol["mean_utility"] - res_m8["mean_utility"])

                writer.writerow([tb, pname, f"{np.mean(achieved_list):.3f}", f"{np.mean(sh_list):.3f}",
                                 f"{np.mean(d_sh_list):+.3f}", f"{np.mean(u_list):.6f}", f"{np.mean(d_u_list):+.6f}"])

    # ----------------------------------------------------------------------
    # STEP 9: Nested Walk-Forward Evaluation (Zero Lookahead across 3 folds)
    # ----------------------------------------------------------------------
    print("\n[Step 9] Running Nested Walk-Forward Evaluation across 3 folds...")
    h_weeks = list(group_by_weekly_rebalance(holdout_records[42]).keys())
    T_hold = len(h_weeks)

    folds = [
        ("Fold_1", 0, min(26, T_hold)),
        ("Fold_2", 26, min(48, T_hold)),
        ("Fold_3", 48, T_hold),
    ]

    wf_records = []
    with open(os.path.join(out_dir, "walkforward_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "start_week", "end_week", "policy", "sharpe", "delta_sharpe_vs_M8", "net_utility", "delta_utility_vs_M8", "annual_turnover"])

        for fname, s_idx, e_idx in folds:
            for pname in ["M8_Constant", "M9_RawTiming", "M15_Band", "M17_Combined"]:
                d_sh_list = []
                d_u_list = []
                sh_list = []
                u_list = []
                to_list = []

                for s in seeds:
                    d_b = group_by_weekly_rebalance(dev_records[s])
                    h_b = group_by_weekly_rebalance(holdout_records[s])
                    full_buckets = {**d_b, **h_b}
                    full_epochs = build_base_portfolio_series(full_buckets)["epochs"]

                    n_dev = len(d_b)
                    train_end = n_dev + s_idx
                    test_end = n_dev + e_idx

                    train_epochs = full_epochs[:train_end]
                    test_epochs = full_epochs[train_end:test_end]

                    u_abs_full = [compute_aggregate_epistemic_uncertainty(ep["sigma_epi"], weights=ep["w_base"], method="q75") for ep in full_epochs]

                    # Re-estimate band delta on train_epochs strictly
                    train_z = [compute_relative_epistemic_uncertainty(u_abs_full[:t], u_abs_full[t], window=8, method="robust_zscore") for t in range(train_end)]
                    train_raw = [compute_relative_epistemic_exposure_scale(z, s_base=0.85, lambda_rel=0.20, tau=0.0) for z in train_z]
                    train_mean_exp = float(np.mean(train_raw))

                    fold_delta = 0.10
                    fold_best_u = -999.0
                    for d_cand in [0.05, 0.10, 0.15]:
                        tr_b = compute_band_exposure_path(train_raw, delta=d_cand, eta=1.0)
                        if np.mean(tr_b) > 0:
                            tr_b = np.clip(tr_b * (train_mean_exp / np.mean(tr_b)), 0.0, 1.0)
                        res_tr = simulate_portfolio_strategy(train_epochs, tr_b, cost_bps=0.0010)
                        if res_tr["mean_utility"] > fold_best_u:
                            fold_best_u = res_tr["mean_utility"]
                            fold_delta = d_cand

                    # Trailing z for test
                    test_z = [compute_relative_epistemic_uncertainty(u_abs_full[:t], u_abs_full[t], window=8, method="robust_zscore") for t in range(train_end, test_end)]
                    raw_test_m9 = [compute_relative_epistemic_exposure_scale(z, s_base=0.85, lambda_rel=0.20, tau=0.0) for z in test_z]
                    mean_t_exp = float(np.mean(raw_test_m9))

                    if pname == "M8_Constant":
                        test_s = np.array([mean_t_exp] * len(test_epochs))
                    elif pname == "M9_RawTiming":
                        test_s = np.array(raw_test_m9)
                    elif pname == "M15_Band":
                        test_s = compute_band_exposure_path(raw_test_m9, delta=fold_delta, eta=1.0)
                    elif pname == "M17_Combined":
                        hys_path = compute_hysteresis_exposure_path(test_z, tau_enter=1.0, tau_exit=0.2, s_normal=0.85, s_reduced=0.45)
                        test_s = compute_band_exposure_path(hys_path, delta=fold_delta, eta=1.0)

                    # Scale to match mean
                    if np.mean(test_s) > 0:
                        test_s = np.clip(test_s * (mean_t_exp / np.mean(test_s)), 0.0, 1.0)
                    s_m8_wf = np.array([mean_t_exp] * len(test_epochs))

                    res_pol = simulate_portfolio_strategy(test_epochs, test_s, cost_bps=0.0010)
                    res_m8 = simulate_portfolio_strategy(test_epochs, s_m8_wf, cost_bps=0.0010)

                    sh_list.append(res_pol["sharpe_ratio"])
                    u_list.append(res_pol["mean_utility"])
                    d_sh_list.append(res_pol["sharpe_ratio"] - res_m8["sharpe_ratio"])
                    d_u_list.append(res_pol["mean_utility"] - res_m8["mean_utility"])
                    to_list.append(res_pol["mean_turnover"] * 52.0)

                writer.writerow([fname, h_weeks[s_idx], h_weeks[min(e_idx-1, T_hold-1)], pname,
                                 f"{np.mean(sh_list):.3f}", f"{np.mean(d_sh_list):+.3f}",
                                 f"{np.mean(u_list):.6f}", f"{np.mean(d_u_list):+.6f}", f"{np.mean(to_list):.2f}"])
                wf_records.append({"fold": fname, "policy": pname, "delta_sharpe": np.mean(d_sh_list), "delta_utility": np.mean(d_u_list)})

    # ----------------------------------------------------------------------
    # STEP 10: Placebo & Negative Control on Cost-Aware Policies (M13 & M17)
    # ----------------------------------------------------------------------
    print("\n[Step 10] Running 1,000 Block & Circular Placebos for M13 and M17...")
    placebo_records = []
    with open(os.path.join(out_dir, "placebo_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "policy", "placebo_type", "actual_utility", "placebo_mean", "percentile_rank", "empirical_pval"])

        for pol_eval in ["M13_Hysteresis", "M17_Combined"]:
            for s in seeds:
                epochs = epochs_by_seed[s]
                actual_s = holdout_exposures[s][pol_eval]
                actual_res = simulate_portfolio_strategy(epochs, actual_s, cost_bps=0.0010)

                # Circular shifts
                circ_paths = generate_placebo_exposure_paths(actual_s, n_placebos=1000, method="circular_shift", seed=s)
                circ_utils = [simulate_portfolio_strategy(epochs, p, cost_bps=0.0010)["mean_utility"] for p in circ_paths]
                rank_circ = float(np.mean(np.array(circ_utils) <= actual_res["mean_utility"]) * 100.0)

                # Block permutations (block size = 4 weeks)
                block_paths = generate_block_placebo_paths(actual_s, n_placebos=1000, block_size=4, seed=s)
                block_utils = [simulate_portfolio_strategy(epochs, p, cost_bps=0.0010)["mean_utility"] for p in block_paths]
                rank_block = float(np.mean(np.array(block_utils) <= actual_res["mean_utility"]) * 100.0)

                writer.writerow([s, pol_eval, "circular_shift", f"{actual_res['mean_utility']:.6f}", f"{np.mean(circ_utils):.6f}", f"{rank_circ:.1f}", f"{1 - rank_circ/100:.4f}"])
                writer.writerow([s, pol_eval, "block_permutation_b4", f"{actual_res['mean_utility']:.6f}", f"{np.mean(block_utils):.6f}", f"{rank_block:.1f}", f"{1 - rank_block/100:.4f}"])
                placebo_records.append({"seed": s, "policy": pol_eval, "rank_circ": rank_circ, "rank_block": rank_block})

    # Summary JSON Output
    summary_data = {
        "selected_parameters": selected_params,
        "walkforward_records": wf_records,
        "placebo_records": placebo_records,
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    # ----------------------------------------------------------------------
    # STEP 11: Generate 9 Publication Figures
    # ----------------------------------------------------------------------
    print("\n[Step 11] Generating 9 Publication Figures...")

    # Figure 1: Exposure Paths Comparison (M8, M9, M13, M15, M17)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(weeks, holdout_exposures[42]["M8_Constant"], color="gray", linestyle="--", lw=1.8, label="M8 Constant (0.70)")
    ax.plot(weeks, holdout_exposures[42]["M9_RawTiming"], color="#E69F00", alpha=0.5, lw=1.2, label="M9 Raw Timing (high churn)")
    ax.plot(weeks, holdout_exposures[42]["M13_Hysteresis"], color="#56B4E9", alpha=0.8, lw=1.8, label="M13 Hysteresis")
    ax.plot(weeks, holdout_exposures[42]["M15_Band"], color="#009E73", alpha=0.8, lw=1.8, label="M15 Rebalancing Band")
    ax.plot(weeks, holdout_exposures[42]["M17_Combined"], color="#0072B2", lw=2.2, label="M17 Combined (Hysteresis + Band)")
    ax.set_xticks(range(0, len(weeks), 8))
    ax.set_xticklabels([weeks[i] for i in range(0, len(weeks), 8)], rotation=30)
    ax.set_ylabel("Portfolio Exposure Scale (s_t)", fontsize=11)
    ax.set_title("Exposure Paths Comparison: Raw Epistemic Timing vs Cost-Aware Rules", fontsize=12, pad=12, fontweight="bold")
    ax.legend(frameon=True, loc="lower left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exposure_paths_comparison.png"))
    plt.close()

    # Figure 2: Cumulative Net Wealth After Costs (c=10 bps)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    colors = {"M8_Constant": "gray", "M9_RawTiming": "#E69F00", "M13_Hysteresis": "#56B4E9",
              "M14_Persistence": "#CC79A7", "M15_Band": "#009E73", "M16_Biweekly": "#F0E442", "M17_Combined": "#0072B2"}
    for pname in ["M8_Constant", "M9_RawTiming", "M13_Hysteresis", "M15_Band", "M17_Combined"]:
        wealth = holdout_sim_results[pname][0]["wealth"]
        ax.plot(wealth, lw=2.0, color=colors[pname], label=pname.replace("_", " "))
    ax.set_ylabel("Cumulative Wealth ($)", fontsize=11)
    ax.set_xlabel("Rebalancing Weeks", fontsize=11)
    ax.set_title("Cumulative Net Wealth After 10 bps Friction", fontsize=12, pad=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cumulative_wealth.png"))
    plt.close()

    # Figure 3: Drawdowns Profile
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    for pname in ["M8_Constant", "M9_RawTiming", "M17_Combined"]:
        dd = np.array(holdout_sim_results[pname][0]["drawdowns"]) * 100.0
        ax.plot(dd, lw=1.8, color=colors[pname], label=pname.replace("_", " "))
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.set_xlabel("Rebalancing Weeks", fontsize=11)
    ax.set_title("Drawdown Profiles: Capital Protection Comparison", fontsize=12, pad=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drawdowns.png"))
    plt.close()

    # Figure 4: Turnover vs Net Utility Scatter
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for pname in policy_names:
        ann_to = np.mean([r["mean_turnover"] for r in holdout_sim_results[pname]]) * 52.0
        u_val = np.mean([r["mean_utility"] for r in holdout_sim_results[pname]])
        ax.scatter(ann_to, u_val, color=colors.get(pname, "#333333"), s=120, label=pname.replace("_", " "), zorder=4)
        ax.annotate(pname.split("_")[0], (ann_to + 0.05, u_val), fontsize=10, fontweight="bold")
    ax.set_xlabel("Annualized Turnover", fontsize=11)
    ax.set_ylabel("Mean Realized Net Utility (c = 10 bps)", fontsize=11)
    ax.set_title("Turnover vs Net Utility Frontier Across Execution Rules", fontsize=12, pad=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "turnover_vs_utility.png"))
    plt.close()

    # Figure 5: Transaction Cost Drag Curve (0 to 50 bps)
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    c_bps_vals = [r["cost_bps"] for r in cost_records["M8_Constant"]]
    for pname in ["M8_Constant", "M9_RawTiming", "M15_Band", "M17_Combined"]:
        sh_vals = [r["sharpe"] for r in cost_records[pname]]
        ax.plot(c_bps_vals, sh_vals, marker="o", lw=2.0, color=colors[pname], label=pname.replace("_", " "))
    ax.set_xlabel("Transaction Cost Drag (Basis Points)", fontsize=11)
    ax.set_ylabel("Realized Sharpe Ratio", fontsize=11)
    ax.set_title("Sharpe Ratio Decay Across Transaction Cost Levels", fontsize=12, pad=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "transaction_cost_drag.png"))
    plt.close()

    # Figure 6: Avoided Loss vs Sacrificed Upside
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    p_sub = ["M9_RawTiming", "M13_Hysteresis", "M15_Band", "M17_Combined"]
    decomp_m9 = decompose_timing_economic_benefit(r_base_42, holdout_exposures[42]["M9_RawTiming"], constant_exposure=mean_exp_42)
    decomp_m13 = decompose_timing_economic_benefit(r_base_42, holdout_exposures[42]["M13_Hysteresis"], constant_exposure=mean_exp_42)
    decomp_m15 = decompose_timing_economic_benefit(r_base_42, holdout_exposures[42]["M15_Band"], constant_exposure=mean_exp_42)
    decomp_m17 = decompose_timing_economic_benefit(r_base_42, holdout_exposures[42]["M17_Combined"], constant_exposure=mean_exp_42)

    av_losses = [d["avoided_losses"] * 100 for d in [decomp_m9, decomp_m13, decomp_m15, decomp_m17]]
    sac_ups = [-d["sacrificed_upside"] * 100 for d in [decomp_m9, decomp_m13, decomp_m15, decomp_m17]]
    tc_drags = [-d["excess_transaction_costs"] * 100 for d in [decomp_m9, decomp_m13, decomp_m15, decomp_m17]]

    x = np.arange(len(p_sub))
    w = 0.25
    ax.bar(x - w, av_losses, width=w, label="Avoided Losses (+)", color="#009E73")
    ax.bar(x, sac_ups, width=w, label="Sacrificed Upside (-)", color="#D55E00")
    ax.bar(x + w, tc_drags, width=w, label="Turnover Costs (-)", color="#CC79A7")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in p_sub])
    ax.set_ylabel("Impact on Portfolio (% of Capital)", fontsize=11)
    ax.set_title("Loss Avoidance vs Sacrifice vs Costs: Preserving the Edge", fontsize=12, pad=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "avoided_loss_vs_sacrificed_upside.png"))
    plt.close()

    # Figure 7: False Alarm Rates
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fp_rates = []
    for pname in ["M9_RawTiming", "M13_Hysteresis", "M14_Persistence", "M15_Band", "M17_Combined"]:
        alm = analyze_uncertainty_alarms(z_42, r_base_42, holdout_exposures[42][pname])
        fp_rates.append(alm["false_positive_rate"])
    ax.bar(["M9 Raw", "M13 Hysteresis", "M14 Persistence", "M15 Band", "M17 Combined"], fp_rates, color="#0072B2", alpha=0.85, width=0.45)
    ax.set_ylabel("False Positive Rate (%)", fontsize=11)
    ax.set_title("False Alarm Rates: Filtering Transient De-risking Whipsaws", fontsize=12, pad=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "false_alarm_rates.png"))
    plt.close()

    # Figure 8: Uncertainty Episode Timeline (Summer 2024 Tech Correction)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sub_weeks = [weeks[i] for i in ep_idx]
    sub_r = np.array([epochs_42[i]["base_return"] for i in ep_idx])
    sub_cum = np.cumprod(1.0 + sub_r) - 1.0
    ax.plot(sub_weeks, sub_cum * 100, color="black", lw=2.2, label="Base Asset Return (%)")
    ax2 = ax.twinx()
    ax2.plot(sub_weeks, holdout_exposures[42]["M9_RawTiming"][ep_idx], color="#E69F00", linestyle=":", lw=1.8, label="M9 Raw Exposure")
    ax2.plot(sub_weeks, holdout_exposures[42]["M17_Combined"][ep_idx], color="#0072B2", lw=2.4, label="M17 Cost-Aware Exposure")
    ax.set_ylabel("Cumulative Base Return (%)", fontsize=11)
    ax2.set_ylabel("Exposure Scale (s_t)", fontsize=11)
    ax.set_xticks(range(0, len(sub_weeks), 2))
    ax.set_xticklabels([sub_weeks[i] for i in range(0, len(sub_weeks), 2)], rotation=25)
    ax.set_title("Summer 2024 Tech Correction Episode: Entry and Exit Dynamics", fontsize=12, pad=12, fontweight="bold")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower left", frameon=True)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "uncertainty_episode_timeline.png"))
    plt.close()

    # Figure 9: Walk-Forward Out-of-Sample Delta Utility
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    f_names = ["Fold_1", "Fold_2", "Fold_3"]
    w_pols = ["M9_RawTiming", "M15_Band", "M17_Combined"]
    x = np.arange(len(f_names))
    w = 0.25
    for i, p in enumerate(w_pols):
        deltas = [np.mean([r["delta_utility"] for r in wf_records if r["fold"] == f and r["policy"] == p]) * 10000 for f in f_names]
        ax.bar(x + (i - 1) * w, deltas, width=w, label=p.replace("_", " "), color=colors[p], alpha=0.85)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(f_names)
    ax.set_ylabel("Out-of-Sample Delta Net Utility (bps)", fontsize=11)
    ax.set_title("Nested Walk-Forward Out-of-Sample Utility vs Matched Baseline", fontsize=12, pad=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "walkforward_results.png"))
    plt.close()

    print(f"\nAll 9 publication figures and 11 CSV/JSON artifacts saved to {out_dir}/")
    print("=" * 80)
    print("PHASE 8 COST-AWARE TIMING PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
