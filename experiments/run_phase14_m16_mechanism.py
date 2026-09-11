#!/usr/bin/env python3
"""Phase 14: Mechanism Decomposition of Uncertainty-Aware Risk Inflation.

Deconstructs the market and cross-sectional conditions that explain why frozen M16
(lambda_u = 0.10) underperformed in 2022 (-0.000208), outperformed strongly in
2023 (+0.000592), and underperformed in 2024-2025 (-0.000181).

Strictly diagnostic:
- No new strategies (no M17, no regime-switching, no conditional M16).
- No parameter tuning (lambda_u = 0.10 remains frozen).
- Primary clean regimes: 2022, 2023, 2024-2025 (N=4,100 clean events, 168 weeks).
- Secondary diagnostic regimes: 2020, 2021 (N=2,389 events, 99 weeks).
- Evaluates seeds 42, 123, 777 separately.
"""

from __future__ import annotations

import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Path setup
repo_root = Path(__file__).resolve().parent.parent
sibling_src = Path("/Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/src")
if str(sibling_src) not in sys.path:
    sys.path.insert(0, str(sibling_src))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from hypergraph_portfolio.mean_variance import MeanVariancePortfolio  # noqa: E402


def compute_asset_relative_uncertainty(sigma_epi: np.ndarray) -> np.ndarray:
    """Compute robust cross-sectional relative epistemic uncertainty within cohort:

    z_i = (sigma_i - median(sigma)) / (1.4826 * MAD(sigma))
    """
    s = np.asarray(sigma_epi, dtype=float)
    m = len(s)
    if m <= 1:
        return np.zeros(m)
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    denom = 1.4826 * mad
    if denom < 1e-6:
        std_val = float(np.std(s, ddof=1))
        denom = std_val if std_val > 1e-6 else 1.0
    return (s - med) / denom


def run_cohort_markowitz(
    mu: np.ndarray,
    Q: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """Solve Markowitz quadratic program for entering cohort."""
    m = len(mu)
    if m == 0:
        return np.array([])
    if m == 1:
        return np.array([1.0])
    try:
        w, _ = MeanVariancePortfolio(Q=gamma * Q, mu=mu, budget=1.0).solve()
        w = np.clip(w, 0.0, 1.0)
        s_w = np.sum(w)
        if s_w > 1e-8:
            return w / s_w
        return np.ones(m) / float(m)
    except Exception:
        return np.ones(m) / float(m)


def build_event_returns_map(rec_path: Path) -> dict[tuple[str, str], list[float]]:
    """Build exact 13-week geometric compounding return map across historical events."""
    rec_df = pd.read_csv(rec_path)
    by_sym = defaultdict(list)
    for _, row in rec_df.iterrows():
        by_sym[str(row["symbol"])].append(dict(row))

    event_returns: dict[tuple[str, str], list[float]] = {}
    for _, row in rec_df.iterrows():
        sym = str(row["symbol"])
        d = str(row["date"])
        m1 = float(row["move_1d"]) if pd.notna(row["move_1d"]) else 0.0

        evs = by_sym.get(sym, [])
        idx = next((i for i, e in enumerate(evs) if str(e["date"]) == d), None)
        if idx is not None:
            match_row = evs[idx]
            m30 = float(match_row["move_30d"]) if pd.notna(match_row.get("move_30d")) else m1
            try:
                p_cur = float(match_row["d_minus_1_close"])
                if idx + 1 < len(evs) and pd.notna(evs[idx + 1].get("d_minus_1_close")):
                    p_next = float(evs[idx + 1]["d_minus_1_close"])
                    r13 = (p_next - p_cur) / p_cur
                else:
                    r13 = m30
            except Exception:
                r13 = m30
        else:
            m30 = m1
            r13 = m30

        if r13 > 2.0 or r13 < -0.8:
            r13 = m30

        ratio_30_1 = max((1.0 + m30) / (1.0 + m1), 1e-4)
        r_drift_w1_4 = (ratio_30_1 ** (1.0 / 3.0)) - 1.0

        ratio_13_30 = max((1.0 + r13) / (1.0 + m30), 1e-4)
        r_drift_w5_13 = (ratio_13_30 ** (1.0 / 9.0)) - 1.0

        rets_by_age = [m1]
        for _ in range(3):
            rets_by_age.append(r_drift_w1_4)
        for _ in range(9):
            rets_by_age.append(r_drift_w5_13)

        event_returns[(d, sym)] = rets_by_age
    return event_returns


def compute_spectral_properties(Q: np.ndarray) -> dict[str, float]:
    """Compute condition number, largest eigenvalue share, and effective rank of covariance matrix."""
    m = Q.shape[0]
    if m <= 1:
        return {
            "largest_eigenvalue_share": 1.0,
            "effective_rank": 1.0,
            "condition_number": 1.0,
            "mean_pairwise_correlation": 1.0,
            "median_pairwise_correlation": 1.0,
        }

    # Compute correlation matrix
    diag_std = np.sqrt(np.maximum(np.diag(Q), 1e-12))
    corr_mat = Q / np.outer(diag_std, diag_std)
    corr_mat = np.clip(corr_mat, -1.0, 1.0)
    np.fill_diagonal(corr_mat, 1.0)

    # Off-diagonal correlations
    mask = ~np.eye(m, dtype=bool)
    off_diags = corr_mat[mask]
    mean_corr = float(np.mean(off_diags)) if len(off_diags) > 0 else 0.0
    med_corr = float(np.median(off_diags)) if len(off_diags) > 0 else 0.0

    # Eigenvalues of Q
    try:
        eigvals = np.linalg.eigvalsh(Q)
        eigvals = np.maximum(eigvals, 1e-12)
        sum_eigs = np.sum(eigvals)
        p = eigvals / sum_eigs
        entropy = -np.sum(p * np.log(p + 1e-15))
        eff_rank = float(np.exp(entropy))
        cond_num = float(eigvals[-1] / eigvals[0])
        largest_share = float(eigvals[-1] / sum_eigs)
    except Exception:
        eff_rank = float(m)
        cond_num = 1.0
        largest_share = 1.0 / float(m)

    return {
        "largest_eigenvalue_share": round(largest_share, 4),
        "effective_rank": round(eff_rank, 4),
        "condition_number": round(cond_num, 4),
        "mean_pairwise_correlation": round(mean_corr, 4),
        "median_pairwise_correlation": round(med_corr, 4),
    }


def compute_concentration_metrics(weights: np.ndarray) -> dict[str, float]:
    """Compute Herfindahl-Hirschman index, effective N, max weight, and top-5 weight."""
    w = np.asarray(weights, dtype=float)
    s = np.sum(w)
    if s > 1e-8:
        w = w / s
    else:
        return {
            "herfindahl_index": 0.0,
            "effective_n_holdings": 0.0,
            "max_asset_weight": 0.0,
            "top5_asset_weight": 0.0,
        }

    hhi = float(np.sum(w**2))
    eff_n = float(1.0 / hhi) if hhi > 1e-8 else 0.0
    max_w = float(np.max(w))
    sorted_w = np.sort(w)[::-1]
    top5_w = float(np.sum(sorted_w[:5]))

    return {
        "herfindahl_index": round(hhi, 5),
        "effective_n_holdings": round(eff_n, 2),
        "max_asset_weight": round(max_w, 4),
        "top5_asset_weight": round(top5_w, 4),
    }


def simulate_and_extract_weekly_mechanisms(
    eval_preds_df: pd.DataFrame,
    rec_path: Path,
    horizon: int = 4,
    cost_bps: float = 0.0010,
    gamma: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate M0 and M16 week-by-week, extracting all ex-ante and post-hoc mechanism variables."""
    ret_map = build_event_returns_map(rec_path)
    rec_df = pd.read_csv(rec_path)
    ret_lookup = {
        str(r["episode_id"]): float(r["move_1d"])
        for _, r in rec_df.iterrows()
        if pd.notna(r["move_1d"])
    }

    regimes_order = ["2020", "2021", "2022", "2023", "2024-2025"]
    clean_regimes = ["2022", "2023", "2024-2025"]
    seeds = [42, 123, 777]

    weekly_rows: list[dict[str, Any]] = []
    weight_shift_rows: list[dict[str, Any]] = []
    error_alignment_rows: list[dict[str, Any]] = []

    # Process each regime and seed
    for reg in regimes_order:
        reg_type = "CLEAN_OOS" if reg in clean_regimes else "TRAINING_OVERLAP_DIAGNOSTIC"

        for seed in seeds:
            sub_preds = eval_preds_df[
                (eval_preds_df["regime"] == reg) & (eval_preds_df["seed"] == seed)
            ].to_dict(orient="records")

            p_mean_all = float(np.mean([r["posterior_mean"] for r in sub_preds]))

            # Group by calendar week
            buckets = defaultdict(list)
            for r in sub_preds:
                dt = datetime.datetime.fromisoformat(r["date"].replace(" ", "T"))
                w_key = f"{dt.year}-W{dt.isocalendar()[1]:02d}"
                buckets[w_key].append(r)
            sorted_buckets = dict(sorted(buckets.items()))
            week_keys = list(sorted_buckets.keys())
            T = len(sorted_buckets)

            # Solve cohort weights for M0 and M16
            cohort_data_m0 = []
            cohort_data_m16 = []
            cohort_metadata = []

            for w_key, basket in sorted_buckets.items():
                m = len(basket)
                assets = [r["ticker"] for r in basket]
                timestamps = [r["date"] for r in basket]
                event_ids = [r["event_id"] for r in basket]
                p_means = np.array([float(r["posterior_mean"]) for r in basket])
                sigma_epis = np.array([float(np.sqrt(max(r["epistemic_uncertainty"], 1e-12))) for r in basket])
                mu_raw = np.array([-(p - p_mean_all) for p in p_means])

                # M0 solution
                Q_base = 0.04 * np.eye(m) + 0.01 * np.ones((m, m))
                w_m0 = run_cohort_markowitz(mu=mu_raw, Q=Q_base, gamma=gamma)

                # M16 solution
                z_cohort = compute_asset_relative_uncertainty(sigma_epis)
                z_pos = np.maximum(z_cohort, 0.0)
                Q_m16 = Q_base + 0.10 * np.diag(z_pos)
                w_m16 = run_cohort_markowitz(mu=mu_raw, Q=Q_m16, gamma=gamma)

                # Store cohort info
                cohort_data_m0.append({
                    "week": w_key,
                    "assets": assets,
                    "timestamps": timestamps,
                    "w_base": w_m0,
                    "sigma_epi": sigma_epis,
                })
                cohort_data_m16.append({
                    "week": w_key,
                    "assets": assets,
                    "timestamps": timestamps,
                    "w_base": w_m16,
                    "sigma_epi": sigma_epis,
                })
                cohort_metadata.append({
                    "week": w_key,
                    "basket": basket,
                    "event_ids": event_ids,
                    "assets": assets,
                    "p_means": p_means,
                    "sigma_epis": sigma_epis,
                    "mu_raw": mu_raw,
                    "w_m0": w_m0,
                    "w_m16": w_m16,
                    "z_pos": z_pos,
                    "Q_base": Q_base,
                    "Q_m16": Q_m16,
                })

            # Run stacked portfolio simulations for M0 and M16
            from hypergraph_portfolio.uncertainty import simulate_cohort_stacked_portfolio
            sim_m0 = simulate_cohort_stacked_portfolio(
                weekly_cohorts=cohort_data_m0,
                event_return_map=ret_map,
                horizon=horizon,
                mode="markowitz",
                exposure_path=np.ones(T),
                cost_bps=cost_bps,
            )
            sim_m16 = simulate_cohort_stacked_portfolio(
                weekly_cohorts=cohort_data_m16,
                event_return_map=ret_map,
                horizon=horizon,
                mode="markowitz",
                exposure_path=np.ones(T),
                cost_bps=cost_bps,
            )

            # Weekly returns and turnovers
            r_m0 = np.array(sim_m0["weekly_net_returns"])
            r_m16 = np.array(sim_m16["weekly_net_returns"])
            to_m0 = np.array(sim_m0["weekly_traded_turnovers"])
            to_m16 = np.array(sim_m16["weekly_traded_turnovers"])

            # Compute wealth trajectories and drawdowns
            wealth_m0 = np.cumprod(1.0 + r_m0)
            peaks_m0 = np.maximum.accumulate(wealth_m0)
            dd_m0 = (wealth_m0 - peaks_m0) / peaks_m0

            wealth_m16 = np.cumprod(1.0 + r_m16)
            peaks_m16 = np.maximum.accumulate(wealth_m16)
            dd_m16 = (wealth_m16 - peaks_m16) / peaks_m16

            # Weekly utilities reconciling to Phase 13C
            # Canonical formula: u_bar = mean(r) - 0.5 * gamma * var(r, ddof=1)
            # Exact weekly reconciliation: u_t = r_t - 0.5 * gamma * (T / (T - 1)) * (r_t - mean_r)^2
            scale_ddof = float(T / (T - 1)) if T > 1 else 1.0
            mean_r_m0 = float(np.mean(r_m0))
            var_contrib_m0 = scale_ddof * ((r_m0 - mean_r_m0) ** 2)
            u_m0 = r_m0 - 0.5 * gamma * var_contrib_m0

            mean_r_m16 = float(np.mean(r_m16))
            var_contrib_m16 = scale_ddof * ((r_m16 - mean_r_m16) ** 2)
            u_m16 = r_m16 - 0.5 * gamma * var_contrib_m16

            delta_u = u_m16 - u_m0
            delta_ret = r_m16 - r_m0
            delta_var = var_contrib_m16 - var_contrib_m0
            delta_dd = dd_m16 - dd_m0
            delta_to = to_m16 - to_m0

            # Verify reconciliation to Phase 13C mean utility
            mean_u_sim_m0 = float(sim_m0["mean_net_utility"])
            mean_u_sim_m16 = float(sim_m16["mean_net_utility"])
            assert np.isclose(np.mean(u_m0), mean_u_sim_m0, atol=1e-6)
            assert np.isclose(np.mean(u_m16), mean_u_sim_m16, atol=1e-6)

            # Extract week-level state variables
            for t in range(T):
                meta = cohort_metadata[t]
                w_key = week_keys[t]
                m_t = len(meta["basket"])

                sigmas = meta["sigma_epis"]
                p_means = meta["p_means"]
                mu_raw = meta["mu_raw"]
                w_m0_t = meta["w_m0"]
                w_m16_t = meta["w_m16"]
                Q_b = meta["Q_base"]

                # A. Uncertainty Level (Ex-Ante)
                mean_unc = float(np.mean(sigmas))
                med_unc = float(np.median(sigmas))
                q75_unc = float(np.percentile(sigmas, 75))
                q90_unc = float(np.percentile(sigmas, 90))

                # B. Uncertainty Dispersion (Ex-Ante)
                std_unc = float(np.std(sigmas, ddof=1)) if m_t > 1 else 0.0
                iqr_unc = float(np.percentile(sigmas, 75) - np.percentile(sigmas, 25)) if m_t > 1 else 0.0
                q90_q10_unc = float(np.percentile(sigmas, 90) - np.percentile(sigmas, 10)) if m_t > 1 else 0.0
                cv_unc = float(std_unc / (mean_unc + 1e-12))

                # C. Forecast Dispersion (Ex-Ante)
                std_p = float(np.std(p_means, ddof=1)) if m_t > 1 else 0.0
                iqr_p = float(np.percentile(p_means, 75) - np.percentile(p_means, 25)) if m_t > 1 else 0.0
                alpha_disp = float(np.std(mu_raw, ddof=1)) if m_t > 1 else 0.0

                # D. Volatility State (Ex-Ante)
                asset_vols = np.sqrt(np.diag(Q_b))
                mean_vol = float(np.mean(asset_vols))
                med_vol = float(np.median(asset_vols))
                disp_vol = float(np.std(asset_vols, ddof=1)) if m_t > 1 else 0.0

                # E. Correlation & Spectrum State (Ex-Ante)
                spec = compute_spectral_properties(Q_b)

                # F. Concentration Metrics (Ex-Ante)
                conc_m0 = compute_concentration_metrics(w_m0_t)
                conc_m16 = compute_concentration_metrics(w_m16_t)
                delta_hhi = conc_m16["herfindahl_index"] - conc_m0["herfindahl_index"]
                delta_eff_n = conc_m16["effective_n_holdings"] - conc_m0["effective_n_holdings"]

                # G. Uncertainty-Weight Alignment (Ex-Ante)
                if m_t > 1 and np.std(w_m0_t) > 1e-8 and np.std(sigmas) > 1e-8:
                    corr_m0_unc = float(np.corrcoef(w_m0_t, sigmas)[0, 1])
                else:
                    corr_m0_unc = 0.0

                # Capital reduction by uncertainty quintile
                # Divide cohort into quintiles (or equal bins if m < 5)
                n_bins = min(5, m_t)
                if n_bins >= 2:
                    unc_ranks = pd.qcut(sigmas, q=n_bins, labels=False, duplicates="drop")
                else:
                    unc_ranks = np.zeros(m_t, dtype=int)

                weight_delta_pos = np.maximum(w_m0_t - w_m16_t, 0.0)  # capital removed by M16
                tot_reallocated = float(np.sum(weight_delta_pos))

                cap_removed_q = defaultdict(float)
                for i in range(m_t):
                    cap_removed_q[int(unc_ranks[i])] += float(weight_delta_pos[i])

                # Highest uncertainty bin
                max_bin = int(np.max(unc_ranks)) if len(unc_ranks) > 0 else 0
                cap_removed_highest = float(cap_removed_q[max_bin])
                second_highest_bin = max_bin - 1 if max_bin > 0 else 0
                cap_removed_top2 = float(cap_removed_q[max_bin] + cap_removed_q[second_highest_bin])

                # H. POST-HOC Realized Mechanism Variables
                # Match 1-day realized returns
                realized_rets = np.array([ret_lookup.get(eid, 0.0) for eid in meta["event_ids"]])
                post_hoc_ret_std = float(np.std(realized_rets, ddof=1)) if m_t > 1 else 0.0
                post_hoc_ret_iqr = float(np.percentile(realized_rets, 75) - np.percentile(realized_rets, 25)) if m_t > 1 else 0.0
                post_hoc_ret_tail = float(np.percentile(realized_rets, 90) - np.percentile(realized_rets, 10)) if m_t > 1 else 0.0
                post_hoc_ret_range = float(np.max(realized_rets) - np.min(realized_rets)) if m_t > 0 else 0.0

                # Forecast error: downside return <= -0.05 or quantile 0.20
                y_true = (realized_rets <= np.percentile(realized_rets, 20)).astype(int) if m_t > 4 else (realized_rets <= 0.0).astype(int)
                abs_errors = np.abs(p_means - y_true)
                mean_err = float(np.mean(abs_errors)) if m_t > 0 else 0.0
                std_err = float(np.std(abs_errors, ddof=1)) if m_t > 1 else 0.0

                if m_t > 1 and np.std(sigmas) > 1e-8 and np.std(abs_errors) > 1e-8:
                    corr_unc_err = float(np.corrcoef(sigmas, abs_errors)[0, 1])
                    spearman_unc_err, _ = stats.spearmanr(sigmas, abs_errors)
                    spearman_unc_err = float(spearman_unc_err)
                else:
                    corr_unc_err = 0.0
                    spearman_unc_err = 0.0

                # Error-aware capital reduction diagnostic:
                # Was capital removed from names with above-median forecast error?
                med_err = float(np.median(abs_errors)) if m_t > 0 else 0.0
                high_err_mask = abs_errors >= med_err
                cap_removed_from_high_err = float(np.sum(weight_delta_pos[high_err_mask]))
                err_reduction_share = float(cap_removed_from_high_err / (tot_reallocated + 1e-12))

                # Also track weight shift by quintile
                for q_idx in range(n_bins):
                    q_mask = unc_ranks == q_idx
                    if np.any(q_mask):
                        weight_shift_rows.append({
                            "regime": reg,
                            "regime_type": reg_type,
                            "seed": seed,
                            "calendar_week": w_key,
                            "quintile": f"Q{q_idx+1}",
                            "n_assets": int(np.sum(q_mask)),
                            "m0_weight": float(np.sum(w_m0_t[q_mask])),
                            "m16_weight": float(np.sum(w_m16_t[q_mask])),
                            "capital_shifted": float(np.sum(w_m16_t[q_mask]) - np.sum(w_m0_t[q_mask])),
                            "mean_forecast_error": float(np.mean(abs_errors[q_mask])),
                            "mean_realized_return": float(np.mean(realized_rets[q_mask])),
                            "realized_pnl_diff": float(np.sum((w_m16_t[q_mask] - w_m0_t[q_mask]) * realized_rets[q_mask])),
                        })

                # Error alignment diagnostic row
                error_alignment_rows.append({
                    "regime": reg,
                    "regime_type": reg_type,
                    "seed": seed,
                    "calendar_week": w_key,
                    "delta_u_m16": round(float(delta_u[t]), 6),
                    "m16_succeeded": bool(delta_u[t] > 0),
                    "total_capital_reallocated": round(tot_reallocated, 4),
                    "cap_removed_from_high_err": round(cap_removed_from_high_err, 4),
                    "err_reduction_share": round(err_reduction_share, 4),
                    "mean_forecast_error": round(mean_err, 4),
                    "corr_unc_err": round(corr_unc_err, 4),
                })

                # Full weekly row
                weekly_rows.append({
                    "regime": reg,
                    "regime_type": reg_type,
                    "seed": seed,
                    "calendar_week": w_key,
                    "week_idx": t,
                    "n_cohort_events": m_t,
                    # Core Outcome Variables
                    "delta_u_m16": round(float(delta_u[t]), 6),
                    "delta_return": round(float(delta_ret[t]), 6),
                    "delta_variance": round(float(delta_var[t]), 6),
                    "delta_drawdown_contrib": round(float(delta_dd[t]), 6),
                    "delta_turnover": round(float(delta_to[t]), 4),
                    "weekly_utility_m0": round(float(u_m0[t]), 6),
                    "weekly_utility_m16": round(float(u_m16[t]), 6),
                    "net_return_m0": round(float(r_m0[t]), 6),
                    "net_return_m16": round(float(r_m16[t]), 6),
                    "turnover_m0": round(float(to_m0[t]), 4),
                    "turnover_m16": round(float(to_m16[t]), 4),
                    # A. Uncertainty Level (Ex-Ante)
                    "mean_uncertainty": round(mean_unc, 6),
                    "median_uncertainty": round(med_unc, 6),
                    "q75_uncertainty": round(q75_unc, 6),
                    "q90_uncertainty": round(q90_unc, 6),
                    # B. Uncertainty Dispersion (Ex-Ante)
                    "std_uncertainty": round(std_unc, 6),
                    "iqr_uncertainty": round(iqr_unc, 6),
                    "q90_q10_uncertainty": round(q90_q10_unc, 6),
                    "cv_uncertainty": round(cv_unc, 4),
                    # C. Forecast Dispersion (Ex-Ante)
                    "std_p_mean": round(std_p, 4),
                    "iqr_p_mean": round(iqr_p, 4),
                    "alpha_dispersion": round(alpha_disp, 4),
                    # D. Volatility State (Ex-Ante)
                    "mean_asset_volatility": round(mean_vol, 4),
                    "median_asset_volatility": round(med_vol, 4),
                    "cross_sectional_vol_dispersion": round(disp_vol, 4),
                    # E. Correlation & Spectrum State (Ex-Ante)
                    "mean_pairwise_correlation": spec["mean_pairwise_correlation"],
                    "median_pairwise_correlation": spec["median_pairwise_correlation"],
                    "largest_eigenvalue_share": spec["largest_eigenvalue_share"],
                    "effective_rank": spec["effective_rank"],
                    "condition_number": spec["condition_number"],
                    # F. Concentration Metrics (Ex-Ante)
                    "hhi_m0": conc_m0["herfindahl_index"],
                    "hhi_m16": conc_m16["herfindahl_index"],
                    "effective_n_m0": conc_m0["effective_n_holdings"],
                    "effective_n_m16": conc_m16["effective_n_holdings"],
                    "max_weight_m0": conc_m0["max_asset_weight"],
                    "max_weight_m16": conc_m16["max_asset_weight"],
                    "top5_weight_m0": conc_m0["top5_asset_weight"],
                    "top5_weight_m16": conc_m16["top5_asset_weight"],
                    "delta_hhi": round(delta_hhi, 5),
                    "delta_effective_n": round(delta_eff_n, 2),
                    # G. Uncertainty-Weight Alignment (Ex-Ante)
                    "corr_m0_weight_uncertainty": round(corr_m0_unc, 4),
                    "total_capital_reallocated": round(tot_reallocated, 4),
                    "cap_removed_highest_uncertainty": round(cap_removed_highest, 4),
                    "cap_removed_top2_uncertainty": round(cap_removed_top2, 4),
                    # H. POST-HOC Mechanism Variables
                    "post_hoc_mean_forecast_error": round(mean_err, 4),
                    "post_hoc_std_forecast_error": round(std_err, 4),
                    "post_hoc_corr_uncertainty_error": round(corr_unc_err, 4),
                    "post_hoc_spearman_uncertainty_error": round(spearman_unc_err, 4),
                    "post_hoc_realized_return_std": round(post_hoc_ret_std, 4),
                    "post_hoc_realized_return_iqr": round(post_hoc_ret_iqr, 4),
                    "post_hoc_realized_return_tail": round(post_hoc_ret_tail, 4),
                    "post_hoc_realized_return_range": round(post_hoc_ret_range, 4),
                    "post_hoc_error_reduction_share": round(err_reduction_share, 4),
                })

    weekly_df = pd.DataFrame(weekly_rows)
    weight_shift_df = pd.DataFrame(weight_shift_rows)
    error_alignment_df = pd.DataFrame(error_alignment_rows)

    return weekly_df, weight_shift_df, error_alignment_df


def compute_regime_comparison(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Build regime comparison table summarizing mechanism state variables across regimes."""
    rows = []
    regimes_order = ["2020", "2021", "2022", "2023", "2024-2025"]
    clean_regimes = ["2022", "2023", "2024-2025"]

    for reg in regimes_order:
        sub = weekly_df[weekly_df["regime"] == reg]
        reg_type = "CLEAN_OOS" if reg in clean_regimes else "TRAINING_OVERLAP_DIAGNOSTIC"

        rows.append({
            "regime": reg,
            "regime_type": reg_type,
            "n_weeks_total": len(sub),
            "mean_delta_u_m16": round(float(sub["delta_u_m16"].mean()), 6),
            "median_delta_u_m16": round(float(sub["delta_u_m16"].median()), 6),
            "positive_weeks_pct": round(float(np.mean(sub["delta_u_m16"] > 0) * 100), 2),
            "mean_uncertainty_level": round(float(sub["mean_uncertainty"].mean()), 6),
            "mean_uncertainty_dispersion": round(float(sub["std_uncertainty"].mean()), 6),
            "mean_cv_uncertainty": round(float(sub["cv_uncertainty"].mean()), 4),
            "mean_forecast_dispersion": round(float(sub["std_p_mean"].mean()), 4),
            "mean_alpha_dispersion": round(float(sub["alpha_dispersion"].mean()), 4),
            "mean_effective_rank": round(float(sub["effective_rank"].mean()), 2),
            "mean_condition_number": round(float(sub["condition_number"].mean()), 2),
            "mean_m0_hhi": round(float(sub["hhi_m0"].mean()), 5),
            "mean_m16_hhi": round(float(sub["hhi_m16"].mean()), 5),
            "mean_delta_hhi": round(float(sub["delta_hhi"].mean()), 5),
            "mean_capital_reallocated": round(float(sub["total_capital_reallocated"].mean()), 4),
            "mean_cap_removed_top2_unc": round(float(sub["cap_removed_top2_uncertainty"].mean()), 4),
            "mean_corr_m0_unc": round(float(sub["corr_m0_weight_uncertainty"].mean()), 4),
            "post_hoc_mean_forecast_error": round(float(sub["post_hoc_mean_forecast_error"].mean()), 4),
            "post_hoc_corr_unc_err": round(float(sub["post_hoc_corr_uncertainty_error"].mean()), 4),
            "post_hoc_return_dispersion": round(float(sub["post_hoc_realized_return_std"].mean()), 4),
            "post_hoc_error_reduction_share": round(float(sub["post_hoc_error_reduction_share"].mean()), 4),
        })

    return pd.DataFrame(rows)


def run_conditional_bucket_analysis(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Run conditional tertile analysis across primary clean regimes."""
    clean_sub = weekly_df[weekly_df["regime_type"] == "CLEAN_OOS"].copy()

    candidate_vars = [
        ("mean_uncertainty", "Uncertainty Level (Mean Sigma)", "EX_ANTE"),
        ("std_uncertainty", "Uncertainty Dispersion (Std Sigma)", "EX_ANTE"),
        ("cv_uncertainty", "Coefficient of Variation of Uncertainty", "EX_ANTE"),
        ("std_p_mean", "Forecast Downside Prob Dispersion", "EX_ANTE"),
        ("alpha_dispersion", "Alpha Signal Dispersion", "EX_ANTE"),
        ("effective_rank", "Covariance Effective Rank", "EX_ANTE"),
        ("condition_number", "Covariance Condition Number", "EX_ANTE"),
        ("hhi_m0", "M0 Concentration (HHI)", "EX_ANTE"),
        ("corr_m0_weight_uncertainty", "Corr(M0 Weight, Uncertainty)", "EX_ANTE"),
        ("total_capital_reallocated", "Total Capital Shifted by M16", "EX_ANTE"),
        ("cap_removed_top2_uncertainty", "Capital Removed from Top 2 Uncertainty Bins", "EX_ANTE"),
        ("post_hoc_realized_return_std", "Realized Cross-Sectional Return Std", "POST_HOC"),
        ("post_hoc_mean_forecast_error", "Realized Mean Forecast Error", "POST_HOC"),
        ("post_hoc_corr_uncertainty_error", "Correlation(Uncertainty, Forecast Error)", "POST_HOC"),
        ("post_hoc_error_reduction_share", "Share of Reduction in High-Error Names", "POST_HOC"),
    ]

    rows = []
    for col, label, var_type in candidate_vars:
        vals = clean_sub[col].values
        # Tertiles
        try:
            tertiles = pd.qcut(vals, q=3, labels=["Low", "Medium", "High"], duplicates="drop")
            clean_sub["bucket"] = tertiles
            for b_label in ["Low", "Medium", "High"]:
                b_sub = clean_sub[clean_sub["bucket"] == b_label]
                if len(b_sub) > 0:
                    rows.append({
                        "variable": col,
                        "variable_label": label,
                        "variable_type": var_type,
                        "bucket": b_label,
                        "n_weeks": len(b_sub),
                        "mean_delta_u_m16": round(float(b_sub["delta_u_m16"].mean()), 6),
                        "median_delta_u_m16": round(float(b_sub["delta_u_m16"].median()), 6),
                        "positive_weeks_pct": round(float(np.mean(b_sub["delta_u_m16"] > 0) * 100), 2),
                        "mean_delta_return": round(float(b_sub["delta_return"].mean()), 6),
                        "mean_delta_variance": round(float(b_sub["delta_variance"].mean()), 6),
                    })
        except Exception:
            continue

    return pd.DataFrame(rows)


def run_association_analysis(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman rank associations between weekly state variables and delta_u_m16."""
    clean_sub = weekly_df[weekly_df["regime_type"] == "CLEAN_OOS"].copy()
    seeds = [42, 123, 777]

    candidate_vars = [
        ("mean_uncertainty", "Uncertainty Level (Mean Sigma)", "EX_ANTE"),
        ("median_uncertainty", "Uncertainty Level (Median Sigma)", "EX_ANTE"),
        ("q90_uncertainty", "Uncertainty Level (Q90 Sigma)", "EX_ANTE"),
        ("std_uncertainty", "Uncertainty Dispersion (Std Sigma)", "EX_ANTE"),
        ("iqr_uncertainty", "Uncertainty Dispersion (IQR Sigma)", "EX_ANTE"),
        ("cv_uncertainty", "Coefficient of Variation of Uncertainty", "EX_ANTE"),
        ("std_p_mean", "Forecast Prob Dispersion", "EX_ANTE"),
        ("alpha_dispersion", "Alpha Signal Dispersion", "EX_ANTE"),
        ("effective_rank", "Covariance Effective Rank", "EX_ANTE"),
        ("condition_number", "Covariance Condition Number", "EX_ANTE"),
        ("mean_pairwise_correlation", "Mean Pairwise Correlation", "EX_ANTE"),
        ("hhi_m0", "M0 Concentration (HHI)", "EX_ANTE"),
        ("delta_hhi", "Delta HHI (M16 - M0)", "EX_ANTE"),
        ("corr_m0_weight_uncertainty", "Corr(M0 Weight, Uncertainty)", "EX_ANTE"),
        ("total_capital_reallocated", "Total Capital Shifted by M16", "EX_ANTE"),
        ("cap_removed_highest_uncertainty", "Capital Removed from Highest Uncertainty Bin", "EX_ANTE"),
        ("cap_removed_top2_uncertainty", "Capital Removed from Top 2 Uncertainty Bins", "EX_ANTE"),
        ("post_hoc_realized_return_std", "Realized Return Std", "POST_HOC"),
        ("post_hoc_realized_return_tail", "Realized Return Tail Spread", "POST_HOC"),
        ("post_hoc_mean_forecast_error", "Mean Forecast Error", "POST_HOC"),
        ("post_hoc_corr_uncertainty_error", "Correlation(Uncertainty, Realized Error)", "POST_HOC"),
        ("post_hoc_spearman_uncertainty_error", "Spearman(Uncertainty, Realized Error)", "POST_HOC"),
        ("post_hoc_error_reduction_share", "Capital Reduction in High-Error Names", "POST_HOC"),
    ]

    rows = []
    for col, label, var_type in candidate_vars:
        # Seed-specific rhos
        seed_rhos = {}
        for s in seeds:
            s_sub = clean_sub[clean_sub["seed"] == s]
            if len(s_sub) > 5 and np.std(s_sub[col]) > 1e-8:
                r_s, _ = stats.spearmanr(s_sub[col], s_sub["delta_u_m16"])
                seed_rhos[s] = float(r_s)
            else:
                seed_rhos[s] = 0.0

        # Pooled rho
        r_pool, p_val = stats.spearmanr(clean_sub[col], clean_sub["delta_u_m16"])

        rows.append({
            "variable": col,
            "variable_label": label,
            "variable_type": var_type,
            "spearman_rho_pooled": round(float(r_pool), 4),
            "p_value_pooled": round(float(p_val), 5),
            "rho_seed_42": round(seed_rhos[42], 4),
            "rho_seed_123": round(seed_rhos[123], 4),
            "rho_seed_777": round(seed_rhos[777], 4),
            "n_observations": len(clean_sub),
        })

    return pd.DataFrame(rows).sort_values("spearman_rho_pooled", key=abs, ascending=False).reset_index(drop=True)


def run_explanatory_model(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Fit standardized linear regression:

    delta_u_m16 ~ std_uncertainty + condition_number + std_p_mean + hhi_m0
    with Leave-One-Regime-Out (LORO) cross validation.
    """
    clean_sub = weekly_df[weekly_df["regime_type"] == "CLEAN_OOS"].copy()
    feature_cols = [
        "std_uncertainty",
        "condition_number",
        "std_p_mean",
        "hhi_m0",
    ]

    # Standardize inputs
    X_raw = clean_sub[feature_cols].values
    y_raw = clean_sub["delta_u_m16"].values

    mu_x = np.mean(X_raw, axis=0)
    sig_x = np.std(X_raw, axis=0, ddof=1)
    sig_x[sig_x < 1e-8] = 1.0
    X_std = (X_raw - mu_x) / sig_x

    # Fit full model
    X_full = np.column_stack([np.ones(len(X_std)), X_std])
    beta_full, _, _, _ = np.linalg.lstsq(X_full, y_raw, rcond=None)
    y_pred_full = X_full @ beta_full
    r2_full = 1.0 - (np.sum((y_raw - y_pred_full)**2) / np.sum((y_raw - np.mean(y_raw))**2))

    rows = [{
        "model": "Full Clean Regimes (2022-2025)",
        "held_out_regime": "None",
        "intercept": round(float(beta_full[0]), 6),
        "beta_std_uncertainty": round(float(beta_full[1]), 6),
        "beta_condition_number": round(float(beta_full[2]), 6),
        "beta_std_p_mean": round(float(beta_full[3]), 6),
        "beta_hhi_m0": round(float(beta_full[4]), 6),
        "train_r2": round(float(r2_full), 4),
        "test_r2": round(float(r2_full), 4),
        "test_rank_corr": round(float(stats.spearmanr(y_pred_full, y_raw)[0]), 4),
    }]

    # Leave-One-Regime-Out (LORO)
    clean_regimes = ["2022", "2023", "2024-2025"]
    for test_reg in clean_regimes:
        train_mask = clean_sub["regime"] != test_reg
        test_mask = clean_sub["regime"] == test_reg

        X_train_raw = clean_sub.loc[train_mask, feature_cols].values
        y_train = clean_sub.loc[train_mask, "delta_u_m16"].values
        X_test_raw = clean_sub.loc[test_mask, feature_cols].values
        y_test = clean_sub.loc[test_mask, "delta_u_m16"].values

        mu_tr = np.mean(X_train_raw, axis=0)
        sig_tr = np.std(X_train_raw, axis=0, ddof=1)
        sig_tr[sig_tr < 1e-8] = 1.0

        X_tr = np.column_stack([np.ones(len(X_train_raw)), (X_train_raw - mu_tr) / sig_tr])
        X_te = np.column_stack([np.ones(len(X_test_raw)), (X_test_raw - mu_tr) / sig_tr])

        beta_loro, _, _, _ = np.linalg.lstsq(X_tr, y_train, rcond=None)
        y_pred_train = X_tr @ beta_loro
        y_pred_test = X_te @ beta_loro

        r2_tr = 1.0 - (np.sum((y_train - y_pred_train)**2) / np.sum((y_train - np.mean(y_train))**2))
        ss_tot_te = np.sum((y_test - np.mean(y_test))**2)
        r2_te = 1.0 - (np.sum((y_test - y_pred_test)**2) / ss_tot_te) if ss_tot_te > 1e-12 else 0.0
        rank_te, _ = stats.spearmanr(y_pred_test, y_test)

        rows.append({
            "model": f"LORO (Train on others, test on {test_reg})",
            "held_out_regime": test_reg,
            "intercept": round(float(beta_loro[0]), 6),
            "beta_std_uncertainty": round(float(beta_loro[1]), 6),
            "beta_condition_number": round(float(beta_loro[2]), 6),
            "beta_std_p_mean": round(float(beta_loro[3]), 6),
            "beta_hhi_m0": round(float(beta_loro[4]), 6),
            "train_r2": round(float(r2_tr), 4),
            "test_r2": round(float(r2_te), 4),
            "test_rank_corr": round(float(rank_te), 4) if pd.notna(rank_te) else 0.0,
        })

    return pd.DataFrame(rows)


def main() -> None:
    print("=== Phase 14: Mechanism Decomposition of Uncertainty-Aware Risk Inflation ===", flush=True)

    out_dir = repo_root / "artifacts" / "phase14_m16_mechanism"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_preds_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "evaluation_predictions.csv"
    assert eval_preds_path.exists(), f"Missing evaluation predictions: {eval_preds_path}"
    print(f"Loading predictions from: {eval_preds_path}", flush=True)
    eval_preds_df = pd.read_csv(eval_preds_path)

    rec_path = repo_root / "data" / "bayesian_lora" / "reconstructed_returns.csv"
    assert rec_path.exists(), f"Missing reconstructed returns: {rec_path}"

    # Step 1: Simulate and extract week-level state & mechanism variables
    print("Simulating portfolios and extracting weekly mechanism variables...", flush=True)
    weekly_df, weight_shift_df, error_alignment_df = simulate_and_extract_weekly_mechanisms(
        eval_preds_df=eval_preds_df,
        rec_path=rec_path,
        horizon=4,
        cost_bps=0.0010,
        gamma=1.0,
    )

    # Save week-level dataset
    weekly_path = out_dir / "phase14_weekly_mechanism.csv"
    weekly_df.to_csv(weekly_path, index=False)
    print(f"Saved weekly mechanism dataset to: {weekly_path} ({len(weekly_df)} rows)")

    # Save weight shift by uncertainty quintile
    shift_path = out_dir / "phase14_weight_shift_by_uncertainty.csv"
    weight_shift_agg = weight_shift_df.groupby(["regime", "regime_type", "quintile"]).mean(numeric_only=True).reset_index()
    weight_shift_agg.to_csv(shift_path, index=False)
    print(f"Saved weight shift by quintile to: {shift_path}")

    # Save error alignment diagnostic
    align_path = out_dir / "phase14_error_alignment.csv"
    error_alignment_df.to_csv(align_path, index=False)
    print(f"Saved error alignment diagnostic to: {align_path}")

    # Step 2: Regime Comparison Table
    print("\nComputing regime comparison summary...", flush=True)
    reg_comp_df = compute_regime_comparison(weekly_df)
    reg_comp_path = out_dir / "phase14_regime_comparison.csv"
    reg_comp_df.to_csv(reg_comp_path, index=False)
    print(f"Saved regime comparison to: {reg_comp_path}")

    # Step 3: Conditional Bucket Analysis
    print("Running conditional tertile bucket analysis...", flush=True)
    bucket_df = run_conditional_bucket_analysis(weekly_df)
    bucket_path = out_dir / "phase14_bucket_analysis.csv"
    bucket_df.to_csv(bucket_path, index=False)
    print(f"Saved conditional bucket analysis to: {bucket_path}")

    # Step 4: Association Analysis
    print("Running Spearman rank correlation association analysis...", flush=True)
    corr_df = run_association_analysis(weekly_df)
    corr_path = out_dir / "phase14_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"Saved association analysis to: {corr_path}")

    # Step 5: Small Explanatory Model
    print("Fitting standardized linear explanatory model with LORO...", flush=True)
    model_df = run_explanatory_model(weekly_df)
    model_path = out_dir / "phase14_explanatory_model.csv"
    model_df.to_csv(model_path, index=False)
    print(f"Saved explanatory model to: {model_path}")

    # Step 6: 2023 vs Non-2023 Forensic Comparison
    clean_sub = weekly_df[weekly_df["regime_type"] == "CLEAN_OOS"].copy()
    w_2023 = clean_sub[clean_sub["regime"] == "2023"]
    w_other = clean_sub[clean_sub["regime"] != "2023"]

    diff_summary = {
        "2023_mean_delta_u": float(w_2023["delta_u_m16"].mean()),
        "non_2023_mean_delta_u": float(w_other["delta_u_m16"].mean()),
        "2023_pct_positive_weeks": float(np.mean(w_2023["delta_u_m16"] > 0) * 100),
        "non_2023_pct_positive_weeks": float(np.mean(w_other["delta_u_m16"] > 0) * 100),
        "uncertainty_dispersion_2023": float(w_2023["std_uncertainty"].mean()),
        "uncertainty_dispersion_other": float(w_other["std_uncertainty"].mean()),
        "forecast_dispersion_2023": float(w_2023["std_p_mean"].mean()),
        "forecast_dispersion_other": float(w_other["std_p_mean"].mean()),
        "effective_rank_2023": float(w_2023["effective_rank"].mean()),
        "effective_rank_other": float(w_other["effective_rank"].mean()),
        "m0_hhi_2023": float(w_2023["hhi_m0"].mean()),
        "m0_hhi_other": float(w_other["hhi_m0"].mean()),
        "post_hoc_return_dispersion_2023": float(w_2023["post_hoc_realized_return_std"].mean()),
        "post_hoc_return_dispersion_other": float(w_other["post_hoc_realized_return_std"].mean()),
        "post_hoc_error_reduction_share_2023": float(w_2023["post_hoc_error_reduction_share"].mean()),
        "post_hoc_error_reduction_share_other": float(w_other["post_hoc_error_reduction_share"].mean()),
    }

    # Step 7: Build Summary JSON
    summary_data = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": "Phase 14: Mechanism Decomposition of Uncertainty-Aware Risk Inflation",
        "frozen_strategy": {
            "strategy": "M16",
            "lambda_u": 0.10,
            "status": "FROZEN_UNTOUCHED",
            "horizon_H": 4,
            "transaction_cost_bps": 10.0,
            "gamma": 1.0,
        },
        "universe": {
            "primary_clean_regimes": ["2022", "2023", "2024-2025"],
            "total_clean_weeks_observations": len(clean_sub),
            "secondary_diagnostic_regimes": ["2020", "2021"],
            "total_diagnostic_weeks_observations": len(weekly_df[weekly_df["regime_type"] != "CLEAN_OOS"]),
        },
        "regime_performance_reconciliation": {
            "2022_mean_delta_u": float(reg_comp_df[reg_comp_df["regime"] == "2022"]["mean_delta_u_m16"].iloc[0]),
            "2023_mean_delta_u": float(reg_comp_df[reg_comp_df["regime"] == "2023"]["mean_delta_u_m16"].iloc[0]),
            "2024_2025_mean_delta_u": float(reg_comp_df[reg_comp_df["regime"] == "2024-2025"]["mean_delta_u_m16"].iloc[0]),
        },
        "structural_2023_vs_other_decomposition": diff_summary,
        "top_ex_ante_associations": corr_df[corr_df["variable_type"] == "EX_ANTE"].head(5).to_dict(orient="records"),
        "top_post_hoc_associations": corr_df[corr_df["variable_type"] == "POST_HOC"].head(5).to_dict(orient="records"),
        "explanatory_model_summary": model_df.to_dict(orient="records"),
    }

    sum_path = out_dir / "phase14_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved phase 14 summary JSON to: {sum_path}")
    print("\nPhase 14 computation completed successfully.", flush=True)


if __name__ == "__main__":
    main()
