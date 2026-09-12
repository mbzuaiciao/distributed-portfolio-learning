#!/usr/bin/env python3
"""Phase 15: Incremental Value of Bayesian Epistemic Uncertainty.

Investigates whether Bayesian epistemic uncertainty contains forecast-reliability
information that simpler confidence (entropy, margin) or market-risk proxies do not.

Strictly diagnostic & information-content study:
- No portfolio strategy search or new trading policy (no M17).
- Primary clean sample: 2022, 2023, 2024-2025 (N=4,100 clean events x 3 seeds = 12,300 rows).
- Secondary diagnostic sample: 2020, 2021 (N=2,389 events x 3 seeds = 7,167 rows).
- Target variable: Absolute forecast error e_i = |p_hat_i - y_i|, and binary top-20% high error.
- Evaluation via Leave-One-Regime-Out (LORO), residual regressions, and week-clustered block bootstrap.
"""

from __future__ import annotations

import datetime
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

repo_root = Path(__file__).resolve().parent.parent


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute exact Area Under the ROC curve via Mann-Whitney U statistic."""
    y_b = np.asarray(y_true, dtype=bool)
    n1 = int(np.sum(y_b))
    n0 = len(y_b) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = stats.rankdata(scores)
    r1 = np.sum(ranks[y_b])
    u = r1 - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def compute_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute Average Precision (area under precision-recall curve)."""
    y_b = np.asarray(y_true, dtype=bool)
    n1 = int(np.sum(y_b))
    if n1 == 0:
        return 0.0
    order = np.argsort(-scores)
    y_sorted = y_b[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    precision = tp / (tp + fp)
    return float(np.sum(precision * y_sorted) / n1)


def fit_ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Fit Ridge regression with unpenalized intercept."""
    d = X.shape[1]
    reg_matrix = alpha * np.eye(d)
    reg_matrix[0, 0] = 0.0  # Do not regularize intercept
    return np.linalg.solve(X.T @ X + reg_matrix, X.T @ y)


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, l2_reg: float = 1.0) -> np.ndarray:
    """Fit L2-regularized Logistic Regression via L-BFGS."""
    _, d = X.shape

    def objective_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = np.clip(X @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        loss = -np.sum(y * np.log(p + 1e-15) + (1.0 - y) * np.log(1.0 - p + 1e-15)) + 0.5 * l2_reg * np.sum(w[1:] ** 2)
        grad = X.T @ (p - y)
        grad[1:] += l2_reg * w[1:]
        return float(loss), grad

    w0 = np.zeros(d)
    res = optimize.minimize(objective_and_grad, w0, jac=True, method="L-BFGS-B", options={"maxiter": 150})
    return res.x


def build_evaluation_features(
    preds_df: pd.DataFrame,
    returns_path: Path,
) -> pd.DataFrame:
    """Merge predictions with realized outcomes and compute all Bayesian, confidence, and market signals."""
    rets_df = pd.read_csv(returns_path)
    rets_df["dt"] = pd.to_datetime(rets_df["date"])
    rets_df = rets_df.sort_values("dt").reset_index(drop=True)

    # Compute trailing asset volatility (ex-ante, shifted to avoid lookahead)
    # Expanding standard deviation of prior moves for each ticker
    rets_df["prev_move"] = rets_df.groupby("symbol")["move_1d"].shift(1)
    trailing_vol = rets_df.groupby("symbol")["prev_move"].expanding().std().reset_index(level=0, drop=True)
    symbol_overall_std = rets_df.groupby("symbol")["move_1d"].transform("std")
    median_vol = float(rets_df["move_1d"].std())

    rets_df["asset_vol"] = trailing_vol.fillna(symbol_overall_std).fillna(median_vol)

    # Map returns and volatility onto predictions
    ret_dict = {str(r["episode_id"]): float(r["move_1d"]) for _, r in rets_df.iterrows() if pd.notna(r["move_1d"])}
    vol_dict = {str(r["episode_id"]): float(r["asset_vol"]) for _, r in rets_df.iterrows() if pd.notna(r["asset_vol"])}

    df = preds_df.copy()
    df["move_1d"] = df["event_id"].map(ret_dict)
    df["asset_volatility"] = df["event_id"].map(vol_dict).fillna(median_vol)

    # Calendar week identifier
    df["dt"] = pd.to_datetime(df["date"])
    df["calendar_week"] = df["dt"].apply(lambda d: f"{d.year}-W{d.isocalendar()[1]:02d}")

    # Market volatility proxy: standard deviation of earnings moves within each calendar week
    week_vol_map = df.groupby("calendar_week")["move_1d"].std().to_dict()
    df["market_vol_proxy"] = df["calendar_week"].map(week_vol_map).fillna(median_vol)

    # Cross-sectional volatility rank within calendar week
    df["cross_sectional_vol_rank"] = df.groupby("calendar_week")["asset_volatility"].rank(pct=True)

    # Binary downside target threshold (clean 20th percentile move_1d)
    clean_mask = df["regime"].isin(["2022", "2023", "2024-2025"])
    clean_thresh = float(df.loc[clean_mask, "move_1d"].quantile(0.20))
    df["y_true"] = (df["move_1d"] <= clean_thresh).astype(int)

    # Primary target: Absolute Forecast Error
    p = np.clip(df["posterior_mean"].values, 1e-7, 1.0 - 1e-7)
    df["forecast_error"] = np.abs(p - df["y_true"].values)

    # Secondary target: Binary Top-20% High Error Indicator (within clean sample)
    clean_err_thresh = float(df.loc[clean_mask, "forecast_error"].quantile(0.80))
    df["high_error_binary"] = (df["forecast_error"] >= clean_err_thresh).astype(int)

    # 1. Bayesian Uncertainty Signals
    df["epistemic_std"] = np.sqrt(np.maximum(df["epistemic_uncertainty"].values, 0.0))

    # 2. Simple Model-Confidence Baselines (deterministic functions of p)
    # Distance from decision boundary: |p - 0.5|. Low margin = high uncertainty.
    # We use inv_confidence_margin = 0.5 - |p - 0.5| = min(p, 1-p) so higher value = higher uncertainty.
    df["confidence_margin"] = np.abs(p - 0.5)
    df["inv_confidence_margin"] = 0.5 - np.abs(p - 0.5)
    df["probability_extremeness"] = np.minimum(p, 1.0 - p)
    df["predictive_entropy"] = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
    df["deterministic_variance"] = p * (1.0 - p)

    return df


def evaluate_univariate_signals(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Evaluate univariate predictive performance of all candidate signals against forecast error."""
    signals = [
        ("epistemic_uncertainty", "Bayesian Epistemic Uncertainty (Variance)", "BAYESIAN_UNCERTAINTY"),
        ("epistemic_std", "Bayesian Epistemic Uncertainty (Std Dev)", "BAYESIAN_UNCERTAINTY"),
        ("total_uncertainty", "Bayesian Total Uncertainty", "BAYESIAN_UNCERTAINTY"),
        ("aleatoric_uncertainty", "Bayesian Aleatoric Uncertainty", "BAYESIAN_UNCERTAINTY"),
        ("predictive_entropy", "Predictive Entropy H(p)", "SIMPLE_CONFIDENCE"),
        ("inv_confidence_margin", "Inverse Confidence Margin (0.5 - |p - 0.5|)", "SIMPLE_CONFIDENCE"),
        ("probability_extremeness", "Probability Extremeness min(p, 1-p)", "SIMPLE_CONFIDENCE"),
        ("deterministic_variance", "Deterministic Variance p(1-p)", "SIMPLE_CONFIDENCE"),
        ("asset_volatility", "Asset Volatility (Historical Trailing Std)", "MARKET_VOLATILITY"),
        ("market_vol_proxy", "Market Volatility Proxy (Weekly Cross-Section)", "MARKET_VOLATILITY"),
        ("cross_sectional_vol_rank", "Cross-Sectional Volatility Rank", "MARKET_VOLATILITY"),
    ]

    y_err = df["forecast_error"].values
    y_high = df["high_error_binary"].values
    n_obs = len(df)

    rows = []
    for col, sig_desc, cat in signals:
        scores = df[col].values
        # Spearman & Pearson
        rho, p_spear = stats.spearmanr(scores, y_err)
        r_p, p_pear = stats.pearsonr(scores, y_err)

        # AUROC and Average Precision for top-20% high error
        auroc = compute_auroc(y_high, scores)
        ap = compute_average_precision(y_high, scores)

        # Q5 / Q1 error ratio
        try:
            q_bins = pd.qcut(scores, q=5, labels=False, duplicates="drop")
            e_q1 = float(y_err[q_bins == 0].mean())
            e_q5 = float(y_err[q_bins == np.max(q_bins)].mean())
            q5_q1_ratio = float(e_q5 / (e_q1 + 1e-12))
        except Exception:
            q5_q1_ratio = 1.0

        rows.append({
            "scope": label,
            "signal_name": col,
            "signal_description": sig_desc,
            "signal_category": cat,
            "spearman_rho": round(float(rho), 4),
            "p_value_spearman": float(f"{p_spear:.2e}"),
            "pearson_r": round(float(r_p), 4),
            "p_value_pearson": float(f"{p_pear:.2e}"),
            "q5_q1_error_ratio": round(q5_q1_ratio, 4),
            "high_error_auroc": round(auroc, 4),
            "high_error_ap": round(ap, 4),
            "n_observations": n_obs,
        })

    return pd.DataFrame(rows)


def run_loro_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Run Leave-One-Regime-Out (LORO) predictive tests comparing nested models A, B, C, D."""
    clean_df = df[df["regime"].isin(["2022", "2023", "2024-2025"])].copy()

    model_features = {
        "Model A (Confidence Only)": ["predictive_entropy", "inv_confidence_margin", "deterministic_variance"],
        "Model B (Market Only)": ["asset_volatility", "market_vol_proxy", "cross_sectional_vol_rank"],
        "Model C (Confidence + Market)": [
            "predictive_entropy",
            "inv_confidence_margin",
            "deterministic_variance",
            "asset_volatility",
            "market_vol_proxy",
            "cross_sectional_vol_rank",
        ],
        "Model D (Confidence + Market + Epistemic)": [
            "predictive_entropy",
            "inv_confidence_margin",
            "deterministic_variance",
            "asset_volatility",
            "market_vol_proxy",
            "cross_sectional_vol_rank",
            "epistemic_uncertainty",
        ],
    }

    clean_regimes = ["2022", "2023", "2024-2025"]
    rows = []

    # Store predictions for pooled out-of-sample evaluation
    pooled_preds: dict[str, list[float]] = defaultdict(list)
    pooled_probs: dict[str, list[float]] = defaultdict(list)
    pooled_y_err: list[float] = []
    pooled_y_high: list[int] = []

    for test_reg in clean_regimes:
        train_mask = clean_df["regime"] != test_reg
        test_mask = clean_df["regime"] == test_reg

        y_train_err = clean_df.loc[train_mask, "forecast_error"].values
        y_test_err = clean_df.loc[test_mask, "forecast_error"].values

        y_train_high = clean_df.loc[train_mask, "high_error_binary"].values
        y_test_high = clean_df.loc[test_mask, "high_error_binary"].values

        reg_results: dict[str, dict[str, float]] = {}

        for m_name, feats in model_features.items():
            X_train_raw = clean_df.loc[train_mask, feats].values
            X_test_raw = clean_df.loc[test_mask, feats].values

            # Standardize using training statistics
            mu_x = np.mean(X_train_raw, axis=0)
            sig_x = np.std(X_train_raw, axis=0, ddof=1)
            sig_x[sig_x < 1e-8] = 1.0

            X_tr = np.column_stack([np.ones(len(X_train_raw)), (X_train_raw - mu_x) / sig_x])
            X_te = np.column_stack([np.ones(len(X_test_raw)), (X_test_raw - mu_x) / sig_x])

            # 1. Continuous error regression (Ridge)
            w_ridge = fit_ridge_regression(X_tr, y_train_err, alpha=1.0)
            y_pred_te = X_te @ w_ridge

            mae = float(np.mean(np.abs(y_test_err - y_pred_te)))
            ss_tot = np.sum((y_test_err - np.mean(y_test_err)) ** 2)
            r2 = float(1.0 - (np.sum((y_test_err - y_pred_te) ** 2) / ss_tot)) if ss_tot > 1e-12 else 0.0

            # 2. Binary high error classification (Logistic)
            w_logit = fit_logistic_regression(X_tr, y_train_high, l2_reg=1.0)
            z_te = np.clip(X_te @ w_logit, -30.0, 30.0)
            prob_te = 1.0 / (1.0 + np.exp(-z_te))

            auroc = compute_auroc(y_test_high, prob_te)
            ap = compute_average_precision(y_test_high, prob_te)

            reg_results[m_name] = {
                "mae": mae,
                "r2": r2,
                "auroc": auroc,
                "ap": ap,
            }

            # Collect for pooled out-of-sample
            pooled_preds[m_name].extend(y_pred_te)
            pooled_probs[m_name].extend(prob_te)

        pooled_y_err.extend(y_test_err)
        pooled_y_high.extend(y_test_high)

        # Record fold metrics
        base_c = reg_results["Model C (Confidence + Market)"]
        for m_name, res in reg_results.items():
            delta_r2 = res["r2"] - base_c["r2"]
            delta_mae = base_c["mae"] - res["mae"]  # Positive means error reduced
            delta_auroc = res["auroc"] - base_c["auroc"]
            delta_ap = res["ap"] - base_c["ap"]

            rows.append({
                "evaluation_scope": f"LORO_HeldOut_{test_reg}",
                "held_out_regime": test_reg,
                "model_name": m_name,
                "features": ", ".join(model_features[m_name]),
                "held_out_r2": round(res["r2"], 5),
                "held_out_mae": round(res["mae"], 5),
                "held_out_auroc": round(res["auroc"], 4),
                "held_out_ap": round(res["ap"], 4),
                "delta_r2_vs_model_c": round(delta_r2, 5),
                "delta_mae_vs_model_c": round(delta_mae, 5),
                "delta_auroc_vs_model_c": round(delta_auroc, 4),
                "delta_ap_vs_model_c": round(delta_ap, 4),
                "n_test_observations": len(y_test_err),
            })

    # Record Pooled Out-of-Sample Results
    y_err_all = np.array(pooled_y_err)
    y_high_all = np.array(pooled_y_high)
    ss_tot_all = np.sum((y_err_all - np.mean(y_err_all)) ** 2)

    pooled_results = {}
    for m_name in model_features:
        preds = np.array(pooled_preds[m_name])
        probs = np.array(pooled_probs[m_name])

        mae_all = float(np.mean(np.abs(y_err_all - preds)))
        r2_all = float(1.0 - (np.sum((y_err_all - preds) ** 2) / ss_tot_all))
        auroc_all = compute_auroc(y_high_all, probs)
        ap_all = compute_average_precision(y_high_all, probs)

        pooled_results[m_name] = {
            "mae": mae_all,
            "r2": r2_all,
            "auroc": auroc_all,
            "ap": ap_all,
        }

    base_c_all = pooled_results["Model C (Confidence + Market)"]
    for m_name, res in pooled_results.items():
        delta_r2 = res["r2"] - base_c_all["r2"]
        delta_mae = base_c_all["mae"] - res["mae"]
        delta_auroc = res["auroc"] - base_c_all["auroc"]
        delta_ap = res["ap"] - base_c_all["ap"]

        rows.append({
            "evaluation_scope": "LORO_Pooled_Out_of_Sample",
            "held_out_regime": "All_Clean_Regimes",
            "model_name": m_name,
            "features": ", ".join(model_features[m_name]),
            "held_out_r2": round(res["r2"], 5),
            "held_out_mae": round(res["mae"], 5),
            "held_out_auroc": round(res["auroc"], 4),
            "held_out_ap": round(res["ap"], 4),
            "delta_r2_vs_model_c": round(delta_r2, 5),
            "delta_mae_vs_model_c": round(delta_mae, 5),
            "delta_auroc_vs_model_c": round(delta_auroc, 4),
            "delta_ap_vs_model_c": round(delta_ap, 4),
            "n_test_observations": len(y_err_all),
        })

    return pd.DataFrame(rows)


def run_residual_information_test(df: pd.DataFrame) -> pd.DataFrame:
    """Regress epistemic uncertainty on simple confidence proxies and test whether residual predicts error."""
    conf_features = ["predictive_entropy", "inv_confidence_margin", "deterministic_variance"]

    scopes = [
        ("Clean_Pooled (2022-2025)", df["regime"].isin(["2022", "2023", "2024-2025"])),
        ("2022", df["regime"] == "2022"),
        ("2023", df["regime"] == "2023"),
        ("2024-2025", df["regime"] == "2024-2025"),
        ("Seed_42 (Clean)", (df["regime"].isin(["2022", "2023", "2024-2025"])) & (df["seed"] == 42)),
        ("Seed_123 (Clean)", (df["regime"].isin(["2022", "2023", "2024-2025"])) & (df["seed"] == 123)),
        ("Seed_777 (Clean)", (df["regime"].isin(["2022", "2023", "2024-2025"])) & (df["seed"] == 777)),
    ]

    rows = []
    for scope_name, mask in scopes:
        sub = df[mask].copy()
        X_raw = sub[conf_features].values
        y_epi = sub["epistemic_uncertainty"].values
        y_err = sub["forecast_error"].values
        y_high = sub["high_error_binary"].values

        # Fit OLS
        X = np.column_stack([np.ones(len(X_raw)), X_raw])
        w_ols, _, _, _ = np.linalg.lstsq(X, y_epi, rcond=None)
        pred_epi = X @ w_ols
        res_epi = y_epi - pred_epi

        # Variance explained by simple confidence
        ss_tot = np.sum((y_epi - np.mean(y_epi)) ** 2)
        r2_conf = float(1.0 - (np.sum(res_epi**2) / ss_tot)) if ss_tot > 1e-12 else 0.0

        # Evaluate residual epistemic uncertainty against error
        rho_res, p_spear = stats.spearmanr(res_epi, y_err)
        r_res, p_pear = stats.pearsonr(res_epi, y_err)
        auroc_res = compute_auroc(y_high, res_epi)
        ap_res = compute_average_precision(y_high, res_epi)

        # Baseline raw epistemic correlation for comparison
        rho_raw, _ = stats.spearmanr(y_epi, y_err)
        auroc_raw = compute_auroc(y_high, y_epi)

        rows.append({
            "sample_scope": scope_name,
            "r2_explained_by_simple_confidence": round(r2_conf, 4),
            "raw_epistemic_spearman_rho": round(float(rho_raw), 4),
            "residual_epistemic_spearman_rho": round(float(rho_res), 4),
            "residual_p_value_spearman": float(f"{p_spear:.2e}"),
            "residual_pearson_r": round(float(r_res), 4),
            "residual_p_value_pearson": float(f"{p_pear:.2e}"),
            "raw_epistemic_auroc": round(auroc_raw, 4),
            "residual_epistemic_auroc": round(auroc_res, 4),
            "residual_epistemic_ap": round(ap_res, 4),
            "n_observations": len(sub),
        })

    return pd.DataFrame(rows)


def run_quintile_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze forecast error calibration across quintiles for candidate signals."""
    clean_df = df[df["regime"].isin(["2022", "2023", "2024-2025"])].copy()

    signals = [
        ("epistemic_uncertainty", "Bayesian Epistemic Uncertainty"),
        ("predictive_entropy", "Predictive Entropy H(p)"),
        ("inv_confidence_margin", "Inverse Confidence Margin"),
        ("asset_volatility", "Historical Asset Volatility"),
    ]

    rows = []
    for col, sig_desc in signals:
        clean_df["q_bin"] = pd.qcut(clean_df[col], q=5, labels=["Q1_Lowest", "Q2", "Q3", "Q4", "Q5_Highest"], duplicates="drop")
        unique_bins = ["Q1_Lowest", "Q2", "Q3", "Q4", "Q5_Highest"]

        errors_by_q = []
        for q_label in unique_bins:
            sub = clean_df[clean_df["q_bin"] == q_label]
            mean_err = float(sub["forecast_error"].mean())
            errors_by_q.append(mean_err)

            rows.append({
                "signal_name": col,
                "signal_description": sig_desc,
                "quintile": q_label,
                "n_events": len(sub),
                "mean_signal_value": round(float(sub[col].mean()), 6),
                "mean_forecast_error": round(mean_err, 4),
                "std_forecast_error": round(float(sub["forecast_error"].std()), 4),
                "top20_error_frequency": round(float(sub["high_error_binary"].mean() * 100), 2),
            })

        # Check monotonicity
        is_monotonic = bool(errors_by_q == sorted(errors_by_q))
        q5_q1 = float(errors_by_q[-1] / (errors_by_q[0] + 1e-12))
        for r in rows[-5:]:
            r["q5_q1_ratio"] = round(q5_q1, 4)
            r["is_monotonic_progression"] = is_monotonic

    return pd.DataFrame(rows)


def run_dependence_aware_bootstrap(
    df: pd.DataFrame,
    n_iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run week-clustered block bootstrap to construct 95% confidence intervals preserving dependence."""
    clean_df = df[df["regime"].isin(["2022", "2023", "2024-2025"])].copy()
    unique_weeks = np.array(clean_df["calendar_week"].unique())
    n_weeks = len(unique_weeks)

    np.random.seed(seed)

    # Pre-index data by week to enable ultra-fast resampling
    week_groups = {w: clean_df[clean_df["calendar_week"] == w] for w in unique_weeks}

    delta_rhos = []
    delta_aurocs_univariate = []
    delta_aurocs_model = []
    delta_maes_model = []

    model_c_feats = [
        "predictive_entropy",
        "inv_confidence_margin",
        "deterministic_variance",
        "asset_volatility",
        "market_vol_proxy",
        "cross_sectional_vol_rank",
    ]
    model_d_feats = model_c_feats + ["epistemic_uncertainty"]

    for b in range(n_iterations):
        # Sample calendar weeks with replacement
        sampled_weeks = np.random.choice(unique_weeks, size=n_weeks, replace=True)
        boot_sub = pd.concat([week_groups[w] for w in sampled_weeks], ignore_index=True)

        y_err = boot_sub["forecast_error"].values
        y_high = boot_sub["high_error_binary"].values

        epi_scores = boot_sub["epistemic_uncertainty"].values
        ent_scores = boot_sub["predictive_entropy"].values

        # 1. Delta Spearman Rho (Epistemic - Entropy)
        rho_epi, _ = stats.spearmanr(epi_scores, y_err)
        rho_ent, _ = stats.spearmanr(ent_scores, y_err)
        delta_rhos.append(float(rho_epi - rho_ent))

        # 2. Delta AUROC (Epistemic - Entropy)
        auc_epi = compute_auroc(y_high, epi_scores)
        auc_ent = compute_auroc(y_high, ent_scores)
        delta_aurocs_univariate.append(float(auc_epi - auc_ent))

        # 3. Incremental Model D vs Model C (Ridge & Logistic on sampled data)
        # Fast evaluation on bootstrap sample
        Xc_raw = boot_sub[model_c_feats].values
        Xd_raw = boot_sub[model_d_feats].values

        mu_c = np.mean(Xc_raw, axis=0)
        sig_c = np.std(Xc_raw, axis=0, ddof=1)
        sig_c[sig_c < 1e-8] = 1.0
        Xc = np.column_stack([np.ones(len(Xc_raw)), (Xc_raw - mu_c) / sig_c])

        mu_d = np.mean(Xd_raw, axis=0)
        sig_d = np.std(Xd_raw, axis=0, ddof=1)
        sig_d[sig_d < 1e-8] = 1.0
        Xd = np.column_stack([np.ones(len(Xd_raw)), (Xd_raw - mu_d) / sig_d])

        # Ridge
        wc = fit_ridge_regression(Xc, y_err, alpha=1.0)
        wd = fit_ridge_regression(Xd, y_err, alpha=1.0)
        mae_c = float(np.mean(np.abs(y_err - Xc @ wc)))
        mae_d = float(np.mean(np.abs(y_err - Xd @ wd)))
        delta_maes_model.append(float(mae_c - mae_d))  # Positive = Model D reduced MAE

        # Logistic
        wc_logit = fit_logistic_regression(Xc, y_high, l2_reg=1.0)
        wd_logit = fit_logistic_regression(Xd, y_high, l2_reg=1.0)
        prob_c = 1.0 / (1.0 + np.exp(-np.clip(Xc @ wc_logit, -30.0, 30.0)))
        prob_d = 1.0 / (1.0 + np.exp(-np.clip(Xd @ wd_logit, -30.0, 30.0)))
        auc_c = compute_auroc(y_high, prob_c)
        auc_d = compute_auroc(y_high, prob_d)
        delta_aurocs_model.append(float(auc_d - auc_c))

    # Compile interval summary
    comparisons = [
        ("Delta_Rho_Univariate (Epistemic - Entropy)", delta_rhos),
        ("Delta_AUROC_Univariate (Epistemic - Entropy)", delta_aurocs_univariate),
        ("Delta_AUROC_Incremental (Model D - Model C)", delta_aurocs_model),
        ("Delta_MAE_Incremental (Model C - Model D)", delta_maes_model),
    ]

    rows = []
    for metric_name, dist in comparisons:
        arr = np.array(dist)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1))
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        # Two-tailed empirical p-value for zero inclusion
        p_val = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
        is_sig = bool(ci_lower > 0.0 or ci_upper < 0.0)

        rows.append({
            "comparison_metric": metric_name,
            "bootstrap_mean": round(mean_val, 5),
            "bootstrap_std": round(std_val, 5),
            "ci_95_lower": round(ci_lower, 5),
            "ci_95_upper": round(ci_upper, 5),
            "empirical_p_value": round(p_val, 4),
            "statistically_significant_at_05": is_sig,
            "n_resamples": n_iterations,
        })

    return pd.DataFrame(rows)


def main() -> None:
    print("=== Phase 15: Incremental Value of Bayesian Epistemic Uncertainty ===", flush=True)

    out_dir = repo_root / "artifacts" / "phase15_incremental_uncertainty"
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "evaluation_predictions.csv"
    assert preds_path.exists(), f"Missing predictions file: {preds_path}"
    preds_df = pd.read_csv(preds_path)

    returns_path = repo_root / "data" / "bayesian_lora" / "reconstructed_returns.csv"
    assert returns_path.exists(), f"Missing reconstructed returns: {returns_path}"

    # Step 1: Feature Construction
    print("Building evaluation features and risk baselines...", flush=True)
    df = build_evaluation_features(preds_df, returns_path)
    clean_df = df[df["regime"].isin(["2022", "2023", "2024-2025"])].copy()
    print(f"Total dataset: {len(df)} rows ({len(clean_df)} primary clean observations).", flush=True)

    # Step 2: Univariate Signal Comparisons
    print("\n1. Running univariate signal comparisons on clean pooled sample...", flush=True)
    sig_comp_df = evaluate_univariate_signals(clean_df, label="Clean_Pooled (2022-2025)")
    sig_comp_path = out_dir / "signal_comparison.csv"
    sig_comp_df.to_csv(sig_comp_path, index=False)
    print(f"Saved signal comparison to: {sig_comp_path}")

    # Regime-by-Regime Breakdown
    print("2. Running regime-by-regime univariate comparisons...", flush=True)
    reg_rows = []
    for reg in ["2020", "2021", "2022", "2023", "2024-2025"]:
        reg_sub = df[df["regime"] == reg]
        reg_type = "CLEAN_OOS" if reg in ["2022", "2023", "2024-2025"] else "TRAINING_OVERLAP_DIAGNOSTIC"
        res_df = evaluate_univariate_signals(reg_sub, label=f"{reg} ({reg_type})")
        res_df["regime"] = reg
        res_df["regime_type"] = reg_type
        reg_rows.append(res_df)
    reg_sig_df = pd.concat(reg_rows, ignore_index=True)
    reg_sig_path = out_dir / "regime_signal_comparison.csv"
    reg_sig_df.to_csv(reg_sig_path, index=False)
    print(f"Saved regime signal comparison to: {reg_sig_path}")

    # Seed-by-Seed Breakdown
    print("3. Running seed-by-seed univariate comparisons...", flush=True)
    seed_rows = []
    for s in [42, 123, 777]:
        seed_sub = clean_df[clean_df["seed"] == s]
        res_df = evaluate_univariate_signals(seed_sub, label=f"Seed_{s} (Clean)")
        res_df["seed"] = s
        seed_rows.append(res_df)
    seed_sig_df = pd.concat(seed_rows, ignore_index=True)
    seed_sig_path = out_dir / "seed_signal_comparison.csv"
    seed_sig_df.to_csv(seed_sig_path, index=False)
    print(f"Saved seed signal comparison to: {seed_sig_path}")

    # Step 3: Incremental Information Test (LORO)
    print("\n4. Running Leave-One-Regime-Out (LORO) incremental predictive tests...", flush=True)
    loro_df = run_loro_model_comparison(df)
    loro_path = out_dir / "loro_model_comparison.csv"
    loro_df.to_csv(loro_path, index=False)
    print(f"Saved LORO model comparison to: {loro_path}")

    # Step 4: Residual Information Test
    print("\n5. Running residual information test (epistemic orthogonalized to simple confidence)...", flush=True)
    resid_df = run_residual_information_test(df)
    resid_path = out_dir / "residual_information_test.csv"
    resid_df.to_csv(resid_path, index=False)
    print(f"Saved residual information test to: {resid_path}")

    # Step 5: Calibration Diagnostics
    print("\n6. Computing error calibration by quintile...", flush=True)
    calib_df = run_quintile_calibration(df)
    calib_path = out_dir / "quintile_calibration.csv"
    calib_df.to_csv(calib_path, index=False)
    print(f"Saved quintile calibration to: {calib_path}")

    # Step 6: Dependence-Aware Block Bootstrap
    print("\n7. Running dependence-aware weekly cluster bootstrap (B=1,000)...", flush=True)
    boot_df = run_dependence_aware_bootstrap(df, n_iterations=1000, seed=42)
    boot_path = out_dir / "block_bootstrap_intervals.csv"
    boot_df.to_csv(boot_path, index=False)
    print(f"Saved block bootstrap intervals to: {boot_path}")

    # Step 7: Build Summary JSON
    print("\n8. Generating Phase 15 summary JSON...", flush=True)
    # Extract key findings
    pooled_loro = loro_df[loro_df["evaluation_scope"] == "LORO_Pooled_Out_of_Sample"].set_index("model_name")
    resid_pooled = resid_df[resid_df["sample_scope"] == "Clean_Pooled (2022-2025)"].iloc[0]

    # Determine scientific classification outcome
    delta_auroc_loro = float(pooled_loro.loc["Model D (Confidence + Market + Epistemic)", "delta_auroc_vs_model_c"])
    delta_mae_loro = float(pooled_loro.loc["Model D (Confidence + Market + Epistemic)", "delta_mae_vs_model_c"])
    resid_spear_rho = float(resid_pooled["residual_epistemic_spearman_rho"])

    if delta_auroc_loro > 0.01 and delta_mae_loro > 0.001 and resid_spear_rho > 0.10:
        verdict = "Outcome A: Epistemic uncertainty clearly adds incremental information"
        verdict_desc = "Bayesian posterior uncertainty captures model-reliability information not available from ordinary confidence or market measures."
    elif abs(delta_auroc_loro) <= 0.01 and abs(delta_mae_loro) <= 0.001:
        verdict = "Outcome B: Epistemic uncertainty and entropy/margin perform similarly"
        verdict_desc = "The forecast-reliability signal is real and statistically robust, but simple confidence proxies (predictive entropy, margin) capture virtually all the same information without requiring Bayesian Monte Carlo sampling."
    elif delta_auroc_loro < -0.01:
        verdict = "Outcome C: Simple confidence dominates"
        verdict_desc = "Bayesian epistemic uncertainty adds computational overhead without incremental diagnostic value."
    else:
        verdict = "Outcome D: Incremental value is regime-dependent"
        verdict_desc = "Bayesian uncertainty contains conditional rather than universally transferable reliability information across regimes."

    summary_data = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": "Phase 15: Incremental Value of Bayesian Epistemic Uncertainty",
        "dataset_scope": {
            "primary_clean_regimes": ["2022", "2023", "2024-2025"],
            "total_clean_observations": len(clean_df),
            "unique_clean_calendar_weeks": int(clean_df["calendar_week"].nunique()),
            "secondary_diagnostic_regimes": ["2020", "2021"],
            "total_diagnostic_observations": len(df[~df["regime"].isin(["2022", "2023", "2024-2025"])]),
        },
        "target_definitions": {
            "primary_target": "absolute_forecast_error (|p_hat - y_true|)",
            "secondary_target": "binary_top20_high_error (error >= 80th percentile)",
            "clean_downside_threshold_move1d": float(clean_df["move_1d"].quantile(0.20)),
            "clean_high_error_threshold": float(clean_df["forecast_error"].quantile(0.80)),
        },
        "scientific_adjudication": {
            "verdict": verdict,
            "interpretation": verdict_desc,
            "delta_auroc_model_d_vs_c": delta_auroc_loro,
            "delta_mae_model_d_vs_c": delta_mae_loro,
            "residual_epistemic_spearman_rho": resid_spear_rho,
        },
        "univariate_top_signals_clean_pooled": sig_comp_df.sort_values("high_error_auroc", ascending=False).head(5).to_dict(orient="records"),
        "loro_pooled_model_comparison": pooled_loro.reset_index().to_dict(orient="records"),
        "residual_information_test": resid_df.to_dict(orient="records"),
        "block_bootstrap_summary": boot_df.to_dict(orient="records"),
    }

    sum_path = out_dir / "phase15_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved Phase 15 summary JSON to: {sum_path}")
    print(f"\nAdjudicated Verdict: {verdict}", flush=True)


if __name__ == "__main__":
    main()
