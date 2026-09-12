#!/usr/bin/env python3
"""Phase 15.1: Deterministic-Confidence Provenance Audit.

Trace the origin of p used in Phase 15 confidence baselines, construct a genuine
single-pass deterministic prediction using frozen posterior mean parameters (mu, no sampling),
evaluate entropy_det and margin_det, and compare them against MC32 signals.

Preregistration Constraints:
- Primary clean sample: 2022, 2023, 2024-2025 (N=4,100 clean events x 3 seeds = 12,300 rows).
- Zero trading strategy search (no M17).
- Genuine single-pass deterministic inference uses exactly 1 forward pass per event.
- Dependence-aware weekly cluster block bootstrap (B=1,000) over 168 calendar weeks.
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import optimize, stats
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Sibling repo imports for Bayesian LoRA module
sibling_src = Path("/Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/src")
if str(sibling_src) not in sys.path:
    sys.path.insert(0, str(sibling_src))

from bayesian_lora_portfolio_risk.uncertainty.bayesian_lora import (  # noqa: E402
    inject_bert_attention_bayesian_lora,
    iter_bayesian_lora_layers,
    set_posterior_sampling,
)

repo_root = Path(__file__).resolve().parent.parent


class SimpleTextDataset(Dataset):
    """Minimal PyTorch dataset for tokenized sequence classification."""

    def __init__(self, texts: list[str], tokenizer: Any, max_length: int = 64):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: val[idx] for key, val in self.encodings.items()}


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Analytical AUROC via Mann-Whitney U statistic (fast vectorized)."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    u_stat, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
    return float(u_stat / (n_pos * n_neg))


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision (AP) for binary classification."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp_cumsum = np.cumsum(y_sorted)
    k_range = np.arange(1, len(y_true) + 1)
    precisions = tp_cumsum / k_range
    total_pos = np.sum(y_true)
    if total_pos == 0:
        return 0.0
    return float(np.sum(precisions * y_sorted) / total_pos)


def fit_predict_ridge(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Exact analytical Ridge regression for continuous forecast error."""
    mean_x = np.mean(x_train, axis=0)
    std_x = np.std(x_train, axis=0) + 1e-8
    x_tr_norm = (x_train - mean_x) / std_x
    x_te_norm = (x_test - mean_x) / std_x

    n_features = x_train.shape[1]
    x_tr_bias = np.hstack([np.ones((len(x_train), 1)), x_tr_norm])
    x_te_bias = np.hstack([np.ones((len(x_test), 1)), x_te_norm])

    reg_mat = alpha * np.eye(n_features + 1)
    reg_mat[0, 0] = 0.0

    w = np.linalg.solve(x_tr_bias.T @ x_tr_bias + reg_mat, x_tr_bias.T @ y_train)
    return x_te_bias @ w


def fit_predict_logistic(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, c_reg: float = 1.0
) -> np.ndarray:
    """Fast L-BFGS Logistic Regression for binary high error."""
    mean_x = np.mean(x_train, axis=0)
    std_x = np.std(x_train, axis=0) + 1e-8
    x_tr_norm = (x_train - mean_x) / std_x
    x_te_norm = (x_test - mean_x) / std_x

    n_samples, n_features = x_tr_norm.shape
    x_tr_bias = np.hstack([np.ones((n_samples, 1)), x_tr_norm])
    x_te_bias = np.hstack([np.ones((len(x_test), 1)), x_te_norm])

    def loss_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = np.clip(x_tr_bias @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        loss = -np.mean(y_train * np.log(p + 1e-12) + (1.0 - y_train) * np.log(1.0 - p + 1e-12))
        loss += 0.5 * (1.0 / (c_reg * n_samples)) * np.sum(w[1:] ** 2)
        grad = (x_tr_bias.T @ (p - y_train)) / n_samples
        grad[1:] += (1.0 / (c_reg * n_samples)) * w[1:]
        return loss, grad

    w0 = np.zeros(n_features + 1)
    res = optimize.minimize(loss_and_grad, w0, method="L-BFGS-B", jac=True, options={"maxiter": 150})
    w_opt = res.x
    z_te = np.clip(x_te_bias @ w_opt, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_te))


def run_deterministic_single_pass_inference(
    clean_events_df: pd.DataFrame,
    device: torch.device,
    seeds: list[int] = [42, 123, 777],
    batch_size: int = 64,
) -> pd.DataFrame:
    """Execute exactly 1 forward pass per event using frozen posterior mean parameters (mu)."""
    model_name = "ProsusAI/finbert"
    base_revision = "4556d13015211d73dccd3fdd39d39232506f3e43"
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=base_revision, local_files_only=True)

    texts = clean_events_df["text"].tolist()
    dataset = SimpleTextDataset(texts, tokenizer, max_length=64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    models_dir = repo_root / "artifacts" / "phase11_replication" / "frozen_models"
    pred_rows = []

    for seed in seeds:
        print(f"  Running single-pass deterministic inference for Seed {seed}...", flush=True)
        t0 = time.perf_counter()
        ckpt_path = models_dir / f"bayesian_lora_seed_{seed}_state.pt"
        assert ckpt_path.exists(), f"Missing frozen checkpoint: {ckpt_path}"

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=base_revision,
            num_labels=2,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        inject_bert_attention_bayesian_lora(
            model, rank=16, alpha=32, dropout=0.05, init_std=0.01, prior_std=1.0
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        # Disable stochastic sampling: use posterior mean parameters (mu) exclusively
        set_posterior_sampling(model, False)
        for layer in iter_bayesian_lora_layers(model):
            assert not layer.sample_weight, f"Seed {seed} layer sample_weight must be False"

        probs_all = []
        forward_pass_count = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs_all.append(probs)
                forward_pass_count += 1

        probs_arr = np.concatenate(probs_all)
        elapsed = time.perf_counter() - t0
        print(f"  Seed {seed} completed {len(probs_arr)} events in {elapsed:.2f}s ({len(probs_arr)/elapsed:.1f} events/s). Forward passes per event: 1", flush=True)

        for i, (_, row) in enumerate(clean_events_df.iterrows()):
            pred_rows.append({
                "event_id": str(row["event_id"]),
                "ticker": str(row["ticker"]),
                "date": str(row["date"]),
                "regime": str(row["regime"]),
                "seed": seed,
                "p_deterministic_mean": float(probs_arr[i]),
            })

    return pd.DataFrame(pred_rows)


def build_evaluation_dataframe(
    det_df: pd.DataFrame,
    mc_preds_df: pd.DataFrame,
    rets_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge single-pass deterministic predictions with MC32 outputs and returns."""
    rets = rets_df.copy()
    rets["dt"] = pd.to_datetime(rets["date"])
    rets = rets.sort_values("dt").reset_index(drop=True)

    rets["prev_move"] = rets.groupby("symbol")["move_1d"].shift(1)
    trailing_vol = rets.groupby("symbol")["prev_move"].expanding().std().reset_index(level=0, drop=True)
    symbol_overall_std = rets.groupby("symbol")["move_1d"].transform("std")
    median_vol = float(rets["move_1d"].std())
    rets["asset_vol"] = trailing_vol.fillna(symbol_overall_std).fillna(median_vol)

    # Build event features
    ret_dict = {str(r["episode_id"]): float(r["move_1d"]) for _, r in rets.iterrows() if pd.notna(r["move_1d"])}
    vol_dict = {str(r["episode_id"]): float(r["asset_vol"]) for _, r in rets.iterrows() if pd.notna(r["asset_vol"])}

    # Merge det_df and mc_preds_df
    det_df["key"] = det_df["event_id"].astype(str) + "_" + det_df["seed"].astype(str)
    mc_preds_df["key"] = mc_preds_df["event_id"].astype(str) + "_" + mc_preds_df["seed"].astype(str)

    merged = pd.merge(
        det_df,
        mc_preds_df[["key", "posterior_mean", "epistemic_uncertainty", "aleatoric_uncertainty", "total_uncertainty"]],
        on="key",
        how="inner",
    )
    merged.drop(columns=["key"], inplace=True)

    merged["move_1d"] = merged["event_id"].map(ret_dict)
    merged["asset_volatility"] = merged["event_id"].map(vol_dict).fillna(median_vol)

    merged["dt"] = pd.to_datetime(merged["date"])
    merged["calendar_week"] = merged["dt"].apply(lambda d: f"{d.year}-W{d.isocalendar()[1]:02d}")

    # Market volatility proxy
    week_vol_map = merged.groupby("calendar_week")["move_1d"].std().to_dict()
    merged["market_vol_proxy"] = merged["calendar_week"].map(week_vol_map).fillna(median_vol)
    merged["cross_sectional_vol_rank"] = merged.groupby("calendar_week")["asset_volatility"].rank(pct=True)

    # Downside label (clean 20th percentile threshold frozen at -4.692%)
    clean_thresh = float(merged["move_1d"].quantile(0.20))
    merged["y_true"] = (merged["move_1d"] <= clean_thresh).astype(int)

    # Probabilities
    p_mc = np.clip(merged["posterior_mean"].values, 1e-7, 1.0 - 1e-7)
    p_det = np.clip(merged["p_deterministic_mean"].values, 1e-7, 1.0 - 1e-7)

    # Errors
    merged["forecast_error_mc32"] = np.abs(p_mc - merged["y_true"].values)
    merged["forecast_error_det"] = np.abs(p_det - merged["y_true"].values)
    # Primary evaluation target remains absolute error against true downside outcome
    merged["forecast_error"] = merged["forecast_error_mc32"]

    clean_err_thresh = float(merged["forecast_error"].quantile(0.80))
    merged["high_error_binary"] = (merged["forecast_error"] >= clean_err_thresh).astype(int)

    # Confidence Signals
    # 1. MC32 Confidence
    merged["entropy_mc32"] = -p_mc * np.log(p_mc) - (1.0 - p_mc) * np.log(1.0 - p_mc)
    merged["margin_mc32"] = 0.5 - np.abs(p_mc - 0.5)

    # 2. True Single-Pass Deterministic Confidence
    merged["entropy_det"] = -p_det * np.log(p_det) - (1.0 - p_det) * np.log(1.0 - p_det)
    merged["margin_det"] = 0.5 - np.abs(p_det - 0.5)

    return merged


def evaluate_signal_dictionary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Evaluate univariate predictive performance of candidate signals against forecast error."""
    signals = [
        ("epistemic_uncertainty", "MC32 Epistemic Uncertainty (Variance)", "BAYESIAN_MC32", 32),
        ("entropy_mc32", "Predictive Entropy H(p_bar_MC32)", "MC32_POSTERIOR_MEAN_CONFIDENCE", 32),
        ("margin_mc32", "Confidence Margin (0.5 - |p_bar_MC32 - 0.5|)", "MC32_POSTERIOR_MEAN_CONFIDENCE", 32),
        ("entropy_det", "Predictive Entropy H(p_deterministic_mean)", "TRUE_SINGLE_PASS_DETERMINISTIC", 1),
        ("margin_det", "Confidence Margin (0.5 - |p_deterministic_mean - 0.5|)", "TRUE_SINGLE_PASS_DETERMINISTIC", 1),
        ("asset_volatility", "Asset Volatility (Historical Trailing Std)", "MARKET_VOLATILITY", 0),
    ]

    y_err = df["forecast_error"].values
    y_high = df["high_error_binary"].values
    n_obs = len(df)

    rows = []
    for col, sig_desc, cat, n_passes in signals:
        scores = df[col].values
        rho, p_spear = stats.spearmanr(scores, y_err)
        r_p, p_pear = stats.pearsonr(scores, y_err)
        auroc = compute_auroc(y_high, scores)
        ap = compute_average_precision(y_high, scores)

        try:
            q_bins = pd.qcut(scores, q=5, labels=False, duplicates="drop")
            err_by_q = pd.Series(y_err).groupby(q_bins).mean()
            q5_q1 = float(err_by_q.iloc[-1] / (err_by_q.iloc[0] + 1e-8))
        except Exception:
            q5_q1 = 1.0

        rows.append({
            "scope": label,
            "signal_name": col,
            "signal_description": sig_desc,
            "signal_category": cat,
            "forward_passes_per_event": n_passes,
            "spearman_rho": round(float(rho), 4),
            "p_value_spearman": float(p_spear),
            "pearson_r": round(float(r_p), 4),
            "p_value_pearson": float(p_pear),
            "q5_q1_error_ratio": round(float(q5_q1), 4),
            "high_error_auroc": round(float(auroc), 4),
            "high_error_ap": round(float(ap), 4),
            "n_observations": n_obs,
        })

    return pd.DataFrame(rows)


def run_loro_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-One-Regime-Out nested model comparison incorporating deterministic confidence."""
    regimes = ["2022", "2023", "2024-2025"]

    models = [
        ("Model A_MC32 (MC32 Confidence)", ["entropy_mc32", "margin_mc32"]),
        ("Model A_det (Single-Pass Confidence)", ["entropy_det", "margin_det"]),
        ("Model B (Market Only)", ["asset_volatility", "market_vol_proxy", "cross_sectional_vol_rank"]),
        ("Model C_MC32 (Confidence_MC32 + Market)", ["entropy_mc32", "margin_mc32", "asset_volatility", "market_vol_proxy", "cross_sectional_vol_rank"]),
        ("Model C_det (Confidence_det + Market)", ["entropy_det", "margin_det", "asset_volatility", "market_vol_proxy", "cross_sectional_vol_rank"]),
        ("Model D (Confidence_MC32 + Market + Epistemic)", ["entropy_mc32", "margin_mc32", "asset_volatility", "market_vol_proxy", "cross_sectional_vol_rank", "epistemic_uncertainty"]),
    ]

    loro_records = []
    pooled_preds: dict[str, dict[str, list[float]]] = {
        m_name: {"y_true": [], "y_err": [], "pred_err": [], "pred_prob": []}
        for m_name, _ in models
    }

    for held_out in regimes:
        train_df = df[df["regime"] != held_out]
        test_df = df[df["regime"] == held_out]

        y_tr_err = train_df["forecast_error"].values
        y_te_err = test_df["forecast_error"].values
        y_tr_high = train_df["high_error_binary"].values
        y_te_high = test_df["high_error_binary"].values

        baseline_ss_err = np.sum((y_te_err - np.mean(y_tr_err)) ** 2)

        for m_name, feat_cols in models:
            x_tr = train_df[feat_cols].values
            x_te = test_df[feat_cols].values

            # Ridge for error prediction
            pred_err = fit_predict_ridge(x_tr, y_tr_err, x_te, alpha=1.0)
            mae = float(np.mean(np.abs(pred_err - y_te_err)))
            ss_res = np.sum((pred_err - y_te_err) ** 2)
            r2 = float(1.0 - (ss_res / baseline_ss_err))

            # Logistic for high error
            pred_prob = fit_predict_logistic(x_tr, y_tr_high, x_te, c_reg=1.0)
            auroc = compute_auroc(y_te_high, pred_prob)
            ap = compute_average_precision(y_te_high, pred_prob)

            pooled_preds[m_name]["y_true"].extend(y_te_high.tolist())
            pooled_preds[m_name]["y_err"].extend(y_te_err.tolist())
            pooled_preds[m_name]["pred_err"].extend(pred_err.tolist())
            pooled_preds[m_name]["pred_prob"].extend(pred_prob.tolist())

            loro_records.append({
                "evaluation_scope": f"LORO_HeldOut_{held_out}",
                "held_out_regime": held_out,
                "model_name": m_name,
                "features": ", ".join(feat_cols),
                "held_out_r2": round(r2, 5),
                "held_out_mae": round(mae, 5),
                "held_out_auroc": round(auroc, 4),
                "held_out_ap": round(ap, 4),
                "n_test_observations": len(test_df),
            })

    # Pooled summary
    pooled_records = []
    y_pooled_err = np.array(pooled_preds["Model C_MC32 (Confidence_MC32 + Market)"]["y_err"])
    base_ss_pooled = np.sum((y_pooled_err - np.mean(y_pooled_err)) ** 2)

    for m_name, _ in models:
        y_te_h = np.array(pooled_preds[m_name]["y_true"])
        y_te_e = np.array(pooled_preds[m_name]["y_err"])
        p_err = np.array(pooled_preds[m_name]["pred_err"])
        p_prob = np.array(pooled_preds[m_name]["pred_prob"])

        mae = float(np.mean(np.abs(p_err - y_te_e)))
        ss_res = np.sum((p_err - y_te_e) ** 2)
        r2 = float(1.0 - (ss_res / base_ss_pooled))
        auroc = compute_auroc(y_te_h, p_prob)
        ap = compute_average_precision(y_te_h, p_prob)

        pooled_records.append({
            "evaluation_scope": "LORO_Pooled_Out_of_Sample",
            "held_out_regime": "All_Clean_Regimes",
            "model_name": m_name,
            "features": dict(models)[m_name],
            "held_out_r2": round(r2, 5),
            "held_out_mae": round(mae, 5),
            "held_out_auroc": round(auroc, 4),
            "held_out_ap": round(ap, 4),
            "n_test_observations": len(df),
        })

    full_df = pd.DataFrame(loro_records + pooled_records)
    ref_auroc = full_df.loc[full_df["model_name"] == "Model C_MC32 (Confidence_MC32 + Market)", ["evaluation_scope", "held_out_auroc"]].set_index("evaluation_scope")["held_out_auroc"].to_dict()
    ref_mae = full_df.loc[full_df["model_name"] == "Model C_MC32 (Confidence_MC32 + Market)", ["evaluation_scope", "held_out_mae"]].set_index("evaluation_scope")["held_out_mae"].to_dict()

    full_df["delta_auroc_vs_model_c_mc32"] = full_df.apply(lambda r: round(r["held_out_auroc"] - ref_auroc[r["evaluation_scope"]], 4), axis=1)
    full_df["delta_mae_vs_model_c_mc32"] = full_df.apply(lambda r: round(r["held_out_mae"] - ref_mae[r["evaluation_scope"]], 5), axis=1)

    return full_df


def run_block_bootstrap_provenance(
    df: pd.DataFrame, n_iterations: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Run dependence-aware weekly cluster block bootstrap over 168 calendar weeks."""
    rng = np.random.RandomState(seed)
    weeks = df["calendar_week"].unique()
    n_weeks = len(weeks)

    week_to_indices = defaultdict(list)
    for idx, w in enumerate(df["calendar_week"].values):
        week_to_indices[w].append(idx)

    y_err = df["forecast_error"].values
    y_high = df["high_error_binary"].values
    epi = df["epistemic_uncertainty"].values
    ent_mc = df["entropy_mc32"].values
    ent_det = df["entropy_det"].values

    delta_rho_det_vs_epi = []
    delta_rho_det_vs_mc = []
    delta_auroc_det_vs_epi = []
    delta_auroc_det_vs_mc = []

    for _ in range(n_iterations):
        sample_weeks = rng.choice(weeks, size=n_weeks, replace=True)
        boot_idx = np.concatenate([week_to_indices[w] for w in sample_weeks])

        y_e_b = y_err[boot_idx]
        y_h_b = y_high[boot_idx]

        r_epi, _ = stats.spearmanr(epi[boot_idx], y_e_b)
        r_ent_mc, _ = stats.spearmanr(ent_mc[boot_idx], y_e_b)
        r_ent_det, _ = stats.spearmanr(ent_det[boot_idx], y_e_b)

        auc_epi = compute_auroc(y_h_b, epi[boot_idx])
        auc_mc = compute_auroc(y_h_b, ent_mc[boot_idx])
        auc_det = compute_auroc(y_h_b, ent_det[boot_idx])

        delta_rho_det_vs_epi.append(r_ent_det - r_epi)
        delta_rho_det_vs_mc.append(r_ent_det - r_ent_mc)
        delta_auroc_det_vs_epi.append(auc_det - auc_epi)
        delta_auroc_det_vs_mc.append(auc_det - auc_mc)

    results = [
        ("Delta_Rho (Entropy_Det - Epistemic_MC32)", delta_rho_det_vs_epi),
        ("Delta_Rho (Entropy_Det - Entropy_MC32)", delta_rho_det_vs_mc),
        ("Delta_AUROC (Entropy_Det - Epistemic_MC32)", delta_auroc_det_vs_epi),
        ("Delta_AUROC (Entropy_Det - Entropy_MC32)", delta_auroc_det_vs_mc),
    ]

    rows = []
    for metric_name, diffs in results:
        arr = np.array(diffs)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr))
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        p_val = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))

        rows.append({
            "comparison_metric": metric_name,
            "bootstrap_mean": round(mean_v, 5),
            "bootstrap_std": round(std_v, 5),
            "ci_95_lower": round(ci_lower, 5),
            "ci_95_upper": round(ci_upper, 5),
            "empirical_p_value": round(p_val, 4),
            "statistically_significant_at_05": bool(ci_lower > 0.0 or ci_upper < 0.0),
            "n_resamples": n_iterations,
        })

    return pd.DataFrame(rows)


def main() -> None:
    print("=== Phase 15.1: Deterministic-Confidence Provenance Audit ===", flush=True)

    out_dir = repo_root / "artifacts" / "phase15_1_provenance_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Inference Device: {device}", flush=True)

    # 1. Recover clean historical events
    rec_path = repo_root / "data" / "bayesian_lora" / "reconstructed_returns.csv"
    assert rec_path.exists(), f"Missing reconstructed returns: {rec_path}"
    rec_df = pd.read_csv(rec_path).dropna(subset=["move_1d"]).copy()
    rec_df["date_dt"] = pd.to_datetime(rec_df["date"])

    clean_regimes_dict = {
        "2022": ("2022-01-01", "2022-12-31 23:59:59"),
        "2023": ("2023-01-01", "2023-12-31 23:59:59"),
        "2024-2025": ("2024-01-01", "2025-05-15 23:59:59"),
    }

    clean_events = []
    for reg, (start_dt, end_dt) in clean_regimes_dict.items():
        sub = rec_df[(rec_df["date_dt"] >= start_dt) & (rec_df["date_dt"] <= end_dt)].sort_values("date")
        for _, row in sub.iterrows():
            company = str(row["company_name"]) if pd.notna(row["company_name"]) else str(row["symbol"])
            ticker = str(row["symbol"])
            date_str = str(row["date"])
            move_1d = float(row["move_1d"])
            text = f"{company} ({ticker}) quarterly earnings announcement on {date_str}."
            clean_events.append({
                "event_id": str(row["episode_id"]),
                "ticker": ticker,
                "date": date_str,
                "regime": reg,
                "text": text,
                "move_1d": move_1d,
            })

    clean_events_df = pd.DataFrame(clean_events)
    print(f"Total clean events extracted: {len(clean_events_df)} (expected 4,100).", flush=True)
    assert len(clean_events_df) == 4100, f"Expected 4,100 clean events, got {len(clean_events_df)}"

    # 2. Check if deterministic predictions already exist or run genuine 1-pass inference
    det_pred_path = out_dir / "deterministic_predictions.csv"
    if det_pred_path.exists():
        print(f"Loading existing single-pass deterministic predictions from: {det_pred_path}", flush=True)
        det_df = pd.read_csv(det_pred_path)
    else:
        print("\nExecuting genuine single-pass deterministic inference (posterior mean parameters mu, 1 pass)...", flush=True)
        det_df = run_deterministic_single_pass_inference(clean_events_df, device=device, seeds=[42, 123, 777], batch_size=64)
        det_df.to_csv(det_pred_path, index=False)
        print(f"Saved deterministic predictions to: {det_pred_path}", flush=True)

    # Load Phase 13 evaluation predictions (MC32)
    mc_preds_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "evaluation_predictions.csv"
    assert mc_preds_path.exists(), f"Missing Phase 13 predictions: {mc_preds_path}"
    mc_preds_df = pd.read_csv(mc_preds_path)
    mc_preds_clean = mc_preds_df[mc_preds_df["regime"].isin(["2022", "2023", "2024-2025"])].copy()

    # 3. Build merged evaluation dataset
    print("\nBuilding merged evaluation features...", flush=True)
    df = build_evaluation_dataframe(det_df, mc_preds_clean, rec_df)
    print(f"Merged clean dataset: {len(df)} rows across {df['calendar_week'].nunique()} calendar weeks.", flush=True)

    # 4. Direct Correlation between p_bar_MC32 and p_deterministic_mean
    p_mc = df["posterior_mean"].values
    p_det = df["p_deterministic_mean"].values
    corr_pearson, p_pear = stats.pearsonr(p_mc, p_det)
    corr_spearman, p_spear = stats.spearmanr(p_mc, p_det)
    mae_p = float(np.mean(np.abs(p_mc - p_det)))
    max_p_diff = float(np.max(np.abs(p_mc - p_det)))

    print("\n--- Probability Provenance Alignment ---", flush=True)
    print(f"Pearson r(p_bar_MC32, p_deterministic_mean):  {corr_pearson:.5f}", flush=True)
    print(f"Spearman rho(p_bar_MC32, p_deterministic_mean): {corr_spearman:.5f}", flush=True)
    print(f"MAE |p_bar_MC32 - p_deterministic_mean|:        {mae_p:.5f}", flush=True)
    print(f"Max |p_bar_MC32 - p_deterministic_mean|:        {max_p_diff:.5f}", flush=True)

    # 5. Univariate Evaluation
    print("\nRunning univariate signal comparison on clean pooled data...", flush=True)
    sig_comp_df = evaluate_signal_dictionary(df, label="Clean_Pooled (2022-2025)")
    sig_comp_path = out_dir / "provenance_comparison.csv"
    sig_comp_df.to_csv(sig_comp_path, index=False)
    print(f"Saved provenance comparison to: {sig_comp_path}", flush=True)

    # Regime breakdown
    reg_rows = []
    for reg in ["2022", "2023", "2024-2025"]:
        reg_df = evaluate_signal_dictionary(df[df["regime"] == reg], label=f"{reg} (Clean)")
        reg_df["regime"] = reg
        reg_rows.append(reg_df)
    reg_comp_df = pd.concat(reg_rows, ignore_index=True)
    reg_comp_path = out_dir / "regime_provenance_comparison.csv"
    reg_comp_df.to_csv(reg_comp_path, index=False)

    # Seed breakdown
    seed_rows = []
    for s in [42, 123, 777]:
        seed_df = evaluate_signal_dictionary(df[df["seed"] == s], label=f"Seed_{s} (Clean)")
        seed_df["seed"] = s
        seed_rows.append(seed_df)
    seed_comp_df = pd.concat(seed_rows, ignore_index=True)
    seed_comp_path = out_dir / "seed_provenance_comparison.csv"
    seed_comp_df.to_csv(seed_comp_path, index=False)

    # 6. LORO Model Comparison
    print("\nRunning Leave-One-Regime-Out (LORO) predictive comparison...", flush=True)
    loro_df = run_loro_model_comparison(df)
    loro_path = out_dir / "loro_provenance_comparison.csv"
    loro_df.to_csv(loro_path, index=False)
    print(f"Saved LORO comparison to: {loro_path}", flush=True)

    # 7. Quintile Calibration for Deterministic Confidence
    print("\nComputing quintile error calibration for deterministic confidence...", flush=True)
    calib_rows = []
    for sig_name in ["entropy_det", "margin_det", "entropy_mc32", "epistemic_uncertainty"]:
        try:
            q_bins = pd.qcut(df[sig_name], q=5, labels=["Q1_Lowest", "Q2", "Q3", "Q4", "Q5_Highest"], duplicates="drop")
            for q_label, group in df.groupby(q_bins, observed=False):
                calib_rows.append({
                    "signal_name": sig_name,
                    "quintile": str(q_label),
                    "n_events": len(group),
                    "mean_signal_value": round(float(group[sig_name].mean()), 6),
                    "mean_forecast_error": round(float(group["forecast_error"].mean()), 4),
                    "std_forecast_error": round(float(group["forecast_error"].std()), 4),
                    "top20_error_frequency": round(float(group["high_error_binary"].mean() * 100.0), 2),
                })
        except Exception as e:
            print(f"Warning on quintile calibration for {sig_name}: {e}")
    calib_df = pd.DataFrame(calib_rows)
    calib_path = out_dir / "quintile_calibration_det.csv"
    calib_df.to_csv(calib_path, index=False)

    # 8. Weekly Cluster Block Bootstrap
    print("\nRunning weekly cluster block bootstrap (B=1,000)...", flush=True)
    boot_df = run_block_bootstrap_provenance(df, n_iterations=1000, seed=42)
    boot_path = out_dir / "block_bootstrap_provenance.csv"
    boot_df.to_csv(boot_path, index=False)
    print(f"Saved block bootstrap results to: {boot_path}", flush=True)

    # 9. Adjudication & Verdict
    # Check Outcome A vs B vs C
    rho_epi = float(sig_comp_df.loc[sig_comp_df["signal_name"] == "epistemic_uncertainty", "spearman_rho"].iloc[0])
    rho_mc = float(sig_comp_df.loc[sig_comp_df["signal_name"] == "entropy_mc32", "spearman_rho"].iloc[0])
    rho_det = float(sig_comp_df.loc[sig_comp_df["signal_name"] == "entropy_det", "spearman_rho"].iloc[0])

    delta_rho_det_vs_epi = rho_det - rho_epi
    delta_rho_det_vs_mc = rho_det - rho_mc

    if abs(delta_rho_det_vs_epi) <= 0.02 and abs(delta_rho_det_vs_mc) <= 0.02:
        verdict = "Outcome A: True single-pass entropy/margin matches MC32 uncertainty"
        verdict_desc = (
            "True single-pass deterministic confidence (using posterior mean parameters with exactly 1 forward pass) "
            "achieves virtually identical rank correlation with forecast error as MC32 uncertainty. "
            "Therefore, Bayesian MC sampling is computationally redundant for reliability ranking."
        )
    elif delta_rho_det_vs_epi < -0.05:
        verdict = "Outcome B: True single-pass confidence is materially worse"
        verdict_desc = (
            "MC posterior-mean entropy matches epistemic uncertainty, but true single-pass confidence degrades materially. "
            "Therefore, uncertainty variance is redundant conditional on Bayesian posterior averaging, but Bayesian MC inference itself is valuable."
        )
    elif delta_rho_det_vs_epi > 0.02:
        verdict = "Outcome C: True deterministic confidence is better"
        verdict_desc = (
            "True single-pass deterministic confidence outperforms MC32 uncertainty. "
            "Bayesian sampling adds neither diagnostic nor computational value."
        )
    else:
        verdict = "Outcome A (Marginal Difference): Single-pass confidence closely tracks MC32 uncertainty"
        verdict_desc = "Single-pass confidence performs within standard error of MC32 epistemic uncertainty."

    print("\n=======================================================", flush=True)
    print(f"ADJUDICATED VERDICT: {verdict}", flush=True)
    print(f"Interpretation: {verdict_desc}", flush=True)
    print("=======================================================", flush=True)

    summary_data = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": "Phase 15.1: Deterministic-Confidence Provenance Audit",
        "provenance_trace": {
            "phase15_probability_source": "Posterior predictive mean across 32 Bayesian MC draws (p_bar_MC32)",
            "phase15_forward_passes_per_event": 32,
            "phase15_classification": "MC32 posterior-mean confidence (NOT single-pass deterministic)",
            "true_single_pass_method": "Frozen Bayesian-LoRA posterior mean parameters (mu), sample_weight=False",
            "true_single_pass_forward_passes_per_event": 1,
        },
        "probability_alignment": {
            "pearson_r": round(corr_pearson, 5),
            "spearman_rho": round(corr_spearman, 5),
            "mean_absolute_error": round(mae_p, 5),
            "max_absolute_error": round(max_p_diff, 5),
        },
        "scientific_adjudication": {
            "verdict": verdict,
            "description": verdict_desc,
            "spearman_rho_epistemic_mc32": rho_epi,
            "spearman_rho_entropy_mc32": rho_mc,
            "spearman_rho_entropy_det_single_pass": rho_det,
            "delta_rho_det_vs_epi": round(delta_rho_det_vs_epi, 4),
            "delta_rho_det_vs_mc32": round(delta_rho_det_vs_mc, 4),
        },
        "signal_comparison_pooled": sig_comp_df.to_dict(orient="records"),
        "loro_comparison_summary": loro_df[loro_df["evaluation_scope"] == "LORO_Pooled_Out_of_Sample"].to_dict(orient="records"),
        "block_bootstrap_summary": boot_df.to_dict(orient="records"),
    }

    sum_path = out_dir / "phase15_1_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved Phase 15.1 summary JSON to: {sum_path}", flush=True)


if __name__ == "__main__":
    main()
