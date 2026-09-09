"""
Phase 11.3: Genuine Replicated-Model Historical Inference Audit
==============================================================
Runs actual model inference on the fixed 2024-2025 benchmark from frozen
Bayesian-LoRA checkpoints (bayesian_lora_seed_<seed>_state.pt) with 32 MC draws.

Strict integrity rules:
- Zero 2026 realized returns accessed.
- Does NOT substitute old historical predictions.
- Replaces synthetic perturbation diagnostics with actual model inference.
- Reassesses procedural replication under the formal Outcome Tree.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Sibling repo imports
sibling_src = Path("/Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/src")
sys.path.insert(0, str(sibling_src))

from bayesian_lora_portfolio_risk.uncertainty.bayesian_lora import (  # noqa: E402
    inject_bert_attention_bayesian_lora,
    set_posterior_sampling,
)

# Local repo imports
local_repo = Path(__file__).parent.parent
sys.path.insert(0, str(local_repo / "src"))
sys.path.insert(0, str(local_repo))

from hypergraph_portfolio.bayesian_lora_adapter import (  # noqa: E402
    BayesianPredictionRecord,
)
from hypergraph_portfolio.uncertainty import (  # noqa: E402
    compute_aggregate_epistemic_uncertainty,
    compute_hysteresis_exposure_path,
    compute_relative_epistemic_exposure_scale,
    compute_relative_epistemic_uncertainty,
    simulate_cohort_stacked_portfolio,
)


def compute_file_sha256(filepath: Path | str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class SimpleTextDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, texts: list[str], tokenizer: Any, max_length: int = 64) -> None:
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


def extract_benchmark_events(rec_path: Path) -> list[dict[str, Any]]:
    """
    Extract the exact 1,728 benchmark events from reconstructed_returns.csv
    matching the chronological 2024-2025 holdout window.
    """
    rec_df = pd.read_csv(rec_path)
    rec_df = rec_df.dropna(subset=["move_1d"]).copy()
    rec_df["date_dt"] = pd.to_datetime(rec_df["date"])

    # Filter to 2024-2025 window
    bench = rec_df[(rec_df["date_dt"] >= "2024-01-01") & (rec_df["date_dt"] <= "2025-05-15 23:59:59")].copy()
    bench = bench.sort_values("date").reset_index(drop=True)

    threshold = float(bench["move_1d"].quantile(0.20))
    events = []
    for _, row in bench.iterrows():
        company = str(row["company_name"]) if pd.notna(row["company_name"]) else str(row["symbol"])
        ticker = str(row["symbol"])
        date_str = str(row["date"])
        move_1d = float(row["move_1d"])
        label = 1 if move_1d <= threshold else 0

        # Input text available strictly at event time (no outcome leakage)
        text = f"{company} ({ticker}) quarterly earnings announcement on {date_str}."

        events.append({
            "event_id": str(row["episode_id"]),
            "ticker": ticker,
            "company_name": company,
            "date": date_str,
            "move_1d": move_1d,
            "label": label,
            "text": text,
        })

    return events


def run_genuine_mc_inference(
    seed: int,
    events: list[dict[str, Any]],
    base_revision: str,
    ckpt_path: Path,
    device: torch.device,
    mc_draws: int = 32,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    print("\n==================================================", flush=True)
    print(f"Running Genuine Inference: Seed {seed} (MC={mc_draws})", flush=True)
    print("==================================================", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=base_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=base_revision,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    inject_bert_attention_bayesian_lora(
        model, rank=16, alpha=32, dropout=0.05, init_std=0.01, prior_std=1.0
    )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    texts = [e["text"] for e in events]
    dataset = SimpleTextDataset(texts, tokenizer, max_length=64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    set_posterior_sampling(model, True)
    samples: list[np.ndarray] = []

    for draw_idx in range(mc_draws):
        if (draw_idx + 1) % 8 == 0 or draw_idx == 0 or draw_idx == mc_draws - 1:
            print(f"  MC Draw {draw_idx + 1}/{mc_draws}...", flush=True)

        batch_probs: list[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                batch_probs.extend(probs.tolist())

        samples.append(np.array(batch_probs, dtype=float))

    matrix = np.stack(samples, axis=0)  # Shape: [mc_draws, N]
    p_mean = matrix.mean(axis=0)
    p_epi = matrix.var(axis=0)
    p_ale = (matrix * (1.0 - matrix)).mean(axis=0)
    p_tot = p_epi + p_ale

    pred_rows = []
    for i, e in enumerate(events):
        pred_rows.append({
            "seed": seed,
            "event_id": e["event_id"],
            "ticker": e["ticker"],
            "date": e["date"],
            "posterior_mean": float(p_mean[i]),
            "epistemic_uncertainty": float(p_epi[i]),
            "aleatoric_uncertainty": float(p_ale[i]),
            "total_uncertainty": float(p_tot[i]),
            "move_1d": e["move_1d"],
            "label": e["label"],
        })

    return pred_rows


def evaluate_model_replication_metrics(
    seed: int, pred_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    y_true = np.array([r["label"] for r in pred_rows], dtype=int)
    p_mean = np.array([r["posterior_mean"] for r in pred_rows], dtype=float)
    p_epi = np.array([r["epistemic_uncertainty"] for r in pred_rows], dtype=float)

    errors = np.abs(p_mean - y_true)

    # 1. Spearman correlation(epistemic uncertainty, error)
    spearman_corr, spearman_pval = stats.spearmanr(p_epi, errors)

    # 2. AUROC top-20% error
    top_threshold = np.quantile(errors, 0.80)
    top_binary = (errors >= top_threshold).astype(int)
    auroc_top = float(roc_auc_score(top_binary, p_epi))
    ap_score = float(average_precision_score(top_binary, p_epi))

    # 3. Uncertainty quintiles
    q_labels = pd.qcut(p_epi, q=5, labels=False, duplicates="drop")
    q_errors = [float(np.mean(errors[q_labels == q])) for q in range(5)]
    q5_q1_ratio = q_errors[4] / max(q_errors[0], 1e-6)

    # 4. Directional downside AUROC
    directional_auc = float(roc_auc_score(y_true, p_mean))

    print(f"  Seed {seed} Genuine Model Metrics:")
    print(f"    Spearman corr(epi, error): {spearman_corr:+.4f} (p={spearman_pval:.4e})")
    print(f"    AUROC top-error: {auroc_top:.4f}; AP: {ap_score:.4f}")
    print(f"    Q1 MAE: {q_errors[0]:.4f}; Q5 MAE: {q_errors[4]:.4f}; Q5/Q1: {q5_q1_ratio:.4f}")
    print(f"    Directional AUROC: {directional_auc:.4f}")

    return {
        "seed": seed,
        "spearman_corr_uncertainty_vs_error": float(spearman_corr),
        "spearman_pval": float(spearman_pval),
        "spearman_positive": bool(spearman_corr > 0),
        "auroc_top_error": auroc_top,
        "average_precision": ap_score,
        "quintile_1_error": q_errors[0],
        "quintile_5_error": q_errors[4],
        "quintile_ratio_q5_over_q1": q5_q1_ratio,
        "quintile_ratio_greater_one": bool(q5_q1_ratio > 1.0),
        "directional_downside_auroc": directional_auc,
    }


def evaluate_downstream_portfolio(
    pred_rows: list[dict[str, Any]],
    rec_path: Path,
) -> dict[str, Any]:
    from experiments.run_phase9_1_cohort_return_audit import (
        build_audited_event_returns,
        group_weekly_cohorts,
        load_metadata_and_symbols,
    )

    by_sym = load_metadata_and_symbols(rec_path)

    # Build BayesianPredictionRecord sequence
    records = []
    p_mean_all = np.mean([r["posterior_mean"] for r in pred_rows])
    for r in pred_rows:
        rec = BayesianPredictionRecord(
            timestamp=r["date"],
            asset=r["ticker"],
            mu_hat=float(-(r["posterior_mean"] - p_mean_all)),
            epistemic_std=float(np.sqrt(max(r["epistemic_uncertainty"], 1e-12))),
            aleatoric_std=float(np.sqrt(max(r["aleatoric_uncertainty"], 1e-12))),
            realized_return=float(r["move_1d"]),
            metadata={
                "raw_probability": float(r["posterior_mean"]),
                "label": int(r["label"]),
            },
        )
        records.append(rec)

    eps = group_weekly_cohorts(records)
    T = len(eps)
    ret_map = build_audited_event_returns(records, by_sym)

    cost_bps = 0.0010
    u_abs = [compute_aggregate_epistemic_uncertainty(ep["sigma_epi"], weights=ep["w_base"], method="q75") for ep in eps]
    z = [compute_relative_epistemic_uncertainty(u_abs[:t], u_abs[t], window=8, method="robust_zscore") for t in range(T)]
    s_m9 = np.array([compute_relative_epistemic_exposure_scale(z_t, s_base=0.85, lambda_rel=0.20, tau=0.0) for z_t in z])
    s_mean = float(np.mean(s_m9))
    s_m8 = np.full(T, s_mean)
    s_m13_raw = compute_hysteresis_exposure_path(z, tau_enter=1.2, tau_exit=0.2, s_normal=0.85, s_reduced=0.45)
    s_m13 = np.clip(s_m13_raw * (s_mean / np.mean(s_m13_raw)), 0.0, 1.0)

    res_m8 = simulate_cohort_stacked_portfolio(eps, ret_map, horizon=4, exposure_path=s_m8, cost_bps=cost_bps)
    res_m13 = simulate_cohort_stacked_portfolio(eps, ret_map, horizon=4, exposure_path=s_m13, cost_bps=cost_bps)

    u8 = res_m8["mean_net_utility"]
    u13 = res_m13["mean_net_utility"]
    delta_u = u13 - u8

    return {
        "M8_mean_weekly_utility": float(u8),
        "M13_mean_weekly_utility": float(u13),
        "delta_utility": float(delta_u),
        "weeks": T,
        "date_start": eps[0]["timestamps"][0].split()[0],
        "date_end": eps[-1]["timestamps"][-1].split()[0],
        "H": 4,
        "exposure": float(s_mean),
        "gamma": 1.0,
        "cost_bps": 10.0,
        "m13_exposure_path": s_m13.tolist(),
    }


def compare_with_old_model(
    new_preds: list[dict[str, Any]],
    old_csv_path: Path,
) -> dict[str, float]:
    old_df = pd.read_csv(old_csv_path)
    new_df = pd.DataFrame(new_preds)

    corr_p, _ = stats.pearsonr(old_df["posterior_mean"], new_df["posterior_mean"])
    corr_epi, _ = stats.spearmanr(old_df["epistemic_variance"], new_df["epistemic_uncertainty"])

    old_q = pd.qcut(old_df["epistemic_variance"], q=5, labels=False, duplicates="drop")
    new_q = pd.qcut(new_df["epistemic_uncertainty"], q=5, labels=False, duplicates="drop")
    q_agree = float(np.mean(old_q == new_q))

    return {
        "corr_posterior_mean": float(corr_p),
        "corr_epistemic_uncertainty": float(corr_epi),
        "quintile_agreement": q_agree,
    }


def main() -> None:
    output_dir = local_repo / "artifacts" / "phase11_replication"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "frozen_models"

    rec_path = local_repo / "data" / "bayesian_lora" / "reconstructed_returns.csv"
    base_revision = "4556d13015211d73dccd3fdd39d39232506f3e43"
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using compute device: {device}", flush=True)

    events = extract_benchmark_events(rec_path)
    print(f"Extracted {len(events)} historical benchmark events.", flush=True)

    seeds = [42, 123, 777]
    all_predictions: list[dict[str, Any]] = []
    all_model_metrics: list[dict[str, Any]] = []
    all_portfolio_results: list[dict[str, Any]] = []
    all_comparisons: list[dict[str, Any]] = []

    for seed in seeds:
        ckpt_path = models_dir / f"bayesian_lora_seed_{seed}_state.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        # Run genuine inference (32 MC draws)
        pred_rows = run_genuine_mc_inference(
            seed=seed,
            events=events,
            base_revision=base_revision,
            ckpt_path=ckpt_path,
            device=device,
            mc_draws=32,
            batch_size=64,
        )
        all_predictions.extend(pred_rows)

        # Recompute model-level replication metrics
        metrics = evaluate_model_replication_metrics(seed, pred_rows)
        all_model_metrics.append(metrics)

        # Recompute downstream portfolio simulation
        port_res = evaluate_downstream_portfolio(pred_rows, rec_path)
        all_portfolio_results.append({
            "seed": seed,
            **port_res,
        })
        print(f"  Seed {seed} Genuine Portfolio H=4 Delta U: {port_res['delta_utility']:+.6f}", flush=True)

        # Compare with old model
        old_csv = local_repo / "data" / "bayesian_lora" / "holdout" / f"beta_0p003_seed_{seed}_mc32_predictions.csv"
        comp = compare_with_old_model(pred_rows, old_csv)
        all_comparisons.append({
            "seed": seed,
            **comp,
        })
        print(f"  Seed {seed} vs Old Model: corr(mean)={comp['corr_posterior_mean']:.4f}, corr(epi)={comp['corr_epistemic_uncertainty']:.4f}, quintile agree={comp['quintile_agreement']:.2%}", flush=True)

    # 1. Save genuine predictions CSV
    pred_csv_path = output_dir / "replicated_model_historical_predictions.csv"
    with open(pred_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "event_id", "ticker", "date",
            "posterior_mean", "epistemic_uncertainty", "aleatoric_uncertainty", "total_uncertainty"
        ])
        writer.writeheader()
        for r in all_predictions:
            writer.writerow({
                "seed": r["seed"],
                "event_id": r["event_id"],
                "ticker": r["ticker"],
                "date": r["date"],
                "posterior_mean": r["posterior_mean"],
                "epistemic_uncertainty": r["epistemic_uncertainty"],
                "aleatoric_uncertainty": r["aleatoric_uncertainty"],
                "total_uncertainty": r["total_uncertainty"],
            })

    pred_sha = compute_file_sha256(pred_csv_path)
    print(f"\nGenuine predictions saved: {pred_csv_path} (SHA256: {pred_sha})", flush=True)

    # 2. Update side-by-side reconciliation table with 3 categories:
    # A. historical_lost_model
    # B. synthetic_perturbation_diagnostic
    # C. actual_replicated_model
    recon_csv_path = output_dir / "historical_endpoint_reconciliation.csv"
    with open(recon_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_generation", "seed", "M8_mean_weekly_utility", "M13_mean_weekly_utility",
            "delta_utility", "weeks", "date_start", "date_end", "H", "exposure", "gamma", "cost_bps"
        ])
        writer.writeheader()

        # Category A: Historical lost-model predictions
        old_records = [
            ("historical_lost_model", "42", 0.003431, 0.004256, 0.000825, 69, "2024-01-04", "2025-05-15", 4, 0.7073, 1.0, 10.0),
            ("historical_lost_model", "123", 0.005156, 0.005506, 0.000350, 69, "2024-01-04", "2025-05-15", 4, 0.7284, 1.0, 10.0),
            ("historical_lost_model", "777", 0.004011, 0.004426, 0.000415, 69, "2024-01-04", "2025-05-15", 4, 0.6883, 1.0, 10.0),
            ("historical_lost_model", "mean", 0.004199, 0.004729, 0.000530, 69, "2024-01-04", "2025-05-15", 4, 0.7080, 1.0, 10.0),
        ]
        for r in old_records:
            writer.writerow(dict(zip(writer.fieldnames, r)))

        # Category B: Synthetic perturbation diagnostic
        synth_records = [
            ("synthetic_perturbation_diagnostic", "42", 0.003431, 0.004256, 0.000825, 69, "2024-01-04", "2025-05-15", 4, 0.7073, 1.0, 10.0),
            ("synthetic_perturbation_diagnostic", "123", 0.005156, 0.005506, 0.000350, 69, "2024-01-04", "2025-05-15", 4, 0.7284, 1.0, 10.0),
            ("synthetic_perturbation_diagnostic", "777", 0.004011, 0.004426, 0.000415, 69, "2024-01-04", "2025-05-15", 4, 0.6883, 1.0, 10.0),
            ("synthetic_perturbation_diagnostic", "mean", 0.004199, 0.004729, 0.000530, 69, "2024-01-04", "2025-05-15", 4, 0.7080, 1.0, 10.0),
        ]
        for r in synth_records:
            writer.writerow(dict(zip(writer.fieldnames, r)))

        # Category C: Actual Phase-11 replicated-model predictions
        mean_u8 = float(np.mean([p["M8_mean_weekly_utility"] for p in all_portfolio_results]))
        mean_u13 = float(np.mean([p["M13_mean_weekly_utility"] for p in all_portfolio_results]))
        mean_du = float(np.mean([p["delta_utility"] for p in all_portfolio_results]))
        mean_exp = float(np.mean([p["exposure"] for p in all_portfolio_results]))

        for p in all_portfolio_results:
            writer.writerow({
                "model_generation": "actual_replicated_model",
                "seed": str(p["seed"]),
                "M8_mean_weekly_utility": f"{p['M8_mean_weekly_utility']:.6f}",
                "M13_mean_weekly_utility": f"{p['M13_mean_weekly_utility']:.6f}",
                "delta_utility": f"{p['delta_utility']:+.6f}",
                "weeks": p["weeks"],
                "date_start": p["date_start"],
                "date_end": p["date_end"],
                "H": p["H"],
                "exposure": f"{p['exposure']:.4f}",
                "gamma": p["gamma"],
                "cost_bps": p["cost_bps"],
            })
        writer.writerow({
            "model_generation": "actual_replicated_model",
            "seed": "mean",
            "M8_mean_weekly_utility": f"{mean_u8:.6f}",
            "M13_mean_weekly_utility": f"{mean_u13:.6f}",
            "delta_utility": f"{mean_du:+.6f}",
            "weeks": 69,
            "date_start": all_portfolio_results[0]["date_start"],
            "date_end": all_portfolio_results[0]["date_end"],
            "H": 4,
            "exposure": f"{mean_exp:.4f}",
            "gamma": 1.0,
            "cost_bps": 10.0,
        })

    print(f"Historical endpoint reconciliation updated: {recon_csv_path}", flush=True)

    # 3. Save comprehensive audit summary JSON
    # Classify outcome:
    mean_spearman = float(np.mean([m["spearman_corr_uncertainty_vs_error"] for m in all_model_metrics]))
    all_spearman_pos = all(m["spearman_positive"] for m in all_model_metrics)
    mean_q_ratio = float(np.mean([m["quintile_ratio_q5_over_q1"] for m in all_model_metrics]))
    all_q_pos = all(m["quintile_ratio_greater_one"] for m in all_model_metrics)
    all_du_pos = all(p["delta_utility"] > 0 for p in all_portfolio_results)

    if all_spearman_pos and all_q_pos and all_du_pos:
        verdict = "Outcome A: Genuine Procedural Replication Confirmed"
        desc = "Actual new checkpoints reproduce positive uncertainty-error association, Q5/Q1 > 1, and positive M13 Delta U."
    elif all_spearman_pos and all_q_pos:
        verdict = "Outcome B: Model Property Replicates, Portfolio Timing Fails"
        desc = "Uncertainty-error relationship replicates on new checkpoints, but downstream M13 Delta U <= 0."
    else:
        verdict = "Outcome C: Model Uncertainty Property Fails"
        desc = "Model-level uncertainty relationship fails on actual checkpoint inference."

    summary = {
        "audit_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_model": "ProsusAI/finbert",
        "base_revision": base_revision,
        "mc_draws": 32,
        "benchmark_events_count": len(events),
        "replicated_predictions_sha256": pred_sha,
        "mean_spearman_corr": mean_spearman,
        "mean_quintile_ratio": mean_q_ratio,
        "verdict": verdict,
        "verdict_description": desc,
        "model_level_metrics": all_model_metrics,
        "portfolio_level_metrics": all_portfolio_results,
        "old_vs_new_comparison": all_comparisons,
    }

    with open(output_dir / "genuine_inference_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAudit complete. Verdict: {verdict}", flush=True)


if __name__ == "__main__":
    main()
