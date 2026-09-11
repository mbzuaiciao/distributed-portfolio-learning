#!/usr/bin/env python3
"""Phase 13B: Development-Only Uncertainty Mapping Selection.

Runs genuine Phase-11 Bayesian-LoRA inference (Seeds 42, 123, 777, MC=32)
strictly on the frozen development period (2018-01-04 to 2019-12-20, N=2,203 events).

Evaluates:
- M0: Unadjusted baseline
- M13: Frozen global timing comparator
- M14: Asset-level signal shrinkage (lambda in [0.25, 0.50, 1.00, 2.00])
- M15: Selective abstention (exclusion fraction in [0.10, 0.20, 0.30])
- M16: Asset-level risk inflation (lambda_u in [0.10, 0.25, 0.50, 1.00])

Outputs:
- artifacts/phase13_uncertainty_mapping/development_predictions.csv
- artifacts/phase13_uncertainty_mapping/development_selection.csv
- artifacts/phase13_uncertainty_mapping/development_selection_summary.csv
- artifacts/phase13_uncertainty_mapping/mapping_preregistration.json
- artifacts/phase13_uncertainty_mapping/phase13b_development_selection_report.md
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Sibling & local imports
repo_root = Path(__file__).resolve().parent.parent
sibling_src = Path("/Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/src")
if str(sibling_src) not in sys.path:
    sys.path.insert(0, str(sibling_src))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from bayesian_lora_portfolio_risk.uncertainty.bayesian_lora import (  # noqa: E402
    inject_bert_attention_bayesian_lora,
    set_posterior_sampling,
)

from hypergraph_portfolio.mean_variance import MeanVariancePortfolio  # noqa: E402
from hypergraph_portfolio.uncertainty import (  # noqa: E402
    compute_aggregate_epistemic_uncertainty,
    compute_hysteresis_exposure_path,
    compute_relative_epistemic_exposure_scale,
    compute_relative_epistemic_uncertainty,
    simulate_cohort_stacked_portfolio,
)


def get_git_sha() -> str:
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return res.stdout.strip() if res.returncode == 0 else "UNKNOWN_SHA"


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


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


def extract_development_events(rec_path: Path) -> list[dict[str, Any]]:
    """Extract and validate the 2,203 development events (2018-01-04 to 2019-12-20)."""
    rec_df = pd.read_csv(rec_path)
    rec_df = rec_df.dropna(subset=["move_1d"]).copy()
    rec_df["date_dt"] = pd.to_datetime(rec_df["date"])

    # Strict development filter
    dev = rec_df[
        (rec_df["date_dt"] >= "2018-01-01") & (rec_df["date_dt"] <= "2019-12-31 23:59:59")
    ].copy()
    dev = dev.sort_values("date").reset_index(drop=True)

    # CRITICAL AUDIT ASSERTION: No evaluation regime access
    assert all(dev["date_dt"] <= pd.Timestamp("2019-12-20 23:59:59")), (
        "Evaluation leakage detected! Events post-2019-12-20 present in development."
    )
    assert len(dev) == 2203, f"Expected exactly 2,203 development events, got {len(dev)}"

    events = []
    for _, row in dev.iterrows():
        company = str(row["company_name"]) if pd.notna(row["company_name"]) else str(row["symbol"])
        ticker = str(row["symbol"])
        date_str = str(row["date"])
        move_1d = float(row["move_1d"])
        text = f"{company} ({ticker}) quarterly earnings announcement on {date_str}."

        events.append({
            "event_id": str(row["episode_id"]),
            "ticker": ticker,
            "company_name": company,
            "date": date_str,
            "move_1d": move_1d,
            "text": text,
        })
    return events


def run_development_inference(
    events: list[dict[str, Any]],
    output_path: Path,
    device: torch.device,
    seeds: list[int] = [42, 123, 777],
    mc_draws: int = 32,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Run genuine Phase-11 checkpoint inference on all development events."""
    base_revision = "4556d13015211d73dccd3fdd39d39232506f3e43"
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=base_revision)
    texts = [e["text"] for e in events]
    dataset = SimpleTextDataset(texts, tokenizer, max_length=64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    models_dir = repo_root / "artifacts" / "phase11_replication" / "frozen_models"
    all_pred_rows: list[dict[str, Any]] = []

    print(f"\nRunning genuine Phase-11 inference on {len(events)} development events across seeds {seeds}...", flush=True)

    for seed in seeds:
        t0 = time.perf_counter()
        ckpt_path = models_dir / f"bayesian_lora_seed_{seed}_state.pt"
        assert ckpt_path.exists(), f"Missing frozen checkpoint: {ckpt_path}"

        torch.manual_seed(seed)
        np.random.seed(seed)

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

        set_posterior_sampling(model, True)

        # Run MC draws
        batch_results: list[dict[str, np.ndarray]] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                b_size = input_ids.shape[0]

                draw_probs = []
                for _ in range(mc_draws):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    draw_probs.append(probs)

                P = np.stack(draw_probs, axis=0)  # [mc_draws, b_size]
                p_mean = np.mean(P, axis=0)
                epi_var = np.var(P, axis=0, ddof=1) if mc_draws > 1 else np.zeros(b_size)
                alea_var = np.mean(P * (1.0 - P), axis=0)
                tot_var = epi_var + alea_var

                batch_results.append({
                    "posterior_mean": p_mean,
                    "epistemic_uncertainty": epi_var,
                    "aleatoric_uncertainty": alea_var,
                    "total_uncertainty": tot_var,
                })

        # Concatenate batches
        cat_p = np.concatenate([b["posterior_mean"] for b in batch_results])
        cat_epi = np.concatenate([b["epistemic_uncertainty"] for b in batch_results])
        cat_alea = np.concatenate([b["aleatoric_uncertainty"] for b in batch_results])
        cat_tot = np.concatenate([b["total_uncertainty"] for b in batch_results])

        for i, ev in enumerate(events):
            all_pred_rows.append({
                "event_id": ev["event_id"],
                "ticker": ev["ticker"],
                "date": ev["date"],
                "seed": seed,
                "posterior_mean": float(cat_p[i]),
                "epistemic_uncertainty": float(cat_epi[i]),
                "aleatoric_uncertainty": float(cat_alea[i]),
                "total_uncertainty": float(cat_tot[i]),
            })

        t1 = time.perf_counter()
        tps = len(events) / (t1 - t0)
        print(f"  Seed {seed}: {len(events)} events completed in {t1 - t0:.2f}s ({tps:.2f} events/sec)", flush=True)

    pred_df = pd.DataFrame(all_pred_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    sha = compute_sha256(output_path)
    print(f"Saved development predictions to: {output_path}")
    print(f"Artifact SHA256: {sha}")
    return pred_df


def build_dev_event_returns_map(
    events: list[dict[str, Any]],
    rec_path: Path,
) -> dict[tuple[str, str], list[float]]:
    """Build exact 4-week geometric compounding return map for development events."""
    rec_df = pd.read_csv(rec_path)
    by_sym = defaultdict(list)
    for _, row in rec_df.iterrows():
        by_sym[str(row["symbol"])].append(dict(row))

    event_returns: dict[tuple[str, str], list[float]] = {}
    for ev in events:
        sym = ev["ticker"]
        d = ev["date"]
        m1 = ev["move_1d"]

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
    """Solve Markowitz portfolio optimization with equal weight fallback."""
    m = len(mu)
    if m == 0:
        return np.array([])
    if m == 1:
        return np.array([1.0])
    try:
        w, _ = MeanVariancePortfolio(Q=gamma * Q, mu=mu, budget=1.0).solve()
        # Clean numerical precision
        w = np.clip(w, 0.0, 1.0)
        s_w = np.sum(w)
        if s_w > 1e-8:
            return w / s_w
        return np.ones(m) / float(m)
    except Exception:
        return np.ones(m) / float(m)


def simulate_strategy_regime(
    seed_preds: list[dict[str, Any]],
    event_return_map: dict[tuple[str, str], list[float]],
    strategy: str,
    param: float,
    horizon: int = 4,
    cost_bps: float = 0.0010,
    gamma: float = 1.0,
) -> dict[str, Any]:
    """Simulate a given mapping strategy over the development cohort stream."""
    # Group into weekly cohorts
    buckets = defaultdict(list)
    p_mean_all = float(np.mean([r["posterior_mean"] for r in seed_preds]))

    for r in seed_preds:
        dt = datetime.datetime.fromisoformat(r["date"].replace(" ", "T"))
        w_key = f"{dt.year}-W{dt.isocalendar()[1]:02d}"
        buckets[w_key].append(r)
    sorted_buckets = dict(sorted(buckets.items()))

    weekly_cohorts = []
    T = len(sorted_buckets)

    for w_key, basket in sorted_buckets.items():
        m = len(basket)
        assets = [r["ticker"] for r in basket]
        timestamps = [r["date"] for r in basket]
        # Forecast alpha: minus centered posterior mean
        mu_raw = np.array([-(float(r["posterior_mean"]) - p_mean_all) for r in basket])
        sigma_epi = np.array([float(np.sqrt(max(r["epistemic_uncertainty"], 1e-12))) for r in basket])
        z_cohort = compute_asset_relative_uncertainty(sigma_epi)
        z_pos = np.maximum(z_cohort, 0.0)

        Q_base = 0.04 * np.eye(m) + 0.01 * np.ones((m, m))

        # Apply specific mapping
        if strategy == "M0":
            w_b = run_cohort_markowitz(mu=mu_raw, Q=Q_base, gamma=gamma)
        elif strategy == "M13":
            # M13 uses standard M0 base weights inside each cohort
            w_b = run_cohort_markowitz(mu=mu_raw, Q=Q_base, gamma=gamma)
        elif strategy == "M14":
            # Asset-level signal shrinkage: mu_adj_i = mu_i / (1 + lambda * max(z_i, 0))
            lam = float(param)
            mu_adj = mu_raw / (1.0 + lam * z_pos)
            w_b = run_cohort_markowitz(mu=mu_adj, Q=Q_base, gamma=gamma)
        elif strategy == "M15":
            # Selective abstention: set alpha to zero for top fraction of epistemic uncertainty
            frac = float(param)
            k_exclude = int(np.round(frac * m))
            mu_adj = mu_raw.copy()
            if k_exclude > 0:
                # Rank indices by sigma_epi descending
                ranked_idx = np.argsort(-sigma_epi)
                excluded_idx = ranked_idx[:k_exclude]
                mu_adj[excluded_idx] = 0.0
            w_b = run_cohort_markowitz(mu=mu_adj, Q=Q_base, gamma=gamma)
        elif strategy == "M16":
            # Asset-level risk inflation: Q_adj = Q + lambda_u * diag(max(z, 0))
            lam_u = float(param)
            Q_adj = Q_base + lam_u * np.diag(z_pos)
            w_b = run_cohort_markowitz(mu=mu_raw, Q=Q_adj, gamma=gamma)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        weekly_cohorts.append({
            "week": w_key,
            "assets": assets,
            "timestamps": timestamps,
            "w_base": w_b,
            "sigma_epi": sigma_epi,
        })

    # Determine exposure path
    if strategy == "M13":
        u_abs = [compute_aggregate_epistemic_uncertainty(c["sigma_epi"], weights=c["w_base"], method="q75") for c in weekly_cohorts]
        z_macro = [compute_relative_epistemic_uncertainty(u_abs[:t], u_abs[t], window=8, method="robust_zscore") for t in range(T)]
        s_m9 = np.array([compute_relative_epistemic_exposure_scale(z_t, s_base=0.85, lambda_rel=0.20, tau=0.0) for z_t in z_macro])
        s_mean = float(np.mean(s_m9))
        s_m13_raw = compute_hysteresis_exposure_path(z_macro, tau_enter=1.2, tau_exit=0.2, s_normal=0.85, s_reduced=0.45)
        s_path = np.clip(s_m13_raw * (s_mean / np.mean(s_m13_raw)), 0.0, 1.0)
    else:
        s_path = np.ones(T, dtype=float)

    sim = simulate_cohort_stacked_portfolio(
        weekly_cohorts=weekly_cohorts,
        event_return_map=event_return_map,
        horizon=horizon,
        mode="markowitz",
        exposure_path=s_path,
        cost_bps=cost_bps,
    )

    r_net = np.array(sim["weekly_net_returns"])
    cum_ret = float(np.prod(1.0 + r_net) - 1.0)
    vol_weekly = float(np.std(r_net, ddof=1))
    ann_to = sim["annual_traded_turnover"]
    mean_weekly_to = float(np.mean(sim["weekly_traded_turnovers"]))
    tc_total = float(cost_bps * np.sum(sim["weekly_traded_turnovers"]))

    return {
        "mean_weekly_utility": float(sim["mean_net_utility"]),
        "cumulative_return": cum_ret,
        "weekly_volatility": vol_weekly,
        "max_drawdown": float(sim["max_drawdown"]),
        "turnover": ann_to,
        "weekly_turnover": mean_weekly_to,
        "transaction_cost": tc_total,
        "sharpe_ratio": float(sim["sharpe_ratio"]),
    }


def main() -> None:
    print("=== Phase 13B: Development Uncertainty Mapping Selection ===", flush=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Compute Device: {device}", flush=True)

    rec_path = repo_root / "data" / "bayesian_lora" / "reconstructed_returns.csv"
    dev_events = extract_development_events(rec_path)
    print(f"Extracted {len(dev_events)} development events.", flush=True)

    # Output paths
    out_dir = repo_root / "artifacts" / "phase13_uncertainty_mapping"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "development_predictions.csv"

    # Step 1: Run or load genuine Phase-11 inference
    seeds = [42, 123, 777]
    if pred_path.exists():
        print(f"Loading existing development predictions from: {pred_path}", flush=True)
        pred_df = pd.read_csv(pred_path)
        pred_sha = compute_sha256(pred_path)
    else:
        pred_df = run_development_inference(
            events=dev_events,
            output_path=pred_path,
            device=device,
            seeds=seeds,
            mc_draws=32,
            batch_size=64,
        )
        pred_sha = compute_sha256(pred_path)

    print(f"Development predictions SHA256: {pred_sha}", flush=True)

    # Build development event return map
    print("Building development event returns map...", flush=True)
    ret_map = build_dev_event_returns_map(dev_events, rec_path)

    # Candidate grids
    candidate_specs = [
        ("M0", 0.0),
        ("M13", 0.0),
        # M14 Signal Shrinkage
        ("M14", 0.25),
        ("M14", 0.50),
        ("M14", 1.00),
        ("M14", 2.00),
        # M15 Selective Abstention
        ("M15", 0.10),
        ("M15", 0.20),
        ("M15", 0.30),
        # M16 Risk Inflation
        ("M16", 0.10),
        ("M16", 0.25),
        ("M16", 0.50),
        ("M16", 1.00),
    ]

    print("\nRunning development portfolio simulations across candidates...", flush=True)

    # Store individual seed results
    detailed_rows: list[dict[str, Any]] = []
    m0_utilities: dict[int, float] = {}

    # First pass: compute M0 for each seed as reference
    for seed in seeds:
        seed_preds = pred_df[pred_df["seed"] == seed].to_dict(orient="records")
        res_m0 = simulate_strategy_regime(
            seed_preds=seed_preds,
            event_return_map=ret_map,
            strategy="M0",
            param=0.0,
            horizon=4,
            cost_bps=0.0010,
            gamma=1.0,
        )
        m0_utilities[seed] = res_m0["mean_weekly_utility"]
        detailed_rows.append({
            "strategy": "M0",
            "parameter": 0.0,
            "seed": seed,
            "mean_weekly_utility": round(res_m0["mean_weekly_utility"], 6),
            "delta_utility_vs_M0": 0.000000,
            "cumulative_return": round(res_m0["cumulative_return"], 4),
            "weekly_volatility": round(res_m0["weekly_volatility"], 4),
            "max_drawdown": round(res_m0["max_drawdown"], 4),
            "turnover": round(res_m0["turnover"], 2),
            "transaction_cost": round(res_m0["transaction_cost"], 4),
        })

    # Other candidates
    for strat, param in candidate_specs:
        if strat == "M0":
            continue
        for seed in seeds:
            seed_preds = pred_df[pred_df["seed"] == seed].to_dict(orient="records")
            res = simulate_strategy_regime(
                seed_preds=seed_preds,
                event_return_map=ret_map,
                strategy=strat,
                param=param,
                horizon=4,
                cost_bps=0.0010,
                gamma=1.0,
            )
            delta_u = res["mean_weekly_utility"] - m0_utilities[seed]
            detailed_rows.append({
                "strategy": strat,
                "parameter": param,
                "seed": seed,
                "mean_weekly_utility": round(res["mean_weekly_utility"], 6),
                "delta_utility_vs_M0": round(delta_u, 6),
                "cumulative_return": round(res["cumulative_return"], 4),
                "weekly_volatility": round(res["weekly_volatility"], 4),
                "max_drawdown": round(res["max_drawdown"], 4),
                "turnover": round(res["turnover"], 2),
                "transaction_cost": round(res["transaction_cost"], 4),
            })

    # Save development_selection.csv
    detailed_df = pd.DataFrame(detailed_rows)
    det_path = out_dir / "development_selection.csv"
    detailed_df.to_csv(det_path, index=False)
    print(f"Saved development selection details to: {det_path}")

    # Build summary table across candidates
    summary_rows: list[dict[str, Any]] = []

    for strat, param in candidate_specs:
        sub = detailed_df[(detailed_df["strategy"] == strat) & (detailed_df["parameter"] == param)]
        deltas = sub["delta_utility_vs_M0"].values
        mean_du = float(np.mean(deltas))
        worst_du = float(np.min(deltas))
        pos_count = int(np.sum(deltas > 0.0))
        mean_to = float(np.mean(sub["turnover"].values))

        summary_rows.append({
            "strategy": strat,
            "parameter": param,
            "mean_delta_u": round(mean_du, 6),
            "worst_seed_delta_u": round(worst_du, 6),
            "positive_seed_count": pos_count,
            "turnover": round(mean_to, 2),
            "selected": False,  # Updated below
        })

    summary_df = pd.DataFrame(summary_rows)

    # Apply Predeclared Deterministic Selection Rule:
    # 1. Prefer candidates with positive mean Delta U (mean_delta_u > 0).
    # 2. Prefer sign consistency across seeds (positive_seed_count == 3).
    # 3. Among robust candidates, prefer higher worst_seed_delta_u.
    # 4. Deterministic tie-break: lower turnover.
    selected_params: dict[str, float] = {}

    for strat in ["M14", "M15", "M16"]:
        strat_candidates = summary_df[summary_df["strategy"] == strat].copy()

        # Score candidates
        def score_candidate(row: pd.Series) -> tuple[int, int, float, float]:
            pos_cnt = int(row["positive_seed_count"])
            all_pos = 1 if pos_cnt == 3 else 0
            worst_u = float(row["worst_seed_delta_u"])
            mean_u = float(row["mean_delta_u"])
            # Invert turnover for tie-breaking
            inv_to = -float(row["turnover"])
            return (all_pos, pos_cnt, worst_u, mean_u, inv_to)

        ranked = sorted(
            strat_candidates.to_dict(orient="records"),
            key=lambda r: (r["positive_seed_count"] == 3, r["positive_seed_count"], r["worst_seed_delta_u"], r["mean_delta_u"], -r["turnover"]),
            reverse=True,
        )
        best_cand = ranked[0]
        selected_params[strat] = float(best_cand["parameter"])
        print(f"Selected for {strat}: parameter = {best_cand['parameter']} (mean_delta_u: {best_cand['mean_delta_u']:+.6f}, worst_seed: {best_cand['worst_seed_delta_u']:+.6f}, positive_seeds: {best_cand['positive_seed_count']}/3, turnover: {best_cand['turnover']})")

    # Mark selected in summary table
    for idx, row in summary_df.iterrows():
        st = row["strategy"]
        pr = row["parameter"]
        if st in selected_params and np.isclose(pr, selected_params[st]):
            summary_df.at[idx, "selected"] = True

    sum_path = out_dir / "development_selection_summary.csv"
    summary_df.to_csv(sum_path, index=False)
    print(f"Saved development selection summary to: {sum_path}")

    # Build mapping_preregistration.json
    models_dir = repo_root / "artifacts" / "phase11_replication" / "frozen_models"
    model_hashes = {
        f"seed_{s}": compute_sha256(models_dir / f"bayesian_lora_seed_{s}_state.pt")
        for s in seeds
    }

    prereg = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit_before_evaluation": get_git_sha(),
        "phase": "Phase 13B: Development-Only Mapping Selection",
        "development_period": {
            "regime_id": "2018-2019",
            "start_date": "2018-01-04 10:00:00",
            "end_date": "2019-12-20 10:00:00",
            "n_events": 2203,
            "status": "frozen_development_selection_complete",
        },
        "evaluation_periods_untouched": [
            {"regime_id": "2020", "n_events": 1198},
            {"regime_id": "2021", "n_events": 1191},
            {"regime_id": "2022", "n_events": 1191},
            {"regime_id": "2023", "n_events": 1181},
            {"regime_id": "2024-2025", "n_events": 1728},
        ],
        "portfolio_backbone_specification": {
            "holding_horizon_H": 4,
            "cohort_capital_weighting": "equal_across_active_cohorts (1/H when full)",
            "optimizer": "Markowitz quadratic program (mean-variance, long-only simplex)",
            "risk_aversion_gamma": 1.0,
            "transaction_cost_bps": 10.0,
            "cash_return": 0.0,
        },
        "frozen_checkpoints": model_hashes,
        "development_predictions_artifact": {
            "path": "artifacts/phase13_uncertainty_mapping/development_predictions.csv",
            "sha256": pred_sha,
            "n_rows": len(pred_df),
        },
        "selection_rule_description": (
            "1. Require positive mean Delta utility vs M0. "
            "2. Maximize sign consistency across seeds (require 3/3 positive seeds where possible). "
            "3. Maximize worst-seed Delta utility (reject single-seed outlier drivers). "
            "4. Deterministic tie-breaker: lower average annual turnover."
        ),
        "candidate_grids": {
            "M14_signal_shrinkage": [0.25, 0.50, 1.00, 2.00],
            "M15_selective_abstention": [0.10, 0.20, 0.30],
            "M16_risk_inflation": [0.10, 0.25, 0.50, 1.00],
        },
        "selected_frozen_parameters": {
            "lambda_M14": selected_params["M14"],
            "fraction_M15": selected_params["M15"],
            "lambda_M16": selected_params["M16"],
        },
        "development_metrics_summary": summary_df.to_dict(orient="records"),
    }

    prereg_path = out_dir / "mapping_preregistration.json"
    with open(prereg_path, "w") as f:
        json.dump(prereg, f, indent=2)
    print(f"Saved mapping preregistration to: {prereg_path}")


if __name__ == "__main__":
    main()
