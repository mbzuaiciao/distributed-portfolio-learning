"""Frozen Phase 16.1 recipe reproduction and Phase 9.1 cohort revalidation.

Commands: --stage train (MPS/CUDA), --stage evaluate (CPU), --stage all.
Never alters historical artifacts. Preregistration must exist in Git unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from experiments.run_phase9_1_cohort_return_audit import (
    build_audited_event_returns,
    group_weekly_cohorts,
    load_metadata_and_symbols,
)
from hypergraph_portfolio.bayesian_lora_adapter import (
    BayesianPredictionRecord,
    validate_prediction_records,
)
from hypergraph_portfolio.uncertainty import (
    build_cohort_stacked_weights,
    build_portfolio_ledger,
    compute_canonical_portfolio_metrics,
    simulate_cohort_stacked_portfolio,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/phase18_clean_cohort_revalidation"
META = ROOT / "data/bayesian_lora/reconstructed_returns.csv"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_freeze():
    path = OUT / "preregistration.json"
    rel = str(path.relative_to(ROOT))
    freeze_sha = git("log", "-1", "--format=%H", "--", rel)
    if not freeze_sha:
        raise ValueError("Preregistration must be committed before the main run")
    committed = subprocess.check_output(["git", "show", f"{freeze_sha}:{rel}"], cwd=ROOT)
    if committed != path.read_bytes():
        raise ValueError("Preregistration differs from committed freeze")
    cfg = json.loads(committed)
    for rel, expected in cfg["file_hashes"].items():
        if sha256(ROOT / rel) != expected:
            raise ValueError(f"Frozen source changed: {rel}")
    return cfg, freeze_sha


def clean_text(row):
    """Only allow-listed causal metadata can enter tokenization."""
    co = row["company_name"] if pd.notna(row["company_name"]) else row["symbol"]
    return f"{co} ({row['symbol']}) quarterly earnings announcement on {row['date']}."


def load_events(cfg):
    events = pd.read_csv(META).dropna(subset=["move_1d"]).copy()
    events["date_dt"] = pd.to_datetime(events["date"])
    # Match Phase16.1's ordering and chronological split exactly.
    events = events.sort_values("date_dt").reset_index(drop=True)
    if events.duplicated(["episode_id"]).any() or events.duplicated(["date", "symbol"]).any():
        raise ValueError("Event identities are not unique")
    pre = events[events.date_dt <= pd.Timestamp(cfg["data"]["train_end"])].copy()
    cutoff = float(pre.move_1d.quantile(cfg["data"]["target_quantile"]))
    pre["label"] = (pre.move_1d <= cutoff).astype(int)
    pre["clean_text"] = pre.apply(clean_text, axis=1)
    n = math.floor(len(pre) * cfg["data"]["train_fraction"])
    train, valid = pre.iloc[:n].copy(), pre.iloc[n:].copy()
    oos = events[events.date_dt >= pd.Timestamp("2022-01-01")].copy()
    oos["label"] = (oos.move_1d <= cutoff).astype(int)
    oos["clean_text"] = oos.apply(clean_text, axis=1)
    assert train.date_dt.max() < valid.date_dt.min()
    assert valid.date_dt.max() < oos.date_dt.min()
    for df in (train, valid, oos):
        if df.clean_text.str.contains(r"1-day move|move_1d|move_30d", case=False).any():
            raise ValueError("Outcome token in clean input")
    return events, train, valid, oos, cutoff


def resolve_historical_universe(events, split, seed=42):
    """Use native return to resolve legacy timestamp ties, then freeze real identities.

    This is identity reconstruction only. Returns never enter features or weights.
    """
    old = pd.read_csv(
        ROOT / f"data/bayesian_lora/{split}/beta_0p003_seed_{seed}_mc32_predictions.csv"
    )
    by_date = {d: g for d, g in events.groupby("date")}
    ids = []
    for row in old.itertuples():
        candidates = by_date[row.date]
        match = candidates[np.isclose(candidates.move_1d, row.move_1d, atol=1e-12, rtol=0)]
        if len(match) != 1:
            raise ValueError(f"Ambiguous native identity at {row.date}")
        ids.append(match.episode_id.iloc[0])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicated historical event")
    return ids


def frame_hash(frame, columns):
    return hashlib.sha256(frame[columns].to_csv(index=False).encode()).hexdigest()


def train_predictions(cfg, freeze_sha, device_name, only_seed=None, cloud_cpu=False):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Reuse the exact Phase16 implementation, not Phase17's checkpoint selector.
    from experiments.run_phase16_deterministic_control import inject_deterministic_lora

    _, train, valid, oos, cutoff = load_events(cfg)
    spec = cfg["model"]
    device_name = device_name or (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    if device_name == "cpu" and not cloud_cpu:
        raise RuntimeError("CPU training exceeds runtime budget; select cloud GPU")
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["identifier"], revision=spec["revision"], local_files_only=True
    )

    class TextDataset(Dataset):
        def __init__(self, df, training):
            self.encoded = tokenizer(
                df.clean_text.tolist(),
                padding=True,
                truncation=True,
                max_length=spec["max_length"],
                return_tensors="pt",
            )
            self.labels = df.label.to_numpy() if training else None

        def __len__(self):
            return len(self.encoded["input_ids"])

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.encoded.items()}
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
            return item

    train_ds = TextDataset(train, True)
    oos_ds = TextDataset(oos, False)
    loader = DataLoader(train_ds, batch_size=spec["train_batch_size"], shuffle=True)
    infer_loader = DataLoader(oos_ds, batch_size=spec["inference_batch_size"], shuffle=False)
    manifest_path = OUT / (
        f"training_manifest_seed_{only_seed}.json"
        if only_seed is not None
        else "training_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    all_predictions = []
    for seed in [only_seed] if only_seed is not None else spec["seeds"]:
        if seed not in spec["seeds"]:
            raise ValueError("Seed outside frozen matrix")
        existing = next((m for m in manifest if m["seed"] == seed), None)
        pred_path = OUT / f"clean_predictions_seed_{seed}.csv"
        checkpoint = OUT / f"clean_phase16_1_seed_{seed}_state.pt"
        if existing:
            if (
                existing["preregistration_sha256"] != sha256(OUT / "preregistration.json")
                or sha256(checkpoint) != existing["checkpoint_sha256"]
                or sha256(pred_path) != existing["predictions_sha256"]
            ):
                raise ValueError("Resume hash mismatch")
            all_predictions.append(pd.read_csv(pred_path))
            continue
        torch.manual_seed(seed)
        base = AutoModelForSequenceClassification.from_pretrained(
            spec["identifier"],
            revision=spec["revision"],
            num_labels=2,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        inject_deterministic_lora(
            base, rank=spec["rank"], alpha=spec["alpha"], dropout=spec["dropout"]
        )
        for name, p in base.named_parameters():
            p.requires_grad = "lora_" in name or "classifier" in name
        model = base.to(device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=spec["lr"],
            weight_decay=spec["weight_decay"],
        )
        t0 = time.monotonic()
        losses = []
        for epoch in range(spec["epochs"]):
            model.train()
            epoch_loss = []
            for i, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss.append(float(loss.detach().cpu()))
                if seed == spec["seeds"][0] and epoch == 0 and i == 9:
                    elapsed = time.monotonic() - t0
                    estimate = elapsed / 10 * len(loader) * spec["epochs"] * len(spec["seeds"]) + 90
                    speed = {
                        "ten_batch_seconds": elapsed,
                        "estimated_total_seconds": estimate,
                        "device": device_name,
                        "batches_per_epoch": len(loader),
                    }
                    (OUT / "runtime_preflight.json").write_text(json.dumps(speed, indent=2) + "\n")
                    print("RUNTIME PREFLIGHT", speed, flush=True)
                    if (
                        device_name == "mps"
                        and estimate > cfg["execution"]["cloud_if_estimate_exceeds_seconds"]
                    ):
                        raise RuntimeError("Local projection exceeds 20 minutes; cloud required")
                if (i + 1) % 100 == 0:
                    print(f"Seed {seed} epoch {epoch + 1} batch {i + 1}/{len(loader)}", flush=True)
            losses.append(float(np.mean(epoch_loss)))
        state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if "lora_" in k or "classifier" in k
        }
        torch.save(state, checkpoint)
        train_seconds = time.monotonic() - t0
        model.eval()
        p_list = []
        t1 = time.monotonic()
        with torch.no_grad():
            for batch in infer_loader:
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                p_list.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        pred = oos[["episode_id", "symbol", "date", "label"]].copy()
        pred["seed"] = seed
        pred["p_clean"] = p_list
        pred.to_csv(pred_path, index=False)
        row = {
            "seed": seed,
            "recipe": spec,
            "freeze_git_sha": freeze_sha,
            "execution_git_sha": git("rev-parse", "HEAD"),
            "device": device_name,
            "preregistration_sha256": sha256(OUT / "preregistration.json"),
            "training_text_sha256": frame_hash(train, ["episode_id", "clean_text", "label"]),
            "validation_text_sha256": frame_hash(valid, ["episode_id", "clean_text", "label"]),
            "inference_text_sha256": frame_hash(oos, ["episode_id", "clean_text"]),
            "checkpoint_sha256": sha256(checkpoint),
            "predictions_sha256": sha256(pred_path),
            "train_seconds": train_seconds,
            "inference_seconds": time.monotonic() - t1,
            "train_rows": len(train),
            "valid_rows": len(valid),
            "oos_rows": len(oos),
            "label_cutoff": cutoff,
            "training_losses": losses,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cpu_threads": torch.get_num_threads(),
            "execution_backend": "dedicated cloud CPU" if cloud_cpu else device_name,
            "runner_sha256": sha256(Path(__file__)),
        }
        manifest.append(row)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        all_predictions.append(pred)
        print(f"Completed fixed recipe seed {seed}: {train_seconds:.1f}s training", flush=True)
        del model, base, optimizer
        if device_name == "mps":
            torch.mps.empty_cache()
    if only_seed is None:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            OUT / "clean_predictions.csv", index=False
        )


def records_for(df, probability, center):
    """Outcomes carried only for return evaluation; formation uses probability."""
    records = [
        BayesianPredictionRecord(
            timestamp=str(r.date),
            asset=str(r.symbol),
            mu_hat=float(center - p),
            epistemic_std=0.0,
            realized_return=float(r.move_1d),
        )
        for r, p in zip(df.itertuples(), probability)
    ]
    records.sort(key=lambda r: (r.timestamp, r.asset))
    validate_prediction_records(records)
    return records


def signal_stat(score, returns, ids):
    if np.ptp(score) < 1e-14:
        return np.array([np.nan, np.nan, *([np.nan] * 5)])
    ranks_s, ranks_r = rankdata(score), rankdata(returns)
    ic = float(np.corrcoef(ranks_s, ranks_r)[0, 1]) if np.ptp(returns) else np.nan
    order = np.lexsort((np.asarray(ids), score))
    quintiles = [float(np.mean(returns[idx])) for idx in np.array_split(order, 5)]
    return np.array([ic, quintiles[4] - quintiles[0], *quintiles])


def moving_blocks(n, block_length, draws, rng):
    # Non-circular moving blocks; calendar lattice includes empty event weeks.
    block_length = min(block_length, n)
    starts = rng.integers(0, n - block_length + 1, size=(draws, math.ceil(n / block_length)))
    return (starts[..., None] + np.arange(block_length)).reshape(draws, -1)[:, :n]


def classify(persistence, economic, mechanical, delta, integrity=True):
    if not integrity:
        return "D"
    if persistence and economic:
        return "A"
    if not persistence and mechanical and delta >= 0:
        return "B"
    if not persistence and delta < 0.0001 and not mechanical:
        return "C"
    return "D"


def evaluate(cfg, freeze_sha):
    events, train, valid, oos, cutoff = load_events(cfg)
    manifest = json.loads((OUT / "training_manifest.json").read_text())
    if {m["seed"] for m in manifest} != set(cfg["model"]["seeds"]):
        raise ValueError("All frozen seeds required")
    for m in manifest:
        seed = m["seed"]
        if (
            m["recipe"] != cfg["model"]
            or m["preregistration_sha256"] != sha256(OUT / "preregistration.json")
            or sha256(OUT / f"clean_phase16_1_seed_{seed}_state.pt") != m["checkpoint_sha256"]
            or sha256(OUT / f"clean_predictions_seed_{seed}.csv") != m["predictions_sha256"]
        ):
            raise ValueError("Invalid prediction/checkpoint provenance")
    preds = pd.concat(
        [pd.read_csv(OUT / f"clean_predictions_seed_{s}.csv") for s in cfg["model"]["seeds"]],
        ignore_index=True,
    )
    ids = resolve_historical_universe(events, "holdout")
    for seed in cfg["model"]["seeds"]:
        if resolve_historical_universe(events, "holdout", seed) != ids:
            raise ValueError("Seed universe mismatch")
    indexed = events.set_index("episode_id")
    sample = indexed.loc[ids].reset_index()
    sample = sample.sort_values(["date", "symbol"]).reset_index(drop=True)
    sample["clean_label"] = (sample.move_1d <= cutoff).astype(int)
    sample["clean_text"] = sample.apply(clean_text, axis=1)
    sample.to_csv(OUT / "event_identity_manifest.csv", index=False)
    center = float(train.label.mean())
    grouped = train.groupby("symbol").label.agg(["sum", "count"])
    risk = (grouped["sum"] + 20 * center) / (grouped["count"] + 20)
    probs = np.stack(
        [
            preds[preds.seed == s].set_index("episode_id").loc[sample.episode_id].p_clean.to_numpy()
            for s in cfg["model"]["seeds"]
        ]
    )
    baseline = sample.symbol.map(risk).fillna(center).to_numpy()
    by_sym = load_metadata_and_symbols(str(META))
    records = records_for(sample, probs[0], center)
    returns_map = build_audited_event_returns(records, by_sym)
    ages = np.array([returns_map[(str(r.date), str(r.symbol))] for r in sample.itertuples()])
    event_weeks = pd.to_datetime(sample.date).dt.to_period("W-SUN")
    calendar = pd.period_range(event_weeks.min(), event_weeks.max(), freq="W-SUN")
    week_codes = calendar.get_indexer(event_weeks)
    week_groups = [np.flatnonzero(week_codes == w) for w in range(len(calendar))]
    rng = np.random.default_rng(cfg["inference"]["bootstrap_seed"])
    blocks = moving_blocks(
        len(calendar),
        cfg["inference"]["block_length_weeks"],
        cfg["inference"]["bootstrap_replicates"],
        rng,
    )
    event_draws = [np.concatenate([week_groups[w] for w in draw]) for draw in blocks]
    views = [
        ("cumulative", h, np.prod(1 + ages[:, :h], axis=1) - 1)
        for h in cfg["data"]["diagnostic_horizons"]
    ]
    views += [("age_week", h, ages[:, h - 1]) for h in range(1, 14)]
    residual = np.prod(1 + ages[:, 1:4], axis=1) - 1
    views += [("incremental_2_4", 4, residual)]
    signal_rows = []
    for system, scores in [
        ("clean_lora", center - probs),
        ("ticker_risk", (center - baseline)[None, :]),
        ("constant_prevalence", np.full((1, len(sample)), 0.0)),
    ]:
        for kind, h, outcome in views:
            stats = np.stack(
                [signal_stat(s, outcome, sample.episode_id.to_numpy()) for s in scores]
            )
            point = np.mean(stats, axis=0)
            if system == "constant_prevalence":
                bounds = np.full((2, 2), np.nan)
            else:
                boot = [
                    np.mean(
                        [
                            signal_stat(s[idx], outcome[idx], sample.episode_id.to_numpy()[idx])[:2]
                            for s in scores
                        ],
                        axis=0,
                    )
                    for idx in event_draws
                ]
                bounds = np.quantile(boot, [0.025, 0.975], axis=0)
            signal_rows.append(
                {
                    "system": system,
                    "kind": kind,
                    "horizon": h,
                    "rank_ic": point[0],
                    "spread": point[1],
                    "ic_ci_lower": bounds[0, 0],
                    "ic_ci_upper": bounds[1, 0],
                    "spread_ci_lower": bounds[0, 1],
                    "spread_ci_upper": bounds[1, 1],
                    "seed_ic_sd": float(np.std(stats[:, 0], ddof=1)) if len(scores) > 1 else 0,
                    **{f"q{q + 1}": point[q + 2] for q in range(5)},
                }
            )
        print("Signal evaluation complete:", system, flush=True)
    signals = pd.DataFrame(signal_rows)
    signals.to_csv(OUT / "clean_signal_decay.csv", index=False)
    cohort_groups = [np.flatnonzero(event_weeks == w) for w in sorted(event_weeks.unique())]
    perm_rng = np.random.default_rng(cfg["inference"]["permutation_seed"])
    perm_values = []
    score = center - probs
    observed = float(
        np.mean([signal_stat(s, residual, sample.episode_id.to_numpy())[0] for s in score])
    )
    for _ in range(cfg["inference"]["permutation_replicates"]):
        order = np.arange(len(sample))
        for group in cohort_groups:
            order[group] = perm_rng.permutation(group)
        perm_values.append(
            np.mean(
                [signal_stat(s[order], residual, sample.episode_id.to_numpy())[0] for s in score]
            )
        )
    p_perm = float((1 + np.sum(np.asarray(perm_values) >= observed)) / (1 + len(perm_values)))
    pd.DataFrame({"incremental_2_4_rank_ic": perm_values}).to_csv(
        OUT / "permutation_null.csv", index=False
    )
    # Forecast labels are training-defined, never a holdout quantile.
    labels = sample.clean_label.to_numpy()
    controls = [
        {
            "system": system,
            "brier": float(np.mean((p - labels) ** 2)),
            "log_loss": float(
                np.mean(
                    -labels * np.log(np.clip(p, 1e-12, 1))
                    - (1 - labels) * np.log(np.clip(1 - p, 1e-12, 1))
                )
            ),
            "prevalence": float(labels.mean()),
            "prediction_mean": float(p.mean()),
        }
        for system, p in [
            ("constant_prevalence", np.full(len(sample), center)),
            ("ticker_risk", baseline),
            *[(f"clean_seed_{s}", probs[i]) for i, s in enumerate(cfg["model"]["seeds"])],
        ]
    ]
    pd.DataFrame(controls).to_csv(OUT / "forecast_controls.csv", index=False)
    portfolio_rows, ledger_rows, seed_sims, reproduction = [], [], {}, []
    for si, seed in enumerate(cfg["model"]["seeds"]):
        recs = records_for(sample, probs[si], center)
        epochs = group_weekly_cohorts(recs)
        for system in ["clean_lora", "equal_weight", "constant_prevalence", "ticker_risk"]:
            eps = (
                epochs
                if system in ["clean_lora", "equal_weight"]
                else group_weekly_cohorts(
                    records_for(
                        sample,
                        np.full(len(sample), center)
                        if system == "constant_prevalence"
                        else baseline,
                        center,
                    )
                )
            )
            mode = "equal_weight" if system == "equal_weight" else "markowitz"
            for h in cfg["portfolio"]["primary_horizons"]:
                sim = simulate_cohort_stacked_portfolio(
                    eps,
                    returns_map,
                    horizon=h,
                    mode=mode,
                    target_exposure=cfg["portfolio"]["exposure"],
                    cost_bps=cfg["portfolio"]["cost_rate"],
                )
                weights = build_cohort_stacked_weights(eps, h, mode)
                neff = [1 / sum(w * w for w in wm.values()) for wm in weights]
                positive = [sum(w > 1e-10 for w in wm.values()) for wm in weights]
                scalar = {k: v for k, v in sim.items() if isinstance(v, (float, int))}
                row = {
                    "system": system,
                    "seed": seed,
                    **scalar,
                    "mean_effective_holdings": float(np.mean(neff)),
                    "mean_positive_active_names": float(np.mean(positive)),
                    "annual_gross_risky_l1_turnover": float(
                        np.mean(sim["weekly_base_turnovers"]) * 52
                    ),
                    "annual_one_way_risky_turnover": sim["annual_traded_turnover"] / 2,
                    "gross_annual_return_at_exposure": float(
                        np.mean(sim["weekly_gross_returns"]) * cfg["portfolio"]["exposure"] * 52
                    ),
                }
                portfolio_rows.append(row)
                if system != "clean_lora":
                    continue
                seed_sims[(seed, h)] = sim
                for t, ep in enumerate(eps):
                    ledger_rows.append(
                        {
                            "seed": seed,
                            "horizon": h,
                            "week": ep["week"],
                            "gross_return_at_exposure": cfg["portfolio"]["exposure"]
                            * sim["weekly_gross_returns"][t],
                            "net_return": sim["weekly_net_returns"][t],
                            "base_turnover": sim["weekly_base_turnovers"][t],
                            "traded_turnover": sim["weekly_traded_turnovers"][t],
                            "transaction_cost": cfg["portfolio"]["cost_rate"]
                            * sim["weekly_traded_turnovers"][t],
                            "effective_holdings": neff[t],
                            "positive_active_names": positive[t],
                            "weight_overlap": sim["weekly_overlaps"][t],
                            "active_cohorts": min(t + 1, h),
                        }
                    )
                if h == 1:
                    led = build_portfolio_ledger(
                        eps,
                        np.full(len(eps), cfg["portfolio"]["exposure"]),
                        "clean_constant",
                        cost_bps=cfg["portfolio"]["cost_rate"],
                    )
                    aligned = np.array([r["net_return"] for r in led])
                    aligned[0] -= cfg["portfolio"]["cost_rate"] * cfg["portfolio"]["exposure"]
                    err = float(np.max(np.abs(aligned - np.array(sim["weekly_net_returns"]))))
                    assert err < 1e-12
                    reproduction.append({"seed": seed, "max_h1_net_return_error": err})
    pd.DataFrame(reproduction).to_csv(OUT / "h1_reproduction.csv", index=False)
    # Random ranking portfolios preserve probability amplitude and cohort identity.
    random_rng = np.random.default_rng(cfg["inference"]["random_portfolio_seed"])
    for run in range(cfg["inference"]["random_portfolio_replicates"]):
        order = np.arange(len(sample))
        for group in cohort_groups:
            order[group] = random_rng.permutation(group)
        for si, seed in enumerate(cfg["model"]["seeds"]):
            eps = group_weekly_cohorts(records_for(sample, probs[si, order], center))
            for h in cfg["portfolio"]["primary_horizons"]:
                sim = simulate_cohort_stacked_portfolio(
                    eps,
                    returns_map,
                    horizon=h,
                    target_exposure=cfg["portfolio"]["exposure"],
                    cost_bps=cfg["portfolio"]["cost_rate"],
                )
                portfolio_rows.append(
                    {
                        "system": "random_ranking",
                        "random_replicate": run,
                        "seed": seed,
                        **{k: v for k, v in sim.items() if isinstance(v, (float, int))},
                    }
                )
    portfolio = pd.DataFrame(portfolio_rows)
    portfolio.to_csv(OUT / "portfolio_seed_metrics.csv", index=False)
    means = (
        portfolio.groupby(["system", "horizon"])
        .mean(numeric_only=True)
        .drop(columns=["seed", "random_replicate"], errors="ignore")
        .reset_index()
    )
    means.to_csv(OUT / "clean_h1_vs_h4.csv", index=False)
    pd.DataFrame(ledger_rows).to_csv(OUT / "portfolio_ledger.csv", index=False)
    # Paired utility bootstrap over the same observed occupied weekly epochs.
    example = group_weekly_cohorts(records)
    # Legacy week keys use Gregorian year with ISO week; dates are mapped via actual timestamps.
    ep_weeks = [pd.Timestamp(ep["timestamps"][0]).to_period("W-SUN") for ep in example]
    ep_calendar = pd.period_range(min(ep_weeks), max(ep_weeks), freq="W-SUN")
    ep_codes = ep_calendar.get_indexer(ep_weeks)
    ep_groups = [np.flatnonzero(ep_codes == w) for w in range(len(ep_calendar))]
    boot_idx = moving_blocks(
        len(ep_calendar),
        cfg["inference"]["block_length_weeks"],
        cfg["inference"]["bootstrap_replicates"],
        np.random.default_rng(cfg["inference"]["bootstrap_seed"]),
    )
    deltas = []
    for draw in boot_idx:
        idx = np.concatenate([ep_groups[w] for w in draw])
        deltas.append(
            np.mean(
                [
                    compute_canonical_portfolio_metrics(
                        np.array(seed_sims[(s, 4)]["weekly_net_returns"])[idx]
                    )["mean_net_utility"]
                    - compute_canonical_portfolio_metrics(
                        np.array(seed_sims[(s, 1)]["weekly_net_returns"])[idx]
                    )["mean_net_utility"]
                    for s in cfg["model"]["seeds"]
                ]
            )
        )
    seed_delta = [
        seed_sims[(s, 4)]["mean_net_utility"] - seed_sims[(s, 1)]["mean_net_utility"]
        for s in cfg["model"]["seeds"]
    ]
    economic_cfg = cfg["criteria"]["economic"]
    persist_cfg = cfg["criteria"]["persistence"]
    mech_cfg = cfg["criteria"]["mechanical"]
    delta = float(np.mean(seed_delta))
    ci = np.quantile(deltas, [0.025, 0.975])
    clean = means[means.system == "clean_lora"].set_index("horizon")
    h1, h4 = clean.loc[1], clean.loc[4]
    persistence_row = signals[
        (signals.system == "clean_lora") & (signals.kind == "incremental_2_4")
    ].iloc[0]
    persistence = bool(
        persistence_row.rank_ic >= persist_cfg["incremental_2_4_rank_ic_min"]
        and persistence_row.spread >= persist_cfg["incremental_2_4_q5_q1_min"]
        and persistence_row.ic_ci_lower > 0
        and persistence_row.spread_ci_lower > 0
        and p_perm <= persist_cfg["permutation_p_max"]
    )
    economic = bool(
        delta >= economic_cfg["delta_mean_net_weekly_utility_min"]
        and ci[0] > 0
        and sum(x > 0 for x in seed_delta) >= economic_cfg["positive_seeds_min"]
    )
    to_red = float(1 - h4.annual_traded_turnover / h1.annual_traded_turnover)
    vol_red = float(1 - h4.annualized_volatility / h1.annualized_volatility)
    eff_ratio = float(h4.mean_effective_holdings / h1.mean_effective_holdings)
    mechanical = bool(
        to_red >= mech_cfg["traded_turnover_reduction_min"]
        and vol_red >= mech_cfg["net_volatility_reduction_min"]
        and eff_ratio >= mech_cfg["effective_holdings_ratio_min"]
    )
    decomposition = []
    for s in cfg["model"]["seeds"]:
        sims = [seed_sims[(s, h)] for h in [1, 4]]
        gross = [cfg["portfolio"]["exposure"] * np.array(x["weekly_gross_returns"]) for x in sims]
        net = [np.array(x["weekly_net_returns"]) for x in sims]
        dg = float(np.mean(gross[1] - gross[0]))
        saved = float(
            cfg["portfolio"]["cost_rate"]
            * np.mean(
                np.array(sims[0]["weekly_traded_turnovers"])
                - np.array(sims[1]["weekly_traded_turnovers"])
            )
        )
        variance = float(0.5 * (np.var(net[0], ddof=1) - np.var(net[1], ddof=1)))
        utility = sims[1]["mean_net_utility"] - sims[0]["mean_net_utility"]
        assert abs(utility - dg - saved - variance) < 1e-12
        decomposition.append(
            {
                "seed": s,
                "delta_gross_weekly_return": dg,
                "weekly_cost_saving": saved,
                "sample_variance_utility_gain": variance,
                "delta_net_utility": utility,
            }
        )
    pd.DataFrame(decomposition).to_csv(OUT / "decomposition.csv", index=False)
    # Quantify reconstruction fallback / gaps / terminal censoring without changing it.
    fallbacks = {"missing_30d": 0, "missing_next_quarter": 0, "corporate_action_guard": 0}
    for row in sample.itertuples():
        evs = by_sym[row.symbol]
        ix = next(i for i, e in enumerate(evs) if e["date"] == row.date)
        ev = evs[ix]
        fallbacks["missing_30d"] += not bool(ev.get("move_30d"))
        if ix + 1 >= len(evs) or not evs[ix + 1].get("d_minus_1_close"):
            fallbacks["missing_next_quarter"] += 1
        else:
            r13 = float(evs[ix + 1]["d_minus_1_close"]) / float(ev["d_minus_1_close"]) - 1
            fallbacks["corporate_action_guard"] += r13 > 2 or r13 < -0.8
    summary = {
        "outcome": classify(persistence, economic, mechanical, delta),
        "persistence_pass": persistence,
        "economic_pass": economic,
        "mechanical_pass": mechanical,
        "delta_net_weekly_utility": delta,
        "delta_utility_ci_95": ci.tolist(),
        "seed_deltas": seed_delta,
        "seed_delta_sd": float(np.std(seed_delta, ddof=1)),
        "incremental_2_4_rank_ic": float(persistence_row.rank_ic),
        "incremental_2_4_spread": float(persistence_row.spread),
        "permutation_p": p_perm,
        "turnover_reduction": to_red,
        "volatility_reduction": vol_red,
        "effective_holdings_ratio": eff_ratio,
        "holdout_events": len(sample),
        "occupied_weeks": len(example),
        "calendar_weeks": len(ep_calendar),
        "empty_calendar_weeks": len(ep_calendar) - len(example),
        "reconstruction_fallback_counts": fallbacks,
        "right_censored_last_cohorts_h4": 3,
        "training_label_cutoff": cutoff,
        "pretest_rows": len(train) + len(valid),
        "freeze_git_sha": freeze_sha,
        "execution_git_sha": git("rev-parse", "HEAD"),
        "execution_dirty_files": git("status", "--short"),
        "runner_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(OUT / "preregistration.json"),
        "qualification": "Retrospective endpoint reconstruction. Genuine observed weekly alpha cannot be confirmed without a weekly price tape. Original Phase9 model leakage unverified.",
    }
    (OUT / "outcome_classification.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame({"delta_net_weekly_utility": deltas}).to_csv(
        OUT / "paired_bootstrap.csv", index=False
    )
    old = pd.read_csv(ROOT / "artifacts/phase9_cohort_stacking/signal_decay_by_horizon.csv")
    old["benchmark_status"] = "historical/compromised-lineage; exact original leakage unverified"
    old.to_csv(OUT / "historical_signal_benchmarks.csv", index=False)
    comparison = signals[(signals.system == "clean_lora") & (signals.kind == "cumulative")].merge(
        old, left_on="horizon", right_on="horizon_weeks", how="left"
    )
    comparison.to_csv(OUT / "signal_comparison.csv", index=False)
    pd.read_csv(ROOT / "artifacts/phase9_1_cohort_return_audit/canonical_phase9_table.csv").assign(
        benchmark_status="historical/compromised-lineage; not a valid control"
    ).to_csv(OUT / "historical_portfolio_benchmarks.csv", index=False)
    report(cfg, summary, means, signals, comparison, decomposition)
    print(json.dumps(summary, indent=2), flush=True)


def md_table(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False, name=None):
        rows.append(
            "| " + " | ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + " |"
        )
    return "\n".join(rows)


def report(cfg, s, means, signals, comparison, decomposition):
    label = {
        "A": "Clean confirmation (endpoint reconstruction only)",
        "B": "Architecture only",
        "C": "No useful continuation",
        "D": "Mixed / inconclusive",
    }[s["outcome"]]
    primary = means[means.system == "clean_lora"][
        [
            "horizon",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "mean_net_utility",
            "annual_gross_risky_l1_turnover",
            "annual_base_turnover",
            "annual_traded_turnover",
            "annual_tc_drag_bps",
            "mean_positive_active_names",
            "mean_effective_holdings",
            "mean_overlap",
        ]
    ]
    text = f"""# Phase 18: Clean deterministic H=4 revalidation

**Frozen outcome {s["outcome"]} — {label}.** Persistence gate: {s["persistence_pass"]}; economic gate: {s["economic_pass"]}; mechanical gate: {s["mechanical_pass"]}.

Preregistration commit `{s["freeze_git_sha"]}` precedes all new predictions and outcomes. The collaborative-LoRA branch stays closed. This is retrospective revalidation, not prospective confirmation or horizon search. All three clean fixed-epoch Phase 16.1 recipe reproductions were run; no Bayesian checkpoint, MC sampling, uncertainty scaling or M13 overlay entered the primary experiment.

## Provenance and scope

Read `provenance_audit.md` for the source table. Phase 16.1 did not persist clean weights or event predictions; new checkpoints reproduce its exact fixed two-epoch recipe. Weight decay 0.0 is the previously selected recipe despite the name “regularized.” Phase 17's best-validation checkpoints were deliberately excluded. Original Phase 9 predictions predate leaked Phase 11 replications; exact original outcome leakage is unverified, so old results remain historical/compromised-lineage benchmarks, never valid controls.

The exact {s["holdout_events"]} historical holdout events were resolved uniquely by timestamp and native one-day return, then keyed by episode ID/symbol/timestamp. Model tokenization accepts only company/ticker/date. Labels use the pre-2022 cutoff {s["training_label_cutoff"]:.10f}; no evaluation-quantile target is used. Training/validation split is chronological. Checkpoints, prediction files, texts, raw data, source, config and Git SHAs are hashed in the freeze and training manifests.

## Primary H=1 versus H=4 portfolio table

Returns/volatility/drawdown are fractions, utility is weekly; turnover is annual L1 units and cost drag is annual basis points. Annual return is arithmetic mean times 52, not CAGR. Values are arithmetic means of separately computed seed metrics.

{md_table(primary)}

Delta net weekly utility = **{s["delta_net_weekly_utility"]:.8f}** ({s["delta_net_weekly_utility"] * 10000:.3f} bp/week), paired 95% CI [{s["delta_utility_ci_95"][0]:.8f}, {s["delta_utility_ci_95"][1]:.8f}]. Per-seed deltas: {s["seed_deltas"]}; seed SD {s["seed_delta_sd"]:.8f}. The 1000 moving-block replicates share 13-calendar-week blocks across seeds; seeds are not independent market histories.

## Predictive persistence

Continuation excludes week one: incremental compounded weeks 2–4 IC {s["incremental_2_4_rank_ic"]:.6f}, Q5-Q1 spread {s["incremental_2_4_spread"]:.6f}, one-sided within-entry-week permutation p={s["permutation_p"]:.4f}. Exact CIs and all quintiles are in `clean_signal_decay.csv`. Cumulative H=4 IC alone cannot establish post-week-one persistence.

{md_table(signals[(signals.system == "clean_lora") & (signals.kind.isin(["cumulative", "incremental_2_4"]))][["kind", "horizon", "rank_ic", "ic_ci_lower", "ic_ci_upper", "spread", "spread_ci_lower", "spread_ci_upper", "seed_ic_sd"]])}

`signal_comparison.csv` directly pairs the clean cumulative metrics with original Phase 9 values. Original Phase 9 signal calculations summed reconstructed returns and had timestamp-tie mapping problems; they are not matched clean controls. `historical_portfolio_benchmarks.csv` preserves the Phase 9.1 audited historical table.

## Decomposition: prediction, diversification, turnover and costs

A. Prediction: the above incremental gate determines persistence, independently of portfolio Sharpe.

B. Mechanics: H=4 volatility reduction {s["volatility_reduction"]:.2%}, effective-holdings ratio {s["effective_holdings_ratio"]:.3f}x. Aggregate holdings are measured by 1/sum(weight squared), after combining duplicate ticker exposures. Positive active names exclude zero optimizer weights; the historical dictionary-key count remains separately exported. Within-cohort weights are frozen at entry; cohorts each receive equal capital while active. This measures overlap/diversification under the preserved architecture; interpolated endpoint returns can also smooth volatility.

C. Turnover: H=4 traded turnover reduction {s["turnover_reduction"]:.2%}. Fees equal .001 times risky L1 volume. Base turnover excludes inception per canonical simulator, while costed traded turnover includes entry; gross risky L1 includes entry. One-way half-L1 is reported separately and never used to charge fees.

D. Net economic value: the identity below separates gross-return change, transaction-cost savings and the canonical sample-variance utility term; closure is asserted to 1e-12 for every seed. Gross-return change reflects the different held ages and cannot alone be called alpha.

{md_table(pd.DataFrame(decomposition))}

Matched-exposure equal-weight, training-prevalence, training-only ticker-risk, and 20 frozen random-ranking controls are in `clean_h1_vs_h4.csv` and `portfolio_seed_metrics.csv`. Constant ranking IC/quintiles are undefined rather than manufactured from tie order. `forecast_controls.csv` evaluates clean labels. These baselines are diagnostic; none changes continuation thresholds.

## Accounting checks and scientific limits

The existing Phase 9.1 return builder and cohort simulator were reused without modification. H=1 matches the canonical ledger after the simulator's entry fee is aligned (see `h1_reproduction.csv`); H=4 age selection, equal capital and expiration have hand-calculated tests.

Returns are **geometrically interpolated event endpoints, not observed weekly marks**: weeks 2–4 share the same per-event drift, as do weeks 5–13. Thus three positive reconstructed week ICs are not three independent persistence observations. H=1 is a one-day move treated as a weekly observation. Next-quarter endpoint fallback counts: {s["reconstruction_fallback_counts"]}. {s["occupied_weeks"]} occupied weeks span {s["calendar_weeks"]} calendar weeks, including {s["empty_calendar_weeks"]} empty weeks. Simulator horizons age by occupied buckets; terminal cohorts are censored without liquidation, exactly as historical accounting. IntrawEEK execution availability and adjusted weekly prices are not established by this data. No endpoint enters features, selection or weights; endpoints only score frozen predictions and returns.

The frozen numerical classification is not revised for these limitations. Even numerical outcome A would establish only an endpoint-reconstruction result; genuine observed multiweek alpha requires actual weekly prices and a prospectively causal execution audit. A positive architecture result without persistence is not alpha confirmation.

## Reproduction

Run `PYTHONPATH=src:. python experiments/run_phase18_clean_cohort_revalidation.py --stage train --device mps`, then the targeted tests, then `--stage evaluate`. A GPU is required for recipe reproduction; cached FinBERT is used offline. A completed seed resumes only after config/prediction/checkpoint hashes match. No new portfolio outcome is evaluated before the Git preregistration check. Execution SHA `{s["execution_git_sha"]}` and runner hash `{s["runner_sha256"]}` are saved. Validation results are not consulted to choose checkpoints.
"""
    (OUT / "scientific_report.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["train", "evaluate", "all"], default="all")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"])
    parser.add_argument(
        "--seed",
        type=int,
        choices=[42, 123, 777],
        help="Cloud worker shard; all seeds remain required for evaluation",
    )
    parser.add_argument(
        "--cloud-cpu", action="store_true", help="Dedicated cloud CPU worker, not local fallback"
    )
    args = parser.parse_args()
    cfg, freeze_sha = load_freeze()
    if args.stage in ["train", "all"]:
        train_predictions(cfg, freeze_sha, args.device, args.seed, args.cloud_cpu)
    if args.stage in ["evaluate", "all"]:
        evaluate(cfg, freeze_sha)


if __name__ == "__main__":
    main()
