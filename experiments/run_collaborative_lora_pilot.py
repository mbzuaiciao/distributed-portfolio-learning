"""Small real-data FinBERT collaboration pilot with a separate frozen evaluation stage.

Run from repository root with PYTHONPATH=src. Requires torch/transformers/sklearn.
Never imports legacy experiment modules or consumes their labels/checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from hypergraph_portfolio.collaborative_lora import (
    combine_predictions,
    fit_simplex_weights,
    subspace_similarity,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/collaborative_lora_mechanism_pilot"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose(rows, cap, salt):
    return sorted(sorted(rows, key=lambda r: hashlib.sha256(
        (salt + r["event_id"]).encode()).hexdigest())[:cap], key=lambda r: r["event_timestamp"])


def build_splits(config):
    """ID-only selection: discard legacy label and all unused metadata immediately."""
    with (ROOT / "artifacts/phase17_clean_bayesian/clean_text_manifest.csv").open() as f:
        rows = [{k: row[k] for k in ("event_id", "ticker", "event_timestamp", "split", "clean_text")}
                for row in csv.DictReader(f)]
    if len({r["event_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate event IDs")
    if any(token in r["clean_text"].lower() for r in rows
           for token in ("move_1d", "move_30d", "1-day move", "downside")):
        raise ValueError("Outcome field in input")
    splits = {}
    ticker_sets = [set(v) for v in config["tasks"].values()]
    if sum(map(len, ticker_sets)) != len(set.union(*ticker_sets)):
        raise ValueError("Task ticker sets overlap")
    for task, tickers in config["tasks"].items():
        splits[task] = choose([r for r in rows if r["split"] == "train" and r["ticker"] in tickers],
                              config["train_cap_per_task"], config["selection_salt"])
    val = sorted([r for r in rows if r["split"] == "validation" and
                  r["ticker"] in config["tasks"]["target"]], key=lambda r: r["event_timestamp"])
    middle_date = val[len(val) // 2]["event_timestamp"][:10]
    splits["development"] = choose([r for r in val if r["event_timestamp"][:10] < middle_date],
                                     config["development_cap"], config["selection_salt"])
    splits["validation"] = choose([r for r in val if r["event_timestamp"][:10] >= middle_date],
                                    config["validation_cap"], config["selection_salt"])
    splits["test"] = choose([r for r in rows if r["split"] == "clean_oos" and
                             r["ticker"] in config["tasks"]["target"]],
                            config["test_cap"], config["selection_salt"])
    flattened = [r["event_id"] for group in splits.values() for r in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Split overlap")
    for name, group in splits.items():
        if not group:
            raise ValueError(f"Empty split {name}")
    training = [r for name in config["tasks"] for r in splits[name]]
    assert max(r["event_timestamp"] for r in training) < min(r["event_timestamp"] for r in splits["development"])
    assert max(r["event_timestamp"] for r in splits["development"]) < min(r["event_timestamp"] for r in splits["validation"])
    assert max(r["event_timestamp"] for r in splits["validation"]) < min(r["event_timestamp"] for r in splits["test"])
    return splits


def labels_for(rows, cutoff):
    """Only convert outcomes for requested IDs, called for test solely in evaluation."""
    wanted = {r["event_id"] for r in rows}
    values = {}
    with (ROOT / "data/bayesian_lora/reconstructed_returns.csv").open() as f:
        for row in csv.DictReader(f):
            if row["episode_id"] in wanted:
                if row["episode_id"] in values:
                    raise ValueError("Duplicate requested return")
                values[row["episode_id"]] = int(float(row["move_1d"]) <= cutoff)
    return np.array([values[r["event_id"]] for r in rows])


def metrics(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score
    p = np.clip(p, 1e-7, 1 - 1e-7)
    ece = 0.0
    for lo in np.arange(0, 1, 0.1):
        mask = (p >= lo) & (p < lo + 0.1)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return {"brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "auroc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "accuracy": float(np.mean((p >= 0.5) == y)), "ece": float(ece)}


def token_similarity(source, target):
    from collections import Counter
    a = Counter(" ".join(r["clean_text"] for r in source).lower().split())
    b = Counter(" ".join(r["clean_text"] for r in target).lower().split())
    return sum(a[k] * b[k] for k in a) / np.sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))


def setup(config):
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"], local_files_only=True)
    return torch, tokenizer


def fresh(config, seed, tasks=1):
    import torch
    from transformers import AutoModelForSequenceClassification

    from hypergraph_portfolio.collaborative_lora_layers import inject_lora
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"], revision=config["revision"], num_labels=2,
        ignore_mismatched_sizes=True, local_files_only=True)
    for p in model.parameters():
        p.requires_grad = False
    inject_lora(model, config, tasks)
    return model


def encode(tokenizer, rows, config):
    return tokenizer([r["clean_text"] for r in rows], padding="max_length", truncation=True,
                     max_length=config["max_length"], return_tensors="pt")


def predict(model, encoded, config):
    import torch
    model.eval()
    start = time.perf_counter()
    values = []
    with torch.no_grad():
        for offset in range(0, len(encoded["input_ids"]), config["batch_size"]):
            batch = {k: v[offset:offset + config["batch_size"]] for k, v in encoded.items()}
            values.extend(torch.softmax(model(**batch).logits, -1)[:, 1].numpy())
    return np.array(values), time.perf_counter() - start


def train(config):
    from hypergraph_portfolio.collaborative_lora_layers import set_task
    if (OUT / "frozen.json").exists():
        raise RuntimeError("Frozen run exists; use a new artifact directory for a changed protocol")
    torch, tokenizer = setup(config)
    splits = build_splits(config)
    save_json(OUT / "splits.json", splits)
    train_names = list(config["tasks"])
    # Crucially no test encoding or test labels in this stage.
    enc = {name: encode(tokenizer, rows, config) for name, rows in splits.items() if name != "test"}
    labels = {name: labels_for(rows, config["label_return_cutoff"]) for name, rows in splits.items() if name != "test"}
    records, signals, freeze = [], [], {"seeds": {}}
    start_run = time.perf_counter()
    for seed in config["seeds"]:
        states, dev, valid, costs = {}, {}, {}, {}
        for method in train_names + ["pooled", "colora"]:
            started = time.perf_counter()
            collaborative = method == "colora"
            model = fresh(config, seed, len(train_names) if collaborative else 1)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=config["learning_rate"], weight_decay=config["weight_decay"])
            domains = train_names if method in ("pooled", "colora") else [method]
            generator = np.random.default_rng(seed)
            steps = 0
            for _ in range(config["epochs"]):
                batches = []
                for name in domains:
                    order = generator.permutation(len(labels[name]))
                    batches.extend((name, order[i:i + config["batch_size"]])
                                   for i in range(0, len(order), config["batch_size"]))
                generator.shuffle(batches)
                model.train()
                for name, ids in batches:
                    if time.perf_counter() - start_run > config["runtime_limit_seconds"]:
                        raise RuntimeError("Local runtime budget exceeded; preserve shards and move to cloud")
                    set_task(model, train_names.index(name) if collaborative else 0)
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(**{k: v[ids] for k, v in enc[name].items()},
                                 labels=torch.tensor(labels[name][ids])).loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
                    optimizer.step()
                    steps += 1
            training_seconds = time.perf_counter() - started
            set_task(model, 0)
            state = {k: v.detach().clone() for k, v in model.state_dict().items()
                     if "lora_" in k or "lambdas" in k or "classifier" in k}
            states[method] = state
            path = OUT / f"seed_{seed}_{method}.pt"
            torch.save(state, path)
            dev[method], _ = predict(model, enc["development"], config)
            valid[method], _ = predict(model, enc["validation"], config)
            costs[method] = {"total_parameters": sum(p.numel() for p in model.parameters()),
                             "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                             "training_seconds": training_seconds, "gradient_steps": steps,
                             "training_examples": config["epochs"] * sum(len(labels[n]) for n in domains),
                             "checkpoint_sha256": digest(path)}
            records.append({"seed": seed, "method": method, **metrics(labels["validation"], valid[method]),
                            **costs[method]})
            print(f"seed={seed} method={method} train={training_seconds:.1f}s", flush=True)
            del model, optimizer
        experts = train_names[1:]
        pdev = np.column_stack([dev[n] for n in experts])
        gate_started = time.perf_counter()
        w = fit_simplex_weights(pdev, labels["development"])
        loo = {name: fit_simplex_weights(np.delete(pdev, j, axis=1), labels["development"]).tolist()
               for j, name in enumerate(experts)}
        gate_seconds = time.perf_counter() - gate_started
        freeze["seeds"][str(seed)] = {"gate_and_loo_seconds": gate_seconds, "weights": w.tolist(), "loo_weights": loo, "models": costs,
                                     "prevalence": float(labels["target"].mean())}
        for j, name in enumerate(experts):
            bkeys = [k for k in states[name] if k.endswith("lora_b")]
            similarity = np.mean([subspace_similarity(states[name][k].numpy(), states["target"][k].numpy()) for k in bkeys])
            p = valid[name]
            signals.append({"seed": seed, "specialist": name,
                            "validation_performance": -metrics(labels["validation"], p)["brier"],
                            "data_similarity": float(token_similarity(splits[name], splits["target"])),
                            "subspace_similarity": float(similarity),
                            "confidence": float(np.mean(abs(p - 0.5))),
                            "negative_ece": -metrics(labels["validation"], p)["ece"],
                            "negative_entropy": float(np.mean(p*np.log(p)+(1-p)*np.log(1-p))),
                            "learned_weight": float(w[j])})
    write_csv(OUT / "validation_metrics.csv", records)
    write_csv(OUT / "explanatory_signals.csv", signals)
    freeze.update({"config_sha256": digest(OUT / "config.json"), "splits_sha256": digest(OUT / "splits.json"),
                   "runner_sha256": digest(__file__),
                   "core_sha256": digest(ROOT / "src/hypergraph_portfolio/collaborative_lora.py"),
                   "layers_sha256": digest(ROOT / "src/hypergraph_portfolio/collaborative_lora_layers.py"),
                   "signals_sha256": digest(OUT / "explanatory_signals.csv"),
                   "training_wall_seconds": time.perf_counter() - start_run,
                   "hardware": platform.platform(), "torch": torch.__version__,
                   "test_labels_accessed": False})
    save_json(OUT / "frozen.json", freeze)


def evaluate(config):
    torch, tokenizer = setup(config)
    freeze = json.loads((OUT / "frozen.json").read_text())
    for file, key in [(OUT / "config.json", "config_sha256"), (OUT / "splits.json", "splits_sha256"),
                      (Path(__file__), "runner_sha256"),
                      (ROOT / "src/hypergraph_portfolio/collaborative_lora.py", "core_sha256"),
                      (ROOT / "src/hypergraph_portfolio/collaborative_lora_layers.py", "layers_sha256"),
                      (OUT / "explanatory_signals.csv", "signals_sha256")]:
        assert digest(file) == freeze[key], f"Frozen input changed: {file}"
    if (OUT / "test_metrics.csv").exists():
        raise RuntimeError("Test already evaluated; do not adapt protocol to these results")
    splits = json.loads((OUT / "splits.json").read_text())
    rows = splits["test"]
    y = labels_for(rows, config["label_return_cutoff"])
    enc = encode(tokenizer, rows, config)
    experts = list(config["tasks"])[1:]
    results, predictions, contributions, comparisons = [], [], [], []
    all_p = {}
    for seed in config["seeds"]:
        specification = freeze["seeds"][str(seed)]
        p, timings = {}, {}
        for method, info in specification["models"].items():
            path = OUT / f"seed_{seed}_{method}.pt"
            assert digest(path) == info["checkpoint_sha256"]
            model = fresh(config, seed, len(config["tasks"]) if method == "colora" else 1)
            missing, unexpected = model.load_state_dict(torch.load(path, weights_only=True), strict=False)
            assert not unexpected and not any("lora_" in k or "classifier" in k or "lambdas" in k for k in missing)
            p[method], timings[method] = predict(model, enc, config)
            del model
        matrix = np.column_stack([p[n] for n in experts])
        p["equal"] = combine_predictions(matrix, np.ones(len(experts)) / len(experts))
        p["learned"] = combine_predictions(matrix, specification["weights"])
        oracle = min(experts, key=lambda n: np.mean((p[n] - y)**2))
        p["oracle_diagnostic"] = p[oracle]
        p["prevalence_null"] = np.full(len(y), specification["prevalence"])
        for j, name in enumerate(experts):
            reduced = np.delete(matrix, j, axis=1)
            for rule in ("equal", "learned"):
                weights = np.ones(len(experts)-1)/(len(experts)-1) if rule == "equal" else specification["loo_weights"][name]
                without = combine_predictions(reduced, weights)
                contributions.append({"seed": seed, "specialist": name, "rule": rule,
                                      "marginal_brier_gain": float(np.mean((without-y)**2) - np.mean((p[rule]-y)**2))})
        all_p[seed] = p
        for method, prob in p.items():
            mixed = method in ("equal", "learned")
            info = specification["models"].get(oracle if method == "oracle_diagnostic" else method)
            if mixed:
                infos = [specification["models"][n] for n in experts]
                base_params = infos[0]["total_parameters"] - infos[0]["trainable_parameters"]
                info = {"total_parameters": base_params + sum(i["trainable_parameters"] for i in infos) + (3 if method == "learned" else 0),
                        "trainable_parameters": sum(i["trainable_parameters"] for i in infos) + (3 if method == "learned" else 0),
                        "training_seconds": sum(i["training_seconds"] for i in infos) + (specification["gate_and_loo_seconds"] if method == "learned" else 0),
                        "gradient_steps": sum(i["gradient_steps"] for i in infos),
                        "training_examples": sum(i["training_examples"] for i in infos)}
            if info is None:
                info = dict.fromkeys(("total_parameters", "trainable_parameters", "training_seconds", "gradient_steps", "training_examples"), 0)
            inference = sum(timings[n] for n in experts) if mixed else timings.get(oracle if method == "oracle_diagnostic" else method, 0)
            results.append({"seed": seed, "method": method, **metrics(y, prob),
                            **{k: info[k] for k in ("total_parameters", "trainable_parameters", "training_seconds", "gradient_steps", "training_examples")},
                            "inference_seconds": inference, "passes_per_event": 3 if mixed else (0 if method == "prevalence_null" else 1),
                            "weights": json.dumps(specification["weights"] if method == "learned" else [1/3]*3 if method == "equal" else []),
                            "oracle_selected": oracle if method == "oracle_diagnostic" else ""})
            predictions.extend({"seed": seed, "method": method, "event_id": r["event_id"], "date": r["event_timestamp"],
                                "y": int(label), "probability": float(probability)} for r, label, probability in zip(rows, y, prob))
    # Shared resampled calendar weeks across seeds: seeds are not independent events.
    weeks = np.array([str(date.fromisoformat(r["event_timestamp"][:10]).isocalendar()[:2]) for r in rows])
    clusters = [np.flatnonzero(weeks == week) for week in np.unique(weeks)]
    rng = np.random.default_rng(20261003)
    for method in ("equal", "learned", "colora"):
        for baseline in ("target", "pooled", "equal"):
            gains = np.stack([(all_p[s][baseline]-y)**2 - (all_p[s][method]-y)**2 for s in config["seeds"]])
            boot = [gains[:, np.concatenate([clusters[j] for j in rng.integers(len(clusters), size=len(clusters))])].mean()
                    for _ in range(config["bootstrap_replicates"])]
            comparisons.append({"method": method, "baseline": baseline, "mean_brier_gain": float(gains.mean()),
                                "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975)),
                                "positive_seeds": int(np.sum(gains.mean(axis=1) > 0)),
                                "passes_continue_criterion": bool(gains.mean() >= config["practical_brier_gain"] and np.quantile(boot, .025) > 0 and np.all(gains.mean(axis=1) > 0))})
    signals = list(csv.DictReader((OUT / "explanatory_signals.csv").open()))
    correlations = []
    for rule in ("equal", "learned"):
        for key in ("validation_performance", "data_similarity", "subspace_similarity", "confidence", "negative_ece", "negative_entropy", "learned_weight"):
            seed_rhos = []
            for seed in config["seeds"]:
                x = [float(next(s[key] for s in signals if int(s["seed"]) == seed and s["specialist"] == n)) for n in experts]
                z = [next(c["marginal_brier_gain"] for c in contributions if c["seed"] == seed and c["specialist"] == n and c["rule"] == rule) for n in experts]
                rho = float(spearmanr(x, z).statistic) if np.std(x) > 1e-12 and np.std(z) > 1e-12 else None
                correlations.append({"rule": rule, "signal": key, "seed": seed, "spearman": rho})
                if rho is not None:
                    seed_rhos.append(rho)
            correlations.append({"rule": rule, "signal": key, "seed": "mean", "spearman": float(np.mean(seed_rhos)) if seed_rhos else None})
    summary = []
    for method in all_p[config["seeds"][0]]:
        subset = [r for r in results if r["method"] == method]
        summary.append({"method": method, **{f"{key}_{stat}": float(np.mean([r[key] for r in subset]) if stat == "mean" else np.var([r[key] for r in subset], ddof=1))
                                            for key in ("brier", "auroc", "log_loss", "ece", "average_precision", "accuracy", "training_seconds", "inference_seconds") for stat in ("mean", "variance")}})
    write_csv(OUT / "test_metrics.csv", results)
    write_csv(OUT / "test_predictions.csv", predictions)
    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "marginal_contributions.csv", contributions)
    write_csv(OUT / "paired_comparisons.csv", comparisons)
    write_csv(OUT / "signal_correlations.csv", correlations)
    save_json(OUT / "decision.json", {"continue": any(all(c["passes_continue_criterion"] for c in comparisons if c["method"] == m) for m in ("learned", "colora")),
                                     "test_rows": len(y), "test_prevalence": float(y.mean()), "weeks": len(clusters)})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "evaluate"], required=True)
    args = parser.parse_args()
    config = json.loads((OUT / "config.json").read_text())
    (train if args.stage == "train" else evaluate)(config)
