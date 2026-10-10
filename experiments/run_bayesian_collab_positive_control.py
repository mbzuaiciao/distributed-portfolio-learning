"""Collaborative LoRA Positive-Control Research Harness.

Implements the experimental protocol for investigating specialist complementarity
before Bayesian uncertainty weighting, as specified in:
- docs/bayesian_collaboration_research_plan.md
- docs/bayesian_collaboration_design_amendment.md

Stages:
  --stage generate : Create and freeze synthetic DGP and splits (test labels isolated)
  --stage train    : Train all models across seeds, fit development weights, freeze checkpoints
  --stage diagnose : Evaluate pre-experiment gates (learnability & complementarity) on dev set
  --stage evaluate : Evaluate held-out test set with Holm step-down correction
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from hypergraph_portfolio.collaborative_lora import (
    combine_predictions,
    fit_simplex_weights,
)
from hypergraph_portfolio.synthetic_models import (
    CapacityMatchedMLP,
    CoLoRAMLP,
    SmallMLP,
)
from hypergraph_portfolio.synthetic_multitask import (
    DGPConfig,
    generate_dgp,
    sample_dataset,
)

# Strict thread limit to keep host machine completely responsive
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts/bayesian_collab_positive_control"


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: Path, obj: Any) -> None:
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute standard calibration and discrimination metrics."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)

    # 10-bin Expected Calibration Error
    ece = 0.0
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        mask = (p >= lo) & (p < hi)
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y_true[mask].mean()))

    brier = float(np.mean((p - y_true) ** 2))
    log_loss = float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    try:
        auroc = float(roc_auc_score(y_true, p))
    except ValueError:
        auroc = 0.5

    try:
        ap = float(average_precision_score(y_true, p))
    except ValueError:
        ap = float(y_true.mean())

    accuracy = float(np.mean((p >= 0.5) == y_true))

    return {
        "brier": brier,
        "log_loss": log_loss,
        "auroc": auroc,
        "average_precision": ap,
        "accuracy": accuracy,
        "ece": ece,
    }


def holm_step_down(
    p_values: Dict[str, float], alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """Perform Holm step-down correction on a family of hypothesis p-values.

    Threshold for step j (1-indexed): alpha / (K - j + 1).
    """
    K = len(p_values)
    sorted_items = sorted(p_values.items(), key=lambda item: item[1])

    results = {}
    stopped = False

    for rank, (method, p_val) in enumerate(sorted_items, start=1):
        threshold = alpha / (K - rank + 1)
        if not stopped and p_val <= threshold:
            rejected = True
        else:
            rejected = False
            stopped = True

        results[method] = {
            "p_value_raw": p_val,
            "rank": rank,
            "threshold": threshold,
            "rejected": rejected,
        }

    return results


def train_single_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.005,
    weight_decay: float = 0.01,
    seed: int = 42,
    model_cls: type = SmallMLP,
) -> nn.Module:
    """Train a SmallMLP or CapacityMatchedMLP model using AdamW and binary cross-entropy."""
    torch.manual_seed(seed)
    model = model_cls(in_features=x_train.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    criterion = nn.BCELoss()

    x_tensor = torch.from_numpy(x_train).float()
    y_tensor = torch.from_numpy(y_train).float()
    n = len(x_train)

    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        indices = rng.permutation(n)
        for i in range(0, n, batch_size):
            batch_idx = indices[i : i + batch_size]
            bx = x_tensor[batch_idx]
            by = y_tensor[batch_idx]

            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def train_colora_model(
    datasets: List[Tuple[np.ndarray, np.ndarray, int]],
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.005,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> CoLoRAMLP:
    """Train a CoLoRAMLP multi-task model with task_id routing."""
    torch.manual_seed(seed)
    in_features = datasets[0][0].shape[1]
    model = CoLoRAMLP(
        in_features=in_features, rank=4, alpha=8.0, tasks=len(datasets)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    criterion = nn.BCELoss()

    rng = np.random.default_rng(seed)
    model.train()

    # Pre-convert tensors
    tensors = [
        (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            tid,
        )
        for x, y, tid in datasets
    ]

    for _ in range(epochs):
        # Create shuffled task batches
        batches = []
        for x_t, y_t, tid in tensors:
            n = len(x_t)
            indices = rng.permutation(n)
            for i in range(0, n, batch_size):
                b_idx = indices[i : i + batch_size]
                batches.append((x_t[b_idx], y_t[b_idx], tid))

        rng.shuffle(batches)

        for bx, by, tid in batches:
            optimizer.zero_grad()
            preds = model(bx, task_id=tid)
            loss = criterion(preds, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def predict(
    model: nn.Module,
    x: np.ndarray,
    task_id: int = 0,
) -> np.ndarray:
    """Generate probabilistic predictions from model."""
    model.eval()
    x_t = torch.from_numpy(x).float()
    with torch.no_grad():
        if isinstance(model, CoLoRAMLP):
            preds = model(x_t, task_id=task_id).cpu().numpy()
        else:
            preds = model(x_t).cpu().numpy()
    return preds


def stage_generate(config_path: Optional[Path] = None) -> None:
    """Stage 1: Generate synthetic DGP and datasets for both regimes."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = DGPConfig()
    spec = generate_dgp(cfg)

    # Save DGP spec
    save_json(OUT_DIR / "dgp_spec.json", spec.to_dict())

    # Generate positive-control dataset
    pos_splits = sample_dataset(spec, regime="positive_control")
    np.savez_compressed(
        OUT_DIR / "splits_positive.npz",
        target_train_x=pos_splits.target_train["x"],
        target_train_y=pos_splits.target_train["y"],
        spec_A_train_x=pos_splits.specialist_A_train["x"],
        spec_A_train_y=pos_splits.specialist_A_train["y"],
        spec_B_train_x=pos_splits.specialist_B_train["x"],
        spec_B_train_y=pos_splits.specialist_B_train["y"],
        dev_x_full=pos_splits.development["x_full"],
        dev_x_A=pos_splits.development["x_A"],
        dev_x_B=pos_splits.development["x_B"],
        dev_y=pos_splits.development["y"],
        val_x_full=pos_splits.validation["x_full"],
        val_x_A=pos_splits.validation["x_A"],
        val_x_B=pos_splits.validation["x_B"],
        val_y=pos_splits.validation["y"],
        test_x_full=pos_splits.test["x_full"],
        test_x_A=pos_splits.test["x_A"],
        test_x_B=pos_splits.test["x_B"],
        test_y=pos_splits.test["y"],
    )

    # Generate negative-control dataset
    neg_splits = sample_dataset(spec, regime="negative_control")
    np.savez_compressed(
        OUT_DIR / "splits_negative.npz",
        target_train_x=neg_splits.target_train["x"],
        target_train_y=neg_splits.target_train["y"],
        spec_A_train_x=neg_splits.specialist_A_train["x"],
        spec_A_train_y=neg_splits.specialist_A_train["y"],
        spec_B_train_x=neg_splits.specialist_B_train["x"],
        spec_B_train_y=neg_splits.specialist_B_train["y"],
        dev_x_full=neg_splits.development["x_full"],
        dev_x_A=neg_splits.development["x_A"],
        dev_x_B=neg_splits.development["x_B"],
        dev_y=neg_splits.development["y"],
        val_x_full=neg_splits.validation["x_full"],
        val_x_A=neg_splits.validation["x_A"],
        val_x_B=neg_splits.validation["x_B"],
        val_y=neg_splits.validation["y"],
        test_x_full=neg_splits.test["x_full"],
        test_x_A=neg_splits.test["x_A"],
        test_x_B=neg_splits.test["x_B"],
        test_y=neg_splits.test["y"],
    )

    config_record = {
        "seeds": [42, 123, 777],
        "dgp": asdict(cfg),
        "analytical_oracle_gap": spec.analytical_oracle_gap,
        "learnability_delta": 0.015,
        "material_brier_gain": 0.005,
        "bootstrap_replicates": 1000,
        "regimes": ["positive_control", "negative_control"],
    }
    save_json(OUT_DIR / "config.json", config_record)
    print("Stage generate completed successfully.")
    print(f"Analytical oracle gap: {spec.analytical_oracle_gap:.6f}")


def stage_train(smoke: bool = False) -> None:
    """Stage 2: Train all models across seeds and fit development weights."""
    if not (OUT_DIR / "splits_positive.npz").exists():
        raise RuntimeError("Must run --stage generate first.")

    config = json.loads((OUT_DIR / "config.json").read_text())
    seeds = config["seeds"]
    epochs = 10 if smoke else 100

    frozen: Dict[str, Any] = {
        "timestamp": time.time(),
        "hardware": platform.platform(),
        "torch_version": torch.__version__,
        "smoke": smoke,
        "runs": {},
    }

    start_all = time.perf_counter()

    for regime in config["regimes"]:
        data = np.load(OUT_DIR / f"splits_{regime.split('_')[0]}.npz")
        dev_y = data["dev_y"]

        frozen["runs"][regime] = {}

        for seed in seeds:
            print(f"Training regime={regime} seed={seed}...")
            # M0: Target-only
            m0 = train_single_model(
                data["target_train_x"],
                data["target_train_y"],
                epochs=epochs,
                seed=seed,
            )
            # M2: Specialist A
            m2 = train_single_model(
                data["spec_A_train_x"],
                data["spec_A_train_y"],
                epochs=epochs,
                seed=seed + 1,
            )
            # M3: Specialist B
            m3 = train_single_model(
                data["spec_B_train_x"],
                data["spec_B_train_y"],
                epochs=epochs,
                seed=seed + 2,
            )

            # M1: Pooled (union of target, A, B)
            x_pooled = np.vstack(
                [
                    data["target_train_x"],
                    data["spec_A_train_x"],
                    data["spec_B_train_x"],
                ]
            )
            y_pooled = np.concatenate(
                [
                    data["target_train_y"],
                    data["spec_A_train_y"],
                    data["spec_B_train_y"],
                ]
            )
            m1 = train_single_model(
                x_pooled, y_pooled, epochs=epochs, seed=seed + 3
            )
            # M1-cap: Capacity-matched pooled (2,199 params, matching CoLoRA's 2,193)
            m1_cap = train_single_model(
                x_pooled, y_pooled, epochs=epochs, seed=seed + 3, model_cls=CapacityMatchedMLP
            )

            # M6: CoLoRA (multi-task training with task routing)
            m6 = train_colora_model(
                [
                    (data["target_train_x"], data["target_train_y"], 0),
                    (data["spec_A_train_x"], data["spec_A_train_y"], 1),
                    (data["spec_B_train_x"], data["spec_B_train_y"], 2),
                ],
                epochs=epochs,
                seed=seed + 4,
            )

            # Save checkpoints
            ckpt_paths = {}
            for name, m in [
                ("m0_target", m0),
                ("m1_pooled", m1),
                ("m1_cap_pooled", m1_cap),
                ("m2_specA", m2),
                ("m3_specB", m3),
                ("m6_colora", m6),
            ]:
                p = OUT_DIR / f"ckpt_{regime}_seed_{seed}_{name}.pt"
                torch.save(m.state_dict(), p)
                ckpt_paths[name] = {"path": str(p), "sha256": digest(p)}

            # Generate development predictions
            p_dev_m0 = predict(m0, data["dev_x_full"])
            p_dev_m1 = predict(m1, data["dev_x_full"])
            p_dev_m1_cap = predict(m1_cap, data["dev_x_full"])
            p_dev_m2 = predict(m2, data["dev_x_A"])
            p_dev_m3 = predict(m3, data["dev_x_B"])
            p_dev_m6 = predict(m6, data["dev_x_full"], task_id=0)

            # Fit M5 learned simplex weights on development
            p_dev_matrix = np.column_stack([p_dev_m2, p_dev_m3])
            weights = fit_simplex_weights(p_dev_matrix, dev_y)

            # M4 equal ensemble on dev
            p_dev_m4 = combine_predictions(p_dev_matrix, np.array([0.5, 0.5]))
            p_dev_m5 = combine_predictions(p_dev_matrix, weights)

            # Development metrics
            dev_metrics = {
                "m0_target": compute_metrics(dev_y, p_dev_m0),
                "m1_pooled": compute_metrics(dev_y, p_dev_m1),
                "m1_cap_pooled": compute_metrics(dev_y, p_dev_m1_cap),
                "m2_specA": compute_metrics(dev_y, p_dev_m2),
                "m3_specB": compute_metrics(dev_y, p_dev_m3),
                "m4_equal": compute_metrics(dev_y, p_dev_m4),
                "m5_learned": compute_metrics(dev_y, p_dev_m5),
                "m6_colora": compute_metrics(dev_y, p_dev_m6),
            }

            frozen["runs"][regime][str(seed)] = {
                "checkpoints": ckpt_paths,
                "learned_weights": weights.tolist(),
                "dev_metrics": dev_metrics,
                "dev_predictions": {
                    "m0": p_dev_m0.tolist(),
                    "m1": p_dev_m1.tolist(),
                    "m1_cap": p_dev_m1_cap.tolist(),
                    "m2": p_dev_m2.tolist(),
                    "m3": p_dev_m3.tolist(),
                    "m4": p_dev_m4.tolist(),
                    "m5": p_dev_m5.tolist(),
                    "m6": p_dev_m6.tolist(),
                },
            }

    frozen["total_training_seconds"] = time.perf_counter() - start_all
    save_json(OUT_DIR / "frozen.json", frozen)
    print("Stage train completed successfully.")


def stage_diagnose() -> None:
    """Stage 3: Pre-experiment diagnostics and gates on development data."""
    if not (OUT_DIR / "frozen.json").exists():
        raise RuntimeError("Must run --stage train before --stage diagnose.")

    config = json.loads((OUT_DIR / "config.json").read_text())
    frozen = json.loads((OUT_DIR / "frozen.json").read_text())
    spec_data = json.loads((OUT_DIR / "dgp_spec.json").read_text())

    data_pos = np.load(OUT_DIR / "splits_positive.npz")
    dev_y = data_pos["dev_y"]
    prev_null_brier = float(dev_y.mean() * (1.0 - dev_y.mean()))
    theta_learn = prev_null_brier - config["learnability_delta"]

    diagnostics: Dict[str, Any] = {
        "prevalence_null_brier": prev_null_brier,
        "theta_learn": theta_learn,
        "gates": {},
    }

    # Gate 1: Learnability (H2) on positive control
    learnable_seeds = []
    for seed in config["seeds"]:
        seed_data = frozen["runs"]["positive_control"][str(seed)]
        brier_A = seed_data["dev_metrics"]["m2_specA"]["brier"]
        brier_B = seed_data["dev_metrics"]["m3_specB"]["brier"]
        pass_seed = (brier_A < theta_learn) and (brier_B < theta_learn)
        learnable_seeds.append(pass_seed)

    gate_h2_pass = all(learnable_seeds)
    diagnostics["gates"]["H2_learnability"] = {
        "pass": gate_h2_pass,
        "threshold": theta_learn,
        "per_seed_pass": learnable_seeds,
    }

    # Gate 2: Complementarity (H3)
    # Check 1: Analytical oracle gap
    analytical_gap = spec_data["analytical_oracle_gap"]
    analytical_pass = analytical_gap > 0.010

    # Check 2: Conditional information regression on dev
    cond_info_passes = []
    for seed in config["seeds"]:
        preds = frozen["runs"]["positive_control"][str(seed)]["dev_predictions"]
        pA = np.array(preds["m2"])
        pB = np.array(preds["m3"])
        X = np.column_stack([pA, pB])
        clf = LogisticRegression(C=1e9, solver="lbfgs")
        clf.fit(X, dev_y)
        coefs = clf.coef_[0]
        # Both coefficients positive indicates complementary predictive signal
        cond_info_passes.append(bool(coefs[0] > 0 and coefs[1] > 0))

    gate_h3_pass = analytical_pass and any(cond_info_passes)
    diagnostics["gates"]["H3_complementarity"] = {
        "pass": gate_h3_pass,
        "analytical_gap": analytical_gap,
        "analytical_pass": analytical_pass,
        "conditional_info_passes": cond_info_passes,
    }

    overall_pass = gate_h2_pass and gate_h3_pass
    diagnostics["overall_pass"] = overall_pass

    save_json(OUT_DIR / "diagnostics_report.json", diagnostics)
    print("Stage diagnose completed.")
    print(f"H2 Learnability pass: {gate_h2_pass}")
    print(
        f"H3 Complementarity pass: {gate_h3_pass} (analytical gap: {analytical_gap:.4f})"
    )
    print(f"Overall pre-experiment gate pass: {overall_pass}")


def stage_evaluate() -> None:
    """Stage 4: Evaluate held-out test set with Holm step-down correction."""
    diag_path = OUT_DIR / "diagnostics_report.json"
    if not diag_path.exists():
        raise RuntimeError("Must run --stage diagnose before --stage evaluate.")

    diagnostics = json.loads(diag_path.read_text())
    if not diagnostics.get("overall_pass", False):
        raise RuntimeError("Pre-experiment gates FAILED. Evaluation aborted.")

    config = json.loads((OUT_DIR / "config.json").read_text())
    frozen = json.loads((OUT_DIR / "frozen.json").read_text())
    seeds = config["seeds"]
    n_boot = config["bootstrap_replicates"]

    results_summary: Dict[str, Any] = {}
    csv_rows = []

    for regime in config["regimes"]:
        data = np.load(OUT_DIR / f"splits_{regime.split('_')[0]}.npz")
        test_y = data["test_y"]
        n_test = len(test_y)

        # Reconstruct models from checkpoints
        seed_predictions: Dict[int, Dict[str, np.ndarray]] = {}

        for seed in seeds:
            seed_info = frozen["runs"][regime][str(seed)]
            ckpts = seed_info["checkpoints"]

            m0 = SmallMLP(in_features=40)
            m0.load_state_dict(
                torch.load(ckpts["m0_target"]["path"], weights_only=True)
            )

            m1 = SmallMLP(in_features=40)
            m1.load_state_dict(
                torch.load(ckpts["m1_pooled"]["path"], weights_only=True)
            )

            m1_cap = CapacityMatchedMLP(in_features=40)
            m1_cap.load_state_dict(
                torch.load(ckpts["m1_cap_pooled"]["path"], weights_only=True)
            )

            m2 = SmallMLP(in_features=40)
            m2.load_state_dict(
                torch.load(ckpts["m2_specA"]["path"], weights_only=True)
            )

            m3 = SmallMLP(in_features=40)
            m3.load_state_dict(
                torch.load(ckpts["m3_specB"]["path"], weights_only=True)
            )

            m6 = CoLoRAMLP(in_features=40)
            m6.load_state_dict(
                torch.load(ckpts["m6_colora"]["path"], weights_only=True)
            )

            p_m0 = predict(m0, data["test_x_full"])
            p_m1 = predict(m1, data["test_x_full"])
            p_m1_cap = predict(m1_cap, data["test_x_full"])
            p_m2 = predict(m2, data["test_x_A"])
            p_m3 = predict(m3, data["test_x_B"])
            p_m6 = predict(m6, data["test_x_full"], task_id=0)

            # Mixture methods
            weights = np.array(seed_info["learned_weights"])
            p_mat = np.column_stack([p_m2, p_m3])
            p_m4 = combine_predictions(p_mat, np.array([0.5, 0.5]))
            p_m5 = combine_predictions(p_mat, weights)

            # Diagnostic best single specialist (test best of M2, M3)
            brier_m2 = float(np.mean((p_m2 - test_y) ** 2))
            brier_m3 = float(np.mean((p_m3 - test_y) ** 2))
            p_m7 = p_m2 if brier_m2 <= brier_m3 else p_m3

            # Prevalence null
            prev = float(data["target_train_y"].mean())
            p_m8 = np.full(n_test, prev)

            preds = {
                "m0_target": p_m0,
                "m1_pooled": p_m1,
                "m1_cap_pooled": p_m1_cap,
                "m2_specA": p_m2,
                "m3_specB": p_m3,
                "m4_equal": p_m4,
                "m5_learned": p_m5,
                "m6_colora": p_m6,
                "m7_best_single_specialist_diag": p_m7,
                "m8_prevalence_null": p_m8,
            }
            seed_predictions[seed] = preds

            for method_name, prob in preds.items():
                m = compute_metrics(test_y, prob)
                csv_rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "method": method_name,
                        **m,
                    }
                )

        # Paired bootstrap comparison vs M0 (target-only)
        methods_to_compare = [
            "m1_pooled",
            "m1_cap_pooled",
            "m4_equal",
            "m5_learned",
            "m6_colora",
            "m7_best_single_specialist_diag",
        ]
        comparison_results = {}
        p_values_family = {}

        boot_rng = np.random.default_rng(20261010)

        for method in methods_to_compare:
            # Paired Brier differences across all seeds: Brier(M0) - Brier(Method)
            diffs = []
            for seed in seeds:
                brier_m0_sample = (seed_predictions[seed]["m0_target"] - test_y) ** 2
                brier_meth_sample = (seed_predictions[seed][method] - test_y) ** 2
                diffs.append(brier_m0_sample - brier_meth_sample)

            diff_matrix = np.array(diffs)  # (n_seeds, n_test)
            mean_gain_per_seed = diff_matrix.mean(axis=1)
            overall_mean_gain = float(diff_matrix.mean())

            # Shared-event bootstrap
            boot_means = []
            for _ in range(n_boot):
                boot_idx = boot_rng.integers(0, n_test, size=n_test)
                boot_means.append(diff_matrix[:, boot_idx].mean())

            ci_low = float(np.quantile(boot_means, 0.025))
            ci_high = float(np.quantile(boot_means, 0.975))
            # One-sided bootstrap p-value: P(gain <= 0)
            p_val = float(np.mean(np.array(boot_means) <= 0.0))

            comparison_results[method] = {
                "mean_brier_gain": overall_mean_gain,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "per_seed_gains": mean_gain_per_seed.tolist(),
                "all_seeds_positive": bool(np.all(mean_gain_per_seed > 0)),
                "p_value_one_sided": p_val,
            }

            if method in ("m4_equal", "m5_learned", "m6_colora"):
                p_values_family[method] = p_val

        # Holm step-down correction on prespecified family
        holm_decisions = holm_step_down(p_values_family, alpha=0.05)

        for method, dec in holm_decisions.items():
            comp = comparison_results[method]
            passes_continuation = bool(
                comp["mean_brier_gain"] >= config["material_brier_gain"]
                and comp["all_seeds_positive"]
                and comp["ci_95_low"] > 0
                and dec["rejected"]
            )
            comp["holm_rank"] = dec["rank"]
            comp["holm_threshold"] = dec["threshold"]
            comp["holm_rejected"] = dec["rejected"]
            comp["passes_continuation"] = passes_continuation

        results_summary[regime] = {
            "comparisons": comparison_results,
            "holm_decisions": holm_decisions,
        }

    # Save outputs
    write_csv(OUT_DIR / "test_metrics.csv", csv_rows)
    save_json(OUT_DIR / "evaluation_summary.json", results_summary)

    # Generate decision JSON
    pos_comp = results_summary["positive_control"]["comparisons"]
    neg_comp = results_summary["negative_control"]["comparisons"]

    h4a_pass = any(
        pos_comp[m]["passes_continuation"]
        for m in ("m4_equal", "m5_learned", "m6_colora")
    )
    h5_pass = not any(
        neg_comp[m]["mean_brier_gain"] >= config["material_brier_gain"]
        for m in ("m4_equal", "m5_learned", "m6_colora")
    )

    decision = {
        "H4a_positive_control_pass": h4a_pass,
        "H5_negative_control_null_holds": h5_pass,
        "conclusive": diagnostics["overall_pass"],
        "summary": results_summary,
    }
    save_json(OUT_DIR / "decision.json", decision)
    print("Stage evaluate completed.")
    print(f"H4a (Positive control pass): {h4a_pass}")
    print(f"H5 (Negative control null holds): {h5_pass}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collaborative LoRA Positive-Control Research Harness"
    )
    parser.add_argument(
        "--stage",
        choices=["generate", "train", "diagnose", "evaluate"],
        required=True,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run short training epochs for smoke testing without altering protocol.",
    )
    args = parser.parse_args()

    if args.stage == "generate":
        stage_generate()
    elif args.stage == "train":
        stage_train(smoke=args.smoke)
    elif args.stage == "diagnose":
        stage_diagnose()
    elif args.stage == "evaluate":
        stage_evaluate()


if __name__ == "__main__":
    main()
