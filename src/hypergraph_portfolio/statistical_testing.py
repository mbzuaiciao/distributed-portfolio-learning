"""Statistical Testing & Paired Resampling for Confirmatory Evaluation.

Implements:
1. Paired instance-level cluster bootstrap over held-out observations.
2. Exact pairing preservation across all models and training seeds.
3. Computation of empirical percentile confidence intervals and p-values.
4. Step-down Holm multiplicity adjustment for the secondary confirmatory family.
5. Strict distinction between effect-size threshold pass and statistical significance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ContrastResult:
    contrast_name: str
    point_estimate: float
    bootstrap_mean: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    p_value_raw: float
    p_value_holm: float
    effect_size_threshold: float
    effect_size_pass: bool
    statistical_significance_pass: bool
    confirmatory_pass: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contrast_name": self.contrast_name,
            "point_estimate": float(self.point_estimate),
            "bootstrap_mean": float(self.bootstrap_mean),
            "standard_error": float(self.standard_error),
            "ci_95": [float(self.ci_lower), float(self.ci_upper)],
            "p_value_raw": float(self.p_value_raw),
            "p_value_holm": float(self.p_value_holm),
            "effect_size_threshold": float(self.effect_size_threshold),
            "effect_size_pass": bool(self.effect_size_pass),
            "statistical_significance_pass": bool(self.statistical_significance_pass),
            "confirmatory_pass": bool(self.confirmatory_pass),
        }


def compute_instance_brier(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Compute per-instance Brier loss (p - y)^2."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)
    return (y_prob - y_true) ** 2


def paired_instance_bootstrap(
    y_true: np.ndarray,
    preds_A: Dict[int, np.ndarray],
    preds_B: Dict[int, np.ndarray],
    n_bootstraps: int = 2000,
    seed: int = 20261012,
    effect_size_threshold: float = 0.0,
    higher_is_better: bool = False,  # False for loss reduction (Brier(B) - Brier(A))
) -> ContrastResult:
    """Run paired cluster-bootstrap resampling across shared evaluation instances.

    Args:
        y_true: 1D array of ground truth binary labels (N,).
        preds_A: dict mapping seed -> predicted probabilities (N,). Evaluated method.
        preds_B: dict mapping seed -> predicted probabilities (N,). Baseline method.
        n_bootstraps: number of paired bootstrap resamples.
        seed: PRNG seed for deterministic resampling.
        effect_size_threshold: minimum required point estimate gain.
        higher_is_better: if False, metric is loss reduction: Loss(B) - Loss(A).
    """
    n_samples = len(y_true)
    seeds = sorted(preds_A.keys())
    assert sorted(preds_B.keys()) == seeds, "Seeds must match across methods"

    # Compute instance loss matrix shape (S, N)
    loss_A = np.array([compute_instance_brier(y_true, preds_A[s]) for s in seeds])
    loss_B = np.array([compute_instance_brier(y_true, preds_B[s]) for s in seeds])

    # Mean loss across seeds for each instance: shape (N,)
    inst_loss_A = np.mean(loss_A, axis=0)
    inst_loss_B = np.mean(loss_B, axis=0)

    # Reduction: baseline loss minus model loss (positive = improvement)
    diff = inst_loss_B - inst_loss_A
    point_est = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_bootstraps, dtype=float)

    for b in range(n_bootstraps):
        idx = rng.integers(0, n_samples, size=n_samples)
        # Resample instances, preserving pairing across methods and seeds
        boot_diffs[b] = np.mean(diff[idx])

    se = float(np.std(boot_diffs, ddof=1))
    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))

    # Two-sided empirical bootstrap p-value for H0: diff <= 0
    p_one_tailed = float(np.mean(boot_diffs <= 0.0))
    p_raw = float(min(1.0, 2.0 * min(p_one_tailed, 1.0 - p_one_tailed)))
    if p_raw == 0.0:
        p_raw = 1.0 / n_bootstraps  # conservative finite-sample lower bound

    effect_pass = bool(point_est >= effect_size_threshold)
    sig_pass = bool(ci_lo > 0.0 and p_raw < 0.05)
    conf_pass = bool(effect_pass and sig_pass)

    return ContrastResult(
        contrast_name="contrast",
        point_estimate=point_est,
        bootstrap_mean=float(np.mean(boot_diffs)),
        standard_error=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        p_value_raw=p_raw,
        p_value_holm=p_raw,  # Updated during family correction
        effect_size_threshold=effect_size_threshold,
        effect_size_pass=effect_pass,
        statistical_significance_pass=sig_pass,
        confirmatory_pass=conf_pass,
    )


def paired_contrast_bootstrap(
    y_true_pos: np.ndarray,
    preds_A_pos: Dict[int, np.ndarray],
    preds_B_pos: Dict[int, np.ndarray],
    y_true_neg: np.ndarray,
    preds_A_neg: Dict[int, np.ndarray],
    preds_B_neg: Dict[int, np.ndarray],
    n_bootstraps: int = 2000,
    seed: int = 20261012,
    effect_size_threshold: float = 0.0,
) -> ContrastResult:
    """Run bootstrap for matched complementarity contrast: Gain_pos - Gain_neg.

    Gain_pos = Brier(B_pos) - Brier(A_pos)
    Gain_neg = Brier(B_neg) - Brier(A_neg)
    Contrast = Gain_pos - Gain_neg
    """
    n_pos = len(y_true_pos)
    n_neg = len(y_true_neg)
    seeds = sorted(preds_A_pos.keys())

    loss_A_pos = np.mean([compute_instance_brier(y_true_pos, preds_A_pos[s]) for s in seeds], axis=0)
    loss_B_pos = np.mean([compute_instance_brier(y_true_pos, preds_B_pos[s]) for s in seeds], axis=0)
    diff_pos = loss_B_pos - loss_A_pos

    loss_A_neg = np.mean([compute_instance_brier(y_true_neg, preds_A_neg[s]) for s in seeds], axis=0)
    loss_B_neg = np.mean([compute_instance_brier(y_true_neg, preds_B_neg[s]) for s in seeds], axis=0)
    diff_neg = loss_B_neg - loss_A_neg

    point_est = float(np.mean(diff_pos) - np.mean(diff_neg))

    rng = np.random.default_rng(seed)
    boot_contrasts = np.empty(n_bootstraps, dtype=float)

    for b in range(n_bootstraps):
        idx_p = rng.integers(0, n_pos, size=n_pos)
        idx_n = rng.integers(0, n_neg, size=n_neg)
        boot_contrasts[b] = np.mean(diff_pos[idx_p]) - np.mean(diff_neg[idx_n])

    se = float(np.std(boot_contrasts, ddof=1))
    ci_lo = float(np.percentile(boot_contrasts, 2.5))
    ci_hi = float(np.percentile(boot_contrasts, 97.5))

    p_one_tailed = float(np.mean(boot_contrasts <= 0.0))
    p_raw = float(min(1.0, 2.0 * min(p_one_tailed, 1.0 - p_one_tailed)))
    if p_raw == 0.0:
        p_raw = 1.0 / n_bootstraps

    effect_pass = bool(point_est > effect_size_threshold)
    sig_pass = bool(ci_lo > 0.0 and p_raw < 0.05)
    conf_pass = bool(effect_pass and sig_pass)

    return ContrastResult(
        contrast_name="matched_complementarity_contrast",
        point_estimate=point_est,
        bootstrap_mean=float(np.mean(boot_contrasts)),
        standard_error=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        p_value_raw=p_raw,
        p_value_holm=p_raw,
        effect_size_threshold=effect_size_threshold,
        effect_size_pass=effect_pass,
        statistical_significance_pass=sig_pass,
        confirmatory_pass=conf_pass,
    )


def apply_holm_bonferroni(results: List[ContrastResult]) -> List[ContrastResult]:
    """Apply step-down Holm-Bonferroni correction to a family of ContrastResults."""
    m = len(results)
    if m == 0:
        return []

    # Sort indices by raw p-value ascending
    sorted_indices = sorted(range(m), key=lambda i: results[i].p_value_raw)

    adjusted_p = [0.0] * m
    running_max = 0.0

    for rank, orig_idx in enumerate(sorted_indices):
        multiplier = m - rank
        raw_p = results[orig_idx].p_value_raw
        adj = min(1.0, raw_p * multiplier)
        running_max = max(running_max, adj)
        adjusted_p[orig_idx] = running_max

    out = []
    for orig_idx, res in enumerate(results):
        p_holm = adjusted_p[orig_idx]
        sig_pass = bool(res.ci_lower > 0.0 and p_holm < 0.05)
        conf_pass = bool(res.effect_size_pass and sig_pass)
        out.append(
            ContrastResult(
                contrast_name=res.contrast_name,
                point_estimate=res.point_estimate,
                bootstrap_mean=res.bootstrap_mean,
                standard_error=res.standard_error,
                ci_lower=res.ci_lower,
                ci_upper=res.ci_upper,
                p_value_raw=res.p_value_raw,
                p_value_holm=p_holm,
                effect_size_threshold=res.effect_size_threshold,
                effect_size_pass=res.effect_size_pass,
                statistical_significance_pass=sig_pass,
                confirmatory_pass=conf_pass,
            )
        )
    return out
