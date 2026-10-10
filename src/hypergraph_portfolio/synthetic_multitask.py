"""Synthetic Multi-Task Data-Generating Process with Controlled Complementarity.

Implements the positive-control and negative-control data generation regimes
as specified in docs/bayesian_collaboration_research_plan.md and amended in
docs/bayesian_collaboration_design_amendment.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import root_scalar


@dataclass(frozen=True)
class DGPConfig:
    """Configuration for synthetic multi-task data generation."""

    seed: int = 20261009
    d_shared: int = 4
    d_A: int = 4
    d_B: int = 4
    d_noise: int = 8
    beta_scale: float = 1.5
    prevalence: float = 0.35
    n_train_target: int = 128
    n_train_specialist: int = 256
    n_development: int = 64
    n_validation: int = 64
    n_test: int = 512

    @property
    def d_total(self) -> int:
        return self.d_shared + self.d_A + self.d_B + self.d_noise

    @property
    def d_model_input(self) -> int:
        # Features + observation mask
        return 2 * self.d_total


@dataclass
class DGPSpec:
    """Frozen parameters of the data-generating process."""

    config: DGPConfig
    beta_s: np.ndarray
    beta_A: np.ndarray
    beta_B: np.ndarray
    intercept: float
    # Observation masks for dim=d_total
    mask_full: np.ndarray
    mask_A: np.ndarray
    mask_B: np.ndarray
    analytical_bayes_brier_full: float = 0.0
    analytical_bayes_brier_A: float = 0.0
    analytical_bayes_brier_B: float = 0.0
    analytical_oracle_gap: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": dataclasses.asdict(self.config),
            "beta_s": self.beta_s.tolist(),
            "beta_A": self.beta_A.tolist(),
            "beta_B": self.beta_B.tolist(),
            "intercept": float(self.intercept),
            "mask_full": self.mask_full.tolist(),
            "mask_A": self.mask_A.tolist(),
            "mask_B": self.mask_B.tolist(),
            "analytical_bayes_brier_full": float(self.analytical_bayes_brier_full),
            "analytical_bayes_brier_A": float(self.analytical_bayes_brier_A),
            "analytical_bayes_brier_B": float(self.analytical_bayes_brier_B),
            "analytical_oracle_gap": float(self.analytical_oracle_gap),
        }


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def generate_dgp(config: Optional[DGPConfig] = None) -> DGPSpec:
    """Create and freeze the DGP specification deterministically from config.seed."""
    if config is None:
        config = DGPConfig()

    rng = np.random.default_rng(config.seed)

    # Draw coefficients for predictive feature blocks
    beta_s = rng.normal(0.0, 1.0, size=config.d_shared)
    beta_A = rng.normal(0.0, 1.0, size=config.d_A)
    beta_B = rng.normal(0.0, 1.0, size=config.d_B)

    # Normalize block norms to beta_scale to ensure predictable signal magnitude
    beta_s = (beta_s / np.linalg.norm(beta_s)) * config.beta_scale
    beta_A = (beta_A / np.linalg.norm(beta_A)) * config.beta_scale
    beta_B = (beta_B / np.linalg.norm(beta_B)) * config.beta_scale

    # Compute variance of logit: Var(beta^T x) = ||beta_s||^2 + ||beta_A||^2 + ||beta_B||^2
    var_logit = config.beta_scale**2 * 3.0
    std_logit = np.sqrt(var_logit)

    # Calibrate intercept so that E[sigmoid(beta^T x + b)] == prevalence
    # Using Monte Carlo simulation with a fixed generator
    mc_rng = np.random.default_rng(config.seed + 1)
    mc_z = mc_rng.normal(0.0, std_logit, size=100000)

    def prevalence_diff(b: float) -> float:
        return float(np.mean(_sigmoid(mc_z + b)) - config.prevalence)

    res = root_scalar(prevalence_diff, bracket=[-10.0, 10.0], method="brentq")
    intercept = float(res.root)

    # Build observation masks (length = d_total = 20)
    d = config.d_total
    mask_full = np.ones(d, dtype=np.float32)

    # Specialist A observes shared (0..d_s-1), A (d_s..d_s+d_A-1), and noise
    # Missing: block B (d_s+d_A .. d_s+d_A+d_B-1)
    mask_A = np.ones(d, dtype=np.float32)
    b_start = config.d_shared + config.d_A
    b_end = b_start + config.d_B
    mask_A[b_start:b_end] = 0.0

    # Specialist B observes shared, B, and noise; missing block A
    mask_B = np.ones(d, dtype=np.float32)
    a_start = config.d_shared
    a_end = a_start + config.d_A
    mask_B[a_start:a_end] = 0.0

    spec = DGPSpec(
        config=config,
        beta_s=beta_s,
        beta_A=beta_A,
        beta_B=beta_B,
        intercept=intercept,
        mask_full=mask_full,
        mask_A=mask_A,
        mask_B=mask_B,
    )

    # Compute analytical Bayes-optimal Brier scores on target distribution
    spec = _compute_analytical_properties(spec, mc_rng)
    return spec


def _compute_analytical_properties(spec: DGPSpec, rng: np.random.Generator) -> DGPSpec:
    """Compute analytical Bayes-optimal Brier scores by Monte Carlo."""
    cfg = spec.config
    n_mc = 200000
    xs = rng.normal(0.0, 1.0, size=(n_mc, cfg.d_shared))
    xa = rng.normal(0.0, 1.0, size=(n_mc, cfg.d_A))
    xb = rng.normal(0.0, 1.0, size=(n_mc, cfg.d_B))

    zs = xs @ spec.beta_s
    za = xa @ spec.beta_A
    zb = xb @ spec.beta_B
    z_full = zs + za + zb + spec.intercept
    p_full = _sigmoid(z_full)
    y = (rng.uniform(0.0, 1.0, size=n_mc) < p_full).astype(np.float64)

    # Bayes-optimal full model Brier
    bayes_brier_full = float(np.mean((p_full - y) ** 2))

    # Bayes-optimal specialist A integrates out xb: xb ~ N(0, I) so zb ~ N(0, ||beta_B||^2)
    # Using MacFadden / probit approx: E[sigmoid(z + eps)] where eps ~ N(0, sigma^2)
    # or Monte Carlo integration over zb
    var_b = float(np.sum(spec.beta_B**2))
    # Approximation for E[sigmoid(u + zb)]: sigmoid(u / sqrt(1 + lambda^2 * var_b)) where lambda^2 = pi/8
    # But Monte Carlo over xb directly:
    zb_mc = rng.normal(0.0, np.sqrt(var_b), size=(n_mc, 100))
    p_A = np.mean(_sigmoid((zs + za + spec.intercept)[:, None] + zb_mc), axis=1)
    bayes_brier_A = float(np.mean((p_A - y) ** 2))

    var_a = float(np.sum(spec.beta_A**2))
    za_mc = rng.normal(0.0, np.sqrt(var_a), size=(n_mc, 100))
    p_B = np.mean(_sigmoid((zs + zb + spec.intercept)[:, None] + za_mc), axis=1)
    bayes_brier_B = float(np.mean((p_B - y) ** 2))

    oracle_gap = float(min(bayes_brier_A, bayes_brier_B) - bayes_brier_full)

    return DGPSpec(
        config=spec.config,
        beta_s=spec.beta_s,
        beta_A=spec.beta_A,
        beta_B=spec.beta_B,
        intercept=spec.intercept,
        mask_full=spec.mask_full,
        mask_A=spec.mask_A,
        mask_B=spec.mask_B,
        analytical_bayes_brier_full=bayes_brier_full,
        analytical_bayes_brier_A=bayes_brier_A,
        analytical_bayes_brier_B=bayes_brier_B,
        analytical_oracle_gap=oracle_gap,
    )


def sample_features_and_labels(
    spec: DGPSpec,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate raw unmasked features x, true probabilities p, and binary labels y."""
    cfg = spec.config
    xs = rng.normal(0.0, 1.0, size=(n_samples, cfg.d_shared))
    xa = rng.normal(0.0, 1.0, size=(n_samples, cfg.d_A))
    xb = rng.normal(0.0, 1.0, size=(n_samples, cfg.d_B))
    xn = rng.normal(0.0, 1.0, size=(n_samples, cfg.d_noise))

    x_raw = np.hstack([xs, xa, xb, xn]).astype(np.float32)
    z = (xs @ spec.beta_s) + (xa @ spec.beta_A) + (xb @ spec.beta_B) + spec.intercept
    probs = _sigmoid(z).astype(np.float32)
    y = (rng.uniform(0.0, 1.0, size=n_samples) < probs).astype(np.float32)
    return x_raw, probs, y


def apply_observation_mask(
    x_raw: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Zero-fill missing features and concatenate binary observation mask.

    Returns:
        concatenated array of shape (n_samples, 2 * d_total).
    """
    x_masked = x_raw * mask
    n = len(x_raw)
    mask_tiled = np.tile(mask, (n, 1)).astype(np.float32)
    return np.hstack([x_masked, mask_tiled]).astype(np.float32)


@dataclass
class DatasetSplits:
    """Complete dataset container with all splits and regimes."""

    regime: str  # 'positive_control' or 'negative_control'
    target_train: Dict[str, np.ndarray]
    specialist_A_train: Dict[str, np.ndarray]
    specialist_B_train: Dict[str, np.ndarray]
    development: Dict[str, np.ndarray]
    validation: Dict[str, np.ndarray]
    test: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)


def sample_dataset(
    spec: DGPSpec,
    regime: str = "positive_control",
    data_seed: Optional[int] = None,
) -> DatasetSplits:
    """Generate all partitions for the specified regime (positive or negative control).

    Args:
        spec: Frozen DGPSpec.
        regime: 'positive_control' or 'negative_control'.
        data_seed: Random seed for sampling instances (defaults to spec.config.seed + 10).
    """
    if regime not in ("positive_control", "negative_control"):
        raise ValueError(f"Unknown regime: {regime}")

    cfg = spec.config
    if data_seed is None:
        data_seed = cfg.seed + 10

    rng = np.random.default_rng(data_seed)

    # 1. Target training data (always observes all features)
    x_raw_tgt, p_tgt, y_tgt = sample_features_and_labels(spec, cfg.n_train_target, rng)
    x_in_tgt = apply_observation_mask(x_raw_tgt, spec.mask_full)
    target_train = {
        "x": x_in_tgt,
        "x_raw": x_raw_tgt,
        "y": y_tgt,
        "p_true": p_tgt,
        "mask": spec.mask_full,
        "task_id": 0,
    }

    # 2. Specialist A and Specialist B training data
    x_raw_A, p_A, y_A = sample_features_and_labels(spec, cfg.n_train_specialist, rng)
    x_raw_B, p_B, y_B = sample_features_and_labels(spec, cfg.n_train_specialist, rng)

    if regime == "positive_control":
        mask_A = spec.mask_A
        mask_B = spec.mask_B
    else:  # negative_control: specialists receive redundant feature masks (both observe mask_A, zero complementarity)
        mask_A = spec.mask_A
        mask_B = spec.mask_A

    x_in_A = apply_observation_mask(x_raw_A, mask_A)
    x_in_B = apply_observation_mask(x_raw_B, mask_B)

    specialist_A_train = {
        "x": x_in_A,
        "x_raw": x_raw_A,
        "y": y_A,
        "p_true": p_A,
        "mask": mask_A,
        "task_id": 1,
    }
    specialist_B_train = {
        "x": x_in_B,
        "x_raw": x_raw_B,
        "y": y_B,
        "p_true": p_B,
        "mask": mask_B,
        "task_id": 2,
    }

    # 3. Development split (target distribution; weights fitted here)
    x_raw_dev, p_dev, y_dev = sample_features_and_labels(spec, cfg.n_development, rng)
    development = {
        "x_full": apply_observation_mask(x_raw_dev, spec.mask_full),
        "x_A": apply_observation_mask(x_raw_dev, mask_A),
        "x_B": apply_observation_mask(x_raw_dev, mask_B),
        "x_raw": x_raw_dev,
        "y": y_dev,
        "p_true": p_dev,
    }

    # 4. Validation split (target distribution; diagnostics here)
    x_raw_val, p_val, y_val = sample_features_and_labels(spec, cfg.n_validation, rng)
    validation = {
        "x_full": apply_observation_mask(x_raw_val, spec.mask_full),
        "x_A": apply_observation_mask(x_raw_val, mask_A),
        "x_B": apply_observation_mask(x_raw_val, mask_B),
        "x_raw": x_raw_val,
        "y": y_val,
        "p_true": p_val,
    }

    # 5. Held-out test split (target distribution; locked until evaluation)
    x_raw_test, p_test, y_test = sample_features_and_labels(spec, cfg.n_test, rng)
    test = {
        "x_full": apply_observation_mask(x_raw_test, spec.mask_full),
        "x_A": apply_observation_mask(x_raw_test, mask_A),
        "x_B": apply_observation_mask(x_raw_test, mask_B),
        "x_raw": x_raw_test,
        "y": y_test,
        "p_true": p_test,
    }

    return DatasetSplits(
        regime=regime,
        target_train=target_train,
        specialist_A_train=specialist_A_train,
        specialist_B_train=specialist_B_train,
        development=development,
        validation=validation,
        test=test,
        metadata={
            "dgp_seed": cfg.seed,
            "data_seed": data_seed,
            "prevalence_target_train": float(y_tgt.mean()),
            "prevalence_dev": float(y_dev.mean()),
            "prevalence_val": float(y_val.mean()),
            "prevalence_test": float(y_test.mean()),
            "null_brier_target": float(y_tgt.mean() * (1.0 - y_tgt.mean())),
        },
    )
