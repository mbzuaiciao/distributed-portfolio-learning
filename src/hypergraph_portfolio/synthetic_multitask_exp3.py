"""Synthetic Multi-Task Data-Generating Process for Experiment 3.

Implements localized support structures:
1. Regime A: Axis-Aligned Regional Covariate Shift (split on x_{s, 1}).
2. Regime B: Smooth Multi-Modal Covariate Support (2D Gaussian mixtures on x_{s, 1}, x_{s, 2}).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import root_scalar

from hypergraph_portfolio.synthetic_multitask import (
    DGPConfig,
    DGPSpec,
    _sigmoid,
    apply_observation_mask,
    generate_dgp,
)


@dataclass(frozen=True)
class Exp3SplitConfig:
    n_train_target: int = 128
    n_train_specialist_dense: int = 256
    n_train_specialist_sparse: int = 16
    n_calibration: int = 128
    n_fusion_dev: int = 128
    n_development: int = 128
    n_test: int = 512
    data_seed: int = 20261015


@dataclass
class Exp3DatasetSplits:
    regime: str  # 'regime_A' or 'regime_B'
    target_train: Dict[str, np.ndarray]
    specialist_A_train: Dict[str, np.ndarray]
    specialist_B_train: Dict[str, np.ndarray]
    calibration: Dict[str, np.ndarray]
    fusion_dev: Dict[str, np.ndarray]
    development: Dict[str, np.ndarray]
    test: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)


def sample_regime_A_specialist(
    spec: DGPSpec,
    specialist: str,  # 'A' or 'B'
    n_dense: int,
    n_sparse: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample specialist training data with axis-aligned regional covariate shift on x_{s,1}."""
    cfg = spec.config
    # Specialist A is dense in x_{s,1} > 0, sparse in x_{s,1} <= 0
    # Specialist B is dense in x_{s,1} <= 0, sparse in x_{s,1} > 0
    target_dense_sign = 1.0 if specialist == "A" else -1.0

    collected_x = []
    collected_p = []
    collected_y = []

    # Sample dense region
    n_dense_collected = 0
    while n_dense_collected < n_dense:
        batch_size = max(64, (n_dense - n_dense_collected) * 2)
        xs = rng.normal(0.0, 1.0, size=(batch_size, cfg.d_shared))
        mask = (xs[:, 0] * target_dense_sign) > 0.0
        n_match = np.sum(mask)
        if n_match > 0:
            take = min(n_match, n_dense - n_dense_collected)
            xs_take = xs[mask][:take]
            xa = rng.normal(0.0, 1.0, size=(take, cfg.d_A))
            xb = rng.normal(0.0, 1.0, size=(take, cfg.d_B))
            xn = rng.normal(0.0, 1.0, size=(take, cfg.d_noise))
            x_raw = np.hstack([xs_take, xa, xb, xn]).astype(np.float32)
            z = (xs_take @ spec.beta_s) + (xa @ spec.beta_A) + (xb @ spec.beta_B) + spec.intercept
            probs = _sigmoid(z).astype(np.float32)
            y = (rng.uniform(0.0, 1.0, size=take) < probs).astype(np.float32)
            collected_x.append(x_raw)
            collected_p.append(probs)
            collected_y.append(y)
            n_dense_collected += take

    # Sample sparse region
    n_sparse_collected = 0
    while n_sparse_collected < n_sparse:
        batch_size = max(64, (n_sparse - n_sparse_collected) * 2)
        xs = rng.normal(0.0, 1.0, size=(batch_size, cfg.d_shared))
        mask = (xs[:, 0] * target_dense_sign) <= 0.0
        n_match = np.sum(mask)
        if n_match > 0:
            take = min(n_match, n_sparse - n_sparse_collected)
            xs_take = xs[mask][:take]
            xa = rng.normal(0.0, 1.0, size=(take, cfg.d_A))
            xb = rng.normal(0.0, 1.0, size=(take, cfg.d_B))
            xn = rng.normal(0.0, 1.0, size=(take, cfg.d_noise))
            x_raw = np.hstack([xs_take, xa, xb, xn]).astype(np.float32)
            z = (xs_take @ spec.beta_s) + (xa @ spec.beta_A) + (xb @ spec.beta_B) + spec.intercept
            probs = _sigmoid(z).astype(np.float32)
            y = (rng.uniform(0.0, 1.0, size=take) < probs).astype(np.float32)
            collected_x.append(x_raw)
            collected_p.append(probs)
            collected_y.append(y)
            n_sparse_collected += take

    x_all = np.vstack(collected_x)
    p_all = np.concatenate(collected_p)
    y_all = np.concatenate(collected_y)

    # Shuffle to mix dense and sparse examples
    perm = rng.permutation(len(y_all))
    return x_all[perm], p_all[perm], y_all[perm]


def sample_regime_B_specialist(
    spec: DGPSpec,
    specialist: str,  # 'A' or 'B'
    n_samples: int,
    rng: np.random.Generator,
    cluster_std: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample specialist training data from 2D Gaussian mixture components in (x_{s,1}, x_{s,2})."""
    cfg = spec.config
    # Centers in 2D space
    if specialist == "A":
        centers = [np.array([1.2, 1.2]), np.array([-1.2, -1.2])]
    else:  # Specialist B
        centers = [np.array([1.2, -1.2]), np.array([-1.2, 1.2])]

    # Equal allocation across clusters
    n_per_cluster = n_samples // 2
    rem = n_samples - n_per_cluster * 2

    xs_list = []
    for idx, c in enumerate(centers):
        count = n_per_cluster + (rem if idx == 0 else 0)
        c2d = rng.normal(loc=c, scale=cluster_std, size=(count, 2))
        c_rest = rng.normal(0.0, 1.0, size=(count, cfg.d_shared - 2))
        xs_list.append(np.hstack([c2d, c_rest]))

    xs = np.vstack(xs_list)
    xa = rng.normal(0.0, 1.0, size=(n_samples, cfg.d_A))
    xb = rng.normal(0.0, 1.0, size=(n_samples, cfg.d_B))
    xn = rng.normal(0.0, 1.0, size=(n_samples, cfg.d_noise))

    x_raw = np.hstack([xs, xa, xb, xn]).astype(np.float32)
    z = (xs @ spec.beta_s) + (xa @ spec.beta_A) + (xb @ spec.beta_B) + spec.intercept
    probs = _sigmoid(z).astype(np.float32)
    y = (rng.uniform(0.0, 1.0, size=n_samples) < probs).astype(np.float32)

    perm = rng.permutation(n_samples)
    return x_raw[perm], probs[perm], y[perm]


def sample_uniform_target_data(
    spec: DGPSpec,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample target data uniformly from the global standard normal distribution."""
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


def sample_exp3_dataset(
    spec: DGPSpec,
    regime: str = "regime_A",
    split_cfg: Optional[Exp3SplitConfig] = None,
) -> Exp3DatasetSplits:
    """Generate all disjoint partitions for Experiment 3."""
    if regime not in ("regime_A", "regime_B"):
        raise ValueError(f"Unknown regime: {regime}")

    if split_cfg is None:
        split_cfg = Exp3SplitConfig()

    rng = np.random.default_rng(split_cfg.data_seed)

    # 1. Target training data (uniform target distribution, all features observed)
    x_raw_tgt, p_tgt, y_tgt = sample_uniform_target_data(
        spec, split_cfg.n_train_target, rng
    )
    target_train = {
        "x": apply_observation_mask(x_raw_tgt, spec.mask_full),
        "x_raw": x_raw_tgt,
        "y": y_tgt,
        "p_true": p_tgt,
        "mask": spec.mask_full,
    }

    # 2. Specialist A and B training data (localized covariate support)
    if regime == "regime_A":
        x_raw_A, p_A, y_A = sample_regime_A_specialist(
            spec,
            "A",
            split_cfg.n_train_specialist_dense,
            split_cfg.n_train_specialist_sparse,
            rng,
        )
        x_raw_B, p_B, y_B = sample_regime_A_specialist(
            spec,
            "B",
            split_cfg.n_train_specialist_dense,
            split_cfg.n_train_specialist_sparse,
            rng,
        )
    else:  # regime_B: smooth multi-modal support
        n_samples_B = split_cfg.n_train_specialist_dense + split_cfg.n_train_specialist_sparse
        x_raw_A, p_A, y_A = sample_regime_B_specialist(spec, "A", n_samples_B, rng)
        x_raw_B, p_B, y_B = sample_regime_B_specialist(spec, "B", n_samples_B, rng)

    specialist_A_train = {
        "x": apply_observation_mask(x_raw_A, spec.mask_A),
        "x_raw": x_raw_A,
        "y": y_A,
        "p_true": p_A,
        "mask": spec.mask_A,
    }
    specialist_B_train = {
        "x": apply_observation_mask(x_raw_B, spec.mask_B),
        "x_raw": x_raw_B,
        "y": y_B,
        "p_true": p_B,
        "mask": spec.mask_B,
    }

    # Helper for evaluating on global target distribution
    def build_eval_split(n: int) -> Dict[str, np.ndarray]:
        x_raw, p, y = sample_uniform_target_data(spec, n, rng)
        return {
            "x_full": apply_observation_mask(x_raw, spec.mask_full),
            "x_A": apply_observation_mask(x_raw, spec.mask_A),
            "x_B": apply_observation_mask(x_raw, spec.mask_B),
            "x_raw": x_raw,
            "y": y,
            "p_true": p,
        }

    # 3. Calibration split (dedicated N=128)
    calibration = build_eval_split(split_cfg.n_calibration)

    # 4. Fusion dev split (dedicated N=128 for fitting all routers)
    fusion_dev = build_eval_split(split_cfg.n_fusion_dev)

    # 5. Development validation split (dedicated N=128 for dev gates and baseline selection)
    development = build_eval_split(split_cfg.n_development)

    # 6. Held-out test split (dedicated N=512, strictly quarantined)
    test = build_eval_split(split_cfg.n_test)

    return Exp3DatasetSplits(
        regime=regime,
        target_train=target_train,
        specialist_A_train=specialist_A_train,
        specialist_B_train=specialist_B_train,
        calibration=calibration,
        fusion_dev=fusion_dev,
        development=development,
        test=test,
        metadata={
            "d_shared": spec.config.d_shared,
            "d_A": spec.config.d_A,
            "d_B": spec.config.d_B,
            "d_noise": spec.config.d_noise,
        },
    )
