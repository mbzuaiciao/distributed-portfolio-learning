"""Synthetic Multi-Task Data-Generating Process for Experiment 2.

Implements the 5-split partition architecture:
1. Model training splits (target_train, specialist_A_train, specialist_B_train)
2. Calibration split (dedicated N=128 for Platt calibration)
3. Fusion development split (dedicated N=128 for fitting simplex & log-odds fusion)
4. Development validation split (dedicated N=128 for evaluating dev hypotheses)
5. Held-out test split (dedicated N=512, locked out and untouched)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from hypergraph_portfolio.synthetic_multitask import (
    DGPConfig,
    DGPSpec,
    apply_observation_mask,
    generate_dgp,
    sample_features_and_labels,
)


@dataclass(frozen=True)
class Exp2SplitConfig:
    n_train_target: int = 128
    n_train_specialist: int = 256
    n_calibration: int = 128
    n_fusion_dev: int = 128
    n_development: int = 128
    n_test: int = 512
    data_seed: int = 20261011


@dataclass
class Exp2DatasetSplits:
    regime: str  # 'positive_control' or 'negative_control'
    target_train: Dict[str, np.ndarray]
    specialist_A_train: Dict[str, np.ndarray]
    specialist_B_train: Dict[str, np.ndarray]
    calibration: Dict[str, np.ndarray]
    fusion_dev: Dict[str, np.ndarray]
    development: Dict[str, np.ndarray]
    test: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)


def sample_exp2_dataset(
    spec: DGPSpec,
    regime: str = "positive_control",
    split_cfg: Optional[Exp2SplitConfig] = None,
) -> Exp2DatasetSplits:
    """Generate all 5 disjoint partitions for Experiment 2."""
    if regime not in ("positive_control", "negative_control"):
        raise ValueError(f"Unknown regime: {regime}")

    if split_cfg is None:
        split_cfg = Exp2SplitConfig()

    rng = np.random.default_rng(split_cfg.data_seed)

    # 1. Target training data (all features observed)
    x_raw_tgt, p_tgt, y_tgt = sample_features_and_labels(
        spec, split_cfg.n_train_target, rng
    )
    target_train = {
        "x": apply_observation_mask(x_raw_tgt, spec.mask_full),
        "x_raw": x_raw_tgt,
        "y": y_tgt,
        "p_true": p_tgt,
        "mask": spec.mask_full,
        "task_id": 0,
    }

    # 2. Specialist A and Specialist B training data
    x_raw_A, p_A, y_A = sample_features_and_labels(
        spec, split_cfg.n_train_specialist, rng
    )
    x_raw_B, p_B, y_B = sample_features_and_labels(
        spec, split_cfg.n_train_specialist, rng
    )

    if regime == "positive_control":
        mask_A = spec.mask_A
        mask_B = spec.mask_B
    else:  # negative control: redundant specialists (both observe mask_A)
        mask_A = spec.mask_A
        mask_B = spec.mask_A

    specialist_A_train = {
        "x": apply_observation_mask(x_raw_A, mask_A),
        "x_raw": x_raw_A,
        "y": y_A,
        "p_true": p_A,
        "mask": mask_A,
        "task_id": 1,
    }
    specialist_B_train = {
        "x": apply_observation_mask(x_raw_B, mask_B),
        "x_raw": x_raw_B,
        "y": y_B,
        "p_true": p_B,
        "mask": mask_B,
        "task_id": 2,
    }

    # Helper for evaluating specialists on target distribution
    def build_eval_split(n: int) -> Dict[str, np.ndarray]:
        x_raw, p, y = sample_features_and_labels(spec, n, rng)
        return {
            "x_full": apply_observation_mask(x_raw, spec.mask_full),
            "x_A": apply_observation_mask(x_raw, mask_A),
            "x_B": apply_observation_mask(x_raw, mask_B),
            "x_raw": x_raw,
            "y": y,
            "p_true": p,
        }

    # 3. Calibration split (dedicated N=128)
    calibration = build_eval_split(split_cfg.n_calibration)

    # 4. Fusion dev split (dedicated N=128)
    fusion_dev = build_eval_split(split_cfg.n_fusion_dev)

    # 5. Development validation split (dedicated N=128)
    development = build_eval_split(split_cfg.n_development)

    # 6. Held-out test split (dedicated N=512, locked out and untouched)
    test = build_eval_split(split_cfg.n_test)

    return Exp2DatasetSplits(
        regime=regime,
        target_train=target_train,
        specialist_A_train=specialist_A_train,
        specialist_B_train=specialist_B_train,
        calibration=calibration,
        fusion_dev=fusion_dev,
        development=development,
        test=test,
        metadata={
            "dgp_seed": spec.config.seed,
            "data_seed": split_cfg.data_seed,
            "split_config": dataclasses.asdict(split_cfg),
            "null_brier_target": float(y_tgt.mean() * (1.0 - y_tgt.mean())),
        },
    )
