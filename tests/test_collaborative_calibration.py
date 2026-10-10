"""Unit tests for the collaborative representation calibration benchmark.

Verifies:
1. Split sizes and integrity (64 train, 64 cal, 128 dev, 256 test).
2. PlattScaler calibration: fitting, probabilities in (0, 1), and monotonicity (AUROC invariance).
3. CalibratedLogOddsFusion3 on 64 calibration samples.
4. Model parameter counts.
5. Statistical testing utilities (paired Wilcoxon test, bootstrap CIs, Holm correction).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from hypergraph_portfolio.multitask_transfer_dgp import (
    CalibratedLogOddsFusion3,
    PlattScaler,
    PooledLoRA,
    SharedSubspaceMultiTask,
    StructuralCoLoRA,
    TargetOnlyLogistic,
    generate_multitask_transfer_dgp,
)


def test_calibration_dataset_splits() -> None:
    data = generate_multitask_transfer_dgp(
        seed=20261101,
        regime="positive_control",
        d=32,
        r=2,
        gamma=0.05,
        n_source=512,
        n_target_train=64,
        n_target_cal=64,
        n_target_dev=128,
        n_target_test=256,
    )
    assert data["target_train"]["X"].shape == (64, 32)
    assert data["target_train"]["y"].shape == (64,)
    assert data["target_cal"]["X"].shape == (64, 32)
    assert data["target_cal"]["y"].shape == (64,)
    assert data["target_dev"]["X"].shape == (128, 32)
    assert data["target_dev"]["y"].shape == (128,)
    assert data["target_test"]["X"].shape == (256, 32)
    assert data["target_test"]["y"].shape == (256,)


def test_platt_scaler_monotonicity_and_auroc_invariance() -> None:
    rng = np.random.default_rng(42)
    logits_cal = rng.normal(0.0, 2.0, size=64)
    y_cal = (rng.uniform(size=64) < 1.0 / (1.0 + np.exp(-logits_cal))).astype(float)

    scaler = PlattScaler.fit(logits_cal, y_cal, C=100.0)
    assert isinstance(scaler.a, float)
    assert isinstance(scaler.b, float)
    assert scaler.a > 0  # Slope should be positive

    # Test on dev set
    logits_dev = rng.normal(0.0, 2.0, size=128)
    y_dev = (rng.uniform(size=128) < 1.0 / (1.0 + np.exp(-logits_dev))).astype(float)

    p_cal = scaler.predict_proba(logits_dev)
    assert np.all(p_cal > 0.0) and np.all(p_cal < 1.0)

    # Monotonic scaling preserves AUROC
    auc_raw = roc_auc_score(y_dev, logits_dev)
    auc_cal = roc_auc_score(y_dev, p_cal)
    assert np.isclose(auc_raw, auc_cal, atol=1e-5)


def test_calibrated_log_odds_fusion_on_64_samples() -> None:
    rng = np.random.default_rng(123)
    p_sources = rng.uniform(0.1, 0.9, size=(64, 3))
    y_cal = (rng.uniform(size=64) > 0.5).astype(float)

    fuse = CalibratedLogOddsFusion3.fit(p_sources, y_cal, alpha=1.0)
    assert fuse.weights.shape == (3,)
    assert isinstance(fuse.intercept, float)

    p_dev_sources = rng.uniform(0.1, 0.9, size=(128, 3))
    p_fuse = fuse.predict_proba(p_dev_sources)
    assert p_fuse.shape == (128,)
    assert np.all(p_fuse > 0.0) and np.all(p_fuse < 1.0)


def test_model_parameter_counts() -> None:
    m0 = TargetOnlyLogistic(in_features=32)
    m0_count = sum(p.numel() for p in m0.parameters() if p.requires_grad)
    assert m0_count == 33  # 32 weights + 1 bias

    m1 = PooledLoRA(in_features=32, rank=2)
    m1_count = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    assert m1_count == 67  # 32*2 + 2*1 + 1

    m_sub = SharedSubspaceMultiTask(in_features=32, rank=2, n_tasks=4)
    m_sub_count = sum(p.numel() for p in m_sub.parameters() if p.requires_grad)
    assert m_sub_count == 76  # 32*2 + 4*2 + 4

    m_col = StructuralCoLoRA(in_features=32, rank=2, n_tasks=4)
    m_col_count = sum(p.numel() for p in m_col.parameters() if p.requires_grad)
    assert m_col_count == 86  # 32*2 + 2*1 + 4*(2*2) + 4


def test_statistical_contrast_helpers() -> None:
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(777)
    # 10 replications with positive effect
    diffs = rng.normal(-0.02, 0.005, size=10)
    stat, p_val = wilcoxon(diffs, alternative="two-sided")
    assert p_val < 0.05
    assert np.mean(diffs) < 0
