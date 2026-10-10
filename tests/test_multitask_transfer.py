"""Unit tests for Multi-Task LoRA Transferability Benchmark."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hypergraph_portfolio.multitask_transfer_dgp import (
    CalibratedLogOddsFusion3,
    PooledLoRA,
    SharedSubspaceMultiTask,
    StructuralCoLoRA,
    TargetOnlyLogistic,
    TargetOnlyLoRA,
    compute_principal_angles,
    compute_subspace_alignment,
    compute_weight_cosine_similarity,
    generate_multitask_transfer_dgp,
)


def test_dgp_generation_positive_and_negative_controls():
    """Verify data generation produces disjoint splits and expected subspace properties."""
    # Positive control
    pos_data = generate_multitask_transfer_dgp(seed=42, regime="positive_control")
    assert pos_data["U_true"].shape == (32, 2)
    np.testing.assert_allclose(pos_data["U_true"].T @ pos_data["U_true"], np.eye(2), atol=1e-6)
    assert pos_data["target_train"]["X"].shape == (64, 32)
    assert pos_data["target_cal"]["X"].shape == (128, 32)
    assert pos_data["target_dev"]["X"].shape == (256, 32)
    assert pos_data["target_test"]["X"].shape == (512, 32)

    # In positive control, tasks share subspace U_true
    U = pos_data["U_true"]
    for task_name, beta in pos_data["betas"].items():
        proj_norm = np.linalg.norm(U.T @ beta)
        total_norm = np.linalg.norm(beta)
        # Shared variance ratio should be > 98%
        assert (proj_norm / total_norm) ** 2 > 0.98

    # Negative control
    neg_data = generate_multitask_transfer_dgp(seed=42, regime="negative_control")
    assert neg_data["U_true"] is None
    for task_name, beta in neg_data["betas"].items():
        assert len(beta) == 32


def test_target_only_and_pooled_models():
    """Verify single-task and pooled LoRA models train and output bounded probabilities."""
    X = np.random.normal(size=(64, 32))
    y = np.random.randint(0, 2, size=64).astype(float)

    # Target-Only Logistic
    m0 = TargetOnlyLogistic(in_features=32)
    p0 = m0.predict_proba(X)
    assert p0.shape == (64,)
    assert np.all((p0 >= 0.0) & (p0 <= 1.0))
    w0 = m0.get_effective_weight()
    assert w0.shape == (32,)

    # Target-Only LoRA
    m_lora = TargetOnlyLoRA(in_features=32, rank=2)
    p_lora = m_lora.predict_proba(X)
    assert p_lora.shape == (64,)
    assert np.all((p_lora >= 0.0) & (p_lora <= 1.0))
    w_lora = m_lora.get_effective_weight()
    assert w_lora.shape == (32,)

    # Pooled LoRA
    m_pooled = PooledLoRA(in_features=32, rank=2)
    p_pooled = m_pooled.predict_proba(X)
    assert p_pooled.shape == (64,)
    assert np.all((p_pooled >= 0.0) & (p_pooled <= 1.0))


def test_calibrated_log_odds_fusion_three_sources():
    """Verify multivariate calibrated log-odds fusion with 3 source logits and learned intercept."""
    rng = np.random.default_rng(123)
    n = 128
    # 3 source probabilities
    p_sources = rng.uniform(0.1, 0.9, size=(n, 3))
    y_cal = rng.integers(0, 2, size=n).astype(float)

    fusion = CalibratedLogOddsFusion3.fit(p_sources, y_cal, alpha=1.0)
    assert len(fusion.weights) == 3
    assert isinstance(fusion.intercept, float)

    p_fused = fusion.predict_proba(p_sources)
    assert p_fused.shape == (n,)
    assert np.all((p_fused >= 0.0) & (p_fused <= 1.0))


def test_shared_subspace_and_structural_colora_models():
    """Verify SharedSubspaceMultiTask and StructuralCoLoRA multi-task forward passes and weights."""
    X = np.random.normal(size=(32, 32))

    # Shared Subspace Multi-Task
    subspace_model = SharedSubspaceMultiTask(in_features=32, rank=2, n_tasks=4)
    for task_idx in range(4):
        p_task = subspace_model.predict_proba_task(X, task_idx)
        assert p_task.shape == (32,)
        assert np.all((p_task >= 0.0) & (p_task <= 1.0))
        w_task = subspace_model.get_effective_weight(task_idx)
        assert w_task.shape == (32,)
    U_shared = subspace_model.get_shared_subspace()
    assert U_shared.shape == (32, 2)

    # Structural CoLoRA
    colora_model = StructuralCoLoRA(in_features=32, rank=2, n_tasks=4)
    for task_idx in range(4):
        p_task = colora_model.predict_proba_task(X, task_idx)
        assert p_task.shape == (32,)
        assert np.all((p_task >= 0.0) & (p_task <= 1.0))
        w_task = colora_model.get_effective_weight(task_idx)
        assert w_task.shape == (32,)
    B_shared = colora_model.get_shared_subspace()
    assert B_shared.shape == (32, 2)


def test_subspace_alignment_and_weight_diagnostics():
    """Verify Grassmannian alignment and cosine similarity metrics on known canonical cases."""
    rng = np.random.default_rng(777)
    d, r = 32, 2

    # Canonical orthonormal basis U
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    U1 = Q[:, :r]
    U2 = Q[:, :r]  # Exact match
    U_ortho = Q[:, r : 2 * r]  # Orthogonal complement

    # Exact match must yield S_proj = 1.0 and cos(theta) = [1.0, 1.0]
    s_exact = compute_subspace_alignment(U1, U2)
    assert pytest.approx(s_exact, abs=1e-5) == 1.0
    angles_exact = compute_principal_angles(U1, U2)
    np.testing.assert_allclose(angles_exact, np.ones(r), atol=1e-5)

    # Orthogonal basis must yield S_proj = 0.0
    s_ortho = compute_subspace_alignment(U1, U_ortho)
    assert pytest.approx(s_ortho, abs=1e-5) == 0.0

    # Weight cosine similarity
    w1 = np.array([1.0, 2.0, 3.0])
    w2 = np.array([2.0, 4.0, 6.0])  # Identical direction
    w_neg = -w1
    assert pytest.approx(compute_weight_cosine_similarity(w1, w2), abs=1e-5) == 1.0
    assert pytest.approx(compute_weight_cosine_similarity(w1, w_neg), abs=1e-5) == -1.0
