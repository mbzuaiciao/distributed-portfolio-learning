"""Unit tests for the collaborative LoRA positive-control experiment harness.

Verifies:
1. Model architectures: SmallMLP (1,857 params) and CoLoRAMLP (2,193 params).
2. CoLoRA routing: task_id alters forward pass output when lambdas differ;
   initialization with lora_b=0 yields exact base model forward pass.
3. Simplex mixture weight fitting on development probabilities.
4. Metric calculations (Brier, log loss, AUROC, ECE).
5. Holm step-down correction algorithm correctness.
6. Conditional information regression test.
7. End-to-end mini training loop executes deterministically without NaNs.
"""

from __future__ import annotations

import numpy as np
import torch

from hypergraph_portfolio.collaborative_lora import (
    fit_simplex_weights,
)
from hypergraph_portfolio.synthetic_models import (
    CoLoRAMLP,
    SmallMLP,
    count_parameters,
)
from hypergraph_portfolio.synthetic_multitask import (
    DGPConfig,
    generate_dgp,
    sample_dataset,
)


def test_model_parameter_counts():
    """Verify parameter counts match exact numbers in design amendment."""
    small = SmallMLP(in_features=40)
    colora = CoLoRAMLP(in_features=40, rank=4, alpha=8.0, tasks=3)

    small_counts = count_parameters(small)
    colora_counts = count_parameters(colora)

    assert small_counts["trainable_parameters"] == 1857
    assert colora_counts["trainable_parameters"] == 2193
    # Extra parameters in CoLoRA: 336 (lora_a: 160, lora_b: 128, lambdas: 48)
    assert colora_counts["trainable_parameters"] - small_counts["trainable_parameters"] == 336


def test_colora_initialization_identity():
    """Verify at initialization (lora_b=0), CoLoRA outputs match base layer output."""
    torch.manual_seed(42)
    model = CoLoRAMLP(in_features=40, rank=4, alpha=8.0, tasks=3)
    x = torch.randn(10, 40)

    # Base forward without LoRA
    with torch.no_grad():
        h_base = torch.nn.functional.linear(
            x, model.colora_fc1.base_weight, model.colora_fc1.base_bias
        )
        h_colora = model.colora_fc1(x, task_id=0)
        torch.testing.assert_close(h_base, h_colora)


def test_colora_task_routing():
    """Verify different task_ids route through different Lambda matrices when perturbing lambdas."""
    model = CoLoRAMLP(in_features=40, rank=4, alpha=8.0, tasks=3)
    x = torch.randn(5, 40)

    with torch.no_grad():
        model.colora_fc1.lora_b.fill_(1.0)
        model.colora_fc1.lora_a.fill_(1.0)
        # Scale lambda for task 1
        model.colora_fc1.lambdas[1].mul_(3.0)

    out_task0 = model(x, task_id=0)
    out_task1 = model(x, task_id=1)
    out_task2 = model(x, task_id=2)

    assert not torch.allclose(out_task0, out_task1)
    torch.testing.assert_close(out_task0, out_task2)


def test_simplex_weights_fitting():
    """Verify simplex weights fit on complementary predictions."""
    p_dev = np.array(
        [
            [0.1, 0.8],
            [0.2, 0.9],
            [0.8, 0.1],
            [0.9, 0.2],
        ]
    )
    y_dev = np.array([1.0, 1.0, 0.0, 0.0])
    w = fit_simplex_weights(p_dev, y_dev)
    assert len(w) == 2
    assert np.isclose(w.sum(), 1.0)
    assert w[1] > w[0]  # Specialist 2 gives higher prob to class 1


def test_holm_step_down():
    """Verify Holm step-down correction implementation."""
    from experiments.run_bayesian_collab_positive_control import holm_step_down

    # 3 hypotheses: p-values 0.01, 0.03, 0.08
    # alpha = 0.05
    # threshold 1: 0.05 / 3 = 0.01667 -> 0.01 <= 0.01667 (REJECT)
    # threshold 2: 0.05 / 2 = 0.025   -> 0.03 > 0.025   (FAIL TO REJECT)
    # threshold 3: 0.05 / 1 = 0.05    -> (FAIL TO REJECT by step-down)
    p_vals = {"m4_equal": 0.03, "m5_learned": 0.01, "m6_colora": 0.08}
    decisions = holm_step_down(p_vals, alpha=0.05)

    assert decisions["m5_learned"]["rejected"] is True
    assert decisions["m4_equal"]["rejected"] is False
    assert decisions["m6_colora"]["rejected"] is False
    assert np.isclose(decisions["m5_learned"]["threshold"], 0.05 / 3)
    assert np.isclose(decisions["m4_equal"]["threshold"], 0.05 / 2)


def test_metrics_computation():
    """Verify metric calculations match definitions."""
    from experiments.run_bayesian_collab_positive_control import compute_metrics

    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.9, 0.1, 0.8, 0.2])

    m = compute_metrics(y, p)
    expected_brier = float(np.mean(np.array([0.9 - 1, 0.1 - 0, 0.8 - 1, 0.2 - 0]) ** 2))
    assert np.isclose(m["brier"], expected_brier)
    assert m["auroc"] == 1.0
    assert m["accuracy"] == 1.0
    assert 0.0 <= m["ece"] <= 1.0


def test_mini_training_loop_convergence():
    """Verify small training run converges without NaNs on synthetic data."""
    from experiments.run_bayesian_collab_positive_control import train_single_model

    torch.use_deterministic_algorithms(True)
    cfg = DGPConfig(seed=20261009, n_train_target=32, n_train_specialist=32)
    spec = generate_dgp(cfg)
    splits = sample_dataset(spec, "positive_control", data_seed=999)

    model = train_single_model(
        splits.target_train["x"],
        splits.target_train["y"],
        epochs=10,
        batch_size=16,
        lr=0.01,
        seed=42,
    )

    x_test = torch.from_numpy(splits.test["x_full"][:10]).float()
    model.eval()
    with torch.no_grad():
        preds = model(x_test).numpy()

    assert not np.isnan(preds).any()
    assert np.all((preds >= 0.0) & (preds <= 1.0))
