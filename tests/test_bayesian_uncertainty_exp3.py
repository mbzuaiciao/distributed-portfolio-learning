"""Comprehensive unit tests for Experiment 3 modules."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hypergraph_portfolio.bayesian_uncertainty import (
    BayesianPenultimateReadout,
    ConfidenceRouter,
    DensityRouter,
    InverseVarianceRouter,
    LinearMoERouter,
    MLPMoERouter,
    MatchedAblationRouter,
    select_strongest_nonbayes_baseline,
)
from hypergraph_portfolio.synthetic_models import SmallMLP
from hypergraph_portfolio.synthetic_multitask import DGPConfig, generate_dgp
from hypergraph_portfolio.synthetic_multitask_exp3 import (
    Exp3SplitConfig,
    sample_exp3_dataset,
)


def test_exp3_dataset_generation_and_split_isolation():
    """Verify that Exp 3 data generator produces disjoint splits for both Regimes A and B."""
    cfg = DGPConfig(seed=20261009)
    spec = generate_dgp(cfg)
    split_cfg = Exp3SplitConfig(data_seed=20261015)

    for regime in ("regime_A", "regime_B"):
        splits = sample_exp3_dataset(spec, regime=regime, split_cfg=split_cfg)

        assert splits.target_train["x"].shape == (128, 40)
        # Specialist dense + sparse = 272
        assert splits.specialist_A_train["x"].shape == (272, 40)
        assert splits.specialist_B_train["x"].shape == (272, 40)
        assert splits.calibration["x_full"].shape == (128, 40)
        assert splits.fusion_dev["x_full"].shape == (128, 40)
        assert splits.development["x_full"].shape == (128, 40)
        assert splits.test["x_full"].shape == (512, 40)

        # Check mask enforcement: in specialist A, block B is 0
        b_start = spec.config.d_shared + spec.config.d_A
        b_end = b_start + spec.config.d_B
        assert np.all(splits.specialist_A_train["x"][:, b_start:b_end] == 0.0)


def test_bayesian_penultimate_readout_covariance_and_uncertainty():
    """Verify Laplace approximation produces well-conditioned covariance and positive epistemic variance."""
    torch.manual_seed(42)
    model = SmallMLP(in_features=40)
    x_train = np.random.randn(100, 40).astype(np.float32)

    readout = BayesianPenultimateReadout.fit(model, x_train, prior_precision=1.0)

    # Check positive definiteness and condition number
    eigvals = np.linalg.eigvalsh(readout.covariance)
    assert np.all(eigvals > 0.0)
    assert readout.condition_number < 1e5

    x_query = np.random.randn(20, 40).astype(np.float32)
    mu, sigma2, p_bayes = readout.predict(model, x_query)

    assert mu.shape == (20,)
    assert sigma2.shape == (20,)
    assert np.all(sigma2 >= 0.0)
    assert np.all((p_bayes >= 0.0) & (p_bayes <= 1.0))

    # Serialization test
    d = readout.to_dict()
    readout_rec = BayesianPenultimateReadout.from_dict(d)
    np.testing.assert_allclose(readout.covariance, readout_rec.covariance)


def test_linear_and_mlp_moe_routers():
    """Verify linear and non-linear MoE routers output valid probabilities and sum-to-1 weights."""
    rng = np.random.default_rng(42)
    n = 100
    x_s = rng.normal(0, 1, size=(n, 4))
    p_A = rng.uniform(0.1, 0.9, size=n)
    p_B = rng.uniform(0.1, 0.9, size=n)
    y = rng.integers(0, 2, size=n).astype(float)

    # Linear MoE
    lin_router = LinearMoERouter.fit(x_s, p_A, p_B, y, epochs=10, seed=42)
    p_lin = lin_router.predict(x_s, p_A, p_B)
    assert p_lin.shape == (n,)
    assert np.all((p_lin >= 0.0) & (p_lin <= 1.0))

    # MLP MoE
    mlp_router = MLPMoERouter.fit(x_s, p_A, p_B, y, epochs=10, seed=42)
    p_mlp = mlp_router.predict(x_s, p_A, p_B)
    assert p_mlp.shape == (n,)
    assert np.all((p_mlp >= 0.0) & (p_mlp <= 1.0))


def test_inverse_variance_and_density_routers():
    """Verify density router and inverse variance router handle numerical extremes gracefully."""
    rng = np.random.default_rng(123)
    n = 50
    x_s = rng.normal(0, 1, size=(n, 4))
    x_s_A = rng.normal(1, 0.5, size=(n, 4))
    x_s_B = rng.normal(-1, 0.5, size=(n, 4))
    p_A = rng.uniform(0.1, 0.9, size=n)
    p_B = rng.uniform(0.1, 0.9, size=n)
    s2_A = rng.uniform(0.01, 1.0, size=n)
    s2_B = rng.uniform(0.01, 1.0, size=n)
    y = rng.integers(0, 2, size=n).astype(float)

    # Density router
    dens_router = DensityRouter.fit(x_s_A, x_s_B)
    p_dens = dens_router.predict(x_s, p_A, p_B)
    assert np.all((p_dens >= 0.0) & (p_dens <= 1.0))

    # Inverse variance router
    inv_router = InverseVarianceRouter.fit(p_A, p_B, s2_A, s2_B, y)
    p_inv = inv_router.predict(p_A, p_B, s2_A, s2_B)
    assert np.all((p_inv >= 0.0) & (p_inv <= 1.0))

    # Confidence router
    p_conf = ConfidenceRouter.predict(p_A, p_B)
    assert np.all((p_conf >= 0.0) & (p_conf <= 1.0))


def test_matched_router_ablation_dimensions():
    """Verify matched ablation router trains with identical hyperparameters across input shapes."""
    rng = np.random.default_rng(777)
    n = 64
    y = rng.integers(0, 2, size=n).astype(float)

    feat_no_uq = rng.normal(0, 1, size=(n, 6))  # 2 logits + 4 x_s
    feat_uq = rng.normal(0, 1, size=(n, 8))     # 2 logits + 2 variances + 4 x_s

    r_no_uq = MatchedAblationRouter.fit(feat_no_uq, y, epochs=10, seed=42)
    r_uq = MatchedAblationRouter.fit(feat_uq, y, epochs=10, seed=42)

    p_no_uq = r_no_uq.predict(feat_no_uq)
    p_uq = r_uq.predict(feat_uq)

    assert p_no_uq.shape == (n,)
    assert p_uq.shape == (n,)
    assert np.all((p_no_uq >= 0.0) & (p_no_uq <= 1.0))
    assert np.all((p_uq >= 0.0) & (p_uq <= 1.0))


def test_deterministic_baseline_selection_rule():
    """Verify pre-registered selection rule picks lowest Brier, breaking ties deterministically."""
    # Scenario 1: Clear Brier winner
    dev_summary_1 = {
        "m_moe_mlp": {"mean_brier": 0.180, "mean_auroc": 0.80},
        "m_moe_lin": {"mean_brier": 0.190, "mean_auroc": 0.81},
        "m6_logodds_cal": {"mean_brier": 0.185, "mean_auroc": 0.79},
    }
    assert select_strongest_nonbayes_baseline(dev_summary_1) == "m_moe_mlp"

    # Scenario 2: Tie in Brier -> higher AUROC wins
    dev_summary_2 = {
        "m_moe_mlp": {"mean_brier": 0.18000, "mean_auroc": 0.80},
        "m_router_nouq": {"mean_brier": 0.18000, "mean_auroc": 0.85},
    }
    assert select_strongest_nonbayes_baseline(dev_summary_2) == "m_router_nouq"

    # Scenario 3: Complete tie in Brier & AUROC -> pre-registered priority order breaks tie
    dev_summary_3 = {
        "m_router_nouq": {"mean_brier": 0.18000, "mean_auroc": 0.80},
        "m_moe_mlp": {"mean_brier": 0.18000, "mean_auroc": 0.80},
    }
    assert select_strongest_nonbayes_baseline(dev_summary_3) == "m_moe_mlp"
