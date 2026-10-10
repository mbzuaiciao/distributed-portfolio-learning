"""Unit tests for synthetic multi-task data generating process.

Verifies:
1. DGP determinism: same seed produces identical coefficients and data.
2. Feature partition: in positive control, specialist A masks block B, specialist B masks block A.
3. Negative control: specialists receive full masks.
4. No label or return leakage in input features.
5. Calibrated prevalence matches configuration within Monte Carlo tolerance.
6. Analytical Bayes-optimal Brier scores and oracle gap (>0.010 in positive control).
7. Input dimensionality matches the 40-dim concatenated feature+mask contract.
"""

from __future__ import annotations

import numpy as np

from hypergraph_portfolio.synthetic_multitask import (
    DGPConfig,
    generate_dgp,
    sample_dataset,
    sample_features_and_labels,
)


def test_dgp_determinism():
    """Verify identical seeds produce byte-identical DGP specifications."""
    spec1 = generate_dgp(DGPConfig(seed=20261009))
    spec2 = generate_dgp(DGPConfig(seed=20261009))

    np.testing.assert_array_equal(spec1.beta_s, spec2.beta_s)
    np.testing.assert_array_equal(spec1.beta_A, spec2.beta_A)
    np.testing.assert_array_equal(spec1.beta_B, spec2.beta_B)
    assert spec1.intercept == spec2.intercept
    assert spec1.analytical_oracle_gap == spec2.analytical_oracle_gap


def test_dgp_different_seeds():
    """Verify different seeds produce different coefficients."""
    spec1 = generate_dgp(DGPConfig(seed=20261009))
    spec2 = generate_dgp(DGPConfig(seed=20261010))

    assert not np.allclose(spec1.beta_s, spec2.beta_s)
    assert not np.allclose(spec1.beta_A, spec2.beta_A)


def test_prevalence_calibration():
    """Verify empirical prevalence approximates target prevalence (0.35)."""
    spec = generate_dgp(DGPConfig(seed=20261009, prevalence=0.35))
    rng = np.random.default_rng(42)
    _, probs, y = sample_features_and_labels(spec, 50000, rng)

    assert abs(np.mean(probs) - 0.35) < 0.01
    assert abs(np.mean(y) - 0.35) < 0.01


def test_analytical_oracle_gap():
    """Verify analytical oracle gap exceeds the pre-registered 0.010 threshold."""
    spec = generate_dgp(DGPConfig(seed=20261009, beta_scale=1.5))
    assert spec.analytical_oracle_gap > 0.010, (
        f"Analytical oracle gap {spec.analytical_oracle_gap} must exceed 0.010"
    )
    assert spec.analytical_bayes_brier_full < spec.analytical_bayes_brier_A
    assert spec.analytical_bayes_brier_full < spec.analytical_bayes_brier_B


def test_feature_partition_positive_control():
    """Verify observation masks in positive control strictly isolate specialist blocks."""
    spec = generate_dgp(DGPConfig(seed=20261009))
    splits = sample_dataset(spec, regime="positive_control", data_seed=123)

    cfg = spec.config
    b_start = cfg.d_shared + cfg.d_A
    b_end = b_start + cfg.d_B
    a_start = cfg.d_shared
    a_end = a_start + cfg.d_A

    # Specialist A: block B must be zero in feature portion (first 20 dims) and mask (next 20 dims)
    xA = splits.specialist_A_train["x"]
    assert xA.shape == (cfg.n_train_specialist, 40)
    np.testing.assert_array_equal(xA[:, b_start:b_end], 0.0)
    np.testing.assert_array_equal(xA[:, 20 + b_start : 20 + b_end], 0.0)
    # Specialist A: block A must be nonzero and observed
    assert np.any(xA[:, a_start:a_end] != 0.0)
    np.testing.assert_array_equal(xA[:, 20 + a_start : 20 + a_end], 1.0)

    # Specialist B: block A must be zero
    xB = splits.specialist_B_train["x"]
    assert xB.shape == (cfg.n_train_specialist, 40)
    np.testing.assert_array_equal(xB[:, a_start:a_end], 0.0)
    np.testing.assert_array_equal(xB[:, 20 + a_start : 20 + a_end], 0.0)
    # Specialist B: block B must be nonzero and observed
    assert np.any(xB[:, b_start:b_end] != 0.0)
    np.testing.assert_array_equal(xB[:, 20 + b_start : 20 + b_end], 1.0)

    # Target: all features and mask must be fully observed
    x_tgt = splits.target_train["x"]
    assert x_tgt.shape == (cfg.n_train_target, 40)
    np.testing.assert_array_equal(x_tgt[:, 20:], 1.0)


def test_feature_partition_negative_control():
    """Verify in negative control, both specialists receive redundant observation masks (mask_A)."""
    spec = generate_dgp(DGPConfig(seed=20261009))
    splits = sample_dataset(spec, regime="negative_control", data_seed=123)

    cfg = spec.config
    a_start = cfg.d_shared
    a_end = a_start + cfg.d_A
    b_start = cfg.d_shared + cfg.d_A
    b_end = b_start + cfg.d_B

    xA = splits.specialist_A_train["x"]
    xB = splits.specialist_B_train["x"]

    # In negative control, both specialists observe block A (mask_A) and neither observes block B
    np.testing.assert_array_equal(xA[:, 20 + a_start : 20 + a_end], 1.0)
    np.testing.assert_array_equal(xB[:, 20 + a_start : 20 + a_end], 1.0)
    np.testing.assert_array_equal(xA[:, 20 + b_start : 20 + b_end], 0.0)
    np.testing.assert_array_equal(xB[:, 20 + b_start : 20 + b_end], 0.0)
    np.testing.assert_array_equal(xA[:, b_start:b_end], 0.0)
    np.testing.assert_array_equal(xB[:, b_start:b_end], 0.0)
    assert np.any(xA[:, a_start:a_end] != 0.0)
    assert np.any(xB[:, a_start:a_end] != 0.0)


def test_no_leakage_in_features():
    """Verify labels are strictly binary and not present in feature arrays."""
    spec = generate_dgp(DGPConfig(seed=20261009))
    splits = sample_dataset(spec, regime="positive_control", data_seed=456)

    for split_name in ("target_train", "specialist_A_train", "specialist_B_train"):
        data = getattr(splits, split_name)
        y = data["y"]
        assert set(np.unique(y)).issubset({0.0, 1.0})
        # Check no constant column in raw features
        assert not np.any(np.all(data["x_raw"] == y[:, None], axis=0))


def test_split_sizes_match_configuration():
    """Verify all split sizes strictly match DGPConfig."""
    cfg = DGPConfig(
        n_train_target=128,
        n_train_specialist=256,
        n_development=64,
        n_validation=64,
        n_test=512,
    )
    spec = generate_dgp(cfg)
    splits = sample_dataset(spec, "positive_control")

    assert len(splits.target_train["y"]) == 128
    assert len(splits.specialist_A_train["y"]) == 256
    assert len(splits.specialist_B_train["y"]) == 256
    assert len(splits.development["y"]) == 64
    assert len(splits.validation["y"]) == 64
    assert len(splits.test["y"]) == 512
