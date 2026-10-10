"""Unit tests for diagnostic utilities in exploratory follow-up investigation."""

from __future__ import annotations

import numpy as np

from hypergraph_portfolio.diagnostics_utils import (
    apply_platt_scaling,
    brier_score_decomposition,
    compute_task_gradient_cosine,
    fit_log_odds_combiner,
    fit_platt_scaling,
    predict_log_odds_combiner,
)
from hypergraph_portfolio.synthetic_multitask import (
    DGPConfig,
    generate_dgp,
    sample_dataset,
)


def test_brier_score_decomposition():
    """Verify Brier decomposition identity: Brier approx Reliability - Resolution + Uncertainty."""
    rng = np.random.default_rng(42)
    y = (rng.uniform(0, 1, 1000) > 0.6).astype(float)
    p = np.clip(y * 0.7 + rng.normal(0, 0.2, 1000), 0.05, 0.95)

    decomp = brier_score_decomposition(y, p, n_bins=10)

    # Within binning approximation error (< 0.01)
    assert abs(decomp["brier"] - decomp["reconstruction_brier"]) < 0.01
    assert decomp["reliability"] >= 0.0
    assert decomp["resolution"] >= 0.0
    assert decomp["uncertainty"] > 0.0


def test_platt_scaling_calibration():
    """Verify Platt scaling calibrates overconfident probabilities."""
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0] * 50, dtype=float)
    # Overconfident extreme probabilities
    p_uncal = np.where(y == 1, 0.999, 0.001)
    # Add miscalibration noise
    p_uncal[0:20] = 0.999  # false positives

    clf = fit_platt_scaling(p_uncal, y)
    p_cal = apply_platt_scaling(clf, p_uncal)

    # Calibrated probabilities should be valid probabilities and improve Brier score
    assert np.all((p_cal >= 0.0) & (p_cal <= 1.0))
    brier_uncal = float(np.mean((p_uncal - y) ** 2))
    brier_cal = float(np.mean((p_cal - y) ** 2))
    assert brier_cal < brier_uncal


def test_log_odds_combiner_dynamic_range():
    """Verify log-odds combiner can output higher joint probability than linear average."""
    p_A = np.array([0.85, 0.15, 0.85, 0.15] * 30)
    p_B = np.array([0.85, 0.15, 0.15, 0.85] * 30)
    y = np.array([1.0, 0.0, 0.5, 0.5] * 30)
    y = (y > 0.4).astype(float)

    clf = fit_log_odds_combiner(p_A, p_B, y)
    p_comb = predict_log_odds_combiner(clf, np.array([0.85]), np.array([0.85]))

    # When both channels agree strongly on 1, joint probability exceeds linear average (0.85)
    assert p_comb[0] >= 0.85


def test_task_gradient_cosine():
    """Verify task gradient cosine computation."""
    g1 = np.array([1.0, 0.0, -1.0])
    g2 = np.array([1.0, 0.0, -1.0])
    g3 = np.array([-1.0, 0.0, 1.0])

    assert np.isclose(compute_task_gradient_cosine(g1, g2), 1.0)
    assert np.isclose(compute_task_gradient_cosine(g1, g3), -1.0)


def test_diagnostic_split_isolation():
    """Verify fresh diagnostic split is disjoint from frozen splits."""
    spec = generate_dgp(DGPConfig(seed=20261009))
    frozen_splits = sample_dataset(spec, "positive_control", data_seed=20261009 + 10)
    diag_splits = sample_dataset(spec, "positive_control", data_seed=20261009 + 999)

    # Features must not be identical
    assert not np.allclose(
        frozen_splits.development["x_full"][:10], diag_splits.development["x_full"][:10]
    )
    assert not np.allclose(
        frozen_splits.test["x_full"][:10], diag_splits.test["x_full"][:10]
    )
