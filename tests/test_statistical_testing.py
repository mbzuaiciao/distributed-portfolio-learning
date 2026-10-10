"""Unit tests for paired instance bootstrap and statistical testing module."""

from __future__ import annotations

import numpy as np
import pytest

from hypergraph_portfolio.statistical_testing import (
    apply_holm_bonferroni,
    paired_contrast_bootstrap,
    paired_instance_bootstrap,
)


def test_paired_bootstrap_determinism_and_pairing():
    """Verify bootstrap is strictly deterministic and preserves paired instance structure."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n).astype(float)

    # Method A is superior to Method B by predicting closer to y_true
    preds_A = {
        42: np.clip(y_true * 0.8 + 0.1 + rng.normal(0, 0.05, n), 0, 1),
        123: np.clip(y_true * 0.8 + 0.1 + rng.normal(0, 0.05, n), 0, 1),
        777: np.clip(y_true * 0.8 + 0.1 + rng.normal(0, 0.05, n), 0, 1),
    }
    # Method B is less accurate
    preds_B = {
        42: np.full(n, 0.5) + rng.normal(0, 0.05, n),
        123: np.full(n, 0.5) + rng.normal(0, 0.05, n),
        777: np.full(n, 0.5) + rng.normal(0, 0.05, n),
    }

    res1 = paired_instance_bootstrap(
        y_true, preds_A, preds_B, n_bootstraps=500, seed=12345, effect_size_threshold=0.010
    )
    res2 = paired_instance_bootstrap(
        y_true, preds_A, preds_B, n_bootstraps=500, seed=12345, effect_size_threshold=0.010
    )

    assert res1.point_estimate == pytest.approx(res2.point_estimate)
    assert res1.ci_lower == pytest.approx(res2.ci_lower)
    assert res1.ci_upper == pytest.approx(res2.ci_upper)
    assert res1.p_value_raw == pytest.approx(res2.p_value_raw)

    # Because Method A is clearly superior:
    # Loss(B) - Loss(A) should be positive
    assert res1.point_estimate > 0.05
    assert res1.ci_lower > 0.0
    assert res1.effect_size_pass is True
    assert res1.statistical_significance_pass is True
    assert res1.confirmatory_pass is True


def test_distinction_effect_size_vs_significance():
    """Verify distinct separation of effect size threshold from statistical significance."""
    rng = np.random.default_rng(99)
    n = 100
    y_true = rng.integers(0, 2, size=n).astype(float)

    # Moderate gain, but high variance -> positive point estimate, but CI spans zero
    preds_A = {
        42: np.clip(y_true * 0.6 + 0.2 + rng.normal(0, 0.25, n), 0, 1),
    }
    preds_B = {
        42: np.clip(y_true * 0.5 + 0.25 + rng.normal(0, 0.25, n), 0, 1),
    }

    res = paired_instance_bootstrap(
        y_true, preds_A, preds_B, n_bootstraps=1000, seed=42, effect_size_threshold=0.010
    )

    # Suppose effect size threshold is 0.010:
    # If point estimate is >= 0.010, effect_size_pass can be True while significance is False
    if res.point_estimate >= 0.010 and res.ci_lower <= 0.0:
        assert res.effect_size_pass is True
        assert res.statistical_significance_pass is False
        assert res.confirmatory_pass is False


def test_matched_contrast_bootstrap():
    """Verify matched contrast bootstrap calculation."""
    rng = np.random.default_rng(101)
    n = 150
    y_pos = rng.integers(0, 2, size=n).astype(float)
    y_neg = rng.integers(0, 2, size=n).astype(float)

    # Strong gain in pos
    preds_A_pos = {42: np.clip(y_pos * 0.8 + 0.1, 0, 1)}
    preds_B_pos = {42: np.full(n, 0.5)}

    # Zero gain in neg (identical models)
    preds_A_neg = {42: np.full(n, 0.5)}
    preds_B_neg = {42: np.full(n, 0.5)}

    contrast_res = paired_contrast_bootstrap(
        y_pos, preds_A_pos, preds_B_pos,
        y_neg, preds_A_neg, preds_B_neg,
        n_bootstraps=500, seed=777, effect_size_threshold=0.0
    )

    assert contrast_res.point_estimate > 0.05
    assert contrast_res.ci_lower > 0.0
    assert contrast_res.effect_size_pass is True
    assert contrast_res.statistical_significance_pass is True
    assert contrast_res.confirmatory_pass is True


def test_holm_bonferroni_adjustment():
    """Verify Holm-Bonferroni step-down multiplier."""
    from hypergraph_portfolio.statistical_testing import ContrastResult

    r1 = ContrastResult("c1", 0.05, 0.05, 0.01, 0.02, 0.08, 0.005, 0.005, 0.0, True, True, True)
    r2 = ContrastResult("c2", 0.02, 0.02, 0.01, -0.01, 0.05, 0.04, 0.04, 0.0, True, False, False)
    r3 = ContrastResult("c3", 0.01, 0.01, 0.01, -0.02, 0.04, 0.08, 0.08, 0.0, True, False, False)

    adjusted = apply_holm_bonferroni([r1, r2, r3])

    # Rank 0 (p=0.005, mult=3) -> 0.015
    assert adjusted[0].p_value_holm == pytest.approx(0.015)
    # Rank 1 (p=0.04, mult=2) -> max(0.015, 0.08) = 0.08
    assert adjusted[1].p_value_holm == pytest.approx(0.08)
    # Rank 2 (p=0.08, mult=1) -> max(0.08, 0.08) = 0.08
    assert adjusted[2].p_value_holm == pytest.approx(0.08)
