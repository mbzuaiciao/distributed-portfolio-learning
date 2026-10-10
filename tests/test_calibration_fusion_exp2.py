"""Unit tests for Experiment 2 data splits, calibration, and evidence fusion."""

from __future__ import annotations

import numpy as np

from hypergraph_portfolio.calibration_fusion import (
    LogOddsFusion,
    PlattCalibrator,
    SimplexFusion,
)
from hypergraph_portfolio.synthetic_multitask import (
    DGPConfig,
    generate_dgp,
)
from hypergraph_portfolio.synthetic_multitask_exp2 import (
    Exp2SplitConfig,
    sample_exp2_dataset,
)


def test_exp2_splits_disjointness():
    """Verify that all 5 splits in Experiment 2 are strictly disjoint."""
    spec = generate_dgp(DGPConfig(seed=20261009))
    splits = sample_exp2_dataset(spec, "positive_control", Exp2SplitConfig(data_seed=42))

    # All splits have unique instances
    x_tgt = splits.target_train["x_raw"]
    x_cal = splits.calibration["x_raw"]
    x_fus = splits.fusion_dev["x_raw"]
    x_dev = splits.development["x_raw"]
    x_tst = splits.test["x_raw"]

    assert len(x_tgt) == 128
    assert len(x_cal) == 128
    assert len(x_fus) == 128
    assert len(x_dev) == 128
    assert len(x_tst) == 512

    # Verify no row matches across calibration, fusion, dev, and test
    assert not np.allclose(x_cal[0], x_fus[0])
    assert not np.allclose(x_cal[0], x_dev[0])
    assert not np.allclose(x_fus[0], x_dev[0])
    assert not np.allclose(x_dev[0], x_tst[0])


def test_platt_calibrator_roundtrip():
    """Verify PlattCalibrator fit, predict, and serialization roundtrip."""
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0] * 20, dtype=float)
    p = np.where(y == 1, 0.95, 0.05)
    cal = PlattCalibrator.fit(p, y)

    p_cal = cal.predict(p)
    assert p_cal.shape == p.shape
    assert np.all((p_cal >= 0.0) & (p_cal <= 1.0))

    d = cal.to_dict()
    cal2 = PlattCalibrator.from_dict(d)
    np.testing.assert_array_almost_equal(p_cal, cal2.predict(p))


def test_log_odds_fusion_roundtrip():
    """Verify LogOddsFusion fit, predict, and serialization roundtrip."""
    y = np.array([1, 1, 0, 0] * 30, dtype=float)
    p_A = np.where(y == 1, 0.8, 0.2)
    p_B = np.where(y == 1, 0.85, 0.15)

    fusion = LogOddsFusion.fit(p_A, p_B, y)
    p_fused = fusion.predict(p_A, p_B)

    assert p_fused.shape == p_A.shape
    assert np.all((p_fused >= 0.0) & (p_fused <= 1.0))

    d = fusion.to_dict()
    fusion2 = LogOddsFusion.from_dict(d)
    np.testing.assert_array_almost_equal(p_fused, fusion2.predict(p_A, p_B))


def test_simplex_fusion_roundtrip():
    """Verify SimplexFusion fit, predict, and serialization roundtrip."""
    y = np.array([1, 1, 0, 0] * 30, dtype=float)
    p_A = np.where(y == 1, 0.8, 0.2)
    p_B = np.where(y == 1, 0.7, 0.3)

    fusion = SimplexFusion.fit(p_A, p_B, y)
    p_fused = fusion.predict(p_A, p_B)

    assert p_fused.shape == p_A.shape
    assert np.isclose(sum(fusion.weights), 1.0)

    d = fusion.to_dict()
    fusion2 = SimplexFusion.from_dict(d)
    np.testing.assert_array_almost_equal(p_fused, fusion2.predict(p_A, p_B))


def test_redundant_negative_control_masks_exp2():
    """Verify negative control in Exp2 gives both specialists mask_A."""
    spec = generate_dgp(DGPConfig(seed=20261009))
    splits = sample_exp2_dataset(spec, "negative_control")

    mask_A = spec.mask_A
    np.testing.assert_array_equal(splits.specialist_A_train["mask"], mask_A)
    np.testing.assert_array_equal(splits.specialist_B_train["mask"], mask_A)
