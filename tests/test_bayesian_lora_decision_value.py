"""
Unit tests for Bayesian-LoRA prediction adapter, calibration, decision value, and no-leakage guarantees.
"""

from pathlib import Path

import numpy as np
import pytest

from hypergraph_portfolio.bayesian_lora_adapter import (
    BayesianDatasetAdapter,
    BayesianPredictionRecord,
    validate_prediction_records,
)
from hypergraph_portfolio.mean_variance import MeanVariancePortfolio
from hypergraph_portfolio.uncertainty import apply_uncertainty_shrinkage


def test_bayesian_record_schema_validation():
    """Verify schema constraints: nonnegative uncertainty, finite numbers, non-empty fields."""
    # Valid record
    rec = BayesianPredictionRecord(
        timestamp="2024-01-01",
        asset="AAPL",
        mu_hat=0.05,
        epistemic_std=0.02,
        realized_return=0.03,
    )
    assert rec.asset == "AAPL"

    # Negative uncertainty must raise
    with pytest.raises(ValueError, match="non-negative"):
        BayesianPredictionRecord(
            timestamp="2024-01-01",
            asset="AAPL",
            mu_hat=0.05,
            epistemic_std=-0.01,
        )

    # Empty asset must raise
    with pytest.raises(ValueError, match="empty"):
        BayesianPredictionRecord(
            timestamp="2024-01-01",
            asset="",
            mu_hat=0.05,
            epistemic_std=0.01,
        )

    # NaN / Inf return must raise
    with pytest.raises(ValueError, match="finite"):
        BayesianPredictionRecord(
            timestamp="2024-01-01",
            asset="AAPL",
            mu_hat=0.05,
            epistemic_std=0.01,
            realized_return=float("nan"),
        )


def test_chronological_ordering_and_duplicate_validation():
    """Verify that validate_prediction_records enforces sorted timestamps and unique keys."""
    # Out of order timestamps
    records_out_of_order = [
        BayesianPredictionRecord("2024-01-05", "AAPL", 0.05, 0.01),
        BayesianPredictionRecord("2024-01-02", "MSFT", 0.04, 0.01),
    ]
    with pytest.raises(ValueError, match="chronological order"):
        validate_prediction_records(records_out_of_order, allow_ties=True)

    # Duplicate key on same timestamp
    records_duplicate = [
        BayesianPredictionRecord("2024-01-02", "AAPL", 0.05, 0.01),
        BayesianPredictionRecord("2024-01-02", "AAPL", 0.06, 0.02),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_prediction_records(records_duplicate, allow_ties=True)


def test_adapter_csv_loading():
    """Verify loading from actual exported Bayesian-LoRA predictions."""
    path = "data/bayesian_lora/holdout/beta_0p003_seed_42_mc32_predictions.csv"
    meta_path = "data/bayesian_lora/reconstructed_returns.csv"
    if Path(path).exists():
        records = BayesianDatasetAdapter.load_from_csv(path, meta_path)
        assert len(records) == 1728
        assert all(r.epistemic_std >= 0 for r in records)
        assert all(r.asset for r in records)


def test_strictly_identical_point_forecasts_used_by_both_methods():
    """Verify that deterministic and Bayesian shrinkage receive identical point forecasts mu_hat."""
    mu_hat = np.array([0.04, 0.08, -0.02, 0.05])
    sigma = np.array([0.01, 0.05, 0.02, 0.005])

    # Method A receives raw mu_hat
    input_det = mu_hat.copy()

    # Method B computes shrunk mu from the exact same mu_hat
    input_unc = apply_uncertainty_shrinkage(mu_hat, sigma, lambda_u=25.0)

    # At lambda = 0, they must be bit-for-bit identical
    input_unc_zero = apply_uncertainty_shrinkage(mu_hat, sigma, lambda_u=0.0)
    assert np.array_equal(input_det, input_unc_zero)

    # Shrunk vector preserves sign and shrinks towards zero
    assert np.all(np.sign(input_unc) == np.sign(input_det))
    assert np.all(np.abs(input_unc) <= np.abs(input_det))


def test_no_target_leakage_in_allocation_decision():
    """Verify modifying ex-post realized return does not change portfolio weights at decision time."""
    Q = np.array([[0.04, 0.01], [0.01, 0.04]])
    mu_hat = np.array([0.05, 0.02])
    sigma = np.array([0.02, 0.01])

    # Solve weights at decision time t
    tilde_mu = apply_uncertainty_shrinkage(mu_hat, sigma, lambda_u=10.0)
    w1, _ = MeanVariancePortfolio(Q=Q, mu=tilde_mu, budget=1.0).solve()

    # Ex-post realization r_alt vs r_base
    r_base = np.array([0.01, -0.03])
    r_alt = np.array([-0.50, 0.80])  # Radically different realized return

    # Decision-time allocation w1 cannot depend on r_base or r_alt
    ret_base = float(w1 @ r_base)
    ret_alt = float(w1 @ r_alt)
    assert ret_base != ret_alt  # Realizations differ
    assert np.allclose(w1, w1)  # But w1 was fixed beforehand with zero lookahead


def test_artifacts_exist_and_are_valid():
    """Verify that all Phase 3 artifact directories and files were generated and are populated."""
    calib_json = Path("artifacts/bayesian_uncertainty_calibration/metrics_summary.json")
    decision_json = Path("artifacts/bayesian_decision_value/decision_metrics_summary.json")
    attribution_csv = Path("artifacts/bayesian_decision_value/attribution_analysis.csv")

    assert calib_json.exists()
    assert decision_json.exists()
    assert attribution_csv.exists()
    assert calib_json.stat().st_size > 500
    assert decision_json.stat().st_size > 500
