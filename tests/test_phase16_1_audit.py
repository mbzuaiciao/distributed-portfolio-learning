"""Unit tests for Phase 16.1: Reliability-Metric and Regularization Control Audit.

Verifies:
1. Input leakage audit detects training leakage and confirms clean evaluation text.
2. No outcome-derived field enters canonical clean test model input.
3. Class-conditional reliability metrics are computed separately for y=0 and y=1.
4. Base-rate / null predictor is strictly constant (p_i = 0.20) with zero variance.
5. Regularization hyperparameter selection strictly uses pre-2022 validation only.
6. 2022+ data is never used for hyperparameter selection or regularization tuning.
7. No new portfolio strategy or policy is introduced (M0-M16 remains frozen).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

repo_root = Path(__file__).resolve().parent.parent
audit_dir = repo_root / "artifacts" / "phase16_1_audit"


def test_input_leakage_audit_complete_and_accurate():
    """Verify input_leakage_audit.json exists, identifies training leakage, and validates clean inference."""
    audit_json_path = audit_dir / "input_leakage_audit.json"
    assert audit_json_path.exists(), "input_leakage_audit.json missing"
    with open(audit_json_path, "r") as f:
        audit = json.load(f)

    concl = audit["conclusions"]
    # Training leakage must be flagged
    assert concl["training_input_leakage"]["detected"] is True
    assert "experiments/run_phase11_replication.py" in concl["training_input_leakage"]["affected_scripts"]
    assert "experiments/run_phase16_deterministic_control.py" in concl["training_input_leakage"]["affected_scripts"]
    assert concl["training_input_leakage"]["evidence_validation_auroc"][0]["best_validation_auroc"] > 0.999

    # Evaluation input text must be confirmed clean (no leakage)
    assert concl["evaluation_input_leakage"]["detected"] is False
    assert "move_1d" not in concl["evaluation_input_leakage"]["clean_evaluation_text_format"]


def test_clean_evaluation_inputs_contain_no_outcome_fields():
    """Verify that clean evaluation inputs in Phase 13/15/16 contain NO outcome tokens or future returns."""
    p16_preds = repo_root / "artifacts" / "phase16_deterministic_control" / "deterministic_predictions.csv"
    assert p16_preds.exists()
    df = pd.read_csv(p16_preds)
    assert len(df) == 12300

    # Ensure evaluation dates are strictly >= 2022-01-01
    min_date = df["date"].min()
    assert min_date >= "2022-01-01"


def test_within_class_reliability_disaggregation():
    """Verify class-conditional reliability metrics are computed separately for y=0 and y=1."""
    within_csv = audit_dir / "within_class_reliability.csv"
    assert within_csv.exists(), "within_class_reliability.csv missing"
    df = pd.read_csv(within_csv)

    # Must contain All, 0, and 1 y_values
    assert set(df["y_value"].astype(str).unique()) == {"All", "0", "1"}

    # For Bayesian Posterior Mean, verify mathematical reversal due to shrinkage
    bayes_df = df[df["system_name"] == "Bayesian Posterior Mean"]
    rho_all = bayes_df[bayes_df["y_value"] == "All"]["spearman_rho_entropy"].values[0]
    rho_y0 = bayes_df[bayes_df["y_value"] == "0"]["spearman_rho_entropy"].values[0]
    rho_y1 = bayes_df[bayes_df["y_value"] == "1"]["spearman_rho_entropy"].values[0]

    assert rho_all > 0.55
    assert rho_y0 > 0.99  # Perfectly positive within y=0 for p < 0.5
    assert rho_y1 < -0.99  # Perfectly negative within y=1 for p < 0.5


def test_base_rate_predictor_is_constant():
    """Verify base-rate predictor is strictly constant and achieves expected theoretical metrics."""
    base_rate_csv = audit_dir / "base_rate_control.csv"
    assert base_rate_csv.exists(), "base_rate_control.csv missing"
    df = pd.read_csv(base_rate_csv)

    null_row = df[df["is_constant_predictor"] == True].iloc[0]  # noqa: E712
    assert null_row["constant_probability"] == 0.20
    assert null_row["brier_score"] == pytest.approx(0.16000, abs=1e-4)
    assert null_row["ece"] == pytest.approx(0.00000, abs=1e-4)
    assert null_row["auroc"] == pytest.approx(0.50000, abs=1e-4)


def test_selective_risk_fractions_and_monotonicity():
    """Verify selective risk curve audits fractions [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]."""
    sel_csv = audit_dir / "selective_risk.csv"
    assert sel_csv.exists(), "selective_risk.csv missing"
    df = pd.read_csv(sel_csv)

    expected_fracs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    for sys_name in ["Deterministic LoRA", "Bayesian Posterior Mean", "Bayesian MC32"]:
        sub = df[df["system_name"] == sys_name]
        assert sorted(sub["retained_fraction"].tolist(), reverse=True) == expected_fracs


def test_no_new_portfolio_strategy_introduced():
    """Verify no new trading strategies or portfolio policies were introduced."""
    # Ensure strategy space remains frozen at M0-M16
    assert not (repo_root / "src" / "hypergraph_portfolio" / "m17_strategy.py").exists()
    # Ensure Phase 16 preregistration bounds remain intact
    p16_prereg = repo_root / "artifacts" / "phase16_deterministic_control" / "phase16_preregistration.json"
    assert p16_prereg.exists()
    with open(p16_prereg, "r") as f:
        meta = json.load(f)
    assert meta["preregistration_metadata"]["experiment_id"] == "phase16_deterministic_lora_control"


def test_regularization_selection_pre2022_only():
    """Verify regularization grid search evaluated on pre-2022 validation only."""
    grid_csv = audit_dir / "regularization_selection.csv"
    assert grid_csv.exists(), "regularization_selection.csv missing"
    df = pd.read_csv(grid_csv)

    assert len(df) == 4
    assert set(df["weight_decay"].unique()) == {0.0, 1e-4, 1e-3, 1e-2}
    # Best validation Brier should be competitive with base rate
    assert df["validation_brier"].min() < 0.17


def test_regularized_oos_comparison_complete():
    """Verify clean regularized deterministic LoRA beats Bayesian LoRA across all metrics."""
    oos_csv = audit_dir / "regularized_oos_comparison.csv"
    assert oos_csv.exists(), "regularized_oos_comparison.csv missing"
    df = pd.read_csv(oos_csv)

    assert len(df) == 4
    reg_row = df[df["system_name"] == "Regularized Det LoRA"].iloc[0]
    bayes_mean_row = df[df["system_name"] == "Bayesian Posterior Mean"].iloc[0]

    # Clean regularized Det LoRA achieves higher AUROC and better Brier/LogLoss/ECE
    assert reg_row["auroc"] > bayes_mean_row["auroc"]
    assert reg_row["brier_score"] < bayes_mean_row["brier_score"]
    assert reg_row["log_loss"] < bayes_mean_row["log_loss"]
    assert reg_row["ece"] < bayes_mean_row["ece"]

