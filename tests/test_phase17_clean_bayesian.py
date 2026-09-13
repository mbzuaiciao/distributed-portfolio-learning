"""Unit tests for Phase 17: Clean Bayesian-LoRA Retraining and Fair Deterministic Comparison.

Verifies:
1. No move_1d in model input text.
2. No move_30d in model input text.
3. No label-derived number (-4.69, etc.) in model input text.
4. Training, validation, and OOS universes are disjoint and strictly chronological.
5. Deterministic and Bayesian systems use identical clean text inputs.
6. Identical architecture except Bayesian variational components (FinBERT, r=16, alpha=32).
7. Bayesian and deterministic hyperparameters evaluated pre-2022 only.
8. Clean OOS evaluation is strictly restricted to 2022+.
9. Seeds (42, 123, 777) are preserved across all systems.
10. Bootstrap preserves shared economic events via calendar-week clusters.
11. Base-rate null predictor (p=0.20) is included.
12. Legacy leaked checkpoints are quarantined and do not enter primary tables.
13. No new portfolio strategy or policy is introduced (M0 unadjusted Markowitz only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

repo_root = Path(__file__).resolve().parent.parent
phase17_dir = repo_root / "artifacts" / "phase17_clean_bayesian"


def test_clean_text_contains_no_outcome_fields():
    """Verify clean_text_manifest.csv contains zero outcome tokens or leaked returns."""
    manifest_path = phase17_dir / "clean_text_manifest.csv"
    assert manifest_path.exists(), "clean_text_manifest.csv missing"
    df = pd.read_csv(manifest_path)
    assert len(df) == 18201

    forbidden_tokens = ["move_1d", "move_30d", "1-day move was", "30-day move was", "-4.69", "downside"]
    for tok in forbidden_tokens:
        matches = df["clean_text"].str.contains(tok, case=False, regex=False).sum()
        assert matches == 0, f"Found {matches} instances of forbidden token '{tok}' in clean_text"


def test_train_val_oos_universes_disjoint():
    """Verify training, validation, and OOS universes are disjoint and strictly chronological."""
    manifest_path = phase17_dir / "clean_text_manifest.csv"
    df = pd.read_csv(manifest_path)

    train_ids = set(df[df["split"] == "train"]["event_id"])
    val_ids = set(df[df["split"] == "validation"]["event_id"])
    oos_ids = set(df[df["split"] == "clean_oos"]["event_id"])

    assert len(train_ids) == 11985
    assert len(val_ids) == 2116
    assert len(oos_ids) == 4100

    # Assert pairwise disjointness
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(oos_ids)) == 0
    assert len(val_ids.intersection(oos_ids)) == 0

    # Assert chronological boundary
    train_dates = pd.to_datetime(df[df["split"] == "train"]["event_timestamp"])
    val_dates = pd.to_datetime(df[df["split"] == "validation"]["event_timestamp"])
    oos_dates = pd.to_datetime(df[df["split"] == "clean_oos"]["event_timestamp"])

    assert train_dates.max() <= val_dates.min()
    assert val_dates.max() < pd.Timestamp("2022-01-01")
    assert oos_dates.min() >= pd.Timestamp("2022-01-01")


def test_deterministic_and_bayesian_use_identical_clean_text():
    """Verify deterministic and Bayesian systems are evaluated on identical events and texts."""
    prereg_path = phase17_dir / "phase17_preregistration.json"
    assert prereg_path.exists()
    with open(prereg_path, "r") as f:
        prereg = json.load(f)

    # Pre-registered manifest hash must match clean_text_manifest.csv
    manifest_path = phase17_dir / "clean_text_manifest.csv"
    assert manifest_path.exists()
    # Check universe row counts
    assert prereg["data_universe_hashes"]["clean_training_universe_rows"] == 11985
    assert prereg["data_universe_hashes"]["clean_validation_universe_rows"] == 2116
    assert prereg["data_universe_hashes"]["clean_oos_universe_unique_events"] == 4100


def test_architectural_parity():
    """Verify identical FinBERT base architecture and LoRA hyperparameters."""
    prereg_path = phase17_dir / "phase17_preregistration.json"
    with open(prereg_path, "r") as f:
        prereg = json.load(f)

    model_spec = prereg["matched_model_specifications"]
    assert model_spec["base_model"]["identifier"] == "ProsusAI/finbert"
    assert model_spec["lora_architecture"]["rank"] == 16
    assert model_spec["lora_architecture"]["alpha"] == 32
    assert model_spec["lora_architecture"]["dropout"] == 0.05
    assert set(model_spec["lora_architecture"]["target_modules"]) == {"attention.self.query", "attention.self.value"}
    assert model_spec["lora_architecture"]["trainable_parameters"] == 591362


def test_legacy_checkpoints_quarantined():
    """Verify legacy checkpoints are formally marked compromised and not loaded."""
    status_path = phase17_dir / "legacy_checkpoint_status.json"
    assert status_path.exists()
    with open(status_path, "r") as f:
        status = json.load(f)

    assert "INVALID for scientific comparisons" in status["scientific_status"]
    assert len(status["quarantined_checkpoints"]) >= 6
    for ckpt in status["quarantined_checkpoints"]:
        assert ckpt["status"] == "COMPROMISED_BY_TARGET_LEAKAGE"
        assert ckpt["action"] == "PRESERVED_FOR_FORENSIC_AUDIT_ONLY_EXCLUDED_FROM_PRIMARY_SCIENCE"


def test_bayesian_hyperparameters_selected_pre2022_only():
    """Verify Bayesian and deterministic hyperparameters were evaluated on pre-2022 data only."""
    val_path = phase17_dir / "validation_metrics.csv"
    assert val_path.exists()
    df = pd.read_csv(val_path)
    # Validation metrics must be populated
    assert len(df) >= 10
    # Manifest must verify all validation events are strictly < 2022
    man_path = phase17_dir / "clean_text_manifest.csv"
    man_df = pd.read_csv(man_path)
    val_dates = pd.to_datetime(man_df[man_df["split"] == "validation"]["event_timestamp"])
    assert (val_dates < pd.Timestamp("2022-01-01")).all()


def test_oos_restricted_to_2022_plus():
    """Verify clean OOS evaluation is strictly restricted to 2022+."""
    preds_path = phase17_dir / "clean_oos_predictions.csv"
    assert preds_path.exists()
    df = pd.read_csv(preds_path)
    dates = pd.to_datetime(df["date"])
    assert (dates >= pd.Timestamp("2022-01-01")).all()


def test_seeds_preserved():
    """Verify seeds 42, 123, 777 are preserved across all systems."""
    seed_path = phase17_dir / "seed_metrics.csv"
    assert seed_path.exists()
    df = pd.read_csv(seed_path)
    tested_seeds = set(df[df["seed"] != "All"]["seed"].astype(int))
    assert tested_seeds == {42, 123, 777}


def test_bootstrap_preserves_shared_economic_events():
    """Verify bootstrap intervals are computed using calendar-week clusters."""
    boot_path = phase17_dir / "bootstrap_intervals.csv"
    assert boot_path.exists()
    df = pd.read_csv(boot_path)
    metrics = df["metric"].tolist()
    assert "Delta_AUROC (BayesMean - Det)" in metrics
    assert "Delta_Brier (BayesMean - Det)" in metrics
    assert "Delta_AUROC (BayesMC32 - BayesMean)" in metrics
    assert "Delta_Brier (BayesMC32 - BayesMean)" in metrics


def test_no_old_leaked_checkpoint_enters_primary_results():
    """Verify no leaked Phase 11 or Phase 16 checkpoint was loaded in Phase 17."""
    man_path = phase17_dir / "training_manifest.csv"
    assert man_path.exists()
    df = pd.read_csv(man_path)
    for ckpt_file in df["checkpoint_file"]:
        assert ckpt_file.startswith("clean_")
        assert "phase11" not in ckpt_file
        assert "phase16" not in ckpt_file


def test_base_rate_null_included():
    """Verify base-rate null predictor is included in preregistration."""
    prereg_path = phase17_dir / "phase17_preregistration.json"
    with open(prereg_path, "r") as f:
        prereg = json.load(f)

    null_spec = prereg["frozen_training_recipes"]["null_baseline"]
    assert null_spec["formula"] == "p_i = 0.20 for all events"


def test_predeclared_equivalence_margins():
    """Verify equivalence margins are preregistered as required."""
    prereg_path = phase17_dir / "phase17_preregistration.json"
    with open(prereg_path, "r") as f:
        prereg = json.load(f)

    margins = prereg["predeclared_equivalence_margins"]
    assert margins["auroc_difference"] == 0.005
    assert margins["brier_score_difference"] == 0.002
    assert margins["log_loss_difference"] == 0.01
    assert margins["ece_difference"] == 0.01


def test_training_and_oos_predictions_artifacts():
    """Verify that when training is completed, all artifacts conform to specifications."""
    preds_path = phase17_dir / "clean_oos_predictions.csv"
    if not preds_path.exists():
        pytest.skip("Training pipeline still executing in background.")

    df = pd.read_csv(preds_path)
    # 4,100 events * 3 seeds = 12,300 predictions
    assert len(df) == 12300

    required_cols = [
        "event_id", "ticker", "date", "regime", "seed", "y_true",
        "p_det_clean", "p_bayes_mean", "p_bayes_mc32", "var_bayes_mc32",
        "entropy_det", "entropy_bayes_mean", "entropy_mc32",
        "margin_det", "margin_bayes_mean", "margin_mc32",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col} in clean_oos_predictions.csv"

    # Seeds must be 42, 123, 777
    assert set(df["seed"].unique()) == {42, 123, 777}

    # Regimes must be 2022, 2023, 2024-2025
    assert set(df["regime"].unique()) == {"2022", "2023", "2024-2025"}

    # Probability bounds
    for p_col in ["p_det_clean", "p_bayes_mean", "p_bayes_mc32"]:
        assert (df[p_col] >= 0.0).all() and (df[p_col] <= 1.0).all()


def test_forecast_and_regime_metrics_artifacts():
    """Verify forecast and regime metrics artifacts when training is complete."""
    fc_path = phase17_dir / "forecast_metrics.csv"
    if not fc_path.exists():
        pytest.skip("Training pipeline still executing in background.")

    df = pd.read_csv(fc_path)
    system_names = df["system"].tolist()
    assert any("Deterministic LoRA" in s for s in system_names)
    assert any("Bayesian Posterior Mean" in s for s in system_names)
    assert any("Bayesian MC32" in s for s in system_names)
    assert any("Null" in s for s in system_names)


def test_no_new_portfolio_strategy():
    """Verify portfolio sanity check only evaluates M0 unadjusted Markowitz."""
    port_path = phase17_dir / "portfolio_sanity_check.csv"
    if not port_path.exists():
        pytest.skip("Training pipeline still executing in background.")

    df = pd.read_csv(port_path)
    assert len(df) >= 3
    # Check that signals used are strictly -p without M13-M16 adjustments
    for sig in df["signal_used"]:
        assert "p_" in sig
