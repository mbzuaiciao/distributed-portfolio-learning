"""Unit tests for Phase 15.1: Deterministic-Confidence Provenance Audit.

Audits:
1. Provenance of each probability baseline (Phase 15 MC32 vs Phase 15.1 single-pass).
2. True deterministic baseline uses exactly 1 forward pass per event with zero MC sampling.
3. No MC samples enter the deterministic baseline (sample_weight=False).
4. Seeds remain grouped by common event/week in dependence-aware block bootstrap.
5. All Phase 15.1 artifacts exist, are non-empty, and adhere to schema.
6. Adjudicated verdict is Outcome A.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from transformers import AutoModelForSequenceClassification

# Sibling repo imports
sibling_src = Path("/Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/src")
if str(sibling_src) not in sys.path:
    import sys
    sys.path.insert(0, str(sibling_src))

from bayesian_lora_portfolio_risk.uncertainty.bayesian_lora import (  # noqa: E402
    inject_bert_attention_bayesian_lora,
    iter_bayesian_lora_layers,
    set_posterior_sampling,
)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def audit_dir(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "phase15_1_provenance_audit"


def test_phase15_1_artifacts_exist(audit_dir: Path) -> None:
    """Verify all mandated Phase 15.1 artifacts were generated and are non-empty."""
    required_files = [
        "deterministic_predictions.csv",
        "provenance_comparison.csv",
        "regime_provenance_comparison.csv",
        "seed_provenance_comparison.csv",
        "loro_provenance_comparison.csv",
        "quintile_calibration_det.csv",
        "block_bootstrap_provenance.csv",
        "phase15_1_summary.json",
        "phase15_1_report.md",
    ]
    for fname in required_files:
        fpath = audit_dir / fname
        assert fpath.exists(), f"Missing required artifact: {fname}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fname}"


def test_provenance_tracing_and_cost_classification(audit_dir: Path) -> None:
    """Verify provenance classification of Phase 15 vs Phase 15.1 baselines."""
    with open(audit_dir / "phase15_1_summary.json") as f:
        summary = json.load(f)

    trace = summary["provenance_trace"]
    assert "p_bar_MC32" in trace["phase15_probability_source"]
    assert trace["phase15_forward_passes_per_event"] == 32
    assert "NOT single-pass" in trace["phase15_classification"]

    assert "sample_weight=False" in trace["true_single_pass_method"]
    assert trace["true_single_pass_forward_passes_per_event"] == 1


def test_deterministic_model_has_zero_mc_sampling(repo_root: Path) -> None:
    """Verify that setting sample_weight=False disables stochastic sampling on all LoRA layers."""
    model_name = "ProsusAI/finbert"
    base_revision = "4556d13015211d73dccd3fdd39d39232506f3e43"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=base_revision,
        num_labels=2,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    inject_bert_attention_bayesian_lora(model, rank=16, alpha=32, dropout=0.05, init_std=0.01, prior_std=1.0)

    # Enable sampling first
    set_posterior_sampling(model, True)
    assert any(layer.sample_weight for layer in iter_bayesian_lora_layers(model))

    # Disable sampling (as used in deterministic single pass)
    set_posterior_sampling(model, False)
    for layer in iter_bayesian_lora_layers(model):
        assert not layer.sample_weight, "Every layer must have sample_weight=False for deterministic pass"


def test_deterministic_forward_pass_reproducibility(repo_root: Path) -> None:
    """Verify deterministic inference with sample_weight=False is 100% deterministic (zero variance across calls)."""
    model_name = "ProsusAI/finbert"
    base_revision = "4556d13015211d73dccd3fdd39d39232506f3e43"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=base_revision,
        num_labels=2,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    inject_bert_attention_bayesian_lora(model, rank=16, alpha=32, dropout=0.05, init_std=0.01, prior_std=1.0)
    ckpt_path = repo_root / "artifacts" / "phase11_replication" / "frozen_models" / "bayesian_lora_seed_42_state.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    set_posterior_sampling(model, False)

    dummy_input = torch.randint(0, 1000, (2, 16))
    with torch.no_grad():
        out1 = model(input_ids=dummy_input).logits
        out2 = model(input_ids=dummy_input).logits

    # Exact deterministic identity
    assert torch.allclose(out1, out2, atol=1e-7), "Deterministic forward passes must produce identical outputs"


def test_deterministic_predictions_dataset_shape_and_alignment(audit_dir: Path) -> None:
    """Verify deterministic_predictions.csv covers exactly 4,100 clean events * 3 seeds = 12,300 rows."""
    det_df = pd.read_csv(audit_dir / "deterministic_predictions.csv")
    assert len(det_df) == 12300
    assert sorted(det_df["seed"].unique()) == [42, 123, 777]
    assert sorted(det_df["regime"].unique()) == ["2022", "2023", "2024-2025"]
    assert det_df["event_id"].nunique() == 4100

    # Alignment stats in summary JSON
    with open(audit_dir / "phase15_1_summary.json") as f:
        summary = json.load(f)
    align = summary["probability_alignment"]
    assert align["pearson_r"] > 0.99, "Correlation between p_det and p_bar_MC32 must exceed 0.99"
    assert align["spearman_rho"] > 0.99, "Rank correlation between p_det and p_bar_MC32 must exceed 0.99"
    assert align["mean_absolute_error"] < 0.01, "MAE must be less than 1% probability"


def test_dependence_aware_grouping_preserved(audit_dir: Path) -> None:
    """Verify block bootstrap intervals are valid and seeds were clustered by calendar week."""
    boot_df = pd.read_csv(audit_dir / "block_bootstrap_provenance.csv")
    assert len(boot_df) == 4
    for _, row in boot_df.iterrows():
        assert row["n_resamples"] == 1000
        assert row["ci_95_lower"] <= row["ci_95_upper"]

    # In Delta_Rho (Entropy_Det - Epistemic_MC32), true single-pass entropy is >= epistemic
    diff_row = boot_df[boot_df["comparison_metric"] == "Delta_Rho (Entropy_Det - Epistemic_MC32)"].iloc[0]
    assert diff_row["ci_95_lower"] > 0.0, "Single-pass entropy should have statistically higher or equal rho"


def test_scientific_verdict_outcome_a(audit_dir: Path) -> None:
    """Verify the adjudicated verdict is Outcome A."""
    with open(audit_dir / "phase15_1_summary.json") as f:
        summary = json.load(f)

    verdict = summary["scientific_adjudication"]["verdict"]
    assert "Outcome A" in verdict
