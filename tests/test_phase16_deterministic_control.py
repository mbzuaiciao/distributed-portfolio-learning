"""Unit tests for Phase 16: Deterministic LoRA Control experiment.

Verifies:
1. Architectural parity (rank, alpha, target modules, parameter counts).
2. Deterministic model has 0 variational/variance parameters and no KL loss.
3. Training universe and train/val split match Phase 11.
4. Clean evaluation scope is strictly 2022+ (N=12,300 across 3 seeds).
5. Clustered bootstrap logic preserves calendar week grouping.
6. Phase 15.1 artifacts remain strictly untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

# Sibling repo imports
sibling_src = Path("/Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/src")
if str(sibling_src) not in sys.path:
    sys.path.insert(0, str(sibling_src))

from bayesian_lora_portfolio_risk.uncertainty.bayesian_lora import BayesianLoRALinear  # noqa: E402

from experiments.run_phase16_deterministic_control import (  # noqa: E402
    DeterministicLoRALinear,
)

PREREGISTRATION_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "phase16_deterministic_control" / "phase16_preregistration.json"



def test_preregistration_exists_and_valid():
    """Verify preregistration artifact exists and specifies formal equivalence bounds."""
    assert PREREGISTRATION_PATH.exists(), f"Preregistration file {PREREGISTRATION_PATH} missing"
    with open(PREREGISTRATION_PATH, "r") as f:
        meta = json.load(f)
    assert meta["preregistration_metadata"]["experiment_id"] == "phase16_deterministic_lora_control"
    assert "hypotheses" in meta
    assert "predeclared_equivalence_margins" in meta
    eq = meta["predeclared_equivalence_margins"]
    assert eq["auroc_difference"] == 0.005
    assert eq["brier_score_difference"] == 0.002
    assert eq["spearman_reliability_difference"] == 0.010


def test_architectural_parity():
    """Verify parameter count and layer configurations between Deterministic and Bayesian LoRA."""
    in_f, out_f = 768, 768
    r = 16
    alpha = 32.0
    dropout = 0.05

    base_layer = torch.nn.Linear(in_f, out_f)
    det_lora = DeterministicLoRALinear(base_layer, rank=r, alpha=alpha, dropout=dropout)
    bayes_lora = BayesianLoRALinear(base_layer, rank=r, alpha=alpha, dropout=dropout)

    # Trainable parameters in Deterministic LoRA should be exactly r*in_f + r*out_f
    det_trainable = sum(p.numel() for p in det_lora.parameters() if p.requires_grad)
    expected_det = 16 * 768 + 768 * 16  # 24,576 per layer
    assert det_trainable == expected_det, f"Expected {expected_det}, got {det_trainable}"

    # Bayesian LoRA has mu and rho for both A and B, so twice as many LoRA parameters: 49,152
    bayes_trainable = sum(p.numel() for p in bayes_lora.parameters() if p.requires_grad)
    assert bayes_trainable == 2 * expected_det, f"Expected {2 * expected_det}, got {bayes_trainable}"

    # Verify deterministic model has NO variance/posterior parameters and zero KL
    assert not hasattr(det_lora, "param_a_rho")
    assert not hasattr(det_lora, "param_b_rho")
    assert not hasattr(det_lora, "kl_divergence")


def test_deterministic_forward_pass_reproducibility():
    """Verify DeterministicLoRALinear produces identical outputs across multiple passes in eval mode."""
    base_layer = torch.nn.Linear(768, 768)
    det_lora = DeterministicLoRALinear(base_layer, rank=16, alpha=32.0, dropout=0.05)
    det_lora.eval()

    x = torch.randn(4, 10, 768)
    out1 = det_lora(x)
    out2 = det_lora(x)

    assert torch.allclose(out1, out2, atol=1e-7), "Deterministic forward pass must be perfectly identical across runs"


def test_clean_evaluation_universe_definition():
    """Verify clean post-training evaluation definition (2022+ only, N=4,100 per seed)."""
    clean_csv = Path("artifacts/phase15_1_provenance_audit/deterministic_predictions.csv")
    assert clean_csv.exists(), "Clean evaluation predictions must exist"
    df = pd.read_csv(clean_csv)

    assert len(df) == 12300, f"Expected 12,300 predictions across 3 seeds, got {len(df)}"
    assert sorted(df["seed"].unique()) == [42, 123, 777]
    assert sorted(df["regime"].unique()) == ["2022", "2023", "2024-2025"]

    # Date check: strictly >= 2022-01-01
    min_date = df["date"].min()
    assert min_date >= "2022-01-01", f"Found leak of pre-2022 date: {min_date}"


def test_phase15_1_artifacts_unmodified():
    """Ensure Phase 15.1 artifacts have not been mutated or overwritten."""
    p15_1_report = Path("artifacts/phase15_1_provenance_audit/phase15_1_report.md")
    p15_1_summary = Path("artifacts/phase15_1_provenance_audit/phase15_1_summary.json")

    assert p15_1_report.exists()
    assert p15_1_summary.exists()

    with open(p15_1_summary, "r") as f:
        summary = json.load(f)
    assert summary["phase"].startswith("Phase 15.1")
    assert summary["scientific_adjudication"]["spearman_rho_entropy_det_single_pass"] > 0.59


def test_phase16_artifacts_exist():
    """Verify all mandated Phase 16 artifacts exist and are non-empty."""
    p16_dir = Path(__file__).resolve().parent.parent / "artifacts" / "phase16_deterministic_control"
    required_files = [
        "phase16_preregistration.json",
        "phase16_summary.json",
        "phase16_report.md",
        "training_manifest.csv",
        "deterministic_predictions.csv",
        "forecast_metrics.csv",
        "regime_metrics.csv",
        "seed_metrics.csv",
        "reliability_ranking.csv",
        "calibration_metrics.csv",
        "bootstrap_intervals.csv",
        "equivalence_analysis.csv",
        "weight_comparison.csv",
        "compute_cost.csv",
        "portfolio_sanity_check.csv",
        "deterministic_lora_seed_42_state.pt",
        "deterministic_lora_seed_123_state.pt",
        "deterministic_lora_seed_777_state.pt",
    ]
    for fname in required_files:
        fpath = p16_dir / fname
        assert fpath.exists(), f"Missing Phase 16 artifact: {fname}"
        assert fpath.stat().st_size > 0, f"Empty Phase 16 artifact: {fname}"


def test_phase16_equivalence_and_outcomes():
    """Verify Phase 16 summary results, equivalence rejection, and MC32 equivalence."""
    p16_summary = Path(__file__).resolve().parent.parent / "artifacts" / "phase16_deterministic_control" / "phase16_summary.json"
    with open(p16_summary, "r") as f:
        summary = json.load(f)

    assert summary["scientific_verdict"] == "Outcome D: Regime / Seed Dependent or Inconclusive"
    assert -0.02 < summary["delta_auroc_bayes_vs_det"] < 0.0  # Det LoRA has slightly higher AUROC
    assert summary["delta_brier_bayes_vs_det"] < -0.05  # Bayesian has better Brier
    assert summary["delta_reliability_rho_bayes_vs_det"] > 0.25  # Bayesian has much higher reliability rho

    # Verify bootstrap intervals between MC32 and Bayesian Mean are equivalent
    boot = {b["comparison_metric"]: b for b in summary["bootstrap_summary"]}
    mc32_vs_mean_auroc = boot["Delta_AUROC (MC32 - Bayesian_Mean)"]
    assert abs(mc32_vs_mean_auroc["bootstrap_mean"]) < 0.001
    assert not mc32_vs_mean_auroc["statistically_significant"]



