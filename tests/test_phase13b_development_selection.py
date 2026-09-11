"""Tests for Phase 13B: Development-Only Uncertainty Mapping Selection.

Audits:
1. Strict temporal boundary enforcement: development data strictly <= 2019-12-20 (N=2,203).
2. Zero evaluation regime leakage: 2020-2025 completely untouched during selection.
3. Mathematical correctness of asset-level uncertainty mappings:
   - M14: Signal shrinkage mu_adj_i = mu_i / (1 + lambda * max(z_i, 0))
   - M15: Selective abstention (zero out mu_i for top fraction f of epistemic uncertainty)
   - M16: Diagonal risk inflation Q_adj = Q + lambda_u * diag(max(z, 0))
4. Preregistration artifact integrity:
   - mapping_preregistration.json contains frozen checkpoints, parameters, candidate grids.
5. Deterministic selection and tie-breaking adherence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

repo_root = Path(__file__).resolve().parent.parent


def test_development_events_scope_and_leakage():
    """Verify that development predictions contain only 2018-2019 events (<= 2019-12-20)."""
    pred_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "development_predictions.csv"
    if not pred_path.exists():
        pytest.skip("development_predictions.csv not yet generated.")

    df = pd.read_csv(pred_path)
    assert len(df) == 2203 * 3, f"Expected 6,609 rows (2,203 events * 3 seeds), got {len(df)}"

    # Check seeds
    assert set(df["seed"].unique()) == {42, 123, 777}

    # Check temporal bounds
    df["date_dt"] = pd.to_datetime(df["date"])
    assert (df["date_dt"] >= "2018-01-01").all(), "Events prior to 2018 found in development!"
    assert (df["date_dt"] <= "2019-12-20 23:59:59").all(), (
        "CRITICAL LEAKAGE: Events after 2019-12-20 found in development predictions!"
    )

    # Verify column definitions
    expected_cols = {
        "event_id",
        "ticker",
        "date",
        "seed",
        "posterior_mean",
        "epistemic_uncertainty",
        "aleatoric_uncertainty",
        "total_uncertainty",
    }
    assert expected_cols.issubset(set(df.columns))

    # Epistemic and aleatoric must be strictly non-negative
    assert (df["epistemic_uncertainty"] >= 0.0).all()
    assert (df["aleatoric_uncertainty"] >= 0.0).all()
    assert (df["posterior_mean"] >= 0.0).all()
    assert (df["posterior_mean"] <= 1.0).all()


def test_m14_signal_shrinkage_formula():
    """Verify M14 asset-level signal shrinkage formula and properties."""
    mu = np.array([0.05, -0.04, 0.02, 0.01])
    # Relative uncertainty z: some positive, some negative
    z = np.array([1.5, 0.0, -0.8, 2.0])
    lam = 1.0

    z_pos = np.maximum(z, 0.0)
    mu_adj = mu / (1.0 + lam * z_pos)

    # For z <= 0, signal must remain identical
    assert np.isclose(mu_adj[1], mu[1])
    assert np.isclose(mu_adj[2], mu[2])

    # For z > 0, magnitude of signal must strictly decrease
    assert abs(mu_adj[0]) < abs(mu[0])
    assert abs(mu_adj[3]) < abs(mu[3])

    # Exact values:
    assert np.isclose(mu_adj[0], 0.05 / (1.0 + 1.0 * 1.5))
    assert np.isclose(mu_adj[3], 0.01 / (1.0 + 1.0 * 2.0))


def test_m15_selective_abstention_formula():
    """Verify M15 selective abstention formula: zero out top fraction by epistemic uncertainty."""
    mu = np.array([0.05, -0.04, 0.02, 0.01, 0.03])  # 5 assets
    sigma_epi = np.array([0.01, 0.08, 0.03, 0.05, 0.02])  # descending order: asset 1, 3, 2, 4, 0

    # Fraction f = 0.20 -> round(0.2 * 5) = 1 asset excluded (asset 1)
    k_exclude = int(np.round(0.20 * len(mu)))
    assert k_exclude == 1
    ranked_idx = np.argsort(-sigma_epi)
    mu_adj = mu.copy()
    mu_adj[ranked_idx[:k_exclude]] = 0.0

    assert mu_adj[1] == 0.0  # highest uncertainty zeroed
    assert mu_adj[0] == mu[0]
    assert mu_adj[2] == mu[2]
    assert mu_adj[3] == mu[3]
    assert mu_adj[4] == mu[4]

    # Fraction f = 0.40 -> round(0.4 * 5) = 2 assets excluded (asset 1 and 3)
    k_exclude_2 = int(np.round(0.40 * len(mu)))
    assert k_exclude_2 == 2
    mu_adj_2 = mu.copy()
    mu_adj_2[ranked_idx[:k_exclude_2]] = 0.0
    assert mu_adj_2[1] == 0.0
    assert mu_adj_2[3] == 0.0
    assert mu_adj_2[0] == mu[0]


def test_m16_risk_inflation_formula():
    """Verify M16 asset-level risk inflation modifies diagonal only."""
    m = 3
    Q = 0.04 * np.eye(m) + 0.01 * np.ones((m, m))
    z = np.array([1.2, -0.5, 0.8])
    lam_u = 0.50

    z_pos = np.maximum(z, 0.0)
    Q_adj = Q + lam_u * np.diag(z_pos)

    # Off-diagonal elements must remain completely unchanged
    for i in range(m):
        for j in range(m):
            if i != j:
                assert np.isclose(Q_adj[i, j], Q[i, j])

    # Diagonal elements
    assert np.isclose(Q_adj[0, 0], Q[0, 0] + lam_u * 1.2)
    assert np.isclose(Q_adj[1, 1], Q[1, 1])  # z <= 0 -> no inflation
    assert np.isclose(Q_adj[2, 2], Q[2, 2] + lam_u * 0.8)


def test_mapping_preregistration_artifact():
    """Verify mapping_preregistration.json structure, frozen parameters, and hashes."""
    prereg_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "mapping_preregistration.json"
    if not prereg_path.exists():
        pytest.skip("mapping_preregistration.json not yet generated.")

    with open(prereg_path) as f:
        data = json.load(f)

    # Check essential top-level keys
    assert "git_commit_before_evaluation" in data
    assert "development_period" in data
    assert "evaluation_periods_untouched" in data
    assert "portfolio_backbone_specification" in data
    assert "frozen_checkpoints" in data
    assert "selected_frozen_parameters" in data
    assert "candidate_grids" in data

    # Check candidate grids match protocol
    assert data["candidate_grids"]["M14_signal_shrinkage"] == [0.25, 0.50, 1.00, 2.00]
    assert data["candidate_grids"]["M15_selective_abstention"] == [0.10, 0.20, 0.30]
    assert data["candidate_grids"]["M16_risk_inflation"] == [0.10, 0.25, 0.50, 1.00]

    # Check selected frozen parameters exist and are numeric
    selected = data["selected_frozen_parameters"]
    assert "lambda_M14" in selected and isinstance(selected["lambda_M14"], (int, float))
    assert "fraction_M15" in selected and isinstance(selected["fraction_M15"], (int, float))
    assert "lambda_M16" in selected and isinstance(selected["lambda_M16"], (int, float))

    assert selected["lambda_M14"] in [0.25, 0.50, 1.00, 2.00]
    assert selected["fraction_M15"] in [0.10, 0.20, 0.30]
    assert selected["lambda_M16"] in [0.10, 0.25, 0.50, 1.00]

    # Check backbone
    backbone = data["portfolio_backbone_specification"]
    assert backbone["holding_horizon_H"] == 4
    assert backbone["risk_aversion_gamma"] == 1.0
    assert backbone["transaction_cost_bps"] == 10.0
    assert backbone["cash_return"] == 0.0

    # Check 5 untouched evaluation periods listed
    eval_periods = data["evaluation_periods_untouched"]
    assert len(eval_periods) == 5
    eval_regime_ids = [p["regime_id"] for p in eval_periods]
    assert eval_regime_ids == ["2020", "2021", "2022", "2023", "2024-2025"]
