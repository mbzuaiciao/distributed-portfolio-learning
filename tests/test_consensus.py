"""Unit tests for consensus constraints and projection operations."""

import numpy as np
import pytest

from hypergraph_portfolio.consensus import (
    check_consensus,
    compute_consensus_discrepancies,
    lift_centralized_to_distributed,
    project_onto_consensus,
)
from hypergraph_portfolio.hypergraph import HypergraphPortfolio


@pytest.fixture
def sample_hypergraph():
    assets = ["A", "B", "C"]
    subportfolios = [
        ["A", "B", "C"],
        ["A", "B"],
        ["A", "C"],
    ]
    return HypergraphPortfolio(asset_names=assets, subportfolios=subportfolios)


def test_consensus_nullspace_and_violations(sample_hypergraph):
    hp = sample_hypergraph

    # Consistent allocation: A=0.15, B=0.10, C=0.20 across all local copies
    # Local copies:
    # SP1: A, B, C -> [0.15, 0.10, 0.20]
    # SP2: A, B    -> [0.15, 0.10]
    # SP3: A, C    -> [0.15, 0.20]
    x_consistent = np.array([0.15, 0.10, 0.20, 0.15, 0.10, 0.15, 0.20])

    is_cons, err = check_consensus(x_consistent, hp)
    assert is_cons
    assert np.isclose(err, 0.0)

    # Inconsistent allocation: A in SP1 is 0.50, in SP2 is 0.10
    x_inconsistent = x_consistent.copy()
    x_inconsistent[0] = 0.50

    is_cons, err = check_consensus(x_inconsistent, hp)
    assert not is_cons
    assert err > 0.1

    # Projecting inconsistent vector yields consistent vector
    x_proj = project_onto_consensus(x_inconsistent, hp)
    is_cons_proj, err_proj = check_consensus(x_proj, hp)
    assert is_cons_proj
    assert np.isclose(err_proj, 0.0)


def test_lift_centralized(sample_hypergraph):
    hp = sample_hypergraph
    z = np.array([0.45, 0.20, 0.35])

    x_lifted = lift_centralized_to_distributed(z, hp)
    # Check consensus
    is_cons, err = check_consensus(x_lifted, hp)
    assert is_cons

    # Check aggregation back to z
    z_rec = hp.aggregate_to_assets(x_lifted)
    assert np.allclose(z_rec, z)


def test_consensus_discrepancy_inspection(sample_hypergraph):
    hp = sample_hypergraph
    x = np.array([0.20, 0.10, 0.30, 0.15, 0.10, 0.25, 0.30])
    disc = compute_consensus_discrepancies(x, hp)

    # Asset A has copies [0.20, 0.15, 0.25] -> max spread = 0.10
    assert np.isclose(disc["A"]["max_discrepancy"], 0.10)
    # Asset B has copies [0.10, 0.10] -> max spread = 0.00
    assert np.isclose(disc["B"]["max_discrepancy"], 0.0)
