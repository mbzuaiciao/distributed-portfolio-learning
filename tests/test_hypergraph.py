"""Unit tests for HypergraphPortfolio and Laplacian matrix construction."""

import numpy as np

from hypergraph_portfolio.hypergraph import (
    HypergraphPortfolio,
    build_powerset_subportfolios,
)


def test_specific_partition_hypergraph_matrices():
    """Verify Section 3.2 hypergraph matrices E and L match the paper exactly."""
    assets = ["x1", "x2", "x3"]
    # Sigma_1 = {x1, x2, x3}, Sigma_2 = {x1, x2}, Sigma_3 = {x1, x3}
    subportfolios = [
        ["x1", "x2", "x3"],
        ["x1", "x2"],
        ["x1", "x3"],
    ]
    hp = HypergraphPortfolio(asset_names=assets, subportfolios=subportfolios)

    assert hp.num_assets == 3
    assert hp.total_nodes == 7
    assert hp.subportfolio_dims == [3, 2, 2]

    # Check E against paper page 11
    expected_E = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=float)
    assert np.allclose(hp.E, expected_E)

    # Edge degrees: asset 1 appears 3 times, asset 2 appears 2 times, asset 3 appears 2 times
    assert np.allclose(hp.edge_degrees, [3.0, 2.0, 2.0])

    # Check L against paper page 11:
    # L_11 = 1 - 1/3 = 2/3, L_22 = 1 - 1/2 = 1/2, etc.
    expected_L = np.array([
        [ 2/3,    0,    0, -1/3,    0, -1/3,    0],
        [   0,  1/2,    0,    0, -1/2,    0,    0],
        [   0,    0,  1/2,    0,    0,    0, -1/2],
        [-1/3,    0,    0,  2/3,    0, -1/3,    0],
        [   0, -1/2,    0,    0,  1/2,    0,    0],
        [-1/3,    0,    0, -1/3,    0,  2/3,    0],
        [   0,    0, -1/2,    0,    0,    0,  1/2],
    ], dtype=float)
    assert np.allclose(hp.L, expected_L)

    # Verify algebraic properties
    props = hp.verify_laplacian_properties()
    assert all(props.values())


def test_powerset_subportfolios_generation():
    """Verify powerset generation for 3 assets."""
    indices = [0, 1, 2]
    # min_size=2: pairs and full set
    subsets = build_powerset_subportfolios(indices, min_size=2, include_full=True)
    assert len(subsets) == 4  # 3 pairs + 1 triplet
    assert [0, 1] in subsets
    assert [0, 2] in subsets
    assert [1, 2] in subsets
    assert [0, 1, 2] in subsets
