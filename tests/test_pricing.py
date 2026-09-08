"""Unit tests for decentralized endogenous pricing mechanism."""

import numpy as np

from hypergraph_portfolio.hypergraph import HypergraphPortfolio
from hypergraph_portfolio.pricing import (
    compute_decentralized_prices,
)


def test_three_asset_paper_pricing_reproduction():
    """Verify exact match with paper Section 3.2 pricing formula."""
    hp = HypergraphPortfolio(
        asset_names=["x1", "x2", "x3"],
        subportfolios=[["x1", "x2", "x3"], ["x1", "x2"], ["x1", "x3"]],
    )

    # Paper's reported asymptotic dual variables v* (page 12)
    v_star = np.array([0.3249, 0.3484, 0.2486, 0.0322, 0.0, 0.0, 0.0])
    z_base = np.array([100.0, 50.0, 80.0])

    pricing = compute_decentralized_prices(v_star=v_star, z_base=z_base, hypergraph=hp)

    # Asset 1 (x1):
    # SP1: 0.3249 / (0.3249 + 0.0322 + 0.0) + 1 = 0.9098 + 1 = 1.9098 -> ~ 1.91 * z1
    # SP2: 0.0322 / 0.3571 + 1 = 0.09017 + 1 = 1.0902 -> ~ 1.09 * z1
    # SP3: 0.0 / 0.3571 + 1 = 1.0 * z1
    sp_prices_1 = pricing.asset_breakdown["x1"]["subportfolio_prices"]
    assert np.isclose(sp_prices_1[0]["price_multiplier"], 1.91, atol=0.01)
    assert np.isclose(sp_prices_1[1]["price_multiplier"], 1.09, atol=0.01)
    assert np.isclose(sp_prices_1[2]["price_multiplier"], 1.00, atol=0.01)

    # Asset 2 (x2):
    # SP1: 0.3484 / 0.3484 + 1 = 2.0 * z2
    # SP2: 0 / 0.3484 + 1 = 1.0 * z2
    sp_prices_2 = pricing.asset_breakdown["x2"]["subportfolio_prices"]
    assert np.isclose(sp_prices_2[0]["price_multiplier"], 2.00, atol=0.01)
    assert np.isclose(sp_prices_2[1]["price_multiplier"], 1.00, atol=0.01)

    # Asset 3 (x3):
    # SP1: 0.2486 / 0.2486 + 1 = 2.0 * z3
    # SP3: 0 / 0.2486 + 1 = 1.0 * z3
    sp_prices_3 = pricing.asset_breakdown["x3"]["subportfolio_prices"]
    assert np.isclose(sp_prices_3[0]["price_multiplier"], 2.00, atol=0.01)
    assert np.isclose(sp_prices_3[2]["price_multiplier"], 1.00, atol=0.01)


def test_zero_dual_fallback():
    """Verify that if dual variables are all zero, prices equal base asset prices."""
    hp = HypergraphPortfolio(
        asset_names=["A", "B"],
        subportfolios=[["A", "B"], ["A", "B"]],
    )
    v_zero = np.zeros(4)
    z_base = np.array([10.0, 20.0])

    pricing = compute_decentralized_prices(v_star=v_zero, z_base=z_base, hypergraph=hp)
    assert np.allclose(pricing.adjusted_prices, [10.0, 20.0, 10.0, 20.0])
    assert np.allclose(pricing.markups, 0.0)
