"""
Decentralized Shadow Pricing Mechanism
======================================
Implements the endogenous pricing equations from Section 2.4:
    p_i^{j*} = ( v_i^j / (sum_l v_l^j) + 1 ) * z_j

Where:
    - z_j is the primitive / base asset price.
    - v_i^j is the dual multiplier associated with copy of asset j in sub-portfolio i.
    - sum_l v_l^j = (E^T v)_j is the aggregate dual pressure on asset j across all subportfolios.
    - The markup v_i^j / sum_l v_l^j reflects the local sub-portfolio's marginal contribution
      to coupling and risk reduction.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from hypergraph_portfolio.hypergraph import HypergraphPortfolio


@dataclass
class DecentralizedPricingResult:
    """Detailed output of the endogenous pricing mechanism."""

    base_prices: np.ndarray             # Primitive prices z (m,)
    dual_variables: np.ndarray          # Dual variables v* (p,)
    aggregate_dual: np.ndarray          # Sum of duals per asset (E^T v*) (m,)
    markups: np.ndarray                 # Local shadow price markups (p,)
    adjusted_prices: np.ndarray         # Local product prices p* (p,)
    asset_breakdown: Dict[str, Dict]    # Hierarchical breakdown per asset and subportfolio


def compute_decentralized_prices(
    v_star: np.ndarray,
    z_base: np.ndarray,
    hypergraph: HypergraphPortfolio,
    eps: float = 1e-12,
) -> DecentralizedPricingResult:
    """
    Compute endogenous decentralized prices for all sub-portfolio asset copies.

    Args:
        v_star: Asymptotic dual variables vector from primal-dual dynamics, shape (p,).
        z_base: Primitive / external asset price vector, shape (m,).
        hypergraph: HypergraphPortfolio instance.
        eps: Small threshold to prevent division by zero when dual sum is zero.

    Returns:
        DecentralizedPricingResult containing markups, adjusted prices, and detailed breakdowns.
    """
    v = np.asarray(v_star, dtype=float)
    z = np.asarray(z_base, dtype=float)

    if len(z) != hypergraph.num_assets:
        raise ValueError(f"Base price vector length {len(z)} != num assets {hypergraph.num_assets}")
    if len(v) != hypergraph.total_nodes:
        raise ValueError(f"Dual variable vector length {len(v)} != total nodes {hypergraph.total_nodes}")

    # Aggregate dual pressure per asset: E^T v
    agg_dual = hypergraph.E.T @ v  # Shape (m,)

    markups = np.zeros(hypergraph.total_nodes, dtype=float)
    adj_prices = np.zeros(hypergraph.total_nodes, dtype=float)
    asset_breakdown: Dict[str, Dict] = {}

    for asset_idx, asset_name in enumerate(hypergraph.asset_names):
        z_j = float(z[asset_idx])
        sum_v_j = float(agg_dual[asset_idx])

        asset_breakdown[asset_name] = {
            "base_price": z_j,
            "aggregate_dual": sum_v_j,
            "subportfolio_prices": {},
        }

        # Find all rows corresponding to this asset
        for row, (sp_idx, local_idx, a_idx) in enumerate(hypergraph.node_meta):
            if a_idx == asset_idx:
                v_local = float(v[row])
                if sum_v_j > eps:
                    rel_markup = v_local / sum_v_j
                else:
                    rel_markup = 0.0

                price_local = (rel_markup + 1.0) * z_j
                markups[row] = rel_markup
                adj_prices[row] = price_local

                asset_breakdown[asset_name]["subportfolio_prices"][sp_idx] = {
                    "local_var_idx": local_idx,
                    "dual_v": v_local,
                    "markup_ratio": rel_markup,
                    "adjusted_price": price_local,
                    "price_multiplier": rel_markup + 1.0,
                }

    return DecentralizedPricingResult(
        base_prices=z,
        dual_variables=v,
        aggregate_dual=agg_dual,
        markups=markups,
        adjusted_prices=adj_prices,
        asset_breakdown=asset_breakdown,
    )


def compute_expected_decentralized_prices(
    v_star: np.ndarray,
    expected_z: np.ndarray,
    hypergraph: HypergraphPortfolio,
) -> DecentralizedPricingResult:
    """
    Stochastic pricing extension (Equation 6 in paper):
        E[p*] = ( inv(E^T v*) * v* + 1 ) * E[z]
    Allows integrating CAPM or APT expected asset prices E[z].
    """
    return compute_decentralized_prices(v_star=v_star, z_base=expected_z, hypergraph=hypergraph)
