"""
Experiment 10: Applied Example — GICS Sectors-Based ETF Allocation
==================================================================
Recreates the 11-sector ETF allocation from Section 5 of Papastaikoudis et al. (2025):
    - 11 GICS sector ETFs: XLK, XLC, XLY, XLP, XLV, XLU, XLF, XLE, XLI, XLRE, XLB
    - 6 thematic subportfolios (Growth, Defensive, Cyclicals, Real Assets, Balanced, Financials)
    - Total variable dimension p = 5 + 4 + 5 + 5 + 7 + 4 = 30
    - Budget scaled to 1000.0 (Section 5.2)
    - Empirical mean returns and covariance matrix from Section 5.2
    - Fixed management costs c_i = kappa_i^T mu_i
    - Primal allocation, dual shadow prices, and decentralized product valuations
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from hypergraph_portfolio.hypergraph import HypergraphPortfolio
from hypergraph_portfolio.mean_variance import MeanVariancePortfolio
from hypergraph_portfolio.pricing import compute_decentralized_prices
from hypergraph_portfolio.primal_dual import PrimalDualSolver


def get_gics_data():
    tickers = ["XLK", "XLC", "XLY", "XLP", "XLV", "XLU", "XLF", "XLE", "XLI", "XLRE", "XLB"]

    # Mean returns from Section 5.2.1
    mu = np.array([
        0.019026, 0.009796, 0.010791, 0.008847, 0.010764,
        0.007266, 0.010661, 0.012848, 0.011831, 0.007280, 0.010612
    ])

    # Sector fixed cost parameters kappa_j (Section 5.2 table)
    kappa = np.array([
        0.0025, 0.0020, 0.0020, 0.0015, 0.0015,
        0.0012, 0.0018, 0.0022, 0.0020, 0.0021, 0.0020
    ])

    # Full 11x11 covariance matrix from Section 5.2.1
    Q = np.array([
        [0.004187, 0.003338, 0.004013, 0.001720, 0.001968, 0.001505, 0.002977, 0.003223, 0.003291, 0.002686, 0.003085],
        [0.003338, 0.003646, 0.003438, 0.001527, 0.001699, 0.001451, 0.003102, 0.003472, 0.003004, 0.002533, 0.002938],
        [0.004013, 0.003438, 0.004902, 0.001706, 0.002086, 0.001666, 0.003378, 0.003811, 0.003632, 0.003238, 0.003624],
        [0.001720, 0.001527, 0.001706, 0.001749, 0.001435, 0.001475, 0.001429, 0.001449, 0.002103, 0.001798, 0.001995],
        [0.001968, 0.001699, 0.002086, 0.001435, 0.002070, 0.001299, 0.002052, 0.002500, 0.002263, 0.001842, 0.002016],
        [0.001505, 0.001451, 0.001666, 0.001475, 0.001299, 0.002357, 0.001653, 0.001955, 0.001836, 0.002041, 0.001959],
        [0.002977, 0.003102, 0.003378, 0.001429, 0.002052, 0.001653, 0.004468, 0.005418, 0.003899, 0.002769, 0.003445],
        [0.003223, 0.003472, 0.003811, 0.001449, 0.002500, 0.001955, 0.005418, 0.011427, 0.004748, 0.003089, 0.004710],
        [0.003291, 0.003004, 0.003632, 0.002103, 0.002263, 0.001836, 0.003899, 0.004748, 0.004228, 0.002868, 0.003778],
        [0.002686, 0.002533, 0.003238, 0.001798, 0.001842, 0.002041, 0.002769, 0.003089, 0.002868, 0.003394, 0.002894],
        [0.003085, 0.002938, 0.003624, 0.001995, 0.002016, 0.001959, 0.003445, 0.004710, 0.003778, 0.002894, 0.003929],
    ])

    subportfolios = [
        ["XLK", "XLC", "XLY", "XLI", "XLB"],                     # SP1: Growth + Innovation
        ["XLP", "XLV", "XLU", "XLI"],                            # SP2: Defensive Core
        ["XLY", "XLV", "XLF", "XLE", "XLI"],                     # SP3: Cyclicals
        ["XLU", "XLE", "XLI", "XLRE", "XLB"],                     # SP4: Real Assets & Inflation
        ["XLK", "XLC", "XLY", "XLP", "XLV", "XLI", "XLRE"],      # SP5: Balanced Growth
        ["XLU", "XLF", "XLI", "XLRE"],                            # SP6: Financials & Real Estate
    ]

    return tickers, mu, kappa, Q, subportfolios


def run_gics_experiment():
    print("=" * 75)
    print("EXPERIMENT 10: 11-SECTOR GICS ETF ALLOCATION & PRICING")
    print("=" * 75)

    tickers, mu, kappa, Q, subportfolios = get_gics_data()
    budget = 1000.0

    # 1. Centralized solution
    c_cent = float(kappa @ mu)
    cent_model = MeanVariancePortfolio(Q=Q, mu=mu, c=c_cent, budget=budget, asset_names=tickers)
    x_cent, f_cent = cent_model.solve()

    print("Centralized Benchmark (Budget = 1000):")
    print(f"Optimal Objective f(x*): {f_cent:.2f} (Paper reported: ~1589.5)")
    print("Centralized Weights:")
    for name, w in zip(tickers, x_cent):
        if w > 1.0:
            print(f"  {name}: {w:.1f}")

    # 2. Decentralized Hypergraph Setup
    hp = HypergraphPortfolio(asset_names=tickers, subportfolios=subportfolios)
    print("\nHypergraph Structure:")
    print(f"- Total Nodes (Local Variables): {hp.total_nodes}")
    print(f"- Sub-portfolio Dimensions: {hp.subportfolio_dims}")
    print("- Asset Degrees (coverage across subportfolios):")
    for name, deg in zip(tickers, hp.edge_degrees):
        print(f"  {name}: {int(deg)} copies")

    # Construct sub-portfolio parameters
    sub_Q = []
    sub_mu = []
    sub_c = []
    for sp_indices in hp.subportfolio_indices:
        idx = np.array(sp_indices)
        sub_Q.append(Q[np.ix_(idx, idx)])
        sub_mu.append(mu[idx])
        sub_c.append(float(kappa[idx] @ mu[idx]))

    sub_a = np.ones(6) / 6.0

    solver = PrimalDualSolver(
        hypergraph=hp,
        sub_Q=sub_Q,
        sub_mu=sub_mu,
        sub_c=sub_c,
        sub_a=sub_a,
        budget=budget,
        scale_gradient_by_weights=False,  # Raw continuous-time dynamics as in Section 5
    )

    x_opt, lam_opt, v_opt, history = solver.solve(
        dt=0.001,
        max_steps=20000,
        record_every=50,
    )

    tot_f, sub_f = solver.total_objective(x_opt)
    z_opt = hp.aggregate_to_assets(x_opt)

    print("\nDecentralized Primal-Dual Simulation Results:")
    print(f"Distributed Objective f(x*): {tot_f:.4f} (Paper reported: 59.5007)")
    print(f"Sub-portfolio Objectives: {np.round(sub_f, 2)}")
    print(f"Total Portfolio Capital Check (sum z): {np.sum(z_opt):.1f}")

    print("\nAggregated Asset Holdings z* = E^T x*:")
    for name, z_val in zip(tickers, z_opt):
        print(f"  {name:<6}: {z_val:.1f}")

    # 3. Decentralized Shadow Pricing
    # Assume base ETF market prices z_base ($)
    base_prices = np.array([200.0, 80.0, 180.0, 75.0, 140.0, 65.0, 40.0, 90.0, 120.0, 40.0, 85.0])
    pricing = compute_decentralized_prices(v_opt, base_prices, hp)

    print("\nSample Endogenous Shadow Price Markups (Selected Assets):")
    for sample_asset in ["XLK", "XLI", "XLF", "XLV"]:
        p_info = pricing.asset_breakdown[sample_asset]
        sp_muls = [f"SP{sp}: {data['price_multiplier']:.2f}x" for sp, data in p_info["subportfolio_prices"].items()]
        print(f"  {sample_asset} (Base: ${p_info['base_price']:.0f}): {', '.join(sp_muls)}")

    # 4. Save Plots
    os.makedirs("experiments/figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Asset weights comparison
    ind = np.arange(len(tickers))
    width = 0.35
    ax1.bar(ind - width/2, x_cent, width, label="Centralized Optimum", color="#2980b9", alpha=0.85)
    ax1.bar(ind + width/2, z_opt, width, label="Distributed Aggregate (z*)", color="#27ae60", alpha=0.85)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(tickers, rotation=45)
    ax1.set_ylabel("Capital Allocation ($)")
    ax1.set_title("Centralized vs Distributed Asset Allocations (Budget = 1000)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Convergence trajectories
    ax2.plot(history.time_points, history.objective_trajectory, color="black", label="Distributed Objective f(x)")
    for i in range(history.sub_objectives_trajectory.shape[1]):
        ax2.plot(history.time_points, history.sub_objectives_trajectory[:, i], label=f"SP{i+1}", alpha=0.7)
    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Objective Value")
    ax2.set_title("GICS Sub-portfolio Risk Convergence")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "experiments/figures/gics_etf_allocations.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nSaved GICS ETF figures to {plot_path}")


if __name__ == "__main__":
    run_gics_experiment()
