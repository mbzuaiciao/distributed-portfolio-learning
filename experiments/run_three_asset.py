"""
Experiment 07: Three-Asset Reproduction
=======================================
Reproduces the numerical results from Section 3 of Papastaikoudis et al. (2025):
    - 3.1: Centralized case
    - 3.2: Specific partitions case (3 sub-portfolios)
    - 3.3: Generalized powerset case (5 sub-portfolios)

Saves convergence plots to experiments/figures/three_asset_convergence.png
and outputs comparative tables against reported numbers.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from hypergraph_portfolio.hypergraph import HypergraphPortfolio
from hypergraph_portfolio.mean_variance import MeanVariancePortfolio
from hypergraph_portfolio.pricing import compute_decentralized_prices
from hypergraph_portfolio.primal_dual import PrimalDualSolver


def run_centralized():
    print("=" * 70)
    print("SECTION 3.1: CENTRALIZED CASE (3 ASSETS)")
    print("=" * 70)

    Q = np.array([
        [4.1294, 1.4527, 0.9273],
        [1.4527, 4.4150, 1.1068],
        [0.9273, 1.1068, 3.3435],
    ])
    mu = np.array([0.3, 0.25, 0.1])
    c = 1.0

    model = MeanVariancePortfolio(Q=Q, mu=mu, c=c, budget=1.0)
    x_opt, f_opt = model.solve()

    print(f"Optimal Asset Allocation x*: {np.round(x_opt, 4)}")
    print(f"Optimal Objective Value f(x*): {f_opt:.4f}")
    print("Paper Reported:")
    print("  x* = [0.3217, 0.2567, 0.4217]")
    print("  f(x*) = 2.8408")
    print(f"Difference in x*: {np.max(np.abs(x_opt - np.array([0.3217, 0.2567, 0.4217]))):.6f}")
    print(f"Difference in f(x*): {abs(f_opt - 2.8408):.6f}\n")

    return x_opt, f_opt, Q, mu


def run_specific_partitions(Q, mu):
    print("=" * 70)
    print("SECTION 3.2: SPECIFIC PARTITIONS CASE (3 SUB-PORTFOLIOS)")
    print("=" * 70)

    assets = ["x1", "x2", "x3"]
    subportfolios = [
        ["x1", "x2", "x3"],  # Sigma_1
        ["x1", "x2"],        # Sigma_2
        ["x1", "x3"],        # Sigma_3
    ]
    hp = HypergraphPortfolio(asset_names=assets, subportfolios=subportfolios)

    sub_Q = [
        Q,
        Q[:2, :2],
        Q[[0, 2]][:, [0, 2]],
    ]
    sub_mu = [
        mu,
        mu[:2],
        mu[[0, 2]],
    ]
    sub_c = [1.0, 0.6, 0.4]
    sub_a = [0.33, 0.33, 0.33]

    print("Incidence Matrix E:")
    print(hp.E.astype(int))
    print("\nLaplacian Matrix L:")
    print(np.round(hp.L, 4))

    # Solver without gradient weighting matches the paper's raw continuous-time simulation
    solver = PrimalDualSolver(
        hypergraph=hp,
        sub_Q=sub_Q,
        sub_mu=sub_mu,
        sub_c=sub_c,
        sub_a=sub_a,
        budget=1.0,
        scale_gradient_by_weights=False,  # Matches Section 2.3 equation (4a)
    )

    x_opt, lam_opt, v_opt, history = solver.solve(
        dt=0.002,
        max_steps=30000,
        record_every=50,
    )

    tot_f, sub_f = solver.total_objective(x_opt)
    z_opt = hp.aggregate_to_assets(x_opt)

    print("\nSimulation Results:")
    print(f"Primal x* (7 local copies): {np.round(x_opt, 4)}")
    print(f"Aggregated Investment z* = E^T x*: {np.round(z_opt, 4)}")
    print(f"Sub-portfolio Objectives: f1={sub_f[0]:.4f}, f2={sub_f[1]:.4f}, f3={sub_f[2]:.4f}")
    print(f"Distributed Objective f(x*): {tot_f:.4f}")
    print(f"Dual Multipliers v*: {np.round(v_opt, 4)}")

    print("\nPaper Reported:")
    print("  x* = [0.1515, 0.1136, 0.1587, 0.1516, 0.1138, 0.1520, 0.1590]")
    print("  z* = [0.4545, 0.2272, 0.3173]")
    print("  f1 = 1.2813, f2 = 0.7281, f3 = 0.5627")
    print("  f(x*) = 0.8573")
    print("  v* = [0.3249, 0.3484, 0.2486, 0.0322, 0, 0, 0]")

    # Decentralized Pricing
    z_base = np.array([100.0, 50.0, 80.0])
    pricing = compute_decentralized_prices(v_opt, z_base, hp)
    print("\nDecentralized Product Pricing Multipliers (over base price z_j):")
    for a_name, data in pricing.asset_breakdown.items():
        multipliers = [f"SP{sp}: {info['price_multiplier']:.2f}*z" for sp, info in data["subportfolio_prices"].items()]
        print(f"  {a_name}: {', '.join(multipliers)}")

    return hp, solver, history, x_opt, v_opt


def run_powerset_case(Q, mu):
    print("=" * 70)
    print("SECTION 3.3: GENERALIZED POWERSET CASE (5 SUB-PORTFOLIOS)")
    print("=" * 70)

    assets = ["x1", "x2", "x3"]
    # Section 3.3:
    # x0 = {x1, x2, x3} with diagonal covariance (unconnected base assets)
    # x1 = {x1, x2, x3}
    # x2 = {x1, x2}
    # x3 = {x1, x3}
    # x4 = {x2, x3}
    subportfolios = [
        ["x1", "x2", "x3"],  # x0
        ["x1", "x2", "x3"],  # x1
        ["x1", "x2"],        # x2
        ["x1", "x3"],        # x3
        ["x2", "x3"],        # x4
    ]
    hp = HypergraphPortfolio(asset_names=assets, subportfolios=subportfolios)

    Q0 = np.diag(np.diag(Q))  # Diagonal variance matrix
    sub_Q = [
        Q0,
        Q,
        Q[:2, :2],
        Q[[0, 2]][:, [0, 2]],
        Q[1:, 1:],
    ]
    sub_mu = [
        mu,
        mu,
        mu[:2],
        mu[[0, 2]],
        mu[1:],
    ]
    sub_c = [0.7, 1.0, 0.6, 0.4, 0.9]
    sub_a = [0.2, 0.2, 0.2, 0.2, 0.2]

    solver = PrimalDualSolver(
        hypergraph=hp,
        sub_Q=sub_Q,
        sub_mu=sub_mu,
        sub_c=sub_c,
        sub_a=sub_a,
        budget=1.0,
        scale_gradient_by_weights=False,
    )

    x_opt, lam_opt, v_opt, history = solver.solve(
        dt=0.002,
        max_steps=30000,
        record_every=50,
    )

    tot_f, sub_f = solver.total_objective(x_opt)
    z_opt = hp.aggregate_to_assets(x_opt)

    print(f"Primal x* (12 local copies): {np.round(x_opt, 4)}")
    print(f"Aggregated Investment z* = E^T x*: {np.round(z_opt, 4)}")
    print(f"Sub-objectives: {np.round(sub_f, 4)}")
    print(f"Distributed Objective f(x*): {tot_f:.4f}")
    print(f"Dual Multipliers v*: {np.round(v_opt, 4)}")

    print("\nPaper Reported:")
    print("  x* = [0.0893, 0.0760, 0.0847, ...] (each asset repeated 4 times)")
    print("  z* = [0.3572, 0.3040, 0.3388]")
    print("  f0=0.7282, f1=1.0761, f2=0.6324, f3=0.4357, f4=0.9323")
    print("  f(x*) = 0.7609")
    print("  v* = [0, 0, 0, 0.3769, 0.4455, 0.3334, 0.2199, 0.2581, 0.1561, 0.1652, 0.1124, 0.1407]")

    return hp, solver, history, x_opt, v_opt


def plot_convergence(history_specific, history_powerset):
    os.makedirs("experiments/figures", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Specific Partitions: Primal convergence
    ax = axes[0, 0]
    for i in range(history_specific.x_trajectory.shape[1]):
        ax.plot(history_specific.time_points, history_specific.x_trajectory[:, i], label=f"x_{i+1}")
    ax.set_title("(a) Specific Partitions: Primal Allocation Trajectory")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Allocation Weight")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # (b) Specific Partitions: Dual convergence
    ax = axes[0, 1]
    for i in range(history_specific.v_trajectory.shape[1]):
        if np.max(history_specific.v_trajectory[:, i]) > 1e-4:
            ax.plot(history_specific.time_points, history_specific.v_trajectory[:, i], label=f"v_{i+1}")
    ax.set_title("(b) Specific Partitions: Dual Shadow Prices v*(t)")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Dual Multiplier v")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # (c) Consensus Error & Lyapunov Function
    ax = axes[1, 0]
    ax.semilogy(history_specific.time_points, history_specific.consensus_error, label="Consensus Error ||Lx||")
    ax.semilogy(history_specific.time_points, history_specific.lyapunov_trajectory, label="Lyapunov Function V(t)", linestyle="--")
    ax.set_title("(c) Convergence of Residuals & Stability")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Log Scale Residual")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # (d) Powerset Objective & Sub-objectives
    ax = axes[1, 1]
    ax.plot(history_powerset.time_points, history_powerset.objective_trajectory, label="Total Objective f(x)", color="black", linewidth=2)
    for k in range(history_powerset.sub_objectives_trajectory.shape[1]):
        ax.plot(history_powerset.time_points, history_powerset.sub_objectives_trajectory[:, k], label=f"Sub-obj f_{k}")
    ax.set_title("(d) Powerset Case: Total & Sub-portfolio Objectives")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Objective Value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plot_path = "experiments/figures/three_asset_convergence.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nSaved convergence figures to {plot_path}")


if __name__ == "__main__":
    x_cent, f_cent, Q, mu = run_centralized()
    hp_spec, solver_spec, hist_spec, x_spec, v_spec = run_specific_partitions(Q, mu)
    hp_pwr, solver_pwr, hist_pwr, x_pwr, v_pwr = run_powerset_case(Q, mu)
    plot_convergence(hist_spec, hist_pwr)
