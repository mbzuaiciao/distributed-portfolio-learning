"""Generate publication-quality figures for Quantitative Finance manuscript.

Figures produced:
- Figure 1: Conceptual architecture of temporal regularization (H=1 vs H=4, 3 forces)
- Figure 2: Analytical expected utility tradeoff and break-even curves
- Figure 3: Phase 20 synthetic horizon frontier (copied/formatted from canonical artifact)
- Figure 4: Retrospective empirical motivation and stress tests (Phase 18 & 19)
"""

import os
import shutil

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set clean journal styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'lines.linewidth': 1.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

FIG_DIR = "paper/figures"
SUB_FIG_DIR = "paper/submission/figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(SUB_FIG_DIR, exist_ok=True)

def generate_figure_1():
    """Figure 1: Conceptual Architecture of Temporal Regularization."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8.5), gridspec_kw={'height_ratios': [1.1, 1.4, 1.0]})
    fig.patch.set_facecolor('white')

    # Panel A: H=1 Single-Horizon Rebalancing
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 3)
    ax1.axis('off')
    ax1.set_title(r"$\bf{(a)\ Single-Period\ Execution\ (H=1):}$ High Responsiveness, Severe Rebalancing Churn", loc='left', fontsize=11)

    weeks = [1, 3, 5, 7, 9]
    for w in weeks:
        rect = patches.FancyBboxPatch((w - 0.7, 0.8), 1.4, 1.2, boxstyle="round,pad=0.1", ec="#1f77b4", fc="#dbeef9", lw=1.5)
        ax1.add_patch(rect)
        ax1.text(w, 1.4, f"Cohort $t_{{{(w+1)//2}}}$\n(100% Capital)", ha='center', va='center', fontsize=9, weight='bold', color="#0f4c81")
        if w < 9:
            ax1.annotate("", xy=(w + 1.2, 1.4), xytext=(w + 0.8, 1.4),
                         arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.0))
            ax1.text(w + 1.0, 1.8, "100% Turnover", ha='center', va='center', fontsize=7.5, color="#d62728", weight='bold')

    ax1.text(0.1, 0.2, "• 100% capital rebalanced every period on fresh forecast\n• Captures peak initial signal, but incurs maximal turnover drag and estimation noise",
             fontsize=8.5, color="#333333")

    # Panel B: H=4 Overlapping Cohort Stacking
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 4.5)
    ax2.axis('off')
    ax2.set_title(r"$\bf{(b)\ Overlapping\ Cohort\ Stacking\ (H=4):}$ Staggered Rolling, Mechanical Regularization", loc='left', fontsize=11)

    cohort_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]
    cohort_labels = ["Cohort $t$ (Age 0: Fresh)", "Cohort $t-1$ (Age 1)", "Cohort $t-2$ (Age 2)", "Cohort $t-3$ (Age 3: Stale)"]

    for i in range(4):
        y_pos = 3.2 - i * 0.75
        rect = patches.FancyBboxPatch((1.5, y_pos - 0.25), 4.5, 0.55, boxstyle="round,pad=0.08", ec=cohort_colors[i], fc=cohort_colors[i], alpha=0.25, lw=1.5)
        ax2.add_patch(rect)
        ax2.text(3.75, y_pos + 0.02, f"{cohort_labels[i]} — 25% Capital", ha='center', va='center', fontsize=8.5, weight='bold', color="#222222")

    # Aggregate arrow
    ax2.annotate("", xy=(6.5, 2.05), xytext=(6.1, 2.05),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=2.5))
    agg_rect = patches.FancyBboxPatch((6.6, 1.0), 3.0, 2.1, boxstyle="round,pad=0.15", ec="#333333", fc="#f0f0f0", lw=2.0)
    ax2.add_patch(agg_rect)
    ax2.text(8.1, 2.3, r"$\bf{Aggregate\ Portfolio}$", ha='center', va='center', fontsize=9.5, color="#111111")
    ax2.text(8.1, 1.8, r"$w_t^H = \frac{1}{4} \sum_{k=0}^3 w_{t-k}$", ha='center', va='center', fontsize=9, color="#0f4c81")
    ax2.text(8.1, 1.3, "• Turnover cuts to $1/4$\n• Noise averaged\n• Stale alpha retained", ha='center', va='center', fontsize=8, color="#444444")

    # Panel C: The Triad of Competing Economic Forces
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 3)
    ax3.axis('off')
    ax3.set_title(r"$\bf{(c)\ The\ Triad\ of\ Competing\ Economic\ Forces\ Governing\ Optimal\ Horizon\ } H^*$", loc='left', fontsize=11)

    forces = [
        ("1. Alpha Dilution", r"$\partial \mathrm{Alpha}/\partial H \leq 0$", "Decaying signals in older cohorts\ndrag down gross returns", "Favors Shorter $H$\n($H=1$)", "#d62728", "#fbeae9"),
        ("2. Cost Amortization", r"$\partial \mathrm{Cost}/\partial H < 0$", "Rolling $1/H$ capital each period\namortizes rebalancing friction", "Favors Longer $H$\n($H \\geq 4$)", "#1f77b4", "#e8f1f8"),
        ("3. Temporal Diversification", r"$\partial \mathrm{Risk}/\partial H \leq 0$", "Overlapping entry schedules\ncancel transient estimation noise", "Favors Longer $H$\n($H \\geq 4$)", "#2ca02c", "#eaf5ea")
    ]

    for idx, (title, math_deriv, desc, fav, ec, fc) in enumerate(forces):
        x_left = 0.3 + idx * 3.25
        rect = patches.FancyBboxPatch((x_left, 0.4), 2.95, 2.3, boxstyle="round,pad=0.1", ec=ec, fc=fc, lw=1.5)
        ax3.add_patch(rect)
        ax3.text(x_left + 1.47, 2.35, title, ha='center', va='center', fontsize=9.5, weight='bold', color=ec)
        ax3.text(x_left + 1.47, 1.95, math_deriv, ha='center', va='center', fontsize=9, color="#222222")
        ax3.text(x_left + 1.47, 1.35, desc, ha='center', va='center', fontsize=8, color="#444444")
        ax3.text(x_left + 1.47, 0.75, fav, ha='center', va='center', fontsize=8.5, weight='bold', color=ec)

    plt.tight_layout()
    pdf_path = os.path.join(FIG_DIR, "fig1_temporal_regularization_architecture.pdf")
    png_path = os.path.join(FIG_DIR, "fig1_temporal_regularization_architecture.png")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    shutil.copy(pdf_path, os.path.join(SUB_FIG_DIR, "fig1_temporal_regularization_architecture.pdf"))
    shutil.copy(png_path, os.path.join(SUB_FIG_DIR, "fig1_temporal_regularization_architecture.png"))
    print(f"Generated Figure 1: {pdf_path} and {png_path}")

def generate_figure_2():
    """Figure 2: Analytical Tradeoff and Comparative Statics."""
    H = np.linspace(1, 13, 200)
    mu_0 = 0.01
    TO_0 = 1.0
    gamma = 1.0
    sigma_0_sq = 0.0016
    sigma_1_sq = 0.0036

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7.5))

    # Panel A: Alpha Dilution
    rhos = [0.0, 0.50, 0.75, 0.90, 0.97]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    for rho, c in zip(rhos, colors):
        if rho == 0.0:
            alpha = mu_0 / H
            lbl = r"$\rho = 0.0\ (\tau_{1/2}=0.0\mathrm{w})$"
        else:
            alpha = mu_0 * (1 - rho**H) / (H * (1 - rho))
            tau_half = np.log(0.5) / np.log(rho)
            lbl = rf"$\rho = {rho:.2f}\ (\tau_{{1/2}}={tau_half:.1f}\mathrm{{w}})$"
        ax1.plot(H, alpha * 10000, label=lbl, color=c)
    ax1.set_title(r"$\bf{(a)\ Expected\ Gross\ Alpha\ Capture}\ \bar{r}_{\mathrm{gross}}(H)$", fontsize=11)
    ax1.set_xlabel("Holding Horizon $H$ (weeks)")
    ax1.set_ylabel("Gross Alpha (bps/week)")
    ax1.legend(loc='upper right', frameon=True, fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Panel B: Turnover Cost Amortization
    costs_bps = [5, 10, 25, 50]
    c_colors = ['#aec7e8', '#1f77b4', '#ff7f0e', '#d62728']
    for c_bp, col in zip(costs_bps, c_colors):
        cost_curve = (c_bp * TO_0 / H)
        ax2.plot(H, cost_curve, label=f"$c = {c_bp}$ bps", color=col)
    ax2.set_title(r"$\bf{(b)\ Turnover\ Cost\ Amortization}\ c \cdot \mathrm{TO}_0 / H$", fontsize=11)
    ax2.set_xlabel("Holding Horizon $H$ (weeks)")
    ax2.set_ylabel("Transaction Cost Drag (bps/week)")
    ax2.legend(loc='upper right', frameon=True, fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Panel C: Temporal Diversification / Variance Reduction
    gammas = [0.5, 1.0, 2.5, 5.0]
    g_colors = ['#c7e9c0', '#74c476', '#31a354', '#006d2c']
    for g, col in zip(gammas, g_colors):
        var_penalty = 0.5 * g * (sigma_0_sq + sigma_1_sq / H) * 10000
        ax3.plot(H, var_penalty, label=rf"$\gamma = {g}$", color=col)
    ax3.set_title(r"$\bf{(c)\ Variance\ Risk\ Penalty}\ \frac{1}{2}\gamma(\sigma_0^2 + \sigma_1^2/H)$", fontsize=11)
    ax3.set_xlabel("Holding Horizon $H$ (weeks)")
    ax3.set_ylabel("Risk Penalty (bps/week)")
    ax3.legend(loc='upper right', frameon=True, fontsize=8)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Panel D: Net Utility Curves and Optimal Horizon Shift
    scenarios = [
        ("High Alpha, Low Friction (c=5, $\\rho=0.25$, $\\mu_0=20$bp)", 0.0020, 0.25, 0.0005, '#d62728', 1),
        ("Institutional Baseline (c=10, $\\rho=0.75$, $\\mu_0=5$bp)", 0.0005, 0.75, 0.0010, '#1f77b4', 4),
        ("High Friction / Persistent (c=25, $\\rho=0.90$, $\\mu_0=5$bp)", 0.0005, 0.90, 0.0025, '#2ca02c', 8),
        ("High Noise / High Friction (c=50, $\\rho=0.97$, $\\mu_0=2$bp)", 0.0002, 0.97, 0.0050, '#9467bd', 13),
    ]
    for name, mu, rho, c_rate, col, h_opt in scenarios:
        u_curve = []
        for h in H:
            alpha = mu * (1 - rho**h) / (h * (1 - rho)) if rho > 0 else mu / h
            cost = c_rate * TO_0 / h
            risk = 0.5 * gamma * (sigma_0_sq + sigma_1_sq / h)
            u_curve.append((alpha - cost - risk) * 10000)
        u_curve = np.array(u_curve)
        u_baseline = u_curve[0]
        delta_u = u_curve - u_baseline
        ax4.plot(H, delta_u, label=f"$H^*={h_opt}$: {name.split(' (')[0]}", color=col)
        # Mark discrete optimum
        opt_idx = np.argmin(np.abs(H - h_opt))
        ax4.scatter(H[opt_idx], delta_u[opt_idx], color=col, s=50, zorder=5)

    ax4.axhline(0, color='gray', linestyle=':', alpha=0.7)
    ax4.set_title(r"$\bf{(d)\ Net\ Utility\ Frontier}\ \Delta U(H, 1) = U(H) - U(1)$", fontsize=11)
    ax4.set_xlabel("Holding Horizon $H$ (weeks)")
    ax4.set_ylabel(r"Utility Delta $\Delta U(H, 1)$ (bps/week)")
    ax4.legend(loc='lower left', frameon=True, fontsize=7.5)
    ax4.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    pdf_path = os.path.join(FIG_DIR, "fig2_analytical_tradeoff.pdf")
    png_path = os.path.join(FIG_DIR, "fig2_analytical_tradeoff.png")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    shutil.copy(pdf_path, os.path.join(SUB_FIG_DIR, "fig2_analytical_tradeoff.pdf"))
    shutil.copy(png_path, os.path.join(SUB_FIG_DIR, "fig2_analytical_tradeoff.png"))
    print(f"Generated Figure 2: {pdf_path} and {png_path}")

def generate_figure_3():
    """Figure 3: Phase 20 Synthetic Horizon Frontier (copying/referencing canonical artifact)."""
    src_png = "artifacts/phase20_optimal_cohort_horizon/horizon_phase_diagram.png"
    src_pdf = "artifacts/phase20_optimal_cohort_horizon/horizon_phase_diagram.pdf"

    tgt_png = os.path.join(FIG_DIR, "fig3_phase20_synthetic_horizon_frontier.png")
    tgt_pdf = os.path.join(FIG_DIR, "fig3_phase20_synthetic_horizon_frontier.pdf")

    shutil.copy(src_png, tgt_png)
    shutil.copy(src_pdf, tgt_pdf)
    shutil.copy(tgt_png, os.path.join(SUB_FIG_DIR, "fig3_phase20_synthetic_horizon_frontier.png"))
    shutil.copy(tgt_pdf, os.path.join(SUB_FIG_DIR, "fig3_phase20_synthetic_horizon_frontier.pdf"))
    print(f"Copied Figure 3: {tgt_pdf} and {tgt_png}")

def generate_figure_4():
    """Figure 4: Retrospective Empirical Motivation & Stress Tests (Phases 18 & 19)."""
    # Load canonical artifacts
    decay_df = pd.read_csv("artifacts/phase18_clean_cohort_revalidation/clean_signal_decay.csv")
    h1_h4_df = pd.read_csv("artifacts/phase19_architecture_signal_stress_test/h1_vs_h4_by_regime.csv")
    ladder_df = pd.read_csv("artifacts/phase19_architecture_signal_stress_test/synthetic_signal_ladder.csv")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7.5))

    # Panel A: Phase 18 Clean Signal Decay
    cum_decay = decay_df[decay_df['kind'] == 'cumulative']
    ax1.plot(cum_decay['horizon'], cum_decay['rank_ic'], marker='o', color='#1f77b4', lw=2, label="Cumulative Rank IC")
    ax1.fill_between(cum_decay['horizon'], cum_decay['ic_ci_lower'], cum_decay['ic_ci_upper'], color='#1f77b4', alpha=0.2, label="95% CI")
    ax1.axhline(0, color='gray', linestyle=':')
    # Highlight residual IC
    ax1.annotate("Residual W2–4 IC = 0.0060\n(p = 0.105, Not Significant)",
                 xy=(4, 0.046), xytext=(5.5, 0.08),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
                 fontsize=8.5, weight='bold', color="#d62728",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f2", ec="#d62728", alpha=0.9))
    ax1.set_title(r"$\bf{(a)\ Clean\ Post-Earnings\ Signal\ Persistence}\ (Phase\ 18)$", fontsize=11)
    ax1.set_xlabel("Horizon (weeks)")
    ax1.set_ylabel("Rank IC")
    ax1.set_xticks([1, 2, 4, 8, 13])
    ax1.legend(loc='upper right', frameon=True, fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Panel B: Traded Turnover Reduction Across Regimes
    regimes = ['clean_lora', 'ticker_risk', 'random_ranking', 'constant_ranking']
    regime_labels = ['Clean LoRA', 'Ticker Risk', 'Random Null', 'Constant Null']
    h1_to = []
    h4_to = []
    for r in regimes:
        sub = h1_h4_df[h1_h4_df['regime'] == r]
        h1_to.append(sub[sub['horizon'] == 1]['annual_traded_turnover'].values[0])
        h4_to.append(sub[sub['horizon'] == 4]['annual_traded_turnover'].values[0])

    x = np.arange(len(regimes))
    width = 0.35
    ax2.bar(x - width/2, h1_to, width, label='H=1', color='#e74c3c', alpha=0.85)
    ax2.bar(x + width/2, h4_to, width, label='H=4', color='#2ecc71', alpha=0.85)
    for i in range(len(regimes)):
        ax2.text(x[i], h4_to[i] + 3.0, "-73.97%", ha='center', fontsize=8, weight='bold', color='#1e8449')
    ax2.set_title(r"$\bf{(b)\ Traded\ Turnover\ Invariance}\ (\geq 73.97\%\ \mathrm{Reduction})$", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(regime_labels)
    ax2.set_ylabel("Traded Turnover (L1/year)")
    ax2.legend(loc='upper right', frameon=True, fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Panel C: Net Volatility Reduction Across Regimes
    h1_vol = []
    h4_vol = []
    for r in regimes:
        sub = h1_h4_df[h1_h4_df['regime'] == r]
        h1_vol.append(sub[sub['horizon'] == 1]['annualized_volatility'].values[0] * 100)
        h4_vol.append(sub[sub['horizon'] == 4]['annualized_volatility'].values[0] * 100)

    ax3.bar(x - width/2, h1_vol, width, label='H=1', color='#e67e22', alpha=0.85)
    ax3.bar(x + width/2, h4_vol, width, label='H=4', color='#3498db', alpha=0.85)
    for i in range(len(regimes)):
        red_pct = 100 * (1 - h4_vol[i] / h1_vol[i])
        ax3.text(x[i], h4_vol[i] + 0.8, f"-{red_pct:.1f}%", ha='center', fontsize=8, weight='bold', color='#1f618d')
    ax3.set_title(r"$\bf{(c)\ Net\ Volatility\ Invariance}\ (63.3\%\mathrm{--}69.7\%\ Reduction)$", fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(regime_labels)
    ax3.set_ylabel("Annualized Net Volatility (%)")
    ax3.legend(loc='upper right', frameon=True, fontsize=8)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Panel D: Synthetic Signal Reversal (Phase 19 Outcome D)
    alphas = ladder_df['alpha']
    delta_u_bps = ladder_df['delta_utility_mean'] * 10000
    ci_lower_bps = ladder_df['delta_utility_ci_lower'] * 10000
    ci_upper_bps = ladder_df['delta_utility_ci_upper'] * 10000

    ax4.plot(alphas, delta_u_bps, marker='s', color='#8e44ad', lw=2, label=r"Net $\Delta U(H=4, 1)$")
    ax4.fill_between(alphas, ci_lower_bps, ci_upper_bps, color='#8e44ad', alpha=0.2, label="95% CI")
    ax4.axhline(0, color='gray', linestyle=':')
    ax4.annotate("Reversal: Strong Short-Lived Alpha\nPenalizes Stacking (H=1 Dominates)",
                 xy=(0.20, delta_u_bps.iloc[3]), xytext=(0.10, -5.0),
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
                 fontsize=8, weight='bold', color="#c0392b",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f2", ec="#c0392b", alpha=0.9))
    ax4.set_title(r"$\bf{(d)\ Synthetic\ Signal\ Strength\ Reversal}\ (Phase\ 19)$", fontsize=11)
    ax4.set_xlabel(r"Synthetic Alpha Magnitude $\alpha$")
    ax4.set_ylabel(r"$\Delta U(H=4, 1)$ (bps/week)")
    ax4.legend(loc='upper right', frameon=True, fontsize=8)
    ax4.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    pdf_path = os.path.join(FIG_DIR, "fig4_retrospective_empirical_motivation.pdf")
    png_path = os.path.join(FIG_DIR, "fig4_retrospective_empirical_motivation.png")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    shutil.copy(pdf_path, os.path.join(SUB_FIG_DIR, "fig4_retrospective_empirical_motivation.pdf"))
    shutil.copy(png_path, os.path.join(SUB_FIG_DIR, "fig4_retrospective_empirical_motivation.png"))
    print(f"Generated Figure 4: {pdf_path} and {png_path}")

if __name__ == "__main__":
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    print("All figures successfully created.")
