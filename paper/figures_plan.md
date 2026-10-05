# Figures Plan: Temporal Regularization in Portfolio Execution

This document specifies the four core figures for the consolidated manuscript and supplementary figures for the appendix.

---

## Main Paper Core Figures

### Figure 1: Conceptual Architecture of Temporal Regularization
- **Purpose**: Illustrate the mechanics of overlapping cohort portfolio formation, showing how staggered sub-portfolio rolling amortizes turnover and averages estimation noise over time.
- **Visual Design**:
  - Panel A: Single-period rebalancing ($H=1$), showing 100% portfolio turnover each week and direct exposure to noisy rank fluctuations.
  - Panel B: Overlapping cohort stacking ($H=4$), showing capital partitioned into four cohorts, with only 1/4 of capital rebalanced each week and aggregate weights formed by averaging active cohorts.
  - Panel C: Flow diagram of the three competing economic forces: Alpha Dilution (favoring $H=1$), Cost Amortization (favoring $H \ge 4$), and Temporal Noise Diversification (favoring $H \ge 4$).

### Figure 2: Analytical Tradeoff & Break-Even Frontiers
- **Source**: `artifacts/phase20_optimal_cohort_horizon/analytical_derivation.md`
- **Purpose**: Demonstrate the analytical tradeoff between alpha capture, turnover costs, and variance penalties as a function of horizon $H$.
- **Panels**:
  - Panel A: Expected Gross Alpha Decay $\bar{r}_{\text{gross}}(H)$ across different persistence levels $\rho \in \{0.0, 0.50, 0.75, 0.90, 0.97\}$.
  - Panel B: Turnover Cost Amortization $c \cdot \text{TO}(H)$ and Portfolio Variance $\sigma^2(H)$ as a function of $H$.
  - Panel C: Net Utility Curves $U(H)$ showing the shift in optimal horizon $H^*$ as transaction costs increase from 0 to 50 bps.
  - Panel D: Closed-form Break-Even Cost Surface $c^*(H; \rho, \beta, \gamma)$.

### Figure 3: Phase 20 Horizon Phase Diagram
- **Source**: `artifacts/phase20_optimal_cohort_horizon/horizon_phase_diagram.png`
- **Purpose**: The primary synthetic result figure of the paper mapping the utility-maximizing horizon $H^*$ across transaction costs $c \in [0, 50]\text{ bps}$ and signal half-life $\tau_{1/2} \in [0, 23]\text{ weeks}$.
- **Panels**:
  - 4-panel grid across observation noise $\sigma_{\text{noise}} \in \{0.0, 1.0\}$ and signal strength $\beta \in \{0.005, 0.020\}$.
  - Color-coded regions for $H^* \in \{1, 2, 4, 8, 13\}$, with hatched or shaded markers indicating statistical tie zones.
  - Clear visual demarcation of the four principal regimes: Short-Horizon Dominance ($H=1$), Intermediate Regularization ($H=2\text{--}4$), Extended Stacking ($H=8\text{--}13$), and Exact-Null Ties.

### Figure 4: Empirical Motivation & Architectural Invariance (Phases 18 & 19 Sanity Check)
- **Source**: `artifacts/phase19_architecture_signal_stress_test/architecture_stress_test_figures.png` & `artifacts/phase18_clean_cohort_revalidation/clean_signal_decay.png`
- **Purpose**: Connect the synthetic frontier to retrospective event-study motivation, illustrating that turnover and volatility savings persisted across several signal regimes without treating the historical data as independent causal confirmation.
- **Panels**:
  - Panel A: Empirical signal decay curve from Phase 18 clean revalidation, showing rapid dissipation of residual Rank IC by week 2 ($p=0.105$).
  - Panel B: Traded turnover reduction ($>73.9\%$) across empirical, random, constant, and synthetic signal regimes.
  - Panel C: Net portfolio volatility reduction ($63.3\%\text{--}69.7\%$) across signal regimes.
  - Panel D: Delta utility reversal: positive $\Delta U$ under weak/null signals, declining to negative $\Delta U$ under strong synthetic alpha.

---

## Appendix Supplementary Figures

- **Figure S1: Exact-Null Baseline Invariant Verification**: Visual confirmation that for $\beta = 0$, differences in aggregate weights, gross returns, turnover, costs, net returns, utility, and variance across all $H$ are identically zero.
- **Figure S2: Break-Even Cost Sensitivity by Risk Aversion**: Detailed heatmaps of breakeven transaction costs across risk aversion levels $\gamma \in \{0.0, 0.5, 1.0, 2.5, 5.0\}$.
- **Figure S3: Historical Forensic Audit of Target Leakage**: Timeline and ROC curves from Phase 16.1 showing artificial validation performance from leaked `move_1d` tokens vs clean out-of-sample collapse.
- **Figure S4: Class-Conditional Entropy-Error Metric Decomposition**: Scatter plots showing mechanical $+1.0000$ and $-1.0000$ within-class correlations between predictive entropy and prediction error.
