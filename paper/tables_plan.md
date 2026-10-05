# Tables Plan: Temporal Regularization in Portfolio Execution

This document specifies the primary LaTeX/Markdown tables for the consolidated manuscript.

---

## Main Paper Core Tables

### Table 1: Primary Comparative Statics Confirmation (Phase 20)
- **Source**: `artifacts/phase20_optimal_cohort_horizon/scientific_report.md`
- **Columns**: Hypothesis, Theoretical Formula, Predicted Direction, Simulation Consistency (%), Statistical Confirmation
- **Rows**:
  - H1: Cost Monotonicity ($\partial H^* / \partial c \ge 0$) — 100.0% — Confirmed
  - H2: Persistence Monotonicity ($\partial H^* / \partial \rho \ge 0$, $\beta > 0$) — 90.0% — Confirmed
  - H3: Short-Lived Alpha Reversal ($H^*=1$ at $\rho=0, c=0$; $\partial H^* / \partial \beta \le 0$) — 100.0% — Confirmed
  - H4: Noise Regularization ($\partial H^* / \partial \sigma_{\text{noise}} \ge 0$) — 100.0% — Confirmed
  - H5: Risk Aversion Preference ($\partial H^* / \partial \gamma \ge 0$) — 100.0% — Confirmed

### Table 2: Exact Null Regime ($\beta = 0.000$) Invariant Audit Summary
- **Source**: `artifacts/phase20_optimal_cohort_horizon/null_regime_invariant_audit.md`
- **Columns**: Evaluated Invariant Quantity, Evaluated Formula, Required Tolerance, Max Absolute Discrepancy, Audit Result
- **Rows**:
  1. Aggregate Portfolio Weights: $\max_t \|w_t^H - w_t^{H=1}\|_\infty$ — $< 10^{-12}$ — 0.00e+00 — PASS
  2. Weekly Gross Return: $\max_t \|r_{\text{gross}, t}^H - r_{\text{gross}, t}^{H=1}\|$ — $< 10^{-12}$ — 0.00e+00 — PASS
  3. Weekly Turnover: $\max_t \|\text{TO}_t^H - \text{TO}_t^{H=1}\|$ — $< 10^{-12}$ — 0.00e+00 — PASS
  4. Weekly Transaction Cost: $\max_t \|\text{Cost}_t^H - \text{Cost}_t^{H=1}\|$ — $< 10^{-12}$ — 0.00e+00 — PASS
  5. Weekly Net Return: $\max_t \|r_{\text{net}, t}^H - r_{\text{net}, t}^{H=1}\|$ — $< 10^{-12}$ — 0.00e+00 — PASS
  6. Mean Weekly Utility: $\|U(H) - U(1)\|$ — $< 10^{-12}$ — 0.00e+00 — PASS
  7. Variance Term: $\|\text{Var}(r_{\text{net}}^H) - \text{Var}(r_{\text{net}}^{H=1})\|$ — $< 10^{-12}$ — 0.00e+00 — PASS

### Table 3: Empirical Parameter Mapping & Sanity Check (Phases 18 & 19)
- **Source**: `artifacts/phase20_optimal_cohort_horizon/empirical_sanity_mapping.json`
- **Columns**: Parameter / Dimension, Empirical Event Strategy Value, Synthetic Grid Mapping, Simulated Optimal Horizon $H^*$, Economic Mechanism
- **Rows**:
  - Transaction Friction ($c$): 10 bps — exact grid match — nearby weak/noisy synthetic cells typically select $H^* \in \{1,2\}$ — higher friction shifts preference toward longer horizons
  - Predictive Signal Quality: Rank IC $\approx 0.05\text{--}0.06$ — descriptive comparison to weak/noisy synthetic cells only; no structural identification of $\beta$ or $\sigma_{\text{noise}}$
  - Information Persistence: historical weeks 2–4 incremental IC $=0.0060$ with $p=0.105$ — no statistically established persistence parameter; any $\rho$ mapping is illustrative
  - Risk Aversion ($\gamma$): 1.0 in the retrospective portfolio accounting — exact utility-parameter match
  - **Overall Empirical Synthesis**: Phase 18/19 motivate the high-churn, weak-signal regime; they do not independently estimate the Phase 20 synthetic frontier.

---

## Appendix Supplementary Tables

- **Table S1: Factorial Simulation Design Manifest**: Full breakdown of parameter grids, seed schedules, and universe sizes.
- **Table S2: Statistical Tie Frequencies across Noise and Cost Slices**: Proportion of cells classified as Dominant Winner vs. 2-way, 3-way, or 5-way ties.
- **Table S3: Clean Retraining Baseline Comparison (Phase 17 Archive)**: Clean Bayesian LoRA vs. deterministic LoRA vs. base-rate null across 2022–2025 holdout regimes.
- **Table S4: Inference Latency and Computational Budget**: Profiling 32-draw Monte Carlo sampling vs. single-pass posterior mean forward pass.
