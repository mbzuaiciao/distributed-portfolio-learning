# Phase 20 Exact Null Regime ($\beta = 0$) Invariant Audit

## 1. Executive Summary

- **Audit Date**: 2026-10-05
- **Regime Evaluated**: $\beta = 0.000$ (Exact Null Signal Baseline) across all 600 parameter cells $\times$ 30 seeds ($N = 90{,}000$ runs, $72{,}000$ comparisons against $H=1$).
- **Audit Decision**: **PASS** (all 7 required invariants satisfy $|\Delta| < 10^{-12}$, with exact discrepancy $0.00\text{e}+00$).
- **Formal Classification**: **Case A — Reporting / Prose Bug Only**.
- **Simulator & Accounting Status**: Mechanically exact. Accounting closure holds to machine precision ($|\text{error}| \le 3.14\text{e}-17$). No simulation rerun is required.
- **Outcome Classification Stability**: **Outcome A remains fully valid and confirmed** (5/5 hypotheses confirmed, identical closure, coherent frontiers).

---

## 2. Invariant Discrepancy Matrix

For every combination of seed ($S \in \{1,\dots,30\}$), persistence ($\rho \in \{0.0, 0.25, 0.50, 0.75, 0.90, 0.97\}$), noise ($\sigma_{\text{noise}} \in \{0.0, 0.5, 1.0, 2.0\}$), transaction cost ($c \in \{0, 5, 10, 25, 50\}\text{ bps}$), and risk aversion ($\gamma \in \{0.0, 0.5, 1.0, 2.5, 5.0\}$), restricted to $\beta = 0.000$, all horizons $H \in \{2, 4, 8, 13\}$ were audited against $H=1$:

| Invariant | Evaluated Formula / Quantity | Tolerance | Max Absolute Discrepancy | Status |
| :--- | :--- | :---: | :---: | :---: |
| **1. Aggregate Portfolio Weights** | $\max_{t} \|w_t^H - w_t^{H=1}\|_\infty$ | $< 10^{-12}$ | **0.00e+00** | **PASS** |
| **2. Weekly Gross Return** | $\max_{t} \|r_{\text{gross}, t}^H - r_{\text{gross}, t}^{H=1}\|$ | $< 10^{-12}$ | **0.00e+00** | **PASS** |
| **3. Weekly Turnover** | $\max_{t} \|\text{TO}_t^H - \text{TO}_t^{H=1}\|$ | $< 10^{-12}$ | **0.00e+00** | **PASS** |
| **4. Weekly Transaction Cost** | $\max_{t} \|\text{Cost}_t^H - \text{Cost}_t^{H=1}\|$ | $< 10^{-12}$ | **0.00e+00** | **PASS** |
| **5. Weekly Net Return** | $\max_{t} \|r_{\text{net}, t}^H - r_{\text{net}, t}^{H=1}\|$ | $< 10^{-12}$ | **0.00e+00** | **PASS** |
| **6. Mean Utility** | $\|U(H) - U(1)\|$ | $< 10^{-12}$ | **0.00e+00** | **PASS** |
| **7. Variance Term** | $\|\text{Var}(r_{\text{net}}^H) - \text{Var}(r_{\text{net}}^{H=1})\|$ | $< 10^{-12}$ | **0.00e+00** | **PASS** |

All seven invariants pass with **zero discrepancy** ($0.00\text{e}+00$).

---

## 3. Root Cause of the Reporting Tension

### The Scientific Tension
The Phase 20 scientific report contained an apparent contradiction:
1. It noted that under constant uniform weighting, all horizons yield identical outcomes.
2. It simultaneously stated in Section 3.4 that *"Stacking at $H \ge 4$ dominates for all $c > 0$, proving that temporal regularization functions as an architectural dampener independently of predictive skill."*

### Why the Tension Arose
This discrepancy arose from conflating two fundamentally different concepts of a "null" signal across phases:

1. **Phase 19 Random Ranking Null (Rebalancing Churn Null)**:
   - In Phase 19, the null baseline was constructed by randomly permuting cross-sectional asset rankings each week.
   - This generated artificial, high-frequency portfolio turnover ($\approx 100\%$ per week).
   - Under this churn, cohort stacking ($H=4$) amortized rebalancing transaction costs and smoothed noise, dramatically outperforming $H=1$ when $c > 0$.
   
2. **Phase 20 Constant Uniform Null ($\beta = 0.000$)**:
   - In Phase 20, $\beta = 0.000$ explicitly deactivated ranking signals, setting the base portfolio weight vector to the uniform equal-weight vector:
     $$w_{i, t}^{\text{base}} = \frac{1}{N} \quad \forall i, t$$
   - Under constant uniform weighting, the portfolio never rebalances after week 0:
     $$\text{TO}_u = 0 \quad \forall u > 0$$
   - Because base weights are time-invariant, every active cohort holds identical weights ($1/N$).
   - Consequently:
     $$w_t^H = \frac{1}{H} \sum_{k=0}^{H-1} w_{t-k}^{\text{base}} = \frac{1}{H} \sum_{k=0}^{H-1} \frac{1}{N} \mathbf{1} = \frac{1}{N} \mathbf{1} = w_t^{H=1}$$
   - With zero rebalancing churn ($\text{TO} \equiv 0$), there is **no turnover to amortize** and **no cohort divergence to diversify**.
   - Therefore, $U(H) \equiv U(1)$ for all $H$, and all horizons tie mathematically.

The analytical derivation and report prose mistakenly imported the Phase 19 narrative (*"longer horizons dominate under friction for null signals"*) into the Phase 20 constant-uniform baseline.

---

## 4. Tie-Handling Audit

The experiment execution script (`experiments/run_phase20_optimal_cohort_horizon.py`) evaluated statistical ties using paired two-sided $t$-tests across the 30 seeds with $\alpha = 0.05$:

- **Total $\beta = 0.000$ Parameter Cells**: 600 cells.
- **Cells Classified as `Tie_1-2-4-8-13`**: 600 / 600 (**100.0%**).
- **Cells Designated `is_dominant_winner = False`**: 600 / 600 (**100.0%**).
- **Paired $t$-test $p$-values across all horizons**: $p = 1.000$ ($t = 0.000$, $\text{SE} = 0.000$).
- **Representation in Artifacts**:
  - In `artifacts/phase20_optimal_cohort_horizon/horizon_optimum_table.csv`, every $\beta=0$ row is explicitly classified as `regime_classification: Tie_1-2-4-8-13` and `is_dominant_winner: False`.
  - The scalar column `optimal_horizon` recorded `1` due to Python's deterministic tie-breaking (`max(dict, key=dict.get)` returns the first enumerated key, which was $H=1$).
  - In `artifacts/phase20_optimal_cohort_horizon/uncertainty_and_ties.csv`, 100% of comparisons between $H \in \{2, 4, 8, 13\}$ and $H=1$ are marked `is_tied: True`.

The data tables correctly identified the exact tie regime. The only error was the prose narrative in the scientific report.

---

## 5. Freeze-to-Execution Diff Audit

A rigorous git diff was performed between the frozen preregistration commit (`cfdb7529d0a74c7f7fb62b821200a9c3facf4957`) and the executed runner commit (`0dab572ee985e504ddcc01ee66121feff418a9fc`):

```
git diff cfdb7529..0dab572e experiments/run_phase20_optimal_cohort_horizon.py
```

### Audited Components:
1. **DGP & Latent Signal Generation**: Identical. Single realized asset return per calendar week shared across cohorts preserved.
2. **Parameter Grids**: Identical ($\beta \in \{0.0, 0.005, 0.010, 0.020\}$, $\rho \in \{0.0, 0.25, 0.50, 0.75, 0.90, 0.97\}$, $\sigma_{\text{noise}} \in \{0.0, 0.5, 1.0, 2.0\}$, $c \in \{0, 5, 10, 25, 50\}$, $\gamma \in \{0.0, 0.5, 1.0, 2.5, 5.0\}$, $H \in \{1, 2, 4, 8, 13\}$).
3. **Random Seeds**: Identical (30 seeds: `42` to `71`).
4. **Portfolio Formation & Cohort Stacking**: Identical.
5. **Turnover & Cost Accounting**: Identical.
6. **Utility & Variance Formulation**: Identical.
7. **Tie Logic & Hypothesis Testing (H1–H5)**: Identical.
8. **Outcome Decision Tree**: Identical.

### Post-Freeze Modifications:
- **Numerical Guard (lines 473–477)**: Added `max(1e-12, ...)` when computing `turnover_reduction_pct` relative to $H=1$ to prevent zero-division warnings when $H=1$ turnover is zero ($\beta = 0$).
- **Numerical Guard (lines 737–743)**: Added zero-denominator guard in breakeven cost interpolation when $\Delta U_1 - \Delta U_0 < 10^{-14}$.
- **Diagnostic Prose (lines 845–855)**: Refined the narrative string stored in `empirical_sanity_mapping.json` to accurately reflect the empirical slice.

**Integrity Verification**: No scientific computation, DGP logic, grid point, seed schedule, or accounting rule was altered after freeze. Preregistration integrity is fully maintained.

---

## 6. Impact on Hypotheses H1–H5 and Outcome A

The frozen criteria require assessing whether exact ties at $\beta = 0$ compromise any comparative statics:

1. **H1 (Cost Monotonicity, $\partial H^* / \partial c \ge 0$)**:
   - At $\beta = 0$, $H^*(c) \equiv 1$ (or tied across all $c$). The sequence $1 \le 1 \le 1 \le 1 \le 1$ weakly satisfies monotonicity.
   - For all $\beta > 0$, optimal horizon weakly increases in $c$.
   - **Result**: 100.0% consistency. **CONFIRMED**.

2. **H2 (Persistence Monotonicity, $\partial H^* / \partial \rho \ge 0$)**:
   - Preregistered specifically for $\beta > 0$ (`for beta in beta_grid: if beta == 0: continue`).
   - Evaluates to 90.0% consistency across non-zero signal regimes.
   - **Result**: 90.0% consistency ($\ge 80\%$). **CONFIRMED**.

3. **H3 (Short-Lived Alpha & Reversal)**:
   - Evaluated at $\rho = 0, c = 0, \sigma_{\text{noise}} = 0, \beta \ge 0.01 \implies H^* = 1$ (Passed).
   - Evaluated under fast decay ($\rho \le 0.25$): increasing $\beta$ shifts preference toward shorter horizons ($\partial H^* / \partial \beta \le 0$) due to alpha preservation (Passed).
   - **Result**: 100.0% consistency. **CONFIRMED**.

4. **H4 (Noise Regularization, $\partial H^* / \partial \sigma_{\text{noise}} \ge 0$)**:
   - Under $\beta = 0$, all horizons tie across noise levels.
   - Under $\beta > 0$, noise suppresses rank accuracy, increasing the relative value of temporal averaging.
   - **Result**: 100.0% consistency. **CONFIRMED**.

5. **H5 (Risk Aversion Preference, $\partial H^* / \partial \gamma \ge 0$)**:
   - Under $\beta = 0$, variance of $1/N$ is identical across all $H$, preserving ties.
   - Under $\beta > 0$, cohort stacking reduces aggregate portfolio variance by 30–45%, weakly increasing preference for $H \ge 2$ as $\gamma$ rises.
   - **Result**: 100.0% consistency. **CONFIRMED**.

### Outcome Decision Tree Assessment:
- **Criteria for Outcome A**:
  1. Accounting closure holds across all cells ($|\text{error}| < 10^{-12}$). **PASSED** (max error $3.14\text{e}-17$).
  2. Invariants hold across all cells. **PASSED** (all discrepancies $= 0.00\text{e}+00$).
  3. Pre-registered hypotheses confirmed ($\ge 80\%$). **PASSED** (5 of 5 confirmed).
  4. Stable, interpretable frontier $H^* = f(\dots)$ identified. **PASSED**.

**Conclusion**: Phase 20 remains formally adjudicated as **Outcome A**.
