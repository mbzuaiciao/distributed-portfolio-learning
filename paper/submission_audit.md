# Phase 20 Submission-Readiness Scientific Audit

This audit systematically verifies every quantitative statement, sample size, equation, statistical test, and architectural claim in `paper/draft.md` and supporting documents against canonical clean artifacts (`artifacts/phase20_optimal_cohort_horizon/`, `artifacts/phase19_architecture_signal_stress_test/`, and `artifacts/phase18_clean_cohort_revalidation/`).

---

## 1. Phase 20 Synthetic Simulation Verification

| Component / Metric | Stated in Draft | Canonical Artifact | Verified Value | Status | Audit Note |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Total Factorial Evaluations** | 360,000 | `experiment_matrix.csv` (rows) | **360,000** | **VERIFIED** | $4 \times 6 \times 4 \times 5 \times 5 \times 5 \times 30 = 360{,}000$. |
| **Random Seed Schedule** | 30 seeds ($S \in \{42, \dots, 71\}$) | `simulation_design.json` | **30 seeds** | **VERIFIED** | Seeds 42 through 71 inclusive. |
| **Asset Universe Size ($N$)** | 30 assets | `simulation_design.json` | **30** | **VERIFIED** | Synthetic cross-section. |
| **Calendar Evaluation Weeks** | 52 weeks (after 52w burn-in) | `simulation_design.json` | **52** | **VERIFIED** | 52 evaluation weeks following 52-week initialization. |
| **Candidate Horizons Grid ($H$)** | $H \in \{1, 2, 4, 8, 13\}$ | `simulation_design.json` | **5 levels** | **VERIFIED** | Frozen discrete candidate set. |
| **Signal Magnitude Grid ($\beta$)** | $\{0.000, 0.005, 0.010, 0.020\}$ | `simulation_design.json` | **4 levels** | **VERIFIED** | Deactivates at 0.000 (uniform null). |
| **Signal Persistence Grid ($\rho$)**| $\{0.0, 0.25, 0.50, 0.75, 0.90, 0.97\}$ | `simulation_design.json` | **6 levels** | **VERIFIED** | Half-lives $\tau_{1/2} \in [0.0, 22.8]\text{ weeks}$. |
| **Observation Noise Grid ($\sigma_{\text{noise}}$)**| $\{0.0, 0.5, 1.0, 2.0\}$ | `simulation_design.json` | **4 levels** | **VERIFIED** | Additive ranking noise. |
| **Transaction Cost Grid ($c$)** | $\{0, 5, 10, 25, 50\}\text{ bps}$ | `simulation_design.json` | **5 levels** | **VERIFIED** | Proportional linear friction. |
| **Risk Aversion Grid ($\gamma$)** | $\{0.0, 0.5, 1.0, 2.5, 5.0\}$ | `simulation_design.json` | **5 levels** | **VERIFIED** | Mean-variance objective weight. |
| **H1 Cost Monotonicity** | 100.0% consistency | `comparative_statics_table.csv:L2` | **100.0%** | **VERIFIED** | $\partial H^* / \partial c \ge 0$. |
| **H2 Persistence Monotonicity** | 90.0% consistency | `comparative_statics_table.csv:L3` | **90.0%** | **VERIFIED** | $\partial H^* / \partial \rho \ge 0$ for $\beta > 0$. |
| **H3 Short-Lived Alpha Reversal**| 100.0% consistency | `comparative_statics_table.csv:L4` | **100.0%** | **VERIFIED** | $H^*=1$ at $\rho=0, c=0$; $\partial H^*/\partial\beta \le 0$. |
| **H4 Noise Regularization** | 100.0% consistency | `comparative_statics_table.csv:L5` | **100.0%** | **VERIFIED** | $\partial H^* / \partial \sigma_{\text{noise}} \ge 0$. |
| **H5 Risk Aversion Preference** | 100.0% consistency | `comparative_statics_table.csv:L6` | **100.0%** | **VERIFIED** | $\partial H^* / \partial \gamma \ge 0$. |
| **Accounting Closure Error** | $\max \|\text{error}\| \le 3.14\times 10^{-17}$ | `scientific_report.md:L7` | **3.144e-17** | **VERIFIED** | Across all 288,000 comparisons vs $H=1$. |
| **Exact Null Invariant Discrepancy** | $0.00\text{e}+00$ across 7 invariants | `null_regime_invariant_audit.json` | **0.00e+00** | **VERIFIED** | Exact mathematical tie across 72,000 cells. |
| **Finite Truncation at $\rho=0.97$** | $0.97^{52} \approx 0.203$ | $(0.97)^{52} = 0.2032$ | **0.2032** | **VERIFIED** | Finite-memory approximation disclosed. |
| **Market Shock $\sigma_{\text{mkt}}$** | 0.02 (2% weekly) | `experiments/run_phase20_*.py:L79` | **0.02** | **VERIFIED** | Shared market factor shock. |
| **Idiosyncratic Shock $\sigma_{\text{idio}}$** | 0.04 (4% weekly) | `experiments/run_phase20_*.py:L80` | **0.04** | **VERIFIED** | Asset-specific calendar return shock. |

---

## 2. Retrospective Empirical Reconciliation Verification (Phases 18 & 19)

| Metric / Parameter | Stated in Draft | Canonical Artifact | Verified Value | Status | Audit Note |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Phase 18 $H=1$ Net Return** | 9.24% | `clean_h1_vs_h4.csv:L2` | **9.24%** | **VERIFIED** | Annualized arithmetic return. |
| **Phase 18 $H=4$ Net Return** | 14.71% | `clean_h1_vs_h4.csv:L3` | **14.71%** | **VERIFIED** | Annualized arithmetic return. |
| **Phase 18 $H=1$ Net Volatility** | 19.59% | `clean_h1_vs_h4.csv:L2` | **19.59%** | **VERIFIED** | Annualized volatility. |
| **Phase 18 $H=4$ Net Volatility** | 6.62% | `clean_h1_vs_h4.csv:L3` | **6.62%** | **VERIFIED** | Annualized volatility. |
| **Phase 18 Net Volatility Reduction**| 66.2% (or ~66%) | `scientific_report.md:L55` | **66.21%** | **VERIFIED** | $1 - 0.066179 / 0.195858 = 66.21\%$. |
| **Phase 18 $H=1$ Traded Turnover** | 73.03 | `clean_h1_vs_h4.csv:L2` | **73.026** | **VERIFIED** | Traded risky L1 turnover/year. |
| **Phase 18 $H=4$ Traded Turnover** | 19.01 | `clean_h1_vs_h4.csv:L3` | **19.012** | **VERIFIED** | Traded risky L1 turnover/year. |
| **Phase 18 Turnover Reduction** | 73.97% (or ~74%) | `scientific_report.md:L57` | **73.97%** | **VERIFIED** | $1 - 19.011677 / 73.026162 = 73.97\%$. |
| **Phase 18 $H=1$ Sharpe** | 0.469 | `clean_h1_vs_h4.csv:L2` | **0.469** | **VERIFIED** | Clean revalidation value (was misstated as 1.508). |
| **Phase 18 $H=4$ Sharpe** | 2.223 | `clean_h1_vs_h4.csv:L3` | **2.223** | **VERIFIED** | Clean revalidation value (was misstated as 2.715). |
| **Phase 18 Incremental W2–4 Rank IC**| 0.0060 | `clean_signal_decay.csv:L7` | **0.006027** | **VERIFIED** | Residual predictive alpha is near zero. |
| **Phase 18 Permutation $p$-value** | 0.105 | `clean_signal_decay.csv:L7` | **0.1050** | **VERIFIED** | Fails statistical significance gate ($p > 0.05$). |
| **Phase 18 Paired $\Delta U$ 95% CI** | [-54.7, +84.3] bp/wk | `scientific_report.md:L34` | **[-54.68, +84.33]** | **VERIFIED** | Paired moving-block bootstrap spans zero. |
| **Phase 18 Gross Return $\Delta$** | 0.145 bp/wk | `decomposition.csv` | **0.145 bp/wk** | **VERIFIED** | Gross alpha gain is negligible. |
| **Phase 18 Cost Saving** | 10.387 bp/wk | `decomposition.csv` | **10.387 bp/wk** | **VERIFIED** | Main driver of point utility delta. |
| **Phase 18 Variance Utility Gain** | 3.272 bp/wk | `decomposition.csv` | **3.272 bp/wk** | **VERIFIED** | Secondary driver of point utility delta. |
| **Phase 19 Turnover Invariance** | $\ge 73.97\%$ across all | `scientific_report.md:L17` | **$\ge 73.97\%$** | **VERIFIED** | Across empirical, null, and synthetic signals. |
| **Phase 19 Volatility Invariance** | 63.3% to 69.7% | `scientific_report.md:L25` | **63.35%–69.71%** | **VERIFIED** | Across all evaluated signal regimes. |
| **Phase 19 Random Null $\Delta U$** | +22.5 bp/wk | `scientific_report.md:L37` | **+22.531 bp/wk** | **VERIFIED** | Rebalancing churn creates regularizer benefit. |
| **Phase 19 Reversal at $\alpha=0.10$** | -0.27 bp/wk | `synthetic_signal_ladder.csv`| **-0.271 bp/wk** | **VERIFIED** | Stacking loses advantage under strong alpha. |
| **Phase 19 Outcome Classification** | Outcome D | `outcome_classification.json`| **Outcome D** | **VERIFIED** | Architecture confirmed; H3 reversed. |

---

## 3. Rhetorical & Terminological Guardrails Audit

| Category / Rule | Requirement | Audited Manuscript Status | Verdict |
| :--- | :--- | :--- | :---: |
| **Synthetic vs Empirical Separation** | Use "empirical" strictly for historical market studies; use "simulation" or "synthetic" for Phase 20. | Table 1 updated to "Simulation Consistency"; Abstract and Section 5 explicitly state controlled synthetic DGP. | **PASS** |
| **"Optimal Horizon" Qualification** | Define optimal horizon as utility-maximizing within discrete candidate set $H \in \{1, 2, 4, 8, 13\}$ under specified model. | Explicit definition added in Introduction and Section 3. | **PASS** |
| **Causal Claims** | Avoid "noise causes longer horizons" or "proves live performance". | Softened to "Within the controlled model, increasing signal noise shifts the utility-maximizing horizon upward." | **PASS** |
| **Empirical Sanity Mapping** | Do not overclaim $H^* \ge 4$ at 10 bps across all seeds in synthetic space. | Accurately explains $H^* \in \{1, 2\}$ at 10 bps under continuous dynamics, shifting to $H=4$ under the extreme panel churn of disjoint earnings baskets. | **PASS** |
| **Limitations Disclosure** | Explicitly detail all 10 boundary conditions in the limitations section. | Section 9 expanded to 10 comprehensive items. | **PASS** |
| **Trading Strategy Disclaimer** | Conclude as an architectural design principle, not a deployable trading strategy. | Section 10 framed strictly as design principles for quantitative portfolio construction. | **PASS** |
