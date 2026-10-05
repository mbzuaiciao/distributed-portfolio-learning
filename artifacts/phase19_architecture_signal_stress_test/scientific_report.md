# Phase 19: Architecture Signal Stress Test

**Outcome D — Mixed / inconclusive under the frozen outcome tree, with strongly confirmed architecture-general mechanical effects.**

## 1. Executive Summary

Phase 18 established that clean LoRA predictive persistence did not survive rigorous revalidation and failed the economic significance gate, yet H=4 cohort stacking delivered massive turnover and volatility reductions across both model forecasts and simple controls. Phase 19 directly tests the foundational question:

> **Is the H=4 cohort-stacking benefit a model-agnostic portfolio-construction effect that persists across signal quality, including weak or null ranking signals?**

Holding all portfolio machinery, constraints, optimizer parameters, costs, and universe completely fixed, Phase 19 evaluates performance across a comprehensive ladder of signal regimes spanning clean empirical predictions, training-only ticker risk, equal/constant controls, 50 predeclared random ranking null seeds, and a 5-level synthetic signal strength ladder (alpha in [0.00, 0.40]).

### Primary Findings

1. **Mechanical Invariance (H1 Confirmed - PASS)**:
   - Traded turnover reduction under H=4 relative to H=1 exceeds the preregistered 50% threshold across **every single regime and realization**:
     - Clean Deterministic LoRA: **73.97%**
     - Ticker-Risk Baseline: **73.97%**
     - Constant Ranking: **73.97%**
     - Equal Weighting: **73.97%**
     - Random Ranking Null (50 seeds): Mean **73.97%** (min **73.97%**)
     - Synthetic Signal Ladder (alpha in [0.00, 0.40]): Mean **73.97%** (min **73.97%**)

2. **Diversification Invariance (H2 Confirmed - PASS)**:
   - Net volatility reduction under H=4 exceeds the preregistered 30% threshold across **all regimes**:
     - Clean Deterministic LoRA: **66.16%**
     - Ticker-Risk Baseline: **69.71%**
     - Constant Ranking: **65.67%**
     - Equal Weighting: **65.67%**
     - Random Ranking Null (50 seeds): Mean **67.10%** (min **63.35%**)
     - Synthetic Signal Ladder: Mean **67.00%** (min **66.07%**)

3. **Utility Decomposition & Reversal of H3 Direction (H3 Falsified / Reversed)**:
   - Algebraic utility closure holds to machine precision (|error| <= 6.94e-18 < 1e-12) for every single portfolio run:
     delta_U = delta_gross_return + weekly_cost_saving + sample_variance_utility_gain
   - Even when signal quality is completely absent (alpha = 0.00 or random ranking), H=4 achieves **positive net utility delta**:
     - Random Ranking Null: Mean delta_U = +22.531 bp/week, composed of gross return contribution (+8.574 bp/week), weekly cost savings (+10.387 bp/week), and sample variance utility gains (+3.570 bp/week), matching the algebraically closed table exactly.
   - **H3 Directional Reversal**: Rather than H=4 utility shrinking as signal quality approaches zero, H=4 is most economically favorable under weak/null signals and loses relative utility as synthetic signal strength increases. Cohort stacking acts like a **temporal regularizer**:
     - valuable under weak/noisy signals because it lowers churn, cost, and variance;
     - potentially costly under strong, short-lived signals because it slows capital rotation.
     *(The synthetic oracle experiment is strictly a diagnostic simulation, never an empirical trading result.)*

4. **No Alpha Under Null Rankings (H4 Confirmed)**:
   - Sharpe or utility improvements under random or constant ranking are strictly classified as implementation architecture mechanics (diversification and trading-cost amortization), **never predictive alpha**.

---

## 2. Regime Comparison Table (H=1 vs H=4)

| regime | category | horizon | alpha | annualized_return | annualized_volatility | sharpe_ratio | max_drawdown | mean_net_utility | annual_gross_risky_l1_turnover | annual_base_turnover | annual_traded_turnover | annual_tc_drag_bps | mean_effective_holdings | mean_positive_active_names | mean_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_lora | empirical | 1 | nan | 0.092381 | 0.195858 | 0.468670 | -0.187676 | 0.001407 | 103.246377 | 104.000000 | 73.026162 | 730.261623 | 6.259840 | 8.734300 | 0.000000 |
| clean_lora | empirical | 4 | nan | 0.147149 | 0.066179 | 2.223427 | -0.049983 | 0.002788 | 26.879227 | 26.509804 | 19.011677 | 190.116773 | 20.634598 | 34.043478 | 0.745098 |
| constant_ranking | null_control | 1 | nan | -0.000703 | 0.177107 | -0.003972 | -0.206348 | -0.000315 | 103.246377 | 104.000000 | 73.026162 | 730.261623 | 25.043478 | 25.043478 | 0.000000 |
| constant_ranking | null_control | 4 | nan | 0.101964 | 0.060794 | 1.677196 | -0.067307 | 0.001925 | 26.879227 | 26.509804 | 19.011677 | 190.116773 | 51.129959 | 96.768116 | 0.745098 |
| equal_weight | null_control | 1 | nan | -0.000703 | 0.177107 | -0.003972 | -0.206348 | -0.000315 | 103.246377 | 104.000000 | 73.026162 | 730.261623 | 25.043478 | 25.043478 | 0.000000 |
| equal_weight | null_control | 4 | nan | 0.101964 | 0.060794 | 1.677196 | -0.067307 | 0.001925 | 26.879227 | 26.509804 | 19.011677 | 190.116773 | 51.129959 | 96.768116 | 0.745098 |
| random_ranking | null_control | 1 | nan | 0.004663 | 0.203644 | 0.027719 | -0.222997 | -0.000311 | 103.246377 | 104.000000 | 73.026162 | 730.261623 | 6.259840 | 8.734300 | 0.000000 |
| random_ranking | null_control | 4 | nan | 0.103260 | 0.066917 | 1.551386 | -0.072281 | 0.001943 | 26.879227 | 26.509804 | 19.011677 | 190.116773 | 20.623632 | 34.033430 | 0.745098 |
| synthetic_ladder | diagnostic_simulation | 1 | 0.000000 | 0.024471 | 0.232213 | 0.121300 | -0.241052 | -0.000051 | 103.246377 | 104.000000 | 73.026162 | 730.261623 | 4.048476 | 5.961594 | 0.000000 |
| synthetic_ladder | diagnostic_simulation | 1 | 0.050000 | 0.131354 | 0.230918 | 0.584894 | -0.214403 | 0.002010 | 103.246377 | 104.000000 | 73.026162 | 730.261623 | 4.049552 | 5.952174 | 0.000000 |
| synthetic_ladder | diagnostic_simulation | 1 | 0.100000 | 0.238125 | 0.230277 | 1.050732 | -0.191538 | 0.004066 | 103.246377 | 104.000000 | 73.026162 | 730.261623 | 4.052280 | 5.956522 | 0.000000 |
| synthetic_ladder | diagnostic_simulation | 1 | 0.200000 | 0.451108 | 0.230192 | 1.977814 | -0.162319 | 0.008163 | 103.246377 | 104.000000 | 73.026162 | 730.261623 | 4.058417 | 5.976812 | 0.000000 |
| synthetic_ladder | diagnostic_simulation | 1 | 0.400000 | 0.892450 | 0.234669 | 3.822528 | -0.125016 | 0.016630 | 103.246377 | 104.000000 | 73.026162 | 730.261623 | 4.051035 | 5.989855 | 0.000000 |
| synthetic_ladder | diagnostic_simulation | 4 | 0.000000 | 0.106027 | 0.075679 | 1.436746 | -0.090726 | 0.001984 | 26.879227 | 26.509804 | 19.011677 | 190.116773 | 14.226095 | 23.246377 | 0.745098 |
| synthetic_ladder | diagnostic_simulation | 4 | 0.050000 | 0.153039 | 0.075371 | 2.070248 | -0.075339 | 0.002888 | 26.879227 | 26.509804 | 19.011677 | 190.116773 | 14.220487 | 23.205072 | 0.745098 |
| synthetic_ladder | diagnostic_simulation | 4 | 0.100000 | 0.200216 | 0.075313 | 2.701949 | -0.062233 | 0.003795 | 26.879227 | 26.509804 | 19.011677 | 190.116773 | 14.217951 | 23.217391 | 0.745098 |
| synthetic_ladder | diagnostic_simulation | 4 | 0.200000 | 0.296212 | 0.075802 | 3.953464 | -0.041908 | 0.005641 | 26.879227 | 26.509804 | 19.011677 | 190.116773 | 14.217003 | 23.290580 | 0.745098 |
| synthetic_ladder | diagnostic_simulation | 4 | 0.400000 | 0.499195 | 0.079459 | 6.319568 | -0.026045 | 0.009539 | 26.879227 | 26.509804 | 19.011677 | 190.116773 | 14.174621 | 23.349275 | 0.745098 |
| ticker_risk | empirical_control | 1 | nan | 0.079158 | 0.237187 | 0.333735 | -0.226305 | 0.000981 | 103.246377 | 104.000000 | 73.026162 | 730.261623 | 2.394258 | 3.449275 | 0.000000 |
| ticker_risk | empirical_control | 4 | nan | 0.134877 | 0.071839 | 1.877489 | -0.034959 | 0.002544 | 26.879227 | 26.509804 | 19.011677 | 190.116773 | 8.439453 | 13.391304 | 0.745098 |

---

## 3. Algebraic Decomposition Table by Regime

| Regime | Gross Return Delta (bp/wk) | Cost Saving (bp/wk) | Variance Gain (bp/wk) | Net Delta Utility (bp/wk) | Traded TO Red. (%) | Vol Red. (%) |
|---|---|---|---|---|---|---|
| Clean LoRA | 0.145 | 10.387 | 3.272 | 13.804 | 73.97% | 66.16% |
| Ticker Risk | 0.328 | 10.387 | 4.913 | 15.628 | 73.97% | 69.71% |
| Random Ranking (Mean) | 8.574 | 10.387 | 3.570 | 22.531 | 73.97% | 67.10% |
| Equal Weight | 9.356 | 10.387 | 2.661 | 22.404 | 73.97% | 65.67% |
| Constant Ranking | 9.356 | 10.387 | 2.661 | 22.404 | 73.97% | 65.67% |

---

## 4. Synthetic Signal Strength Ladder (Diagnostic Simulation)

| alpha | mean_rank_ic | to_reduction_mean | to_reduction_std | vol_reduction_mean | vol_reduction_std | eff_ratio_mean | delta_gross_mean | cost_saving_mean | var_gain_mean | delta_utility_mean | delta_utility_std | delta_utility_ci_lower | delta_utility_ci_upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | -0.007087 | 0.739659 | 0.000000 | 0.673530 | 0.017595 | 3.514179 | 0.000530 | 0.001039 | 0.000466 | 0.002035 | 0.001816 | -0.001564 | 0.005043 |
| 0.050000 | 0.039050 | 0.739659 | 0.000000 | 0.673045 | 0.017756 | 3.511915 | -0.000622 | 0.001039 | 0.000461 | 0.000878 | 0.001795 | -0.002696 | 0.003927 |
| 0.100000 | 0.085325 | 0.739659 | 0.000000 | 0.672383 | 0.017747 | 3.508926 | -0.001768 | 0.001039 | 0.000458 | -0.000271 | 0.001754 | -0.003792 | 0.002752 |
| 0.200000 | 0.177761 | 0.739659 | 0.000000 | 0.670119 | 0.017354 | 3.503331 | -0.004018 | 0.001039 | 0.000457 | -0.002522 | 0.001675 | -0.005847 | 0.000418 |
| 0.400000 | 0.364197 | 0.739659 | 0.000000 | 0.660692 | 0.017165 | 3.499542 | -0.008601 | 0.001039 | 0.000471 | -0.007091 | 0.001628 | -0.010046 | -0.004237 |

*Note: The synthetic ladder is strictly a controlled simulation to map architecture behavior across signal strength, completely separated from real-world empirical strategy claims.*

---

## 5. Statistical Intervals (Paired Moving Block Bootstrap)

| regime | metric | point_estimate | ci_lower_95 | ci_upper_95 | boot_se |
| --- | --- | --- | --- | --- | --- |
| clean_lora | delta_net_weekly_utility | 0.001380 | -0.005045 | 0.008132 | 0.003389 |
| ticker_risk | delta_net_weekly_utility | 0.001563 | -0.005955 | 0.009169 | 0.003774 |
| constant_ranking | delta_net_weekly_utility | 0.002240 | -0.003089 | 0.008766 | 0.002942 |
| equal_weight | delta_net_weekly_utility | 0.002240 | -0.003089 | 0.008766 | 0.002942 |

---

## 6. Primary Scientific Conclusions

1. **Outcome D: Mechanical Invariance Confirmed, but Frozen Tree Inconclusive due to H3 Reversal**.
   The turnover reduction (~74%) and volatility reduction (~66%) do not depend on predictive model quality or alpha; they are structural invariants arising from holding assets for 4 weekly decision epochs rather than rebalancing 100% of capital every week. However, the preregistered utility-dependence hypothesis H3 was directionally reversed: H=4 produces its highest relative economic utility gains under weak/null signals and yields negative delta utility at high synthetic signal strength where instantaneous 1-week rebalancing capitalizes on short-lived clairvoyant information. The frozen A/B/C/D tree did not anticipate this pattern, yielding formal classification Outcome D.
2. **Cohort Stacking Acts as a Temporal Regularizer**.
   Multi-week cohort stacking operates as a temporal regularizer: highly valuable under noisy, weak, or null signals because it curtails excessive portfolio churn, eliminates trading costs, and diversifies idiosyncratic risk; but potentially costly under strong, short-lived alpha because it delays instantaneous capital reallocation.
3. **No Alpha Under Null Rankings (H4 Confirmed)**:
   Any performance or utility improvements observed under random or constant ranking are strictly classified as portfolio implementation mechanics (diversification and trading-cost amortization), never predictive alpha.
4. **Implications for Quant System Design**:
   Portfolio construction mechanisms such as multi-period overlapping cohort stacking should not be conflated with signal forecasting power. When evaluating Machine Learning alpha, signal decay and portfolio stacking must be audited independently.
