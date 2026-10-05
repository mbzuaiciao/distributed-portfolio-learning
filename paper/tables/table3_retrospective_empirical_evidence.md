# Table 3: Retrospective Empirical Evidence and Stress Tests (Phases 18 & 19)

### Panel A: Clean Empirical Revalidation (Phase 18, Corporate Earnings Holdout 2022–2025)
| Performance / Diagnostic Metric | Single Horizon ($H=1$) | Cohort Stacking ($H=4$) | Difference / Impact | Status / Audit Note |
| :--- | :---: | :---: | :---: | :--- |
| **Annualized Net Return** | 9.24% | 14.71% | +5.47% | Arithmetic annual return |
| **Annualized Net Volatility** | 19.59% | 6.62% | **-66.21%** | Substantial variance reduction |
| **Traded Risky Turnover (L1/yr)** | 73.03 | 19.01 | **-73.97%** | Proportional cost amortization |
| **Annualized Sharpe Ratio** | **0.469** | **2.223** | **+1.754** | Clean revalidation value |
| **Maximum Drawdown** | -18.77% | -5.00% | +13.77% | Downside risk mitigation |
| **Mean Net Weekly Utility** | 14.072 bp/wk | 27.876 bp/wk | **+13.804 bp/wk** | 95% CI: $[-54.68, +84.33]$ bp/wk |
| **Gross Return Contribution** | — | — | +0.145 bp/wk | Gross alpha contribution is near zero |
| **Transaction Cost Savings** | — | — | +10.387 bp/wk | Primary driver of point utility delta |
| **Variance Term Utility Gain** | — | — | +3.272 bp/wk | Secondary driver of point utility delta |
| **Residual Weeks 2–4 Rank IC** | — | — | **0.0060** | $p = 0.105$ (Statistically Insignificant) |

### Panel B: Architecture-General Signal Stress Test (Phase 19)
| Evaluated Signal Regime | Traded Turnover Red. (%) | Net Volatility Red. (%) | Net $\Delta U$ ($H=4$ vs $H=1$) | Outcome Classification |
| :--- | :---: | :---: | :---: | :--- |
| **Clean Empirical LoRA** | **73.97%** | **66.16%** | +13.80 bp/week | Architecture-general confirmation |
| **Ticker-Risk Baseline** | **73.97%** | **69.71%** | +15.63 bp/week | Cross-cohort noise smoothing |
| **Random Ranking Null (50 seeds)**| **73.97%** | **67.10%** | **+22.53 bp/week** | Churn amortization without alpha |
| **Constant / Equal-Weight Null** | **73.97%** | **65.67%** | **+22.40 bp/week** | Churn amortization without alpha |
| **Synthetic $\alpha = 0.00$** | **73.97%** | **67.35%** | +2.04 bp/week | Stacking dominates under noise |
| **Synthetic $\alpha = 0.05$** | **73.97%** | **67.30%** | +0.88 bp/week | Modest stacking benefit |
| **Synthetic $\alpha = 0.10$** | **73.97%** | **67.24%** | **-0.27 bp/week** | **Reversal Point (H3 Reversed)** |
| **Synthetic $\alpha = 0.20$** | **73.97%** | **67.01%** | -2.52 bp/week | Instantaneous rebalancing wins |
| **Synthetic $\alpha = 0.40$** | **73.97%** | **66.07%** | -7.09 bp/week | Instantaneous rebalancing wins |
| **Phase 19 Final Adjudication** | — | — | — | **Outcome D (Inconclusive / H3 Reversed)** |

*Note: Panel A reflects clean revalidation on uncompromised metadata inputs (Phase 18). Panel B reports the stress test across signal quality (Phase 19). These historical results serve as retrospective empirical motivation and reconciliation, not prospective trading confirmation.*
