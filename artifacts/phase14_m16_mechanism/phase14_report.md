# Phase 14: Mechanism Decomposition of Uncertainty-Aware Risk Inflation Report

**Document Status**: Final Diagnostic Study  
**Strategy Under Investigation**: Frozen $M16$ ($\lambda_u = 0.10$, strictly unchanged)  
**Baseline Strategy**: Frozen $M0$ (Unadjusted Markowitz, strictly unchanged)  
**Portfolio Backbone**: $H = 4$ weeks, $\gamma = 1.0$, $10$ bps transaction cost, cash return $= 0.0$  
**Evaluation Scope**: Primary clean post-training regimes ($2022$, $2023$, $2024\text{--}2025$; $N=4,100$ events, $504$ week-seed observations) across Seeds $42, 123, 777$. Secondary diagnostic regimes ($2020$, $2021$; $N=2,389$ events, $297$ week-seed observations).  

---

## 1. Executive Summary & Problem Formulation

In Phase 13C, frozen $M16$ (asset-level diagonal covariance risk inflation, $\lambda_u = 0.10$) emerged as the top-performing uncertainty mapping, achieving a positive clean average utility delta ($\overline{\Delta U}_{\text{clean}} = +0.000068$). However, its performance exhibited severe cross-regime instability:
- **2022 (Bear / Rates)**: Hurt ($\Delta U = -0.000208$, $0/3$ positive seeds)
- **2023 (Tech Rally)**: Helped strongly ($\Delta U = +0.000592$, $3/3$ positive seeds)
- **2024–2025 (Extended)**: Hurt ($\Delta U = -0.000181$, $1/3$ positive seeds)

The primary purpose of Phase 14 is to perform a granular, week-level forensic decomposition explaining **under what market and cross-sectional conditions uncertainty-aware risk inflation helps**.

> [!IMPORTANT]
> **Core Empirical Conclusion**:  
> $M16$ operates as an **asymmetric error hedge**. It generates strong positive decision value if and only if epistemic uncertainty is positively correlated with realized forecast error in that specific weekly cross-section ($\rho = +0.3681$, $p < 10^{-16}$).  
> In 2023, epistemic uncertainty was exceptionally well-calibrated to actual model errors (+41.3% stronger correlation) and uncertainty dispersion was +40.6% wider, allowing $M16$ to successfully de-risk deceptive alpha signals. In 2022 and 2024–2025, the uncertainty signal was less informative regarding specific forecast errors, causing variance penalties to misallocate capital away from winning assets.

---

## 2. Regime Comparison & Structural Contrast

The table below contrasts the macroeconomic and portfolio state across all evaluated regimes (aggregated across Seeds 42, 123, 777):

### Cross-Regime Mechanism Summary Table

| Metric | 2020 (Diag) | 2021 (Diag) | **2022 (Clean)** | **2023 (Clean)** | **2024–2025 (Clean)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regime Type** | Diagnostic | Diagnostic | **Primary Clean** | **Primary Clean** | **Primary Clean** |
| **Total Weeks ($T \times \text{seeds}$)** | 150 | 147 | **150** | **147** | **207** |
| **Mean Weekly $\Delta U_{M16}$** | $+0.000002$ | $-0.000086$ | **$-0.000236$** | **$+0.000529$** | **$-0.000144$** |
| **Median Weekly $\Delta U_{M16}$** | $-0.000002$ | $0.000000$ | **$-0.000060$** | **$-0.000001$** | **$0.000000$** |
| **Positive Weeks %** | 41.3% | 47.6% | **40.7%** | **47.6%** | **42.0%** |
| **Mean Uncertainty Level ($\bar{\sigma}_{\text{epi}}$)** | 0.016395 | 0.018115 | **0.014134** | **0.021037** | **0.013384** |
| **Uncertainty Dispersion ($\text{std}(\sigma_{\text{epi}})$)** | 0.011997 | 0.013416 | **0.010948** | **0.014772** | **0.010190** |
| **Uncertainty Coeff of Variation (CV)** | 0.5386 | 0.5748 | **0.5582** | **0.5561** | **0.5622** |
| **Forecast Prob Dispersion ($\text{std}(p)$)** | 0.0270 | 0.0280 | **0.0225** | **0.0296** | **0.0222** |
| **Covariance Effective Rank** | 17.12 | 17.29 | **16.95** | **17.24** | **18.00** |
| **Covariance Condition Number** | 6.97 | 7.07 | **6.93** | **7.02** | **7.26** |
| **$M0$ Herfindahl Concentration ($HHI$)**| 0.2851 | 0.2628 | **0.2793** | **0.2567** | **0.2262** |
| **$\Delta HHI$ ($M16 - M0$)** | +0.0214 | +0.0232 | **+0.0232** | **+0.0239** | **+0.0238** |
| **Capital Shifted by $M16$** | 9.86% | 11.15% | **10.92%** | **11.04%** | **11.49%** |
| **Capital Removed from Top 2 Uncertainty Bins**| 9.55% | 10.72% | **10.45%** | **10.65%** | **11.01%** |
| **Realized Return Std (Post-Hoc)** | 0.0553 | 0.0443 | **0.0575** | **0.0534** | **0.0698** |
| **Realized Forecast Error (Post-Hoc)** | 0.3030 | 0.3494 | **0.2848** | **0.2826** | **0.2737** |
| **Corr(Uncertainty, Error) (Post-Hoc)** | +0.0348 | -0.0059 | **+0.0996** | **+0.1086** | **+0.0604** |
| **Error Reduction Share (Post-Hoc)** | 39.3% | 45.5% | **49.3%** | **50.8%** | **54.9%** |

---

## 3. Structural Breakdown: 2023 vs Non-2023

Comparing 2023 against the other clean regimes ($2022 + 2024\text{--}2025$) reveals the top structural differences:

```
+-------------------------------------------------------------------------------------------------------+
| Structural Metric               | 2023 Mean    | Non-2023 Mean | Absolute Diff | Percentage Diff      |
+-------------------------------------------------------------------------------------------------------+
| Mean Weekly Delta U (M16 vs M0) | +0.000529    | -0.000183     | +0.000712     | +389.8%              |
| Mean Net Return Delta           | +0.000556    | -0.000168     | +0.000724     | +431.1%              |
| Epistemic Uncertainty Level     | 0.021037     | 0.013699      | +0.007338     | +53.6% (Ex-Ante)     |
| Epistemic Uncertainty Dispersion| 0.014772     | 0.010508      | +0.004264     | +40.6% (Ex-Ante)     |
| Forecast Alpha Dispersion       | 0.029556     | 0.022334      | +0.007221     | +32.3% (Ex-Ante)     |
| Corr(Uncertainty, Error)        | +0.108574    | +0.076854     | +0.031721     | +41.3% (Post-Hoc)    |
| Realized Return Volatility      | 0.053392     | 0.064615      | -0.011223     | -17.4% (Post-Hoc)    |
+-------------------------------------------------------------------------------------------------------+
```

### Forensic Observations on 2023
1. **Elevated Model Tension (Ex-Ante)**: In 2023, following the generative AI surge and the Silicon Valley Bank failure, language models encountered highly unusual earnings rhetoric. This produced $+53.6\%$ higher uncertainty levels and $+40.6\%$ higher cross-sectional uncertainty dispersion.
2. **High Signal Calibration (Post-Hoc)**: In 2023, names that received high epistemic uncertainty were substantially more prone to large forecast errors ($\bar{\rho} = +0.1086$ vs $+0.0769$). The risk inflation penalty successfully targeted the forecasts that were actually failing.
3. **Macro Bull Market Context (Post-Hoc)**: 2023 was a sustained equity bull market with lower cross-sectional return volatility (0.0534 vs 0.0646). When $M16$ shifted $+5.78\%$ capital into low-uncertainty assets (Q1), these assets produced positive returns ($+0.00357$), generating $+0.000687$ weekly PnL difference in Q1 alone.

---

## 4. Conditional Tertile Analysis

Partitioning all 504 clean weekly observations into Low, Medium, and High tertiles across candidate state variables reveals strong conditional asymmetries:

| Variable | Variable Type | Bucket | $N$ Weeks | Mean $\Delta U_{M16}$ | Median $\Delta U_{M16}$ | Positive Weeks % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Capital Removed from Top 2 Uncertainty Bins** | EX-ANTE | Low | 168 | -0.000038 | 0.000000 | 32.1% |
| | | Medium | 168 | -0.000260 | -0.000129 | 45.2% |
| | | **High** | **168** | **+0.000373** | **+0.000189** | **52.4%** |
| **$M0$ Concentration ($HHI$)** | EX-ANTE | Low | 168 | -0.000183 | -0.000084 | 42.3% |
| | | Medium | 168 | -0.000039 | 0.000000 | 42.3% |
| | | **High** | **168** | **+0.000297** | **0.000000** | **45.2%** |
| **Total Capital Shifted by $M16$** | EX-ANTE | Low | 168 | -0.000059 | 0.000000 | 32.1% |
| | | Medium | 168 | -0.000159 | -0.000077 | 47.0% |
| | | **High** | **168** | **+0.000293** | **+0.000096** | **50.6%** |
| **Correlation(Uncertainty, Realized Error)** | POST-HOC | Low | 168 | -0.001380 | -0.000610 | 35.7% |
| | | Medium | 168 | -0.000103 | -0.000002 | 37.5% |
| | | **High** | **168** | **+0.001559** | **+0.000111** | **56.6%** |
| **Share of Capital Reduction in High-Error Names** | POST-HOC | Low | 169 | -0.000661 | -0.000001 | 30.2% |
| | | Medium | 167 | -0.000588 | -0.000262 | 33.5% |
| | | **High** | **168** | **+0.001324** | **+0.000790** | **66.1%** |

### Key Tertile Takeaways
1. **$M16$ Fails When Passive**: When $M16$ only shifts a small amount of capital (Low bin of capital removed), it produces negative or near-zero decision value ($-0.000038$). It only generates positive utility ($+0.000373$) when $M0$ would have concentrated significant weight in high-uncertainty names.
2. **The Post-Hoc Error Wedge**: The difference between the Low and High tertiles of `Correlation(Uncertainty, Error)` is massive: **$-0.001380$ vs $+0.001559$ ($\Delta = 0.002939$)**. This confirms that the primary determinant of $M16$ success is whether uncertainty correctly anticipated error.

---

## 5. Association Analysis (Spearman Rank Correlations)

Spearman rank correlations evaluated across all 504 clean weekly observations ($168 \text{ weeks} \times 3 \text{ seeds}$):

| Rank | Variable | Variable Type | Pooled $\rho$ | $p$-value | Seed 42 $\rho$ | Seed 123 $\rho$ | Seed 777 $\rho$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Correlation(Uncertainty, Realized Error)** | POST-HOC | **+0.3681** | $< 10^{-16}$ | +0.0919 | +0.3174 | +0.5934 |
| 2 | **Spearman(Uncertainty, Realized Error)** | POST-HOC | **+0.3408** | $< 10^{-14}$ | +0.1227 | +0.4401 | +0.4433 |
| 3 | **Capital Reduction in High-Error Names** | POST-HOC | **+0.2374** | $< 10^{-7}$ | +0.2086 | +0.3057 | +0.3373 |
| 4 | **Realized Return Tail Spread** | POST-HOC | **-0.1111** | $0.0125$ | -0.0465 | -0.1241 | -0.1591 |
| 5 | **$M0$ Concentration ($HHI$)** | EX-ANTE | **+0.0724** | $0.1043$ | -0.0308 | +0.0375 | +0.1014 |
| 6 | **Covariance Condition Number** | EX-ANTE | **-0.0547** | $0.2201$ | +0.0225 | -0.0492 | -0.1011 |
| 7 | **Covariance Effective Rank** | EX-ANTE | **-0.0547** | $0.2201$ | +0.0225 | -0.0492 | -0.1011 |
| 8 | **Uncertainty Level (Mean $\sigma_{\text{epi}}$)** | EX-ANTE | **+0.0467** | $0.2954$ | +0.1746 | +0.1824 | +0.0374 |
| 9 | **Uncertainty Dispersion (Std $\sigma_{\text{epi}}$)**| EX-ANTE | **+0.0355** | $0.4266$ | +0.1112 | +0.0686 | +0.0903 |
| 10| **Alpha Signal Dispersion** | EX-ANTE | **+0.0325** | $0.4667$ | +0.0176 | +0.0364 | +0.2148 |

---

## 6. Explanatory Regression Model & Leave-One-Regime-Out (LORO)

We fit a standardized linear explanatory model:
$$\Delta U_{M16, t} = \beta_0 + \beta_1 \text{StdUncertainty}_t + \beta_2 \text{ConditionNumber}_t + \beta_3 \text{StdProb}_t + \beta_4 HHI_{M0, t}$$

| Model Specification | Held-Out Regime | $\beta_{\text{StdUncertainty}}$ | $\beta_{\text{CondNumber}}$ | $\beta_{\text{StdProb}}$ | $\beta_{HHI_{M0}}$ | Train $R^2$ | Test $R^2$ | Test Rank $\rho$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full Clean Sample | None | $+0.000033$ | $-0.000081$ | $-0.000061$ | $+0.000115$ | **0.0029** | **0.0029** | +0.0633 |
| LORO Fold 1 | **2022** | $-0.000007$ | $-0.000108$ | $-0.000079$ | $+0.000117$ | 0.0043 | **-0.0122** | +0.0157 |
| LORO Fold 2 | **2023** | $+0.000143$ | $+0.000018$ | $-0.000058$ | $+0.000083$ | 0.0016 | **-0.0743** | +0.0820 |
| LORO Fold 3 | **2024–2025** | $-0.000099$ | $-0.000159$ | $-0.000011$ | $+0.000172$ | 0.0084 | **-0.0155** | +0.0031 |

### Interpretation of the Explanatory Model
- **Ex-Ante Unpredictability**: The explanatory $R^2$ on ex-ante cross-sectional state variables is essentially zero ($R^2 = 0.29\%$). Out-of-sample test $R^2$ across all three leave-one-regime-out folds is strictly negative ($-0.012$ to $-0.074$).
- **Coefficient Instability**: The coefficient for uncertainty dispersion flips signs across regimes (positive when training without 2023, negative when training without 2024–2025), proving that ex-ante uncertainty dispersion alone cannot reliably forecast whether $M16$ will help or hurt in an unseen regime.

---

## 7. Answers to Core Scientific Questions

### Q1: Why does 2023 differ most from 2022 and 2024–2025?
2023 was characterized by **the confluence of three distinct factors**:
1. **High Model Uncertainty & Dispersion**: Unprecedented textual dispersion post-banking crisis and GenAI explosion produced $+53.6\%$ higher uncertainty levels and $+40.6\%$ higher cross-sectional uncertainty dispersion.
2. **Elevated Signal Validity**: The correlation between epistemic uncertainty and realized forecast error was $+41.3\%$ stronger ($\bar{\rho} = 0.1086$ vs $0.0769$), meaning high uncertainty accurately flagged bad forecasts.
3. **Macro Bull Market Tailwinds**: In a trending bull market with lower realized volatility ($0.0534$ vs $0.0646$), capital reallocated into low-uncertainty assets (Q1) generated positive returns ($+0.00357$), whereas in 2022 bear-market conditions, reallocating into low-uncertainty assets did not prevent market-wide losses.

### Q2: Which variables are most associated with $M16$ benefit?
The single most powerful predictor is **`post_hoc_corr_uncertainty_error`** ($\rho = \mathbf{+0.3681}$, $p < 10^{-16}$). When uncertainty correctly identifies errors, $M16$ achieves $\Delta U = +0.001559$. Among ex-ante variables, baseline portfolio concentration **`hhi_m0`** ($\rho = +0.0724$) and **`cap_removed_top2_uncertainty`** ($\Delta U = +0.000373$ in the top tertile) show that $M16$ creates value only when $M0$ would otherwise over-concentrate in high-uncertainty names.

### Q3: Does uncertainty dispersion matter more than uncertainty level?
**No.** Both were elevated simultaneously in 2023 (+53.6% level, +40.6% dispersion). While cross-sectional dispersion is mathematically required for $z_i^+$ to discriminate among names, dispersion without overall signal calibration produces no portfolio benefit (as observed in 2021).

### Q4: Does $M16$ help when high-uncertainty names would otherwise receive meaningful $M0$ weight?
**Yes, decisively.** When $M0$ naturally avoids high-uncertainty assets, $M16$ removes virtually no capital and produces slightly negative utility ($-0.000038$). In contrast, when $M0$ assigns heavy weight to high-uncertainty names, $M16$ intervenes aggressively, generating **$+0.000373$ mean $\Delta U$ ($52.4\%$ win rate)**.

### Q5: Does correlation / effective-rank state matter?
**Minimally.** The covariance effective rank was nearly identical across 2022 ($16.95$), 2023 ($17.24$), and 2024–2025 ($18.00$). The capacity of the covariance matrix to accommodate reallocations was not a limiting factor.

### Q6: Is the mechanism ex-ante observable or only post-hoc?
**The mechanism is predominantly POST-HOC.** Ex-ante variables indicate the *capacity* to reallocate capital (explaining $< 0.3\%$ of utility variance), but the *economic payoff* depends strictly on the realized accuracy of the uncertainty signal.

### Q7: Is there enough evidence to justify a future conditional strategy?
**NO.** There is currently **insufficient evidence** to justify creating a tradable conditional $M16$ policy or regime-switching rule based on cross-sectional uncertainty statistics alone. Any rule calibrated to these variables would overfit the idiosyncratic 2023 tech rally.

### Q8: Strongest supported scientific interpretation
Uncertainty-aware risk inflation ($M16$) is a **second-order error hedge** whose decision value is contingent on macroeconomic regime alignment. It succeeds when model uncertainty is genuinely correlated with downside forecast failure and when the broader equity market provides positive returns to the surviving high-conviction assets.
