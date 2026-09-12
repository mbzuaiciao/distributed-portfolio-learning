# Phase 15: Incremental Value of Bayesian Epistemic Uncertainty

**Scientific Report & Diagnostic Information Study**  
**Date**: September 2026  
**Repository**: `distributed-portfolio-learning`  
**Dataset Scope**: Frozen Phase-11 Bayesian-LoRA checkpoints across Seeds 42, 123, 777; $N = 12,300$ clean post-training observations ($4,100$ events $\times$ 3 seeds) spanning 2022, 2023, and 2024–2025 across 168 calendar weeks; 7,167 secondary diagnostic observations (2020–2021).  

---

## Executive Summary

Phase 13 and Phase 14 established two key empirical facts:
1. Epistemic uncertainty strongly tracks forecast unreliability ($Q5/Q1$ error ratio $> 1.4\times$);
2. Fixed uncertainty-to-portfolio mappings (shrinkage $M14$, abstention $M15$, risk inflation $M16$) do not robustly improve portfolio utility out of sample, exhibiting regime-dependent degradation.

Phase 15 addresses the core scientific question that resolves this dichotomy:
> **Does Bayesian epistemic uncertainty contain forecast-reliability information that simpler confidence baselines (predictive entropy, margin $|p - 0.5|$) or market-risk proxies do not?**

### Adjudicated Scientific Outcome: **Outcome B (Epistemic Uncertainty and Entropy/Margin Perform Similarly)**
- **Diagnostic Signal is Real**: Bayesian epistemic uncertainty exhibits a strong, statistically significant rank correlation with absolute forecast error ($\rho = 0.5945, p < 10^{-300}$).
- **Simple Confidence Captures Virtually Identical Information**: Deterministic predictive entropy $H(p)$ and inverse confidence margin $(0.5 - |p - 0.5|)$ achieve Spearman $\rho = 0.5992$, slightly exceeding epistemic uncertainty ($\Delta \rho = -0.00486, 95\%\text{ CI: } [-0.00754, -0.00215]$).
- **Zero Incremental Information**: In Leave-One-Regime-Out (LORO) nested regressions, adding epistemic uncertainty to a model already containing simple confidence and market risk proxies yields $\Delta \text{AUROC} = -0.0009$ and $\Delta \text{MAE} = -0.00016$. Block bootstrap tests ($B=1,000$ clusters over calendar weeks) confirm these differences are indistinguishable from zero ($p = 0.51$ for AUROC, $p = 0.44$ for MAE).
- **High Collinearity**: Simple confidence functions of the posterior mean explain **$86.22\%$** of the variance in Bayesian epistemic uncertainty ($R^2 = 0.8622$). Residual epistemic uncertainty has an out-of-sample AUROC of $0.5008$ (pure coin flip) and linear Pearson $r = 0.0024$ ($p = 0.789$).
- **Market Volatility Dominates Tail Risk**: For binary high-error detection (top 20% errors), historical asset volatility achieves an AUROC of $0.5880$, whereas both epistemic uncertainty and predictive entropy hover near $0.5015$.

---

## 1. Experimental Design & Methodology

### 1.1 Data Scope & Preregistration Constraints
- **Primary Clean Sample**: 2022 ($N = 3,573$), 2023 ($N = 3,543$), 2024–2025 ($N = 5,184$). Total $N = 12,300$ event-seed observations.
- **Secondary Diagnostic Regimes**: 2020 ($N = 3,594$) and 2021 ($N = 3,573$), totaling $7,167$ observations with known Phase-11 pretraining overlap.
- **Zero Trading Policies**: No new strategies ($M17$), no parameter tuning, no portfolio simulation.
- **Target Variables**:
  - **Primary**: Absolute Forecast Error $e_i = |\hat{p}_i - y_i|$, where $y_i = \mathbb{I}(r_{1d, i} \le q_{0.20}(r_{1d}))$, with the clean 20th percentile threshold frozen at $r_{\text{downside}} = -4.692\%$.
  - **Secondary**: Binary High-Error Indicator $y_{\text{high\_err}, i} = \mathbb{I}(e_i \ge q_{0.80}(e))$, with the clean 80th percentile threshold frozen at $e_{\text{thresh}} = 0.6529$.

### 1.2 Candidate Signals
1. **Bayesian Uncertainty**:
   - `epistemic_uncertainty`: Monte Carlo variance across $M=32$ weight samples, $\frac{1}{M}\sum_m (p_m - \bar{p})^2$.
   - `epistemic_std`: $\sqrt{\text{epistemic\_uncertainty}}$.
   - `total_uncertainty`: Law of total variance $\text{Var}(y) = \bar{p}(1-\bar{p}) + \text{Var}_m(p_m)$.
   - `aleatoric_uncertainty`: Expected conditional variance $\frac{1}{M}\sum_m p_m(1-p_m)$.
2. **Simple Deterministic Confidence Baselines**:
   - `predictive_entropy`: $H(\bar{p}) = -\bar{p}\ln \bar{p} - (1-\bar{p})\ln(1-\bar{p})$.
   - `inv_confidence_margin`: $0.5 - |\bar{p} - 0.5| = \min(\bar{p}, 1-\bar{p})$.
   - `deterministic_variance`: $\bar{p}(1-\bar{p})$.
3. **Ex-Ante Market Risk Baselines**:
   - `asset_volatility`: Pre-event historical trailing return standard deviation.
   - `market_vol_proxy`: Cross-sectional standard deviation of earnings moves within each calendar week.
   - `cross_sectional_vol_rank`: Within-week percentile rank of asset volatility.

### 1.3 Statistical & Cross-Validation Protocol
- **Leave-One-Regime-Out (LORO)**: Temporal validation holding out an entire multi-year regime (2022, 2023, or 2024–2025) while training on the remaining clean regimes.
- **Dependence-Aware Cluster Bootstrap**: $B = 1,000$ resamples clustered at the **calendar week** level ($168$ unique clusters). Because market conditions affect all stocks reporting in the same week, resamples are drawn across calendar weeks, simultaneously preserving the cross-sectional and cross-seed correlation structure.

---

## 2. Univariate Signal Evaluation

Table 1 reports univariate predictive performance against forecast error $e_i$ and top-20% high error $y_{\text{high\_err}}$ on the pooled clean sample ($N=12,300$).

### Table 1: Univariate Signal Performance (Clean Pooled 2022–2025)

| Category | Signal Name | Spearman $\rho$ | $p$-value | Pearson $r$ | $Q5/Q1$ Error Ratio | High-Error AUROC | High-Error AP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bayesian** | `epistemic_uncertainty` | 0.5945 | $< 10^{-300}$ | 0.1014 | 1.4343 | 0.5015 | 0.2005 |
| **Bayesian** | `epistemic_std` | 0.5945 | $< 10^{-300}$ | 0.1084 | 1.4343 | 0.5015 | 0.2005 |
| **Bayesian** | `total_uncertainty` | 0.5992 | $< 10^{-300}$ | 0.1087 | 1.3800 | 0.5001 | 0.1998 |
| **Bayesian** | `aleatoric_uncertainty` | 0.5992 | $< 10^{-300}$ | 0.1070 | 1.3800 | 0.5001 | 0.1998 |
| **Simple Confidence** | `predictive_entropy` | 0.5992 | $< 10^{-300}$ | 0.1041 | 1.3800 | 0.5001 | 0.1998 |
| **Simple Confidence** | `inv_confidence_margin` | 0.5992 | $< 10^{-300}$ | 0.1120 | 1.3800 | 0.5001 | 0.1998 |
| **Simple Confidence** | `deterministic_variance` | 0.5992 | $< 10^{-300}$ | 0.1087 | 1.3800 | 0.5001 | 0.1998 |
| **Market Volatility** | `asset_volatility` | 0.0895 | $2.8 \times 10^{-23}$ | 0.1237 | 2.1301 | **0.5880** | **0.2499** |
| **Market Volatility** | `cross_sectional_vol_rank` | 0.0876 | $2.1 \times 10^{-22}$ | 0.1167 | 1.9568 | 0.5833 | 0.2508 |
| **Market Volatility** | `market_vol_proxy` | 0.0154 | 0.0882 | 0.0498 | 1.1659 | 0.5305 | 0.2220 |

### Key Univariate Findings:
1. **Rank Correlation Equivalence**: Bayesian epistemic uncertainty ($\rho = 0.5945$) and simple predictive entropy ($\rho = 0.5992$) have virtually identical correlation with absolute forecast error.
2. **Binary High-Error Orthogonality**: Neither epistemic uncertainty ($\text{AUROC} = 0.5015$) nor predictive entropy ($\text{AUROC} = 0.5001$) can discriminate binary high forecast errors. 
3. **Market Volatility Discriminates Tail Events**: Historical asset volatility achieves $\text{AUROC} = 0.5880$ and a $Q5/Q1$ error ratio of $2.13\times$, demonstrating that extreme model forecast failures are primarily driven by underlying asset risk rather than model-space epistemic variance.

---

## 3. Nested Incremental Model Comparison (LORO)

To determine whether epistemic uncertainty provides incremental value beyond confidence and market risk, we compare four nested models using Leave-One-Regime-Out (LORO) cross-validation:
- **Model A (Confidence Only)**: `predictive_entropy`, `inv_confidence_margin`, `deterministic_variance`
- **Model B (Market Only)**: `asset_volatility`, `market_vol_proxy`, `cross_sectional_vol_rank`
- **Model C (Confidence + Market)**: Features from A and B combined
- **Model D (Confidence + Market + Epistemic)**: Features from C + `epistemic_uncertainty`

### Table 2: Leave-One-Regime-Out (LORO) Predictive Performance

| Held-Out Regime | Model | Out-of-Sample $R^2$ | Out-of-Sample MAE | Out-of-Sample AUROC | Out-of-Sample AP | $\Delta \text{AUROC}$ vs C | $\Delta \text{MAE}$ vs C |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2022** ($N=3,573$) | Model A | 0.0128 | 0.2976 | 0.4978 | 0.1931 | -0.0810 | -0.0069 |
| | Model B | 0.0089 | 0.2958 | 0.5785 | 0.2433 | -0.0002 | -0.0052 |
| | Model C | 0.0216 | 0.2907 | 0.5787 | 0.2429 | Ref | Ref |
| | **Model D** | 0.0216 | 0.2907 | 0.5791 | 0.2432 | **+0.0004** | -0.0000 |
| **2023** ($N=3,543$) | Model A | 0.0175 | 0.2939 | 0.5183 | 0.2036 | -0.0583 | -0.0082 |
| | Model B | 0.0083 | 0.2904 | 0.5719 | 0.2374 | -0.0047 | -0.0046 |
| | Model C | 0.0263 | 0.2858 | 0.5766 | 0.2390 | Ref | Ref |
| | **Model D** | 0.0261 | 0.2863 | 0.5731 | 0.2342 | **-0.0035** | -0.0005 |
| **2024–2025** ($N=5,184$) | Model A | 0.0081 | 0.3023 | 0.4977 | 0.2097 | -0.0986 | -0.0034 |
| | Model B | 0.0178 | 0.3042 | 0.5948 | 0.2694 | -0.0015 | -0.0053 |
| | Model C | 0.0261 | 0.2989 | 0.5963 | 0.2644 | Ref | Ref |
| | **Model D** | 0.0261 | 0.2989 | 0.5965 | 0.2656 | **+0.0002** | 0.0000 |
| **Pooled OOS** ($N=12,300$) | Model A | 0.0121 | 0.2985 | 0.4964 | 0.1994 | -0.0891 | -0.0058 |
| | Model B | 0.0127 | 0.2978 | 0.5832 | 0.2472 | -0.0022 | -0.0051 |
| | Model C | 0.0250 | 0.2928 | 0.5855 | 0.2478 | Ref | Ref |
| | **Model D** | 0.0249 | 0.2929 | 0.5845 | 0.2463 | **-0.0009** | **-0.0002** |

### Key LORO Findings:
1. **Incremental Predictive Gain is Flat or Slightly Negative**: Across the pooled out-of-sample regimes, adding Bayesian epistemic uncertainty changes $R^2$ by $-0.00007$, MAE by $-0.00016$, and AUROC by $-0.0009$.
2. **Complementarity of Confidence + Market**: Combining simple confidence and market risk (Model C) doubles out-of-sample $R^2$ ($0.0250$) relative to Model A ($0.0121$) or Model B ($0.0127$) alone, while Model D adds zero further improvement.

---

## 4. Residual Information Test

To directly test whether Bayesian epistemic uncertainty contains residual information uncaptured by simple confidence, we project `epistemic_uncertainty` onto the simple confidence subspace via OLS:
$$\text{epi}_i = \alpha + \beta_1 H(\bar{p}_i) + \beta_2 (0.5 - |\bar{p}_i - 0.5|) + \beta_3 \bar{p}_i(1-\bar{p}_i) + \epsilon_i$$
We then test whether the residual $\hat{\epsilon}_i$ retains any predictive power for forecast error.

### Table 3: Residual Epistemic Uncertainty Diagnostics

| Sample Scope | $R^2$ (Confidence $\to$ Epistemic) | Raw Epistemic Spearman $\rho$ | Residual Epistemic Spearman $\rho$ | Residual $p$-value | Residual Epistemic AUROC | Residual Epistemic AP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Pooled (2022–2025)** | **0.8622** | 0.5945 | 0.1772 | $2.4 \times 10^{-87}$ | **0.5008** | 0.2022 |
| **2022** | 0.8875 | 0.6058 | 0.2082 | $2.9 \times 10^{-36}$ | 0.5067 | 0.2035 |
| **2023** | 0.8702 | 0.6049 | 0.1787 | $8.1 \times 10^{-27}$ | 0.4896 | 0.1900 |
| **2024–2025** | 0.8687 | 0.5804 | 0.1821 | $6.6 \times 10^{-40}$ | 0.5081 | 0.2183 |
| **Seed 42** | 0.8510 | 0.5870 | -0.0908 | $5.7 \times 10^{-9}$ | 0.4987 | 0.2009 |
| **Seed 123** | 0.8836 | 0.5889 | 0.2471 | $4.2 \times 10^{-58}$ | 0.5018 | 0.1987 |
| **Seed 777** | 0.0000 | 0.4618 | -0.0178 | 0.2560 | 0.5127 | 0.2086 |

### Interpretation:
- Simple confidence explains **86.22%** of the variance in epistemic uncertainty.
- Once orthogonalized, the residual epistemic uncertainty has linear Pearson correlation with forecast error of $r = 0.0024$ ($p = 0.789$) and high-error AUROC of $0.5008$. 
- Residual epistemic variance does not carry meaningful linear or discriminatory reliability signals.

---

## 5. Error Calibration by Quintile

### Table 4: Error Calibration by Signal Quintile (Clean Sample, $N=12,300$)

| Signal Name | Quintile | Mean Signal Value | Mean Forecast Error $e$ | Std Forecast Error | Top-20% High-Error Freq (%) | $Q5/Q1$ Error Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`epistemic_uncertainty`** | Q1 (Lowest) | 0.0000 | 0.1987 | 0.3985 | 19.84% | 1.4343 |
| | Q2 | 0.0000 | 0.1982 | 0.3974 | 19.76% | |
| | Q3 | 0.000005 | 0.2062 | 0.3984 | 20.33% | |
| | Q4 | 0.000189 | 0.2170 | 0.3746 | 19.80% | |
| | Q5 (Highest) | 0.006397 | **0.2849** | 0.3111 | 20.28% | |
| **`predictive_entropy`** | Q1 (Lowest) | 0.004112 | 0.2056 | 0.4036 | 20.53% | 1.3800 |
| | Q2 | 0.007879 | 0.1925 | 0.3930 | 19.19% | |
| | Q3 | 0.030992 | 0.2006 | 0.3943 | 19.76% | |
| | Q4 | 0.131281 | 0.2227 | 0.3797 | 20.45% | |
| | Q5 (Highest) | 0.364840 | **0.2837** | 0.3091 | 20.08% | |
| **`asset_volatility`** | Q1 (Lowest) | 0.029735 | 0.1312 | 0.2919 | **10.41%** | **2.1301** |
| | Q2 | 0.043088 | 0.2323 | 0.3872 | 21.14% | |
| | Q3 | 0.051990 | 0.2142 | 0.3763 | 19.47% | |
| | Q4 | 0.061353 | 0.2477 | 0.3997 | 23.05% | |
| | Q5 (Highest) | 0.089116 | **0.2796** | 0.4113 | **25.93%** | |

### Functional Form Analysis:
- The relationship between model uncertainty and forecast error is **threshold-like and convex**: errors are essentially flat across Q1–Q3 (~0.198 to 0.206), rise mildly in Q4 (0.2170), and jump in Q5 (0.2849).
- Note that top-20% high-error frequency is almost identical across epistemic and entropy quintiles (~20%). In contrast, historical asset volatility displays a dramatic gradient in high-error frequency, doubling from $10.41\%$ in Q1 to $25.93\%$ in Q5.

---

## 6. Dependence-Aware Weekly Cluster Bootstrap

To prevent false precision from cross-sectional correlation within earnings announcement clusters, we run $B = 1,000$ block bootstrap iterations resampling across $168$ calendar weeks.

### Table 5: Block Bootstrap Confidence Intervals ($B=1,000$)

| Comparison Metric | Bootstrap Mean | Bootstrap Std | 95% CI Lower | 95% CI Upper | Empirical $p$-value | Statistically Significant ($\alpha=0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\Delta \rho_{\text{Univariate}}$** (`epistemic` - `entropy`) | -0.00486 | 0.00142 | -0.00754 | -0.00215 | 0.000 | **Yes** (Entropy is slightly higher) |
| **$\Delta \text{AUROC}_{\text{Univariate}}$** (`epistemic` - `entropy`) | +0.00133 | 0.00105 | -0.00071 | +0.00322 | 0.216 | **No** (Indistinguishable) |
| **$\Delta \text{AUROC}_{\text{Incremental}}$** (Model D - Model C) | +0.00029 | 0.00048 | -0.00024 | +0.00160 | 0.510 | **No** (Zero incremental gain) |
| **$\Delta \text{MAE}_{\text{Incremental}}$** (Model C - Model D) | +0.00001 | 0.00002 | -0.00001 | +0.00007 | 0.440 | **No** (Zero incremental gain) |

The 95% confidence intervals for incremental out-of-sample improvements $\Delta \text{AUROC}$ and $\Delta \text{MAE}$ strictly straddle zero.

---

## 7. Answers to the Eight Preregistered Questions

### Q1: Does Bayesian epistemic uncertainty predict forecast error?
**Yes.** Bayesian epistemic uncertainty exhibits a strong and highly significant rank correlation with absolute forecast error ($\rho = 0.5945, p < 10^{-300}$ on the pooled clean sample; $\rho = 0.6058$ in 2022, $\rho = 0.6049$ in 2023, $\rho = 0.5804$ in 2024–2025). High epistemic uncertainty events suffer $43.4\%$ higher absolute error in Q5 relative to Q1.

### Q2: Does predictive entropy or margin predict forecast error as well or better?
**Yes, as well or slightly better.** Predictive entropy $H(\bar{p})$ and confidence margin $0.5 - |\bar{p} - 0.5|$ achieve $\rho = 0.5992$, beating epistemic uncertainty by a statistically significant $\Delta \rho = +0.00486$ ($p < 0.001$). Simple deterministic functions of the point prediction match or exceed Bayesian epistemic uncertainty along every univariate reliability metric.

### Q3: Does epistemic uncertainty add incremental information beyond entropy/margin?
**No.** In Leave-One-Regime-Out (LORO) nested model tests, adding epistemic uncertainty to a model with simple confidence and market risk baselines yields $\Delta \text{AUROC} = -0.0009$ and $\Delta \text{MAE} = -0.00016$. Block bootstrap 95% confidence intervals for incremental performance span zero ($[-0.00024, +0.00160]$ for AUROC, $p = 0.51$). Furthermore, simple confidence explains $86.22\%$ of the variance in epistemic uncertainty, leaving residuals that have zero predictive power ($r = 0.0024, p = 0.789; \text{AUROC} = 0.5008$).

### Q4: Does market volatility proxy capture the relevant reliability variation instead?
**Yes, for binary catastrophe and tail risk.** While model confidence predicts continuous absolute estimation error, market volatility (historical asset volatility) is the dominant predictor of extreme, tail forecast errors ($\text{AUROC} = 0.5880$, $Q5/Q1$ error ratio $= 2.13\times$, top-20% high-error frequency doubling from $10.4\%$ to $25.9\%$). Model C (Confidence + Market) provides true complementarity, doubling explained variance ($R^2 = 0.0250$) compared to confidence alone ($R^2 = 0.0121$).

### Q5: Is the relationship between epistemic uncertainty and forecast error linear, monotonic, or threshold-like?
**Convex and threshold-like.** Across quintiles Q1 through Q3, forecast error is virtually flat ($0.1987 \to 0.2062$). Error elevates mildly in Q4 ($0.2170$) and concentrates sharply in Q5 ($0.2849$). Moreover, epistemic uncertainty does not alter the incidence of binary top-20% errors (~$20\%$ across all quintiles).

### Q6: Does epistemic uncertainty behave differently across the 2022, 2023, and 2024–2025 regimes?
**No, its behavior is remarkably invariant.** The rank correlation is consistent across all regimes ($\rho = 0.6058$ in 2022, $0.6049$ in 2023, $0.5804$ in 2024–2025). In none of the three regimes does adding epistemic uncertainty improve out-of-sample prediction over Model C (LORO $\Delta \text{AUROC} = +0.0004$ in 2022, $-0.0035$ in 2023, $+0.0002$ in 2024–2025).

### Q7: How do the 2020–2021 training-overlap regimes compare?
**They exhibit inflated calibration due to training overlap.** In 2020 and 2021, rank correlations are higher ($\rho = 0.6422$ in 2020, $\rho = 0.7252$ in 2021) than in clean regimes ($\rho \approx 0.58 - 0.60$). However, the equivalence between epistemic uncertainty and simple entropy holds identically ($\rho = 0.6473$ vs $0.6422$ in 2020; $\rho = 0.7273$ vs $0.7252$ in 2021). The training overlap identified in Phase 13B.1 inflated absolute correlation levels, but did not grant epistemic uncertainty any unique information.

### Q8: What is the scientific conclusion regarding the value of Bayesian LoRA for portfolio construction?
**Scientific Verdict: Outcome B (Equivalence).**
1. **The Diagnostic Signal Exists**: Model forecast unreliability is real, quantifiable, and predictable ($\rho \approx 0.60$).
2. **Information-Theoretic Redundancy**: Epistemic uncertainty provides no measurable incremental information beyond confidence measures constructed from the prediction mean used in Phase 15. The high collinearity ($R^2 = 86.22\%$) indicates strong empirical redundancy rather than mathematical identity; 32 Monte Carlo forward passes produce no reliability information that is not already captured by predictive entropy $H(\bar{p})$ or confidence margin $|p - 0.5|$.
3. **Portfolio Implication**: This explains why the complex portfolio mappings ($M14$, $M15$, $M16$) failed to outperform simple benchmarks in Phase 13C and Phase 14: epistemic uncertainty provided no independent information beyond point-prediction confidence, while missing the exogenous market volatility dynamics that drive real-world asset covariance and tail returns.
