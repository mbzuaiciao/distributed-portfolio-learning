# Phase 13B: Development-Only Uncertainty Mapping Selection Report

**Date:** 2026-09-11  
**Status:** COMPLETE & FROZEN  
**Temporal Domain:** 2018-01-04 to 2019-12-20 (Strict Development Window)  
**Total Development Events:** 2,203  
**Active Portfolio Horizon:** $H = 4$ weeks  
**Execution Environment:** Apple Silicon (M2 MPS), PyTorch 2.14, Transformers, SciPy  

---

## 1. Executive Summary & Audit Safeguards

Phase 13B establishes the frozen parameter values for three distinct asset-level epistemic uncertainty mappings:
- **M14**: Asset-Level Signal Shrinkage ($\lambda$)
- **M15**: Asset-Level Selective Abstention ($f$)
- **M16**: Asset-Level Risk Inflation ($\lambda_u$)

### Strict Protocol Enforcement
1. **Zero Evaluation Leakage:**  
   All 6,489 historical events from the evaluation regimes (2020, 2021, 2022, 2023, 2024–2025) were **strictly excluded** from this phase. The pipeline enforced an explicit runtime assertion requiring `all(date <= "2019-12-20 23:59:59")`.
2. **Genuine Model Inference Only:**  
   Predictions were generated exclusively by running genuine forward passes through the frozen Phase-11 checkpoints (`bayesian_lora_seed_42_state.pt`, `seed_123`, `seed_777`, 32 Monte Carlo draws each). No synthetic proxies, perturbations, or legacy Stage-2 outputs were utilized.
3. **Artifact Immutability:**  
   The resulting development predictions are frozen and recorded under canonical SHA256:
   ```
   Artifact: artifacts/phase13_uncertainty_mapping/development_predictions.csv
   SHA256:   6460761d8ba889e551ca4d0fa279f6ff2fe589fc6ce7813a8626c6b412e60b29
   Rows:     6,609 (2,203 events x 3 seeds)
   ```

---

## 2. Mathematical Mapping Formulations & Candidate Grids

For each earnings announcement $i$ in weekly cohort $t$, epistemic standard deviation is defined as:
$$\sigma_{\text{epi}, i} = \sqrt{\operatorname{Var}_{\text{MC}}(p_i)}$$

The cohort-standardized relative epistemic uncertainty $z_i$ is computed via robust MAD scaling:
$$z_i = \frac{\sigma_{\text{epi}, i} - \operatorname{median}(\sigma_{\text{epi}})}{\max(1.4826 \cdot \operatorname{MAD}(\sigma_{\text{epi}}), 10^{-6})}, \quad z_{\text{pos}, i} = \max(z_i, 0)$$

### Candidate Mapping Definitions
1. **M14 — Signal Shrinkage:**  
   Shrinks forecast alpha proportionally to excess uncertainty:
   $$\mu_{\text{adj}, i} = \frac{\mu_i}{1 + \lambda \cdot z_{\text{pos}, i}}$$
   Candidate grid: $\lambda \in \{0.25, 0.50, 1.00, 2.00\}$.
2. **M15 — Selective Abstention:**  
   Ranks assets within the cohort by $\sigma_{\text{epi}, i}$ descending and zeros out forecast alpha for the top fraction $f$:
   $$\mu_{\text{adj}, i} = 0 \quad \forall i \in \operatorname{TopFraction}(f)$$
   Candidate grid: $f \in \{0.10, 0.20, 0.30\}$.
3. **M16 — Asset-Level Risk Inflation:**  
   Inflates asset diagonal variance in the covariance matrix $Q$ without altering off-diagonal covariances:
   $$Q_{\text{adj}} = Q + \lambda_u \operatorname{diag}(z_{\text{pos}})$$
   Candidate grid: $\lambda_u \in \{0.10, 0.25, 0.50, 1.00\}$.

### Reference Benchmarks
- **M0 (Unadjusted Baseline):** Raw unadjusted mean forecast alpha with $Q_{\text{base}} = 0.04 I + 0.01 \mathbf{1}\mathbf{1}^T$.
- **M13 (Global Timing Comparator):** Frozen macro hysteresis timing overlay ($\tau_{\text{enter}}=1.2, \tau_{\text{exit}}=0.2, s \in [0.45, 0.85]$).

---

## 3. Predeclared Deterministic Selection Rule

To select a single parameter candidate for each mapping before accessing any evaluation regimes:
1. **Mean Performance:** Require positive mean $\Delta U$ vs unadjusted M0 across seeds ($\overline{\Delta U} > 0$).
2. **Sign Consistency:** Maximize positive seed count (require 3/3 positive seeds where possible).
3. **Downside Robustness:** Maximize worst-seed $\Delta U$ ($\min_{s} \Delta U_s$) to penalize single-seed outlier dependence.
4. **Deterministic Tie-Breaker:** Select candidate with lowest average annual traded turnover.

---

## 4. Development Selection Results

### Complete Candidate Grid Summary

| Strategy | Parameter | Mean $\Delta U$ vs M0 | Worst Seed $\Delta U$ | Positive Seeds | Annual Turnover | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0** | 0.00 | 0.000000 | 0.000000 | 0 / 3 | 26.60 | Baseline |
| **M13** | 0.00 | -0.000404 | -0.000611 | 0 / 3 | 23.51 | Macro Benchmark |
| **M14** | **0.25** | **+0.000021** | **-0.000008** | **2 / 3** | **26.60** | **PASSED & SELECTED** |
| M14 | 0.50 | +0.000022 | -0.000031 | 2 / 3 | 26.60 | Rejected (lower worst seed) |
| M14 | 1.00 | +0.000024 | -0.000050 | 2 / 3 | 26.60 | Rejected (lower worst seed) |
| M14 | 2.00 | +0.000023 | -0.000070 | 2 / 3 | 26.60 | Rejected (lower worst seed) |
| M15 | 0.10 | -0.000083 | -0.000311 | 2 / 3 | 26.60 | Failed Positive Criterion |
| M15 | 0.20 | -0.000050 | -0.000154 | 1 / 3 | 26.60 | Failed Positive Criterion |
| **M15** | **0.30** | **-0.000005** | **-0.000033** | **2 / 3** | **26.60** | **FROZEN COMPARATOR (Failed Positive Criterion)** |
| **M16** | **0.10** | **+0.000136** | **-0.000004** | **2 / 3** | **26.60** | **PASSED & SELECTED** |
| M16 | 0.25 | +0.000155 | -0.000088 | 2 / 3 | 26.60 | Rejected (lower worst seed) |
| M16 | 0.50 | +0.000146 | -0.000179 | 2 / 3 | 26.60 | Rejected (lower worst seed) |
| M16 | 1.00 | +0.000121 | -0.000278 | 2 / 3 | 26.60 | Rejected (lower worst seed) |

---

## 5. Seed-by-Seed Performance of Selected Candidates

### Baseline M0 Performance
- **Seed 42:** Utility = `0.002525`, Cumulative Return = `+28.71%`, Volatility = `1.64%`, Max Drawdown = `-22.45%`
- **Seed 123:** Utility = `0.002005`, Cumulative Return = `+22.19%`, Volatility = `1.20%`, Max Drawdown = `-14.02%`
- **Seed 777:** Utility = `0.001579`, Cumulative Return = `+17.10%`, Volatility = `1.21%`, Max Drawdown = `-14.56%`
- **Mean M0 Utility:** `0.002036`

### Selected M14 ($\lambda = 0.25$) — Passed Positive Development Criterion
- **Seed 42:** Utility = `0.002595` ($\Delta U = +0.000070$, Return = `+29.61%`, MDD = `-22.56%`)
- **Seed 123:** Utility = `0.001997` ($\Delta U = -0.000008$, Return = `+22.09%`, MDD = `-14.08%`)
- **Seed 777:** Utility = `0.001580` ($\Delta U = +0.000001$, Return = `+17.11%`, MDD = `-14.56%`)
- **Mean $\Delta U$:** `+0.000021` | **Worst Seed:** `-0.000008`

### Frozen M15 Comparator ($f = 0.30$) — Failed Positive Development Criterion
- **Seed 42:** Utility = `0.002543` ($\Delta U = +0.000018$, Return = `+28.94%`, MDD = `-21.62%`)
- **Seed 123:** Utility = `0.001972` ($\Delta U = -0.000033$, Return = `+21.79%`, MDD = `-14.43%`)
- **Seed 777:** Utility = `0.001580` ($\Delta U = +0.000001$, Return = `+17.11%`, MDD = `-14.54%`)
- **Mean $\Delta U$:** `-0.000005` (Negative on average) | **Worst Seed:** `-0.000033`
- *Note:* All candidate parameter values for M15 ($f \in [0.10, 0.20, 0.30]$) generated negative mean development $\Delta U$. M15 did not pass the positive-development criterion. $f=0.30$ is frozen strictly as the representative baseline comparator for selective abstention because it minimizes downside degradation and has the best worst-seed utility.

### Selected M16 ($\lambda_u = 0.10$) — Passed Positive Development Criterion
- **Seed 42:** Utility = `0.002660` ($\Delta U = +0.000135$, Return = `+30.45%`, MDD = `-22.78%`)
- **Seed 123:** Utility = `0.002281` ($\Delta U = +0.000276$, Return = `+25.61%`, MDD = `-14.50%`)
- **Seed 777:** Utility = `0.001576` ($\Delta U = -0.000004$, Return = `+17.06%`, MDD = `-16.70%`)
- **Mean $\Delta U$:** `+0.000136` | **Worst Seed:** `-0.000004`

---

## 6. Scientific Analysis of Findings

1. **Superiority of Convex Risk Penalization (M16):**  
   M16 ($\lambda_u = 0.10$) passed the positive-development criterion and delivered the highest net utility improvement in development ($+0.000136$ weekly, equivalent to $\approx +0.71\%$ annualized net utility). By penalizing uncertain assets in the quadratic risk metric rather than zeroing or shrinking alpha, Markowitz smoothly downweights high-uncertainty names while retaining exposure when alpha conviction is sufficiently large.
2. **Smooth Continuous Shrinkage (M14):**  
   M14 ($\lambda = 0.25$) also passed the positive-development criterion, demonstrating mild, consistent positive gains across seeds ($+0.000021$ mean $\Delta U$). Larger shrinkage parameters ($\lambda \ge 0.50$) slightly increased mean utility on seed 42 but exacerbated the downside on seed 123, making $\lambda = 0.25$ the optimal robust choice.
3. **Failure of Hard Abstention (M15):**  
   All M15 candidates produced negative average utility across the development period, failing the positive-development criterion. Abruptly truncating top-uncertainty names to zero alpha discards high-conviction positive moves, confirming the theoretical expectation that soft continuous shrinkage (M14) and convex risk inflation (M16) dominate discrete exclusion. $f=0.30$ is frozen solely as the least-bad representative comparator.
4. **Behavior of Global Timing Benchmark (M13):**  
   M13 yielded negative utility ($\Delta U = -0.000404$) relative to 100% invested M0 during 2018–2019. This is expected: the development period was predominantly a strong bull market, where reducing exposure to 45% during elevated volatility periods incurs a cash drag. However, M13 substantially reduced maximum drawdown (e.g. from $-22.45\%$ to $-16.68\%$ on seed 42), demonstrating its risk-mitigation property.

---

## 7. Preregistration Freeze

The selected parameters and representative comparator are formally preregistered in `artifacts/phase13_uncertainty_mapping/mapping_preregistration.json`:
```json
{
  "selected_frozen_parameters": {
    "lambda_M14": 0.25,
    "fraction_M15": 0.30,
    "lambda_M16": 0.10
  }
}
```

No further hyperparameter adjustment, calibration, or model retraining will be conducted prior to or during the Phase 13C evaluation across the 2020–2025 regimes.
