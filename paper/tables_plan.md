# Tables Plan & Formal Table Specifications

This document defines the formal tables to be included in the consolidated paper draft. All numbers are populated directly from verified clean primary artifacts (`artifacts/phase17_clean_bayesian/` and `artifacts/phase16_1_audit/`).

---

## Table 1: Clean Model Out-of-Sample Performance Comparison

**Caption**: *Table 1: Out-of-sample forecast performance on clean historical regimes (pooled 2022–2025; $N=12,300$ evaluations across 3 random seeds). All models were trained on strictly clean text without target outcome leakage. Clustered bootstrap 95% confidence intervals across $B=1,000$ calendar-week clusters are reported in brackets.*

| Model System | Inference Passes | AUROC | Average Precision | Brier Score | Log Loss | ECE | Mean Predicted $p$ | Std $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base-Rate Null ($p \equiv 0.20$)** | 0 | 0.50000 | 0.20000 | 0.16000 | 0.50040 | 0.00000 | 0.20000 | 0.00000 |
| **Clean Deterministic LoRA** | 1 | 0.51312<br>`[0.503, 0.523]` | 0.21049 | 0.16079<br>`[0.159, 0.162]` | 0.50332<br>`[0.500, 0.507]` | 0.01785 | 0.18747 | 0.03083 |
| **Clean Bayesian Posterior Mean** | 1 | 0.52531<br>`[0.516, 0.535]` | 0.22358 | 0.15987<br>`[0.158, 0.161]` | 0.50007<br>`[0.497, 0.503]` | 0.00686 | 0.19314 | 0.02049 |
| **Clean Bayesian MC32** | 32 | 0.52504<br>`[0.516, 0.535]` | 0.22315 | 0.15987<br>`[0.158, 0.161]` | 0.50009<br>`[0.497, 0.503]` | 0.00695 | 0.19331 | 0.02053 |

---

## Table 2: Clean Performance Across Historical Market Regimes

**Caption**: *Table 2: Out-of-sample forecast accuracy disaggregated across three distinct clean post-training historical regimes: the 2022 bear market / inflation tightening, the 2023 recovery, and the 2024–2025 continuation.*

| Historical Regime | Calendar Period | Clean Events ($N$) | Det LoRA AUROC | Bayes Mean AUROC | Det LoRA Brier | Bayes Mean Brier | Null Base-Rate Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2022 Regime** | 2022-01-01 to 2022-12-31 | 3,573 | 0.50925 | 0.51501 | 0.15767 | 0.15688 | 0.15688 |
| **2023 Regime** | 2023-01-01 to 2023-12-31 | 3,543 | 0.51303 | 0.51661 | 0.15646 | 0.15570 | 0.15583 |
| **2024–2025 Regime** | 2024-01-01 to 2025-05-15 | 5,184 | 0.51692 | 0.53786 | 0.16589 | 0.16477 | 0.16500 |
| **Full Clean Sample** | **2022 to 2025** | **12,300** | **0.51312** | **0.52531** | **0.16079** | **0.15987** | **0.16000** |

---

## Table 3: Bayesian Posterior Mean vs. MC32 Computational & Statistical Comparison

**Caption**: *Table 3: Inference latency, forward passes, and predictive metrics comparing single-pass evaluation of the variational posterior mean ($\mu$) against 32-draw Monte Carlo sampling (MC32). Statistical equivalence is tested against predeclared margins.*

| Metric / Attribute | Posterior Mean ($\mu$) | MC32 Sampling | Pairwise Difference (MC32 - Mean) | 95% Clustered CI | Predeclared Equivalence Margin | Equivalence Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Forward Passes / Event** | 1 | 32 | $+31$ | — | — | — |
| **Inference Latency** | 7.89 ms | 241.02 ms | $+233.13\text{ ms } (+30.6\times)$ | — | — | Compute Inefficient |
| **AUROC** | 0.52531 | 0.52504 | -0.00028 | `[-0.00080, +0.00027]` | $\pm 0.005$ | **EQUIVALENT** |
| **Average Precision** | 0.22358 | 0.22315 | -0.00043 | `[-0.00120, +0.00035]` | — | **EQUIVALENT** |
| **Brier Score** | 0.15987 | 0.15987 | +0.00001 | `[+0.00000, +0.00002]` | $\pm 0.002$ | **EQUIVALENT** |
| **Log Loss** | 0.50007 | 0.50009 | +0.00003 | `[+0.00000, +0.00005]` | $\pm 0.010$ | **EQUIVALENT** |
| **Expected Calibration Error** | 0.00686 | 0.00695 | +0.00022 | `[-0.00064, +0.00186]` | $\pm 0.010$ | **EQUIVALENT** |

---

## Table 4: Forensic Audit Summary & Methodological Lessons

**Caption**: *Table 4: Summary of experimental vulnerabilities identified during the audit, their initial misleading appearance, forensic diagnosis, and final scientific consequence.*

| Vulnerability / Issue | What Initially Appeared | Forensic Audit Finding | Final Scientific Consequence |
| :--- | :--- | :--- | :--- |
| **Target Outcome Leakage** | Model achieved near-perfect validation AUROC ($>0.999$) and apparent high confidence. | Continuous label realization `move_1d` was concatenated into training text strings (`". 1-day move was " + move_1d`). | Historical checkpoints were compromised; clean test-time evaluation suffered distribution collapse to chance ($\text{AUROC} \approx 0.50$). |
| **Reliability Metric Artifact** | Predictive entropy strongly tracked prediction error ($\rho \approx 0.60$), suggesting exceptional error-awareness. | In binary classification under class imbalance, rank correlation within $y=0$ is mechanically $+1.0$ and within $y=1$ is $-1.0$ for $p < 0.5$. | Pooled correlation reflects probability shrinkage toward zero rather than learned epistemic error awareness. |
| **Base-Rate / Null Baseline** | Bayesian training appeared to provide superior calibration compared to unregularized models. | A zero-parameter constant base-rate predictor ($p \equiv 0.20$) achieves Brier score of $0.16000$ and zero calibration error. | Fine-tuned models barely move off the base rate ($\bar{p} \approx 0.19$, $\sigma_p \approx 0.02$), showing negligible text signal. |
| **Monte Carlo Redundancy** | 32-draw posterior sampling was presumed necessary to capture weight uncertainty. | Posterior mean parameters $\mu$ produce predictions that correlate with MC32 at $r > 0.9999$, with all metrics within equivalence bounds. | MC sampling introduces a $30.6\times$ latency penalty with zero measurable forecast gain; single-pass inference is strictly preferred. |
| **Portfolio Instability** | Uncertainty-aware risk inflation ($M_{16}$) appeared to deliver substantial Sharpe gains in 2023. | $M_{16}$ acted as an unconditional bear-market de-risking multiplier, causing severe underperformance in 2022 and 2024–2025. | No robust ex-ante portfolio mapping survived clean adjudication; under unadjusted Markowitz ($M_0$), deterministic LoRA achieved higher utility. |
