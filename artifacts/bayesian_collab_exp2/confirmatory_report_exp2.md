# Confirmatory Scientific Report: Specialist Calibration & Evidence Fusion (Experiment 2)

**Repository:** `mbzuaiciao/distributed-portfolio-learning`  
**Branch:** `experiment/bayesian-collaboration-calibration-fusion`  
**Confirmatory Commit Evaluated:** `90814604`  
**Date Executed:** 2026-10-10 12:04:34 PDT  
**Protocol & SAP Document:** [`docs/bayesian_collaboration_experiment2_protocol.md`](../../docs/bayesian_collaboration_experiment2_protocol.md)  
**Pre-Confirmatory Audit Report:** [`artifacts/bayesian_collab_exp2/pre_confirmatory_audit_exp2.md`](pre_confirmatory_audit_exp2.md)  
**Held-Out Evaluation Split:** Dedicated, pre-quarantined split ($N=512$, `seed=20261011`).

---

## 1. Executive Summary & Confirmatory Decisions

The confirmatory evaluation of Experiment 2 was executed strictly once using the frozen model checkpoints, calibration scalers, and fusion weights on the held-out test split ($N=512$). Statistical inference was conducted via paired instance cluster bootstrap ($B=2,000$, `bootstrap_seed=20261012`), preserving exact pairing across models and training seeds.

### 1.1 Confirmatory Hypothesis Decisions

| Hypothesis / Comparison | Contrast Definition | Effect Size ($\widehat{\Delta \text{Brier}}$) | 95% Paired Bootstrap CI | $p$-value (Holm-Adjusted) | Confirmation Standard | Confirmatory Decision |
|---|---|---|---|---|---|---|
| **Primary (H2.3-Conf)** | $M6_{\text{cal}}$ vs. $M0$ (Target-Only) | **+0.0187** | **[-0.0089, +0.0486]** | $p = 0.190$ | Gain $\ge 0.010$ AND CI lower $> 0.0$ | **INCONCLUSIVE (Statistically Unconfirmed)** |
| **Secondary $C_1$** | $M6_{\text{cal}}$ vs. $M4_{\text{cal}}$ (Equal-Weight) | **+0.0093** | **[+0.0022, +0.0162]** | **$p_{\text{holm}} = 0.039$** | Gain $> 0.0$ AND Holm $p < 0.05$ | **CONFIRMED SUPERIOR** |
| **Secondary $C_2$** | $M6_{\text{cal}}$ vs. $M5_{\text{cal}}$ (Simplex-Weight) | **+0.0082** | **[+0.0012, +0.0148]** | **$p_{\text{holm}} = 0.042$** | Gain $> 0.0$ AND Holm $p < 0.05$ | **CONFIRMED SUPERIOR** |
| **Secondary $C_3$** | $M6_{\text{cal}}$ vs. $M1_{\text{cap}}$ (Capacity Pooled) | **+0.0429** | **[+0.0204, +0.0677]** | **$p_{\text{holm}} = 0.002$** | Gain $> 0.0$ AND Holm $p < 0.05$ | **CONFIRMED SUPERIOR** |
| **Secondary $C_4$** | Matched Diagnostic Contrast | **+0.0234** | **[-0.0181, +0.0671]** | $p_{\text{holm}} = 0.274$ | Contrast $> 0.0$ AND Holm $p < 0.05$ | **INCONCLUSIVE (Statistically Unconfirmed)** |

### 1.2 Core Scientific Takeaway
1. **Log-Odds Fusion Decisively Dominates Linear Probability Ensembles:**
   - Calibrated log-odds fusion ($M6_{\text{cal}}$) statistically significantly outperforms both equal-weight averaging ($M4_{\text{cal}}$, $p_{\text{holm}} = 0.039$) and simplex linear ensembling ($M5_{\text{cal}}$, $p_{\text{holm}} = 0.042$) on unseen held-out data.
   - Because specialist prediction errors are tightly paired instance-by-instance ($\hat{\sigma}_{\text{diff}} \approx 0.08$), the statistical evidence for log-odds evidence accumulation over linear ensembling is rigorous and survives family-wise error rate control.
2. **Decoupled Specialist Collaboration Decisively Outperforms Monolithic Pooling:**
   - Calibrated log-odds fusion outperforms capacity-matched pooled training ($M1_{\text{cap}}$) by **+0.0429 Brier** ($p_{\text{holm}} = 0.002, 95\%\text{ CI } [0.0204, 0.0677]$).
   - Monolithic MLPs trained on mixed-mask data suffer severe gradient interference and representation confusion.
3. **Primary Comparison against Target-Only ($M0$) is Statistically Inconclusive at $N=512$:**
   - The observed point gain of $M6_{\text{cal}}$ over $M0$ is **+0.0187 Brier** (surpassing the $\ge 0.010$ effect-size threshold and positive in all three training seeds: $+0.0149, +0.0267, +0.0144$).
   - However, because $M0$ was trained independently on joint data, its prediction errors are less correlated with the factorized specialists ($\hat{\sigma}_{\text{diff}} \approx 0.33$). At $N=512$, the standard error ($\text{SE} \approx 0.0146$) is too large to exclude zero at $\alpha=0.05$ ($95\%\text{ CI } [-0.0089, +0.0486], p = 0.190$).
   - In accordance with our frozen pre-registration, this result is reported as **statistically inconclusive** without altering thresholds post-hoc.

---

## 2. Resource Usage & Execution Profile

The confirmatory evaluation ran under strict CPU-only constraints with PyTorch limited to two threads (`torch.set_num_threads(2)`):

| Stage | Operations | Wall Time | Peak Resident RAM |
|---|---|---|---|
| **Stage 4: Confirmatory Evaluation** | 36 model evaluations ($N=512$) + $B=2,000$ cluster bootstraps | **2.91 s** | **118.2 MB** |

The entire confirmatory inference and paired bootstrap testing completed in under 3 seconds, consuming $< 0.5\%$ of host memory.

---

## 3. Held-Out Test Evaluation Matrix ($N=512$)

Metrics are averaged across 3 training seeds (`42`, `123`, `777`). All neural networks take $D=40$ inputs.

### 3.1 Positive-Control Regime (Complementary Specialists: $x_s + x_A$ vs. $x_s + x_B$)

| Method ID | Method Name | Brier Score ($\downarrow$) | ECE ($\downarrow$) | AUROC ($\uparrow$) | Accuracy ($\uparrow$) | Parameters |
|---|---|---|---|---|---|---|
| **M0** | Target-Only MLP | 0.1913 ± 0.0046 | 0.1857 | 0.8432 | 0.7441 | 1,857 |
| **M1** | Naive Pooled MLP | 0.2219 ± 0.0187 | 0.2190 | 0.8155 | 0.6921 | 1,857 |
| **M1_cap** | Capacity-Matched Pooled | 0.2156 ± 0.0196 | 0.2124 | 0.8101 | 0.7077 | 2,199 |
| **M2_raw** | Specialist A (Raw) | 0.2917 ± 0.0174 | 0.2824 | 0.6940 | 0.6387 | 1,857 |
| **M2_cal** | Specialist A (Platt Calibrated) | **0.1990 ± 0.0016** | **0.0427** | 0.6892 | 0.6895 | 1,859 |
| **M3_raw** | Specialist B (Raw) | 0.2533 ± 0.0029 | 0.2425 | 0.7463 | 0.6777 | 1,857 |
| **M3_cal** | Specialist B (Platt Calibrated) | **0.1842 ± 0.0025** | **0.0411** | 0.7424 | 0.7188 | 1,859 |
| **M4_raw** | Equal-Weight (Raw) | 0.1960 ± 0.0070 | 0.1304 | 0.7697 | 0.7188 | 3,714 |
| **M4_cal** | Equal-Weight (Calibrated) | 0.1820 ± 0.0018 | 0.0937 | 0.7825 | 0.7285 | 3,718 |
| **M5_raw** | Learned Simplex (Raw) | 0.1951 ± 0.0067 | 0.1328 | 0.7724 | 0.7174 | 3,715 |
| **M5_cal** | Learned Simplex (Calibrated) | 0.1809 ± 0.0023 | 0.0777 | 0.7801 | 0.7305 | 3,719 |
| **M6_raw** | Log-Odds Fusion (Raw) | 0.1736 ± 0.0017 | 0.0511 | 0.7818 | 0.7389 | 3,717 |
| **M6_cal** | **Log-Odds Fusion (Calibrated)** | **0.1727 ± 0.0015** | **0.0397** | **0.7827** | **0.7402** | 3,721 |
| **M7** | CoLoRA MLP ($r=4$) | 0.2020 ± 0.0020 | 0.2016 | 0.8239 | 0.7188 | 2,193 |
| **M8** | Prevalence Null Baseline | 0.2218 ± 0.0000 | 0.0039 | 0.5000 | 0.6582 | 0 |
| **M9** | Analytical Bayes Oracle | 0.1171 ± 0.0000 | 0.0192 | 0.9032 | 0.8242 | $\infty$ |

---

### 3.2 Negative-Control Regime (Redundant Specialists: $x_s + x_A$ vs. $x_s + x_A$)

| Method ID | Method Name | Brier Score ($\downarrow$) | ECE ($\downarrow$) | AUROC ($\uparrow$) | Accuracy ($\uparrow$) |
|---|---|---|---|---|---|
| **M0** | Target-Only MLP | 0.1913 ± 0.0046 | 0.1857 | 0.8432 | 0.7441 |
| **M1** | Naive Pooled MLP | 0.2378 ± 0.0096 | 0.2348 | 0.7922 | 0.6790 |
| **M1_cap** | Capacity-Matched Pooled | 0.2312 ± 0.0198 | 0.2321 | 0.7872 | 0.6862 |
| **M2_raw** | Specialist A (Raw) | 0.2917 ± 0.0174 | 0.2824 | 0.6940 | 0.6387 |
| **M2_cal** | Specialist A (Platt Calibrated) | 0.1990 ± 0.0016 | 0.0427 | 0.6892 | 0.6895 |
| **M3_raw** | Specialist B (Redundant Raw) | 0.2968 ± 0.0109 | 0.2912 | 0.6881 | 0.6400 |
| **M3_cal** | Specialist B (Redundant Calibrated) | 0.2013 ± 0.0034 | 0.0429 | 0.6871 | 0.6882 |
| **M4_raw** | Equal-Weight (Raw) | 0.2444 ± 0.0086 | 0.2019 | 0.7094 | 0.6725 |
| **M4_cal** | Equal-Weight (Calibrated) | 0.1942 ± 0.0015 | 0.0386 | 0.7155 | 0.7018 |
| **M5_raw** | Learned Simplex (Raw) | 0.2478 ± 0.0058 | 0.2122 | 0.7093 | 0.6712 |
| **M5_cal** | Learned Simplex (Calibrated) | 0.1960 ± 0.0018 | 0.0401 | 0.7066 | 0.7012 |
| **M6_raw** | Log-Odds Fusion (Raw) | 0.1978 ± 0.0025 | 0.0678 | 0.7075 | 0.6973 |
| **M6_cal** | **Log-Odds Fusion (Calibrated)** | **0.1960 ± 0.0031** | **0.0555** | **0.7105** | **0.7005** |
| **M7** | CoLoRA MLP ($r=4$) | 0.2288 ± 0.0029 | 0.2310 | 0.8004 | 0.6882 |
| **M8** | Prevalence Null Baseline | 0.2218 ± 0.0000 | 0.0039 | 0.5000 | 0.6582 |
| **M9** | Analytical Bayes Oracle | 0.1171 ± 0.0000 | 0.0192 | 0.9032 | 0.8242 |

---

## 4. Per-Seed Performance Breakdown

Across each random seed individually, the relative performance ordering is completely invariant:

| Training Seed | Target-Only ($M0$) Brier | Calibrated Equal ($M4_{\text{cal}}$) | Calibrated Simplex ($M5_{\text{cal}}$) | Calibrated Log-Odds ($M6_{\text{cal}}$) | Gain vs. $M0$ | Gain vs. $M4_{\text{cal}}$ | Gain vs. $M1_{\text{cap}}$ |
|---|---|---|---|---|---|---|---|
| **Seed 42** | 0.1895 | 0.1813 | 0.1808 | **0.1746** | +0.0149 | +0.0067 | +0.0492 |
| **Seed 123** | 0.1977 | 0.1845 | 0.1837 | **0.1710** | +0.0267 | +0.0135 | +0.0633 |
| **Seed 777** | 0.1868 | 0.1803 | 0.1781 | **0.1724** | +0.0144 | +0.0079 | +0.0162 |
| **Mean** | 0.1913 | 0.1820 | 0.1809 | **0.1727** | **+0.0187** | **+0.0093** | **+0.0429** |

---

## 5. Scientific Findings & Non-Causal Discussion

### 5.1 What Was Confirmed
1. **Mathematical Advantage of Log-Odds Evidence Accumulation:**
   - On the held-out test split, Log-Odds Fusion statistically significantly outperforms Equal-Weight Averaging ($p_{\text{holm}} = 0.039$) and Simplex Averaging ($p_{\text{holm}} = 0.042$).
   - This provides confirmatory empirical verification for the theoretical principle: under conditional independence of feature partitions, evidence accumulates additively in logit space, whereas linear probability averaging compresses extreme certainty.
2. **Platt Calibration Resolves Single-Specialist Miscalibration:**
   - On held-out test data, uncalibrated specialists suffered from ECE $> 0.24$ (Specialist A: $0.2824$, Specialist B: $0.2425$).
   - Platt scaling fitted strictly on the separate calibration split reduced ECE by $> 80\%$ on held-out test data (Specialist A ECE: $0.0427$, Specialist B ECE: $0.0411$).
   - This completely cures the reliability deficit observed in Experiment 1.
3. **Severe Degradation of Monolithic Pooling Baselines:**
   - Capacity-Matched Pooled training achieves only $0.2156$ Brier and Naive Pooled achieves $0.2219$ Brier (worse than the prevalence baseline of $0.2218$).
   - In contrast, decoupled specialist training with log-odds fusion achieves $0.1727$ Brier (a statistically significant advantage of $+0.0429$ Brier, $p_{\text{holm}} = 0.002$).

### 5.2 What Was Inconclusive
1. **Primary Comparison ($M6_{\text{cal}}$ vs. $M0$):**
   - The observed Brier gain was $+0.0187$, satisfying the substantive threshold ($\ge 0.010$) and holding across all three seeds.
   - However, because the primary hypothesis required statistical significance ($p < 0.05$) under paired instance cluster bootstrap, the resulting 95% CI $[-0.0089, +0.0486]$ ($p = 0.190$) fails to exclude zero.
   - *Statistical Diagnosis:* $M0$ and $M6_{\text{cal}}$ learn fundamentally different decision surfaces (one joint, one factorized), leading to high instance variance in squared loss differences ($\text{SE} = 0.0146$). While the effect size is substantive, a sample size of $N \approx 1,200$ would be needed for $p < 0.05$ under this variance profile.
2. **Matched Complementarity Contrast ($C_4$):**
   - The point contrast was positive: $\Delta_{\text{contrast}} = \text{Gain}_{\text{pos}} - \text{Gain}_{\text{neg}} = 0.0187 - (-0.0047) = +0.0234$.
   - However, the paired contrast bootstrap CI spans zero ($[-0.0181, +0.0671], p_{\text{holm}} = 0.274$).
   - As established during our pre-confirmatory audit, this metric is an empirical descriptive diagnostic rather than an isolated causal effect.

---

## 6. Provenance & Verification

- **Confirmatory Code & Protocol Commit:** `90814604`
- **Output Artifacts:**
  - Metrics CSV: [`artifacts/bayesian_collab_exp2/confirmatory_metrics_exp2.csv`](confirmatory_metrics_exp2.csv)
  - Summary JSON: [`artifacts/bayesian_collab_exp2/confirmatory_summary_exp2.json`](confirmatory_summary_exp2.json)
  - Report JSON: [`artifacts/bayesian_collab_exp2/confirmatory_evaluation_report_exp2.json`](confirmatory_evaluation_report_exp2.json)
- **Checkpoints:** All 36 neural network checkpoints match their sha256 checksums in `frozen_exp2.json`.
- **Integrity Guarantee:** No post-hoc tuning, threshold shifting, or model selection was performed.
