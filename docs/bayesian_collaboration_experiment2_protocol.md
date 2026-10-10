# Experiment 2 Protocol: Specialist Calibration & Evidence Fusion

**Repository:** `mbzuaiciao/distributed-portfolio-learning`  
**Branch:** `experiment/bayesian-collaboration-calibration-fusion`  
**Date Frozen:** 2026-10-10  
**Status:** PROSPECTIVE PRE-REGISTRATION — frozen before execution.

---

## 1. Scientific Objective

Experiment 1 demonstrated that while synthetic specialist data contains genuine complementary predictive information
($\Delta_{\text{oracle}} = 0.0419$), single specialists suffer from epistemic overconfidence under partial feature
observability (ECE $\approx 0.30$), and linear probability averaging mathematically compresses joint dynamic range.

Experiment 2 prospectively tests whether:
1. **Post-hoc calibration** on held-out calibration data cures specialist overconfidence on unseen data.
2. **Log-odds evidence fusion** in logit space outperforms equal-weight and simplex linear averaging.
3. The resulting **calibrated/fused specialist system** outperforms target-only and capacity-matched pooled baselines.
4. Performance gains are causally attributable to **complementary feature information** via matched redundant controls.

---

## 2. Split Partition Architecture & Data Hygiene

To strictly prevent data leakage and circular weight fitting, Experiment 2 establishes a **5-split partition architecture**:

```
                          ┌─ Specialist A Train (N=256, xs, xA)
1. Model Training Splits ─┼─ Specialist B Train (N=256, xs, xB / xs, xA in neg)
                          └─ Target Train       (N=128, all features)

2. Calibration Split     ─── Dedicated N=128 (fits Platt scaling for M2, M3 only)

3. Fusion Fitting Split  ─── Dedicated N=128 (fits Simplex w and Log-odds weights only)

4. Development Split     ─── Dedicated N=128 (evaluates development hypotheses & gates)

5. Held-Out Test Split   ─── Dedicated N=512 (STRICTLY LOCKED & UNTOUCHED)
```

### Isolation Guarantees
- `calibration_split` is used **only** to fit 1-parameter Platt calibration scalers on specialist outputs.
- `fusion_fitting_split` is used **only** to fit fusion weights on calibrated specialist predictions.
- `development_split` evaluates development hypotheses without any example reuse from calibration or fusion fitting.
- `held_out_test_split` is completely unread and unblinded.

---

## 3. Method Matrix (16 Models / Combinations)

| Method ID | Model Class | Training Data | Post-Hoc Transform | Inference Mask |
|---|---|---|---|---|
| **$M0$ target-only** | `SmallMLP` (1,857 params) | Target train ($N=128$) | Raw output | Full mask |
| **$M1$ pooled** | `SmallMLP` (1,857 params) | Union ($N=640$) | Raw output | Full mask |
| **$M1_{\text{cap}}$ pooled** | `CapacityMatchedMLP` (2,199 params) | Union ($N=640$) | Raw output | Full mask |
| **$M2_{\text{raw}}$ spec A** | `SmallMLP` (1,857 params) | Spec A train ($N=256$) | Raw output | A-mask |
| **$M2_{\text{cal}}$ spec A** | `SmallMLP` + Platt | Spec A train ($N=256$) | Platt calibrated on cal split | A-mask |
| **$M3_{\text{raw}}$ spec B** | `SmallMLP` (1,857 params) | Spec B train ($N=256$) | Raw output | B-mask (pos) / A-mask (neg) |
| **$M3_{\text{cal}}$ spec B** | `SmallMLP` + Platt | Spec B train ($N=256$) | Platt calibrated on cal split | B-mask (pos) / A-mask (neg) |
| **$M4_{\text{raw}}$ equal raw** | $0.5 p_A + 0.5 p_B$ | Spec A + B models | Raw linear average | A-mask + B-mask |
| **$M4_{\text{cal}}$ equal cal** | $0.5 p_{A,\text{cal}} + 0.5 p_{B,\text{cal}}$ | Spec A + B models | Calibrated linear average | A-mask + B-mask |
| **$M5_{\text{raw}}$ simplex raw** | $w_A p_A + w_B p_B$ | Fit on fusion split | Raw simplex mixture | A-mask + B-mask |
| **$M5_{\text{cal}}$ simplex cal** | $w_A p_{A,\text{cal}} + w_B p_{B,\text{cal}}$ | Fit on fusion split | Calibrated simplex mixture | A-mask + B-mask |
| **$M6_{\text{raw}}$ log-odds raw** | $\sigma(w_0 + w_A \text{logit}(p_A) + w_B \text{logit}(p_B))$ | Fit on fusion split | Raw log-odds fusion | A-mask + B-mask |
| **$M6_{\text{cal}}$ log-odds cal** | $\sigma(w_0 + w_A \text{logit}(p_{A,\text{cal}}) + w_B \text{logit}(p_{B,\text{cal}}))$ | Fit on fusion split | Calibrated log-odds fusion | A-mask + B-mask |
| **$M7$ CoLoRA** | `CoLoRAMLP` (2,193 params) | Multi-task union ($N=640$) | Raw output, task 0 | Full mask |
| **$M8$ prevalence null** | Constant $\hat{p} = 0.35$ | Zero parameters | Fixed base rate | Baseline |
| **$M9_{\text{oracle}}$ Bayes oracle** | Analytical ground truth | DGP coefficients | $p^*_{\text{full}}(x)$ | Joint upper bound |

---

## 4. Prospective Research Hypotheses & Success Criteria

### Hypothesis H2.1 (Specialist Calibration Efficacy)
> On the unseen development split, post-hoc Platt calibration strictly reduces Brier score and ECE for both individual specialists:
> $$\text{Brier}(M2_{\text{cal}}) < \text{Brier}(M2_{\text{raw}}), \quad \text{Brier}(M3_{\text{cal}}) < \text{Brier}(M3_{\text{raw}})$$
> with mean $\text{ECE} \le 0.15$ across all training seeds.

### Hypothesis H2.2 (Log-Odds Fusion vs. Linear Ensembling)
> Because feature blocks $x_A$ and $x_B$ are conditionally independent given $x_s$, evidence accumulates additively in logit space.
> Calibrated log-odds fusion strictly outperforms both equal-weight and simplex linear probability averaging on unseen development data:
> $$\text{Brier}(M6_{\text{cal}}) < \text{Brier}(M5_{\text{cal}}) \le \text{Brier}(M4_{\text{cal}})$$

### Hypothesis H2.3 (System-Level Superiority over Baselines)
> The calibrated log-odds fusion system ($M6_{\text{cal}}$) outperforms the target-only baseline ($M0$) by at least $0.010$ Brier on development data:
> $$\text{Brier}(M0) - \text{Brier}(M6_{\text{cal}}) \ge 0.010$$
> and achieves Brier performance within $0.010$ of the capacity-matched pooled baseline ($M1_{\text{cap}}$).

### Hypothesis H2.4 (Attribution to Complementary Information)
> In the redundant negative control (where both specialists observe $x_A$), combining specialists produces only variance reduction.
> The gain of log-odds fusion over target-only is strictly greater in the positive control than in the negative control:
> $$\Delta \text{Brier}_{\text{pos}}(M6_{\text{cal}}, M0) - \Delta \text{Brier}_{\text{neg}}(M6_{\text{cal}}, M0) > 0$$

---

## 5. Execution & Compute Constraints

- Hardware: CPU-only execution.
- PyTorch thread limit: strictly capped at **2 threads** (`torch.set_num_threads(2)`).
- Model architecture: `SmallMLP` (1,857 params), `CapacityMatchedMLP` (2,199 params), `CoLoRAMLP` (2,193 params).
- Training schedule: AdamW, lr=0.005, wd=0.01, 100 epochs, batch size 32.
- Memory limit: $< 400$ MB RAM.
- Held-out test set: **strictly untouched** until final design audit.

---

## 6. Confirmatory Statistical Analysis Plan (SAP)

### 6.1 Sample Size and Quarantined Data
- Confirmatory evaluation is conducted strictly on the quarantined held-out test split ($N=512$).
- Ground truth test labels remain unread and unblinded until the execution of Stage 4.

### 6.2 Primary Confirmatory Hypothesis (H2.3-Confirmatory)
- **Comparison:** Calibrated Log-Odds Fusion ($M6_{\text{cal}}$) vs. Target-Only ($M0$).
- **Metric:** Brier Score reduction: $\Delta \text{Brier} = \text{Brier}(M0) - \text{Brier}(M6_{\text{cal}})$.
- **Effect-Size Criterion:** Point estimate $\widehat{\Delta \text{Brier}} \ge 0.010$.
- **Statistical Superiority Criterion:** Two-sided $\alpha = 0.05$ (equivalent to 95% bootstrap CI lower bound $> 0.0$).
- **Confirmatory Confirmation Requirement:** Both effect-size success ($\ge 0.010$) AND statistical superiority (95% CI $> 0.0$) must be satisfied simultaneously.

### 6.3 Secondary Confirmatory Family & Multiplicity Adjustment
A prespecified family of four secondary confirmatory contrasts is evaluated with step-down Holm-Bonferroni correction ($\alpha = 0.05$):
1. **$C_1$ (Log-Odds vs. Equal-Weight):** $\text{Brier}(M4_{\text{cal}}) - \text{Brier}(M6_{\text{cal}}) > 0.0$
2. **$C_2$ (Log-Odds vs. Simplex-Weight):** $\text{Brier}(M5_{\text{cal}}) - \text{Brier}(M6_{\text{cal}}) > 0.0$
3. **$C_3$ (Log-Odds vs. Capacity-Matched Pooled):** $\text{Brier}(M1_{\text{cap}}) - \text{Brier}(M6_{\text{cal}}) > 0.0$
4. **$C_4$ (Matched Diagnostic Contrast):** $\Delta \text{Brier}_{\text{pos}}(M0, M6_{\text{cal}}) - \Delta \text{Brier}_{\text{neg}}(M0, M6_{\text{cal}}) > 0.0$

### 6.4 Paired Instance Resampling Design
- **Cluster Unit:** Evaluation instances $i \in \{0, \dots, N-1\}$ ($N=512$).
- **Bootstrap Replicates:** $B = 2,000$ paired instance resamples with replacement, using fixed seed `bootstrap_seed = 20261012`.
- **Exact Pairing:** Each bootstrap draw resamples the exact same instance index tuple across all methods and across all training seeds ($s \in \{42, 123, 777\}$).
- **Seed Treatment:** For each bootstrap draw $b$, loss is evaluated for each seed $s$ and averaged across seeds prior to computing contrast differences.
- **Inference:** 95% percentile bootstrap confidence intervals $[q_{0.025}, q_{0.975}]$ and two-sided empirical bootstrap $p$-values.

