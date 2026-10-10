# Exploratory Diagnostic Follow-Up Plan: Experiment 1 Gate Analysis

**Branch:** `experiment/bayesian-collaboration-diagnostic-followup`  
**Date:** 2026-10-10  
**Parent Branch:** `experiment/bayesian-collaboration-positive-controls` (frozen at `efc4ca99`)  
**Status:** EXPLORATORY PRE-SPECIFICATION — specifies hypotheses and diagnostic metrics before execution.

---

## 1. Context & Motivation

In Experiment 1, the pre-registered development gate H2 failed because single specialists trained on
partial feature blocks failed to achieve Brier $< 0.19374$ on the development set ($N=64$).
As required by research integrity protocols, Stage 4 (held-out test set) was locked out, and all frozen
checkpoints and results were preserved.

This exploratory follow-up investigates four targeted scientific questions arising from Experiment 1's development results:
1. **Q1: Specialist B underperformance:** Why did Specialist B exhibit high Brier ($0.3232$) and high ECE ($0.377$) despite holding positive predictive information ($\Delta_{\text{oracle}} = 0.0419$)?
2. **Q2: Redundant vs. Complementary comparison:** Why did the ensemble of redundant specialists achieve lower development Brier ($0.1627$) than the ensemble of complementary specialists ($0.2023$)?
3. **Q3: CoLoRA vs. Capacity-matched pooled gap:** Why did structural CoLoRA ($0.2562$) underperform the capacity-matched pooled baseline ($0.2166$)?
4. **Q4: Non-linear complementarity exploitation:** Does simple linear probability averaging fail to fully exploit the synthetic DGP's complementary information compared to log-odds combinations and the Bayes oracle?

---

## 2. Research Data Protocol: Fresh Diagnostic Split

To avoid overfitting to or repeatedly adapting on the original development split ($N=64$), all diagnostic
analyses use a **fresh, independent diagnostic split** ($N_{\text{diag}} = 256$) sampled from the frozen DGP
with independent seed:
$$\text{data\_seed} = \text{cfg.seed} + 999 = 20262008$$

### Test Set Isolation Guarantee
The confirmatory held-out test split (`test_x_full`, `test_x_A`, `test_x_B`, `test_y`) remains **strictly unread and unblinded**.
No code in this investigation accesses test labels or artifacts.

---

## 3. Pre-Specified Exploratory Hypotheses

### Hypothesis EH1 (Specialist B Overconfidence & Calibration)
> Single specialists trained with unregularized BCE on partial feature blocks become overconfident:
> because the unobserved block (e.g. $x_A$ for Specialist B) is missing from the input, the model pushes
> output probabilities to $\approx 0$ or $\approx 1$ based on incomplete signal. When an unobserved feature
> alters the true outcome, extreme overconfidence incurs maximum quadratic error ($(1 - 0)^2 = 1$).
> Temperature scaling or Platt calibration on validation data will significantly reduce Specialist B's Brier score
> and ECE without altering its discrimination (AUROC).

- **Comparison Metrics:** Raw Brier vs. Platt-calibrated Brier; ECE; Sanders Brier decomposition (Reliability vs. Resolution).
- **Expected Outcome:** Reliability error accounts for $\ge 40\%$ of Specialist B's Brier score; calibration brings Brier below $0.22$.

### Hypothesis EH2 (Ensemble Dynamic Range Compression)
> Linear probability averaging ($0.5 p_A + 0.5 p_B$) compresses the dynamic range of joint predictions:
> it is mathematically bounded by $[\min(p_A, p_B), \max(p_A, p_B)]$ and cannot output high joint certainty ($> 0.90$)
> even when both specialists observe agreeing evidence. In contrast, in the redundant regime, averaging two estimates
> of the same feature block $x_A$ acts purely as variance reduction on a single linear predictor without dynamic-range compression.
> Furthermore, in the complementary regime, Specialist B's calibration error dragged down the linear average.

- **Comparison Metrics:** Dynamic range of predictions ($\max - \min$, variance of predictions); Brier decomposition of $M4_{\text{pos}}$ vs. $M4_{\text{neg}}$.
- **Expected Outcome:** Redundant ensemble has lower reliability error than complementary linear ensemble; log-odds combination restores dynamic range and surpasses redundant ensemble.

### Hypothesis EH3 (CoLoRA Gradient Bottleneck & Interference)
> Structural CoLoRA underperforms the capacity-matched pooled baseline due to two architectural factors:
> 1. **Rank-4 bottleneck:** Only 4 dimensions can pass through $\Delta W = \frac{\alpha}{r} B \Lambda_{\text{task}} A$,
>    constraining task-specific adaptations relative to the full-rank dense projection in `CapacityMatchedMLP` (width 38).
> 2. **Shared base gradient interference:** The shared base weight $W_{\text{base}}$ receives gradient updates from
>    tasks with opposing mask structures (Task 1 has $x_B=0$, Task 2 has $x_A=0$), creating conflicting gradient directions.

- **Comparison Metrics:** Training loss trajectory; cosine similarity between task gradients on $W_{\text{base}}$; rank sensitivity ($r=4$ vs. $r=8$).
- **Expected Outcome:** Task gradients on $W_{\text{base}}$ exhibit negative or low cosine similarity; increasing rank reduces the deficit against capacity-matched pooled.

### Hypothesis EH4 (Log-Odds Combination vs. Joint Oracle)
> Under conditional independence of feature blocks $x_A$ and $x_B$ given $x_s$, the Bayes-optimal combination
> of specialist predictions is additive in log-odds space ($\text{logit}(p_{\text{joint}}) \approx \text{logit}(p_A) + \text{logit}(p_B) + c$),
> not probability space. A log-odds combination rule will strictly outperform linear probability averaging
> and approach the analytical joint Bayes oracle $p^*_{\text{full}}$.

- **Comparison Metrics:** Brier score on fresh diagnostic split across:
  1. Equal probability mixture: $0.5 p_A + 0.5 p_B$
  2. Simplex probability mixture: $w_A p_A + w_B p_B$
  3. Log-odds logistic combination: $\sigma(\alpha_0 + \alpha_A \text{logit}(p_A) + \alpha_B \text{logit}(p_B))$
  4. Analytical joint Bayes oracle: $p^*_{\text{full}}$
- **Expected Outcome:** Brier(Log-odds) $<$ Brier(Simplex) $<$ Brier(Equal); Log-odds combination recovers $\ge 60\%$ of the gap to the Bayes oracle.

---

## 4. Execution Protocol & Safeguards

- **Compute Environment:** Local CPU only.
- **Thread Limit:** PyTorch capped at 2 CPU threads (`torch.set_num_threads(2)`).
- **Target Wall Time:** $< 30$ seconds.
- **Target Peak Memory:** $< 400$ MB RAM.
- **Unit Testing:** Deterministic tests verify log-odds combination and calibration routines before running diagnostic scripts.
