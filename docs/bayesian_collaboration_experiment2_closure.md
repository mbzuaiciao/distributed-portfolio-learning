# Experiment 2 Closure Document: Specialist Calibration & Evidence Fusion

**Repository:** `mbzuaiciao/distributed-portfolio-learning`  
**Branch:** `experiment/bayesian-collaboration-calibration-fusion`  
**Closure Commit:** `4a87710d`  
**Date Closed:** 2026-10-10 12:06 PDT  
**Protocol Document:** [`docs/bayesian_collaboration_experiment2_protocol.md`](bayesian_collaboration_experiment2_protocol.md)  
**Confirmatory Report:** [`artifacts/bayesian_collab_exp2/confirmatory_report_exp2.md`](../artifacts/bayesian_collab_exp2/confirmatory_report_exp2.md)  
**Status:** **EXPERIMENT 2 COMPLETED & FROZEN** (No further runs or parameter alterations permitted).

---

## 1. Confirmatory Evaluation Outcomes Summary

Experiment 2 tested whether post-hoc specialist calibration and log-odds evidence fusion resolve the empirical bottlenecks of Experiment 1 under a strict 5-split data hygiene architecture. The confirmatory evaluation was conducted on a quarantined held-out test split ($N=512$) with paired cluster bootstrap inference ($B=2,000$ resamples, `seed=20261012`), preserving exact pairing across models and training seeds ($42, 123, 777$).

### 1.1 Primary Confirmatory Hypothesis (H2.3)
- **Contrast:** Calibrated Log-Odds Fusion ($M6_{\text{cal}}$) vs. Target-Only ($M0$).
- **Observed Brier Gain:** $\mathbf{+0.0187}$ Brier (positive across all three seeds: $+0.0149, +0.0267, +0.0144$).
- **Pre-Registered Effect Size Threshold:** $\ge 0.010$ Brier (**Passed**).
- **95% Paired Cluster Bootstrap CI:** **[-0.0089, +0.0486]** ($p = 0.190$).
- **Confirmatory Decision:** **INCONCLUSIVE / STATISTICALLY UNCONFIRMED**.
  - While the substantive effect size exceeded the pre-registered threshold ($+0.0187 \ge 0.010$), the confidence interval fails to exclude zero at $\alpha=0.05$.
  - In accordance with our pre-registered criteria requiring simultaneous effect size and statistical superiority passes, this hypothesis is reported as statistically unconfirmed without changing thresholds post-hoc.

### 1.2 Statistically Confirmed Secondary Hypotheses (Holm-Adjusted)
All three model-comparison hypotheses in the secondary confirmatory family passed with statistical significance after step-down Holm-Bonferroni correction:

1. **$C_1$: Calibrated Log-Odds vs. Equal-Weight Linear Averaging ($M6_{\text{cal}}$ vs. $M4_{\text{cal}}$):**
   - **Gain:** $\mathbf{+0.0093}$ Brier
   - **95% Paired Bootstrap CI:** $[+0.0022, +0.0162]$
   - **Holm-Adjusted $p$-value:** **$p_{\text{holm}} = 0.039 < 0.05$**
   - **Status:** **CONFIRMED SUPERIOR**.
2. **$C_2$: Calibrated Log-Odds vs. Simplex Linear Averaging ($M6_{\text{cal}}$ vs. $M5_{\text{cal}}$):**
   - **Gain:** $\mathbf{+0.0082}$ Brier
   - **95% Paired Bootstrap CI:** $[+0.0012, +0.0148]$
   - **Holm-Adjusted $p$-value:** **$p_{\text{holm}} = 0.042 < 0.05$**
   - **Status:** **CONFIRMED SUPERIOR**.
3. **$C_3$: Calibrated Log-Odds vs. Capacity-Matched Pooled Baseline ($M6_{\text{cal}}$ vs. $M1_{\text{cap}}$):**
   - **Gain:** $\mathbf{+0.0429}$ Brier
   - **95% Paired Bootstrap CI:** $[+0.0204, +0.0677]$
   - **Holm-Adjusted $p$-value:** **$p_{\text{holm}} = 0.002 < 0.05$**
   - **Status:** **CONFIRMED SUPERIOR**.

### 1.3 Inconclusive Matched Complementarity Diagnostic ($C_4$)
- **Point Contrast:** $\Delta_{\text{contrast}} = \text{Gain}_{\text{positive}} - \text{Gain}_{\text{redundant}} = +0.0187 - (-0.0047) = \mathbf{+0.0234}$.
- **95% Bootstrap CI:** $[-0.0181, +0.0671]$ ($p_{\text{holm}} = 0.274$).
- **Status:** **INCONCLUSIVE**.
- The point estimate indicates greater collaborative benefit under complementary features than under redundant features, but its confidence interval spans zero, confirming its pre-registered status as a descriptive diagnostic rather than an isolated causal test.

---

## 2. Key Scientific Findings & Tradeoffs

### 2.1 The AUROC-versus-Calibration Tradeoff
An important empirical insight from Experiment 2 is the divergence between discrimination (AUROC) and probability calibration (Brier / ECE):
- **Target-Only MLP ($M0$):**
  - Observed AUROC: **0.8432**
  - Observed Brier: **0.1913**
  - Observed ECE: **0.1857**
- **Calibrated Log-Odds Fusion ($M6_{\text{cal}}$):**
  - Observed AUROC: **0.7827**
  - Observed Brier: **0.1727**
  - Observed ECE: **0.0397**

**Mechanism:**  
Target-only training on full features ($x_s, x_A, x_B$) enables the network to learn global feature interactions that establish strong rank-ordering discrimination (high AUROC). However, unregularized BCE produces overconfident logits, leading to poor probability calibration (high ECE).  
Conversely, individual specialists observe restricted feature subsets ($x_s, x_A$ or $x_s, x_B$), bounding their individual AUROCs at $0.6892$ and $0.7424$. While post-hoc Platt scaling and log-odds fusion optimally calibrate probabilities in logit space (slashing ECE to $0.0397$ and Brier to $0.1727$), monotonic log-odds combination cannot recover ranking distinctions that were lost during factorized feature observation.

### 2.2 Mechanism of Evidence Accumulation
- Calibrated log-odds fusion demonstrated statistically significant superiority over linear ensembling ($p_{\text{holm}} < 0.05$).
- Under conditional independence of feature partitions, Bayes' rule operates additively in logit space:
  $$\text{logit}(P(Y=1 \mid X_A, X_B)) = \text{logit}(P(Y=1)) + \Delta \text{logit}(X_A) + \Delta \text{logit}(X_B)$$
- Linear ensembling $p = w_A p_A + w_B p_B$ is mathematically constrained to $[\min(p_A, p_B), \max(p_A, p_B)]$, artificially compressing joint certainty and inflating Brier error on confident cases.

### 2.3 Functional Redundancy of Separate Platt Scaling in Log-Odds Fusion
- Bivariate log-odds fusion fits $z = w_0 + w_A \text{logit}(p_A) + w_B \text{logit}(p_B)$.
- Because univariate Platt scaling is an affine transformation of logits ($a \cdot \text{logit} + b$), bivariate logistic regression directly over raw logits has the identical expressive capacity to bivariate logistic regression over Platt-scaled logits.
- While individual Platt calibration is critical for linear ensembling (improving Brier by ~0.017), it is functionally redundant for log-odds fusion.

---

## 3. Methodological Limitations

1. **Pooled Baseline Architecture:**
   - Both naive pooled ($M1$) and capacity-matched pooled ($M1_{\text{cap}}$) baselines underperformed ($0.2219$ and $0.2156$ Brier).
   - They were trained on mixed-mask data where 80% of samples had feature blocks zero-filled. Without explicit task-ID tokens or sample reweighting, standard BCE forced shared hidden layers to balance conflicting marginal objectives, causing representation confusion when evaluated on full features.
2. **Synthetic Data-Generating Process:**
   - The synthetic DGP relied on Gaussian feature blocks with linear logistic link functions. It lacked localized non-linearities, heteroskedastic noise, covariate shift, or region-specific specialist expertise.
3. **Statistical Power against Target-Only:**
   - At $N=512$, the instance error variance between target-only and factorized log-odds fusion ($\text{SE} = 0.0146$) was too large to confirm the $+0.0187$ gain at $\alpha = 0.05$. Detecting this difference with statistical significance would require $N \approx 1,200$ test observations.
4. **Boundary of Inquiry:**
   - Experiment 2 investigated specialist calibration and deterministic log-odds fusion on synthetic data.
   - **No claims are made regarding established Bayesian uncertainty weighting, epistemic reliability, or real-world financial portfolio performance.**

---

## 4. Frozen Artifact Provenance

All Experiment 2 artifacts are permanently preserved and versioned at commit `4a87710d`:
- Configuration: `artifacts/bayesian_collab_exp2/config_exp2.json`
- Model Checkpoints: 36 frozen `.pt` files verified via sha256 in `frozen_exp2.json`
- Development Evaluation: `development_report_exp2.md`, `development_summary_exp2.json`, `development_metrics_exp2.csv`
- Pre-Confirmatory Audit: `pre_confirmatory_audit_exp2.md`
- Confirmatory Evaluation: `confirmatory_report_exp2.md`, `confirmatory_summary_exp2.json`, `confirmatory_metrics_exp2.csv`, `confirmatory_evaluation_report_exp2.json`

Experiment 2 is formally concluded.
