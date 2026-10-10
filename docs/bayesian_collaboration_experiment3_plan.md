# Experiment 3 Research Plan: Bayesian Uncertainty-Aware Specialist Fusion

**Repository:** `mbzuaiciao/distributed-portfolio-learning`  
**Branch:** Prospective (to be branched as `experiment/bayesian-collaboration-uncertainty-fusion`)  
**Date Drafted:** 2026-10-10 12:07 PDT  
**Status:** **PROSPECTIVE DESIGN ONLY — AWAITING SEPARATE USER APPROVAL BEFORE EXECUTION**

---

## 1. Scientific Motivation & Problem Statement

Experiment 2 confirmed that:
1. Post-hoc calibration cures specialist overconfidence under partial feature observability.
2. Under conditional independence, log-odds evidence accumulation statistically significantly outperforms linear probability averaging ($p_{\text{holm}} < 0.05$).
3. Decoupled specialist learning decisively outperforms monolithic capacity-matched pooling ($+0.0429$ Brier gain, $p_{\text{holm}} = 0.002$).

However, Experiment 2 employed **static, global fusion weights** ($w_A, w_B$) fitted across the entire feature space. In Experiment 2's DGP, feature support was homogeneous across all samples, so fixed weights were globally optimal.

**The Central Question for Experiment 3:**  
Does dynamic, instance-level **Bayesian epistemic uncertainty weighting** provide measurable, incremental predictive value over calibrated static log-odds fusion when specialist expertise is **regionally localized**?

---

## 2. Formal Definition & Validation of Epistemic Uncertainty

### 2.1 Distinguishing Aleatoric from Epistemic Uncertainty
- **Aleatoric Uncertainty (Data Entropy):** Irreducible Bayes error arising from overlapping class densities:
  $$H_{\text{aleatoric}}(x) = -p^*(x) \log p^*(x) - (1 - p^*(x)) \log (1 - p^*(x))$$
  Aleatoric uncertainty is maximal near the true decision boundary ($p^* \approx 0.5$), regardless of training data abundance.
- **Epistemic Uncertainty (Parameter / Subspace Uncertainty):** Reducible uncertainty stemming from sparsity or absence of specialist training observations in feature region $x$:
  $$\sigma_{\text{epistemic}}^2(x) = \text{Var}_{\theta \sim p(\theta \mid \mathcal{D})}[f_\theta(x)] = \phi(x)^T \Sigma_\theta \phi(x)$$
  where $\phi(x)$ is the representation vector and $\Sigma_\theta$ is the posterior parameter covariance matrix (via Laplace approximation or Bayesian linear readout).

### 2.2 Validation of the Epistemic Uncertainty Metric
Prior to model combination, the epistemic uncertainty estimator $\sigma_{\text{epistemic}}^2(x)$ must be validated against two pre-specified sanity checks:
1. **Covariate Distance Monotonicity:** On synthetic validation samples, $\sigma_{\text{epistemic}}^2(x)$ must exhibit positive rank correlation ($r_{\text{Spearman}} > 0.70$) with the Mahalanobis distance from the specialist's training distribution:
   $$d_M(x, \mu_{\text{train}}, \Sigma_{\text{train}}) = \sqrt{(x - \mu_{\text{train}})^T \Sigma_{\text{train}}^{-1} (x - \mu_{\text{train}})}$$
2. **Independence from Aleatoric Ambiguity:** In in-distribution regions near the decision boundary ($|f(x)| < 0.1$), epistemic uncertainty must remain low ($\sigma_{\text{epistemic}}^2(x) \ll \sigma_{\text{OOD}}^2$), verifying that the estimator does not conflate boundary proximity with lack of knowledge.

---

## 3. Data-Generating Process with Controlled Regional Specialization

To create an environment where instance-level uncertainty weighting can provide incremental value, Experiment 3 introduces a **localized covariate-shift DGP**:

### 3.1 Feature Structure & Regional Partitioning
- Features: $x = [x_{\text{shared}}, x_A, x_B, x_{\text{noise}}] \in \mathbb{R}^{20}$.
- Partitioning Variable: Let $s = x_{\text{shared}, 1}$.
  - **Region 1 ($s > 0$):** Specialist A has abundant training observations ($N=256$), while Specialist B has sparse observations ($N=16$ or zero).
  - **Region 2 ($s \le 0$):** Specialist B has abundant training observations ($N=256$), while Specialist A has sparse observations ($N=16$ or zero).
- **Target Distribution:** The target evaluation split samples uniformly across both Region 1 and Region 2.

### 3.2 Theoretical Mechanism of Expected Advantage
- Under Region 1, Specialist A is reliable ($\sigma_{A}^2(x) \approx 0$), whereas Specialist B is operating out-of-distribution with high epistemic variance ($\sigma_{B}^2(x) \gg 0$).
- Under Region 2, the reverse holds: Specialist B is reliable, whereas Specialist A is operating out-of-distribution.
- A **static log-odds combiner** with fixed weights $(w_A, w_B)$ is forced to adopt compromise weights ($w_A \approx w_B$), contaminating Region 1 predictions with Specialist B's noise and Region 2 predictions with Specialist A's noise.
- An **uncertainty-aware Bayesian combiner** dynamically computes precision weights:
  $$w_A(x) = \frac{1 / \sigma_A^2(x)}{1 / \sigma_A^2(x) + 1 / \sigma_B^2(x)}, \quad w_B(x) = \frac{1 / \sigma_B^2(x)}{1 / \sigma_A^2(x) + 1 / \sigma_B^2(x)}$$
  automatically delegating prediction to the regionally competent specialist.

---

## 4. Method Matrix & Fair Baseline Architecture

All models receive identical input features and observation masks. Parameter counts and compute budgets are matched fairly.

| Method ID | Model Description | Combination Rule | Uncertainty Mechanism | Budget / Notes |
|---|---|---|---|---|
| **M0** | Target-Only MLP | Direct prediction | None | Standard MLP ($1,857$ params) |
| **M1_cap** | Capacity-Matched Pooled | Direct prediction | None | Standard MLP ($2,199$ params) |
| **M6_cal** | Static Log-Odds Fusion | $\sigma(w_0 + w_A \ell_A + w_B \ell_B)$ | None (Fixed weights from Exp 2) | Static global combiner |
| **M_conf** | Confidence-Gated Fusion | Weights gated by $|p - 0.5|$ | Predictive confidence (heuristic) | Non-Bayesian baseline |
| **M_MoE** | Mixture-of-Experts Gating | Gating network $w(x_{\text{shared}})$ | Deterministic input-conditioning | MLP gating network |
| **M_Bayes** | **Bayesian Uncertainty Fusion** | Dynamic precision weights $w(x) \propto 1/\sigma^2(x)$ | Laplace / Bayesian top-layer variance | **Primary Candidate Method** |
| **M9_oracle** | Regional Bayes Oracle | Optimal regional ground truth | Full conditional Bayes upper bound | Oracle reference |

---

## 5. Statistical Hypotheses & Success / Failure Criteria

### 5.1 Primary Confirmatory Hypothesis (H3.1: Incremental Bayesian Value)
- **Comparison:** Bayesian Uncertainty Fusion ($M_{\text{Bayes}}$) vs. Static Log-Odds Fusion ($M6_{\text{cal}}$).
- **Metric:** Brier Score reduction on regional-shift test set:
  $$\Delta \text{Brier} = \text{Brier}(M6_{\text{cal}}) - \text{Brier}(M_{\text{Bayes}})$$
- **Effect-Size Success Criterion:** $\widehat{\Delta \text{Brier}} \ge 0.015$.
- **Statistical Superiority Criterion:** 95% paired cluster bootstrap CI lower bound $> 0.0$ ($p < 0.05$).
- **Combined Success:** Both effect-size and statistical superiority must hold simultaneously.

### 5.2 Secondary Confirmatory Hypotheses (Holm-Adjusted)
1. **$C_{3.1}$ (Bayesian vs. Confidence Gating):** $M_{\text{Bayes}}$ strictly outperforms confidence-heuristic gating ($M_{\text{conf}}$): $\Delta \text{Brier} > 0.0, p_{\text{holm}} < 0.05$.
2. **$C_{3.2}$ (Bayesian vs. Deterministic MoE):** $M_{\text{Bayes}}$ matches or outperforms input-conditioned MoE gating ($M_{\text{MoE}}$) without requiring end-to-end gating supervision.
3. **$C_{3.3}$ (Region-Specific Delegation):** On Region 1 test instances, $w_A(x) > 0.80$ for $>90\%$ of samples; on Region 2 test instances, $w_B(x) > 0.80$ for $>90\%$ of samples.

### 5.3 Explicit Falsification / Failure Criteria
- **Falsification 1:** If static log-odds fusion ($M6_{\text{cal}}$) performs within $0.005$ Brier of $M_{\text{Bayes}}$, Bayesian uncertainty weighting provides negligible incremental value.
- **Falsification 2:** If confidence heuristic gating ($M_{\text{conf}}$) or deterministic MoE matches $M_{\text{Bayes}}$, epistemic uncertainty is redundant with standard predictive confidence.

---

## 6. Data Hygiene & Split Architecture

To prevent data contamination, Experiment 3 requires **completely fresh data draws**:
- **New PRNG Seeds:** DGP configuration `dgp_seed = 20261014`, partition split `data_seed = 20261015`.
- **Absolute Prohibition:** The held-out test split from Experiment 2 ($N=512$) will **not be loaded, queried, or reused** for any model selection or training in Experiment 3.
- **5-Split Structure:**
  1. Specialist Training Splits: Target ($N=128$), Spec A ($N=256$, Region 1 bias), Spec B ($N=256$, Region 2 bias)
  2. Calibration Split ($N=128$): Dedicated for post-hoc Platt scaling
  3. Gating / Fusion Split ($N=128$): Dedicated for fitting fusion baselines
  4. Development Split ($N=128$): For exploratory validation and gate checks
  5. Quarantined Confirmatory Test Split ($N=512$): Strictly locked until final audit

---

## 7. Resource Estimate & Pre-Execution Approval Gate

- **Hardware:** CPU-only, strictly limited to 2 PyTorch threads (`torch.set_num_threads(2)`).
- **Compute Operations:** Training 18 base MLPs + top-layer Laplace covariance estimation + bootstrap testing.
- **Estimated Wall-Clock Time:** $25 - 40$ seconds total.
- **Peak RAM Estimate:** $< 350$ MB.
- **Approval Gate:** **NO CODE EXECUTION OR TRAINING WILL OCCUR UNTIL FORMAL USER APPROVAL IS PROVIDED.**
