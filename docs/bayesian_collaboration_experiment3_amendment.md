# Experiment 3 Protocol Amendment: Rigorous Baselines, Uncertainty Ablation, & Deterministic Selection

**Repository:** `mbzuaiciao/distributed-portfolio-learning`  
**Branch:** `experiment/bayesian-collaboration-uncertainty-routing`  
**Date Frozen:** 2026-10-10 12:15 PDT  
**Parent Plan:** [`docs/bayesian_collaboration_experiment3_plan.md`](bayesian_collaboration_experiment3_plan.md)  
**Status:** **PROSPECTIVE AMENDMENT — PRE-REGISTERED BEFORE MODEL TRAINING**

---

## 1. Motivation & Scope of Amendment

This amendment incorporates key scientific clarifications into the Experiment 3 research design:
1. Prevents any artificial handicap of deterministic baselines by equipping both **linear and non-linear deterministic Mixture-of-Experts (MoE)** routers.
2. Introduces an **uncertainty-feature ablation** ($M_{\text{Router-NoUQ}}$ vs. $M_{\text{Router-UQ}}$) to isolate whether Bayesian epistemic variance provides incremental information beyond representation features alone.
3. Formally pre-specifies the **deterministic selection and tie-breaking rule** for the strongest non-Bayesian baseline before observing development data.
4. Pre-specifies numerical conditioning and bias handling in Bayesian covariance and inverse-variance weighting.

---

## 2. Strong Deterministic Mixture-of-Experts Baselines

To ensure deterministic routing is not artificially handicapped, Experiment 3 implements two distinct MoE routers:

### 2.1 Linear MoE Router ($M_{\text{MoE-Lin}}$)
- **Architecture:** Multinomial logistic gating network on shared features $x_s \in \mathbb{R}^{d_s}$:
  $$g(x_s) = \text{softmax}(W_g x_s + b_g), \quad W_g \in \mathbb{R}^{2 \times d_s}, \; b_g \in \mathbb{R}^2$$
- **Parameter Count:** $(d_s + 1) \times 2 = 10$ parameters (for $d_s=4$).
- **Role:** Optimal for axis-aligned linear separations (Regime A).

### 2.2 Non-Linear MoE Router ($M_{\text{MoE-MLP}}$)
- **Architecture:** 2-layer MLP gating network with hidden representations:
  $$h(x_s) = \text{ReLU}(W_1 x_s + b_1), \quad g(x_s) = \text{softmax}(W_2 h(x_s) + b_2)$$
  where $W_1 \in \mathbb{R}^{16 \times d_s}$, $W_2 \in \mathbb{R}^{2 \times 16}$.
- **Parameter Count:** $4 \times 16 + 16 + 16 \times 2 + 2 = 114$ parameters.
- **Regularization:** L2 weight decay $\lambda = 0.01$ and early stopping to prevent overfitting on the $N=128$ fusion split.
- **Role:** High expressive capacity capable of learning non-linear, multi-modal boundaries (Regime B).

---

## 3. Matched Uncertainty-Feature Ablation

To isolate whether Bayesian epistemic uncertainty provides true informational value or whether performance gains are simply due to additional router capacity, we implement a **strictly matched ablation pair**:

### 3.1 Architecture Specification
Both routers share the identical 2-layer MLP architecture (hidden dimension 16, ReLU activation, Sigmoid output, identical AdamW learning rate $0.005$, weight decay $0.01$, and batch size $32$ for 100 epochs on the $N=128$ fusion split):

1. **Ablation Baseline ($M_{\text{Router-NoUQ}}$):**
   - **Inputs:** Specialist logits and shared features:
     $$\mathbf{z}_{\text{NoUQ}} = [\ell_A, \ell_B, x_{s, 1}, \dots, x_{s, d_s}] \in \mathbb{R}^{2 + d_s}$$
2. **Bayesian-Augmented Router ($M_{\text{Router-UQ}}$):**
   - **Inputs:** Specialist logits, shared features, **plus Bayesian epistemic variances**:
     $$\mathbf{z}_{\text{UQ}} = [\ell_A, \ell_B, \sigma_{A,\text{epistemic}}^2, \sigma_{B,\text{epistemic}}^2, x_{s, 1}, \dots, x_{s, d_s}] \in \mathbb{R}^{4 + d_s}$$

### 3.2 Hypothesis A3 (Uncertainty Feature Attribution)
- Contrast: $\Delta \text{Brier}_{\text{ablation}} = \text{Brier}(M_{\text{Router-NoUQ}}) - \text{Brier}(M_{\text{Router-UQ}})$.
- A positive, statistically significant contrast demonstrates that Bayesian epistemic uncertainty inputs provide predictive signal that cannot be extracted from raw shared features and logits alone.

---

## 4. Bayesian Covariance Conditioning & Error Independence

### 4.1 Numerical Conditioning of Laplace Covariance
For each specialist $i \in \{A, B\}$, let $\phi_i(x) \in \mathbb{R}^{16}$ denote the penultimate representation. The Hessian of the negative log-likelihood is:
$$H_i = \sum_{n=1}^{N_i} \hat{p}_{i,n} (1 - \hat{p}_{i,n}) \phi_i(x_n) \phi_i(x_n)^T$$
To guarantee strict numerical stability:
1. **Adaptive Damping:** Regularized inverse $\Sigma_i = (H_i + \lambda I)^{-1}$ with prior precision $\lambda = 1.0$.
2. **Eigenvalue Conditioning Check:** Compute condition number $\kappa = \lambda_{\max}(\Sigma_i) / \lambda_{\min}(\Sigma_i)$. If $\kappa > 10^5$, clip eigenvalues at floor $\epsilon = 10^{-4} \lambda_{\max}$.
3. **Epistemic Variance Formulation:** $\sigma_i^2(x) = \phi_i(x)^T \Sigma_i \phi_i(x) \ge 0$.

### 4.2 Relaxation of Unbiased/Independent Error Assumptions
Standard inverse-variance weighting $w_i \propto 1/\sigma_i^2$ assumes estimators are unbiased and uncorrelated. To avoid this restrictive assumption in real classifiers:
- Specialist inputs are first **Platt-calibrated** on the calibration split to ensure zero mean logit bias.
- In $M_{\text{Bayes-Learn}}$, the fusion model includes a learned intercept and cross-term:
  $$z = w_0 + w_A(x) \ell_A(x) + w_B(x) \ell_B(x)$$
  where weights are parameterized via softplus of precision plus learned affine adjustments, explicitly absorbing correlation and scale mismatches.

---

## 5. Pre-Registered Deterministic Baseline Selection Rule

To prevent cherry-picking the non-Bayesian comparator after inspecting held-out test data, the comparator $M_{\text{strongest\_nonbayes}}$ is **deterministically frozen based on development performance**:

### 5.1 Eligible Candidate Pool
$$\mathcal{M}_{\text{nonbayes}} = \left\{ M0, M1_{\text{cap}}, M6_{\text{cal}}, M_{\text{conf}}, M_{\text{dens}}, M_{\text{MoE-Lin}}, M_{\text{MoE-MLP}}, M_{\text{Router-NoUQ}} \right\}$$

### 5.2 Deterministic Selection Hierarchy
1. **Primary Criterion:** Minimum mean Brier score on the development split:
   $$M^* = \arg\min_{M \in \mathcal{M}_{\text{nonbayes}}} \overline{\text{Brier}}_{\text{dev}}(M)$$
2. **Tie-Breaker 1:** If two methods have development Brier within $10^{-5}$, select the method with higher development AUROC:
   $$M^* = \arg\max_{M \in \text{Ties}} \overline{\text{AUROC}}_{\text{dev}}(M)$$
3. **Tie-Breaker 2:** If still tied, select according to pre-registered priority order (favoring the most expressive baseline):
   $$M_{\text{MoE-MLP}} \succ M_{\text{Router-NoUQ}} \succ M_{\text{MoE-Lin}} \succ M_{\text{dens}} \succ M_{\text{conf}} \succ M6_{\text{cal}} \succ M1_{\text{cap}} \succ M0$$

The chosen $M^*$ becomes the fixed comparator for the Primary Confirmatory Hypothesis on the held-out test split.

---

## 6. Pre-Registered Confirmatory Statistical Decision Rules

### 6.1 Primary Confirmatory Hypothesis (H3.1)
- **Comparison:** $M_{\text{Bayes-Learn}}$ vs. $M^*$.
- **Metric:** Brier score reduction on held-out test split ($N=512$):
  $$\Delta \text{Brier} = \text{Brier}(M^*) - \text{Brier}(M_{\text{Bayes-Learn}})$$
- **Effect-Size Threshold:** $\widehat{\Delta \text{Brier}} \ge 0.010$.
- **Statistical Superiority Criterion:** 95% paired cluster bootstrap CI lower bound $> 0.0$ ($p < 0.05$).
- **Success Standard:** Requires both effect-size $\ge 0.010$ and 95% CI $> 0.0$ simultaneously.

### 6.2 Secondary Confirmatory Family (Holm Step-Down, $\alpha = 0.05$)
1. **$C_{3.1}$ (Uncertainty Feature Ablation):** $M_{\text{Router-UQ}}$ vs. $M_{\text{Router-NoUQ}}$ ($\Delta \text{Brier} > 0$).
2. **$C_{3.2}$ (Bayes vs. Static Log-Odds):** $M_{\text{Bayes-Learn}}$ vs. $M6_{\text{cal}}$ ($\Delta \text{Brier} > 0$).
3. **$C_{3.3}$ (Bayes vs. Deterministic MoE):** $M_{\text{Bayes-Learn}}$ vs. $M_{\text{MoE-MLP}}$ ($\Delta \text{Brier} > 0$).
4. **$C_{3.4}$ (Bayes vs. Inverse-Variance Heuristic):** $M_{\text{Bayes-Learn}}$ vs. $M_{\text{Bayes-Inv}}$ ($\Delta \text{Brier} > 0$).

### 6.3 Inconclusive vs. Equivalence Rule (TOST)
- If $|\widehat{\Delta \text{Brier}}| < 0.005$ and CI spans zero, report as **statistically inconclusive**.
- Claiming equivalence requires Two One-Sided Tests (TOST) confirming 90% CI is bounded within $[-\delta, +\delta] = [-0.005, +0.005]$.
- If $M^*$ statistically beats $M_{\text{Bayes-Learn}}$, Bayesian routing is **falsified**.
