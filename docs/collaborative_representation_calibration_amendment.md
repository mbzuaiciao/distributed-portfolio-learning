# Prospective Amendment: Equal-Budget Collaborative Representation Calibration Diagnostic

**Document Version:** 1.0  
**Date:** 2026-10-10  
**Status:** **FROZEN PROSPECTIVE DESIGN (PRIOR TO IMPLEMENTATION & EXECUTION)**  
**Repository:** `distributed-portfolio-learning`  
**Branch:** `experiment/collaborative-representation-calibration`  
**Base Commit:** `cf44abfb`  
**Preceding Closure:** [`docs/collaborative_lora_transfer_closure.md`](collaborative_lora_transfer_closure.md)  
**Parent Plan:** [`docs/collaborative_lora_calibration_diagnostic_plan.md`](collaborative_lora_calibration_diagnostic_plan.md)  

---

## 1. Primary Statistical Units & Inferential Architecture

1. **Replications as Statistical Units:**
   - The diagnostic evaluates **$S = 10$ independent simulation replications** (random seeds $20261101, 20261102, \dots, 20261110$).
   - Each replication generates an entirely fresh synthetic dataset instance (source data, target train, target cal, target dev, and quarantined test).
   - Across-dataset inference treats each replication $s \in \{1, \dots, S\}$ as an independent statistical unit.

2. **Paired Metric Contrasts & Uncertainty Reporting:**
   - For any two models $A$ and $B$, the primary contrast is the paired difference in Brier score on the development split ($N_{\text{dev}}=128$):
     $$\Delta_s = \text{Brier}(M_A)_s - \text{Brier}(M_B)_s$$
   - We report:
     - Sample mean paired difference: $\bar{\Delta} = \frac{1}{S} \sum_{s=1}^S \Delta_s$
     - Standard error of the mean: $\text{SE}(\bar{\Delta}) = \frac{s_{\Delta}}{\sqrt{S}}$
     - 95% paired bootstrap confidence interval (1,000 resamples)
     - Paired two-sided Wilcoxon signed-rank test $p$-value
     - Adjusted significance thresholds via Holm-Bonferroni step-down correction across the family of primary comparisons.

---

## 2. Regularization, Model Selection, & Parameter Accounting

To prevent post-hoc hyperparameter optimization or calibration overfitting:

1. **Platt Scaling Regularization:**
   - For all 1D post-hoc calibrated models ($M_{0,\text{cal}}$, $M_{1,\text{cal}}$, $M_{\text{subspace},\text{cal}}$, $M_{\text{CoLoRA},\text{cal}}$):
     $$p_{\text{cal}}(x) = \sigma(a \cdot z(x) + b)$$
   - Fit strictly on the Target Calibration split ($N_{\text{cal}} = 64$) via $L_2$-regularized logistic regression with fixed penalty $\lambda_{\text{cal}} = 0.01$ (`C=100.0`, L-BFGS solver, max iterations 1000).
   - Parameter count: **2 trainable parameters** ($a \in \mathbb{R}$, $b \in \mathbb{R}$).

2. **Three-Source Calibrated Fusion Regularization:**
   - For $M_{\text{fuse}}$:
     $$\text{logit}(p_{\text{fuse}}(x)) = w_0 + \sum_{j=1}^3 w_j \cdot \text{logit}(p_j(x))$$
   - Fit strictly on Target Calibration ($N_{\text{cal}} = 64$) via $L_2$-regularized multivariate logistic regression with fixed penalty $\lambda_{\text{fuse}} = 0.1$ (`C=10.0`, L-BFGS solver).
   - Parameter count: **4 trainable parameters** ($w_0, w_1, w_2, w_3$).

3. **Fixed Budgets Without Model Tuning:**
   - No cross-validation or grid searching on validation splits is allowed for calibration or fusion. Both use deterministic convex optimization to exact convergence.

---

## 3. Strict Target-Label Access Accounting

All primary comparisons maintain **strictly matched target-label access**:

| Model ID | Method Description | Target Train ($N$) | Target Cal ($N$) | Total Target Labels Available | Feature Source |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **$M_{0,\text{raw}}$** | Target-Only Logistic | 64 | 0 | 64 | Target task only |
| **$M_{0,\text{cal}}$** | Target-Only Logistic + Platt | 64 | 64 | **128** | Target task only |
| **$M_{0,\text{pooled-128}}$** | Target-Only Logistic Retrained | 128 | 0 | **128** | Target task only |
| **$M_{1,\text{raw}}$** | Pooled Global LoRA | 64 | 0 | 64 | Source union + target |
| **$M_{1,\text{cal}}$** | Pooled Global LoRA + Platt | 64 | 64 | **128** | Source union + target |
| **$M_{\text{subspace},\text{raw}}$** | Shared-Subspace Multi-Task | 64 | 0 | 64 | Multi-task subspace |
| **$M_{\text{subspace},\text{cal}}$** | Shared-Subspace + Platt | 64 | 64 | **128** | Multi-task subspace |
| **$M_{\text{CoLoRA},\text{raw}}$** | Structural CoLoRA | 64 | 0 | 64 | Multi-task CoLoRA |
| **$M_{\text{CoLoRA},\text{cal}}$** | Structural CoLoRA + Platt | 64 | 64 | **128** | Multi-task CoLoRA |
| **$M_{\text{fuse}}$** | 3-Source Calibrated Fusion | 0 | 64 | **64** (cal only) | 3 source specialists |
| **$M_{\text{oracle}}$** | Analytical Bayes Oracle | 0 | 0 | 0 | Analytical DGP |

Notice that $M_{\text{fuse}}$ operates with **half** the total target labels ($N=64$ calibration labels, 0 training labels) compared to the 128 total target labels available to $M_{\text{CoLoRA},\text{cal}}$, $M_{0,\text{cal}}$, and $M_{0,\text{pooled-128}}$. This provides a strict test of whether parameter-level representation learning requires equal or greater target data than simple output fusion.

---

## 4. Predefined Primary Comparisons vs. Exploratory Diagnostics

### Predefined Primary Comparisons (Evaluated at $\alpha = 0.05$ with Holm correction):
1. **Primary Comparison 1 (The Calibration Bridge):**
   - Contrast: $M_{\text{CoLoRA},\text{cal}}$ vs. $M_{\text{CoLoRA},\text{raw}}$
   - Hypothesis: Post-hoc Platt scaling significantly improves Brier score ($\Delta < 0$).
2. **Primary Comparison 2 (Collaborative Value under Equal Calibrated Budget):**
   - Contrast: $M_{\text{CoLoRA},\text{cal}}$ vs. $M_{0,\text{cal}}$
   - Hypothesis: With matched calibration access ($N_{\text{total}}=128$), collaborative representation sharing achieves lower Brier score than single-task learning ($\Delta < 0$).
3. **Primary Comparison 3 (Representation Sharing vs. Direct Target Data Collection):**
   - Contrast: $M_{\text{CoLoRA},\text{cal}}$ vs. $M_{0,\text{pooled-128}}$
   - Hypothesis: CoLoRA with calibration outperforms a single-task model trained directly on the combined $N=128$ target examples.
4. **Primary Comparison 4 (Subspace Invariance):**
   - Contrast: $M_{\text{CoLoRA},\text{cal}}$ vs. $M_{\text{subspace},\text{cal}}$
   - Hypothesis: Both low-rank collaborative architectures achieve statistically indistinguishable calibrated performance ($|\bar{\Delta}| \le 0.005$).
5. **Primary Comparison 5 (Representation vs. Output Fusion):**
   - Contrast: $M_{\text{CoLoRA},\text{cal}}$ vs. $M_{\text{fuse}}$
   - Objective: Determine whether calibrated representation sharing outperforms 3-source output fusion.

### Exploratory Diagnostics (Descriptive, not decision gates):
- Expected Calibration Error (ECE) pre- and post-calibration across all architectures.
- AUROC invariance check under Platt scaling (verifying that strictly monotonic Platt scaling preserves ranking).
- Subspace Grassmannian projection alignment $S_{\text{proj}}(\hat{B}, U_{\text{true}})$ and parameter cosine similarity distributions across the 10 replications.

---

## 5. Mechanistic Interpretation Guardrails

We explicitly precommit to avoiding overinterpreting predictive differences:
- If $M_{\text{fuse}}$ or $M_{\text{CoLoRA},\text{cal}}$ outperforms baselines, we will **not** claim "proven error cancellation" or "proven variance reduction" without separate decomposed empirical proofs (e.g. bias-variance decompositions or error covariance analyses). We will describe such outcomes strictly as empirical predictive improvements under the evaluated loss functions.

---

## 6. Execution Constraints & Test Isolation

- **CPU Only:** Strictly 2 PyTorch threads (`torch.set_num_threads(2)`).
- **Runtime Budget:** Pre-execution benchmark must project total runtime $< 20$ minutes.
- **Strict Held-Out Quarantine:** Target Held-Out Test ($N=256$) is strictly quarantined; Stage 4 will **not** be executed. All development decisions terminate at Stage 3.
