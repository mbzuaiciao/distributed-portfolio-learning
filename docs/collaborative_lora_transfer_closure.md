# Formal Closure Report: Multi-Task LoRA Transferability Benchmark

**Document Version:** 1.0  
**Date:** 2026-10-10  
**Status:** **CLOSED — DEVELOPMENT-GATE FAILURE (CONFIRMATORY EVALUATION CANCELLED)**  
**Repository:** `distributed-portfolio-learning`  
**Branch:** `experiment/collaborative-lora-transferability`  
**Current Commit:** `8052f0a9`  
**Preceding Protocol:** [`docs/collaborative_lora_transfer_amendment.md`](collaborative_lora_transfer_amendment.md) (`cf0b8175`)  
**Pre-Run Audit:** [`docs/collaborative_lora_transfer_audit_amendment.md`](collaborative_lora_transfer_audit_amendment.md) (`a29ea28d`)  
**Development Artifacts:** [`artifacts/multitask_transfer_benchmark/development_report_transfer.md`](../artifacts/multitask_transfer_benchmark/development_report_transfer.md)  

---

## 1. Executive Summary & Gate Decision

This document formally records the scientific closure of the Multi-Task LoRA Transferability Experiment. 

Following the frozen protocol specified in `docs/collaborative_lora_transfer_amendment.md`, the benchmark evaluated whether **Structural CoLoRA** ($M_{\text{CoLoRA}}$) achieves positive transfer to a severely data-starved target task ($N=64$). 

### Prospective Gate Evaluation

| Gate / Stopping Criterion | Prospectively Frozen Rule | Observed Development Result ($N=256$) | Gate Status |
| :--- | :--- | :--- | :---: |
| **Criterion 1: Positive Transfer over $M_0$** | Mean $\Delta \text{Brier} \ge +0.010$ over $M_0$; positive across all 3 seeds | Mean $\Delta = +0.0019$ ($0.2878$ vs. $0.2897$); failed on Seed 123 ($\Delta = -0.0478$) | **FAILED** |
| **Criterion 2: Superiority over Pooling $M_1$** | Mean $\Delta \text{Brier} \ge +0.005$ over Pooled LoRA ($M_1$) | $M_{\text{CoLoRA}}$ ($0.2878$) trailed $M_1$ ($0.2164$) by $\Delta = -0.0714$ Brier | **FAILED** |
| **Criterion 3: Superiority over Calibrated Fusion $M_{\text{fuse}}$** | $M_{\text{CoLoRA}}$ must statistically outperform $M_{\text{fuse}}$ | $M_{\text{CoLoRA}}$ ($0.2878$) substantially trailed $M_{\text{fuse}}$ ($0.1923$) by $\Delta = -0.0955$ Brier | **FAILED** |
| **Confirmatory Held-Out Test Evaluation** | Execute Stage 4 on quarantined test split ($N=512$, seed=20261024) only if Criteria 1 & 2 pass | Stage 4 execution strictly aborted; test split quarantined | **CANCELLED** |

**Formal Decision:**  
The experiment is formally **CLOSED as a development-gate failure**. In accordance with the prospective stopping rules, **no confirmatory evaluation was performed on the quarantined held-out test split ($N=512$)**, preserving the quarantine and precluding post-hoc data dredging.

---

## 2. Observed Positive Representation Recovery in Positive Control

While the prospective probability Brier criteria failed, the experiment produced unambiguous evidence of representation-level transfer:

1. **Latent Subspace Recovery:**
   - Structural CoLoRA successfully recovered the ground-truth rank-2 latent subspace $U_{\text{true}} \in \mathbb{R}^{32 \times 2}$ with a mean Grassmannian projection alignment of **$S_{\text{proj}}(\hat{B}, U_{\text{true}}) = 0.6815 \pm 0.0663$**.
   - Direct Shared-Subspace Multi-Task ($M_{\text{subspace}}$) achieved comparable alignment: **$S_{\text{proj}}(\hat{U}, U_{\text{true}}) = 0.6584 \pm 0.1313$**.
2. **Target Weight Orientation Recovery:**
   - Single-task target logistic regression ($M_0, N=64$) suffered severe parameter estimation variance, recovering the true parameter direction $\beta^*_{\text{target}}$ with a mean cosine similarity of only **$0.6004 \pm 0.0418$**.
   - Structural CoLoRA ($M_{\text{CoLoRA}}$) reconstructed the target parameter orientation with a mean cosine similarity of **$0.7311 \pm 0.1015$** (+0.1308 gain).
   - Shared-Subspace ($M_{\text{subspace}}$) similarly achieved **$0.7199 \pm 0.0989$**.
3. **Discriminative Ranking (AUROC):**
   - Single-task target logistic regression achieved mean AUROC of **$0.6746 \pm 0.0210$**.
   - Structural CoLoRA achieved mean AUROC of **$0.7253 \pm 0.0260$** (+0.0507 gain).
   - Shared-Subspace achieved mean AUROC of **$0.7180 \pm 0.0368$** (+0.0434 gain).

These findings demonstrate that multi-task representation sharing successfully pools statistical power across related tasks to learn the shared feature subspace, significantly improving feature orientation and ranking over single-task training under severe data scarcity ($N=64$).

---

## 3. Mechanistic Cause of Primary Brier Failure: Logit Calibration vs. Discrimination

The divergence between strong discrimination gains (+0.0507 AUROC) and near-zero probability Brier gains (+0.0019 Brier) highlights a critical methodological finding:

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2 = \text{Reliability (Calibration Error)} - \text{Resolution} + \text{Uncertainty}$$

1. **Overconfident Small-Sample Logits:**
   - Both $M_{\text{CoLoRA}}$ and $M_{\text{subspace}}$ updated target-specific routing parameters ($\Lambda_{\text{target}}$ and $v_{\text{target}}$) via gradient descent on small target batches ($N=64$) without explicit temperature scaling or validation-based early stopping.
   - This optimization dynamic resulted in severe probability miscalibration: Expected Calibration Error (ECE) was **$0.2669$** for $M_{\text{CoLoRA}}$ and **$0.2409$** for $M_{\text{subspace}}$, comparable to unregularized $M_0$ (ECE **$0.2495$**).
2. **Brier Penalization of Extreme Probabilities:**
   - Because the Brier score quadratically penalizes probability errors, overconfident probabilities ($p \to 0$ or $p \to 1$) on misclassified examples incur severe quadratic losses, offsetting the rank-ordering gains observed in AUROC.
   - In contrast, methods that had access to larger sample sizes or calibration procedures ($M_1$ with ECE $0.0665$; $M_{\text{fuse}}$ with ECE $0.0726$; $M_{0,\text{cal-matched}}$ with ECE $0.0842$) achieved substantially lower Brier scores ($0.1923$ to $0.2164$) primarily through well-calibrated probabilities around the base rate.

---

## 4. Negative Control Findings: Harmful Transfer & Pooling Failure

The Negative-Control regime (where tasks have mutually orthogonal parameter vectors $\beta_k \sim \mathcal{N}(0, I_d)$) validated the necessity of task-specific parameterization:

1. **Catastrophic Collapse of Unstructured Pooling ($M_1$):**
   - Pooled Global LoRA ($M_1$, trained on all 1,600 samples across tasks) completely collapsed on the target task:
     - Mean AUROC dropped to **$0.5020 \pm 0.0236$** (pure random guessing).
     - Target parameter cosine similarity collapsed to **$0.0208 \pm 0.0515$**.
   - Naive data pooling forced incompatible task directions to compromise on a noise vector, causing devastating negative transfer.
2. **Breakdown of Prediction-Level Calibrated Fusion ($M_{\text{fuse}}$):**
   - Output calibrated fusion achieved an AUROC of only **$0.5642 \pm 0.0042$** and a Brier score of **$0.2550 \pm 0.0032$**, performing worse than the constant prevalence null baseline ($0.2506$).
   - When source models have zero true correlation with the target task, combining their outputs introduces noise that cannot be fully corrected even on $N=128$ calibration examples.
3. **Protection by Task-Specific Routing:**
   - Structural CoLoRA ($M_{\text{CoLoRA}}$) and Shared-Subspace ($M_{\text{subspace}}$) isolated task loadings via $\Lambda_k$ and $v_k$, preventing catastrophic interference and preserving target discrimination (AUROC **$0.6749$** and **$0.6833$**; cosine similarities **$0.5008$** and **$0.5064$**).

---

## 5. Scientific Correction: Clarification of the Calibrated Fusion Advantage

In our pre-run scientific audit, we identified that $M_{\text{fuse}}$ had access to 128 additional labeled target calibration examples ($N=128$) beyond the target training set ($N=64$). To control for this label confound, we introduced $M_{0,\text{cal-matched}}$ (Target-Only Logistic trained on $N = 64 + 128 = 192$ target labels).

### Provenance-Preserving Correction
- **Previous Informal Assertion:** *"The low Brier score of $M_{\text{fuse}}$ is entirely an artifact of having 128 extra target calibration labels."*
- **Empirical Correction:** While the additional 128 labels accounted for the vast majority of the performance gap between $M_0$ ($N=64$, Brier $0.2897$) and $M_{\text{fuse}}$ (Brier $0.1923$), **it did not explain 100% of the fusion advantage in the positive-control regime**:
  - $M_{0,\text{cal-matched}}$ ($N=192$) achieved Brier **$0.2072 \pm 0.0064$** and AUROC **$0.7615 \pm 0.0096$**.
  - $M_{\text{fuse}}$ achieved Brier **$0.1923 \pm 0.0120$** and AUROC **$0.7788 \pm 0.0250$**.
- In the positive control, where source tasks genuinely correlate with the target task, calibrated multivariate log-odds fusion provided a modest incremental gain ($\Delta = -0.0149$ Brier, $+0.0173$ AUROC) beyond equal-label single-task learning. This incremental benefit stems from variance reduction via ensembling independent source estimators, coupled with explicit logistic calibration.

---

## 6. Shared-Subspace Equivalence: No CoLoRA-Specific Advantage

The benchmark directly evaluated whether Structural CoLoRA ($w_k = B \Lambda_k A$) confers any architectural benefit over the simpler low-rank shared-subspace model ($w_k = U v_k$):

| Metric (Positive Control) | Shared-Subspace Multi-Task ($M_{\text{subspace}}$) | Structural CoLoRA ($M_{\text{CoLoRA}}$) | Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Trainable Parameters** | **76** | **86** | +10 |
| **Mean Brier Score $\downarrow$** | $0.2775 \pm 0.0370$ | $0.2878 \pm 0.0355$ | $-0.0103$ (favoring Subspace) |
| **Mean AUROC $\uparrow$** | $0.7180 \pm 0.0368$ | $0.7253 \pm 0.0260$ | $+0.0073$ (favoring CoLoRA) |
| **Mean ECE $\downarrow$** | $0.2409$ | $0.2669$ | $-0.0260$ (favoring Subspace) |
| **Target Weight Cosine Sim $\uparrow$** | $0.7199 \pm 0.0989$ | $0.7311 \pm 0.1015$ | $+0.0112$ (favoring CoLoRA) |
| **Subspace Alignment $S_{\text{proj}} \uparrow$** | $0.6584 \pm 0.1313$ | $0.6815 \pm 0.0663$ | $+0.0231$ (favoring CoLoRA) |

### Key Theoretical Conclusion
As proven in the pre-run audit, for scalar outputs ($d_{\text{out}}=1$), any linear weight vector has matrix rank 1. Consequently:
$$\operatorname{span}(B \Lambda_k A) = \operatorname{span}(B) = \operatorname{span}(U)$$
The reachable parameter manifolds of CoLoRA and Shared-Subspace Multi-Task are **mathematically identical**. Empirically, their performance is indistinguishable within experimental noise. Any positive transfer observed in this benchmark is attributable to **shared-subspace representation learning**, not to LoRA's specific trilinear parameterization.

---

## 7. Seed-Level Variability & Limitations of 3-Seed Inference

The empirical metrics exhibited notable variability across the three random seeds ($42, 123, 777$):

| Seed | Positive Control $M_0$ Brier | Positive Control CoLoRA Brier | $\Delta$ Brier ($M_0 - \text{CoLoRA}$) | Positive Control CoLoRA AUROC | Negative Control CoLoRA AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 0.2834 | 0.2581 | **+0.0253** | 0.7556 | 0.7012 |
| **123** | 0.2899 | 0.3377 | **-0.0478** | 0.6921 | 0.6389 |
| **777** | 0.2957 | 0.2676 | **+0.0281** | 0.7282 | 0.6845 |

### Statistical Limitations
1. On seeds 42 and 777, CoLoRA achieved Brier improvements of $+0.0253$ and $+0.0281$ (satisfying the $+0.010$ threshold on those seeds in isolation).
2. However, on seed 123, CoLoRA's target adapter optimization produced extreme logits (Log Loss = $2.48$, ECE = $0.326$), resulting in a Brier deterioration of $-0.0478$.
3. With only $S=3$ random seeds, the standard deviation of Brier for CoLoRA was $0.0355$. A sample of 3 seeds is vulnerable to small-sample optimization outliers. Any future calibration study must utilize at least $S \ge 10$ independent dataset/seed replications to achieve bounded confidence intervals.

---

## 8. Audit of Evaluation Script Modifications & Provenance

To maintain complete research integrity, we audited the code modifications between commit `a29ea28d` (the frozen commit before training) and commit `8052f0a9` (the development evaluation commit):

### Audit Log
1. **Commit `a29ea28d`:**
   - Completed pre-run scientific audit.
   - Implemented `stage_train()` and `stage_evaluate_dev()` including $M_{0,\text{cal-matched}}$.
   - Authored the initial markdown report generation string in `stage_evaluate_dev()`.
2. **First Run of `stage evaluate_dev`:**
   - Evaluated models on Target Development ($N=256$).
   - Terminal summary printed negative delta relative to $M_{\text{fuse}}$ ($-0.0955$) and small delta over $M_0$ ($+0.0019$).
   - A syntax inspection revealed that the draft report template in `a29ea28d` contained an unescaped LaTeX backslash in `\cos` and unescaped braces, which raised a Python `SyntaxWarning`. Furthermore, the template prose drafted prior to execution had inverted sign interpretations (treating negative Brier differences as improvements).
3. **Commit `8052f0a9`:**
   - Corrected the template string formatting and LaTeX escaping (`\\cos`, `\\text`).
   - Refined Section 3 commentary to accurately describe the sign of Brier differences, the role of probability calibration (ECE), and the $M_{0,\text{cal-matched}}$ control.
   - **Verification:** No model definitions, data splits, loss functions, optimization hyperparameters, seed lists, or evaluation metrics were modified. The changes were strictly confined to report string formatting and LaTeX syntax.

---

## 9. Final Closure Declaration

Experiment `experiment/collaborative-lora-transferability` is **OFFICIALLY CLOSED**.
- Primary prospective Brier gates: **FAILED**.
- Confirmatory test evaluation: **ABORTED / QUARANTINED**.
- Mathematical mechanism established: Shared-subspace learning recovers target feature orientation (+0.131 cos sim, +0.051 AUROC), but small-sample logit miscalibration impairs probability Brier performance.
- Any future investigation of this mechanism must proceed via a separate, prospective calibration-diagnostic plan.
