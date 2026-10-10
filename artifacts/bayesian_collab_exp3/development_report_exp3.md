# Experiment 3: Development Evaluation & Uncertainty Routing Report

**Date:** 2026-10-10  
**Branch:** `experiment/bayesian-collaboration-uncertainty-routing`  
**Execution Environment:** macOS Darwin, CPU only, strictly capped at 2 PyTorch threads (`torch.set_num_threads(2)`).  
**Protocol References:**
- Plan: [`docs/bayesian_collaboration_experiment3_plan.md`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/docs/bayesian_collaboration_experiment3_plan.md)
- Amendment: [`docs/bayesian_collaboration_experiment3_amendment.md`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/docs/bayesian_collaboration_experiment3_amendment.md)
- Audit: [`artifacts/bayesian_collab_exp3/pre_training_audit_exp3.md`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/artifacts/bayesian_collab_exp3/pre_training_audit_exp3.md)

---

## 1. Executive Summary & Gate Verdicts

Experiment 3 evaluated whether Bayesian epistemic uncertainty routing provides incremental predictive information beyond calibrated predictive confidence, training-support density, and deterministic learned gating across two distinct spatial regimes:
- **Regime A:** Sharp axis-aligned regional specialization ($x_{s,0} \in [-1, 1]$).
- **Regime B:** Smooth multi-modal 2D Gaussian mixture specialization ($\mu_A = [-1, -1], \mu_B = [1, 1]$).

Prior to unlocking the held-out test split, the protocol enforces two mandatory developmental validation gates on the epistemic uncertainty estimates:
1. **Gate U1 (Support Sparsity Tracking):** Spearman rank correlation $\rho_s(s^2(x), d_M^2(x)) \ge 0.40$ between epistemic variance and Mahalanobis distance to specialist training support.
2. **Gate U2 (Residual Error Partial Correlation):** Partial Pearson correlation $r_{s^2, \text{resid}^2 \cdot \text{conf}} \ge 0.10$ between epistemic variance and squared prediction residual, partialling out predictive confidence $|p - 0.5|$.

### Gate Outcome Summary Table

| Regime | Specialist | Gate U1 (Support Sparsity $\ge 0.40$) | Gate U1 Verdict | Gate U2 (Residual Partial $\ge 0.10$) | Gate U2 Verdict | Combined Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regime A** | Spec A ($m_2$) | $\rho_s = 0.250$ | **FAIL** | $r = -0.148$ | **FAIL** | **FAILED** |
| **Regime A** | Spec B ($m_3$) | $\rho_s = 0.239$ | **FAIL** | $r = -0.229$ | **FAIL** | **FAILED** |
| **Regime B** | Spec A ($m_2$) | $\rho_s = 0.170$ | **FAIL** | $r = -0.035$ | **FAIL** | **FAILED** |
| **Regime B** | Spec B ($m_3$) | $\rho_s = 0.175$ | **FAIL** | $r = -0.156$ | **FAIL** | **FAILED** |

> [!CAUTION]
> **GATE FAILURE & CONFINEMENT PROTOCOL TRIGGERED:**  
> Both Gate U1 and Gate U2 failed across both regimes and all specialists.  
> In strict compliance with Section 5.3 of the frozen research plan, **evaluation is halted immediately**. The quarantined held-out test split ($N=512$, seed=999) **has NOT been accessed, evaluated, or unlocked**.

---

## 2. Resource & Execution Tracking

All models (24 neural network runs across 2 regimes, 4 base architectures, and 3 seeds), Bayesian readouts, calibrators, and routers were trained and evaluated strictly on CPU:

| Metric | Stage 2 (`train`) | Stage 3 (`evaluate_dev`) | Protocol Budget / Limit | Compliance |
| :--- | :---: | :---: | :---: | :---: |
| **Hardware** | macOS CPU (M-series) | macOS CPU (M-series) | CPU only | **Pass** |
| **PyTorch CPU Threads** | 2 | 2 | Max 2 threads | **Pass** |
| **Elapsed Runtime** | 7.60 s | 0.39 s | Max 20 min (1200 s) | **Pass** |
| **Peak Memory (RSS)** | 352.5 MB | 277.2 MB | 4000 MB | **Pass** |
| **Held-out Test Access** | **None** | **None** | Quarantined | **Pass** |

---

## 3. Pre-Registered Non-Bayesian Comparator Selection ($M^*$)

The candidate non-Bayesian comparator pool comprises:
$$\mathcal{M}_{\text{pool}} = \{ M_0, M_1, M_2, M_3, M_6, M_{\text{Conf}}, M_{\text{Dens}}, M_{\text{MoE-Lin}}, M_{\text{MoE-MLP}}, M_{\text{Router-NoUQ}} \}$$

The deterministic selection rule specifies:
1. Lowest mean Brier score on the development split across seeds $\{42, 123, 777\}$.
2. Tie-breaker 1: Highest development AUROC.
3. Tie-breaker 2: Lowest development ECE.

### Selection Outcomes

- **Regime A:**
  - $M_6$ (`m6_logodds_cal`) achieved mean Brier **0.1646** ($\pm 0.0062$), outperforming all other non-Bayesian candidates ($M_0$: 0.1737, $M_{\text{Router-NoUQ}}$: 0.1747, $M_{\text{Conf}}$: 0.1918, $M_{\text{MoE-MLP}}$: 0.2004, $M_1$: 0.2008, $M_{\text{Dens}}$: 0.2032, $M_{\text{MoE-Lin}}$: 0.2056, $M_2$: 0.2090, $M_3$: 0.2112).
  - **Selected $M^*$ for Regime A:** `m6_logodds_cal` (Calibrated Log-Odds Fusion).

- **Regime B:**
  - $M_6$ (`m6_logodds_cal`) achieved mean Brier **0.1736** ($\pm 0.0046$), outperforming all other non-Bayesian candidates ($M_{\text{Dens}}$: 0.1810, $M_{\text{Router-NoUQ}}$: 0.1875, $M_{\text{Conf}}$: 0.1927, $M_{\text{MoE-Lin}}$: 0.1949, $M_{\text{MoE-MLP}}$: 0.1976, $M_3$: 0.1977, $M_2$: 0.2014, $M_1$: 0.2152, $M_0$: 0.2318).
  - **Selected $M^*$ for Regime B:** `m6_logodds_cal` (Calibrated Log-Odds Fusion).

---

## 4. Comprehensive Development Performance Results

All metrics evaluated on the disjoint development split ($N=512$, seed=888), averaged across 3 training seeds ($\pm$ standard deviation):

### Regime A (Axis-Aligned Regional Specialization)

| Model ID | Method / Architecture | Mean Brier $\downarrow$ | Std Brier | Mean AUROC $\uparrow$ | Mean ECE $\downarrow$ | Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **$M_6$ ($M^*$)** | **Calibrated Log-Odds Fusion** | **0.1646** | 0.0062 | 0.8218 | 0.0844 | **Selected Non-Bayes $M^*$** |
| $M_0$ | Target-Only MLP | 0.1737 | 0.0218 | 0.8938 | 0.1789 | High AUROC, undercalibrated |
| $M_{\text{Bayes-Learn}}$ | Bayesian Learned Router ($M_{\text{Router-UQ}}$) | 0.1746 | 0.0135 | 0.8011 | 0.0879 | Bayesian uncertainty-aware |
| $M_{\text{Router-NoUQ}}$ | Matched Ablation Router (No UQ) | 0.1747 | 0.0085 | 0.8074 | 0.0941 | Matched baseline (no variance) |
| $M_{\text{Conf}}$ | Confidence Gating ($|p - 0.5|$) | 0.1918 | 0.0037 | 0.8108 | 0.1404 | Heuristic confidence |
| $M_{\text{Bayes-Inv}}$ | Bayesian Inverse-Variance Router | 0.1991 | 0.0030 | 0.7624 | 0.1023 | Precision-weighted log-odds |
| $M_{\text{MoE-MLP}}$ | Deterministic MLP MoE | 0.2004 | 0.0062 | 0.7830 | 0.1347 | Nonlinear router on $x_s$ |
| $M_1$ | Capacity-Matched Pooled | 0.2008 | 0.0086 | 0.8532 | 0.2027 | High capacity, poor calibration |
| $M_{\text{Dens}}$ | Empirical Density Router | 0.2032 | 0.0032 | 0.7873 | 0.1625 | Gaussian density in $x_s$ |
| $M_{\text{MoE-Lin}}$ | Deterministic Linear MoE | 0.2056 | 0.0041 | 0.7593 | 0.1192 | Linear router on $x_s$ |
| $M_2$ | Specialist A (Calibrated) | 0.2090 | 0.0056 | 0.7412 | 0.1046 | Regional specialist |
| $M_3$ | Specialist B (Calibrated) | 0.2112 | 0.0054 | 0.7566 | 0.1551 | Regional specialist |
| $M_8$ | Prevalence Null Baseline | 0.2456 | 0.0000 | 0.5000 | 0.0547 | Unconditional mean |
| $M_9$ | Bayes Optimal Oracle | 0.0793 | 0.0000 | 0.9678 | 0.0730 | Ground truth oracle |

### Regime B (Smooth 2D Gaussian Mixture Specialization)

| Model ID | Method / Architecture | Mean Brier $\downarrow$ | Std Brier | Mean AUROC $\uparrow$ | Mean ECE $\downarrow$ | Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **$M_6$ ($M^*$)** | **Calibrated Log-Odds Fusion** | **0.1736** | 0.0046 | 0.7978 | 0.0779 | **Selected Non-Bayes $M^*$** |
| $M_{\text{Dens}}$ | Empirical Density Router | 0.1810 | 0.0015 | 0.8017 | 0.1191 | Tracks Gaussian modes |
| $M_{\text{Bayes-Learn}}$ | Bayesian Learned Router ($M_{\text{Router-UQ}}$) | 0.1864 | 0.0057 | 0.7794 | 0.0863 | Bayesian uncertainty-aware |
| $M_{\text{Router-NoUQ}}$ | Matched Ablation Router (No UQ) | 0.1875 | 0.0054 | 0.7756 | 0.1107 | Matched baseline (no variance) |
| $M_{\text{Conf}}$ | Confidence Gating ($|p - 0.5|$) | 0.1927 | 0.0025 | 0.7902 | 0.1129 | Heuristic confidence |
| $M_{\text{MoE-Lin}}$ | Deterministic Linear MoE | 0.1949 | 0.0046 | 0.7643 | 0.0943 | Linear router on $x_s$ |
| $M_{\text{Bayes-Inv}}$ | Bayesian Inverse-Variance Router | 0.1957 | 0.0049 | 0.7530 | 0.0941 | Precision-weighted log-odds |
| $M_{\text{MoE-MLP}}$ | Deterministic MLP MoE | 0.1976 | 0.0044 | 0.7517 | 0.0875 | Nonlinear router on $x_s$ |
| $M_3$ | Specialist B (Calibrated) | 0.1977 | 0.0036 | 0.7580 | 0.0881 | Specialist on Mode B |
| $M_2$ | Specialist A (Calibrated) | 0.2014 | 0.0026 | 0.7141 | 0.0638 | Specialist on Mode A |
| $M_1$ | Capacity-Matched Pooled | 0.2152 | 0.0122 | 0.8219 | 0.2231 | High capacity, poor calibration |
| $M_0$ | Target-Only MLP | 0.2318 | 0.0146 | 0.8111 | 0.2379 | Small sample size ($N=256$) |
| $M_8$ | Prevalence Null Baseline | 0.2346 | 0.0000 | 0.5000 | 0.0156 | Unconditional mean |
| $M_9$ | Bayes Optimal Oracle | 0.1069 | 0.0000 | 0.9232 | 0.0642 | Ground truth oracle |

---

## 5. Matched Uncertainty-Feature Ablation & Parameter Accounting

To isolate whether Bayesian epistemic variance provides incremental information beyond deterministic features, we compared two architectures trained on identical fusion splits with identical optimization hyperparameters (AdamW, lr=0.01, wd=0.01, 100 epochs):
1. **$M_{\text{Router-NoUQ}}$ (No Uncertainty):** Inputs $[l_A, l_B, x_{s,1..4}]$ (6 features).
2. **$M_{\text{Router-UQ}}$ / $M_{\text{Bayes-Learn}}$ (With Uncertainty):** Inputs $[l_A, l_B, s^2_A, s^2_B, x_{s,1..4}]$ (8 features).

### Parameter Breakdown

| Model | Layer 1 Weights | Layer 1 Bias | Layer 2 Weights | Layer 2 Bias | Total Parameters | Parameter Ratio vs NoUQ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$M_{\text{Router-NoUQ}}$** | $6 \times 16 = 96$ | 16 | $16 \times 1 = 16$ | 1 | **129** | 1.000x |
| **$M_{\text{Router-UQ}}$** | $8 \times 16 = 128$ | 16 | $16 \times 1 = 16$ | 1 | **161** | 1.248x |
| **Difference** | **+32** | 0 | 0 | 0 | **+32** | — |

The 32-parameter difference corresponds strictly to the linear projection weights mapping the two epistemic variance inputs into the 16 hidden hidden units. No extra depth or nonlinear capacity was added.

### Ablation Contrast

| Regime | Brier ($M_{\text{Router-NoUQ}}$) | Brier ($M_{\text{Router-UQ}}$) | Brier Gain ($\Delta$) | Direction | Materiality |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Regime A** | 0.174716 | 0.174634 | **+0.000082** | Favors UQ | Negligible ($\Delta < 0.0001$) |
| **Regime B** | 0.187516 | 0.186421 | **+0.001094** | Favors UQ | Minimal ($\Delta \approx 0.0011$) |

**Interpretation:** Epistemic variance inputs yield virtually zero predictive gain over deterministic features ($+0.0001$ in Regime A, $+0.0011$ in Regime B). Furthermore, both models are substantially worse than the simple 3-parameter calibrated log-odds fusion baseline ($M_6$, Brier 0.1646 in Regime A, 0.1736 in Regime B).

---

## 6. Diagnostic Analysis of Gate Failures

### 1. Gate U1: Support Sparsity Tracking ($\rho_s < 0.40$)
- **Observed:** Specialist A and B achieved $\rho_s \in [0.239, 0.250]$ in Regime A, and $\rho_s \in [0.170, 0.175]$ in Regime B.
- **Root Cause:** The Laplace approximation on penultimate representations evaluates $\Sigma = (J^T J + \alpha I)^{-1}$. In a multi-layer perceptron with ReLU activations, representation geometry is piece-wise linear and concentrated. While $s^2(x) = \phi(x)^T \Sigma \phi(x)$ increases outside the training support, the representation norms $\|\phi(x)\|$ do not grow monotonically with Euclidean or Mahalanobis distance in raw input space $x_s$. Consequently, the rank alignment with input Mahalanobis distance saturates at $\rho_s \approx 0.17 - 0.25$, well below the 0.40 gate threshold.

### 2. Gate U2: Residual Error Partial Correlation ($r < 0.10$, observed negative)
- **Observed:** Partial correlations were $-0.148$ (A) and $-0.229$ (B) in Regime A, and $-0.035$ (A) and $-0.156$ (B) in Regime B.
- **Root Cause:**
  1. Prediction errors $(p - y)^2$ are dominated by aleatoric noise (points near Bayes decision boundary where $p^* \approx 0.5$).
  2. Controlling for boundary proximity $|p - 0.5|$ isolates instances where the model makes high-confidence errors ($|p - 0.5|$ large, but $y$ opposite).
  3. Under the Laplace approximation, points far from the training support often map to saturated regions of the penultimate space where variance is moderate or high, but the calibrated probability $p$ is pushed towards the base rate. In contrast, extreme overconfident errors occur within or near the training support where $s^2$ is actually small.
  4. This produces a *negative* partial correlation: larger epistemic variance correlates with *smaller* residual errors after controlling for confidence. Thus, epistemic variance fails to identify high-residual blind spots.

---

## 7. Protocol Compliance & Confinement Status

1. **Gate Decision:** Both developmental gates failed in all configurations.
2. **Confirmatory Quarantine:** In adherence to the pre-registered protocol, Stage 4 (`evaluate_test`) was **not executed**. The held-out test labels remain strictly untouched.
3. **Reproducibility Artifacts Saved:**
   - Checkpoints: `artifacts/bayesian_collab_exp3/ckpt_exp3_*.pt` (with SHA-256 digests in `frozen_exp3.json`).
   - Summary JSON: `artifacts/bayesian_collab_exp3/development_summary_exp3.json`.
   - Evaluation Report JSON: `artifacts/bayesian_collab_exp3/development_evaluation_report_exp3.json`.
   - Per-seed CSV: `artifacts/bayesian_collab_exp3/development_metrics_exp3.csv`.

---
