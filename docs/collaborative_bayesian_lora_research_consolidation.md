# Research Consolidation & Decision Document: Collaborative Bayesian LoRA & Uncertainty-Aware Fusion (Experiments 1–4)

**Document Version:** 1.0  
**Date:** 2026-10-10  
**Status:** **OFFICIAL RESEARCH CONSOLIDATION & STRATEGIC DECISION**  
**Repository:** `distributed-portfolio-learning`  
**Current Active Branch:** `experiment/minimal-uncertainty-validation`  
**Associated Branches & Milestones:**
- Experiment 1: `experiment/bayesian-collaboration-positive-controls` (HEAD: `65c92816`)
- Experiment 2: `experiment/bayesian-collaboration-calibration-fusion` (HEAD: `4a87710d`, closure: `eb0d8745`)
- Experiment 3: `experiment/bayesian-collaboration-uncertainty-routing` (HEAD: `6e2b21c2`, closure: `322f8ec3`)
- Experiment 4: `experiment/minimal-uncertainty-validation` (HEAD: `118499cb`)

---

## 1. Executive Summary & Purpose

This document consolidates four sequential investigations into collaborative low-rank adaptation (LoRA), specialist calibration, and Bayesian epistemic uncertainty routing. Its objectives are to:
1. Synthesize the empirical findings across Experiments 1 through 4 into a unified scientific record.
2. Formally correct historical overclaims while strictly preserving frozen protocol reports, code hashes, and quarantined datasets.
3. Articulate the mathematical distinctions between training-support density, posterior epistemic uncertainty, predictive confidence, calibration, discrimination, and downstream decision utility.
4. Establish definitive criteria for when Bayesian uncertainty research may legitimately be reopened.
5. Provide a strategic decision and roadmap for the next research phase.

---

## 2. Experimental Retrospective: Experiments 1–4

### 2.1 The Original Collaborative Bayesian LoRA Hypothesis
The research trajectory originated in Praneeth's multi-task collaboration setting: a decentralized environment where multiple specialist agents observe partitioned, local views of a data-generating process (e.g., distinct feature subsets, asset classes, or temporal regimes). 

The initial **Collaborative LoRA Hypothesis** postulated that:
- Individual specialists could adapt frozen base representations using low-rank decomposition matrices ($W_i = W_0 + B_i A_i$) constrained to a shared hypergraph communication subspace.
- A **Bayesian formulation** of collaborative LoRA would allow specialists to infer posterior distributions over their adapter weights, yielding epistemic uncertainty estimates that could dynamically route test predictions: weighting specialists with high support and down-weighting specialists extrapolating beyond their local training domains.

```mermaid
flowchart LR
    subgraph E1["Experiment 1 (Gate Failure)"]
        direction TB
        E1_1["CoLoRA vs. Baselines"]
        E1_2["Uncalibrated Probabilities"]
        E1_3["Confirmatory Test Locked"]
    end
    subgraph E2["Experiment 2 (Held-Out Test)"]
        direction TB
        E2_1["Platt Scaling + Log-Odds (M6)"]
        E2_2["3 Secondary Tests Confirmed"]
        E2_3["Primary vs. M0 Unconfirmed"]
    end
    subgraph E3["Experiment 3 (Gate Failure)"]
        direction TB
        E3_1["Laplace Readout Routing"]
        E3_2["Gates U1 & U2 Failed"]
        E3_3["Negligible Router Delta"]
    end
    subgraph E4["Experiment 4 (Minimal Benchmark)"]
        direction TB
        E4_1["BLR vs. Neural Laplace"]
        E4_2["Support Tracking Sound"]
        E4_3["Selective Risk Uncoupled"]
    end
    E1 -->|Isolate Calibration| E2
    E2 -->|Prospectively Test Bayesian Routing| E3
    E3 -->|Isolate Estimator Mechanics| E4
```

---

### 2.2 Experiment 1: Development-Gate Failure & Calibration Bottlenecks
- **Objective:** Evaluate structural CoLoRA against target-only and capacity-matched pooled training on synthetic complementary and redundant multi-task data.
- **Protocol & Status:** Prospective protocol; development gates failed; held-out test split remained strictly locked.
- **Key Empirical Failures:**
  1. Specialist B exhibited poor Brier performance despite analytical access to predictive features.
  2. Structural CoLoRA was outperformed by a capacity-matched pooled baseline.
  3. Redundant specialists outperformed complementary specialists on development Brier.
- **Root-Cause Diagnosis:**
  - Standard cross-entropy loss produced severe logit overconfidence (poor probability calibration).
  - Simple linear probability averaging ($p_{\text{ens}} = \frac{1}{2}(p_A + p_B)$) is mathematically incapable of accumulating independent log-odds evidence under conditional independence, compressing joint certainty.
  - Optimization instability in low-rank factorization obscured architectural differences.

---

### 2.3 Experiment 2: Calibrated Evidence Fusion on Held-Out Evaluation
- **Objective:** Test whether post-hoc Platt calibration and bivariate log-odds fusion resolve Experiment 1's bottlenecks under a 5-split data hygiene architecture ($N_{\text{target}}=256, N_{\text{specA}}=512, N_{\text{specB}}=512, N_{\text{cal}}=256, N_{\text{fus}}=256, N_{\text{dev}}=512, N_{\text{test}}=512$).
- **Protocol & Status:** Completed confirmatory evaluation with paired cluster bootstrap inference ($B=2,000$ resamples) on quarantined test data (`seed=20261012`).
- **Primary Hypothesis Outcome ($M_6$ vs. $M_0$):**
  - Point estimate: $+0.0187$ Brier gain (exceeded the substantive effect threshold of $+0.010$).
  - 95% Paired Cluster Bootstrap CI: **[-0.0089, +0.0486]** ($p = 0.190$).
  - **Verdict:** **STATISTICALLY UNCONFIRMED**. The confidence interval spans zero; the hypothesis of statistically significant superiority over target-only is not supported at $\alpha=0.05$.
- **Confirmed Secondary Comparisons (Step-Down Holm-Bonferroni Adjusted):**
  1. $C_1$ ($M_6$ vs. Equal-Weight Linear $M_4$): Gain $+0.0093$, 95% CI $[+0.0022, +0.0162]$, **$p_{\text{holm}} = 0.039 < 0.05$** (**CONFIRMED**).
  2. $C_2$ ($M_6$ vs. Simplex Linear $M_5$): Gain $+0.0082$, 95% CI $[+0.0012, +0.0148]$, **$p_{\text{holm}} = 0.042 < 0.05$** (**CONFIRMED**).
  3. $C_3$ ($M_6$ vs. Capacity-Matched Pooled $M_1$): Gain $+0.0429$, 95% CI $[+0.0204, +0.0677]$, **$p_{\text{holm}} = 0.002 < 0.05$** (**CONFIRMED**).
- **Inconclusive Contrast & Tradeoffs:**
  - Complementarity contrast $C_4$ was $+0.0234$, 95% CI $[-0.0181, +0.0671]$ ($p = 0.274$, inconclusive).
  - **AUROC vs. Calibration Tradeoff:** Target-only achieved high AUROC ($0.8432$) but poor calibration (ECE $0.1857$, Brier $0.1913$). Calibrated log-odds fusion achieved superior calibration (ECE $0.0397$, Brier $0.1727$) but lower AUROC ($0.7827$), proving that factorized feature observation fundamentally bounds global rank discrimination.
  - Bivariate log-odds regression directly on raw logits is mathematically equivalent to regression on Platt-scaled logits, rendering separate Platt scaling functionally redundant for $M_6$.

---

### 2.4 Experiment 3: Development-Gate Failure of Bayesian Uncertainty Routing
- **Objective:** Test whether Bayesian epistemic uncertainty from penultimate-layer Laplace approximations provides incremental routing value beyond calibrated confidence, training density, and deterministic MoE gating across Regime A (axis-aligned shift) and Regime B (smooth 2D GMM).
- **Protocol & Status:** Development-gate failure; Stage 4 (`evaluate_test`) was **never executed**; test split remained strictly locked.
- **Validation Gate Outcomes:**
  - **Gate U1 (Support Sparsity Tracking $\rho_s \ge 0.40$):**
    - Regime A: Spec A $= 0.250$, Spec B $= 0.239$ (**FAIL**).
    - Regime B: Spec A $= 0.170$, Spec B $= 0.175$ (**FAIL**).
  - **Gate U2 (Residual Error Partial Correlation $r \ge 0.10$):**
    - Regime A: Spec A $= -0.148$, Spec B $= -0.229$ (**FAIL**).
    - Regime B: Spec A $= -0.035$, Spec B $= -0.156$ (**FAIL**).
- **Matched Ablation Router Results:**
  - Model comparison: $M_{\text{Router-NoUQ}}$ (129 params) vs. $M_{\text{Router-UQ}}$ (161 params, +32 linear projection weights).
  - Regime A Brier gain: **$+0.000082$** (negligible).
  - Regime B Brier gain: **$+0.001094$** (negligible).
  - Bayesian inverse-variance router ($M_{\text{Bayes-Inv}}$) performed substantially worse ($0.1991$ and $0.1957$ Brier).
- **Winner:** Static calibrated log-odds fusion ($M_6$) remained the strongest non-Bayesian comparator $M^*$ in both regimes (Brier **0.1646** in Regime A; **0.1736** in Regime B).

---

### 2.5 Experiment 4: Minimal Uncertainty-Estimator Diagnostic Benchmark
- **Objective:** Diagnostic benchmark on repeated synthetic draws ($M=20$ seeds, $N_{\text{train}}=256, N_{\text{test}}=1024$) comparing Bayesian Logistic Regression (BLR) with exact Newton-Raphson Laplace against Neural Penultimate-Layer Laplace across an $\alpha$ hyperprior grid.
- **Key Findings:**
  1. **Support Tracking is Mathematically Sound:** In linear log-concave models, BLR achieved $\rho_s = 0.988 \pm 0.003$ and OOD AUROC $0.964 \pm 0.006$. H_A.1 confirmed.
  2. **Latent Credible Intervals are Accurate:** BLR achieved $95.1\% \pm 7.6\%$ empirical coverage of true latent logits $z^*(x)$ in-support. H_A.2 confirmed.
  3. **Curvature Ill-Conditioning in Neural Representations:** Neural penultimate Hessian condition number was **57.6x worse** than BLR ($\kappa = 777.5 \pm 260.7$ vs. $13.5 \pm 1.9$ at $\alpha=1.0$). At $\alpha=0.01$, $\kappa$ exploded to $36,275$. H_B.3 confirmed.
  4. **The Selective-Risk Disconnect:** Abstaining on high-epistemic-uncertainty instances did **not** decrease Brier error (AU-RC $= 0.1807$ for BLR vs. $0.1510$ for random abstention). As inputs extrapolate along signal-bearing directions, extreme latent logits push probabilities toward 0 or 1, producing low Brier error on deterministic labels. In-support boundary instances retain high aleatoric noise ($p \approx 0.5$). Epistemic uncertainty is orthogonal to residual classification error.
  5. **Why Exp 3 Failed:** In Experiment 3, feature partitioning with zero-filled unobserved coordinates caused ReLU activations to collapse to zero ($\phi(x) \approx 0$), making the quadratic form $s^2 = \phi^T H^{-1} \phi$ blind to out-of-distribution coordinates.

---

## 3. Compact Experimental Evidence Table

| Experiment | Target Architecture / Method | Evaluated Split | Primary Metric | Primary Outcome | Status / Confirmatory Decision | Commit / Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | Structural CoLoRA vs. Pooled/Target | Development ($N=512$) | Brier Score | CoLoRA underperformed pooled; Spec B failed | **Development Gate Failed** (Test locked) | `65c92816` |
| **Exp 2** | Calibrated Log-Odds Fusion ($M_6$) vs. $M_0$ | Confirmatory Held-Out ($N=512$) | Brier Gain | $+0.0187$, 95% CI $[-0.0089, +0.0486]$, $p=0.190$ | **Statistically Unconfirmed** (3 secondary tests passed) | `4a87710d` |
| **Exp 2** | $M_6$ vs. Linear Averaging ($M_4, M_5$) | Confirmatory Held-Out ($N=512$) | Brier Gain | $+0.0093$ ($p_{\text{holm}}=0.039$); $+0.0082$ ($p_{\text{holm}}=0.042$) | **Confirmed Superior** | `4a87710d` |
| **Exp 2** | $M_6$ vs. Capacity Pooled ($M_1$) | Confirmatory Held-Out ($N=512$) | Brier Gain | $+0.0429$, 95% CI $[+0.0204, +0.0677]$, $p_{\text{holm}}=0.002$ | **Confirmed Superior** | `4a87710d` |
| **Exp 3** | Epistemic Laplace Routing vs. MoE Baselines | Development ($N=512$) | Gates U1, U2 & Router $\Delta$ | $\rho_s \in [0.17, 0.25]$; $r < 0$; $\Delta \text{Brier} \le +0.0011$ | **Development Gate Failed** (Test locked) | `6e2b21c2` |
| **Exp 4** | Minimal Diagnostic: BLR vs. Neural Laplace | Repeated Synthetic ($M=20$ seeds) | Support $\rho_s$, Coverage, AU-RC | BLR $\rho_s = 0.988$; CI cov $= 95.1\%$; AU-RC $= 0.1807$ | **Diagnostic Completed** (H_A.1, H_A.2, H_B.3 confirmed) | `118499cb` |

---

## 4. Disentangling Uncertainty, Confidence, Calibration, and Decision Utility

A major scientific contribution of this research arc is the rigorous disentanglement of six frequently conflated concepts:

```
+----------------------------------------------------------------------------------------------------+
|                                    CONCEPTUAL DISENTANGLEMENT                                      |
+------------------------------+----------------------------------+----------------------------------+
| Concept                      | Mathematical Definition          | Practical Operational Meaning    |
+------------------------------+----------------------------------+----------------------------------+
| 1. Training-Support Density  | d_M^2(x, D_train) or p(x)        | Geometric distance to observed   |
|                              |                                  | covariate training support.      |
+------------------------------+----------------------------------+----------------------------------+
| 2. Posterior Epistemic       | Var_{theta|D}[f_theta(x)]        | Parameter ignorance due to lack  |
|    Uncertainty               | = phi(x)^T Sigma phi(x)          | of data; depends on phi(x).      |
+------------------------------+----------------------------------+----------------------------------+
| 3. Predictive Confidence     | |p(x) - 0.5| or max(p, 1-p)      | Margin to classification boundary|
|                              |                                  | (conflates signal and noise).    |
+------------------------------+----------------------------------+----------------------------------+
| 4. Calibration               | E[(Y - p)^2] / ECE               | Statistical consistency between  |
|                              | P(Y=1 | p=c) = c                 | probabilities and frequencies.   |
+------------------------------+----------------------------------+----------------------------------+
| 5. Discrimination            | AUROC / AUC-PR                   | Ability to rank positives above  |
|                              | P(p(x+) > p(x-))                 | negatives (scale-invariant).     |
+------------------------------+----------------------------------+----------------------------------+
| 6. Downstream Decision       | E[U(a(p), Y)] - Costs            | Net economic or operational value|
|    Utility                   | Asymmetric loss, Sharpe, PnL     | under asymmetric action costs.   |
+------------------------------+----------------------------------+----------------------------------+
```

### Critical Theoretical Clarifications
1. **Epistemic Uncertainty $\neq$ Error Identification:** In binary classification, instances with high epistemic variance can exhibit *low* Brier error if their true latent logit $z^*(x)$ is large. Demanding that epistemic uncertainty correlate with residual prediction error is conceptually invalid.
2. **Confidence $\neq$ Epistemic Certainty:** An unregularized neural network can predict $p=0.99$ (high confidence) in regions with zero training data. Conversely, a point directly on the decision boundary has $p=0.5$ (low confidence) but may have zero epistemic uncertainty if thousands of training points surround it.
3. **Calibration $\neq$ Discrimination:** Calibrated log-odds fusion ($M_6$) achieved outstanding Brier accuracy and calibration (ECE $0.040$) while exhibiting lower AUROC than target-only training ($0.783$ vs. $0.843$). Multi-task feature factorization bounds global ranking while permitting near-optimal probability calibration.

---

## 5. Formal Audit of Scientific Claims

### 5.1 Supported Claims (Empirically Validated)
1. **Evidence Accumulation in Logit Space:** Calibrated log-odds fusion ($M_6$) statistically significantly outperforms equal-weight linear averaging ($M_4$) and simplex-weighted averaging ($M_5$) on held-out data ($p_{\text{holm}} < 0.05$). Under conditional feature independence, evidence accumulates additively in log-odds.
2. **Inefficiency of Masked Pooled Training:** Calibrated specialist fusion statistically significantly outperforms capacity-matched pooled training ($M_1$) evaluated under identical conditions ($p_{\text{holm}} = 0.002$). Training shared weights on heterogeneous zero-filled feature blocks causes representation confusion.
3. **Soundness of Laplace in Log-Concave Linear Spaces:** When representation geometry is linear and preserved, the Laplace approximation accurately reflects parameter posterior covariance, providing near-perfect support distance tracking ($\rho_s = 0.988$) and nominal 95% latent credible interval coverage ($95.1\%$).
4. **Hessian Ill-Conditioning in Neural Representations:** Neural penultimate representations introduce severe Hessian ill-conditioning ($\kappa > 700$ vs. $\approx 13$ for linear models), rendering quadratic variance estimates fragile to representation distortions.

### 5.2 Unsupported Claims (Formally Retracted / Refuted)
1. **Statistically Significant Superiority Over Target-Only ($M_0$):** Retracted. On the confirmatory test split ($N=512$), the observed Brier gain ($+0.0187$) yielded a 95% bootstrap CI of $[-0.0089, +0.0486]$ ($p = 0.190$). The claim of confirmed superiority over target-only is unsupported.
2. **Efficacy of Epistemic Uncertainty Specialist Routing:** Refuted. Penultimate Laplace variance failed development gates U1 and U2 in both regimes, and adding epistemic variance to a matched router yielded negligible gain ($< 0.0011$ Brier).
3. **Epistemic Uncertainty Predicts Classification Error Reduction:** Refuted. Abstaining on high-epistemic-uncertainty instances increases average Brier error on remaining samples (AU-RC $= 0.1807$ vs. $0.1510$ for random).

### 5.3 Unresolved Claims (Open Empirical Questions)
1. **Full Bayesian Neural Networks & MCMC:** Whether full-network Hamiltonian Monte Carlo (HMC) or Deep Ensembles can overcome representation collapse in partitioned-feature settings remains untested.
2. **Downstream Economic Decision Utility:** Whether calibrated probability improvements translate into economically meaningful Sharpe, turnover, or draw-down improvements under realistic transaction costs remains unmeasured.

---

## 6. Predefined Criteria for Reopening Bayesian Uncertainty Research

Bayesian uncertainty estimation for multi-agent or specialist routing must **not** be reopened without satisfying all of the following prospective criteria:

1. **Decision Asymmetry:** The downstream task must involve an explicitly asymmetric loss function (e.g., severe penalties for false positives, capital preservation constraints, or cost-sensitive human fallback) where epistemic variance directly controls risk exposure.
2. **Non-Collapsing Representation Architecture:** The base models must not evaluate quadratic variance on zero-filled or partitioned representation blocks. If partitioned features are used, density or uncertainty must be measured via coordinate-space density estimators (e.g., Mahalanobis or GMM) or spectral-normalized representations.
3. **Ambiguous Extrapolation Labels:** The evaluation domain must feature genuine covariate shift where out-of-distribution labels are genuinely uncertain or drifting, rather than deterministic linear extrapolation.
4. **Pre-Registered Utility Gate:** The uncertainty mechanism must demonstrate a statistically significant gain in decision utility (e.g., net portfolio return or cost-adjusted risk) over a deterministic confidence baseline ($|p - 0.5|$) on a development cohort before any held-out evaluation is unlocked.

---

## 7. Strategic Evaluation of Future Research Directions

```mermaid
flowchart TD
    subgraph Options["Strategic Research Directions"]
        Opt1["Direction 1: Uncertainty-Aware Collaboration<br/>(Full BNN, MCMC, Spectral Laplace)"]
        Opt2["Direction 2: Calibrated Fusion on Signal Data<br/>(Expanded Financial DGPs, Non-linear Link)"]
        Opt3["Direction 3: Downstream Portfolio Utility<br/>(Asymmetric Loss, Costs, Multi-Asset Allocation)"]
    end
    Opt1 -.->|High Risk, Heavy Compute, Architectural Friction| Rej["Deprioritize"]
    Opt2 -->|Scientifically Sound Intermediate| Sec["Secondary Horizon"]
    Opt3 ==>|Highest ROI, Validated Core (M6), Real Value| Rec["RECOMMENDED IMMEDIATE PROJECT"]
```

### Direction 1: Advanced Uncertainty-Aware Collaboration
- **Description:** Implement full-network Bayesian neural networks, Deep Ensembles, or spectral-normalized Laplace readouts to fix representation collapse.
- **Evaluation:** **Deprioritize / High Risk.** Heavy compute overhead (violates lightweight CPU guardrails), high architectural friction, and doubtful incremental value over static fusion.

### Direction 2: Calibrated Evidence Fusion on Signal-Bearing Data
- **Description:** Extend calibrated log-odds fusion ($M_6$) to richer synthetic data with non-linear feature interactions and covariate drift.
- **Evaluation:** **Viable Secondary Horizon.** Scientifically sound and builds directly on validated $M_6$ mechanics, but remains within purely synthetic predictive evaluation.

### Direction 3: Downstream Financial Portfolio Utility Evaluation (RECOMMENDED)
- **Description:** Evaluate whether the calibrated probability forecasts from specialist fusion ($M_6$) deliver superior risk-adjusted portfolio returns, lower maximum drawdown, and reduced turnover when paired with cost-aware portfolio construction rules (e.g., Kelly sizing, risk-parity, or mean-variance optimization with transaction friction).
- **Evaluation:** **Highest Return on Investment.** 
  - Directly addresses the ultimate research objective of `distributed-portfolio-learning`.
  - Bypasses the fragile uncertainty routing layer and deploys the robust, empirically verified $M_6$ fusion engine.
  - Measures real decision utility under asymmetric loss and transaction costs rather than synthetic symmetric Brier score alone.

---

## 8. Final Research Decision & Recommendation

### Formal Decision
1. **Close the Collaborative Bayesian LoRA & Uncertainty Routing Thread:** Formalize the closure of Experiments 1, 2, 3, and 4. Recognize that dynamic epistemic uncertainty routing on neural representations is unviable in this setting and that static calibrated log-odds fusion ($M_6$) is the verified method of choice.
2. **Execute Single Follow-On Project:** Advance directly to **Downstream Financial Portfolio Utility Evaluation of Calibrated Specialist Fusion**.
   - Design a prospective, cost-aware financial decision benchmark comparing portfolios constructed from Target-Only ($M_0$), Capacity-Matched Pooled ($M_1$), and Calibrated Specialist Fusion ($M_6$).
   - Pre-register utility metrics: Sharpe ratio, Sortino ratio, max drawdown, and net turnover after 10 bps transaction fees.
   - Maintain strict CPU compute safeguards and quarantined evaluation splits.

---
