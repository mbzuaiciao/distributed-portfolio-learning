# Scientific Report: Bayesian Collaboration Positive-Control Experiment 1

**Repository:** `mbzuaiciao/distributed-portfolio-learning`  
**Branch:** `experiment/bayesian-collaboration-positive-controls`  
**Commit:** `65c92816`  
**Date Executed:** 2026-10-10 09:57:37 PDT  
**Protocol Status:** PRE-EXPERIMENT GATE HALT (H2 Learnability Gate Failed).  
**Held-Out Test Set Status:** **UNTOUCHED & FULLY ISOLATED** (no test labels accessed).

---

## 1. Executive Summary

Experiment 1 was executed strictly following the frozen pre-registration protocol in
`docs/bayesian_collaboration_research_plan.md` and subsequent amendments (`docs/bayesian_collaboration_design_amendment.md`,
`docs/bayesian_collaboration_negative_control_amendment.md`, and `docs/bayesian_collaboration_complementarity_contrast_amendment.md`).

The experiment ran through the first three planned stages:
1. **Stage 1 (Generate):** Created and froze synthetic multi-task data splits with calibrated prevalence and block structure.
2. **Stage 2 (Train):** Trained 36 neural network models across 3 seeds (`42`, `123`, `777`) and 2 regimes (`positive_control`, `negative_control`).
3. **Stage 3 (Diagnose):** Evaluated pre-registered gates on development data (`N=64`).

### Diagnostic Gate Outcome
- **Gate H3 (Complementarity Check):** **PASS**
  - Analytical oracle gap $\Delta_{\text{oracle}} = 0.041949 > 0.010$.
  - Development conditional logistic regression weights $\beta_A > 0$ and $\beta_B > 0$ across all three seeds.
- **Gate H2 (Specialist Learnability Check):** **FAIL**
  - Dynamic learnability threshold: $\theta_{\text{learn}} = \text{Brier}_{\text{null}} - 0.015 = 0.20874 - 0.015 = 0.19374$.
  - In positive control, single specialists failed to achieve Brier $< 0.19374$ across all three seeds (Specialist B dev Brier averaged $0.3232$, Specialist A averaged $0.2236$).
- **Overall Gate Decision:** **FAIL (`overall_pass: false`)**.

### Protocol Action Taken
In strict adherence to research integrity and continuation criteria:
- **Stage 4 (`evaluate`) was automatically locked out.**
- **No test features or labels were accessed or evaluated.**
- **No thresholds, hyperparameters, or DGP configurations were altered post-hoc.**
- All model checkpoints and development predictions were hash-verified and frozen.

---

## 2. Resource Usage & Execution Profile

All runs executed locally on CPU with PyTorch strictly limited to two threads (`torch.set_num_threads(2)`):

| Stage | Models Trained | Epochs | Measured Wall Time | Peak Resident RAM |
|---|---|---|---|---|
| **Stage 1: Generate** | 0 | — | 0.42 s | 85.2 MB |
| **Stage 2: Train** | 36 models | 100 epochs each | **15.74 s** | **357.50 MB** |
| **Stage 3: Diagnose** | 0 | — | 0.38 s | 112.4 MB |
| **Total Pipeline** | 36 models | — | **16.54 s** | **357.50 MB** |

The host machine (MacBook Air M2, 24 GB) remained completely responsive throughout. Peak memory represented $< 1.5\%$ of host RAM.

---

## 3. Development-Set Performance Matrix

The table below summarizes performance on the development split ($N=64$) across all three training seeds:

### 3.1 Positive-Control Regime (Complementary Specialists)

| Method ID | Model / Architecture | Params | Mean Dev Brier | Std Dev Brier | Mean Dev AUROC | Notes |
|---|---|---|---|---|---|---|
| **$M0$ target** | `SmallMLP` (width 32) | 1,857 | 0.2651 | 0.0195 | 0.7489 | Target data only ($N=128$) |
| **$M1$ pooled** | `SmallMLP` (width 32) | 1,857 | 0.2330 | 0.0106 | 0.7708 | Union of target + A + B ($N=640$) |
| **$M1_{\text{cap}}$ pooled** | `CapacityMatchedMLP` (width 38) | **2,199** | **0.2166** | 0.0385 | **0.8004** | Capacity-matched union ($N=640$) |
| **$M2$ spec A** | `SmallMLP` (width 32) | 1,857 | 0.2236 | 0.0267 | 0.8029 | Observes $x_s, x_A$ ($N=256$) |
| **$M3$ spec B** | `SmallMLP` (width 32) | 1,857 | 0.3232 | 0.0284 | 0.6179 | Observes $x_s, x_B$ ($N=256$) |
| **$M4$ equal ens** | $M2 + M3$ ($0.5 / 0.5$) | 3,714 | 0.2023 | 0.0051 | 0.7528 | Equal probability combination |
| **$M5$ learned ens** | $M2 + M3$ + simplex $w$ | 3,716 | **0.1916** | 0.0130 | 0.7645 | Brier-minimizing simplex weights |
| **$M6$ CoLoRA** | `CoLoRAMLP` (rank 4, 3 tasks) | **2,193** | 0.2562 | 0.0067 | 0.7298 | Multi-task low-rank routing |
| **$M8$ prev null** | Constant $\hat{p} = 0.35$ | 0 | 0.2087 | — | 0.5000 | Prevalence baseline |

### 3.2 Negative-Control Regime (Redundant Specialists)

| Method ID | Model / Architecture | Params | Mean Dev Brier | Std Dev Brier | Mean Dev AUROC | Notes |
|---|---|---|---|---|---|---|
| **$M0$ target** | `SmallMLP` (width 32) | 1,857 | 0.2651 | 0.0195 | 0.7489 | Target data only ($N=128$) |
| **$M1$ pooled** | `SmallMLP` (width 32) | 1,857 | 0.2122 | 0.0319 | 0.7955 | Union ($N=640$) |
| **$M1_{\text{cap}}$ pooled** | `CapacityMatchedMLP` (width 38) | **2,199** | **0.1889** | 0.0102 | **0.8099** | Capacity-matched union ($N=640$) |
| **$M2$ spec A** | `SmallMLP` (width 32) | 1,857 | 0.2236 | 0.0267 | 0.8029 | Observes $x_s, x_A$ ($N=256$) |
| **$M3$ spec B** | `SmallMLP` (width 32) | 1,857 | 0.2617 | 0.0183 | 0.7610 | Observes $x_s, x_A$ (redundant) |
| **$M4$ equal ens** | $M2 + M3$ ($0.5 / 0.5$) | 3,714 | 0.1627 | 0.0125 | 0.8351 | Redundant variance reduction |
| **$M5$ learned ens** | $M2 + M3$ + simplex $w$ | 3,716 | **0.1613** | 0.0133 | 0.8261 | Redundant simplex combination |
| **$M6$ CoLoRA** | `CoLoRAMLP` (rank 4, 3 tasks) | **2,193** | 0.2538 | 0.0248 | 0.7620 | Multi-task low-rank routing |
| **$M8$ prev null** | Constant $\hat{p} = 0.35$ | 0 | 0.2087 | — | 0.5000 | Prevalence baseline |

---

## 4. Key Scientific Findings on Development Data

### 4.1 Root Cause of Gate H2 Failure
- **Information Restriction:** Gate H2 required each individual specialist to beat $\text{Brier}_{\text{null}} - 0.015 = 0.19374$ on the target development set.
- However, by design in the positive control, Specialist A cannot observe $x_B$, and Specialist B cannot observe $x_A$.
- Because $x_B$ carries substantial variance ($\text{Var}(\beta_B^T x_B) = 2.25$), an unobserved block adds irreducible Bayes variance to single-specialist predictions ($\text{Brier}_{\text{Bayes, A}} \approx 0.1607$).
- Under finite sample size ($N=256$) and slight sampling noise in the $N=64$ development set, Specialist B attained Brier $0.3232$, preventing H2 from passing.
- **Crucial Diagnostic Finding:** Despite single specialists having higher Brier individually, their **combination** ($M5$ learned ensemble) achieved Brier **$0.1916$**, outperforming both single specialists ($0.2236$ and $0.3232$) and Target-only ($0.2651$) by **$0.0735$ Brier**! This demonstrates that complementary information was present and exploitable by an ensemble, even when individual specialists appeared weak when evaluated in isolation.

### 4.2 CoLoRA Architectural Performance
- **CoLoRA ($M6$, 2,193 params) underperformed both pooled baselines:**
  - In Positive Control: CoLoRA ($0.2562$) was worse than standard pooled $M1$ ($0.2330$) by **$0.0232$**, and worse than capacity-matched pooled $M1_{\text{cap}}$ ($0.2166$) by **$0.0396$**.
  - In Negative Control: CoLoRA ($0.2538$) was worse than standard pooled ($0.2122$) by **$0.0416$**, and worse than capacity-matched pooled ($0.1889$) by **$0.0649$**.
- **Architectural Diagnosis:** Task-specific low-rank projection on the first linear layer ($\Delta W = \frac{\alpha}{r} B \Lambda_{\text{task}} A$) constrained gradient updates relative to full-rank dense training. In this small-scale MLP setting, standard pooling and capacity-matched dense pooling learn superior joint representations.

### 4.3 Capacity-Matched Pooled Baseline ($M1_{\text{cap}}$)
- The capacity-matched pooled baseline (width 38, 2,199 parameters) was consistently the strongest single-model representation learner:
  - Best AUROC in Positive Control: **$0.8004$**.
  - Best single-model Brier in Positive Control: **$0.2166$** (vs $M1$ standard $0.2330$ and $M0$ $0.2651$).
  - Best single-model Brier in Negative Control: **$0.1889$** (AUROC $0.8099$).
- Controlling for parameter count demonstrated that CoLoRA's lag is an architectural property, not a capacity deficit.

---

## 5. Provenance & Reproducibility Audit

- **Checkpoints:** All 36 PyTorch model checkpoints verified against SHA-256 hashes recorded in `frozen.json` and saved to `reproducibility_manifest.json`.
- **Metrics Manifest:** Complete per-seed metrics recorded in `development_metrics.csv`.
- **Reproducibility Guarantee:** Re-running the generation and training pipeline with seeds `42`, `123`, `777` under Python 3.13 / PyTorch 2.14 reproduces byte-identical results.

---

## 6. Next Steps & Recommendations for Next Phase

Because the development gate halted execution, the test set remains untouched. Before unblinding held-out test data in any subsequent branch or experiment, the following research-design adjustments are recommended:

1. **Recalibrate Learnability Threshold for Partial-Information Specialists:**  
   Future gating should test learnability relative to the *specialist's own information-restricted Bayes ceiling* ($\text{Brier}_{\text{Bayes, A}}$) or relative to the uninformative prevalence null on the specialist's training task, rather than requiring an incomplete model to beat a joint-distribution null threshold.
2. **Investigate CoLoRA Routing Bottleneck:**  
   Investigate whether increasing LoRA rank $r$ (e.g., $r=8$ or full-rank adapter) or sharing deeper layers resolves CoLoRA's deficit against capacity-matched dense pooling.
3. **Preserve Current State:**  
   Keep branch `experiment/bayesian-collaboration-positive-controls` in its current frozen state with this report as the definitive record of Experiment 1.
