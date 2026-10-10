# Pre-Confirmatory Scientific Audit Report: Experiment 2

**Target Commit:** `83851e50`  
**Branch:** `experiment/bayesian-collaboration-calibration-fusion`  
**Audit Date:** 2026-10-10 11:48 PDT  
**Audit Scope:** Data hygiene, split isolation, statistical claims, calibration/fusion mechanics, baseline fairness, contrast interpretation, Bayes oracle, parameter counting, and confirmatory freeze integrity.  
**Held-Out Test Set Status:** **FULLY ISOLATED & UNTOUCHED** (Zero access, zero evaluation).

---

## Executive Classification Summary

| Item | Focus Area | Status | Classification |
|---|---|---|---|
| **1** | Five-split disjointness & data hygiene | Verified disjoint, sequential PRNG sampling, zero leakage | **Acceptable Choice** |
| **2a** | Reported Brier gains & sign conventions | Gains correctly computed as $\text{Brier}_{\text{baseline}} - \text{Brier}_{\text{model}}$ | **Acceptable Choice** |
| **2b** | Confidence intervals & $p$-value claim | Unverified $p < 0.001$ in narrative; high sample loss variance at $N=128$ | **Blocker (Narrative Correction Required)** |
| **2c** | Repeated seed evaluations on shared test set | 3 seeds evaluated on identical 128 instances without cluster adjustment | **Limitation (Methodological Warning)** |
| **3** | Raw vs. Calibrated Log-Odds Fusion | Bivariate log-odds subsumes 1D Platt scaling; L2 penalty interaction | **Acceptable Choice / Scientific Finding** |
| **4** | Pooled baselines optimization & conditioning | 2,000 steps (fair); lacks explicit task ID token under zero-masking | **Limitation (Architectural Constraint)** |
| **5** | Matched Complementarity Contrast | Empirical subtractive diagnostic overclaimed as strict causal proof | **Limitation (Narrative Correction Required)** |
| **6a** | Analytical Bayes Oracle implementation | Exact ground-truth formulation on full target feature vector | **Acceptable Choice** |
| **6b** | Parameter count documentation in report table | Table reported 1,473 (from Exp 1 D=28) instead of 1,857 (Exp 2 D=40) | **Blocker (Documentation Correction Required)** |
| **7** | Confirmatory configuration freeze | Configuration frozen; zero post-hoc changes | **Acceptable Choice** |

---

## Detailed Audit Findings

### 1. Data Hygiene & Five-Split Disjointness
- **Implementation Audit:** Verified in `src/hypergraph_portfolio/synthetic_multitask_exp2.py` (`sample_exp2_dataset`).
- **Splits:**
  - `target_train`: $N=128$ (drawn first from `data_seed=20261011`)
  - `specialist_A_train`: $N=256$ (drawn second)
  - `specialist_B_train`: $N=256$ (drawn third)
  - `calibration`: $N=128$ (drawn fourth)
  - `fusion_dev`: $N=128$ (drawn fifth)
  - `development`: $N=128$ (drawn sixth)
  - `test`: $N=512$ (drawn seventh)
- **Disjointness:** In continuous Gaussian feature space ($\mathbb{R}^{20}$), consecutive pseudorandom draws from a single generator are almost surely disjoint.
- **Fitting Isolation:**
  - `PlattCalibrator.fit` is called exclusively with predictions on `data["calibration"]`.
  - `LogOddsFusion.fit` and `SimplexFusion.fit` are called exclusively with predictions on `data["fusion_dev"]`.
  - Development evaluation strictly queries `data["development"]`.
  - Keys `test_x_full`, `test_x_A`, `test_x_B`, `test_y` are never indexed in `experiments/run_bayesian_collab_experiment2.py`.
- **Verdict:** **Acceptable Choice**. Zero data leakage across splits.

---

### 2. Statistical Claims, Seed Handling, & Confidence Intervals
- **Reported Brier Gains:**
  - In positive control, $\text{Gain} = \overline{\text{Brier}}(M0) - \overline{\text{Brier}}(M6_{\text{cal}}) = 0.213968 - 0.187607 = \mathbf{+0.026360}$.
  - In negative control, $\text{Gain} = 0.213968 - 0.199965 = \mathbf{+0.014003}$.
  - Contrast: $+0.026360 - 0.014003 = \mathbf{+0.012357}$.
  - The gains and signs are mathematically exact.
- **Audit of $p < 0.001$ Claim:**
  - In `development_report_exp2.md`, Section 1 Table stated: `Gain = +0.0264 Brier (p < 0.001)`.
  - **Finding:** No formal hypothesis test or p-value calculation exists in `run_bayesian_collab_experiment2.py`.
  - When recomputed on the $N=128$ development instances:
    - Instance-level loss difference $d_i = (p_{0, i} - y_i)^2 - (p_{6, i} - y_i)^2$ has high empirical standard deviation ($\hat{\sigma} \approx 0.301$).
    - Standard error of the mean on $N=128$ instances is $\text{SE} \approx 0.301 / \sqrt{128} \approx 0.0266$.
    - The observed gain of $0.0264$ is approximately $1.0 \, \text{SE}$.
    - A paired $t$-test yields $t \approx 0.99, p \approx 0.32$.
    - The 95% bootstrap confidence interval on the development set is $[-0.026, +0.079]$, which includes zero.
  - **Verdict:** **Blocker (Narrative Correction)**. The text claiming $p < 0.001$ must be retracted and corrected. On $N=128$ dev instances, the point estimate indicates a healthy $+0.0264$ margin over baseline, but the sample size is underpowered for asymptotic $p < 0.001$. Confirmatory evaluation on $N=512$ will provide $2\times$ tighter confidence intervals.
- **Seed Handling:**
  - The script evaluates 3 seeds on the identical 128 instances and reports the mean of seed scores.
  - While reporting the average across seeds is standard practice to marginalize optimizer variance, treating $3 \times 128 = 384$ predictions as independent observations would be a pseudo-replication error.
  - **Verdict:** **Limitation**. For confirmatory evaluation, hypothesis tests must use instance-level clustering or ensemble-averaged instance bootstrap.

---

### 3. Investigation: Raw (0.1872) vs. Calibrated (0.1876) Log-Odds Fusion
- **Mathematical Equivalence:**
  - Let raw specialist logits be $\ell_A = \text{logit}(p_A)$ and $\ell_B = \text{logit}(p_B)$.
  - Univariate Platt scaling fits affine transformations: $\ell_{A,\text{cal}} = a_A \ell_A + b_A$ and $\ell_{B,\text{cal}} = a_B \ell_B + b_B$.
  - Bivariate log-odds fusion fits:
    $$z = w_0 + w_A \ell_A + w_B \ell_B$$
  - When log-odds fusion is applied to calibrated logits:
    $$z = w_0 + w_A (a_A \ell_A + b_A) + w_B (a_B \ell_B + b_B) = (w_0 + w_A b_A + w_B b_B) + (w_A a_A) \ell_A + (w_B a_B) \ell_B$$
  - Under unregularized estimation ($C \to \infty$), the hypothesis class of bivariate linear models over $(\ell_A, \ell_B)$ is **identical** to the hypothesis class over $(\ell_{A,\text{cal}}, \ell_{B,\text{cal}})$.
- **Empirical Cause of the $0.0004$ Discrepancy:**
  - `LogOddsFusion.fit` uses `LogisticRegression(C=1.0)` (mild L2 penalty $\frac{1}{2} \|\mathbf{w}\|^2$).
  - For raw logits, logits are large ($|\ell| \in [2, 6]$), so optimal weights are small ($w_A \approx 0.096, w_B \approx 0.093$). The L2 penalty $\frac{1}{2}(0.096^2 + 0.093^2) \approx 0.009$ is negligible.
  - For calibrated logits, logits are pre-compressed by Platt scaling ($a \approx 0.07$), so the fusion model requires larger weights ($w_A \approx 1.28, w_B \approx 0.88$). Here, the L2 penalty $\frac{1}{2}(1.28^2 + 0.88^2) \approx 1.21$ induces slightly stronger shrinkage.
  - Furthermore, two-stage estimation splits sample information ($N=128$ for Platt, $N=128$ for fusion), while raw log-odds estimates scale and combination simultaneously on the fusion split.
- **Separation of Calibration vs. Fusion:**
  - **For Linear Ensembling (M4, M5):** Individual calibration is critical. Calibration reduces Equal-Weight Brier from $0.2137$ to $0.1972$ ($\Delta = -0.0165$) and Simplex Brier from $0.2158$ to $0.1990$ ($\Delta = -0.0168$) because linear mixtures cannot correct for extreme logits.
  - **For Log-Odds Fusion (M6):** The bivariate logistic model inherently performs calibration (scaling and intercept shift) while simultaneously fusing evidence. Pre-calibration is functionally redundant.
- **Verdict:** **Acceptable Choice / Key Scientific Finding**.

---

### 4. Pooled & Capacity-Matched Pooled Baseline Fairness
- **Optimization Budget:**
  - Union dataset size is $N=640$ ($128 + 256 + 256$).
  - At batch size 32 over 100 epochs, pooled models receive $20 \text{ steps/epoch} \times 100 = 2,000 \text{ gradient steps}$, compared to 400 steps for Target-only ($N=128$).
  - Optimization steps are more than sufficient for convergence.
- **Task Conditioning & Observation Masking:**
  - Input to pooled models is a 40-dimensional vector: $[x_{\text{masked}}, \text{mask}]$.
  - While the observation mask implicitly identifies the task (e.g. mask for A has block B zeroed), the pooled models do not receive an explicit one-hot task embedding.
  - In contrast, CoLoRA receives explicit integer task IDs ($0, 1, 2$) routed to task-specific parameter matrices $\Lambda_t$.
  - 80% of the pooled training set ($512/640$ samples) contains missing feature blocks. With unweighted BCE, the network's shared layers are heavily updated to predict without $x_A$ or without $x_B$. When evaluated on target $x_{\text{full}}$, these conflicting marginal gradients degrade full-information prediction (Brier $\approx 0.246 - 0.249$).
- **Verdict:** **Limitation**. The poor performance of pooled models reflects genuine cross-task interference and mask confusion in monolithic architectures, but adding explicit task conditioning would provide a more competitive multi-task pooling baseline.

---

### 5. Matched Complementarity Contrast & Causal Attribution
- **Contrast Formulation:**
  $$\Delta_{\text{contrast}} = \text{Gain}_{\text{positive}} - \text{Gain}_{\text{negative}} = 0.0264 - 0.0140 = \mathbf{+0.0124}$$
- **Audit of Claims:**
  - The positive control provides complementary features ($X_A$ vs $X_B$).
  - The negative control provides redundant features ($X_A$ vs $X_A$).
  - The gain in the negative control ($+0.0140$) reflects pure sample-size expansion and ensembling variance reduction.
  - The fact that the positive control gain ($+0.0264$) exceeds the negative control gain demonstrates that complementary predictive information provides an incremental advantage over variance reduction alone.
  - **However:** In `development_report_exp2.md`, this is described as "formally confirming that the collaborative gain is causally driven by complementary predictive feature information."
  - This overstates causal identification: changing the feature block from $X_B$ to $X_A$ also changes the marginal logit variance, specialist training dynamics, and calibration parameter values between regimes.
- **Verdict:** **Limitation (Narrative Correction)**. The contrast is a valid empirical subtractive diagnostic, but should not be framed as a formal proof of causal isolation.

---

### 6. Bayes Oracle & Parameter Count Labeling
- **Bayes Oracle Correctness:**
  - The DGP generates labels from $p^* = \sigma(x_s \beta_s + x_A \beta_A + x_B \beta_B + b)$.
  - In `experiments/run_bayesian_collab_experiment2.py` (lines 533–538), the oracle prediction on `dev_x_full` is computed directly using the ground-truth coefficients and intercept.
  - Dev Brier for $M9_{\text{oracle}}$ is $0.1382$, matching the analytical expectation ($0.1448 \pm \text{finite-sample error}$).
  - **Verdict:** **Acceptable Choice**.
- **Parameter Count Labeling Audit:**
  - Let $D = 40$ (20 features + 20 masks).
  - `SmallMLP`:
    - Layer 1: $40 \times 32 + 32 = 1,312$
    - Layer 2: $32 \times 16 + 16 = 528$
    - Layer 3: $16 \times 1 + 1 = 17$
    - **Total: 1,857 parameters**.
  - `CapacityMatchedMLP`:
    - Layer 1: $40 \times 38 + 38 = 1,558$
    - Layer 2: $38 \times 16 + 16 = 624$
    - Layer 3: $16 \times 1 + 1 = 17$
    - **Total: 2,199 parameters**.
  - `CoLoRAMLP` ($r=4, \alpha=8, \text{tasks}=3$):
    - Base layer: $1,312$; LoRA A: $160$; LoRA B: $128$; Lambdas: $48$; FC2: $528$; FC3: $17$.
    - **Total: 2,193 parameters**.
  - In `docs/bayesian_collaboration_experiment2_protocol.md`, parameter counts were correctly listed as **1,857**, **2,199**, and **2,193**.
  - In `artifacts/bayesian_collab_exp2/development_report_exp2.md`, Table 1 incorrectly reported **1,473** for SmallMLP (and 2,946 for ensembles), which was a legacy figure carried over from Experiment 1's 28-dimensional input.
- **Verdict:** **Blocker (Documentation Correction Required)**. The table in `development_report_exp2.md` must be updated to state the correct parameter counts (1,857 for SmallMLP; 3,714 for two-specialist raw ensemble; 3,723 for calibrated log-odds system).

---

### 7. Confirmatory Configuration Freeze & Tampering Check
- **Audit:**
  - Verified git commit `83851e50`.
  - Inspected `config_exp2.json` and `docs/bayesian_collaboration_experiment2_protocol.md`.
  - Confirmed that neither the DGP parameters, sample sizes, random seeds (`42, 123, 777`), nor hypothesis thresholds ($\ge 0.010$ gain, $> 0.0$ contrast) were adjusted after observing development results.
- **Verdict:** **Acceptable Choice**. Complete frozen configuration integrity.

---

## Required Remediation Actions Before Confirmatory Run

1. **Retract Unsubstantiated $p$-value Claim:** Remove `(p < 0.001)` from `development_report_exp2.md`; document empirical standard error on $N=128$ development instances.
2. **Correct Parameter Counts in Documentation:** Update Table 1 of `development_report_exp2.md` to reflect the actual 40-input parameter count ($1,857$ for SmallMLP, $3,714$ for two-model ensembles).
3. **Tone Down Causal Language:** Replace claims of "causal proof" with "matched empirical diagnostic contrast."
4. **Pre-Specify Cluster Bootstrap for Held-Out Evaluation:** Ensure the confirmatory script runs a clustered instance bootstrap across the $N=512$ held-out test instances rather than a naive paired test over seed-replicates.
