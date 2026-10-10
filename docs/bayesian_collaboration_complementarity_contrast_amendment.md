# Bayesian Collaboration Positive-Control: Complementarity Contrast & Diagnostic Protocol Amendment

**Amends:**
- [`docs/bayesian_collaboration_research_plan.md`](bayesian_collaboration_research_plan.md) (commit `6e7fe17b`)
- [`docs/bayesian_collaboration_design_amendment.md`](bayesian_collaboration_design_amendment.md) (commit `891468ab`)
- [`docs/bayesian_collaboration_negative_control_amendment.md`](bayesian_collaboration_negative_control_amendment.md) (commit `20a22c56`)

**Date frozen:** 2026-10-10 (America/Los_Angeles)  
**Status:** FROZEN before main training execution.

---

## 1. Scope & Epistemic Principles

### 1.1 Redundant Control Tests Feature Complementarity, Not Absolute Invariance
The redundant-specialist negative control is designed to enforce the **absence of complementary predictive features**
($\Delta_{\text{oracle}} = 0.0000$). It does **not** assume or require that ensembling two independently trained models
yields zero variance reduction. When Specialist A and Specialist B both observe $[x_{\text{shared}}, x_A, x_{\text{noise}}]$,
averaging their predictions can still dampen stochastic estimation noise.

### 1.2 Non-Proof of Single Empirical Differences
A single point estimate from a scratch simulation (e.g., $\Delta \text{Brier} = -0.0047$) is **not mathematical proof**
that H5 will pass across all three seeds and bootstrap resamples in the frozen evaluation. Finite-sample variance,
batch permutation order, and seed variability mean empirical gains are random variables.

---

## 2. Reclassification of H5 as a Descriptive Diagnostic

### 2.1 Limitation of a Rigid Scalar Threshold
Hypothesis H5 was originally written with a rigid scalar threshold:
$$\text{Gain}_{\text{negative}}(M_k \text{ vs } M0) < 0.005$$
Because $M0$ has $N=128$ while the specialist ensemble $M4$ combines two $N=256$ models ($N_{\text{total}}=512$),
the difference $\text{Brier}(M0) - \text{Brier}(M4_{\text{neg}})$ reflects three competing components:
$$\Delta \text{Brier}_{\text{neg}} = \text{Gain}(\text{sample size } 512 \text{ vs } 128) + \text{Gain}(\text{variance reduction}) - \text{Penalty}(\text{unobserved } x_B)$$

If the unobserved $x_B$ penalty slightly fluctuates in finite samples, $\Delta \text{Brier}_{\text{neg}}$ could take values
near the boundary of 0.005. Treating a boundary crossing as a rigid "mechanistic falsification" would conflate
sample-size/variance-reduction dynamics with the presence or absence of feature complementarity.

### 2.2 Protocol Adjustment: H5 as Descriptive Diagnostic
1. **H5 is formally classified as a Descriptive Diagnostic Control**:
   - In the negative control, we report the empirical mean Brier gain, 95% bootstrap CI, and whether it falls below the nominal 0.005 threshold.
   - It is monitored as a calibration diagnostic rather than an automatic falsification gate that aborts scientific reporting.
2. **Primary Mechanistic Measure: Matched Complementarity Contrast**:
   To strictly isolate the effect of **complementary feature information**, we define the matched difference-in-differences contrast:
   $$\Delta \text{Contrast}(M_k) = \Delta \text{Brier}_{\text{positive}}(M_k, M_0) - \Delta \text{Brier}_{\text{negative}}(M_k, M_0)$$
   where $\Delta \text{Brier}(M_k, M_0) = \text{Brier}(M_0) - \text{Brier}(M_k)$ (positive means $M_k$ improves over $M_0$).

#### Interpretation of Matched Contrast
- In Positive Control: Specialists hold non-overlapping features ($x_A$ and $x_B$).
- In Negative Control: Specialists hold redundant features ($x_A$ and $x_A$).
- Both regimes share identical sample sizes ($N_{\text{tgt}}=128, N_A=256, N_B=256$), identical architectures, identical optimization schedules, and identical ensembling rules.
- Subtracting $\Delta \text{Brier}_{\text{negative}}$ cancels out the shared benefits of sample scaling ($N=512$ vs $128$) and stochastic variance reduction, isolating the **pure marginal value of specialist feature complementarity**.
- A positive contrast $\Delta \text{Contrast} > 0$ proves that collaborative ensembling gained more when specialists held complementary information than when they held redundant information.

---

## 3. Capacity-Matched Baseline Comparison Protocol

To rigorously assess CoLoRA ($M6$, 2,193 parameters):
1. **$M6$ vs. $M1$ (Standard Pooled, 1,857 parameters):** Measures total gain from pooling + LoRA architecture (+18.1% parameters).
2. **$M6$ vs. $M1_{\text{cap}}$ (Capacity-Matched Pooled, 2,199 parameters):** Measures the pure architectural effect of shared low-rank representations and task-specific $\Lambda_{\text{task}}$ routing, controlling for parameter count within 0.27%.
3. Both comparisons are computed via paired shared-event bootstrap resamples across seeds and reported in `evaluation_summary.json`.

---

## 4. Evaluation Gate Integrity & Resource Policy

- **Gating Integrity:** `stage_diagnose` gates (H2 learnability on dev set, H3 analytical oracle gap > 0.010, and dev conditional regression) remain strictly required before test-set evaluation can proceed.
- **Test-Set Isolation:** Held-out test labels remain untouched.
- **Host Responsiveness Constraint:** Local execution is capped at 2 CPU threads (`torch.set_num_threads(2)`), keeping peak RAM $< 250$ MB. Main training and held-out evaluation will not run without explicit user authorization.

---

*Frozen on 2026-10-10 before main training execution on branch `experiment/bayesian-collaboration-positive-controls`.*
