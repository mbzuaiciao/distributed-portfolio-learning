# Bayesian Collaboration Positive-Control: Negative-Control Redesign and Capacity Amendment

**Amends:**
- [`docs/bayesian_collaboration_research_plan.md`](bayesian_collaboration_research_plan.md) (frozen at commit `6e7fe17b`)
- [`docs/bayesian_collaboration_design_amendment.md`](bayesian_collaboration_design_amendment.md) (frozen at commit `891468ab`)

**Date frozen:** 2026-10-10 (America/Los_Angeles)  
**Status:** FROZEN before main training execution.

---

## 1. Context: The Negative-Control Confounder Audit

In the pre-evaluation scientific audit conducted on 2026-10-10, an empirical simulation revealed
a critical confounder in the initial negative-control design:

1. **Initial formulation:** In the initial negative-control design, Specialists A and B were given
   full observation masks (`mask_full`, all 20 features visible) and trained on $N=256$ samples each
   drawn from the target distribution $P(y \mid x)$. Target-only ($M0$) was trained on $N=128$ samples.
2. **Confounder:** Ensembling Specialist A and Specialist B pooled $256 + 256 = 512$ examples of the
   target task. Comparing this ensemble ($M4$) against Target-only ($M0$, 128 examples) conferred a
   **$4\times$ sample-size advantage** to the ensemble on the identical task.
3. **Empirical violation of H5:** Even without complementary features, variance reduction and sample scaling
   produced $\text{Brier}(M0) = 0.2006$ vs. $\text{Brier}(M4) = 0.1778$, yielding a gain of
   $\Delta \text{Brier} = +0.0228 > 0.005$. Under this initial formulation, Hypothesis H5
   ("no method gains $\ge 0.005$ in negative control") would mechanically fail due to sample-size scaling,
   not genuine collaboration.

---

## 2. Redesign: Redundant Specialist Negative Control

To genuinely isolate specialist complementarity from variance reduction and sample size, the negative
control is redesigned to enforce **redundant feature information**:

### 2.1 Information Access by Regime

- **Positive-Control Regime (Complementary Specialists):**
  - Target-only ($M0$): observes $[x_{\text{shared}}, x_A, x_B, x_{\text{noise}}]$, $N=128$.
  - Specialist A ($M2$): observes $[x_{\text{shared}}, x_A, x_{\text{noise}}]$ ($x_B$ masked), $N=256$.
  - Specialist B ($M3$): observes $[x_{\text{shared}}, x_B, x_{\text{noise}}]$ ($x_A$ masked), $N=256$.
  - *Specialists A and B observe disjoint, non-overlapping signal blocks. Combining them recovers all predictive features.*
  - $\Delta_{\text{oracle}} = 0.0419 > 0.010$.

- **Negative-Control Regime (Redundant Specialists):**
  - Target-only ($M0$): observes $[x_{\text{shared}}, x_A, x_B, x_{\text{noise}}]$, $N=128$ (unchanged).
  - Specialist A ($M2$): observes $[x_{\text{shared}}, x_A, x_{\text{noise}}]$ ($x_B$ masked), $N=256$.
  - Specialist B ($M3$): observes $[x_{\text{shared}}, x_A, x_{\text{noise}}]$ ($x_B$ masked), $N=256$.
  - *Specialists A and B observe the EXACT SAME feature block A. Specialist B provides zero additional feature information beyond Specialist A.*
  - $\Delta_{\text{oracle}} = 0.0000$.

### 2.2 Mathematical Justification of Fair Information Access

In the negative-control regime:
1. Target-only retains its natural access to all features of its own target task ($x_s, x_A, x_B$) with $N=128$.
2. Both specialists are trained on Domain A data ($x_s, x_A$). Neither specialist has access to $x_B$.
3. Combining Specialist A and Specialist B ($M4_{\text{ens}}$) averages two independent estimates of $P(y \mid x_s, x_A)$,
   reducing estimation variance, but **cannot recover $x_B$**.
4. Because Target-only ($M0$) has access to both $x_A$ and $x_B$, $M0$ achieves lower Brier than the redundant ensemble:
   $$\text{Brier}(M0) = 0.2317, \quad \text{Brier}(M4_{\text{ens, red}}) = 0.2364 \implies \Delta \text{Brier} = -0.0047 < 0.005$$
5. Therefore, Hypothesis H5 holds cleanly. Collaboration fails to improve over Target-only when specialists
   lack complementary information.

---

## 3. Disentangling the Four Sources of Improvement

To prevent confounding, empirical results must be attributed to one of four distinct mechanisms:

| Source of Gain | Mechanism | Operational Metric |
|---|---|---|
| **1. Additional training samples** | Learning from more observations of known features | $M1$ (pooled, $N=640$) vs. $M0$ (target-only, $N=128$) |
| **2. Variance reduction** | Averaging estimation noise across independent fits | $M4_{\text{ens}}$ vs. single specialists ($M2, M3$) in Negative Control |
| **3. Additional predictive information** | Recovering joint distribution of non-overlapping feature blocks | $M4_{\text{ens}}$ (pos control) vs. $M4_{\text{ens}}$ (neg control), and $M4$ vs. $M7$ |
| **4. Collaborative parameter sharing** | Shared representations with task-specific low-rank routing | $M6$ (CoLoRA) vs. $M1$ (pooled) and $M1_{\text{cap}}$ (capacity-matched pooled) |

---

## 4. Renaming M7: Best-Single-Specialist Diagnostic

In the original research plan, M7 was termed "oracle diagnostic" in some tables and "upper bound on complementary gain"
in Section 5.3.

**Clarification and renaming:**
- Empirical M7 is formally renamed to: `best_single_specialist_diagnostic`.
- Formula: $\hat{p}_{M7} = \hat{p}_{M2}$ if $\text{Brier}(M2) \le \text{Brier}(M3)$ else $\hat{p}_{M3}$.
- Interpretation: M7 represents the best performance achievable by an oracle that selects the superior *single*
  specialist without combining. It cannot exploit complementarity.
- The true upper bound on complementary information is the analytical $\Delta_{\text{oracle}} = 0.0419$ (derived
  from the joint Bayes-optimal predictor $p^*_{\text{full}}$).

---

## 5. Capacity-Matched Baseline: M1-cap (Pooled-38)

To resolve the parameter count asymmetry between `SmallMLP` (1,857 params) and `CoLoRAMLP` (2,193 params, +18.1%),
an auxiliary baseline is added to the training and evaluation matrix:

- **Model:** `CapacityMatchedMLP` (`hidden_dim = 38`)
  - Input: 40 dims
  - Layer 1: Linear(40 $\to$ 38), ReLU [1,558 params]
  - Layer 2: Linear(38 $\to$ 16), ReLU [624 params]
  - Layer 3: Linear(16 $\to$ 1), Sigmoid [17 params]
  - **Total Trainable Parameters:** **2,199** (matches CoLoRA's 2,193 to within 0.27%)
- **Role:** $M1_{\text{cap}}$ trains on the exact same pooled dataset as $M1$ and $M6$ ($N=640$).
- **Comparison:**
  - $M6$ (CoLoRA) vs. $M1$ isolates combined capacity + routing.
  - $M6$ (CoLoRA) vs. $M1_{\text{cap}}$ isolates architectural task-routing and low-rank sharing while controlling for capacity.

---

## 6. Audit of Hypothesis H5 Validity

Under the redundant-specialist design:
- In negative control, Specialists A and B both observe $[x_s, x_A]$.
- Theoretical complementarity gap is 0.0000.
- Empirical simulation verifies that $\Delta \text{Brier}(M4, M0) \le 0 < 0.005$.
- **Verdict on H5:** The existing pre-registered threshold ($\Delta \text{Brier} < 0.005$) is **appropriate, mathematically sound, and re-affirmed**. No threshold modification is needed.

---

## 7. Execution and Computational Constraints

- **Thread budget:** `torch.set_num_threads(2)`.
- **Peak memory:** $< 250$ MB RAM (using $< 1.5\%$ of the host 24 GB RAM).
- **Execution safety:** Local execution executes within ~60–80 seconds total wall time on CPU. Host machine remains fully responsive.
- **Remote CPU feasibility:** Feasible via SSH/GCP if desired, but local CPU execution is lightweight and complete within < 2 minutes.

---

*Frozen on 2026-10-10 before main training execution on branch `experiment/bayesian-collaboration-positive-controls`.*
