# Bayesian Collaboration Positive-Control: Design Amendment

**Amends:** [`docs/bayesian_collaboration_research_plan.md`](bayesian_collaboration_research_plan.md)
(frozen at commit `6e7fe17b`, 2026-10-09)

**Date frozen:** 2026-10-09 (America/Los_Angeles)
**Status:** FROZEN before implementation. Do not modify after the first commit.

This amendment resolves five design decisions left open in the original plan,
adds scientific corrections that strengthen the integrity of the experiment,
and supersedes contradictory threshold language in the original plan.
The original plan itself is preserved unchanged at `6e7fe17b`.

---

## Design Decision 1 — Feature Masking

**Decision:** Zero-fill unavailable features and provide an explicit binary observation
mask that is concatenated to the input. All models receive the 40-dimensional vector
`concat([x_masked, obs_mask])` as input.

**Rationale:** Zero-filling preserves input dimensionality, which allows the same
architecture for all methods. The observation mask prevents the model from treating
structural zeros as genuine feature values and removes any ambiguity about which features
were observed. The mask is applied **before** any preprocessing; no normalization is
performed that could indirectly reveal masked values.

**Implementation contract:**

- DGP generates features from a standard normal distribution (mean 0, std 1).
  No standardization is applied at training time. Because features are already
  mean-zero, zero-filling does not introduce systematic bias toward either class.
- The observation mask $m \in \{0, 1\}^{d_{\text{total}}}$ is 1 for each observed
  feature and 0 for each zeroed feature.
- The model input is always `cat([x_masked, obs_mask])`, dimension 40.
- All methods — including pooled and CoLoRA — receive this 40-dimensional input;
  each training example carries its own mask determined by its task assignment.

**Information-access matrix:**

| Method | $x_s$ | $x_A$ | $x_B$ | $x_n$ | obs_mask | task_id | $N_{\text{train}}$ | Test input |
|---|---|---|---|---|---|---|---|---|
| M0 target-only | ✓ | ✓ | ✓ | ✓ | full | — | 128 | full mask |
| M1 pooled | mixed | mixed | mixed | ✓ | per-example | — | 640 | full mask |
| M2 specialist-A | ✓ | ✓ | 0 | ✓ | A-mask | — | 256 | A-mask |
| M3 specialist-B | ✓ | 0 | ✓ | ✓ | B-mask | — | 256 | B-mask |
| M4 equal ensemble | M2 + M3 test predictions | — | 3 fwd passes |
| M5 learned ensemble | M2 + M3 test predictions | — | 3 fwd passes |
| M6 CoLoRA | mixed | mixed | mixed | ✓ | per-example | ✓ target | 640 | full mask |
| M7 oracle (diag.) | dev-Brier-best of M2, M3 | — | 1 fwd pass |
| M8 prevalence null | — | — | 0 |

Notes:
- M1 pooled and M6 CoLoRA have 640 total training examples (128 target + 256 A + 256 B),
  more than M0 (128). This is intentional: pooled represents the data-pooling baseline.
- M2 and M3 test at inference time with their training-time masks (A-mask and B-mask
  respectively). This reflects their specialist identity.
- M6 (CoLoRA) uses the target task Lambda at inference (task_id = 0) with full features.
- M7 is an unreachable oracle (selected by test Brier). It is a diagnostic only.

**Improvement source attribution (positive-control regime):**

| Comparison | Source of improvement if any gain is observed |
|---|---|
| M0 → M1 | Additional training data + pooled representation learning |
| M0 → M4/M5 | Complementary information aggregation (each specialist sees a different feature partition) |
| M0 → M6 | Additional training data + task-specific routing |
| M4/M5 → M6 | Task-routing mechanism vs. simple prediction aggregation |

In the negative-control regime, specialists use full features; the same comparisons then
measure variance reduction from averaging (M4/M5) and data pooling plus routing (M6).

---

## Design Decision 2 — Learnability Threshold

**Decision:** Replace the contradictory threshold language in the original plan (which
stated both "Brier < 0.20" and "prevalence_null_brier − 0.015") with a single
**dynamic threshold** computed from the DGP configuration:

$$\theta_{\text{learn}} = \text{Brier}_{\text{null}} - 0.015$$

where $\text{Brier}_{\text{null}} = \text{prevalence}(1 - \text{prevalence})$.

At the target prevalence 0.35, $\text{Brier}_{\text{null}} = 0.35 \times 0.65 = 0.2275$,
giving $\theta_{\text{learn}} = 0.2125$.

**Hypothesis H2 (amended):** A specialist is learnable if its development-set Brier score
is strictly below $\theta_{\text{learn}}$ (i.e., Brier < 0.2125 at the configured prevalence).

**Rationale:** A fixed threshold of 0.20 is too tight at 35% prevalence, since the
irreducible Bayes-optimal Brier is approximately 0.202 (estimated analytically from the DGP).
A threshold of 0.20 could therefore fail even a Bayes-optimal specialist. The dynamic
threshold of $\text{Brier}_{\text{null}} - 0.015$ ensures the specialist must exceed
the null model by a meaningful margin without requiring Bayes-optimality.

---

## Design Decision 3 — CoLoRA Task Routing

**Decision:** CoLoRA uses an explicit integer task ID for Lambda selection in the
first-layer LoRA path. The task-ID-to-Lambda mapping is:

| task_id | Task at training time | Lambda at inference |
|---|---|---|
| 0 | target (full mask) | ✓ used at test inference |
| 1 | specialist A (A-mask) | not used at inference |
| 2 | specialist B (B-mask) | not used at inference |

**Rationale:** Implicit task routing (from the zero pattern alone) is not reliable at
small model sizes, and it conflates the architectural routing mechanism with the masking
signal that all methods already receive. Explicit task IDs give CoLoRA a routing
capability that other methods lack. This is an **input-access difference**; we do not
claim the comparison between M0–M5 and M6 is fully matched. The improvement source
attribution table (Decision 1) documents this explicitly.

**Informational note:** The obs_mask itself encodes task identity (A-mask and B-mask
are distinguishable), so all methods implicitly have access to task-identity information.
CoLoRA's advantage from explicit task IDs is the ability to route through a dedicated
per-task low-rank subspace, not the knowledge of which task it is.

---

## Design Decision 4 — Bayesian/UQ Weighting

**Decision:** Bayesian/UQ uncertainty weighting is **excluded** from this experiment.

**Rationale:** As documented in the original plan and the collaborative-LoRA branch
closure (`docs/collaborative_lora_branch_closure.md`), Bayesian specialist weighting is
conditional on first establishing deterministic collaborative gain. This experiment
is the prerequisite step. Including Bayesian weighting here would conflate the
prerequisite test with the downstream hypothesis.

---

## Design Decision 5 — Multiple-Comparison Correction

**Decision:** Apply **Holm step-down correction** to the confirmatory family of three
comparisons: {M4 (equal), M5 (learned), M6 (CoLoRA)} each vs. M0 (target-only).

**Holm procedure:**

1. Compute a one-sided bootstrap p-value for each comparison in the family:
   $p_k = $ proportion of 1,000 shared-event bootstrap resamples where the mean
   across seeds of $[\text{Brier}(M0) - \text{Brier}(M_k)]$ is $\leq 0$.

2. Sort p-values: $p_{(1)} \leq p_{(2)} \leq p_{(3)}$ with corresponding methods $k_{(1)}, k_{(2)}, k_{(3)}$.

3. Apply Holm step-down at family-wise error rate $\alpha = 0.05$:
   - Reject $H_0$ for $k_{(i)}$ if $p_{(j)} \leq \alpha / (K + 1 - j)$ for all $j \leq i$,
     where $K = 3$ (family size).
   - Corrected thresholds: 0.0167, 0.025, 0.05 for the first, second, and third ordered hypothesis.

4. Report: raw effect sizes, raw 95% two-sided CIs, raw p-values, Holm-corrected thresholds,
   and Holm-corrected pass/fail decisions.

**The continuation criterion** for reporting a positive result additionally requires:
- Mean Brier gain $\geq 0.005$ (pre-specified materiality threshold).
- Positive gain in all three seeds.
- Lower 95% raw CI strictly above zero.
- Holm-corrected decision: PASS (p-value ≤ Holm threshold for that method).

All four conditions must hold simultaneously. This is more stringent than the Holm
correction alone.

---

## Scientific Corrections

### Correction C1 — Complementarity criterion

**Original plan (H3):** "The error correlation between M2 and M3 on the development set is negative."

**Amendment:** The requirement for **negatively correlated specialist errors** is removed.
Negative error correlation is sufficient but not necessary for complementarity, and it is
an empirical proxy that can fail when models are poorly calibrated or when the development
set is small (n=64).

**Replacement criterion:** Complementarity is established independently through:

1. **Analytical oracle gap** (computed from DGP before any training):
   $\Delta_{\text{oracle}} = \min(\text{Brier}_{\text{Bayes,A}}, \text{Brier}_{\text{Bayes,B}}) - \text{Brier}_{\text{Bayes,full}}$

   where each Bayes-optimal Brier is computed by Monte Carlo over the known DGP distribution.
   If $\Delta_{\text{oracle}} > 0.010$, complementary information exists in the DGP by construction.

2. **Conditional information test** (on development set after training):
   Fit a logistic regression of $y$ on $[\hat{p}_A, \hat{p}_B]$ from development predictions.
   If both coefficients are positive, both specialists contribute information conditional
   on the other.

3. **Empirical oracle gap** (on development set):
   $\hat{\Delta}_{\text{dev}} = \text{Brier}(M0, \text{dev}) - \text{Brier}(M7, \text{dev})$

   where M7 is the dev-Brier-best of M2 and M3. A positive gap confirms that a single
   specialist can already improve over target-only on development data.

**Revised H3:** The complementarity check passes if:
- $\Delta_{\text{oracle}} > 0.010$ (analytical, always verifiable from DGP), AND
- at least one of: (a) both conditional-regression coefficients are positive on development,
  (b) $\hat{\Delta}_{\text{dev}} > 0$.

The analytical check is the primary gate; the empirical checks are informative diagnostics.

### Correction C2 — Negative-control expected outcome

**Amendment:** The negative control has a mathematically justified expected outcome.

In the negative-control regime, both specialists receive **full masks** (all features visible).
Both specialists train on the same-distribution feature space, approximating the same
Bayes-optimal predictor $p^*(x)$ from independent training samples.

**Expected outcome:**
- Both specialists converge toward $p^*(x)$ with estimation error $\epsilon \sim O(1/\sqrt{N})$.
- The average of two independent estimates $(\hat{p}_A + \hat{p}_B)/2$ reduces variance by
  approximately 50%, giving a Brier improvement of order $O(\sigma^2_\epsilon)$.
- For $N_{\text{train}} = 256$ examples and model capacity $\sim\!1800$ parameters, this
  variance-reduction gain is expected to be $\ll 0.005$.
- H5 (negative control null holds) tests that no method achieves the materiality threshold.

The negative control is thus expected to show small improvements from variance reduction,
not zero improvement, but well below the materiality threshold.

### Correction C3 — DGP independence

The DGP configuration and coefficients are fixed at generation time using a frozen seed.
They are **not revised based on model training outcomes**. The analytical oracle gap is
computed from the DGP coefficients directly, confirming complementarity before any model
is trained. This eliminates the circularity risk of tuning the DGP to guarantee CoLoRA wins.

### Correction C4 — Test set isolation

The test set is not accessed until the `evaluate` stage. The experiment harness enforces this:
- `generate`: saves raw data; no labels are consumed.
- `train`: accesses train/development/validation labels only.
- `diagnose`: checks H2 and H3 gates using development predictions only.
- `evaluate`: unlocks test label access only after H2 and H3 pass.

The `evaluate` stage refuses to run if `diagnose` has not produced a PASS report.

---

## Architecture Specification (Amendment to Section 8 of original plan)

**All non-CoLoRA methods:** `SmallMLP`
- Input: `concat([x_masked, obs_mask])` → 40 features
- Layer 1: Linear(40 → 32), ReLU
- Layer 2: Linear(32 → 16), ReLU
- Layer 3: Linear(16 → 1), Sigmoid
- Trainable parameters: 40×32+32 + 32×16+16 + 16×1+1 = **1,857**
- All parameters trained from random initialization (no frozen backbone)

**CoLoRA (M6):** `CoLoRAMLP`
- Input: same 40 features
- Layer 1 base: Linear(40 → 32), bias=True (shared across tasks, fully trainable)
- Layer 1 LoRA: `W_eff = W_base + (alpha/rank) × lora_B @ Lambda_task @ lora_A`
  where lora_A ∈ ℝ^{rank×40}, lora_B ∈ ℝ^{32×rank}, Lambda_task ∈ ℝ^{rank×rank}
- Layer 2: Linear(32 → 16), ReLU
- Layer 3: Linear(16 → 1), Sigmoid
- Trainable parameters: base(1312) + lora_A(160) + lora_B(128) + lambdas(3×16=48) + L2+L3(528+17) = **2,193**
  (extra 336 parameters vs SmallMLP, +18%; reported explicitly in all tables)

LoRA hyperparameters: rank=4, alpha=8. Lambda initialized to identity matrices.

**Training hyperparameters (all methods):**
- Optimizer: AdamW (lr=0.005, weight_decay=0.01, betas=(0.9, 0.999))
- Loss: binary cross-entropy
- Epochs: 100
- Batch size: 32
- Gradient clipping: max norm 1.0
- No early stopping; no development-set model selection

---

## Reproducibility Commands (Amendment to Section 11)

```bash
# All commands from repository root with PYTHONPATH=src
PYTHONPATH=src python experiments/run_bayesian_collab_positive_control.py --stage generate
PYTHONPATH=src python experiments/run_bayesian_collab_positive_control.py --stage train
PYTHONPATH=src python experiments/run_bayesian_collab_positive_control.py --stage diagnose
# (Do not run evaluate until diagnose reports PASS for both regimes)
PYTHONPATH=src python experiments/run_bayesian_collab_positive_control.py --stage evaluate
```

Unit tests (no torch required for DGP tests):
```bash
PYTHONPATH=src pytest tests/test_synthetic_multitask.py -v
PYTHONPATH=src pytest tests/test_bayesian_collab_positive_control.py -v
```

---

*This document is frozen. Do not modify after the first commit on
`experiment/bayesian-collaboration-positive-controls`.*
