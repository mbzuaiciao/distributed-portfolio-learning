# Bayesian Collaboration Positive-Control Research Plan

**Branch:** `experiment/bayesian-collaboration-positive-controls`  
**Date frozen:** 2026-10-09 (America/Los_Angeles)  
**Status:** PRE-REGISTRATION — no experiment has been executed. Do not modify after first commit.

---

## 1. Context and Motivation

### Prior negative finding

The frozen collaborative-LoRA pilot
(`artifacts/collaborative_lora_mechanism_pilot/`, closed 2026-10-04)
established no collaborative gain on real FinBERT/earnings-event data.
Key pathologies:

| Pathology | Evidence |
|---|---|
| Learned weights collapsed to one specialist | Banking weight ≈ 1.0 across all three seeds |
| Energy/consumer reduced performance | Negative marginal Brier contribution under equal weighting |
| Target-only beat all multi-source methods on mean Brier | Mean Brier: target 0.1937, pooled 0.1964, CoLoRA 0.1964 |
| Prevalence null beat all trained models on Brier/log-loss | Null Brier 0.1841 vs. best trained 0.1921 |
| Near-chance AUROC for most methods | Mean AUROC range 0.46–0.53 |

The branch closure states:
> The branch is closed unless a materially different task regime or data-generating setup
> provides evidence of exploitable specialist complementarity.

The critical ambiguity from the pilot is whether the failure reflects:

- **(A) Absent complementary information** — the source tasks genuinely contain no useful signal for the target.
- **(B) Training failure** — complementary information exists in principle but the LoRA adapters fail to learn and transfer it at this scale (128 examples, 3 epochs, one backbone).
- **(C) Model failure** — the backbone/architecture cannot route task-specific representations to a shared head at this rank/alpha.

The pilot cannot distinguish (A) from (B) and (C) because the real data has no known ground truth for complementarity. **This is the core motivation for a synthetic positive-control experiment.**

### Revival criteria being addressed

The branch closure requires, for revival:
1. A materially different task regime or data-generating setup.
2. Demonstrated specialist complementarity (complementary predictive signal exists and is learnable).
3. Fresh development and held-out evaluation.

This plan addresses all three. It does **not** test Bayesian uncertainty weighting; that remains conditional on first establishing collaborative gain.

---

## 2. Scientific Objectives

This experiment has four separable goals, tested in order:

| Goal | Question | Test layer |
|---|---|---|
| **G1 Code correctness** | Does the training loop, prediction combination, and metric pipeline behave correctly? | Unit tests on deterministic synthetic data |
| **G2 Learnability** | Can a single LoRA specialist learn a task with known structure from small synthetic data? | Single-task learning curves with ground-truth signal |
| **G3 Complementarity** | When two specialists are trained on non-overlapping feature subspaces, do their predictions contain complementary information? | Mutual information diagnostic, oracle ensemble gap |
| **G4 Collaborative improvement** | Does any combination method (equal, learned, CoLoRA) improve on target-only, using complementary specialists? | Full method comparison under the frozen evaluation protocol |

**These must not be conflated.** Failure at G4 is only interpretable if G1–G3 pass first.

---

## 3. Synthetic Data-Generating Process

### 3.1 Design principles

- No method has an informational advantage not derivable from its training allocation.
- The DGP is transparent: all latent variables are documented and frozen before evaluation.
- The positive-control regime is *necessary but not sufficient* for collaboration to succeed.
- The negative-control regime makes collaboration provably suboptimal.

### 3.2 Latent structure

All tasks share a binary classification setting: predict $y \in \{0, 1\}$.

Inputs are $d$-dimensional real-valued feature vectors.
The feature space is partitioned into non-overlapping blocks:
$$x = [x_{\text{shared}},\ x_A,\ x_B,\ x_{\text{noise}}]$$

where:
- $x_{\text{shared}} \in \mathbb{R}^{d_s}$ — correlated with $y$ in all tasks.
- $x_A \in \mathbb{R}^{d_A}$ — correlated with $y$ in the **target** task and **specialist A only**.
- $x_B \in \mathbb{R}^{d_B}$ — correlated with $y$ in the **target** task and **specialist B only**.
- $x_{\text{noise}} \in \mathbb{R}^{d_n}$ — i.i.d. Gaussian noise, uncorrelated with $y$ in all tasks.

> [!IMPORTANT]
> The target task observes all of $x$ during training but has only $N_{\text{target}}$ examples — not enough to reliably learn $x_A$ and $x_B$ independently. Specialists A and B see only their respective partitions plus $x_{\text{noise}}$ (they cannot observe each other's partition or $x_{\text{shared}}$). This enforces that no specialist has access to more signal than its designated block.

### 3.3 Label generation

$$P(y=1 \mid x) = \sigma\!\left(\beta_s^T x_{\text{shared}} + \beta_A^T x_A + \beta_B^T x_B\right)$$

where $\sigma$ is the sigmoid function, and $\beta_s, \beta_A, \beta_B$ are frozen coefficient vectors drawn at DGP initialization time and stored before any experiment.

All tasks share *identical* $\beta_s, \beta_A, \beta_B$ over the same shared latent distribution. The difference is which features each task observes at training time:

| Task | Observes at train time | Label rule |
|---|---|---|
| Target specialist | $x_{\text{shared}}, x_A, x_B, x_{\text{noise}}$ | Full $P(y=1\|x)$ |
| Specialist A | $x_{\text{shared}}, x_A, x_{\text{noise}}$ | Full $P(y=1\|x)$ (labels shared) |
| Specialist B | $x_{\text{shared}}, x_B, x_{\text{noise}}$ | Full $P(y=1\|x)$ (labels shared) |

> [!NOTE]
> All tasks share the same binary label $y$, but specialists A and B are trained on different subsets of the input features. The oracle ensemble (A+B) should provably outperform either specialist alone on the target distribution, because A and B together recover all signal blocks. This creates the positive control.

### 3.4 Controlled complementarity parameters

```python
DGP_CONFIG = {
    "seed": 20261009,           # DGP seed — frozen before any experiment
    "d_shared": 4,              # Shared predictive features
    "d_A": 4,                   # Specialist-A-only features
    "d_B": 4,                   # Specialist-B-only features
    "d_noise": 8,               # Noise features (irrelevant)
    "beta_scale": 1.5,          # Signal-to-noise scale for coefficients
    "n_train_target": 128,      # Target training examples
    "n_train_specialist": 256,  # Per-specialist training examples (more = better learned)
    "n_development": 64,        # Development set (weight fitting only)
    "n_validation": 64,         # Validation set (signals only)
    "n_test": 512,              # Held-out test set
    "prevalence": 0.35,         # P(y=1) target; fixed intercept term
}
```

The total feature dimension is $d = d_s + d_A + d_B + d_n = 20$ input features. This is intentionally small to ensure fast training.

### 3.5 Positive-control regime

In the **positive-control regime**, specialists A and B hold genuinely complementary signal.
The oracle ensemble (equal-weight combination of A and B predictions) must outperform
either specialist alone by a measurable margin that we can compute analytically
from the DGP coefficients.

**Pre-computed oracle gap (analytical):** Before any model training, we compute the
expected Brier improvement from combining two ideal Bayes-optimal specialists versus
the best single specialist. This serves as a theoretical upper bound and sanity check.

### 3.6 Negative-control regime

In the **negative-control regime**, specialists A and B are trained on the *same*
feature set as the target (all features). No specialist has complementary information.
Any mixture of A and B should not improve over target-only in expectation.

This tests whether methods correctly *fail* to improve when collaboration is not warranted.

---

## 4. Method Comparison Matrix

The following methods are evaluated. All except the oracle diagnostic are usable at test time.

| ID | Method | Training data | Description |
|---|---|---|---|
| M0 | **target-only** | Target $N_{\text{target}}$ examples, all features | Baseline; only learns from target |
| M1 | **pooled** | Union of target + A + B training data | All features visible; data augmentation baseline |
| M2 | **specialist-A** | A's training set, A's features only | Source specialist, no target training data |
| M3 | **specialist-B** | B's training set, B's features only | Source specialist, no target training data |
| M4 | **equal ensemble** | M2 + M3 predictions, equal weights | Trivial combination |
| M5 | **learned ensemble** | M2 + M3 predictions, simplex Brier weights fit on development | Combination with development fitting |
| M6 | **CoLoRA** | Union, shared adapter, task-specific Lambda | Structural baseline from prior pilot |
| M7 | **oracle diagnostic** | Test-Brier-best of M2/M3 | Unreachable upper bound; diagnostic only |
| M8 | **prevalence null** | None | Constant prediction = training prevalence |

> [!NOTE]
> In the positive-control regime, the informational-oracle combination of M2 + M3 should outperform M0 (target-only). Whether any of M4–M6 discover this improvement is the experimental question. We explicitly do **not** engineer the DGP to ensure that CoLoRA wins — the question is whether any combination method discovers the complementarity.

---

## 5. Evaluation Metrics and Protocols

### 5.1 Primary metric

**Brier score** (lower is better), consistent with the prior pilot. A method clears the
positive-control bar if it achieves mean Brier improvement ≥ 0.005 over M0 (target-only)
across all three seeds, with positive improvement in all seeds.

The threshold of 0.005 is tighter than the prior pilot's 0.002 against multiple baselines
because: (1) the DGP is synthetic and improvements should be more reliable, and (2) we
want to distinguish real effects from numerical noise with high confidence.

### 5.2 Secondary metrics

- **AUROC** — Discrimination quality
- **Log loss** — Calibration-aware proper scoring
- **ECE** (10-bin) — Calibration error
- **Average precision** — Ranking quality under class imbalance

### 5.3 Complementarity diagnostics

These are computed before running the main experiment on held-out data:

| Diagnostic | Definition | Purpose |
|---|---|---|
| **Oracle Brier gap** | Brier(M0) − Brier(M7) | Upper bound on achievable complementary gain |
| **Mutual information (MI)** | $I(y; \hat{p}_A, \hat{p}_B)$ estimated on development | Is there complementary information in the two specialist predictions? |
| **Error correlation** | Pearson ρ between M2 and M3 prediction errors on development | Negative correlation → complementarity; positive → redundancy |
| **Prediction diversity** | Variance of ensemble prediction vs. mean of individual variances | Diversity quantification |
| **Feature attribution** | Regression of M2/M3 errors onto $x_A, x_B$ blocks | Confirms specialists learned their designated features |

> [!IMPORTANT]
> If the oracle Brier gap (M0 − M7) is < 0.010 on the test set, this indicates the DGP did not generate sufficient complementarity. The experiment would be **inconclusive**, not a negative finding for collaboration. Complementarity must be confirmed by the diagnostics on development data, before we look at M4–M6 test results.

### 5.4 Learning curves

For G2 (learnability), we measure per-specialist Brier score as a function of training set size:
$N \in \{16, 32, 64, 128, 256\}$. This confirms that the model can learn from the synthetic data
and establishes the sample-efficiency curve. Learning curves are computed on the development
set; they are informative only and do not affect any comparison thresholds.

### 5.5 Calibration

For each method, we compute reliability diagrams (10 bins) and store the ECE.
The prevalence null provides the calibration upper-bound reference.

---

## 6. Pre-Registered Hypotheses

All hypotheses are registered here before any model is trained or test data evaluated.

### H1 — Code correctness (G1)
> All unit tests pass deterministically across machines with the same Python/NumPy version.

Failure invalidates the experiment. No correction.

### H2 — Learnability (G2)
> Specialist A trained on $N=256$ examples achieves Brier < 0.20 on the development set.
> Specialist B trained on $N=256$ examples achieves Brier < 0.20 on the development set.

Threshold chosen to be reliably above the prevalence null (Brier = prevalence × (1−prevalence) ≈ 0.228 at 35% prevalence). Failure means the architecture or DGP needs adjustment.

### H3 — Complementarity exists (G3)
> The oracle Brier gap (M0 − M7) on the *development* set exceeds 0.010.
> The error correlation between M2 and M3 on the development set is negative.

**This must be confirmed before H4 is evaluated.** If H3 fails, the experiment is inconclusive;
report the failure and do not interpret M4–M6 test results as evidence against collaboration.

### H4a — Positive-control passes (G4, primary)
> In the positive-control regime, at least one of M4, M5, or M6 achieves mean Brier
> improvement ≥ 0.005 over M0 (target-only) across all three seeds,
> with positive improvement in all seeds.

### H4b — Positive-control ordering (G4, secondary)
> In the positive-control regime, the learned ensemble (M5) improves at least as much
> as the equal ensemble (M4) on the development set. This tests whether the mixture
> weight learner can discover that both specialists are useful.

### H5 — Negative-control null holds (G4)
> In the negative-control regime, no method achieves mean Brier improvement ≥ 0.005 over M0.

Failure of H5 would indicate leakage, implementation error, or a flaw in the negative-control DGP.

---

## 7. Seeds, Randomness, and Reproducibility

| Seed role | Value | Scope |
|---|---|---|
| DGP coefficient initialization | 20261009 | Fixed; not changed after this document |
| DGP data generation (train/dev/val/test) | 20261009 | Shared across all methods |
| Method training seed 1 | 42 | Model initialization and batch ordering |
| Method training seed 2 | 123 | Model initialization and batch ordering |
| Method training seed 3 | 777 | Model initialization and batch ordering |
| Bootstrap resampling | 20261010 | Evaluation stage only |

DGP and training seeds are independent: the same data is used for all three training seeds.
Training seeds affect model initialization and training batch order only.

All random state is set before any data is generated or model is initialized. NumPy `default_rng`
is used throughout; global state is never modified.

---

## 8. Proposed Architecture

### 8.1 Backbone and adapter

The synthetic experiment uses a **lightweight linear model** to replace FinBERT:

```
Input: d=20 dimensional feature vector
→ Linear layer (d → 32), ReLU
→ Linear layer (32 → 16), ReLU  
→ Linear layer (16 → 1), Sigmoid
```

This is approximately 800 trainable parameters per model.
LoRA is applied to the first linear layer as a low-rank update (rank 4, alpha 8),
consistent with the prior pilot's architecture convention.

The CoLoRA structure follows `collaborative_lora_layers.py`: shared B, A matrices
with a per-task Lambda diagonal scaling matrix.

**Design rationale:** Using a simple MLP rather than FinBERT ensures:
1. The experiment runs in < 5 minutes locally.
2. Interpretability: we can verify feature attribution directly.
3. No dependence on large model weights or tokenizer.
4. Gradients are analytically tractable for sanity checking.

### 8.2 Training details

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Consistent with prior pilot |
| Learning rate | 0.001 | Higher than FinBERT (simpler model) |
| Weight decay | 0.01 | L2 regularization |
| Epochs | 50 | Small data; more epochs needed than FinBERT |
| Batch size | 32 | Full dataset fits in memory |
| LoRA rank | 4 | Consistent with prior pilot |
| LoRA alpha | 8 | Consistent with prior pilot |
| Runtime limit | 300s | Well within 20-minute budget |

### 8.3 Feature masking for specialists

Specialist A receives a masked input where $x_B$ columns are zero-filled (not dropped).
Specialist B receives a masked input where $x_A$ columns are zero-filled.
Target-only and pooled methods receive all features.
CoLoRA receives all features but routes by task-specific Lambda during training.

> [!WARNING]
> Zero-filling rather than column deletion preserves the input dimension across all methods,
> allowing the same architecture without shape changes. However, zero-filled columns
> carry implicit information (their position identifies the task). We document this
> as a known confounder. An alternative design using separate input heads is discussed
> in Section 10.

---

## 9. Continuation Criteria and Failure Modes

### Pre-experiment gates (run before held-out evaluation)

| Gate | Criterion | Action on failure |
|---|---|---|
| Unit tests pass | All deterministic tests pass | Fix code; do not proceed |
| Learnability (H2) | Both specialists Brier < 0.20 on dev | Revisit DGP parameters; re-register |
| Complementarity (H3) | Oracle gap > 0.010, negative error correlation | DGP insufficient; report and halt |

### Post-experiment continuation criteria

If H4a is satisfied (positive control passes):
→ Document the successful positive control. Design the Bayesian weighting experiment.

If H4a fails but H3 passes (complementarity exists but no method discovers it):
→ This is the finding that LoRA collaboration *fails to exploit* complementary information.
→ Document as a training/architecture failure, not a complementarity absence.
→ Do not modify thresholds or methods to force a positive result.

If H4a fails and H3 also fails:
→ The synthetic DGP did not generate sufficient complementarity.
→ Re-examine DGP parameters; re-register a new protocol before any further evaluation.

If H5 fails (negative control false positive):
→ Treat as a data or implementation bug; investigate and report.

---

## 10. Potential Confounders and Leakage Risks

### 10.1 Informational confounders

| Confounder | Mechanism | Mitigation |
|---|---|---|
| **Zero-filling task identification** | Specialists can infer their task identity from which feature block is zero | All methods receive the same zero-filled inputs in the same way; the zero structure is not correlated with $y$ at test time |
| **Label homogeneity** | All tasks share the same $y$; specialists A and B see partial signal for the same labels | Intentional by design; documents that complementarity comes from feature coverage, not label shift |
| **DGP coefficient visibility** | The DGP coefficients are stored before training; if they leaked into the feature generation they would bias results | DGP generation is isolated in a separate `generate_dgp` function that saves coefficients to disk only — never into the model training loop |
| **Target feature access** | Target-only method sees all features but has fewer examples | This is a deliberate asymmetry: the experiment tests data scarcity, not informational restriction |

### 10.2 Evaluation leakage risks

| Risk | Mechanism | Mitigation |
|---|---|---|
| **Threshold snooping** | Hypothesis thresholds set by looking at pilot results | All thresholds frozen in this document before any experiment |
| **Architecture search** | MLP depth/width chosen to favor one method | Architecture chosen to be simple and uniform across methods; same architecture for all |
| **DGP parameter tuning** | DGP parameters chosen to guarantee CoLoRA wins | DGP parameters chosen to satisfy analytic complementarity criteria (documented), not chosen by iterating on outcomes |
| **Development-set leakage** | Development set used for more than weight fitting and diagnostics | Development set is used only for: (a) mixture weight fitting, (b) learnability thresholds H2/H3, (c) learning curves. Never used for architecture decisions or threshold calibration |
| **Seed fishing** | Only favorable training seeds reported | All three seeds are reported; the continuation criterion requires positive improvement in all seeds |

### 10.3 Comparison fairness

| Fairness concern | Analysis |
|---|---|
| **Data advantage for pooled** | M1 (pooled) trains on more data than M0. This is expected and not unfair — the experiment tests whether combination discovers the *type* of complementarity, not whether more data helps |
| **Multiple passes for ensemble** | M4/M5 require 2 forward passes vs. 1 for M0/M1. Inference cost is logged but not penalized in the primary metric |
| **CoLoRA training cost** | M6 sees all tasks during training (same data as M1) but has more parameters (Lambda matrices). We report parameter counts and training cost explicitly |
| **Oracle upper bound** | M7 is isolated as a diagnostic and cannot be selected using test data |

### 10.4 External validity limitations

This experiment deliberately uses synthetic data to establish mechanism.
Results cannot be directly extrapolated to:
- Real NLP tasks (different noise structure, semantic complexity)
- High-dimensional feature spaces (different overfitting regime)
- Task-heterogeneous label distributions (this experiment uses a homogeneous DGP)

Any positive finding here is a *mechanism demonstration*, not a performance claim.

---

## 11. Implementation Plan

### Phase 0 — Repository setup (this document)
- [x] Create branch `experiment/bayesian-collaboration-positive-controls`
- [x] Write and freeze this research plan
- [ ] Commit research plan (no code yet)

### Phase 1 — Synthetic DGP module
**File:** `src/hypergraph_portfolio/synthetic_multitask.py`

Implements:
- `generate_dgp(config) → DGPSpec` — deterministic from config seed
- `sample_dataset(dgp_spec, split_sizes, data_seed) → DataSplits`
- `compute_bayes_optimal_brier(dgp_spec) → float` — analytical lower bound
- `compute_oracle_brier_gap(dgp_spec, specialist_predictions) → float`

### Phase 2 — Unit tests
**File:** `tests/test_synthetic_multitask.py`

Covers:
- DGP determinism: same seed → same data
- Feature partition: specialist A input has $x_B$ zeroed
- No label leakage in input features
- Prevalence matches config parameter within tolerance
- Analytical Brier bound is finite and positive

**File:** `tests/test_bayesian_collab_positive_control.py`

Covers:
- Training loop convergence on simple DGP
- Mixture combination correctness
- Metric correctness
- Frozen checkpoint integrity (hash comparison)

### Phase 3 — Experiment harness
**File:** `experiments/run_bayesian_collab_positive_control.py`

Stages (separate, ordered, idempotent):
1. `generate` — create and freeze DGP data
2. `train` — train all models, freeze checkpoints and predictions
3. `diagnose` — run G2/G3 diagnostics on development data (no test labels)
4. `evaluate` — run G4 comparison on held-out test set (only after diagnose passes)

Produces artifacts in:
`artifacts/bayesian_collab_positive_control/`

### Phase 4 — Report
**File:** `artifacts/bayesian_collab_positive_control/report.md`

Generated by the experiment harness. Includes:
- DGP configuration and frozen parameters
- Learning curves (dev Brier vs. training set size)
- Complementarity diagnostics
- Full method comparison table
- Hypothesis decisions (H1–H5)
- Paired bootstrap confidence intervals
- Cloud execution plan (for larger-scale follow-up)

---

## 12. Cloud Execution Plan (For Future Larger Runs)

The initial experiment is designed to run locally in < 5 minutes on a single CPU.
For larger-scale validation (more tasks, higher feature dimensions, larger N), the
following cloud plan is proposed but **not yet executed**:

### Proposed scale-up parameters

| Parameter | Local value | Cloud value | Rationale |
|---|---|---|---|
| N_train_target | 128 | 512 | Approach more realistic data regimes |
| N_train_specialist | 256 | 1024 | Ensure specialist learning saturation |
| N_test | 512 | 2048 | Tighter confidence intervals |
| Seeds | 3 | 10 | Better characterize seed variability |
| d_A, d_B | 4 | 16 | Higher-dimensional complementarity |
| Bootstrap replicates | 1000 | 5000 | More precise intervals |

### Estimated compute

At $d=52$ total features, $N=3072$ total examples, 10 seeds, 8 methods:
- CPU: ~2 hours on a single 4-core machine
- GPU: ~20 minutes on a single T4

### Cloud execution pattern

Use the existing `experiments/cloud/manage_gcp_execution.py` pattern with:
- Separate shards per (method, seed)
- Atomic completion manifests
- Cost ledger and auto-shutdown
- Stored per-shard checksums before aggregation

**Cloud execution must be pre-registered** in its own protocol document before launch.
This plan is a proposal only; it is not committed to or executed here.

---

## 13. Unresolved Scientific Decisions

The following questions require discussion before implementation begins or are flagged as
open in case initial results are ambiguous:

### Unresolved Decision 1: Feature masking vs. separate input heads

**Question:** Should specialist A receive a zero-filled $d=20$ vector (simple, preserves architecture),
or a true $d_s + d_A + d_n = 16$ vector with a different-shaped input layer?

**Tradeoff:** Zero-filling is simpler and directly reusable, but a network can learn to exploit
the zero pattern as a task identity signal. Separate heads are cleaner but require a different
architecture for each specialist.

**Tentative decision:** Use zero-filling for the initial experiment, document as a limitation,
and test the separate-head variant as a robustness check if H4a fails.

### Unresolved Decision 2: Threshold for H2 (learnability)

**Question:** Is Brier < 0.20 the right threshold, given that the prevalence null Brier is
≈0.228 (at 35% prevalence) and Bayes-optimal Brier requires access to all features?

**Tentative decision:** Set H2 threshold to prevalence\_null\_brier − 0.015 = 0.213.
This ensures learning is clearly above chance without requiring Bayes-optimality.
Revise if the DGP generates surprisingly easy or hard tasks.

### Unresolved Decision 3: CoLoRA task routing in the synthetic setting

**Question:** In the prior pilot, CoLoRA uses FinBERT's task routing via task-specific Lambda.
In the MLP setting, task identity must be injected differently. Should task identity be:
(a) inferred implicitly from zero-filled features, or
(b) passed as an explicit one-hot task embedding?

**Tradeoff:** Option (a) is consistent with the pilot's architecture. Option (b) gives CoLoRA
an explicit routing signal that other methods don't have, potentially biasing results.

**Tentative decision:** Use option (a), consistent with the pilot. Document as a limitation.

### Unresolved Decision 4: Whether to include a Bayesian variant in this experiment

**Question:** Should we include a Bayesian-uncertainty-weighted ensemble (the eventual target
of the research program) in this experiment, or reserve it for a later phase?

**Scientific rationale for exclusion:** Including Bayesian weighting before establishing
deterministic collaboration would conflate G4 with a higher-level question (G5). The
closure document explicitly states Bayesian weighting is conditional on establishing G4.

**Decision: Exclude Bayesian weighting from this experiment.** It is listed as a future
phase, not part of the positive-control deliverable.

### Unresolved Decision 5: Multiple-comparison correction

**Question:** Five hypotheses (H1–H5) are tested. Should we apply Bonferroni or
Holm-Bonferroni correction to the significance levels?

**Tentative decision:** No correction in the primary thresholds (the continuation criterion
uses Brier magnitude, not p-values). The bootstrap confidence intervals are exploratory and
carry no multiple-comparison adjustment, consistent with the prior pilot. Report this as a
limitation.

---

## 14. Summary of Key Design Choices

| Choice | Value | Justification |
|---|---|---|
| Synthetic vs. real data | Synthetic | Ground-truth complementarity control; revival criterion |
| Architecture | Lightweight MLP with LoRA | Fast; interpretable; consistent with prior pilot |
| Feature partition | Non-overlapping blocks | Ensures complementarity is mechanistic, not accidental |
| Labels | Shared across tasks | Tests information transfer, not label-distribution shift |
| Baselines | Identical to prior pilot | Enables direct comparison of method behavior |
| Evaluation | Test locked before evaluation stage | Prevents threshold snooping |
| Seeds | 3 (same as prior pilot) | Directly comparable; insufficient for broad characterization |
| Runtime | < 5 minutes | Locally executable; no cloud resource provisioned |

---

## 15. What This Experiment Cannot Establish

Even a fully positive result (all of H1–H4 pass) would not establish:

- That CoLoRA or any combination method works on real NLP tasks.
- That Bayesian uncertainty weighting improves over deterministic collaboration.
- That the specific architecture or hyperparameters generalize to other settings.
- That the positive control validates the specific conditions of the prior FinBERT pilot.

A positive result establishes only: **under controlled synthetic conditions where complementary
information is known to exist, some combination method can discover and exploit it.**
This is a necessary but not sufficient condition for the broader research program.

---

*Document frozen before any experiment is executed. Do not modify this file after the first
commit on branch `experiment/bayesian-collaboration-positive-controls`.*
