# Collaborative LoRA mechanism pilot: protocol frozen before training

Date: 2026-10-03 (America/Los_Angeles). Parent commit: 2eb719e1.

## Repository inspection

Working tree was clean. README scientific status, recent commits through the submission
readiness audit, Phase 16.1/17 clean-input code and results, paper/submission_audit.md,
Phase 13 cloud manager/sharded inference, and the supplied CoLoRA reference were inspected.
No collaborative-LoRA or collaboration mechanism pilot existed. Existing papers and
artifacts are preserved. README contains superseded claims; Phase 17 numeric tables and
paper audit are the latest evidence. The Phase 17 summary also contains prose inconsistent
with its numeric AUROC difference; this branch does not resolve or reinterpret it.
Legacy outcome-leaking checkpoints are excluded. No Bayesian mechanism is presumed.

## File-level implementation

- `src/hypergraph_portfolio/collaborative_lora.py`: convex development-only Brier
  weighting, probability combination, exact effective-update merge utility, subspace overlap.
- `src/hypergraph_portfolio/collaborative_lora_layers.py`: frozen base query/value
  LoRA using Phase 17 architecture convention; shared B Lambda_task A CoLoRA structure.
  Isolated from legacy script imports, which depend on an absolute sibling Bayesian path.
- `experiments/run_collaborative_lora_pilot.py`: separate train/freeze and evaluation
  commands, output checksums, predictions, paired comparisons and signal attribution.
- `tests/test_collaborative_lora.py`: simplex optimization, invalid inputs, update
  merge cross terms, subspace invariance, split separation and CoLoRA routing.
- All new experiment data, checkpoints, configuration, protocol and report live in
  `artifacts/collaborative_lora_mechanism_pilot/`.

## Exact matrix and label definition

The frozen `config.json` defines 3 seeds (42, 123, 777), 4 domains and all hyperparameters.
Target: AAPL, CSCO, INTC, IBM, AMD, GLW, EA, EBAY, ADP, AKAM, AMAT, GOOGL, CTSH, MCHP, APH, FIS, KLAC, LRCX, TXN, ADI.
The initial 8-ticker proposal was expanded after an ID/date-only inventory showed only
70 target test events and 16 events per development partition; no outcomes were inspected.
Specialist banking: JPM, BAC, C, WFC, GS, MS.
Specialist energy: XOM, CVX, COP, SLB, EOG, OXY.
Specialist consumer: WMT, TGT, COST, PG, KO, PEP, MCD, NKE.
These are hand-declared company groups, not a claim to exact sector classification.
Tasks differ by input/label distribution; all share binary downside semantics.
Fixed event label: `move_1d <= -0.05`; no test-quantile label threshold. Phase 17 labels
are discarded. Only its outcome-free metadata text and split assignment are reused.
No raw return is an input feature.

For EACH seed, train six models:
1. Target-only: 128 target training events.
2. Pooled/global: union of the four task training sets, up to 512 events.
3-5. Three specialists: up to 128 events each, excluding target training data.
6. CoLoRA: same union as pooled; shared rank-4 A/B, per-task rank-4 by rank-4 Lambda;
   a shared trainable binary classifier, target Lambda used at inference. This implements
   the paper's adapter structure, not its entire experimental/distributed algorithm.

For EACH seed, evaluate ten entries: those six models; equal probability mixture;
learned simplex probability mixture; Brier-test-best specialist oracle (diagnostic
only, not deployable or selected for any later method); target-training prevalence null.
Mixture experts are the three SOURCE specialists only. Mixtures are prediction ensembles,
not parameter-space adapter merges; no claim of one-pass inference for these methods.
The exact merge utility is tested but not a tested model baseline in this matrix.

FinBERT revision 4556d13015211d73dccd3fdd39d39232506f3e43, query/value rank 4,
alpha 8, dropout .05, trainable classifier; AdamW lr .0002, weight decay .01,
3 fixed epochs, batch 16, 48 tokens. Same seed creates identical initial base classifier
and adapters for independent models. Joint methods get each task's same number of
sample exposures per epoch, shuffled task batches, no extra hyperparameter search.
Pooled/CoLoRA consume more total data/compute than target-only; this is explicitly measured.

## Data boundary and leakage controls

Training is the existing Phase 17 train partition. Target validation partition is split
chronologically at its midpoint calendar date into earlier development (cap 64) and
later validation (cap 64). Test is target tickers in the existing clean_oos partition
(cap 256, 2022+). Select by salted SHA256(event_id), NEVER by outcome; chronological
ordering after selection. All domains and all partitions have disjoint event IDs.
Development optimizes simplex weights and leave-one-out refits only. Validation supplies
signals only. Epoch count/hyperparameters/task groups/rank/test cap are fixed before
training; no early stopping or specialist selection except the isolated diagnostic oracle.

Train stage does not tokenize the test or retrieve its labels. It freezes checksums of
configuration, splits, code, signals, checkpoints and per-seed weights before evaluation.
Evaluation rejects changed frozen inputs and repeat evaluation in the same artifact folder.
The existing historical holdout has already been evaluated by other branches; this is a
new branch-specific locked retrospective pilot, not fresh prospective confirmation.

## Signals and attribution

Before evaluation save each specialist's target validation negative Brier, mean absolute
margin, negative entropy, negative ECE; source-target training token-frequency cosine;
and mean column-space overlap of query/value LoRA B with target-only adapter B.
Similarity never uses target test. Column spaces use numerical-rank SVD, no factor
averaging. Bayesian uncertainty is omitted: existing clean Bayesian models were not
trained as these heterogeneous specialists, so their uncertainty is not interchangeable.

For equal mixture: marginal contribution = Brier(without specialist, renormalized equal)
minus Brier(full mixture). For learned mixture: without-specialist weights are refit on
development and frozen BEFORE test, then analogous difference. Positive means useful.
Report within-seed Spearman correlation across three specialists for every signal and
both rules, plus mean across seeds; no pooled independence assumption, no optimization
on these correlations. With three specialists this is exploratory and cannot establish
which signal is a general predictive weighting rule. No held-out signal-regression claim.

## Compute plan

Use existing sibling virtual environment containing torch/transformers/sklearn; project
venv contains NumPy/SciPy/pytest/ruff but no torch/transformers. CPU deterministic operations,
2 threads, local cached FinBERT; no installation or download. Warm training-batch benchmark
and projection in compute_plan.json. 864 gradient batches if all train caps fill, plus
pretest/test inference and model-load allowance. Abort local training after 1200 seconds.
Expected local duration is below 20 minutes, so no cloud resources are provisioned.
For larger follow-up: benchmark Azure first; use existing seed/task checkpoint shards,
atomic completion manifests and cost ledger/auto-shutdown pattern. Existing cloud code is
GCP-focused and does not itself establish Azure access. This pilot requires no cloud path.

## Falsifiable continuation criteria

Primary metric Brier (lower better); secondary AUROC, AP, log loss, ECE, accuracy.
For learned or CoLoRA to pass: mean Brier improvement >= .002 against EACH of target-only,
pooled and equal; positive improvement in ALL three seeds; lower 95% paired calendar-week
cluster bootstrap bound > 0 for EACH comparison (1000 shared-week resamples across seeds).
One method passing all three comparisons warrants a larger task/seed study. Otherwise
no effect is established under this protocol; no Bayesian optimization follows by default.
Bootstrap is conditional on these seeds; 3 seeds cannot characterize broad seed variation.
Intervals are exploratory, without multiplicity correction. Passing is a screening
criterion and cannot be reported as broad statistical confirmation.

Report per-seed metrics, sample mean/sample variance (ddof=1), trained/stored parameter
counts, optimization sample exposures/steps, training/inference wall time, passes/event,
mixture weights, comparisons and negative findings. Parameter counts for mixtures assume
one shared frozen base plus all source adapters/heads; actual inference still takes three
passes. CoLoRA counts include all task Lambda matrices. Wall timings include model load
in training and forward inference only in inference; logical operations and passes also
reported because timings depend on hardware. Common tokenization, diagnostics and
leave-one-out refits are analysis overhead; record total training-stage wall time separately.
