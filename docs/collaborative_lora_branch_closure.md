# Collaborative LoRA branch closure

Closure date: 2026-10-04 (America/Los_Angeles).

**Outcome: No collaborative LoRA effect established under the frozen pilot protocol.
The branch is closed unless a materially different task regime or data-generating setup
provides evidence of exploitable specialist complementarity.**

## Research question and frozen hypotheses

Can heterogeneous LoRA specialists trained on related but different tasks improve a
held-out target task, and what development-only weighting rule predicts marginal usefulness?
The operational hypothesis was that learned collaboration or CoLoRA would improve target
Brier performance over target-only, pooled/global and equal weighting. The explanatory
signal comparison was exploratory; no signal-specific confirmation threshold was frozen.
Bayesian uncertainty was a possible later mechanism, conditional on first establishing
collaborative gain, rather than a presumed source of benefit.

The [original protocol](../artifacts/collaborative_lora_mechanism_pilot/protocol.md),
[configuration](../artifacts/collaborative_lora_mechanism_pilot/config.json) and
[preregistration](../artifacts/collaborative_lora_mechanism_pilot/preregistration.json)
remain unchanged. A learned or CoLoRA method had to satisfy **all** of the following
against **each** of target-only, pooled/global and equal weighting:

- Mean Brier improvement at least **0.002** (lower Brier is better).
- Positive improvement in **all three seeds**, 42, 123 and 777.
- Lower **95% paired calendar-week bootstrap** confidence bound strictly above zero,
  using 1,000 shared-week resamples across seeds.

AUROC, average precision, log loss, ECE and accuracy remain secondary. Beating equal
weighting alone, a positive but smaller pooled advantage, or an isolated favorable seed
cannot satisfy this rule. The closure does not substitute another metric or threshold.

## Experiment matrix and information boundaries

FinBERT revision `4556d13015211d73dccd3fdd39d39232506f3e43`; fresh query/value
LoRA rank 4, alpha 8, dropout 0.05; three fixed epochs; AdamW learning rate 0.0002,
weight decay 0.01, batch 16 and maximum 48 tokens. Inputs are outcome-free company/date
announcement metadata. Binary labels use the fixed rule `move_1d <= -0.05`, not a
test-derived percentile. Company groups and all settings are in the frozen config.

| Method per seed | Training data / role |
| --- | --- |
| Target-only LoRA | 128 target computing, semiconductor and internet-company events |
| Pooled/global LoRA | Union of target plus banking, energy and consumer training sets: 512 events |
| Each of three source specialists | 128 events in its own domain; no target training examples |
| Best-single-specialist oracle | Test-Brier-best source; diagnostic only, never used to fit/select another method |
| Equal mixture | Three source specialists, probability weights 1/3 each |
| Learned mixture | Three source specialists, simplex Brier weights fitted only on development |
| CoLoRA structural baseline | Same 512-event union as pooled; shared B/A and task-specific Lambda, target Lambda at inference |
| Prevalence null | Constant target-training prevalence, 0.1484375 |

There are 18 trained models and 10 evaluated methods per seed. Target development and
validation contain 64 events each, in chronological order; target test contains 256 events
in 72 calendar-week clusters (2022+). ID-hash selection does not use outcomes. Tasks and
partitions have disjoint event IDs. Development fits full and leave-one-out weights;
validation supplies explanatory signals. Code, models, signals and weights were frozen
before test evaluation. The historical holdout was used in prior repository branches:
this is a locked retrospective pilot, not fresh prospective confirmation.

## Principal results and seed variability

Brier scores and sample variance across seeds (`ddof=1`):

| Method | Seed 42 | Seed 123 | Seed 777 | Mean | Variance |
| --- | --- | --- | --- | --- | --- |
| target | 0.192368 | 0.189558 | 0.199110 | 0.193679 | 2.41e-05 |
| banking | 0.192871 | 0.195623 | 0.195903 | 0.194799 | 2.81e-06 |
| energy | 0.219807 | 0.219631 | 0.218184 | 0.219208 | 7.93e-07 |
| consumer | 0.217600 | 0.212708 | 0.214740 | 0.215016 | 6.04e-06 |
| pooled | 0.192189 | 0.198607 | 0.198463 | 0.196420 | 1.34e-05 |
| colora | 0.192172 | 0.198601 | 0.198325 | 0.196366 | 1.32e-05 |
| equal | 0.208696 | 0.208400 | 0.208723 | 0.208606 | 3.21e-08 |
| learned | 0.192871 | 0.195623 | 0.195903 | 0.194799 | 2.81e-06 |
| oracle_diagnostic | 0.192871 | 0.195623 | 0.195903 | 0.194799 | 2.81e-06 |
| prevalence_null | 0.184082 | 0.184082 | 0.184082 | 0.184082 | 0 |

The [full report](../artifacts/collaborative_lora_mechanism_pilot/report.md) retains
secondary metrics, parameter counts, timing and all confidence intervals.
[Seed metrics](../artifacts/collaborative_lora_mechanism_pilot/test_metrics.csv) and
[summary](../artifacts/collaborative_lora_mechanism_pilot/summary.csv) are the numeric record.
The prevalence null has the lowest Brier **and log loss in every seed**, and the lowest
mean ECE. It is strongest on those proper-score/calibration measures, not on every
metric: its AUROC is 0.5 and some trained models rank slightly above chance. All methods
have identical majority-class accuracy, 0.769531. The null is a central substantive
control, not an irrelevant comparator.

| Candidate | Brier gain vs target | Gain vs pooled | Gain vs equal | Positive seeds vs target / pooled / equal | Overall screen |
| --- | --- | --- | --- | --- | --- |
| Learned | -0.001120 | +0.001621 | +0.013807 | 1 / 2 / 3 | Failed |
| CoLoRA | -0.002687 | +0.000054 | +0.012240 | 2 / 3 / 3 | Failed |

Positive gain means lower Brier than the comparator. Learned weighting's pooled gain
falls below 0.002 and lacks all-seed consistency. CoLoRA's pooled gain is only about
0.000054, despite a positive conditional bootstrap interval: statistical direction does
not replace the frozen materiality requirement. Both are worse than target-only on average.
Equal weighting is worse than target-only and pooled in every seed. Both candidates beat
this weak equal baseline, which does not establish the required collaboration effect.
See [paired comparisons](../artifacts/collaborative_lora_mechanism_pilot/paired_comparisons.csv).

## Learned weights and marginal contributions

In banking / energy / consumer order, all three learned weight vectors are effectively
**[1, 0, 0]**: banking weight exceeds `1 - 1e-12`, and residual weights are numerical noise
of order `1e-15`. These predictions match banking and the isolated best-single diagnostic.
The result is single-specialist selection, with no demonstrated complementary mixture.
Exact weights and all reduced-mixture refits remain in
[frozen.json](../artifacts/collaborative_lora_mechanism_pilot/frozen.json).

Marginal contribution is Brier without the specialist minus full-mixture Brier. Positive
means useful. Equal removal renormalizes remaining equal weights; learned removal uses
weights refitted on development and frozen before test.

| Rule | Specialist | Seed 42 | Seed 123 | Seed 777 | Mean gain |
| --- | --- | --- | --- | --- | --- |
| equal | banking | +0.009999 | +0.007690 | +0.007719 | +0.008469 |
| equal | energy | -0.004916 | -0.004952 | -0.004248 | -0.004705 |
| equal | consumer | -0.004036 | -0.002048 | -0.002806 | -0.002963 |
| learned | banking | +0.024729 | +0.017085 | +0.018837 | +0.020217 |
| learned | energy | -0.000000 | +0.000000 | +0.000000 | +0.000000 |
| learned | consumer | -0.000000 | +0.000000 | +0.000000 | +0.000000 |

Rounded zeros in learned energy/consumer contributions are at most floating-point noise
(~2.8e-17); the signed raw values are preserved in
[marginal_contributions.csv](../artifacts/collaborative_lora_mechanism_pilot/marginal_contributions.csv).
Banking helps relative to mixtures dominated by harmful sources; this is not evidence
that multiple useful specialists complement one another. Under equal weighting,
validation negative Brier and negative ECE rank marginal usefulness with Spearman +1
in each seed, token similarity +0.5, and subspace overlap, confidence margin and negative
entropy -1. With only three specialists and shared data, these coarse correlations cannot
identify a general weighting rule or prove a causal calibration mechanism. All explanatory
signals and correlations, including unfavorable ones, remain saved.

## Compute and limitations

The pilot ran locally on deterministic CPU operations with two threads. Training stage:
876.97 seconds; evaluation stage from log timestamps: 139.09 seconds; total launch through
evaluation completion: 1110.18 seconds (**18.50 minutes**, including the inter-stage gap).
No cloud job was launched for this repository. These are recorded wall times, not FLOP
estimates or cloud-billing measurements.

Independent/pooled adapters have 148,994 trainable parameters; CoLoRA has 150,530.
Pooled and CoLoRA each process 1,536 training sample exposures, versus 384 per individual
adapter. Mixtures retain three adapters/heads (446,982 parameters, plus three learned
weight scalars) and require three forward passes per event. Their timing records sum
source forward times and omit trivial mixing arithmetic. See
[compute records](../artifacts/collaborative_lora_mechanism_pilot/actual_compute.json)
and the original report for stored-base parameter counts and per-seed timings.

The result is bounded by metadata inputs, 128 training examples per task, three epochs,
three seeds, one backbone, three source specialists and one target regime. The CoLoRA
baseline implements the shared-factor adapter structure, not a full replication of its
paper's experimental/distributed algorithm. Seeds share events, and bootstrap intervals
condition on those seeds without multiplicity correction. Source base rates, calibration,
input quality and complementarity are not causally separated. These limitations bound
the negative finding; they do not license changing this pilot after seeing its test outcome.

## Reconciliation with prior repository findings

Earlier Bayesian-LoRA uncertainty hypotheses were already substantially weakened by
later audits. [Phase 15](../artifacts/phase15_incremental_uncertainty/phase15_report.md)
and [Phase 15.1](../artifacts/phase15_1_provenance_audit/phase15_1_report.md) questioned
incremental Bayesian sampling value beyond simpler confidence. Phase 16.1 subsequently
identified outcome leakage in legacy training and the mechanical, class-dependent
entropy/error relationship. The later [submission audit](../paper/submission_audit.md)
and [limitations](../paper/limitations.md) constrain the clean Phase 17 conclusions to
weak metadata inputs; they do not justify a broad claim that Bayesian LoRA never helps.
Clean Bayesian-versus-deterministic differences must be read from those numeric tables,
not superseded early claims or inconsistent summary prose.

This deterministic collaboration pilot independently fails to establish the prerequisite
collaborative gain. It does not test Bayesian weighting and cannot establish its efficacy
or universal inefficacy. Because the prerequisite gain is absent, **Bayesian/UQ specialist
weighting was not pursued**. Together with the prior audits, there is currently **no
empirical basis in this repository for escalating to Bayesian or portfolio-style
specialist weighting**. Prior portfolio findings and their audit history are preserved;
a portfolio analogy is not evidence of specialist complementarity.

## Research implications and branch verdict

- **Failure to establish collaboration:** the frozen comparisons fail; learned weighting
  degenerates to selection and CoLoRA does not materially beat pooled training. This is a
  failure under this protocol, not a proof against collaborative LoRA in general.
- **Weak underlying task signal:** near-chance discrimination and a stronger prevalence
  null indicate that these metadata inputs and limited training do not supply robust
  target predictive signal for sophisticated combination to exploit.
- **Possible lack of cross-task complementarity:** harmful energy/consumer contributions
  and collapsed weights show no exploited complementarity here. They cannot distinguish
  absent complementary information from a failure of these trained adapters to learn it.

**Outcome: No collaborative LoRA effect established under the frozen pilot protocol.
The branch is closed unless a materially different task regime or data-generating setup
provides evidence of exploitable specialist complementarity.**

Revival requires a materially different setup in which individual specialists
**demonstrably contain complementary predictive signal**, plus fresh development and
held-out evaluation. Another weighting-rule search, hyperparameter retuning, lowered
threshold or new ensemble on the current pilot test is not a revival criterion. Failed
methods, the prevalence null, original thresholds and unfavorable results remain intact.

## Closure audit

All 41 files covered by the original artifact manifest match their saved hashes, as do
preregistration, source, config, code and checkpoint identities. From the saved predictions,
the closure audit reproduced Brier scores/sample variances, mixture identities, all 18
marginal contributions and the original paired bootstrap intervals/decision. This is
verification of existing results, not new model evaluation or an expensive experiment.
See [closure verification](../artifacts/collaborative_lora_mechanism_pilot/closure_audit/artifact_verification.json).
The frozen report, manifest, configuration, predictions and decision were not edited.
Final closure test/lint logs are saved separately under `closure_audit/`.
