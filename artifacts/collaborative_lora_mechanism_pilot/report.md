# Collaborative LoRA mechanism pilot

Decision: **NO collaborative effect established; do not scale this recipe**.

This is a locked retrospective pilot on real earnings-event metadata, not an earnings-call NLP benchmark or prospective confirmation. All prior repository findings and checkpoints were preserved. No Bayesian weighting was used.

## Protocol

See [frozen protocol](protocol.md) and [configuration](config.json) for task tickers, exact matrix and preregistered criteria. FinBERT with fresh query/value rank-4 adapters, three fixed epochs, seeds 42/123/777. Target-only and three source-domain specialists each train on 128 events; pooled and CoLoRA train on their 512-event union. The three-source prediction mixtures exclude the target adapter. CoLoRA uses shared B Lambda_task A with a shared binary head, evaluated through the target Lambda. It is a structural baseline based on the supplied paper, not a full replication of its training experiments.

Actual splits: target=128, banking=128, energy=128, consumer=128, development=64, validation=64, test=256. Target test prevalence 0.2305; 72 calendar-week clusters. Label is fixed return <= -5%, not a test-derived percentile. Events were selected by salted ID hash without outcomes; training, earlier development, later validation and test are disjoint and chronological. Three-source simplex weights and leave-one-out refits use development only; validation provides explanatory signals only. No tuning or early stopping follows test evaluation. Test-best specialist is isolated as an oracle diagnostic, upper only among these three single specialists.

## Target-task results

Means and sample variances across three seeds (ddof=1); lower Brier/log loss is better. Full AP, ECE, accuracy, seed results and predictions are in the CSV files.

| Method | Brier mean | Brier variance | AUROC mean | AUROC variance | Log loss mean |
| --- | --- | --- | --- | --- | --- |
| target | 0.193679 | 2.41e-05 | 0.5096 | 0.000502 | 0.60953 |
| banking | 0.194799 | 2.81e-06 | 0.4627 | 0.000658 | 0.61439 |
| energy | 0.219208 | 7.93e-07 | 0.5271 | 0.00074 | 0.86333 |
| consumer | 0.215016 | 6.04e-06 | 0.5039 | 0.0013 | 0.79426 |
| pooled | 0.196420 | 1.34e-05 | 0.5182 | 0.000509 | 0.62451 |
| colora | 0.196366 | 1.32e-05 | 0.5164 | 0.00039 | 0.62417 |
| equal | 0.208606 | 3.21e-08 | 0.4830 | 0.00104 | 0.71661 |
| learned | 0.194799 | 2.81e-06 | 0.4627 | 0.000658 | 0.61439 |
| oracle_diagnostic | 0.194799 | 2.81e-06 | 0.4627 | 0.000658 | 0.61439 |
| prevalence_null | 0.184082 | 0 | 0.5000 | 0 | 0.56329 |

## Collaboration comparisons

Positive gain means lower Brier than the baseline. Paired 95% calendar-week cluster bootstrap shares event resamples across seeds (1000 draws). Continue requires gain >= .002, positive gains in all seeds and lower CI > 0 against ALL three baselines. Equal versus equal is necessarily zero. These intervals condition on the three seeds and have no multiple-comparison adjustment.

| Method | Baseline | Brier gain | 95% CI | Positive seeds | Pass |
| --- | --- | --- | --- | --- | --- |
| equal | target | -0.014927 | [-0.019896, -0.009937] | 0 | False |
| equal | pooled | -0.012186 | [-0.016518, -0.008287] | 0 | False |
| equal | equal | +0.000000 | [+0.000000, +0.000000] | 0 | False |
| learned | target | -0.001120 | [-0.001717, -0.000605] | 1 | False |
| learned | pooled | +0.001621 | [+0.000954, +0.002279] | 2 | False |
| learned | equal | +0.013807 | [+0.009420, +0.018569] | 3 | True |
| colora | target | -0.002687 | [-0.003871, -0.001644] | 2 | False |
| colora | pooled | +0.000054 | [+0.000032, +0.000075] | 3 | False |
| colora | equal | +0.012240 | [+0.008289, +0.016105] | 3 | True |

## Parameters and compute

Device: deterministic CPU, 2 threads. Training-stage wall time 877.0s. Cloud was unnecessary: the preflight estimate was 1050s (<20 minutes). No Azure/GCP instance was provisioned. Total launch-to-evaluation wall time: 18.50 minutes (actual_compute.json). Training timing includes model construction; inference is timed model forwards only. Common tokenization, diagnostics and optimization overhead are included in stage wall time but excluded from per-model inference. Mixture inference sums three specialists' measured forward times; mixing arithmetic is not timed. Stored parameter counts assume one shared frozen base plus all adapters/classifier heads; CoLoRA includes all four Lambda sets. Learned weights add three stored/trainable scalars with two effective simplex degrees of freedom. The trained parameter count is the total jointly optimized storage, not target-only inference capacity.

| Method | Stored params | Trainable params | Train s mean | Train exposures | Gradient steps | Test forwards s mean | Passes/event |
| --- | --- | --- | --- | --- | --- | --- | --- |
| target | 109631234 | 148994 | 22.15 | 384 | 24 | 6.10 | 1 |
| banking | 109631234 | 148994 | 22.63 | 384 | 24 | 7.65 | 1 |
| energy | 109631234 | 148994 | 21.05 | 384 | 24 | 8.18 | 1 |
| consumer | 109631234 | 148994 | 21.31 | 384 | 24 | 6.13 | 1 |
| pooled | 109631234 | 148994 | 86.41 | 1536 | 96 | 7.34 | 1 |
| colora | 109632770 | 150530 | 103.17 | 1536 | 96 | 6.36 | 1 |
| equal | 109929222 | 446982 | 64.99 | 1152 | 72 | 21.96 | 3 |
| learned | 109929225 | 446985 | 65.00 | 1152 | 72 | 21.96 | 3 |
| oracle_diagnostic | 109631234 | 148994 | 22.63 | 384 | 24 | 7.65 | 1 |
| prevalence_null | 0 | 0 | 0.00 | 0 | 0 | 0.00 | 0 |

For the test-oracle diagnostic, single-specialist costs omit the search across all three specialists; it cannot be deployed with test labels. Seed-level timing and timing variances are saved. Checkpoints contain adapters/heads only; the base revision is fixed in config.

## Specialist weights and marginal usefulness

Weights ordered banking, energy, consumer; equal is [1/3, 1/3, 1/3]. A near-one-hot learned rule is specialist selection, so a screening pass from such a rule would not by itself establish a benefit from combination.

| Seed | Learned weights |
| --- | --- |
| 42 | [0.9999999999999989, 0.0, 1.169203622808367e-15] |
| 123 | [0.9999999999999996, 0.0, 4.0245584642661905e-16] |
| 777 | [0.9999999999999996, 4.442682554217892e-16, 0.0] |

Realized marginal Brier gain is without-specialist Brier minus full-mixture Brier, so positive means useful. For equal weights, removal renormalizes remaining weights; for learned weights, each reduced mixture was refit on development BEFORE test. See marginal_contributions.csv for every seed, specialist and rule. Negative contributions are retained.

| Rule | Signal | Mean within-seed Spearman |
| --- | --- | --- |
| equal | validation_performance | +1.000 |
| equal | data_similarity | +0.500 |
| equal | subspace_similarity | -1.000 |
| equal | confidence | -1.000 |
| equal | negative_ece | +1.000 |
| equal | negative_entropy | -1.000 |
| equal | learned_weight | +0.833 |
| learned | validation_performance | +0.866 |
| learned | data_similarity | +0.866 |
| learned | subspace_similarity | -0.866 |
| learned | confidence | -0.866 |
| learned | negative_ece | +0.866 |
| learned | negative_entropy | -0.866 |
| learned | learned_weight | +0.866 |

Signals were frozen pretest: negative target-validation Brier, training token cosine, LoRA B column-space overlap versus target adapter, absolute margin, negative ECE and negative entropy. The learned weight is also audited. Only three specialists exist per seed: these rank correlations are very coarse and dependent across seeds. They do not establish a best explanatory mechanism or incremental predictive value; no significance or winning-signal claim is made. Metadata token cosine is dominated by the common announcement template and may be uninformative. Bayesian uncertainty is omitted because no clean heterogeneous Bayesian specialist checkpoints exist.

## Observed findings

No method meets the predeclared continuation screen. Equal weighting worsens mean Brier by .014927 versus target-only and .012186 versus pooled. Learned weighting improves .013807 versus equal (all seeds), but worsens .001120 versus target-only on average and wins only one of three seeds there. It improves .001621 versus pooled, below the .002 practical threshold and with only two positive seeds. Weights collapse to banking in every seed and match the test-best single-specialist diagnostic: this is successful selection of the least harmful source, not a collaborative mixture benefit. CoLoRA improves only .000054 versus pooled, far below .002, and is worse than target-only by .002687 on average. It beats equal weighting, which is a weak baseline here. The training-prevalence null has lower Brier (.184082) than every trained method; all methods have the same majority-class accuracy (.769531). Discrimination remains near chance, so none of these findings establishes useful target-task learning from the available metadata.

For equal-mixture marginal contributions, validation negative Brier and negative ECE both have within-seed Spearman +1.0 in all three seeds; token similarity is +.5, while LoRA-subspace overlap, margin confidence and negative entropy are -1.0. For development-refitted mixtures the corresponding correlations have ties and magnitude .866. The ranking is consistent with calibration dominating the usefulness of these weak models that underestimate downside probability: high binary classification confidence does not imply a useful specialist. This is a description of three sources, not a fitted or independently validated explanatory rule. The source datasets, priors and input templates remain confounded; richer tasks are needed before a causal mechanism claim.

## Limitations, negative findings and next decision

This pilot uses company/date metadata, only 128 train events per task and three epochs. A null result cannot rule out collaboration with rich text, better tasks, greater training budgets or more specialists. Pooled and CoLoRA consume more total labeled data and training compute than target-only; their comparisons control the union and exposure budget. The specialist ensemble takes three inference passes; its parameter count and compute differ from one model. Seeds share event partitions and pretrained weights. Source specialists can fail through calibration or base-rate shift. The existing 2022+ holdout was used in earlier repository work; this branch freezes new weights before its evaluation but supplies no fresh prospective evidence.

The primary screen passes only if decision.json says continue. If it fails, stop this metadata recipe and do not tune on its test set. A future branch would need richer inputs and a fresh held-out task or time period, broader specialist diversity, and a separately locked signal-validation study. Uncertainty remains an optional later candidate and is not justified by this pilot.

## Reproduction and verification

Install the `collaboration` optional dependency group and use a locally cached copy of the pinned FinBERT revision. From the repository root: `PYTHONPATH=src python experiments/run_collaborative_lora_pilot.py --stage train`, then `--stage evaluate`, then `python experiments/report_collaborative_lora_pilot.py`. The delivered artifact directory is frozen and deliberately refuses retraining/re-evaluation; archive it before an exact replay and restore the same frozen config/protocol in a clean working copy. No rerun may change settings based on the test results. Check frozen.json for code, input and checkpoint checksums. Training used the available sibling virtual environment without importing sibling project code.

Validation: full pytest suite 266 passed, 3 skipped; new logic tests 15 passed in the training environment; repository ruff passed. See pytest.log and train.log. See environment.json for runtime versions and source_manifest.json for source-data identities.
