"""Render the pilot's immutable result tables into a concise scientific report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "artifacts/collaborative_lora_mechanism_pilot"


def read(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    summary = read("summary.csv")
    costs = read("test_metrics.csv")
    comparisons = read("paired_comparisons.csv")
    signals = read("signal_correlations.csv")
    freeze = json.loads((OUT / "frozen.json").read_text())
    decision = json.loads((OUT / "decision.json").read_text())
    splits = json.loads((OUT / "splits.json").read_text())
    methods = [r["method"] for r in summary]
    actual = json.loads((OUT / "actual_compute.json").read_text()) if (OUT / "actual_compute.json").exists() else None
    verdict = "CONTINUE to a larger independently held-out study" if decision["continue"] else "NO collaborative effect established; do not scale this recipe"
    lines = ["# Collaborative LoRA mechanism pilot", "", f"Decision: **{verdict}**.", "",
             "This is a locked retrospective pilot on real earnings-event metadata, not an earnings-call NLP benchmark or prospective confirmation. "
             "All prior repository findings and checkpoints were preserved. No Bayesian weighting was used.", "",
             "## Protocol", "",
             "See [frozen protocol](protocol.md) and [configuration](config.json) for task tickers, exact matrix and preregistered criteria. "
             "FinBERT with fresh query/value rank-4 adapters, three fixed epochs, seeds 42/123/777. "
             "Target-only and three source-domain specialists each train on 128 events; pooled and CoLoRA train on their 512-event union. "
             "The three-source prediction mixtures exclude the target adapter. CoLoRA uses shared B Lambda_task A with a shared binary head, evaluated through the target Lambda. "
             "It is a structural baseline based on the supplied paper, not a full replication of its training experiments.", "",
             f"Actual splits: {', '.join(f'{k}={len(v)}' for k,v in splits.items())}. "
             f"Target test prevalence {decision['test_prevalence']:.4f}; {decision['weeks']} calendar-week clusters. "
             "Label is fixed return <= -5%, not a test-derived percentile. "
             "Events were selected by salted ID hash without outcomes; training, earlier development, later validation and test are disjoint and chronological. "
             "Three-source simplex weights and leave-one-out refits use development only; validation provides explanatory signals only. "
             "No tuning or early stopping follows test evaluation. Test-best specialist is isolated as an oracle diagnostic, upper only among these three single specialists.", "",
             "## Target-task results", "",
             "Means and sample variances across three seeds (ddof=1); lower Brier/log loss is better. "
             "Full AP, ECE, accuracy, seed results and predictions are in the CSV files.", "",
             table(["Method", "Brier mean", "Brier variance", "AUROC mean", "AUROC variance", "Log loss mean"],
                   [[r['method'], f"{float(r['brier_mean']):.6f}", f"{float(r['brier_variance']):.3g}",
                     f"{float(r['auroc_mean']):.4f}", f"{float(r['auroc_variance']):.3g}", f"{float(r['log_loss_mean']):.5f}"] for r in summary]), "",
             "## Collaboration comparisons", "",
             "Positive gain means lower Brier than the baseline. Paired 95% calendar-week cluster bootstrap shares event resamples across seeds (1000 draws). "
             "Continue requires gain >= .002, positive gains in all seeds and lower CI > 0 against ALL three baselines. "
             "Equal versus equal is necessarily zero. These intervals condition on the three seeds and have no multiple-comparison adjustment.", "",
             table(["Method", "Baseline", "Brier gain", "95% CI", "Positive seeds", "Pass"],
                   [[r['method'], r['baseline'], f"{float(r['mean_brier_gain']):+.6f}",
                     f"[{float(r['ci_low']):+.6f}, {float(r['ci_high']):+.6f}]", r['positive_seeds'], r['passes_continue_criterion']] for r in comparisons]), "",
             "## Parameters and compute", "",
             f"Device: deterministic CPU, 2 threads. Training-stage wall time {freeze['training_wall_seconds']:.1f}s. "
             "Cloud was unnecessary: the preflight estimate was 1050s (<20 minutes). No Azure/GCP instance was provisioned. "
             f"Total launch-to-evaluation wall time: {actual['total_launch_to_evaluation_finish_seconds']/60 if actual else 0:.2f} minutes (actual_compute.json). "
             "Training timing includes model construction; inference is timed model forwards only. "
             "Common tokenization, diagnostics and optimization overhead are included in stage wall time but excluded from per-model inference. "
             "Mixture inference sums three specialists' measured forward times; mixing arithmetic is not timed. "
             "Stored parameter counts assume one shared frozen base plus all adapters/classifier heads; CoLoRA includes all four Lambda sets. "
             "Learned weights add three stored/trainable scalars with two effective simplex degrees of freedom. "
             "The trained parameter count is the total jointly optimized storage, not target-only inference capacity.", "",
             table(["Method", "Stored params", "Trainable params", "Train s mean", "Train exposures", "Gradient steps", "Test forwards s mean", "Passes/event"],
                   [[m, next(r['total_parameters'] for r in costs if r['method']==m),
                     next(r['trainable_parameters'] for r in costs if r['method']==m),
                     f"{sum(float(r['training_seconds']) for r in costs if r['method']==m)/3:.2f}",
                     next(r['training_examples'] for r in costs if r['method']==m),
                     next(r['gradient_steps'] for r in costs if r['method']==m),
                     f"{sum(float(r['inference_seconds']) for r in costs if r['method']==m)/3:.2f}",
                     next(r['passes_per_event'] for r in costs if r['method']==m)] for m in methods]), "",
             "For the test-oracle diagnostic, single-specialist costs omit the search across all three specialists; it cannot be deployed with test labels. "
             "Seed-level timing and timing variances are saved. Checkpoints contain adapters/heads only; the base revision is fixed in config.", "",
             "## Specialist weights and marginal usefulness", "",
             "Weights ordered banking, energy, consumer; equal is [1/3, 1/3, 1/3]. A near-one-hot learned rule is specialist selection, so a screening pass from such a rule would not by itself establish a benefit from combination.", "",
             table(["Seed", "Learned weights"], [[s, str(v['weights'])] for s,v in freeze['seeds'].items()]), "",
             "Realized marginal Brier gain is without-specialist Brier minus full-mixture Brier, so positive means useful. "
             "For equal weights, removal renormalizes remaining weights; for learned weights, each reduced mixture was refit on development BEFORE test. "
             "See marginal_contributions.csv for every seed, specialist and rule. Negative contributions are retained.", "",
             table(["Rule", "Signal", "Mean within-seed Spearman"],
                   [[r['rule'], r['signal'], f"{float(r['spearman']):+.3f}" if r['spearman'] else "undefined (constant)"] for r in signals if r['seed']=='mean']), "",
             "Signals were frozen pretest: negative target-validation Brier, training token cosine, LoRA B column-space overlap versus target adapter, "
             "absolute margin, negative ECE and negative entropy. The learned weight is also audited. "
             "Only three specialists exist per seed: these rank correlations are very coarse and dependent across seeds. "
             "They do not establish a best explanatory mechanism or incremental predictive value; no significance or winning-signal claim is made. "
             "Metadata token cosine is dominated by the common announcement template and may be uninformative. "
             "Bayesian uncertainty is omitted because no clean heterogeneous Bayesian specialist checkpoints exist.", "",
             "## Observed findings", "",
             "No method meets the predeclared continuation screen. Equal weighting worsens mean Brier by .014927 versus target-only and .012186 versus pooled. "
             "Learned weighting improves .013807 versus equal (all seeds), but worsens .001120 versus target-only on average and wins only one of three seeds there. "
             "It improves .001621 versus pooled, below the .002 practical threshold and with only two positive seeds. "
             "Weights collapse to banking in every seed and match the test-best single-specialist diagnostic: this is successful selection of the least harmful source, not a collaborative mixture benefit. "
             "CoLoRA improves only .000054 versus pooled, far below .002, and is worse than target-only by .002687 on average. "
             "It beats equal weighting, which is a weak baseline here. The training-prevalence null has lower Brier (.184082) than every trained method; all methods have the same majority-class accuracy (.769531). "
             "Discrimination remains near chance, so none of these findings establishes useful target-task learning from the available metadata.", "",
             "For equal-mixture marginal contributions, validation negative Brier and negative ECE both have within-seed Spearman +1.0 in all three seeds; "
             "token similarity is +.5, while LoRA-subspace overlap, margin confidence and negative entropy are -1.0. "
             "For development-refitted mixtures the corresponding correlations have ties and magnitude .866. "
             "The ranking is consistent with calibration dominating the usefulness of these weak models that underestimate downside probability: "
             "high binary classification confidence does not imply a useful specialist. "
             "This is a description of three sources, not a fitted or independently validated explanatory rule. "
             "The source datasets, priors and input templates remain confounded; richer tasks are needed before a causal mechanism claim.", "",
             "## Limitations, negative findings and next decision", "",
             "This pilot uses company/date metadata, only 128 train events per task and three epochs. "
             "A null result cannot rule out collaboration with rich text, better tasks, greater training budgets or more specialists. "
             "Pooled and CoLoRA consume more total labeled data and training compute than target-only; their comparisons control the union and exposure budget. "
             "The specialist ensemble takes three inference passes; its parameter count and compute differ from one model. "
             "Seeds share event partitions and pretrained weights. Source specialists can fail through calibration or base-rate shift. "
             "The existing 2022+ holdout was used in earlier repository work; this branch freezes new weights before its evaluation but supplies no fresh prospective evidence.", "",
             "The primary screen passes only if decision.json says continue. If it fails, stop this metadata recipe and do not tune on its test set. "
             "A future branch would need richer inputs and a fresh held-out task or time period, broader specialist diversity, and a separately locked signal-validation study. "
             "Uncertainty remains an optional later candidate and is not justified by this pilot.", "",
             "## Reproduction and verification", "",
             "Install the `collaboration` optional dependency group and use a locally cached copy of the pinned FinBERT revision. "
             "From the repository root: `PYTHONPATH=src python experiments/run_collaborative_lora_pilot.py --stage train`, "
             "then `--stage evaluate`, then `python experiments/report_collaborative_lora_pilot.py`. "
             "The delivered artifact directory is frozen and deliberately refuses retraining/re-evaluation; archive it before an exact replay and restore the same frozen config/protocol in a clean working copy. "
             "No rerun may change settings based on the test results. Check frozen.json for code, input and checkpoint checksums. "
             "Training used the available sibling virtual environment without importing sibling project code.", "",
             "Validation: full pytest suite 266 passed, 3 skipped; new logic tests 15 passed in the training environment; repository ruff passed. "
             "See pytest.log and train.log. See environment.json for runtime versions and source_manifest.json for source-data identities."]
    (OUT / "report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
