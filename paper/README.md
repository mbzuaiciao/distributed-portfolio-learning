# Paper Consolidation: When Uncertainty Looks Useful

This directory consolidates the complete empirical and methodological findings of the research project into a submission-ready scientific manuscript and supporting documentation.

## Recommended Working Title
**"When Uncertainty Looks Useful: Target Leakage, Reliability Metrics, and Bayesian LoRA in Financial Event Risk"**  
*(Alternative: "Target Leakage, Reliability Metrics, and Bayesian LoRA: A Forensic Audit and Empirical Re-Evaluation in Earnings Event Risk")*

## Central Research Question
*When does Bayesian epistemic uncertainty provide measurable value in financial event risk forecasting, and how can subtle experimental pitfalls artificially inflate its apparent utility?*

## Key Documents in this Directory

| Document | Purpose |
| :--- | :--- |
| [`draft.md`](./draft.md) | Complete submission-ready manuscript draft (~5,800 words) adhering to strict scientific scoping and neutral phrasing. |
| [`submission_audit.md`](./submission_audit.md) | Comprehensive quantitative audit verifying every numerical statement, sample size, and metric against canonical artifacts. |
| [`claims.md`](./claims.md) | Authoritative claims ledger recording empirical status, supporting artifacts, allowed wording, forbidden overclaims, and scope bounds. |
| [`results_inventory.md`](./results_inventory.md) | Master provenance taxonomy classifying all results across Phases 11–17 into primary clean evidence vs. quarantined forensic artifacts. |
| [`outline.md`](./outline.md) | Structural overview and section-by-section breakdown (11 sections). |
| [`figures_plan.md`](./figures_plan.md) | Minimal, publication-ready figure specifications (timeline, model comparisons, MC equivalence, regime breakdowns). |
| [`tables_plan.md`](./tables_plan.md) | Formal LaTeX/Markdown table specifications with verified canonical numbers. |
| [`limitations.md`](./limitations.md) | Explicit boundary conditions detailing the minimal metadata input representation, task scope, and architectural focus. |
| [`reproducibility.md`](./reproducibility.md) | Complete provenance manifest, environment details, git commits, random seeds, and verification instructions. |

## Core Scientific Finding Summary
1. **Target Leakage Discovery**: Historical training pipelines (Phases 11–16) concatenated continuous post-event price moves (`. 1-day move was ` + `move_1d`) into input text, providing the model with label ground truth and causing artificial validation performance ($\text{AUROC} \approx 1.0$) followed by out-of-sample distribution collapse on clean text.
2. **Reliability Metric Artifact**: The apparent correlation between predictive entropy and prediction error ($\rho \approx 0.60$) was mathematically proven to be largely an artifact of probability shrinkage on class-imbalanced data ($80\%$ majority class): within $y=0$, $\rho \equiv +1.0$; within $y=1$, $\rho \equiv -1.0$.
3. **Clean Retraining Adjudication**: When retrained from scratch on minimal clean metadata across 3 seeds ($N_{\text{train}}=11,985$, $N_{\text{val}}=2,116$, $N_{\text{test}}=12,300$ across 2022–2025):
   - Clean Bayesian Posterior Mean achieves $\text{AUROC} = 0.5253$, $\text{Brier} = 0.15987$, $\text{LogLoss} = 0.50007$.
   - Clean Deterministic LoRA achieves $\text{AUROC} = 0.5131$, $\text{Brier} = 0.16079$, $\text{LogLoss} = 0.50332$.
   - Trivial Constant Base-Rate Null ($p \equiv 0.20$) achieves $\text{Brier} = 0.16000$, outperforming Deterministic LoRA and matching Bayesian LoRA.
4. **Computational Redundancy of Monte Carlo Sampling**: 32-draw Monte Carlo posterior sampling was practically equivalent to a single forward pass of the Bayesian posterior mean under preregistered equivalence margins ($\Delta \text{Brier} = +0.00001, \Delta \text{AUROC} = -0.00028$) while imposing a $30.6\times$ latency penalty in this experiment.
5. **Secondary Portfolio Evidence**: No robust, positive economic benefit from asset-level or portfolio-level uncertainty adjustments was established out of sample. Under unadjusted Markowitz ($M_0$), Deterministic LoRA achieved higher Sharpe ($0.5580$ vs. $0.3360$) and higher utility ($+0.1392$ vs. $+0.0559$), though portfolio conclusions remain secondary given weak underlying point-forecast signals.
