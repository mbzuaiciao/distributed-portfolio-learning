# Paper Consolidation: When Uncertainty Looks Useful

This directory consolidates the complete empirical and methodological findings of the research project into a publication-grade scientific manuscript and supporting documentation.

## Working Title
**"When Uncertainty Looks Useful: Target Leakage, Metric Artifacts, and Bayesian LoRA for Financial NLP"**

## Central Research Question
*When does Bayesian epistemic uncertainty actually provide measurable value in financial NLP, and how can evaluation pitfalls artificially inflate its apparent utility?*

## Key Documents in this Directory

| Document | Purpose |
| :--- | :--- |
| [`outline.md`](./outline.md) | Structural overview and section-by-section breakdown (11 sections). |
| [`claims.md`](./claims.md) | Comprehensive claims ledger with empirical status, supporting artifacts, and allowed/forbidden wording. |
| [`results_inventory.md`](./results_inventory.md) | Phase-by-phase provenance taxonomy classifying all results as Valid Clean Primary, Valid Clean Secondary, Diagnostic Only, Compromised by Leakage, or Superseded. |
| [`figures_plan.md`](./figures_plan.md) | Minimal, publication-ready figure specifications (timeline, model comparisons, MC equivalence, regime breakdowns). |
| [`tables_plan.md`](./tables_plan.md) | Formal LaTeX/Markdown table specifications (clean model comparison, regime breakdown, compute cost, audit summary). |
| [`draft.md`](./draft.md) | Complete manuscript draft (~6,000 words) following the authoritative scientific narrative. |
| [`limitations.md`](./limitations.md) | Explicit bounding of task, dataset, text representation, market regime, and model scope. |
| [`reproducibility.md`](./reproducibility.md) | Complete provenance manifest, environment details, git commits, random seeds, and verification instructions. |

## Core Scientific Finding Summary
1. **Target Leakage Discovery**: Historical training pipelines (Phases 11–16) concatenated continuous post-event price moves (`. 1-day move was ` + `move_1d`) into input text, providing the model with label ground truth and causing artificial validation performance ($\text{AUROC} \approx 1.0$) followed by out-of-sample distribution collapse.
2. **Reliability Metric Artifact**: The apparent correlation between predictive entropy and prediction error ($\rho \approx 0.60$) was mathematically proven to be an artifact of probability shrinkage on class-imbalanced data ($80\%$ non-downside): within $y=0$, $\rho \equiv +1.0$; within $y=1$, $\rho \equiv -1.0$.
3. **Clean Retraining Adjudication**: When retrained from scratch on strictly clean text across 3 seeds on clean regimes (2022–2025, $N=12,300$):
   - Clean Bayesian Posterior Mean achieves $\text{AUROC} = 0.5253$, $\text{Brier} = 0.15987$, $\text{LogLoss} = 0.50007$.
   - Clean Deterministic LoRA achieves $\text{AUROC} = 0.5131$, $\text{Brier} = 0.16079$, $\text{LogLoss} = 0.50332$.
   - Trivial Constant Base-Rate Null ($p \equiv 0.20$) achieves $\text{Brier} = 0.16000$, outperforming Deterministic LoRA and matching Bayesian LoRA.
4. **Computational Redundancy of Monte Carlo Sampling**: 32-draw Monte Carlo posterior sampling is statistically and practically indistinguishable from a single forward pass of the Bayesian posterior mean ($\Delta \text{Brier} = +0.00001, \Delta \text{AUROC} = -0.00028$, strictly within $\pm 0.002$ and $\pm 0.005$ equivalence margins) while imposing a $30.6\times$ latency penalty.
5. **Downstream Portfolio Utility**: No robust, positive economic benefit from asset-level or portfolio-level uncertainty adjustments was established out of sample. Under unadjusted Markowitz ($M_0$), Deterministic LoRA achieved higher Sharpe ($0.5580$ vs. $0.3360$) and higher utility ($+0.1392$ vs. $+0.0559$).
