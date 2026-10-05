# Paper Consolidation: Temporal Regularization in Portfolio Execution

This directory consolidates the theoretical derivations, synthetic simulations, empirical reconciliations, and manuscript assets for the paper:

## Title
**"Temporal Regularization in Portfolio Execution: Optimal Holding Horizons Under Signal Decay and Trading Friction"**

*(Working Paper, October 2026)*

## Central Research Question
*How should the optimal cohort holding horizon $H^*$ depend on signal persistence ($\tau$), observation noise ($\sigma_{\text{noise}}$), signal strength ($\beta$), transaction costs ($c$), and risk aversion ($\gamma$)?*

## Working Thesis
Overlapping cohort portfolios function as a **temporal regularization mechanism** that balances the responsiveness to fresh predictive signals against execution turnover, transaction friction, and estimation noise. The optimal holding horizon increases monotonically with signal persistence, transaction costs, signal noise, and risk aversion, while strong, rapidly decaying alpha favors shorter horizons.

## Document Manifest

| Document | Purpose |
| :--- | :--- |
| [`draft.md`](./draft.md) | Consolidated, submission-ready manuscript draft (10 sections) adhering to neutral scientific phrasing. |
| [`outline.md`](./outline.md) | Structural overview and section-by-section specification of the 10 manuscript sections. |
| [`claims.md`](./claims.md) | Authoritative claims ledger recording empirical status, supporting artifacts, allowed wording, and forbidden overclaims. |
| [`figures_plan.md`](./figures_plan.md) | Specifications for the 4 core figures (conceptual, analytical, phase diagram, empirical sanity check). |
| [`tables_plan.md`](./tables_plan.md) | Formal table specifications for comparative statics, accounting closure, and parameter mapping. |
| [`limitations.md`](./limitations.md) | Explicit boundary conditions detailing synthetic DGP assumptions, cost models, and prospective scope. |
| [`results_inventory.md`](./results_inventory.md) | Master inventory tracking primary Phase 20 evidence vs. evolutionary forensic/negative results. |
| [`reproducibility.md`](./reproducibility.md) | Complete provenance manifest, environment details, git commits, seeds, and execution instructions. |
| [`submission_audit.md`](./submission_audit.md) | Quantitative verification audit linking numerical statements to canonical artifacts. |

## Core Scientific Finding Summary
1. **Analytical Frontier**: Derived stylized expected utility $U(H) = \mu_0 \frac{1 - \rho^H}{H(1 - \rho)} - \frac{c \cdot \text{TO}_0}{H} - \frac{1}{2}\gamma(\sigma_0^2 + \frac{\sigma_1^2}{H})$, establishing monotonic comparative statics across friction, persistence, noise, alpha value, and risk aversion.
2. **Synthetic DGP Confirmation**: Preregistered simulation across 360,000 evaluations (30 seeds) with a single realized calendar return per asset per week confirmed 5/5 comparative static hypotheses (H1: 100%, H2: 90%, H3: 100%, H4: 100%, H5: 100%).
3. **Accounting Closure & Null Regime**: Audited numerical closure error to machine precision ($3.14\times 10^{-17} < 10^{-12}$) and proved that time-invariant uniform weights ($\beta = 0$) result in exact 5-way ties across all horizons ($0.00\text{e}+00$ discrepancy).
4. **Empirical Reconciliation**: Mapped earlier Phase 18/19 findings into the synthetic phase space, confirming that empirical $H=4$ survival was an architectural regularizer effect driven by noise suppression and turnover reduction rather than persistent alpha.
5. **Evolutionary Negative Results**: Quarantined earlier investigations (target leakage in text serialization, class-imbalance entropy-error artifact, collaborative LoRA pilot failure, lack of independent 2026 data) as context explaining how the temporal regularization question emerged.
