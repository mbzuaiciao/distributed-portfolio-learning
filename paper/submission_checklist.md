# Journal Submission Checklist: Quantitative Finance

**Target Journal**: *Quantitative Finance* (Taylor & Francis)  
**Paper Title**: Temporal Regularization in Portfolio Execution: Optimal Holding Horizons Under Signal Decay and Trading Friction  
**Date**: October 5, 2026  
**Status**: Ready for Submission  

---

## 1. Primary Submission Documents

- [x] **Main Manuscript Source**: `paper/submission/manuscript.tex` (Clean, self-contained LaTeX source following journal conventions).
- [x] **Main Manuscript PDF**: `paper/submission/manuscript.pdf` (15 pages, 0 compilation errors, 0 bibtex warnings, 0 overfull hboxes).
- [x] **Bibliography Database**: `paper/submission/references.bib` (12 verified references with complete DOIs/metadata).
- [x] **Declarations & Disclosures**: `paper/submission/declarations.md` (Funding, conflicts of interest, data/code availability, author contributions).
- [x] **Typesetting & Visual Quality Audit**: `paper/submission/pdf_quality_check.md` (Visual inspection of margins, font kerning, float alignment, and table scaling).

---

## 2. Manuscript Content & Scientific Guardrails

- [x] Scope the headline contribution as analytical + controlled synthetic, with retrospective empirical motivation.
- [x] Define "optimal horizon" as utility-maximizing within the frozen candidate set $H \in \{1, 2, 4, 8, 13\}$ and model.
- [x] Correct Phase 20 calendar length to 156 weeks and seed schedule to 200001–200030.
- [x] Qualify the Phase 18/19-to-Phase 20 mapping as descriptive rather than structurally identified.
- [x] Remove causal wording that treats the retrospective $H=4$ result as proof of the synthetic mechanism.
- [x] Perform a final line edit after figures and tables are inserted.
- [x] Disclose author names, affiliations, acknowledgements, disclosures, and data/code availability statements in `declarations.md`.
- [x] Confirm no claims of:
  - prospective or live trading validation;
  - statistically established multi-week post-earnings alpha in Phase 18;
  - universal superiority of $H=4$;
  - structural identification of Phase 18/19 parameters inside the Phase 20 DGP;
  - real-world global optimality outside the frozen model, utility, and candidate horizon set.

---

## 3. Figure Package Audit

All figures are compiled in vector PDF and 300 DPI PNG formats in `paper/figures/` and mirrored in `paper/submission/figures/`:

- [x] **Figure 1**: `fig1_temporal_regularization_architecture.{pdf,png}`
  - *Content*: Architectural comparison of $H=1$ versus $H=4$ overlapping cohorts; labels the 3 competing economic forces.
  - *Type*: Conceptual schematic.
- [x] **Figure 2**: `fig2_analytical_tradeoff.{pdf,png}`
  - *Content*: 4-panel analytical curves of alpha dilution, cost amortization, risk reduction, and expected utility across $\rho$ and $c$.
  - *Type*: Stylized theoretical model curves.
- [x] **Figure 3**: `fig3_phase20_synthetic_horizon_frontier.{pdf,png}`
  - *Content*: 2D phase diagrams of utility-maximizing horizon $H^* \in \{1, 2, 4, 8, 13\}$ across $(c, \rho)$ with exact-null tie baseline.
  - *Type*: Controlled synthetic simulation ($N=30, T=156, S=30, 360{,}000$ evaluations).
  - *Labeling*: Explicitly labeled as a controlled simulation frontier, not an empirical frontier.
- [x] **Figure 4**: `fig4_retrospective_empirical_motivation.{pdf,png}`
  - *Content*: 4-panel retrospective empirical evidence (Phase 18 signal decay, turnover reduction, volatility reduction, and Phase 19 synthetic alpha reversal).
  - *Type*: Retrospective empirical motivation and reconciliation (not independent confirmation).

---

## 4. Table Package Audit

All tables are compiled in LaTeX (`.tex`) and Markdown (`.md`) formats in `paper/tables/` and mirrored in `paper/submission/tables/`:

- [x] **Table 1**: `table1_comparative_statics.{tex,md}`
  - *Content*: Hypotheses H1–H5 confirmation rates across 360,000 evaluations (H1: 100%, H2: 90%, H3: 100%, H4: 100%, H5: 100%).
- [x] **Table 2**: `table2_phase20_design_and_audit.{tex,md}`
  - *Content*: Complete factorial parameter grid ($N=30, T=156, K=52$, 30 seeds, 360,000 evaluations), closure error ($3.144\times 10^{-17}$), exact-null tie ($0.00\text{e}+00$).
- [x] **Table 3**: `table3_retrospective_empirical_evidence.{tex,md}`
  - *Content*: Panel A (Phase 18 clean revalidation) and Panel B (Phase 19 signal stress tests across 5 regimes).

---

## 5. References and Literature Positioning

- [x] Verify the exact bibliographic record for Papastaikoudis et al. (2025).
- [x] Verify and add the cited Bayesian neural-network reference (Blundell et al., 2015).
- [x] Correct and verify the LoRA conference citation as Hu et al. (2022).
- [x] Add the closest portfolio-execution comparator: Gârleanu and Pedersen (2013, 2016).
- [x] Add Bernard and Thomas (1989, 1990) only for post-earnings-announcement-drift motivation.
- [x] Add Markowitz (1952) and Merton (1969, 1971) for the theoretical portfolio execution foundations.
- [x] Add Jegadeesh and Titman (1993, 2001) for empirical staggered cohort momentum precedents.
- [x] Check every current in-text citation against the verified reference ledger.
- [x] Avoid unsupported novelty claims such as "first" or "universal".

---

## 6. Reproducibility & Computational Integrity

- [x] Recenter `paper/reproducibility.md` on the frozen Phase 20 experiment.
- [x] Verify all 30 raw CSV artifact hashes across Phases 18, 19, and 20 are unchanged.
- [x] Re-run full test suite (`pytest`, 305 tests passed).
- [x] Re-run linter (`ruff check .`, 0 errors).
- [x] Confirm accounting closure error $\le 3.144\times 10^{-17} < 10^{-12}$.
- [x] Confirm exact-null ($\beta=0$) discrepancy $0.00\text{e}+00$ across all 7 invariants.

---

## 7. ScholarOne Submission Portal Placeholders

The following items are prepared for entry during the online submission workflow:

- [ ] **Corresponding Author ORCID**: `[Author ORCID to be entered via ScholarOne]`
- [ ] **Suggested Reviewers**:
  - Reviewer 1: Expert in transaction costs and dynamic portfolio execution (e.g., Gârleanu/Pedersen literature).
  - Reviewer 2: Expert in empirical factor investing and portfolio turnover heuristics.
  - Reviewer 3: Expert in machine learning regularization and financial econometrics.
- [ ] **Opposed Reviewers**: `[None]`
- [ ] **Cover Letter**: To Editor-in-Chief, *Quantitative Finance*. (Draft included in journal package).
- [ ] **Copyright & Open Access Selection**: Standard Taylor & Francis author publishing agreement.
