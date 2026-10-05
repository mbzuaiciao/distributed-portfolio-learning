# Submission Checklist

This checklist tracks the remaining non-experimental work required before external submission.

## Manuscript
- [x] Scope the headline contribution as analytical + controlled synthetic, with retrospective empirical motivation.
- [x] Define "optimal horizon" as utility-maximizing within the frozen candidate set and model.
- [x] Correct Phase 20 calendar length to 156 weeks and seed schedule to 200001-200030.
- [x] Qualify the Phase 18/19-to-Phase 20 mapping as descriptive rather than structurally identified.
- [x] Remove causal wording that treats the retrospective H=4 result as proof of the synthetic mechanism.
- [ ] Perform a final line edit after figures and tables are inserted.
- [ ] Freeze author names, affiliations, acknowledgements, disclosures, and data/code availability statement.

## Figures and tables
- [ ] Produce final publication versions of Figures 1-4.
- [ ] Ensure Figure 3 is labeled as a synthetic/controlled-simulation frontier, not an empirical frontier.
- [ ] Ensure Figure 4 is labeled retrospective empirical motivation/sanity checking, not independent confirmation.
- [ ] Build the three main tables from canonical artifacts and verify every displayed number.
- [ ] Move detailed null-audit and forensic figures/tables to supplementary material.

## References and literature positioning
The current Markdown draft contains provisional author-year citations but no verified reference list in the paper package. Do not submit until every citation is backed by a complete bibliographic entry.

- [x] Verify the exact bibliographic record for Papastaikoudis et al. (2025).
- [x] Verify and add the cited Bayesian neural-network reference (Blundell et al., 2015).
- [x] Correct and verify the LoRA conference citation as Hu et al. (2022).
- [x] Add the closest portfolio-execution comparator: Gârleanu and Pedersen (2013).
- [x] Add Bernard and Thomas (1989) only for post-earnings-announcement-drift motivation.
- [x] Add Markowitz (1952) for the mean-variance foundation.
- [x] Check every current in-text citation against the verified reference ledger.
- [ ] Avoid novelty claims such as "first" unless the literature review establishes them.

## Reproducibility
- [x] Recenter `paper/reproducibility.md` on the frozen Phase 20 experiment.
- [ ] Record the final dependency lockfile/environment snapshot.
- [ ] Re-run full tests and Ruff immediately before submission.
- [ ] Record final raw-artifact hashes and final manuscript commit SHA.
- [ ] Archive code/data in the form required by the target venue.

## Venue formatting
A target journal has not yet been frozen, so venue-specific formatting should remain separate from scientific content.

- [x] Select target venue: **Quantitative Finance** (primary); *Mathematical Finance* retained as a more theory-intensive secondary option.
- [ ] Apply its word/page limits, abstract length, heading format, citation style, figure-resolution rules, and supplementary-material policy.
- [ ] Prepare anonymous/blinded version if required.
- [ ] Prepare cover letter and required declarations.
- [ ] Generate and visually inspect the final PDF.

## Final scientific guardrails
The submission must not claim:
- prospective or live trading validation;
- statistically established multi-week post-earnings alpha in Phase 18;
- universal superiority of H=4;
- structural identification of Phase 18/19 parameters inside the Phase 20 DGP;
- real-world global optimality outside the frozen model, utility, and candidate horizon set.
