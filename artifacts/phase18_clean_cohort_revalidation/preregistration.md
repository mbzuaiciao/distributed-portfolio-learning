# Frozen clean cohort revalidation

The accompanying `preregistration.json` is authoritative. Commit it before generating any new predictions or portfolio outcomes. This is retrospective, with historical results already known; it is not prospective confirmation.

Primary comparison: H=4 minus H=1 seed-mean net weekly utility, defined by the existing simulator as mean net return minus half its sample variance (ddof=1). Both use exposure 0.7073, cash return 0, initial entry cost, 10 bps per unit of risky L1 trading, equal capital per active cohort, and frozen Phase 9.1 within-cohort Markowitz weights. Horizon search and timing overlays are prohibited.

Economic continuation requires delta utility >= .0001 (1 bp/week), a paired 95% 13-week moving-block bootstrap lower bound >0, and positive delta in >=2 of 3 seeds. Persistence requires seed-mean incremental weeks 2–4 rank IC >=.02, incremental Q5-Q1 spread >=.0025, both 95% lower bounds >0, and a one-sided within-cohort permutation p<=.05. Week-one returns cannot contribute to this persistence gate. Quintiles use non-overlapping stable equal-count bins. Bootstrap draws preserve shared calendar blocks across seeds; three model seeds are not three independent market histories.

Mechanical benefit requires all three: traded turnover falls >=25%, net volatility falls >=20%, and effective holdings rise >=1.5x. Effective holdings is the reciprocal sum of squared aggregate risky weights; positive-weight active names are reported separately from the historical simulator's dictionary-key count. Base turnover excludes inception per the simulator; traded turnover includes it. Gross risky L1 turnover and one-way half-L1 turnover are exported with their conventions.

Apply categories in order, without revision:

1. **A — Clean confirmation:** persistence AND economic continuation.
2. **B — Architecture only:** NOT persistence, all mechanical thresholds, and nonnegative delta net utility.
3. **C — No useful continuation:** NOT persistence, delta net utility < .0001, and NOT all mechanical thresholds.
4. **D — Mixed / inconclusive:** all remaining combinations; any integrity failure also forces D.

An A under reconstructed endpoint returns must be qualified as endpoint-reconstruction confirmation. Genuine observed week-by-week alpha remains unresolved without actual weekly prices. Positive mechanics without persistence cannot be claimed as alpha confirmation.

Matrix: seeds 42/123/777, diagnostic cumulative H=1/2/4/8/13 and reconstructed age weeks 1–13; primary H=1/4 portfolios; training-prevalence constant, fixed training-only ticker-risk baseline (20 pseudo-events), same-exposure equal weight, 20 random-ranking portfolio controls and 199 signal permutations. Report 2022–2023 only descriptively; primary evaluation is the exact historical 2024–2025 holdout event set. B=1000 paired bootstrap, block length 13 calendar weeks. No model or H selection uses these diagnostics.

The canonical Phase 16.1 recipe is reproduced at fixed epoch two, without repeating its regularization search or loading leaked weights. Clean labels use the pre-2022 cutoff; no OOS quantile is used. Clean text contains only company, ticker and announcement date. Input mutation tests, explicit event mapping, file hashes, checkpoint hashes, parent/freeze/execution git SHAs, seeds and full configuration are mandatory.

Estimated local MPS runtime: 1140 seconds. First-ten-batch speed check projects training plus inference; use cloud if the projection exceeds 1200 seconds, preferring Azure where practical and using only dedicated project resources. Checkpoints resume only completed seeds with matching recipe and hashes. Tests and lint run before evaluation. Historical files are preserved.
