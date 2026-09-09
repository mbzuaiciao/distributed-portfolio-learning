# Phase 12.2: Frozen 2026 Input Provenance Audit

## 1. Executive Summary
- **Total Frozen Input Rows Evaluated**: 52
- **Retrospective-but-Frozen Rows**: 34 (5 Tranche-1 corporate earnings events + 29 pre-tranche-1 calendar episodes)
- **Genuinely Prospective Rows**: 0 (0 completed $H=4$; NVDA event occurred before freeze and is right-censored; future events do not yet exist)
- **Future Placeholders / Not Yet Predictable**: 18 (Calendar weeks 2026-W35 through 2026-W52)
- **Causal-Input Violations**: **0** (All predictions generated using documents available prior to generation timestamp)
- **Rows Eligible Toward Next Formal Milestone**: **0**
- **Canonical Prediction SHA256**: `6542b33fa5810985beff8c835e71b598ffaf47bf943e88d43f7c50d7dcb30aeb` (**Verified Byte-for-Byte Unchanged**)

---

## 2. Temporal Provenance and Date Reconciliation Table

| Event ID | Ticker | Fiscal Quarter | Event Timestamp | Filing Timestamp | First Tradable Date | Cohort Week | Doc Type | Text Available? | Prediction Exists? | Prospective Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `GOOGL_2026_Q2` | GOOGL | 2026_Q2 | 2026-07-23 16:00:00 | 2026-07-23 16:00:00 | 2026-07-24 | 2026-W30 | 10-Q | True | True | `RETROSPECTIVE-BUT-FROZEN` |
| `MSFT_2026_Q4` | MSFT | 2026_Q4 | 2026-07-28 16:05:00 | 2026-07-28 16:05:00 | 2026-07-29 | 2026-W31 | 10-K | True | True | `RETROSPECTIVE-BUT-FROZEN` |
| `AAPL_2026_Q3` | AAPL | 2026_Q3 | 2026-07-31 16:30:00 | 2026-07-31 16:30:00 | 2026-08-03 | 2026-W31 | 10-Q | True | True | `RETROSPECTIVE-BUT-FROZEN` |
| `AMZN_2026_Q2` | AMZN | 2026_Q2 | 2026-08-01 16:00:00 | 2026-08-01 16:00:00 | 2026-08-03 | 2026-W31 | 10-Q | True | True | `RETROSPECTIVE-BUT-FROZEN` |
| `NVDA_2026_Q2` | NVDA | 2026_Q2 | 2026-08-26 16:20:00 | 2026-08-26 16:20:00 | 2026-08-27 | 2026-W35 | 10-Q | True | True | `RETROSPECTIVE-BUT-FROZEN` |
| `CALENDAR_2026_W01..W29` (29 rows) | UNIVERSE | Q1_2026 | None | None | None | W01..W29 | None | False | False | `RETROSPECTIVE-BUT-FROZEN` |
| `CALENDAR_2026_W35..W52` (18 rows) | UNIVERSE | Q3/Q4_2026 | None | None | None | W35..W52 | None | False | False | `FUTURE PLACEHOLDER / NOT YET PREDICTABLE` |

---

## 3. Investigation of the 18 Future-Dated Episodes
- **Classification**: **Category C — Scheduled-event placeholder with no actual filing text**.
- **Finding**: For weeks 2026-W35 through 2026-W52, no filing documents or text excerpts were provided to FinBERT. No predictions exist in `frozen_2026_predictions.csv` for these rows (`prediction_row_exists = False`).
- **Scientific Implication**: A valid prospective prediction for an earnings announcement requires the text excerpt of that earnings release. Future Q3/Q4 2026 announcements cannot be scored until the actual filings are published. They are strictly future placeholders and do not constitute pre-computed prospective predictions.

---

## 4. Reconciliation of Event Dates
1. **`MSFT_2026_Q4`**:
   - Canonical SEC filing timestamp: `2026-07-28 16:05:00` (after market close).
   - First tradable session: `2026-07-29`.
   - Cause of discrepancy: The Phase 12.1 funnel narrative inadvertently listed `2026-07-23` due to a clerical copy-paste typo from GOOGL. The underlying SEC manifest and trading ledger have always correctly used `2026-07-28`.
2. **`AAPL_2026_Q3`**:
   - Canonical SEC filing timestamp: `2026-07-31 16:30:00` (Friday afternoon).
   - First tradable session: `2026-08-03` (Monday post-weekend).
   - Cause of discrepancy: Some narrative summaries cited `2026-08-01` because of weekend calendar grouping and co-cohorting with AMZN.
3. **`AMZN_2026_Q2`**:
   - Canonical filing timestamp: `2026-08-01 16:00:00` (Saturday).
   - First tradable session: `2026-08-03` (Monday).
   - Verification: `2026-08-01` is the actual SEC EDGAR filing timestamp on a weekend day, not a derived cohort date.

---

## 5. Prediction-Time Causality Verification
For all 5 corporate events:
$$\text{input\_document\_timestamp} \le \text{prediction\_generation\_timestamp (2026-09-09)}$$
- Zero predictions were generated using future text.
- Zero look-ahead leakage exists in the text inputs.

---

## 6. Milestone Eligibility Clarification
The stopping criterion remains strictly:
$$\min(\text{unique completed } H=4 \text{ prospective events} \ge 20,\; \text{cutoff date} = 2026\text{-}12\text{-}31)$$
- **Rule Enforcement**: Only events classified as `GENUINELY PROSPECTIVE` count toward the $\ge 20$ target.
- **Current Count**: **0** completed genuinely prospective events.
  - The 4 completed events in Tranche 1 are `RETROSPECTIVE-BUT-FROZEN` (their market outcomes transpired in August 2026, before the September 9 freeze).
  - `NVDA_2026_Q2` is right-censored ($<20$ days observed as of September 9) and its event date preceded the freeze.
  - The 18 future rows are placeholders with no text or predictions yet.
