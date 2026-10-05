# Claims Ledger & Rhetorical Guardrails

This ledger records every primary and contextual empirical and methodological claim in the paper, specifying its verification status, supporting artifacts, provenance classification, allowed scientific wording, forbidden overclaims, and explicit scope boundaries.

---

## 1. Primary Theoretical & Synthetic Claims (Phase 20)

### Claim 1.1: Cohort Stacking Acts as a Temporal Regularizer
- **Status**: **CONFIRMED (ANALYTICALLY DERIVED & SYNTHETICALLY VALIDATED)**
- **Supporting Artifacts**:
  - `artifacts/phase20_optimal_cohort_horizon/analytical_derivation.md`
  - `artifacts/phase20_optimal_cohort_horizon/scientific_report.md`
  - `artifacts/phase20_optimal_cohort_horizon/outcome_classification.json`
- **Data Provenance**: Valid Clean Synthetic DGP & Closed-Form Derivations
- **Scope Limitation**: Applies to overlapping rolling cohort portfolios with discrete rebalancing and decaying predictive signals.
- **Allowed Wording**:
  - *"Cohort stacking acts as a temporal regularization mechanism that balances the responsiveness to fresh predictive information against execution turnover, transaction friction, and estimation noise."*
  - *"Overlapping cohort portfolios reduce portfolio turnover proportionally to $1/H$ and diversify cross-cohort entry variance, but incur an alpha dilution penalty as decaying signals age."*
- **Forbidden Overclaims**:
  - *"Cohort stacking creates new predictive information out of nothing."*
  - *"Cohort stacking is universally superior to single-period rebalancing regardless of signal strength or friction."*

---

### Claim 1.2: Monotonic Optimal Horizon Comparative Statics (H1–H5)
- **Status**: **CONFIRMED (PREREGISTERED SIMULATION, 360,000 EVALUATIONS)**
- **Supporting Artifacts**:
  - `artifacts/phase20_optimal_cohort_horizon/experiment_matrix.csv`
  - `artifacts/phase20_optimal_cohort_horizon/horizon_optimum_table.csv`
  - `artifacts/phase20_optimal_cohort_horizon/outcome_classification.json`
- **Data Provenance**: Controlled Synthetic Factorial Simulation across 30 seeds
- **Verified Consistency**:
  - H1 (Cost Monotonicity, $\partial H^* / \partial c \ge 0$): **100.0%**
  - H2 (Persistence Monotonicity, $\partial H^* / \partial \rho \ge 0$): **90.0%** (for $\beta > 0$)
  - H3 (Short-Lived Alpha Reversal, $\partial H^* / \partial \beta \le 0$ under fast decay): **100.0%**
  - H4 (Noise Regularization, $\partial H^* / \partial \sigma_{\text{noise}} \ge 0$): **100.0%**
  - H5 (Risk Aversion Preference, $\partial H^* / \partial \gamma \ge 0$): **100.0%**
- **Allowed Wording**:
  - *"The optimal cohort holding horizon $H^*$ increases monotonically with transaction costs, signal half-life, observation noise, and risk aversion, while strong, rapidly decaying alpha favors shorter, more responsive horizons."*
- **Forbidden Overclaims**:
  - *"The comparative statics hold identically in every single individual seed without exception."* (H2 holds across 90% of slices, with minor plateau ties).

---

### Claim 1.3: Machine-Precision Accounting Closure & Exact-Null Invariant
- **Status**: **CONFIRMED (MACHINE-PRECISION AUDIT)**
- **Supporting Artifacts**:
  - `artifacts/phase20_optimal_cohort_horizon/null_regime_invariant_audit.md`
  - `artifacts/phase20_optimal_cohort_horizon/null_regime_invariant_audit.json`
  - `artifacts/phase20_optimal_cohort_horizon/outcome_classification.json`
- **Data Provenance**: Exhaustive Invariant Audit across all 288,000 pairwise cells and 72,000 null-regime cells
- **Verified Metrics**:
  - Maximum numerical closure error: $3.144 \times 10^{-17} \ll 10^{-12}$.
  - Null regime ($\beta = 0.000$) maximum absolute discrepancy across 7 invariants: **0.00e+00**.
  - All $\beta = 0$ cells classified as `Tie_1-2-4-8-13` with $p = 1.000$.
- **Allowed Wording**:
  - *"Accounting closure between utility differences, gross return deltas, turnover cost savings, and risk differentials holds to machine precision ($3.14\times 10^{-17}$)."*
  - *"Under constant uniform weighting ($\beta = 0.000$), rebalancing turnover is identically zero, producing an exact mathematical tie across all evaluated horizons."*
- **Forbidden Overclaims**:
  - *"Stacking at $H \ge 4$ dominates under transaction friction when weights are time-invariant."* (This claim was falsified; friction only advantages stacking when base weights churn).

---

## 2. Empirical Reconciliation Claims (Phases 18 & 19)

### Claim 2.1: Empirical $H=4$ Survival was an Architectural Effect
- **Status**: **SUPPORTED (CLEAN EMPIRICAL REVALIDATION & STRESS TESTING)**
- **Supporting Artifacts**:
  - `artifacts/phase18_clean_cohort_revalidation/scientific_report.md`
  - `artifacts/phase19_architecture_signal_stress_test/scientific_report.md`
  - `artifacts/phase20_optimal_cohort_horizon/empirical_sanity_mapping.json`
- **Data Provenance**: Quarterly earnings event study (2022–2025) and architecture stress tests
- **Allowed Wording**:
  - *"In clean empirical revalidation on uncompromised metadata, residual post-earnings predictive alpha across weeks 2–4 was statistically insignificant (residual IC $= 0.0060, p=0.105$)."*
  - *"Nevertheless, multi-week cohort stacking ($H=4$) delivered robust turnover reductions ($-74\%$) and volatility reductions ($-66\%$) that generalized across empirical, random, constant, and synthetic signal regimes."*
  - *"Mapping empirical parameters ($c = 10\text{ bps}$, Rank IC $\approx 0.05$, $\tau_{1/2} \approx 1\text{--}2.4\text{ wks}$) into the synthetic phase diagram confirms that $H=4$ was optimal due to execution cost amortization and noise smoothing rather than persistent alpha."*
- **Forbidden Overclaims**:
  - *"We have discovered a high-capacity, live-trading alpha strategy that beats the market."*
  - *"The empirical results constitute independent prospective confirmation of the Phase 20 frontier."* (Empirical data is retrospective only).

---

## 3. Quarantined Contextual & Forensic Claims (Phases 11–17)

### Claim 3.1: Target Outcome Leakage in Historical Training Text
- **Status**: **CONFIRMED (FORENSIC CODE AUDIT)**
- **Supporting Artifacts**: `artifacts/phase16_1_audit/input_leakage_audit.json`
- **Role in Paper**: Contextual background explaining how the research inquiry was redirected away from model complexity toward execution architecture.

### Claim 3.2: Mechanical Distortion in Spearman Reliability Metrics
- **Status**: **CONFIRMED (ANALYTICAL DERIVATION & CLEAN AUDIT)**
- **Supporting Artifacts**: `artifacts/phase16_1_audit/within_class_reliability.csv`
- **Role in Paper**: Methodological finding explaining why predictive entropy appeared correlated with forecast error in earlier phases.

### Claim 3.3: Collaborative LoRA Pilot Closure
- **Status**: **CONFIRMED NEGATIVE (PILOT EVALUATION)**
- **Supporting Artifacts**: `docs/collaborative_lora_branch_closure.md`
- **Role in Paper**: Transparent negative result establishing that heterogeneous specialist coordination failed continuation criteria.
