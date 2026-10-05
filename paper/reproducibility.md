# Reproducibility & Provenance Guide

This document identifies the frozen specifications, canonical artifacts, commits, and commands needed to reproduce the paper's headline temporal-regularization result. Earlier Bayesian/LoRA experiments remain part of the repository's scientific provenance but are not the primary reproducibility target for this manuscript.

---

## 1. Repository and Authoritative Phase 20 Commits

- **Repository**: `https://github.com/mbzuaiciao/distributed-portfolio-learning.git`
- **Primary branch**: `main`
- **Phase 20 preregistration**: `8f3470c5` — original optimal-horizon preregistration.
- **Shared-calendar-return DGP correction**: `2a68e6f6` — pre-execution correction eliminating cohort-specific realized shocks.
- **Final frozen Phase 20 specification**: `cfdb7529` — adds the frozen signal-strength dimension and 52-week finite signal history.
- **Frozen Phase 20 execution and deliverables**: `0dab572e`.
- **Exact-null invariant audit**: `1de89121`.
- **Paper consolidation**: `9a29482e`.
- **Scientific accuracy / publication-readiness review**: `554b10e4`.

The scientific result should be interpreted relative to the frozen specification at `cfdb7529`; later paper commits do not alter the raw experiment outputs.

---

## 2. Frozen Phase 20 Design

### Universe and calendar
- Assets: **30**
- Calendar weeks: **156**
- Finite signal-history truncation: **52 weeks**
- Candidate horizons: `H in {1, 2, 4, 8, 13}`

### Parameter grids
- Signal strength `beta`: `{0.000, 0.005, 0.010, 0.020}`
- Persistence `rho`: `{0.00, 0.25, 0.50, 0.75, 0.90, 0.97}`
- Observation noise: `{0.0, 0.5, 1.0, 2.0}`
- Transaction costs: `{0, 5, 10, 25, 50}` bps
- Risk aversion `gamma`: `{0.0, 0.5, 1.0, 2.5, 5.0}`
- Seeds: `200001, 200002, ..., 200030`

### Experiment count
- 12,000 parameter cells per seed
- 30 seeds
- **360,000 total parameter-seed evaluations**
- 14,400 unique simulated rollouts before cost/risk-aversion reuse

### Return DGP
Each asset has one realized return per calendar week, shared by every active cohort holding that asset:

```text
R_{i,u} = beta * sum_{k=0}^{52} rho^k * s_{i,u-k}
          + eps_{mkt,u} + xi_{i,u}
```

with market-factor standard deviation 0.02 and idiosyncratic standard deviation 0.04.

The `rho=0.97` case is explicitly finite-memory: `0.97^52 ~= 0.203`.

---

## 3. Canonical Phase 20 Files

The primary specification and outputs are under:

`artifacts/phase20_optimal_cohort_horizon/`

Key files:
- `preregistration.json`
- `preregistration.md`
- `simulation_design.json`
- `analytical_derivation.md`
- `experiment_matrix.csv`
- `horizon_optimum_table.csv`
- `comparative_statics_table.csv`
- `uncertainty_and_ties.csv`
- `utility_decomposition_table.csv`
- `breakeven_boundaries.csv`
- `outcome_classification.json`
- `scientific_report.md`
- `null_regime_invariant_audit.json`
- `null_regime_invariant_audit.md`

The experiment runner is:

`experiments/run_phase20_optimal_cohort_horizon.py`

The dedicated tests are:

`tests/test_phase20_optimal_cohort_horizon.py`

---

## 4. Reproduction Commands

From the repository root, using the repository's configured virtual environment:

```bash
# Verify the frozen preregistration from the runner.
.venv/bin/python3 -c "from experiments.run_phase20_optimal_cohort_horizon import load_freeze; cfg, sha = load_freeze(); print(sha)"

# Reproduce the frozen Phase 20 experiment.
.venv/bin/python3 experiments/run_phase20_optimal_cohort_horizon.py

# Run the Phase 20 tests.
.venv/bin/pytest tests/test_phase20_optimal_cohort_horizon.py

# Run the full project test suite.
.venv/bin/pytest tests/

# Static checks.
.venv/bin/ruff check .
```

A reproduction run will regenerate the Phase 20 output artifacts. For manuscript-only edits, do not rerun the simulation.

---

## 5. Primary Verification Targets

A correct reproduction should recover:

- Outcome A under the frozen decision tree.
- H1 cost monotonicity: 100% simulation consistency.
- H2 persistence monotonicity: 90% consistency for active-signal slices.
- H3 short-lived-alpha reversal: 100%.
- H4 noise regularization: 100%.
- H5 risk-aversion preference: 100%.
- Maximum accounting-closure error approximately `3.144e-17`, below the frozen `1e-12` tolerance.
- Exact `beta=0` invariant: all candidate horizons tie under constant uniform weights, with zero discrepancy across the seven audited invariants.

---

## 6. Retrospective Empirical Provenance

The paper uses Phases 18 and 19 only as retrospective motivation and reconciliation. These historical analyses do not constitute prospective validation of the Phase 20 frontier.

Canonical directories:
- `artifacts/phase18_clean_cohort_revalidation/`
- `artifacts/phase19_architecture_signal_stress_test/`

Important scope limitation: the empirical return history terminates in May 2025, and Phase 18 uses reconstructed/interpolated weekly returns rather than continuous observed weekly marks.

---

## 7. Historical ML / Bayesian Branches

Earlier Bayesian LoRA, uncertainty-calibration, leakage-audit, and collaborative-LoRA work remains preserved in the repository as negative or provenance evidence. It should not be treated as the primary reproduction path for the current manuscript.

Those branches explain how the project moved from model-level uncertainty claims toward the portfolio-execution question, but the headline paper result is the Phase 20 temporal-regularization frontier.

---

## 8. Submission-Time Reproducibility Check

Before external submission:

1. verify the manuscript's numerical statements against the canonical artifacts;
2. verify raw Phase 18-20 artifact hashes have not changed during manuscript editing;
3. run the full test suite and Ruff;
4. record the final manuscript commit SHA;
5. preserve the exact environment specification used for the final reproducibility run;
6. archive a machine-readable dependency lockfile and, if required by the venue, a DOI-backed code/data snapshot.
