# Phase 13B.1: Model-Training / Evaluation Overlap Audit Report

**Date:** 2026-09-11  
**Status:** COMPLETE & FROZEN  
**Audit Scope:** 8,692 valid historical earnings events across 2018–2025  
**Model Examined:** Replicated Bayesian-LoRA checkpoints (Seeds 42, 123, 777)  
**Artifacts Generated:**
- `artifacts/phase13_uncertainty_mapping/model_training_overlap_audit.csv`
- `artifacts/phase13_uncertainty_mapping/model_training_overlap_summary.json`

---

## 1. Executive Summary & Forensic Findings

This forensic audit investigates whether the events assigned to Phase-13 temporal regimes were used during the training, gradient backpropagation, or validation model selection of the Phase-11 Bayesian-LoRA models.

### Key Audit Findings
1. **Exact Training Universe Recovered:**  
   The Phase-11 replicated models were trained on pre-2022 records from `reconstructed_returns.csv` under a strict cutoff date of **2021-12-31 23:59:59** ($N=15,752$ total pretest events).
   - **Gradient Training Slice:** Earliest 85% ($N=13,389$ events, 2005-10-13 to 2020-02-13), used directly for parameter optimization via gradient descent.
   - **Model-Selection Validation Slice:** Final 15% ($N=2,363$ events, 2020-02-13 to 2021-12-22), used across epochs for checkpoint selection by validation AUROC.
2. **Definite Overlap in 2018–2021 ($N = 4,592$ events, 100% overlap):**  
   Every event in `2018-2019`, `2020`, and `2021` was an active constituent of either the gradient training slice or the model-selection validation slice.
3. **Definite Non-Overlap in 2022–2025 ($N = 4,100$ events, 0% overlap):**  
   All events in `2022`, `2023`, and `2024-2025` postdate the training cutoff of `2021-12-31`. Mathematically and causally, zero of these 4,100 events were seen by the model during training or validation.

---

## 2. Regime Overlap Breakdown & Classification

Matching was conducted using hierarchical keys:
1. Exact document ID (`episode_id`)
2. Exact synthetic text hash (`compute_sha256_text`)
3. Ticker + event date (`symbol` + `date`)
4. Normalized company name + event date

| Regime | Date Range | Total Events | Definite Overlap | Definite Non-Overlap | Unknown | Regime Classification | Evaluation Eligibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`2018-2019`** | 2018-01-04 to 2019-12-20 | 2,203 | 2,203 (100%) | 0 (0%) | 0 | **A. MODEL-TRAINING-OVERLAP** | Development Only (Parameter Selection) |
| **`2020`** | 2020-01-03 to 2020-12-22 | 1,198 | 1,198 (100%) | 0 (0%) | 0 | **A. MODEL-TRAINING-OVERLAP** | Secondary Diagnostic (In-Distribution) |
| **`2021`** | 2021-01-07 to 2021-12-22 | 1,191 | 1,191 (100%) | 0 (0%) | 0 | **A. MODEL-TRAINING-OVERLAP** | Secondary Diagnostic (In-Distribution) |
| **`2022`** | 2022-01-06 to 2022-12-22 | 1,191 | 0 (0%) | 1,191 (100%) | 0 | **B. TEMPORALLY POST-TRAINING** | **PRIMARY CLEAN EVALUATION (Clean OOS)** |
| **`2023`** | 2023-01-05 to 2023-12-21 | 1,181 | 0 (0%) | 1,181 (100%) | 0 | **B. TEMPORALLY POST-TRAINING** | **PRIMARY CLEAN EVALUATION (Clean OOS)** |
| **`2024-2025`** | 2024-01-04 to 2025-05-15 | 1,728 | 0 (0%) | 1,728 (100%) | 0 | **B. TEMPORALLY POST-TRAINING** | **PRIMARY CLEAN EVALUATION (Clean OOS)** |
| **Total Scope** | **2018-01-04 to 2025-05-15** | **8,692** | **4,592 (52.8%)**| **4,100 (47.2%)**| **0** | — | — |

---

## 3. Definition of Two Predeclared Evaluation Views

Because 2020 and 2021 events were used in model fitting and validation checkpoint selection, they **cannot be characterized as clean out-of-sample tests**. To maintain scientific integrity, Phase 13 pre-specifies two distinct evaluation views:

### View 1: Primary Clean Out-of-Sample Evaluation ($N = 4,100$ events)
- **Included Regimes:** `2022` (Bear Market), `2023` (Banking Stress / GenAI), `2024-2025` (Canonical Benchmark Holdout).
- **Temporal Span:** 2022-01-06 to 2025-05-15 (178 portfolio weeks).
- **Integrity Guarantee:** Strictly postdates the model training cutoff (`2021-12-31`). True out-of-sample generalization test with 0% data leakage.
- **Primary Scientific Adjudication:** Hypothesis tests regarding whether uncertainty mappings (M14, M15, M16) add decision value over unadjusted M0 will be adjudicated primarily on this view.

### View 2: Secondary In-Distribution Diagnostic Evaluation ($N = 2,389$ events)
- **Included Regimes:** `2020` (COVID-19 Shock), `2021` (Stimulus Boom).
- **Temporal Span:** 2020-01-03 to 2021-12-22 (103 portfolio weeks).
- **Integrity Classification:** In-distribution / training-overlap diagnostic.
- **Scientific Utility:** Serves as a diagnostic control to observe how uncertainty mappings function on macro regimes seen during model validation, but explicitly decoupled from clean out-of-sample claims.

---

## 4. M15 Language Correction & Adjudication Clarification

In the Phase 13B development selection analysis, all tested candidates for **M15 (Selective Abstention)** exhibited **negative mean $\Delta U$** relative to unadjusted baseline M0:
- $f = 0.10$: $\overline{\Delta U} = -0.000083$
- $f = 0.20$: $\overline{\Delta U} = -0.000050$
- $f = 0.30$: $\overline{\Delta U} = -0.000005$

### Mandatory Clarifications:
1. **M15 Failed the Positive Development Criterion:**  
   Unlike M14 ($\lambda = 0.25, \overline{\Delta U} = +0.000021$) and M16 ($\lambda_u = 0.10, \overline{\Delta U} = +0.000136$), M15 failed to achieve positive incremental utility on the development period.
2. **Role of $f = 0.30$:**  
   $f = 0.30$ was frozen strictly as the **representative baseline comparator** for selective abstention because it had the least negative mean utility and the best worst-seed utility ($-0.000033$) among abstention candidates.
3. **Scientific Context:**  
   Discrete truncation (zeroing alpha for top-uncertainty names) mechanically discards valuable high-conviction moves. The development evidence demonstrates that continuous shrinkage (M14) and convex quadratic risk penalization (M16) strongly dominate hard abstention.

---

## 5. Confirmation of Frozen Parameters

All portfolio mapping parameters remain **strictly unchanged** from commit `8ed3430ea559e91f83e71f1da27b590a01aa5ff2`:
- **M14 (Asset-Level Signal Shrinkage):** $\lambda = 0.25$
- **M15 (Selective Abstention Comparator):** $f = 0.30$ (failed positive development criterion; frozen as comparator)
- **M16 (Asset-Level Risk Inflation):** $\lambda_u = 0.10$
- **M13 (Global Macro Timing Comparator):** $\tau_{\text{enter}} = 1.2, \tau_{\text{exit}} = 0.2, s \in [0.45, 0.85]$
- **Portfolio Backbone:** $H = 4$ weeks, equal cohort capital ($1/K$), frozen-entry Markowitz QP, $10\text{ bps}$ transaction cost, $\gamma = 1.0$, zero cash return.
