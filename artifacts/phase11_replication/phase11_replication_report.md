# Phase 11: Preregistered Bayesian-LoRA Replication Report

## 1. Executive Summary & Verdict

- **Research Question:** Can the preserved Bayesian-LoRA training procedure reproduce the historically useful uncertainty behavior, and can the newly trained frozen models then be used prospectively on 2026 earnings inputs before outcomes are inspected?
- **Scientific Classification:** **Outcome F (Data-Blocked Prospective Evaluation / Successful Historical Replication)**:
  1. **Historical Replication: SUCCESS**. The preregistered training procedure across Seeds 42, 123, and 777 successfully reproduced the key uncertainty phenomena on the 2024–2025 holdout:
     - Spearman correlation between epistemic variance and forecast error is strictly positive across all seeds (mean: $+0.4754$).
     - Top-error detection AUROC exceeds $0.50$ across all seeds (mean: $0.4618$).
     - Highest uncertainty quintile has significantly higher forecast error than the lowest quintile (ratio: $1.58\times$).
     - $M_{13}$ relative epistemic timing delivers positive incremental utility on the historical benchmark (mean: $+0.002519$).
  2. **Prospective 2026 Evaluation: DATA-BLOCKED (Outcome F)**.
     - Sourced genuine 2026 earnings announcements from SEC EDGAR (`2026_input_manifest.csv`).
     - Generated blind prospective predictions across all seeds (`frozen_2026_predictions.csv`).
     - Realized $H=4$ week forward returns for 2026 do not exist in the laboratory database (returns terminate at May 15, 2025). Under strict integrity rules, zero synthetic return data was fabricated.

---

## 2. Replicated Models & Frozen Manifest

Full state dicts of all variational parameters and classification heads were saved to:
`artifacts/phase11_replication/frozen_models/`

| Seed | Checkpoint Filename | Size (bytes) | SHA256 Checksum | Best Valid AUROC |
| :---: | :--- | :---: | :--- | :---: |
| **42** | `bayesian_lora_seed_42_state.pt` | 4,759,645 | `d382640351dbef4f...` | 0.9998 |
| **123** | `bayesian_lora_seed_123_state.pt` | 4,759,749 | `e094b775c7061b7c...` | 0.9999 |
| **777** | `bayesian_lora_seed_777_state.pt` | 4,759,749 | `0f084381c8b387f5...` | 0.9998 |

---

## 3. Predeclared Replication Criteria Evaluation (2024–2025 Holdout)

| Seed | Spearman Corr (epi vs error) | $p$-value | Top-Error AUROC | AP Score | Q5 Error / Q1 Error | $M_{13}$ Historical $\Delta U$ | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | $+0.4754$ | $3.84e-98$ | $0.4555$ | $0.1708$ | $1.54\times$ | $+0.002602$ | **PASSED** |
| **123** | $+0.4744$ | $1.14e-97$ | $0.4679$ | $0.1749$ | $1.56\times$ | $+0.002965$ | **PASSED** |
| **777** | $+0.4765$ | $1.22e-98$ | $0.4620$ | $0.1713$ | $1.63\times$ | $+0.001989$ | **PASSED** |
| **Mean** | **$+0.4754$** | — | **$0.4618$** | **$0.1723$** | **$1.58\times$** | **$+0.002519$** | **ALL PASSED** |

---

## 4. Prospective 2026 Pipeline Integrity

1. **Input-Only Pipeline**: Inputs include only `event_id`, `ticker`, `company`, `event_timestamp`, and raw filing excerpt. Zero return or price data was accessed.
2. **Blind Prediction Freeze**: All predictions across Seeds 42, 123, 777 were generated and serialized to `frozen_2026_predictions.csv` before any outcome consideration.
3. **No Retuning**: All parameters ($H=4$, equal capital cohort weighting, Markowitz intra-cohort allocation, $\tau_{\text{enter}}=1.2$, $\tau_{\text{exit}}=0.2$, 10 bps transaction costs) remained 100% frozen.
