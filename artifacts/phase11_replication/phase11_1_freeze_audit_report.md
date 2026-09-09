# Phase 11.1: Pre-Outcome Replication Freeze Audit Report

**Date:** 2026-09-09  
**Status:** PREREGISTERED FREEZE FINALIZED & VERIFIED  
**Integrity Invariant:** Realized 2026 market returns have **NOT** been acquired, inspected, joined, or summarized. Prospective outcomes remain completely unseen.

---

## 1. Monte Carlo Draw Count Audit

- **Preregistration Specification:** `MC draws = 32` (`replication_preregistration.json:35`).
- **Code Audit:** In `experiments/run_phase11_replication.py:430`, inference was explicitly configured with `"mc_samples": 32`.
- **Artifact Audit:** In `artifacts/phase11_replication/frozen_2026_predictions.csv`, the `mc_samples` column records `32` for every row across all seeds:
  ```csv
  event_id,ticker,event_timestamp,seed,point_prediction,epistemic_uncertainty,mc_samples,document_hash
  AAPL_2026_Q3,AAPL,2026-07-31 16:30:00,42,0.12668188974275013,4.07482429854856e-06,32,c49a7522...
  ```
- **Historical Benchmark:** Predictions in `data/bayesian_lora/holdout/beta_0p003_seed_<seed>_mc32_predictions.csv` also use 32 draws.
- **Discrepancy Resolution:** The phrase `"T = 30"` in the Phase 11 chat response summary was purely a **reporting typo** (transcription error in the markdown response). Both the generation script and the committed artifact executed with exactly `T = 32`.
- **Action Taken:** Under the preregistered audit protocol ("If actual prospective predictions used 32: correct documentation only, preserve existing prediction SHA"), the existing canonical prediction file and its SHA256 checksum are preserved:
  - **Canonical File:** `artifacts/phase11_replication/frozen_2026_predictions.csv`
  - **SHA256 Checksum:** `6542b33fa5810985beff8c835e71b598ffaf47bf943e88d43f7c50d7dcb30aeb`

---

## 2. Frozen Base Model Provenance & Reconstruction Recipe

Each frozen checkpoint (`bayesian_lora_seed_<seed>_state.pt`, ~4.76 MB) was inspected directly:

### Exact State Dict Contents (98 Tensors)
1. **Bayesian LoRA Variational Parameters (96 tensors):**
   - 12 BERT attention layers $\times$ 2 target modules (`attention.self.query`, `attention.self.value`) $\times$ 2 low-rank matrices ($A \in \mathbb{R}^{16 \times 768}$, $B \in \mathbb{R}^{768 \times 16}$) $\times$ 2 variational tensors ($\mu, \rho$) $= 96$ tensors.
   - Total variational parameters: $1{,}179{,}648$ floats.
2. **Classifier Head (2 tensors):**
   - `classifier.weight`: $\mathbb{R}^{2 \times 768}$ ($1{,}536$ floats).
   - `classifier.bias`: $\mathbb{R}^{2}$ ($2$ floats).
3. **Base FinBERT Weights:** **Not included** in the 4.76 MB state dict (base weights remain frozen in external registry).
4. **Tokenizer Assets:** **Not included** in the state dict (external).

### External Base Model Exact Specifications
- **Model Identifier:** `ProsusAI/finbert`
- **Hugging Face Commit SHA:** `4556d13015211d73dccd3fdd39d39232506f3e43`
- **Config SHA256:** `dc41f453cd64ac234c321bfbc43acb6b3c7e8fb31b48356a0496ebdadb0d36e8`
- **Tokenizer Vocab SHA256:** `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`
- **Environment:** PyTorch `2.14.0`, Transformers `5.16.1`.

### Exact Checkpoint Reconstruction Recipe (Zero Retraining)
```python
import torch
from transformers import AutoModelForSequenceClassification
from bayesian_lora_portfolio_risk.uncertainty.bayesian_lora import inject_bert_attention_bayesian_lora

# 1. Instantiate frozen base model at exact git revision
model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert",
    revision="4556d13015211d73dccd3fdd39d39232506f3e43",
    num_labels=2,
    ignore_mismatched_sizes=True,
)

# 2. Inject variational Bayesian LoRA modules
inject_bert_attention_bayesian_lora(
    model, rank=16, alpha=32, dropout=0.05, init_std=0.01, prior_std=1.0
)

# 3. Load frozen adapter state dict
state = torch.load("artifacts/phase11_replication/frozen_models/bayesian_lora_seed_42_state.pt", weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()
```

---

## 3. Round-Trip Checkpoint Test

A fresh independent process was spawned to load each seed's frozen state dict onto the base FinBERT revision and score identical input texts. The outputs were compared against immediate post-training outputs under identical random seed initialization:

| Seed | Max $|\Delta \bar{p}|$ (Posterior Mean) | Max $|\Delta \sigma_{\text{epi}}^2|$ (Epistemic Var) | Max $|\Delta \sigma_{\text{ale}}^2|$ (Aleatoric Var) | Status |
| :---: | :---: | :---: | :---: | :---: |
| **42** | `0.00e+00` | `0.00e+00` | `0.00e+00` | **PERFECT MATCH** |
| **123** | `0.00e+00` | `0.00e+00` | `0.00e+00` | **PERFECT MATCH** |
| **777** | `0.00e+00` | `0.00e+00` | `0.00e+00` | **PERFECT MATCH** |

All three seeds demonstrate exact machine-precision numerical reproducibility ($0.0$ difference), well within the $10^{-6}$ tolerance.

---

## 4. Historical Replication Sample Audit

### Clarification of Terminology
In the Phase 11 preliminary summary, the phrase *"on the training distribution"* was used loosely. Forensic audit confirms that the replication metrics were evaluated **strictly out-of-sample** on the fixed 2024–2025 historical benchmark:

| Dataset Segment | Source File | Date Range | Sample Count $N$ | Used in Training? |
| :--- | :--- | :--- | :---: | :---: |
| **Training Slice** | `data/bayesian_lora/reconstructed_returns.csv` (pre-2022) | `2005-10-13` to `2021-12-22` | $15{,}752$ | **YES** |
| **Historical Benchmark** | `data/bayesian_lora/holdout/beta_0p003_seed_<seed>_mc32_predictions.csv` | `2024-01-04` to `2025-05-15` | $1{,}728$ | **NO (0% overlap)** |

The 2024–2025 holdout benchmark postdates the training slice by more than two years. Zero samples from the holdout benchmark were seen during model fitting, ensuring genuine out-of-sample replication evidence. No Phase-11 Bayesian-LoRA model-training hyperparameters were tuned using the fixed 2024-2025 benchmark. However, the 2024-2025 period was previously used extensively in downstream portfolio research and is not an untouched investment holdout.

---

## 5. M13 Delta-U Reconciliation

### Discrepancy & Granularity Distinction
- **Event-Level Diagnostic (+0.002519):**
  - **Classification:** `EVENT-LEVEL DIAGNOSTIC — NOT CANONICAL PORTFOLIO ENDPOINT`.
  - Must **not** be used in headline portfolio performance tables.
  - Computed as an unpooled heuristic mean across $N = 1{,}728$ individual earnings releases:
    $$u_t = s_t r_{1d, t} - \frac{1}{2}\gamma s_t^2 \frac{\sigma_0^2}{52} - c \cdot \text{TO}_{\text{event}}$$
    with $\gamma = 2.5$ and static annualized volatility proxy $\sigma_0 = 0.20$.
  - Yielded: Seed 42 ($+0.002602$), Seed 123 ($+0.002965$), Seed 777 ($+0.001989$), Mean ($+0.002519$).
- **Canonical Weekly Portfolio Endpoint (+0.000530):**
  - **Classification:** `CANONICAL HISTORICAL PORTFOLIO REPLICATION ENDPOINT`.
  - Evaluated on weekly mark-to-market returns across $T = 69$ calendar weeks under multi-week $H=4$ cohort stacking, equal capital weighting ($1/H$), Markowitz intra-cohort allocation, $\gamma = 1.0$, and turnover netting.
  - Both historical lost-model predictions and newly replicated models produce identical canonical weekly portfolio endpoints:
    - **Seed 42:** $M_8 = 0.003431, M_{13} = 0.004256, \mathbf{\Delta U = +0.000825}$
    - **Seed 123:** $M_8 = 0.005156, M_{13} = 0.005506, \mathbf{\Delta U = +0.000350}$
    - **Seed 777:** $M_8 = 0.004011, M_{13} = 0.004426, \mathbf{\Delta U = +0.000415}$
    - **3-Seed Arithmetic Mean:** $M_8 = 0.004199, M_{13} = 0.004729, \mathbf{\Delta U = +0.000530}$

### Provenance of "+0.000652"
- The value `+0.000652` originated strictly in Phase 8.2 / Phase 9 (`artifacts/phase9_audit/metrics_summary.json:103`) as **Seed 42 alone under the one-week ($H=1$) architecture** without multi-cohort stacking ($M_{13}\ -0.001048$ vs $M_8\ -0.001699$).
- It was neither a 3-seed mean nor an $H=4$ portfolio endpoint.
- Under the frozen primary $H=4$ architecture, the canonical 3-seed mean historical Delta U is **$+0.000530$** (Seed 42: $+0.000825$).


---

## 6. Prospective Freeze Manifest Checksums

The complete prospective freeze manifest is recorded in:  
`artifacts/phase11_replication/prospective_freeze_manifest.json`

| Asset | File Path | SHA256 Checksum |
| :--- | :--- | :--- |
| **Canonical Predictions (T=32)** | `artifacts/phase11_replication/frozen_2026_predictions.csv` | `6542b33fa5810985beff8c835e71b598ffaf47bf943e88d43f7c50d7dcb30aeb` |
| **2026 Input Manifest** | `artifacts/phase11_replication/2026_input_manifest.csv` | `cca8e8f6af30fa8225476a77a5802403979541d0701c2faf376814033a1c9314` |
| **Frozen Checkpoint Seed 42** | `artifacts/phase11_replication/frozen_models/bayesian_lora_seed_42_state.pt` | `d382640351dbef4fd40b47c8b033b1a4823c57d8fd6740b6fd9d1709043dc309` |
| **Frozen Checkpoint Seed 123** | `artifacts/phase11_replication/frozen_models/bayesian_lora_seed_123_state.pt` | `e094b775c7061b7c60341704525719b2c3fd4055e1aa325581eacabc23059301` |
| **Frozen Checkpoint Seed 777** | `artifacts/phase11_replication/frozen_models/bayesian_lora_seed_777_state.pt` | `0f084381c8b387f5a4946e337470c8c25707228c40cf676604d1db22f3a13e2a` |

---

## 7. Audit Sign-Off

- [x] T=32 draw count verified in code and artifacts (T=30 confirmed as reporting typo).
- [x] Canonical prospective prediction SHA preserved.
- [x] Base model provenance fully documented down to Hugging Face commit SHA and config hash.
- [x] Round-trip checkpoint test achieves machine-precision reproducibility (`0.00e+00`).
- [x] Historical benchmark confirmed to be strictly out-of-sample ($N=1,728$, 2024–2025).
- [x] Delta U difference fully reconciled between event-level and weekly portfolio-level models.
- [x] Pre-outcome prospective freeze finalized and clean.
