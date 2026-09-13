# Reproducibility & Provenance Guide

This document provides complete instructions, environment specifications, cryptographic hashes, and git commit references required to independently reproduce every finding in the paper.

---

## 1. Code Repository & Authoritative Git Commits

- **Repository**: `https://github.com/mbzuaiciao/distributed-portfolio-learning.git`
- **Key Git Commits**:
  - `bd84126`: *"Preregister clean Bayesian LoRA comparison"* (Frozen preregistration, data hashes, and equivalence margins pre-committed *before* clean OOS evaluation).
  - `b0d67af`: *"Retrain Bayesian LoRA on clean inputs"* (Complete clean retraining, inference, metrics, bootstrap intervals, and report).
  - `1b33ce6`: *"Audit reliability metric mechanics, input leakage, and regularized LoRA"* (Phase 16.1 audit proving input leakage and within-class reliability artifact).

---

## 2. Environment & Hardware Specifications

- **Operating System**: macOS (Darwin 25.3.0, Apple Silicon)
- **Python Version**: Python 3.14.2
- **PyTorch Version**: 2.14.0 (MPS acceleration enabled)
- **Transformers Version**: 4.49.0
- **Hardware Profile**: Apple Silicon M-Series, MPS device acceleration
  - Deterministic training time: $383.52\text{ s}$ (~6.4 min) per seed
  - Bayesian training time: $402.42\text{ s}$ (~6.7 min) per seed
  - Deterministic inference: $8.26\text{ ms}$ per event
  - Bayesian Posterior Mean inference: $7.89\text{ ms}$ per event
  - Bayesian MC32 inference: $241.02\text{ ms}$ per event

---

## 3. Pre-Trained Model & Adapter Specifications

- **Base Model**: `ProsusAI/finbert`
- **HuggingFace Git Revision**: `4556d13015211d73dccd3fdd39d39232506f3e43`
- **Architecture**: `BertForSequenceClassification` (`num_labels=2`, `ignore_mismatched_sizes=True`)
- **LoRA Configuration**:
  - Injected Modules: `attention.self.query`, `attention.self.value`
  - Rank ($r$): 16
  - Scaling Alpha ($\alpha$): 32.0 (scaling factor $\frac{\alpha}{r} = 2.0$)
  - Dropout: 0.05
  - Trainable Parameters: 591,362 ($0.54\%$ of backbone)
- **Deterministic Optimizer Recipe**:
  - Optimizer: AdamW
  - Learning Rate: $2.0 \times 10^{-4}$
  - Weight Decay: $0.0$
  - Batch Size: 32
  - Epochs: 2
- **Bayesian Optimizer Recipe**:
  - Parameterization: Independent Gaussian variational posteriors $q(\theta) = \mathcal{N}(\mu, \text{softplus}(\rho)^2)$
  - Prior Distribution: $\mathcal{N}(0, \sigma_0^2 \mathbf{I})$ with $\sigma_0 = 1.0$
  - Variational Initialization: $\mu_A \sim \text{Kaiming}$, $\mu_B = 0$, $\sigma_{\text{init}} = 0.01$
  - Scaled KL Loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \frac{\beta}{N_{\text{train}}} D_{\text{KL}}(q(\theta) \,\|\, p(\theta))$ with $\beta = 0.003, N_{\text{train}} = 11,985$
  - Optimizer: AdamW ($\text{lr} = 2.0 \times 10^{-4}, \text{weight\_decay} = 0.01$, 2 epochs, batch size 32)

---

## 4. Canonical Data Splits & Cryptographic Hashes

- **Data Source**: `data/bayesian_lora/reconstructed_returns.csv` (18,201 total earnings events)
- **Label Cutoff**: 20th percentile of pre-2022 1-day post-earnings returns ($y_i = \mathbb{I}(\text{move\_1d}_i \le -4.692\%)$)
- **Clean Input Text Template**:
  ```python
  f"{company} ({ticker}) quarterly earnings announcement on {date_str}."
  ```
- **Strict Leakage Invariant**:
  ```python
  assert not any(tok in text for tok in ["move_1d", "move_30d", "1-day move was", "30-day move was", "-4.69", "downside"])
  ```
- **Universe Hashes**:
  - Training Universe ($N=11,985$, pre-2022): `5a6e316546c59eeb98b42df3c8a4459f8bfb291e9216d2d8109355f45c4c601b`
  - Validation Universe ($N=2,116$, pre-2022): `9c998f01ca784b7200103e698ca316b8f2678ad167dc1643d88736590cae7a2e`
  - Clean OOS Universe ($N=4,100$ unique events, 2022–2025): `cf89dbe5785c8cbd5ccae1e2002eaa4626242fdaa8f3ef68af54e5bd44cd7c8d`
  - Clean Text Manifest Hash (`clean_text_manifest.csv`): `4560ff86b543260822a7d8dd266bd40c84b0f1d5dde0fff5ecdc3d588c99beb6`

---

## 5. Statistical Protocols & Equivalence Bounds

- **Seeds Evaluated**: 42, 123, 777 ($N=4,100 \times 3 = 12,300$ evaluations).
- **Cluster Resampling**: Calendar-week clustered block bootstrap ($B=1,000$ iterations across 168 unique calendar weeks). All observations occurring in the same calendar week are resampled together to preserve cross-asset return dependencies.
- **Predeclared Equivalence Margins**:
  - $\Delta \text{AUROC}$: $\pm 0.005$
  - $\Delta \text{Brier Score}$: $\pm 0.002$
  - $\Delta \text{Log Loss}$: $\pm 0.010$
  - $\Delta \text{ECE}$: $\pm 0.010$

---

## 6. Execution Commands

To reproduce the complete pipeline and run the full test suite from the repository root:

```bash
# 1. Activate environment
source /Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/.venv/bin/activate

# 2. Run the clean training, evaluation, and bootstrap analysis
python experiments/run_phase17_clean_bayesian.py

# 3. Run full automated test suite (251 passing tests)
pytest tests/ -v

# 4. Run static linter and formatting check
ruff check .
```
