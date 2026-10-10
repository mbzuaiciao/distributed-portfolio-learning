# Pre-Training Scientific Audit: Multi-Task LoRA Transferability Benchmark

**Date:** 2026-10-10  
**Repository:** `distributed-portfolio-learning`  
**Branch:** `experiment/collaborative-lora-transferability`  
**Base Commit:** `de85a8c4`  
**Execution Environment:** macOS Darwin, CPU only, strictly capped at 2 PyTorch threads (`torch.set_num_threads(2)`).  
**Protocol References:**
- Prospective Amendment: [`docs/collaborative_lora_transfer_amendment.md`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/docs/collaborative_lora_transfer_amendment.md)
- Prospective Research Design: [`docs/collaborative_lora_transfer_experiment_design.md`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/docs/collaborative_lora_transfer_experiment_design.md)
- Data Artifacts:
  - Bounded Benchmark: [`artifacts/multitask_transfer_benchmark/bounded_benchmark_transfer.json`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/artifacts/multitask_transfer_benchmark/bounded_benchmark_transfer.json)
  - Generation Summary: [`artifacts/multitask_transfer_benchmark/generation_summary.json`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/artifacts/multitask_transfer_benchmark/generation_summary.json)

---

## 1. Executive Summary & Audit Purpose

This audit serves as a pre-training gate before any multi-seed model training or confirmatory held-out evaluation is conducted. Its purpose is to verify:
1. **DGP Contrast:** That the Positive Control DGP possesses mathematically verified shared latent factor structure ($> 99.7\%$ shared variance) while the Negative Control DGP possesses independent task vectors ($\approx 6.25\%$ expected random overlap).
2. **Information Boundaries & Split Disjointness:** That target train ($N=64$), target calibration ($N=128$), target development ($N=256$), and target test ($N=512$) splits are completely disjoint with zero data leakage.
3. **Training Budget Parity:** That all models operate under matched gradient steps (500 steps) with balanced multi-task sampling preventing target-task starvation.
4. **Subspace Recovery Diagnostic Protocol:** That geometric diagnostics are frozen to isolate whether collaborative LoRA actually recovers the latent factor subspace.
5. **Execution Guardrails:** That runtime and memory have been empirically bounded on CPU prior to main training.

---

## 2. Information Boundaries & Data Split Verification

The target task is partitioned into four distinct splits generated independently from source tasks:

```
+----------------------------------------------------------------------------------------------------+
|                                      DATA SPLIT ARCHITECTURE                                       |
+------------------------------+--------------------+---------------+--------------------------------+
| Split Name                   | Tasks Included     | Sample Size   | Permitted Usage                |
+------------------------------+--------------------+---------------+--------------------------------+
| 1. Source Train              | Source 1, 2, 3     | 512 x 3 = 1536| Specialist & multi-task train  |
+------------------------------+--------------------+---------------+--------------------------------+
| 2. Target Train              | Target Task        | N = 64        | Target task local adaptation   |
+------------------------------+--------------------+---------------+--------------------------------+
| 3. Target Calibration        | Target Task        | N = 128       | Fitting M_fuse log-odds weights|
+------------------------------+--------------------+---------------+--------------------------------+
| 4. Target Development        | Target Task        | N = 256       | Dev evaluation & model tuning  |
+------------------------------+--------------------+---------------+--------------------------------+
| 5. Target Test (Quarantined) | Target Task        | N = 512       | Final confirmatory evaluation  |
+------------------------------+--------------------+---------------+--------------------------------+
```

**Audit Verification:**
- Source and Target splits share zero observation overlap.
- Target Calibration is strictly reserved for the 4-parameter logistic regression in $M_{\text{fuse}}$.
- Target Test labels are saved in `dataset_transfer_*.npz` under strict programmatic quarantine and have **not been evaluated**.

---

## 3. Evaluated Model Architecture & Trainable Parameters

| Method ID | Model Name | Architecture Details | Parameters | Role in Core Contrast |
| :--- | :--- | :--- | :---: | :--- |
| **$M_0$** | **Target-Only Logistic** | $z = w^T x + b$ ($d=32$) | **33** | Single-task linear baseline on scarce data ($N=64$) |
| **$M_{\text{LoRA0}}$** | **Target-Only LoRA** | $z = x^T (B A) + b$ ($r=2$) | **67** | Single-task low-rank adaptation baseline |
| **$M_1$** | **Pooled / Global LoRA** | $z = x^T (B_{\text{glob}} A_{\text{glob}}) + b$ | **67** | Unstructured data pooling across all 1600 samples |
| **$M_{\text{fuse}}$** | **3-Source Calibrated Fusion** | 3 independent source LoRAs + calibrated log-odds | **205** | Prediction-level modular evidence accumulation |
| **$M_{\text{subspace}}$** | **Shared-Subspace Multi-Task** | Shared basis $U \in \mathbb{R}^{32 \times 2}$ + task $v_k \in \mathbb{R}^2$ | **76** | Low-rank multi-task linear benchmark (non-LoRA) |
| **$M_{\text{CoLoRA}}$** | **Structural CoLoRA** | Shared $B \in \mathbb{R}^{32 \times 2}, A \in \mathbb{R}^{2 \times 1}$, task $\Lambda_k$ | **86** | Structural parameter-level hypergraph sharing |
| **$M_{\text{null}}$** | **Prevalence Null** | Constant empirical target prevalence | **0** | Baseline for uninformative predictions |
| **$M_{\text{oracle}}$** | **Bayes Optimal Oracle** | Exact analytical $\beta^*_{\text{target}}$ | **0** | Theoretical upper bound on performance |

### Representation Sharing vs. Prediction Fusion Contrast
- **$M_{\text{fuse}}$ vs. $M_0$:** Measures benefit of independent source models combined purely at the prediction logit scale via target calibration.
- **$M_{\text{subspace}}$ vs. $M_0$:** Measures benefit of generic shared-subspace multi-task learning.
- **$M_{\text{CoLoRA}}$ vs. $M_{\text{subspace}}$:** Isolates whether CoLoRA's specific parameter factorization ($B \Lambda_k A$) provides any architectural advantage over direct low-rank factor projection ($U v_k$).

---

## 4. Balanced Multi-Task Optimization Protocol

To prevent the target task ($N=64$) from being drowned out by the source tasks ($N=1536$), the training loop uses a **Balanced Multi-Task Sampler**:
- At each gradient step, batches of size $B = 16$ are sampled independently from each of the 4 tasks.
- Target training examples are sampled with replacement, ensuring **exactly 25% of the total gradient exposure** is dedicated to target loss.
- All models train for exactly **500 gradient steps** using AdamW ($\text{lr} = 0.01$, $\text{weight\_decay} = 0.01$).

---

## 5. Subspace Recovery Diagnostics

To prevent attributing performance gains to spurious artifacts, three geometric diagnostics are prospectively frozen:
1. **Grassmannian Projection Alignment ($S_{\text{proj}} \in [0, 1]$):**
   $$S_{\text{proj}}(\hat{U}, U_{\text{true}}) = \frac{1}{r} \|\hat{Q}^T U_{\text{true}}\|_F^2$$
   Where $\hat{Q}$ is the orthonormalized basis of the learned adapter $B$. Perfect recovery yields $1.0$.
2. **Principal Angle Cosines:** $\cos(\theta_1), \cos(\theta_2) = \text{svd}(\hat{Q}^T U_{\text{true}})$.
3. **Target Parameter Cosine Similarity:**
   $$\cos(\hat{\beta}_{\text{target}}, \beta^*_{\text{target}}) = \frac{\hat{\beta}_{\text{target}}^T \beta^*_{\text{target}}}{\|\hat{\beta}_{\text{target}}\|_2 \|\beta^*_{\text{target}}\|_2}$$

---

## 6. Predefined Confirmatory Decision Rules

On the quarantined held-out target test split ($N=512$, seed=20261024):
1. **Positive Transfer Confirmation:** CoLoRA ($M_{\text{CoLoRA}}$) must achieve Brier gain $\ge +0.010$ over Target-Only ($M_0$) across all 3 seeds ($42, 123, 777$) with a 95% paired bootstrap CI strictly $> 0$.
2. **Negative Transfer Protection:** CoLoRA must beat Pooled LoRA ($M_1$) by $\ge +0.005$ Brier, proving that task-specific routing matrices $\Lambda_k$ prevent cross-task interference.
3. **Representation vs. Prediction Winner:**
   - If $M_{\text{CoLoRA}}$ outperforms $M_{\text{fuse}}$ ($p < 0.05$): Confirms structural representation sharing.
   - If $M_{\text{fuse}}$ matches or outperforms $M_{\text{CoLoRA}}$: Concludes that prediction-level fusion is the parsimonious mechanism.
4. **Negative Control Rejection Gate:** Neither CoLoRA nor Pooled LoRA may exhibit statistically significant positive transfer on the Negative Control regime.

---

## 7. Compute Safeguards & Bounded Benchmark Audit

A bounded 1-seed benchmark was executed on CPU with PyTorch threads capped at two:

| Component | Measured Latency | Measured Peak Memory | Guardrail Ceiling | Audit Status |
| :--- | :---: | :---: | :---: | :---: |
| **Target-Only Logistic ($M_0$)** | 0.908 s | — | — | **Pass** |
| **Target-Only LoRA ($M_{\text{LoRA0}}$)** | 0.063 s | — | — | **Pass** |
| **Pooled LoRA ($M_1$)** | 0.064 s | — | — | **Pass** |
| **Independent Specialists + Fusion ($M_{\text{fuse}}$)** | 0.191 s | — | — | **Pass** |
| **Shared-Subspace Multi-Task ($M_{\text{subspace}}$)** | 0.156 s | — | — | **Pass** |
| **Structural CoLoRA ($M_{\text{CoLoRA}}$)** | 0.185 s | — | — | **Pass** |
| **Total Single Replication** | **1.57 s** | **328.3 MB** | — | **Pass** |
| **Projected 6-Run Full Experiment** | **0.16 min** (9.4 s) | **328.3 MB** | Max 20 min / 4000 MB | **Pass** |

---

## 8. Confinement & Execution Status

- **Datasets Generated:** `dataset_transfer_positive_control.npz`, `dataset_transfer_negative_control.npz`.
- **Unit Tests:** 5/5 multi-task transfer unit tests passing (48/48 across full repository).
- **Multi-Seed Training:** **NOT EXECUTED** (Locked pending user authorization).
- **Held-Out Test Split:** **STRICTLY QUARANTINED & UNTOUCHED**.
- **Pretrained Models / GPUs:** **NONE USED**.
