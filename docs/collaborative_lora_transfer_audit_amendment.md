# Prospective Design Amendment: Targeted Pre-Run Scientific Audit & Refinements

**Document Version:** 1.0  
**Date:** 2026-10-10  
**Status:** **FROZEN AUDIT AMENDMENT (PRE-EXECUTION)**  
**Repository:** `distributed-portfolio-learning`  
**Branch:** `experiment/collaborative-lora-transferability`  
**Base Commit:** `91a29834`  
**Preceding Milestones:**
- Prospective Design Amendment: [`docs/collaborative_lora_transfer_amendment.md`](collaborative_lora_transfer_amendment.md)
- Pre-Training Audit: [`artifacts/multitask_transfer_benchmark/pre_training_audit_transfer.md`](../artifacts/multitask_transfer_benchmark/pre_training_audit_transfer.md)

---

## 1. Scientific Audit Findings & Clarifications

### 1.1 Representational Capacity of Scalar-Output Rank-2 LoRA
- **Mathematical Fact:** In a scalar-output linear model ($d_{\text{out}} = 1$), the effective weight vector $w = B A$ has dimension $d \times 1$. The algebraic rank of any vector is at most $1$.
- **Consequence:** For any rank $r \ge 1$ (here $r=2$), the rank-2 LoRA decomposition $B A \in \mathbb{R}^{d \times 1}$ can represent **arbitrary** linear coefficient vectors in $\mathbb{R}^d$. Specifically, setting the first column of $B$ to $w$ and $A = [1, 0]^T$ yields $B A = w$.
- **Scientific Limitation:** Single-task scalar LoRA ($M_{\text{LoRA0}}$) has the **identical expressive capacity** to full-rank Target-Only Logistic ($M_0$). Low-rank structure acts as a true rank constraint **only** across tasks (constraining the multi-task coefficient matrix $W = [\beta_1, \dots, \beta_K] \in \mathbb{R}^{d \times K}$ to rank $\le r$) or in multi-output layers ($d_{\text{out}} > r$).
- **Interpretation Rule:** Differences between $M_0$ and $M_{\text{LoRA0}}$ reflect gradient optimization dynamics (bi-linear factor descent vs. direct vector descent), not subspace capacity restrictions. Findings from scalar linear LoRA must not be overgeneralized to multi-layer matrix LoRA in large transformers without qualification.

### 1.2 Representational Equivalence of CoLoRA and Shared-Subspace Multi-Task
- **Mathematical Equivalence:**
  - `SharedSubspaceMultiTask`: $w_k = U v_k$, with $U \in \mathbb{R}^{d \times r}$ and $v_k \in \mathbb{R}^r$.
  - `StructuralCoLoRA`: $w_k = B \Lambda_k A$, with $B \in \mathbb{R}^{d \times r}$, $\Lambda_k \in \mathbb{R}^{r \times r}$, and $A \in \mathbb{R}^{r \times 1}$.
  - Because $u_k = \Lambda_k A \in \mathbb{R}^r$ can assume any vector in $\mathbb{R}^r$ (for $A \ne 0$), the span of reachable task weights is identical:
    $$\{w_k \mid w_k = B \Lambda_k A\} = \{w_k \mid w_k = U v_k\} = \text{span}(B)$$
- **Consequence:** Structural CoLoRA and the Shared-Subspace baseline possess **identical representational power**. Any performance discrepancy between them is an optimization artifact of tri-linear parameterization ($B \Lambda_k A$) versus bi-linear parameterization ($U v_k$).

### 1.3 Audit of Trainable Parameter Counts
Verified via actual PyTorch parameter tensors (`sum(p.numel() for p in model.parameters() if p.requires_grad)`):
- Target-Only Logistic ($M_0$): **33** parameters ($32 \times 1 + 1$).
- Target-Only LoRA ($M_{\text{LoRA0}}$): **67** parameters ($32 \times 2 + 2 \times 1 + 1$).
- Pooled Global LoRA ($M_1$): **67** parameters ($32 \times 2 + 2 \times 1 + 1$).
- Shared-Subspace Multi-Task ($M_{\text{subspace}}$): **76** parameters ($32 \times 2 + 4 \times 2 + 4$).
- Structural CoLoRA ($M_{\text{CoLoRA}}$): **86** parameters ($32 \times 2 + 2 \times 1 + 4 \times 4 + 4$).
- Independent Specialists + 3-Source Fusion ($M_{\text{fuse}}$): **205** parameters ($3 \times 67$ source adapters + $4$ logistic fusion parameters).

### 1.4 Target Data Budget & Calibration-Matched Baseline ($M_{0,\text{cal-matched}}$)
- **Confound Identification:** $M_{\text{fuse}}$ uses 3 source models plus $N = 128$ labeled target calibration examples to fit the fusion weights. In contrast, $M_0$ only sees $N = 64$ target training examples. If $M_{\text{fuse}}$ outperforms $M_0$, the gain could be driven by the 128 additional target labels rather than transfer from source models.
- **Remediation:** We introduce an explicit, fair target-only baseline:
  - **$M_{0,\text{cal-matched}}$ (Target-Only Equal-Label-Budget):** Full-rank $L_2$-regularized Logistic Regression trained on the union of Target Train ($N=64$) and Target Calibration ($N=128$), giving an identical total budget of **$N = 192$ target labels**.
  - A transfer method can claim true collaborative benefit over single-task learning only if it outperforms $M_{0,\text{cal-matched}}$.

### 1.5 Verification of Gradient Update Steps & Bounded Runtimes
- All models execute exactly **500 gradient steps** with AdamW ($\text{lr} = 0.01$, $\text{weight\_decay} = 0.01$).
- Initial vs. final loss checks confirm active, rapid convergence (BCE loss drops from $\approx 0.82$ to $\approx 0.17$ within 500 steps).
- The short execution time ($\approx 1.57\text{ s}$ per replication) is driven by compact tensor dimensions ($32 \times 16$) running in hardware-accelerated BLAS on Apple Silicon CPU.

### 1.6 Probabilistic Independence of Negative Control Task Coefficients
- The Negative Control generates task vectors independently from $\mathcal{N}(0, I_{32})$.
- Expected subspace overlap is $\mathbb{E}[\|\Pi_U \beta_j\|^2 / \|\beta_j\|^2] = r/d = 6.25\%$.
- However, random fluctuations in finite sample regimes ($N=64$) can occasionally produce empirical alignments up to $15\%$. Zero transfer is a **probabilistic expectation**, not an absolute guarantee.

### 1.7 Frozen Gates, Test Isolation, Seeds, and Decision Rules
- **Seeds:** 42, 123, 777.
- **Hardware Safeguard:** CPU only, strictly capped at 2 PyTorch threads (`torch.set_num_threads(2)`).
- **Test Isolation:** Target Held-Out Test ($N=512$) remains strictly quarantined. Stage 4 (`evaluate_test`) will **not** be executed.

---
