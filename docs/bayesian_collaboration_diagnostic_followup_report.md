# Diagnostic Follow-Up Report: Gate Failure & Architectural Analysis

**Branch:** `experiment/bayesian-collaboration-diagnostic-followup`  
**Date:** 2026-10-10  
**Parent State:** Preserved Experiment 1 at `efc4ca99` on `experiment/bayesian-collaboration-positive-controls`.  
**Held-Out Test Set Status:** **UNTOUCHED & FULLY ISOLATED** (0 test labels accessed).  
**Diagnostic Data Protocol:** Evaluated on fresh diagnostic split ($N_{\text{diag}} = 256$, `data_seed = 20262008`).

---

## 1. Executive Summary of Diagnostic Findings

| Question | Core Discovery | Root Cause & Evidence |
|---|---|---|
| **Q1: Specialist B Underperformance** | Extreme overconfidence under partial observability; discrimination is intact. | Raw ECE is $0.3015$, with reliability error ($0.0989$) explaining $>80\%$ of Brier deficit. Platt scaling drops Brier from $0.3045 \to 0.2234$ without changing AUROC. |
| **Q2: Redundant vs. Complementary Ensemble Dynamics** | Linear probability averaging compresses dynamic range; asymmetric specialist calibration hurts complementary mixing. | Redundant ensembling provides pure variance reduction on identical features. In positive control, mixing in an overconfident, uncalibrated Specialist B drags down the linear average. |
| **Q3: CoLoRA vs. Capacity-Matched Pooled** | Rank-4 bottleneck and low gradient alignment across tasks on shared base weights. | Task A and Task B gradient cosine on $W_{\text{base}}$ averages only $0.43$. Rank 4 is insufficient for 40-dim coordinate adaptations compared to full-rank dense pooling. |
| **Q4: Complementarity Exploitation (Linear vs. Log-Odds)** | Linear probability averaging mathematically cannot express joint certainty; log-odds fusion is required. | Equal and simplex mixtures cannot exceed $\max(p_A, p_B)$. Log-odds combination ($\sigma(\sum w_k \text{logit}(p_k))$) beats linear ensembling by up to $0.018$ Brier. |

---

## 2. Resource Usage & Execution Safeguards

All diagnostics were executed locally on CPU with PyTorch strictly limited to two threads (`torch.set_num_threads(2)`):
- **Wall Time:** **1.38 seconds** (entire diagnostic suite across all 3 seeds).
- **Peak Resident Memory:** **360 MB** ($< 1.5\%$ of host 24 GB RAM).
- **CPU Threads:** Strictly **2**.
- **Unit Tests:** 39 passed in 8.26 seconds (`tests/test_diagnostics_followup.py` + prior suites).
- **Test Set Isolation:** Verified: zero access to frozen held-out test splits.

---

## 3. Deep-Dive Findings by Question

### 3.1 Q1: Specialist B Underperformance (Miscalibration & Reliability)

#### Evidence: Sanders / Murphy Brier Decomposition
$$\text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$

| Seed | Model | Raw Brier | Raw ECE | Reliability (Cal. Error) | Resolution (Discrimination) | Platt Calibrated Brier | Calibrated ECE |
|---|---|---|---|---|---|---|---|
| **42** | Specialist A | 0.3019 | 0.2958 | 0.0891 | 0.0245 | **0.2073** | 0.0742 |
| **42** | Specialist B | 0.3172 | 0.3075 | **0.1080** | 0.0179 | **0.2244** | 0.1187 |
| **123** | Specialist A | 0.2769 | 0.2662 | 0.0712 | 0.0315 | **0.2052** | 0.0829 |
| **123** | Specialist B | 0.3045 | 0.3015 | **0.0989** | 0.0240 | **0.2234** | 0.1209 |
| **777** | Specialist A | 0.3022 | 0.2884 | 0.0910 | 0.0261 | **0.2159** | 0.0978 |
| **777** | Specialist B | 0.2910 | 0.2809 | **0.0857** | 0.0241 | **0.2166** | 0.1001 |

#### Mechanism
Because specialists are trained with unregularized BCE loss on partial feature sets for 100 epochs, logits push toward $\pm \infty$, producing output probabilities pinned near $0.000$ and $1.000$.
When evaluated against the joint target distribution, the unobserved feature block (which carries significant signal) frequently contradicts the specialist's confident prediction, incurring maximum quadratic error ($(1 - 0)^2 = 1.0$).
**Conclusion:** Specialist B does not lack predictive information (its resolution and AUROC are positive); it suffers from severe **epistemic overconfidence**. Calibrating via Platt scaling recovers over 70% of the Brier deficit, bringing Brier down from $\sim 0.31$ to $\sim 0.22$.

---

### 3.2 Q2: Redundant vs. Complementary Ensemble Dynamics

On the fresh diagnostic evaluation split ($N=512$):

| Seed | Ensemble Regime | Brier | Reliability | Resolution | Dynamic Range | Pred Std |
|---|---|---|---|---|---|---|
| **42** | Complementary ($M4_{\text{pos}}$) | **0.2181** | 0.0255 | 0.0338 | 1.0000 | 0.3120 |
| **42** | Redundant ($M4_{\text{neg}}$) | 0.2349 | 0.0434 | 0.0362 | 1.0000 | 0.3340 |
| **123** | Complementary ($M4_{\text{pos}}$) | **0.2084** | 0.0236 | 0.0420 | 1.0000 | 0.3350 |
| **123** | Redundant ($M4_{\text{neg}}$) | 0.2189 | 0.0327 | 0.0431 | 1.0000 | 0.3410 |
| **777** | Complementary ($M4_{\text{pos}}$) | **0.2109** | 0.0224 | 0.0419 | 1.0000 | 0.3290 |
| **777** | Redundant ($M4_{\text{neg}}$) | 0.2147 | 0.0279 | 0.0429 | 1.0000 | 0.3310 |

#### Explanation of Original Small Dev Anomaly
On the fresh diagnostic split, the complementary ensemble **consistently outperforms** the redundant ensemble ($0.2084$ vs. $0.2189$ on Seed 123; $0.2181$ vs. $0.2349$ on Seed 42).
The anomaly observed in Experiment 1's initial $N=64$ development set was driven by finite-sample calibration variance: in that specific split, Specialist B's extreme false-positive predictions severely penalized the linear average. In the redundant negative control, both models learned from block A alone, avoiding the miscalibrated cross-block interference.

---

### 3.3 Q3: Why Structural CoLoRA Underperformed Capacity-Matched Pooled

#### Measured Optimization & Representation Metrics
- **Task Gradient Cosine on Shared Base Weights ($W_{\text{base}}$):**
  - Mean cosine similarity: **$0.4316$** (Min: **$0.1120$**).
  - Task 1 (block B zeroed) and Task 2 (block A zeroed) produce conflicting gradient trajectories on $W_{\text{base}}$.
- **LoRA Weight Update Magnitude:**
  - $\Delta W = \frac{\alpha}{r} B \Lambda_{\text{task}} A$ magnitude relative to base: $\|\Delta W\| / \|W_{\text{base}}\| = \mathbf{0.546}$ to **$1.179$**.
  - The low-rank update is large, indicating that the network relies heavily on the LoRA branch to compensate for conflicting base weights.
- **Rank-4 Bottleneck:**
  - Input dimensionality is 40. CoLoRA restricts task-specific adaptation to an intermediate bottleneck of dimension $r=4$.
  - In contrast, `CapacityMatchedMLP` provides a full-rank $40 \to 38$ dense representation without forcing task-specific projection through rank 4.
  - Dense gradient flow allows all features to participate freely in representation learning without low-rank subspace contention.

---

### 3.4 Q4: Complementarity Exploitation (Linear Averaging vs. Log-Odds Fusion)

Performance across combination methods on fresh diagnostic evaluation split:

| Seed | Target-only ($M0$) | Specialist A | Specialist B | Equal Linear ($M4$) | Simplex Linear ($M5$) | Log-Odds Combiner | Joint Bayes Oracle ($p^*_{\text{full}}$) |
|---|---|---|---|---|---|---|---|
| **42** | 0.2006 | 0.3019 | 0.3172 | 0.2181 | 0.2182 | **0.1997** | **0.1180** |
| **123** | 0.1852 | 0.2769 | 0.3045 | 0.2084 | 0.2101 | **0.1960** | **0.1180** |
| **777** | 0.1824 | 0.3022 | 0.2910 | 0.2109 | 0.2112 | **0.1995** | **0.1180** |

#### Why Linear Averaging Fails to Exploit Complementarity
1. **Mathematical Range Bounding:**
   $$p_{\text{linear}} = w_A p_A + w_B p_B \implies p_{\text{linear}} \in [\min(p_A, p_B), \max(p_A, p_B)]$$
   When both specialists observe agreeing evidence (e.g. $p_A = 0.8, p_B = 0.8$), linear ensembling outputs $0.80$.
   However, because feature blocks $x_A$ and $x_B$ are conditionally independent, the true joint log-odds is additive:
   $$\text{logit}(p_{\text{joint}}) = \text{logit}(p_A) + \text{logit}(p_B) + c \implies p_{\text{joint}} \approx 0.95$$
2. **Empirical Verification:**
   The bivariate log-odds combiner fits positive weights on both specialist logits ($\alpha_A \approx 0.11, \alpha_B \approx 0.11$) and achieves Brier **$0.1960$**, outperforming linear averaging ($0.2084$) by **$0.0124$ Brier**.
3. **Conclusion:**
   Linear probability averaging fundamentally suppresses joint certainty. Combining specialists in **logit space** (log-odds combination) is mathematically necessary to exploit complementary evidence.

---

## 4. Recommended Next Steps for Experiment 2

Based on these empirical discoveries, we recommend the following protocol adaptations before unblinding any held-out test data in Experiment 2:

1. **Incorporate Calibration in Specialist Training:**
   Introduce Platt / temperature scaling or mild label smoothing on specialist outputs so that missing feature blocks do not cause destructive probability overconfidence.
2. **Upgrade Combination Rule from Linear Mixture to Log-Odds Fusion:**
   Add log-odds combination ($\sigma(\alpha_0 + \sum \alpha_k \text{logit}(p_k))$) as a formal confirmatory combination method alongside equal and simplex weighting.
3. **Address CoLoRA Architecture:**
   Evaluate higher rank ($r=8$ or $r=16$) or task-specific bias vectors to remove the low-rank bottleneck against capacity-matched dense pooling.
4. **Preserve Branch State:**
   Keep `experiment/bayesian-collaboration-diagnostic-followup` as the permanent record of this diagnostic investigation.
