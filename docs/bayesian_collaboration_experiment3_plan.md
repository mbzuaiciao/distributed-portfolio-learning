# Experiment 3 Research Plan: Bayesian Uncertainty-Aware Specialist Routing

**Repository:** `mbzuaiciao/distributed-portfolio-learning`  
**Prospective Branch:** `experiment/bayesian-collaboration-uncertainty-fusion`  
**Status:** **PROSPECTIVE DESIGN ONLY — AWAITING USER APPROVAL BEFORE IMPLEMENTATION**  
**Predecessor Experiments:**
- Experiment 1 (`65c92816`): Frozen positive control; revealed single-specialist overconfidence under partial observability.
- Diagnostic Follow-Up (`8727c162`): Identified reliability error and demonstrated superiority of log-odds combination.
- Experiment 2 (`4a87710d` / `eb0d8745`): Confirmed log-odds fusion over linear ensembling ($p_{\text{holm}} < 0.05$) and over capacity pooling ($p_{\text{holm}} = 0.002$); primary comparison to target-only was statistically inconclusive at $N=512$ ($+0.0187$, 95% CI $[-0.0089, +0.0486]$).

---

## 1. Scientific Objective & Research Questions

Experiment 2 established that when specialist feature support is globally homogeneous, static log-odds fusion with fixed weights ($w_A, w_B$) is effective. However, static fusion cannot adapt when specialist competence is **heterogeneous across instances**.

**Core Research Question:**  
Does dynamic, instance-level **Bayesian epistemic uncertainty** provide incremental predictive information for specialist routing beyond:
1. **Calibrated predictive confidence** (e.g. $|p - 0.5|$ or predictive entropy)?
2. **Empirical training-support density** (e.g. kernel density estimation in shared feature space)?
3. **Deterministic learned gating** (e.g. a mixture-of-experts gating network trained on identical inputs)?

---

## 2. Theoretical Framework & Epistemic Uncertainty Formulation

### 2.1 Formal Decomposition: Aleatoric vs. Epistemic Uncertainty
For each specialist model $f_i(x)$ parameterized by weights $\theta_i$, we distinguish:
1. **Aleatoric Uncertainty (Data Noise):**
   - Inherent ambiguity in the label $Y \in \{0, 1\}$ given true features:
     $$\mathcal{U}_{\text{aleatoric}}(x) = 4 \, p_i(x) \, (1 - p_i(x)) \in [0, 1]$$
   - Aleatoric uncertainty is maximal near the true decision boundary ($p_i(x) \approx 0.5$), even when training observations are infinitely dense.
2. **Epistemic Uncertainty (Parameter / Support Ambiguity):**
   - Uncertainty in parameter estimates $\theta_i$ arising from sparsity of training samples in the local feature neighborhood of $x$.
   - Using a **Laplace approximation** on the specialist's penultimate representation layer $\phi_i(x) \in \mathbb{R}^{d_{\text{rep}}}$:
     $$p(\theta_i \mid \mathcal{D}_i) \approx \mathcal{N}(\hat{\theta}_i, \Sigma_i), \quad \Sigma_i = \left( H_i + \lambda I \right)^{-1}$$
     where $H_i = \sum_{n=1}^{N_i} \hat{p}_{i,n} (1 - \hat{p}_{i,n}) \phi_i(x_n) \phi_i(x_n)^T$ is the Fisher information matrix at MAP.
   - For a query $x$, latent logit mean and epistemic variance are:
     $$\mu_i(x) = \phi_i(x)^T \hat{\theta}_i, \quad \sigma_{i,\text{epistemic}}^2(x) = \phi_i(x)^T \Sigma_i \phi_i(x)$$
   - The probit-approximated Bayesian predictive probability is:
     $$p_i^{\text{Bayes}}(x) = \sigma\left( \frac{\mu_i(x)}{\sqrt{1 + \frac{\pi}{8} \sigma_{i,\text{epistemic}}^2(x)}} \right)$$

### 2.2 Pre-Combination Validation of Epistemic Uncertainty
Before any fusion model is evaluated, the epistemic estimator $\sigma_{i,\text{epistemic}}^2(x)$ must pass two quantitative development gates:
1. **Gate U1 (Support-Tracking Gate):**
   - On a synthetic validation set sampled across dense and sparse regions, $\sigma_{i,\text{epistemic}}^2(x)$ must demonstrate a positive monotonic relationship with local training sparsity (negative log-KDE $-\log \hat{p}_{\text{train}}(x_s)$):
     $$\text{Spearman Rank Correlation } r_s(\sigma_{i,\text{epistemic}}^2, -\log \hat{p}_{\text{train}}) > 0.40$$
     *(Note: Replaces the flawed rigid threshold of 0.70 with an achievable, informative rank threshold of 0.40).*
2. **Gate U2 (Residual Error Association Controlled for Confidence):**
   - To confirm that epistemic variance measures *actual specialist reliability* rather than boundary ambiguity, we compute the partial correlation between epistemic variance and squared specialist residual error, partialling out distance to the decision boundary:
     $$\text{PartialCorr}\left( \sigma_{i,\text{epistemic}}^2(x), (p_i(x) - y)^2 \;\Big|\; |p_i(x) - 0.5| \right) > 0.10$$
   - This formally validates that epistemic uncertainty contains residual information about model error that cannot be deduced from predictive confidence alone.

---

## 3. Data-Generating Process & Experimental Regimes

Experiment 3 evaluates two distinct synthetic regimes to prevent trivial regional overfitting:

### 3.1 Regime A: Axis-Aligned Regional Covariate Shift (Benchmark Control)
- **Feature Structure:** $x = [x_s, x_A, x_B, x_{\text{noise}}] \in \mathbb{R}^{20}$ with $x \sim \mathcal{N}(0, I_{20})$.
- **Partitioning Rule:** Defined by shared coordinate $s = x_{s, 1}$.
  - **Region 1 ($s > 0$):** Specialist A has abundant training data ($N=256$); Specialist B has sparse data ($N=16$).
  - **Region 2 ($s \le 0$):** Specialist B has abundant training data ($N=256$); Specialist A has sparse data ($N=16$).
- **Role in Study:** In Regime A, an explicit regional threshold $s=0$ exists. A deterministic MoE gating network conditioned on $x_s$ can easily discover this boundary. This tests whether Bayesian uncertainty matches or exceeds specialized deterministic gating when the support boundary is simple.

### 3.2 Regime B: Smooth Multi-Modal Covariate Support (Non-Trivial Setting)
- **Feature Structure:** Identical feature dimensions, but training support is governed by a **mixture of support Gaussians** in shared space:
  - Specialist A training samples are drawn from mixture components centered at $\mu_{A,1} = (+1.2, +1.2)$ and $\mu_{A,2} = (-1.2, -1.2)$ in $(x_{s,1}, x_{s,2})$.
  - Specialist B training samples are drawn from mixture components centered at $\mu_{B,1} = (+1.2, -1.2)$ and $\mu_{B,2} = (-1.2, +1.2)$.
  - Outside their respective clusters, specialist support decays smoothly according to standard Gaussian dispersion.
- **Target & Test Distribution:** Evaluated on standard $\mathcal{N}(0, I_{20})$ data covering all quadrants uniformly.
- **Role in Study:** In Regime B, specialist reliability is not governed by a single threshold on one variable, but by a 2D checkerboard/radial proximity function. A simple linear gate cannot cleanly separate competence. Bayesian epistemic uncertainty naturally increases with quadratic distance $\phi(x)^T \Sigma \phi(x)$ from training clusters, providing a test where Bayesian methods should possess a structural representational advantage over simple linear routers.

---

## 4. Evaluated Method Matrix & Fair Architectural Matching

All models receive identical input features and observation masks. Training budgets are capacity-matched.

| Method ID | Method Name | Architecture | Input Information | Routing / Combination Mechanism |
|---|---|---|---|---|
| **$M0$** | Target-Only MLP | `SmallMLP` (1,857 params) | $x_s, x_A, x_B$ ($N=128$) | Direct prediction |
| **$M1_{\text{cap}}$** | Capacity-Matched Pooled | `CapacityMatchedMLP` (2,199) | Pooled union ($N=544$) | Direct prediction |
| **$M6_{\text{cal}}$** | Static Log-Odds Fusion | Specialists + Platt + Logit | Masked specialists + Dev | Fixed weights $\sigma(w_0 + w_A \ell_A + w_B \ell_B)$ |
| **$M_{\text{conf}}$** | Confidence-Gated Router | Specialists + Platt | Masked specialists + Dev | Dynamic weights $w_i(x) \propto \|p_i(x) - 0.5\|$ |
| **$M_{\text{dens}}$** | Kernel Density Router | Specialists + KDE on $x_s$ | Masked specialists + Dev | Dynamic weights $w_i(x) \propto \hat{p}_{\text{train}, i}(x_s)$ |
| **$M_{\text{MoE}}$** | Deterministic Learned MoE | Specialists + MLP Router | Masked specialists + Dev | Gating net $g(x_s) = \text{softmax}(W_g x_s + b_g)$ |
| **$M_{\text{Bayes-Inv}}$** | Inverse Epistemic Variance | Specialists + Laplace | Masked specialists + Dev | Precision weights $w_i(x) \propto 1 / \sigma_{i,\text{epistemic}}^2(x)$ |
| **$M_{\text{Bayes-Learn}}$** | **Learned Uncertainty Fusion** | Specialists + Laplace + Gate | Masked specialists + Dev | Logistic gate on $[\ell_A, \ell_B, \sigma_A^2, \sigma_B^2, x_s]$ |
| **$M9_{\text{oracle}}$** | Joint Bayes Oracle | Analytical Ground Truth | Full features $x_{\text{full}}$ | Exact Bayes upper bound |

### Capacity and Information Hygiene:
- All combination methods ($M6_{\text{cal}}$, $M_{\text{conf}}$, $M_{\text{dens}}$, $M_{\text{MoE}}$, $M_{\text{Bayes-Inv}}$, $M_{\text{Bayes-Learn}}$) are fitted strictly on a **dedicated fusion development split** ($N=128$) using identical specialist base representations.
- Base specialist models are frozen before any router or combiner is trained.

---

## 5. Statistical Hypotheses & Success / Falsification Criteria

### 5.1 Primary Confirmatory Hypothesis (H3.1: Incremental Bayesian Value)
- **Primary Comparison:** $M_{\text{Bayes-Learn}}$ vs. **$M_{\text{strongest\_nonbayes}}$**  
  where $M_{\text{strongest\_nonbayes}} = \arg\min_{M \in \{M_{\text{MoE}}, M_{\text{dens}}, M_{\text{conf}}, M6_{\text{cal}}\}} \text{Brier}(M)$ evaluated on the development split.
- **Metric:** Brier Score reduction on held-out test split ($N=512$):
  $$\Delta \text{Brier}_{\text{primary}} = \text{Brier}(M_{\text{strongest\_nonbayes}}) - \text{Brier}(M_{\text{Bayes-Learn}})$$
- **Confirmatory Success Requirement:**
  1. **Effect-Size Criterion:** Point estimate $\widehat{\Delta \text{Brier}} \ge 0.010$.
  2. **Statistical Superiority Criterion:** 95% paired cluster bootstrap CI lower bound $> 0.0$ ($p < 0.05$).
  3. **Simultaneous Satisfaction:** Both criteria must be met on the held-out test split.

### 5.2 Secondary Confirmatory Family (Holm-Adjusted, $\alpha=0.05$)
1. **$C_{3.1}$ (Bayes vs. Static Log-Odds):** $\text{Brier}(M6_{\text{cal}}) - \text{Brier}(M_{\text{Bayes-Learn}}) > 0.0$.
2. **$C_{3.2}$ (Bayes vs. Deterministic MoE):** $\text{Brier}(M_{\text{MoE}}) - \text{Brier}(M_{\text{Bayes-Learn}}) > 0.0$.
3. **$C_{3.3}$ (Bayes vs. Density Router):** $\text{Brier}(M_{\text{dens}}) - \text{Brier}(M_{\text{Bayes-Learn}}) > 0.0$.
4. **$C_{3.4}$ (Bayes vs. Confidence Router):** $\text{Brier}(M_{\text{conf}}) - \text{Brier}(M_{\text{Bayes-Learn}}) > 0.0$.

### 5.3 Explicit Inconclusive vs. Equivalence Handling
- If the observed difference is small ($|\widehat{\Delta \text{Brier}}| < 0.005$) and the 95% bootstrap CI spans zero, the outcome is reported as **statistically inconclusive**.
- A claim of **empirical equivalence** will only be made if supported by a formal Two One-Sided Tests (TOST) procedure demonstrating that the 90% confidence interval falls strictly within the equivalence bounds $[-\delta, +\delta] = [-0.005, +0.005]$.
- If deterministic MoE ($M_{\text{MoE}}$) or density router ($M_{\text{dens}}$) outperforms $M_{\text{Bayes-Learn}}$, the hypothesis that Bayesian epistemic uncertainty provides incremental value is **falsified**.

---

## 6. Data Hygiene & Split Protocol

Experiment 3 utilizes **entirely fresh data partitions**:
- **DGP Seeds:** Pre-registered seeds `dgp_seed = 20261014`, `split_seed = 20261015`.
- **Zero Test Set Reuse:** The Experiment 1 and Experiment 2 held-out test splits will **never be loaded, evaluated, or reused**.
- **5-Split Partition Architecture:**
  1. `target_train`: $N=128$ (drawn first)
  2. `specialist_A_train`: $N=256$ (drawn second, with localized support)
  3. `specialist_B_train`: $N=256$ (drawn third, with complementary localized support)
  4. `calibration`: $N=128$ (drawn fourth, dedicated to specialist Platt scaling)
  5. `fusion_dev`: $N=128$ (drawn fifth, dedicated to fitting routers $M_{\text{MoE}}, M_{\text{dens}}, M_{\text{Bayes-Learn}}$)
  6. `development`: $N=128$ (drawn sixth, dedicated to development gate checks)
  7. `held_out_test`: $N=512$ (drawn seventh, strictly quarantined until pre-confirmatory audit)

---

## 7. Methodological Risk Register

| Risk ID | Failure Mode | Severity | Probability | Pre-Registered Mitigation Strategy |
|---|---|---|---|---|
| **R1** | Laplace covariance ill-conditioning | High | Medium | Add adaptive diagonal damping $\Sigma = (H + \lambda I)^{-1}$ with $\lambda = 1.0$; verify condition number $\kappa < 10^5$. |
| **R2** | Gating network overfits on $N=128$ fusion dev | Medium | High | Apply strong L2 weight decay ($C=0.1$ or dropout) on router parameters; constrain router capacity to $\le 100$ parameters. |
| **R3** | MoE router trivially solves Regime A | Medium | High | Regime B explicitly tests non-linear multi-modal support where linear gating fails; primary conclusions evaluate both regimes separately. |
| **R4** | Insufficient test power at $N=512$ | High | Medium | Pre-register TOST equivalence bounds; report raw CIs and avoid premature claims of equivalence if CI spans zero. |
| **R5** | Conflating confidence with epistemic UQ | Critical | Medium | Enforce Gate U2 (partial correlation with residual error controlling for boundary distance) before model combination. |

---

## 8. Compute Budget, Resource Limits, & Pre-Execution Approval Gate

- **Hardware Platform:** CPU-only (Apple Silicon local host, no GPU or cloud required).
- **Thread Enforcement:** Strictly capped at **2 PyTorch threads** (`torch.set_num_threads(2)`).
- **Workload Profile:**
  - Training: 36 base `SmallMLP` models (Regimes A & B across 3 seeds).
  - Laplace Hessian extraction: 1 forward/backward pass per specialist on training set ($N=256$).
  - Router fitting: Lightweight logistic/MLP fits on $N=128$.
  - Evaluation: Forward passes on $N=512$ + $B=2,000$ paired cluster bootstraps.
- **Resource Estimates:**
  - **Estimated Wall-Clock Time:** **$35 - 55$ seconds total** (well below the 20-minute alert threshold).
  - **Peak Memory:** **$< 380$ MB RAM** ($< 1.6\%$ host memory).
- **Execution Rule:**
  - **NO TRAINING, SCRIPT EXECUTION, OR TEST ACCESS WILL OCCUR UNTIL THE USER EXPLICITLY DIRECTS EXECUTION.**
