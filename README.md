# Hypergraph Portfolio Lab

[![CI Tests](https://img.shields.io/badge/pytest-43%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A pedagogical, mathematically rigorous, and auditable laboratory based on the research paper:
> **"Distributed Portfolio Optimization & Decentralized Pricing"**  
> *Ioannis Papastaikoudis, Jeremy Watson, Ioannis Lestas (2025)*

This repository serves three core purposes:
1. **First-Principles Education**: Teach distributed portfolio optimization, hypergraphs, consensus methods, continuous-time primal-dual dynamics, and decentralized pricing step-by-step (`intuition -> tiny example -> mathematics -> code -> experiment -> interpretation`).
2. **Critical Mathematical Audit**:
   - Provide an independent, controlled mathematical audit of the paper's central claims (auditing why the distributed objective is reported as 0.8573 vs 2.8408, proving that the problems are not informationally equivalent, and demonstrating true lifted equivalence $\tilde{Q}_{\text{lifted}} = E Q E^T$).
3. **Bayesian Uncertainty Decision Benchmark**:
   - Ingest real **Bayesian-LoRA epistemic uncertainty** and rigorously evaluate whether uncertainty can be converted into economic portfolio decisions across multiple mappings (symmetric shrinkage, covariance inflation, downside probability).

---

## Current Scientific Status

| Research Question | Result / Finding | Holdout Evidence (3 Seeds) | Status |
| :--- | :--- | :--- | :---: |
| **Does epistemic uncertainty predict forecast error?** | Strong monotonic information | Spearman $\rho_s = \mathbf{+0.5502}$, Pearson $r = \mathbf{+0.5707}$, AUROC $= \mathbf{0.8182}$ for top 20% errors. | **Confirmed / Supported** |
| **Does epistemic uncertainty predict downside?** | Weak / Near random chance | AUROC $= \mathbf{0.5613}$ for negative return ($r < 0$); AUROC $= \mathbf{0.5914}$ for large downside event. | **Not Supported / Weak** |
| **Does symmetric shrinkage improve allocation?** | Dampens volatility, drops returns | Volatility reduced by $1.4\%$, return dropped from $43.8\%$ to $35.7\%$; paired $\Delta U$ 95% CIs span zero. | **Not Supported / Neutral** |
| **Does covariance inflation improve allocation?** | Mild drawdown gain, no utility gain | Max drawdown improved ($-14.88\%$ vs $-15.12\%$), but paired $\Delta U$ 95% CI $[-0.00037, +0.00037]$ spans zero. | **Not Supported / Neutral** |
| **Does downside-aware allocation improve allocation?** | Softens negative forecast penalties | M3 dropped return to $27.5\%$, Sharpe to $0.974$, MaxDD to $-22.75\%$; M4 identical to deterministic. | **Not Supported / Negative** |
| **Is distributed uncertainty coordination justified yet?** | Centralized value not established | Centralized uncertainty mappings do not beat deterministic baseline; consensus coordination premature. | **Not Supported / Premature** |

---

## Repository Structure

```
hypergraph-portfolio-lab/
├── pyproject.toml                             # Project metadata and dependencies
├── README.md                                  # Repository overview and scientific guide
├── data/bayesian_lora/                        # Ingested real Bayesian-LoRA predictions
│   ├── development/                           # 2022-2023 development split (Seeds 42, 123, 777)
│   ├── holdout/                               # 2024-2025 frozen holdout split (Seeds 42, 123, 777)
│   └── reconstructed_returns.csv              # Reconstructed asset daily returns
├── src/hypergraph_portfolio/                  # Core Python library
│   ├── __init__.py                            # Public API
│   ├── mean_variance.py                       # Centralized Markowitz solver & KKT
│   ├── hypergraph.py                          # Incidence E, degree D_E, Laplacian L
│   ├── consensus.py                           # Nullspace projectors, audit, lifted system
│   ├── primal_dual.py                         # Continuous-time ODE dynamics & Lyapunov
│   ├── pricing.py                             # Endogenous shadow price mechanism
│   ├── uncertainty.py                         # True confidence shrinkage, cov inflation, downside prob
│   └── bayesian_lora_adapter.py               # Pydantic schema validation & dataset adapter
├── tutorials/                                 # 22 first-principles tutorials (00 to 21)
│   ├── 00_mean_variance.md                    # Centralized Markowitz Mean-Variance
│   ├── 01_why_distributed.md                  # Why Distributed Portfolio Optimization?
│   ├── 02_hypergraphs.md                      # Overlapping Portfolios as Hypergraphs
│   ├── 03_consensus.md                        # Consensus Constraints (Lx = 0)
│   ├── 04_distributed_objective.md            # The Distributed Mean-Variance Problem
│   ├── 05_duality.md                          # Lagrangians and Shadow Prices
│   ├── 06_primal_dual.md                      # Continuous-Time Primal-Dual Dynamics
│   ├── 07_three_asset_reproduction.md         # Section 3 Faithful Reproduction
│   ├── 08_decentralized_pricing.md            # Decentralized Pricing Mechanism
│   ├── 09_equivalence_audit.md                # Critical Audit of the Central Claim
│   ├── 10_gics_example.md                     # 11-Sector GICS ETF Applied Example
│   ├── 11_bayesian_uncertainty.md             # Bayesian Predictive Uncertainty Extension
│   ├── 12_research_synthesis.md               # Research Synthesis & Production Roadmap
│   ├── 13_equivalence_derivation.md           # Analytical Derivation of Objective Equivalence
│   ├── 14_value_of_uncertainty.md             # Empirical Value of Uncertainty (30 Seeds)
│   ├── 15_synthetic_uncertainty_audit.md      # Synthetic Audit & True Confidence Shrinkage
│   ├── 16_bayesian_uncertainty_calibration.md # Hypothesis H1: Uncertainty Calibration
│   ├── 17_bayesian_decision_value.md          # Hypothesis H2: Decision Value & Attribution
│   ├── 18_distributed_bayesian_allocation.md  # Distributed Extension & Local Shrinkage
│   ├── 19_epistemic_risk_vs_alpha.md          # Method 2: Bayesian Covariance Inflation
│   ├── 20_downside_probability.md             # Methods 3 & 4: Posterior Downside Risk
│   └── 21_decision_mapping_comparison.md      # Phase 4 Benchmark & Decision Tree
├── experiments/                               # Reproducible experimental scripts
│   ├── run_three_asset.py                     # Reproduces Section 3.1, 3.2, 3.3
│   ├── run_audit_experiments.py               # Closed-form reduction & audit table
│   ├── run_gics_etf.py                        # 11-Sector GICS ETF simulation
│   ├── run_uncertainty_experiments.py         # 5 candidate uncertainty mechanisms
│   ├── run_bayesian_allocation_value.py       # Controlled 30-seed synthetic simulation
│   ├── run_bayesian_uncertainty_calibration.py# H1 statistical calibration pipeline
│   ├── run_bayesian_decision_value.py         # H2 holdout decision value & attribution
│   └── run_phase4_decision_mapping.py         # Phase 4 decision mapping & bootstrap pipeline
├── artifacts/                                 # Output artifacts & publication figures
│   ├── bayesian_allocation_value/             # Multi-seed synthetic figures & JSON
│   ├── bayesian_uncertainty_calibration/      # Calibration plots, tables, JSON metrics
│   ├── bayesian_decision_value/               # Holdout curves, drawdown, HHI attribution
│   └── phase4_decision_mapping/               # Phase 4 metrics, selectivity, bootstrap, plots
└── tests/                                     # Unit test suite (43 tests)
    ├── test_mean_variance.py
    ├── test_hypergraph.py
    ├── test_consensus.py
    ├── test_primal_dual.py
    ├── test_pricing.py
    ├── test_uncertainty.py
    ├── test_equivalence_reduction.py          # Closed-form reduction & lifted tests
    ├── test_bayesian_simulation.py            # DGP causality, shrinkage, & leakage tests
    ├── test_bayesian_lora_decision_value.py   # Schema validation, adapter, & leakage tests
    └── test_phase4_decision_mapping.py        # Covariance inflation, downside, & bootstrap tests
```

---

## Quickstart & Installation

### 1. Environment Setup

Using `uv` (recommended) or standard Python `venv`:

```bash
# Clone the repository
git clone https://github.com/mbzuaiciao/distributed-portfolio-learning.git
cd distributed-portfolio-learning

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Run All Unit Tests

```bash
.venv/bin/pytest -v tests/
# 35 passed in ~2s
```

### 3. Run Experiments

```bash
# 1. Reproduce paper's 3-asset numerical results (Section 3)
.venv/bin/python experiments/run_three_asset.py

# 2. Run the closed-form mathematical audit suite (Tutorials 09 & 13)
.venv/bin/python experiments/run_audit_experiments.py

# 3. Run 11-sector GICS ETF allocation & pricing (Section 5)
.venv/bin/python experiments/run_gics_etf.py

# 4. Run the 30-seed synthetic Bayesian allocation audit (Tutorials 14 & 15)
.venv/bin/python experiments/run_bayesian_allocation_value.py

# 5. Run Hypothesis H1: Real Bayesian-LoRA Calibration (Tutorial 16)
.venv/bin/python experiments/run_bayesian_uncertainty_calibration.py

# 6. Run Hypothesis H2: Real Decision Value & Attribution (Tutorials 17 & 18)
.venv/bin/python experiments/run_bayesian_decision_value.py

# 7. Run Phase 4: Decision-Relevant Bayesian Risk Mappings (Tutorials 19–21)
.venv/bin/python experiments/run_phase4_decision_mapping.py
```

All figures are automatically saved to `experiments/figures/`, `artifacts/bayesian_decision_value/`, and `artifacts/phase4_decision_mapping/`.

---

## Phase 2: Audit and Value of Uncertainty

Phase 2 deepens the repository across two critical scientific dimensions:

### A. Mathematical Equivalence Audit (Tutorial 13)

By substituting the consensus condition $x = E D_E^{-1} w$ into the paper's distributed objective $F(x) = \sum_{i=1}^K a_i (x_i^T Q_i x_i - \mu_i^T x_i + c_i)$, we obtain the exact closed-form reduced quadratic form:

$$(Q_{\text{eff}})_{j, k} = \frac{\sum_{i: j, k \in \Sigma_i} a_i}{d_j d_k} Q_{j, k}, \quad (\mu_{\text{eff}})_j = \frac{\sum_{i: j \in \Sigma_i} a_i}{d_j} \mu_j, \quad c_{\text{eff}} = \sum_{i=1}^K a_i c_i$$

Comparing this term-by-term with the centralized Markowitz objective $J(w) = w^T Q w - \mu^T w + c$ under uniform desk weights $a_i = 1/K$:
1. **Diagonal variance is downscaled** by $\frac{1}{K d_j}$ (a $3\times$ to $6\times$ reduction in effective asset variance).
2. **Covariances are downscaled or zeroed** by $\frac{n_{jk}}{K d_j d_k}$ (cross-subportfolio covariances vanish).
3. **Linear returns are downscaled** by $\frac{1}{K}$.

When evaluated under the **same ground-truth market objective** $J_{\text{market}}(w) = w^T Q w - \mu^T w$:

| Formulation | Paper Obj Value | Standardized Market Obj | Market Variance | Market Return | Max Asset Weight | Herfindahl Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Markowitz** | $1.8408$ | **$1.8408$** | $2.0436$ | $0.2028$ | $0.4217$ | $0.3471$ |
| **True Lifted Equivalent** | $1.8408$ | **$1.8408$** | $2.0436$ | $0.2028$ | $0.4217$ | $0.3471$ |
| **Distributed (Specific Partitions)** | $0.1906$ | **$1.9239$** | $2.1490$ | $0.2251$ | $0.4632$ | $0.3541$ |
| **Distributed (Powerset Partitions)** | $0.0942$ | **$1.8688$** | $2.0860$ | $0.2171$ | $0.4352$ | $0.3486$ |

*(Note: Market objectives exclude constant offset $c=1.0$ for direct comparability with optimization output).*

**Audit Conclusion**: The paper's distributed objective does not achieve lower true risk. Its lower reported objective is a mathematical artifact of slicing the budget vector into local copies and omitting cross-subportfolio covariances. Constructing the **true lifted system** ($\tilde{Q}_{\text{lifted}} = E Q E^T$) exactly recovers Markowitz equivalence.

---

### B. Controlled Bayesian Allocation Experiment (Tutorial 14)

We evaluate whether epistemic predictive uncertainty $u_t = \operatorname{Var}(\mu_t \mid \mathcal{D})$ adds economic value when the point forecast $\hat{\mu}_t$ is held **strictly identical** between policies:

$$\tilde{\mu}_{t, j} = \frac{\hat{\mu}_{t, j}}{1 + \lambda_u u_{t, j}}$$

Tested across **30 independent random seeds** $\times$ **5 information calibration regimes** with 10 bps transaction costs and single-period utility $U_t = r_t - \gamma \cdot \text{risk}_t - \kappa \cdot \text{turnover}_t$:

| Calibration Regime | Paired $\Delta$ Sharpe | $t$-statistic | Paired $\Delta$ Utility | $t$-statistic | Max Drawdown Imp. |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Well-Calibrated** ($u_t \propto \text{Error}_t$) | **$+0.094 \pm 0.033$** | **$+2.85^{**}$** | **$+0.00114 \pm 0.00037$** | **$+3.08^{**}$** | **$+1.71\%$** ($-24.7\%$ vs $-23.0\%$) |
| **Regime Shift** (structural break) | **$+0.095 \pm 0.031$** | **$+3.06^{**}$** | **$+0.00116 \pm 0.00037$** | **$+3.14^{**}$** | **$+3.26\%$** ($-25.6\%$ vs $-22.3\%$) |
| **Weakly-Calibrated** (noisy uncertainty) | $-0.030 \pm 0.024$ | $-1.22$ | $-0.00041 \pm 0.00028$ | $-1.46$ | $-0.47\%$ |
| **Uninformative** (random noise) | **$-0.091 \pm 0.023$** | **$-4.00^{***}$** | **$-0.00115 \pm 0.00028$** | **$-4.05^{***}$** | **$-1.60\%$** |
| **Miscalibrated** (overconfident on errors) | **$-0.107 \pm 0.024$** | **$-4.47^{***}$** | **$-0.00149 \pm 0.00032$** | **$-4.63^{***}$** | **$-1.19\%$** |

**Empirical Takeaway**:
- Epistemic uncertainty is not alpha; it functions as an **adaptive Bayesian regularizer** that mitigates error maximization.
- When uncertainty is well-calibrated or signals regime changes, it delivers statistically significant gains in Sharpe ratio, net utility, and maximum drawdown protection.
- When uncertainty is uninformative or miscalibrated, it actively destroys portfolio value. Reliable out-of-sample calibration is a mandatory prerequisite before deployment.

---

## Phase 3: Real Bayesian-LoRA Decision Value

Phase 3 transitions from synthetic uncertainty to evaluating actual deep learning predictions from **Bayesian-LoRA** models trained on equity returns. We test two central hypotheses:

1. **Hypothesis H1 (Informational Value)**: Does out-of-sample Bayesian-LoRA epistemic uncertainty contain statistically significant information about future forecast error magnitude?
2. **Hypothesis H2 (Decision Value)**: Does incorporating this uncertainty via confidence shrinkage improve out-of-sample portfolio performance when point forecasts are held strictly identical?

---

### Part A: Synthetic Audit & True Confidence Shrinkage (Tutorial 15)

Prior to ingesting real neural network outputs, we audited the Phase 2 synthetic simulation engine:
- **Temporal Causality**: Enforced strict causal ordering $F_t \to \hat{\mu}_t \to \sigma_{\text{epi}, t} \to w_t \to r_{t+1}$ via an ex-ante latent state $h_t$ that jointly generates forecast noise and epistemic uncertainty. Next-period market return $r_{t+1}$ is drawn strictly ex-post.
- **True Confidence Shrinkage**: Replaced pseudo-shrinkage ($\hat{\mu} - \lambda \sigma$) with true Bayesian shrinkage toward a neutral prior $\mu_{\text{prior}}$:
  $$\tilde{\mu}_j = \rho_j \hat{\mu}_j + (1 - \rho_j) \mu_{\text{prior}}, \quad \rho_j = \frac{1}{1 + \lambda_u \sigma_{\text{epi}, j}}$$
- **Controlled Re-run**: Re-ran the 30-seed simulation ($\lambda_u = 50.0$), confirming Phase 2 findings: informative uncertainty gains $+0.069$ Sharpe ($t = 3.14$), regime shifts gain $+0.078$ Sharpe ($t = 3.90$), while random noise hurts ($-0.031$ Sharpe).

---

### Part B: Real Dataset Ingestion & Validation

We ingest out-of-sample predictions from Bayesian-LoRA models across three random initializations (Seeds 42, 123, 777) spanning:
- **Development Split (2022–2023)**: 501 trading days across 4 liquid assets (NVDA, MSFT, AAPL, AMZN) used strictly for walk-forward parameter tuning.
- **Holdout Split (2024–2025)**: 432 trading days (1,728 asset-day events) frozen until final evaluation.
- **Data Integrity**: Enforced via `BayesianPredictionRecord` schema validation, checking timestamps, non-negative epistemic/aleatoric variances, and identical point forecasts.

---

### Part C: Hypothesis H1 — Informational Value (Tutorial 16)

We evaluate whether out-of-sample epistemic uncertainty $\sigma_{\text{epi}, t}$ predicts realized absolute forecast error $|\hat{\mu}_t - r_{t+1}|$:

| Metric | Seed 42 | Seed 123 | Seed 777 | Mean $\pm$ Std | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Spearman Rank Corr ($\rho_s$)** | $+0.5961$ | $+0.5768$ | $+0.4777$ | $\mathbf{+0.5502 \pm 0.0519}$ | $p < 10^{-100}$ |
| **Pearson Corr ($r$)** | $+0.5898$ | $+0.5901$ | $+0.5323$ | $\mathbf{+0.5707 \pm 0.0319}$ | $p < 10^{-100}$ |
| **AUROC (Top 20% Errors)** | $0.8402$ | $0.8443$ | $0.7701$ | $\mathbf{0.8182 \pm 0.0341}$ | Highly Discriminative |
| **Top Quintile Mean Error (Q4)** | $25.70\%$ | $26.79\%$ | $22.95\%$ | $\mathbf{25.15\%}$ | $2.122\times$ baseline error |
| **Bottom Quintile Mean Error (Q1)**| $11.75\%$ | $11.95\%$ | $11.87\%$ | $\mathbf{11.86\%}$ | Stable, accurate regime |
| **Ratio (Q4 / Q1)** | $2.188\times$ | $2.242\times$ | $1.934\times$ | $\mathbf{2.122\times}$ | Strong separation |

> **Scientific Verdict for H1**: **CONFIRMED / SUPPORTED**.  
> Bayesian-LoRA epistemic uncertainty contains strong, statistically significant information about future forecast error magnitude. It successfully identifies when neural network predictions are unreliable.

---

### Part D & E: Hypothesis H2 — Decision Value & Attribution (Tutorial 17)

We optimize the shrinkage hyperparameter $\lambda_u^*$ on 2022–2023 development data ($\lambda_u^* = 100.0$), freeze it, and run the out-of-sample backtest on the 2024–2025 holdout across all three seeds:

| Seed | Policy | Ann. Return | Ann. Vol | Sharpe Ratio | Max DD | Net Utility | Paired $\Delta$ Utility [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | Deterministic | $35.33\%$ | $29.41\%$ | $1.201$ | $-14.75\%$ | $0.00067$ | — |
| | Uncertainty-Aware | $24.32\%$ | $27.60\%$ | $0.881$ | $-16.94\%$ | $-0.00010$ | $-0.00077$ $[-0.0043, +0.0023]$ |
| **123** | Deterministic | $51.27\%$ | $29.89\%$ | $1.715$ | $-13.92\%$ | $0.00125$ | — |
| | Uncertainty-Aware | $45.41\%$ | $28.16\%$ | $1.613$ | $-14.94\%$ | $0.00148$ | $+0.00023$ $[-0.0025, +0.0027]$ |
| **777** | Deterministic | $44.89\%$ | $27.31\%$ | $1.644$ | $-11.73\%$ | $0.00115$ | — |
| | Uncertainty-Aware | $37.56\%$ | $26.55\%$ | $1.415$ | $-11.70\%$ | $0.00055$ | $-0.00060$ $[-0.0028, +0.0012]$ |

#### Why Did Decision Value Fail Despite Strong Calibration?

We performed an orthogonal attribution analysis deconstructing the mechanism:
1. **Exposure Dampener vs Alpha Filter**: Real-world epistemic uncertainty correlates positively with absolute forecast magnitude $|\hat{\mu}_j|$. When the network forecasts large positive returns, its epistemic uncertainty is elevated.
2. **Indiscriminate Penalization**: Symmetric confidence shrinkage penalizes high-uncertainty assets regardless of whether the forecast error is positive or negative. It successfully reduced exposure to large negative misses, but simultaneously penalized the portfolio's highest-performing winning trades.
3. **Concentration Effect**: Portfolio concentration (Herfindahl Index) dropped by $12.2\%$ to $16.8\%$. However, the ratio of capital allocated to bad forecasts vs good forecasts actually slightly increased ($0.737 \to 0.773$).
4. **Utility Impact**: In all three seeds, the paired net utility difference spans zero. While annualized volatility was reduced by $1.5\%$ to $1.8\%$, return fell proportionally.

> **Scientific Verdict for H2**: **NOT SUPPORTED / NEUTRAL**.  
> Real informational value in epistemic uncertainty (H1) does **not** automatically translate into economic decision value (H2) under naive symmetric confidence shrinkage. Epistemic uncertainty functions as a volatility dampener rather than a selective alpha enhancer.

---

### Part F: Distributed Extension (Tutorial 18)

We demonstrated how Bayesian confidence shrinkage naturally integrates into the paper's hypergraph primal-dual consensus dynamics:
- Local desks compute shrunk expected returns $\tilde{\mu}_i = \rho_i \odot \hat{\mu}_i$ autonomously without sharing model weights or epistemic variance matrices.
- Primal-dual consensus dynamics converge to the exact lifted Bayesian optimum ($100.0\%$ numerical agreement).
- Future architectural directions to bridge the gap from H1 to H2:
  1. **Asymmetric / Downside-Only Shrinkage**: Shrink only when $\hat{\mu}_j > 0$ and downside epistemic variance is elevated;
  2. **Endogenous Risk Augmentation**: Expand localized covariance matrices $Q_i \leftarrow Q_i + \lambda_u \operatorname{diag}(\sigma_{\text{epi}, i}^2)$;
  3. **Confidence-Weighted Consensus**: Weight agent disagreement penalties by relative epistemic certainty.

---

## Phase 4: Decision-Relevant Bayesian Risk

Phase 4 evaluates whether Bayesian-LoRA epistemic uncertainty can be converted into economic portfolio value using decision mappings other than symmetric mean shrinkage:

1. **Method 2 — Bayesian Covariance Inflation**: Leaves expected returns intact ($\hat{\mu}$) and treats epistemic uncertainty as quadratic estimation risk ($Q_{\text{robust}} = Q_{\text{market}} + \lambda_{\text{cov}}^* \operatorname{diag}(\sigma_{\text{epi}}^2)$). [Tutorial 19](tutorials/19_epistemic_risk_vs_alpha.md).
2. **Method 3 — Downside-Adjusted Alpha**: Computes posterior downside probability $p_{\text{down}} = P(\mu < 0 \mid \mathcal{D})$ and shrinks alpha only when downside probability is elevated ($\tilde{\mu}_j = \hat{\mu}_j \cdot \max(0, 1 - \lambda_{\text{down}}^* p_{\text{down}, j})$). [Tutorial 20](tutorials/20_downside_probability.md).
3. **Method 4 — Downside Covariance Inflation**: Penalizes covariance along downside-prone assets ($Q_{\text{down}} = Q_{\text{market}} + \lambda_{\text{down}}^* \operatorname{diag}(p_{\text{down}} \cdot \sigma_{\text{epi}}^2)$). [Tutorial 21](tutorials/21_decision_mapping_comparison.md).

All hyperparameters were tuned strictly on 2022–2023 development data ($\lambda_{\text{cov}}^* = 100.0, \lambda_{\text{down}, 3}^* = 2.0, \lambda_{\text{down}, 4}^* = 0.3$) and frozen before evaluating the 2024–2025 holdout across Seeds 42, 123, 777.

---

### Comparative Holdout Results (3 Seeds Aggregated)

| Policy | Annualized Return | Annualized Volatility | Sharpe Ratio | Sortino Ratio | Max Drawdown | Mean Net Utility | Bad/Good Capital Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0: Deterministic** | $\mathbf{43.80\% \pm 6.59\%}$ | $28.87\% \pm 1.12\%$ | $\mathbf{1.519 \pm 0.227}$ | $\mathbf{2.894 \pm 0.488}$ | $-15.12\%$ | $\mathbf{-0.00621 \pm 0.00140}$ | $\mathbf{0.651 \pm 0.056}$ |
| **M1: Symmetric Shrink** | $35.73\% \pm 8.72\%$ | $\mathbf{27.45\% \pm 0.66\%}$ | $1.301 \pm 0.310$ | $2.473 \pm 0.580$ | $-16.51\%$ | $-0.00659 \pm 0.00167$ | $0.680 \pm 0.056$ |
| **M2: Covariance Inflation**| $41.99\% \pm 6.31\%$ | $28.50\% \pm 1.07\%$ | $1.475 \pm 0.219$ | $2.831 \pm 0.461$ | $\mathbf{-14.88\%}$ | $-0.00636 \pm 0.00130$ | $0.658 \pm 0.056$ |
| **M3: Downside Alpha** | $27.53\% \pm 4.51\%$ | $28.19\% \pm 1.20\%$ | $0.974 \pm 0.131$ | $1.648 \pm 0.257$ | $-22.75\%$ | $-0.00791 \pm 0.00086$ | $0.709 \pm 0.034$ |
| **M4: Downside Cov** | $43.80\% \pm 6.59\%$ | $28.87\% \pm 1.12\%$ | $1.518 \pm 0.227$ | $2.893 \pm 0.488$ | $-15.12\%$ | $-0.00621 \pm 0.00140$ | $0.651 \pm 0.056$ |

---

### Paired Circular Block Bootstrap Statistics

Using circular block bootstrap ($B = 4$ weeks, 1,000 resamples) evaluating paired net utility difference $\Delta U = U_{\text{policy}} - U_{\text{M0}}$:
- **M2 (Covariance Inflation)**: $\Delta U = -0.000011$ (95% CI: $[-0.000372, +0.000374]$ in Seed 42); $[-0.000795, +0.000318]$ in Seed 123; $[-0.001011, +0.000158]$ in Seed 777. All 95% CIs span zero ($p > 0.48$).
- **M3 (Downside Alpha)**: Consistently underperformed the baseline across all three seeds ($p = 0.21 - 0.75$).
- **M4 (Downside Cov)**: Performance was statistically indistinguishable from the deterministic baseline ($\Delta U \approx 0.00000$).

---

### Key Empirical Takeaways & Scientific Decision Tree

1. **Failure of Selective Filtering**:
   No uncertainty-aware mapping was selective. The deterministic baseline achieved the lowest (best) Bad-to-Good capital allocation ratio ($0.651$). All uncertainty mappings either worsened the ratio ($0.658 - 0.709$) or left it unchanged.
2. **The High-Alpha / High-Uncertainty Trap**:
   Assets with high point alpha and high uncertainty produced the worst realized returns ($-1.36\%$ to $-1.68\%$). The deterministic optimizer already minimized exposure to them ($0.85\% - 1.10\%$). Heuristic downside alpha shrinkage (M3) softened these negative forecasts toward zero, accidentally doubling allocation to toxic assets ($2.07\% - 2.82\%$).
3. **Forecast Error vs. Downside Calibration**:
   Epistemic uncertainty strongly predicts *absolute forecast error* ($\text{AUROC} = \mathbf{0.818}$), but is nearly uninformative for predicting *directional downside return* ($\text{AUROC} = \mathbf{0.561}$).
4. **Decision Tree Classification: OUTCOME D (Clean Negative Result)**:
   The Bayesian-LoRA uncertainty signal predicts model error, but under the tested portfolio architectures, it does not provide incremental economic value.
5. **Distributed Coordination Implications**:
   Modifying the distributed consensus mechanism (e.g., confidence-weighted consensus or uncertainty-dependent dual variables) is **premature and scientifically unjustified** until a centralized decision mapping demonstrates credible economic value.

---

## Code Quality & Standards

- **Formatting & Linting**: Verified with `ruff` (`ruff check .` passes with 0 errors).
- **Testing**: 43 comprehensive unit tests covering all core modules with `pytest`.
- **Determinism**: Fixed random seeds, exact analytical benchmarks, and verifiable hand calculations.

```bash
.venv/bin/ruff check .
.venv/bin/pytest -v
```

---

## Citation & Reference

If using this codebase for research or teaching, please cite:

```bibtex
@article{papastaikoudis2025distributed,
  title={Distributed Portfolio Optimization & Decentralized Pricing},
  author={Papastaikoudis, Ioannis and Watson, Jeremy and Lestas, Ioannis},
  journal={Working Paper, University of Cambridge / University of Canterbury},
  year={2025}
}
```
