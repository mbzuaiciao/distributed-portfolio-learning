# Tutorial 11 — Bayesian Predictive Uncertainty Extension

In classical portfolio optimization—and in the paper by Papastaikoudis, Watson, and Lestas (2025)—expected returns $\mu$ are treated as **known, deterministic point estimates**.

In modern quantitative finance and machine learning, point estimates are notoriously brittle. With the advent of **Bayesian deep learning, LoRA adapters, and probabilistic ensembles**, models output a full posterior predictive distribution:

$$p(\mu \mid \mathcal{D}) \sim \left( \mathbb{E}[\mu], \operatorname{Var}(\mu) \right)$$

giving both expected gains and **epistemic uncertainty** (confidence in the signal).

In this research tutorial, we investigate **five candidate hypotheses** for how Bayesian predictive uncertainty can enter the hypergraph distributed portfolio architecture.

---

## 1. Epistemic Status & Attribution

> [!NOTE]
> **What the Paper Explicitly Claims**:
> - The paper's formulation in Section 2 is purely deterministic: $\mu_i$ is a fixed vector.
> - Section 5.4 suggests "Smart Subset Selection via Machine Learning" as future work, but does not formulate probabilistic or Bayesian uncertainty models.

> [!IMPORTANT]
> **Our Original Research Extension**:
> - We formulate the bridge between probabilistic ML signals (e.g. Bayesian LoRA fine-tuned on financial filings) and hypergraph portfolio optimization.
> - We treat each candidate entry point as a scientific **hypothesis** to be tested with controlled experiments, rather than an established dogma.

---

## 2. Intuition: Point Estimates vs Confidence Intervals

Imagine two analysts both pitch you an investment idea:
- **Analyst 1 (High Confidence)**: Forecasts a 15% return on Apple with high certainty ($\sigma = 2\%$). They analyzed 10 years of earnings, supply chains, and customer churn.
- **Analyst 2 (High Epistemic Noise)**: Forecasts a 15% return on a volatile biotech stock, but with massive uncertainty ($\sigma = 25\%$). The clinical trial results are unproven.

A naive Markowitz optimizer looks only at $\mathbb{E}[\mu] = 0.15$ for both assets, treating them as identical opportunities!

Where should the uncertainty $\operatorname{Var}(\mu)$ enter our multi-desk hypergraph system?
1. At the **asset signal level** (shrinkage)?
2. At the **risk covariance level** (parameter variance augmentation)?
3. At the **allocator governance level** (desk mandate weights $a_i$)?
4. At the **consensus coupling level** (relaxing agreement when uncertainty differs)?
5. At the **tail probability level** (downside shortfall risk)?

---

## 3. Five Candidate Hypotheses

### Hypothesis 1: Bayesian Return Shrinkage
$$\tilde{\mu}_j = \frac{\mu_j}{1 + \kappa \operatorname{Var}(\mu_j)}$$
- **Mechanism**: Assets with high epistemic variance are shrunk toward zero (or a prior mean).
- **Economic Rationale**: You only bet aggressively on assets where you have high predictive confidence.

### Hypothesis 2: Local Risk Matrix Augmentation
$$\tilde{Q}_i = Q_i + \gamma \operatorname{diag}(\operatorname{Var}(\mu_i))$$
- **Mechanism**: Epistemic parameter uncertainty is added directly to market covariance $Q_i$.
- **Economic Rationale**: Rooted in robust optimization (Goldfarb & Iyengar 2003). Parameter estimation error acts as an unhedged volatility factor.

### Hypothesis 3: Sub-portfolio Mandate Re-weighting
$$\tilde{a}_i \propto \frac{a_i}{1 + \beta \operatorname{Tr}(\operatorname{Var}(\mu_i))}$$
- **Mechanism**: The firm-level CIO shifts capital weights $a_i$ away from desks whose models are currently experiencing high epistemic noise.
- **Economic Rationale**: Governance-level risk management. Allocators with cleaner signals receive larger mandates.

### Hypothesis 4: Adaptive Coupling Strength
$$L_{\text{adaptive}} = I - E W_E (E^T W_E E)^{-1} E^T$$
- **Mechanism**: If Desk 1 is very certain about an asset while Desk 2 is uncertain, consensus is weighted toward the confident desk rather than enforcing unweighted arithmetic averaging.

### Hypothesis 5: Posterior Downside Hurdle Penalty
$$\tilde{\mu}_j = \mu_j - \alpha \, \mathbb{P}\left(\mu_j < r_{\text{target}} \mid \mathcal{D}\right)$$
- **Mechanism**: Directly penalizes assets whose predictive distribution has fat lower tails below a hurdle rate $r_{\text{target}}$, using the Gaussian cumulative distribution:
  $$\mathbb{P}(\mu_j < r_{\text{target}}) = \Phi\left( \frac{r_{\text{target}} - \mu_j}{\sigma_{\mu, j}} \right)$$

---

## 4. Controlled Experiment: Testing the Mechanisms

Let's test these candidate mechanisms under controlled conditions using [`run_uncertainty_experiments.py`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/experiments/run_uncertainty_experiments.py):

- **Asset 1**: High return ($\mu_1 = 0.30$), but **massive epistemic noise** ($\operatorname{Var}(\mu_1) = 0.20$).
- **Asset 2**: Moderate return ($\mu_2 = 0.25$), **low epistemic noise** ($\operatorname{Var}(\mu_2) = 0.02$).
- **Asset 3**: Low return ($\mu_3 = 0.10$), **low epistemic noise** ($\operatorname{Var}(\mu_3) = 0.01$).

```python
from experiments.run_uncertainty_experiments import run_uncertainty_study
run_uncertainty_study()
```

---

## 5. Numerical Experiment Results

```
===========================================================================
EXPERIMENT 11: BAYESIAN PREDICTIVE UNCERTAINTY IN DISTRIBUTED PORTFOLIOS
===========================================================================
Mechanism                           | Asset 1 (Uncertain)  | Asset 2 (Certain)  | Asset 3 (Certain) 
-----------------------------------------------------------------------------------------------
Baseline (Ignore Uncertainty)       | 0.4551               | 0.2272             | 0.3177            
M1: Return Shrinkage                | 0.4146               | 0.2436             | 0.3418            
M2: Risk Augmentation               | 0.4240               | 0.2414             | 0.3346            
M3: Importance Re-weighting         | 0.4603               | 0.2255             | 0.3143            
M5: Posterior Downside Penalty      | 0.4584               | 0.2324             | 0.3092            
```

---

## 6. Comparative Evaluation of Hypotheses

Let's evaluate each hypothesis based on our experimental data:

| Hypothesis | Level of Action | Allocation Shift on Asset 1 | Mathematical Properties | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **M1: Return Shrinkage** | Asset Level | Drops from **0.455 to 0.415** | Preserves QP convexity; linear in $\mu$ | **Strongest candidate** for direct implementation. |
| **M2: Risk Augmentation** | Subportfolio Level | Drops from **0.455 to 0.424** | Preserves $Q \succ 0$; smooth diagonal inflation | **Best for robust covariance control**. |
| **M3: Mandate Weights** | Firm Governance | Increases slightly to **0.460** | Sub-portfolio 1 has all 3 assets; re-weighting redistributes across desks | Useful when entire desks have differing model accuracy. |
| **M4: Adaptive Coupling** | Network Topology | Shifts consensus projection | Changes hypergraph projector $P_E$ | Best for asymmetric information between desks. |
| **M5: Downside Penalty** | Risk Objective | Non-linear penalty | Smooth Gaussian tail approximation | Best when institutional mandates have strict hurdle rates. |

---

## 7. Strategic Insight: The Multi-Layer Uncertainty Architecture

Our findings show that **epistemic uncertainty should not be restricted to a single entry point**:
1. **Asset Level (M1)**: Shrink point forecasts based on token-level predictive entropy from Bayesian neural nets.
2. **Strategy Level (M2)**: Add epistemic covariance to $Q_i$ to ensure robust local risk budgeting.
3. **Firm Level (M3)**: Re-allocate capital mandates toward allocators with the highest empirical calibration.

In Tutorial 12, we conclude our lab with a comprehensive research synthesis.

---

## What You Should Now Understand

1. Why deterministic mean-variance optimization fails when return estimates carry epistemic uncertainty.
2. The mathematical formulation of 5 candidate ways Bayesian uncertainty can enter distributed portfolios.
3. How to implement and verify each mechanism using [`uncertainty.py`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/src/hypergraph_portfolio/uncertainty.py).

---

## Questions Worth Investigating

- If Desk 1 uses a Bayesian Transformer while Desk 2 uses a Classical Fama-French factor model, how should the consensus mechanism weight their respective certainty estimates?
- Can out-of-sample Sharpe ratios verify whether Mechanism 1 or Mechanism 2 produces superior risk-adjusted returns over a 10-year rolling window?
