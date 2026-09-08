# Tutorial 19: Epistemic Risk vs. Directional Alpha

In [Tutorial 17](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/tutorials/17_bayesian_decision_value.md), we found that applying symmetric confidence shrinkage to expected returns ($\tilde{\mu}_j = \rho_j \hat{\mu}_j$) destroyed out-of-sample portfolio utility. The model penalized high-uncertainty assets indiscriminately, suppressing profitable winning trades just as aggressively as losing trades.

This tutorial explores **Method 2 (Bayesian Covariance Inflation)**:
> *Does epistemic uncertainty belong in the quadratic risk term rather than the expected-return term?*

---

## 1. Mathematical Formulation

### The Centralized Mean-Variance Problem
Recall the standard Markowitz problem:
$$\min_{w \in \Delta} \frac{1}{2} w^T Q w - \mu^T w$$
where $\Delta = \{w \in \mathbb{R}^m : \sum_j w_j = 1, w_j \ge 0\}$.

### Epistemic Covariance Inflation
Instead of modifying the directional alpha forecast $\hat{\mu}$, we leave expected returns intact:
$$\mu_{\text{used}} = \hat{\mu}$$
and augment the market covariance matrix $Q$ with diagonal epistemic estimation variance:
$$Q_{\text{robust}} = Q_{\text{market}} + \lambda_{\text{cov}} \operatorname{diag}(\sigma_{\text{epi}}^2)$$

where:
- $\sigma_{\text{epi}, j}^2 = \operatorname{Var}(\mu_j \mid \mathcal{D})$ is the epistemic predictive variance for asset $j$;
- $\lambda_{\text{cov}} \ge 0$ is a risk-aversion hyperparameter tuned strictly on development data.

### Structural Properties
1. **Symmetry Preservation**: Because $Q_{\text{market}}$ is symmetric and $\operatorname{diag}(\sigma_{\text{epi}}^2)$ is diagonal, $Q_{\text{robust}}$ is strictly symmetric:
   $$Q_{\text{robust}} = Q_{\text{robust}}^T$$
2. **Positive Semi-Definiteness (PSD)**: For any non-zero vector $z \in \mathbb{R}^m$:
   $$z^T Q_{\text{robust}} z = z^T Q_{\text{market}} z + \lambda_{\text{cov}} \sum_{j=1}^m z_j^2 \sigma_{\text{epi}, j}^2 \ge 0$$
   Since $Q_{\text{market}} \succeq 0$ and $\sigma_{\text{epi}, j}^2 \ge 0$, $Q_{\text{robust}} \succeq 0$.
3. **Identity at $\lambda_{\text{cov}} = 0$**: Exactly reproduces the deterministic baseline.
4. **Off-Diagonal Preservation**: Off-diagonal market correlations and cross-asset covariances remain completely untouched:
   $$(Q_{\text{robust}})_{ij} = (Q_{\text{market}})_{ij} \quad \forall i \ne j$$

---

## 2. Python Implementation

The transformation is implemented in [`src/hypergraph_portfolio/uncertainty.py`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/src/hypergraph_portfolio/uncertainty.py#L89-L134):

```python
import numpy as np

def apply_epistemic_covariance_inflation(
    Q: np.ndarray,
    epistemic_std: np.ndarray,
    lambda_cov: float = 1.0,
) -> np.ndarray:
    """Inflate covariance matrix by adding diagonal epistemic risk."""
    Q = np.asarray(Q, dtype=float)
    epistemic_std = np.asarray(epistemic_std, dtype=float)
    lambda_cov = float(lambda_cov)

    if lambda_cov == 0.0:
        return Q.copy()

    var_epi = epistemic_std ** 2
    Q_robust = Q + lambda_cov * np.diag(var_epi)
    return 0.5 * (Q_robust + Q_robust.T)
```

---

## 3. Development Set Tuning (2022–2023)

To ensure zero test leakage, $\lambda_{\text{cov}}$ was swept across `[0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]` on the 2022–2023 development set:

```
Development Grid Search (Mean Net Utility):
  lambda_cov =   0.0 -> Mean Utility = -0.00437
  lambda_cov =   0.1 -> Mean Utility = -0.00437
  lambda_cov =   1.0 -> Mean Utility = -0.00437
  lambda_cov =  10.0 -> Mean Utility = -0.00437
  lambda_cov =  30.0 -> Mean Utility = -0.00437
  lambda_cov = 100.0 -> Mean Utility = -0.00436  <-- Optimal
```
The parameter $\lambda_{\text{cov}}^* = 100.0$ was selected and frozen prior to holdout evaluation.

---

## 4. Out-of-Sample Empirical Results (Holdout 2024–2025)

We evaluated M2 (Covariance Inflation with $\lambda_{\text{cov}}^* = 100.0$) against M0 (Deterministic) across three random initializations:

| Metric | M0 (Deterministic) | M2 (Covariance Inflation) | Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Annualized Return** | $43.80\% \pm 6.59\%$ | $41.99\% \pm 6.31\%$ | $-1.81\%$ |
| **Annualized Volatility** | $28.87\% \pm 1.12\%$ | $28.50\% \pm 1.07\%$ | $-0.37\%$ |
| **Sharpe Ratio** | $1.519 \pm 0.227$ | $1.475 \pm 0.219$ | $-0.044$ |
| **Sortino Ratio** | $2.894 \pm 0.488$ | $2.831 \pm 0.461$ | $-0.063$ |
| **Maximum Drawdown** | $-15.12\%$ | $-14.88\%$ | **$+0.24\%$** (mild improvement) |
| **Mean Net Utility** | $-0.00621 \pm 0.00140$ | $-0.00636 \pm 0.00130$ | $-0.00015$ |
| **Bad-to-Good Weight Ratio** | $0.651 \pm 0.056$ | $0.658 \pm 0.056$ | $+0.007$ |

### Paired Block Bootstrap (95% Confidence Intervals)
Using circular block bootstrap ($B = 4$ weeks, 1,000 resamples):
- **Seed 42**: $\Delta U = -0.000011$, 95% CI: $[-0.000372, +0.000374]$ ($t = -0.057, p = 0.968$)
- **Seed 123**: $\Delta U = -0.000142$, 95% CI: $[-0.000795, +0.000318]$ ($t = -0.499, p = 0.704$)
- **Seed 777**: $\Delta U = -0.000302$, 95% CI: $[-0.001011, +0.000158]$ ($t = -0.940, p = 0.482$)

All 95% bootstrap confidence intervals span zero. The performance difference is statistically indistinguishable from zero.

---

## 5. Mechanistic Interpretation

Why did Covariance Inflation (M2) perform better than Symmetric Shrinkage (M1), but still fail to beat Deterministic (M0)?

1. **Relative Alpha Ordering Preserved**:
   Unlike return shrinkage, which compresses $\hat{\mu}_j$ toward zero and can flip the relative attractiveness of assets, covariance inflation preserves the sign and rank ordering of expected returns. The optimizer still knows which asset has the highest point alpha.
2. **Effective Variance Dampening**:
   Since $\sigma_{\text{epi}, j} \approx 0.005 - 0.012$, the epistemic variance $\sigma_{\text{epi}, j}^2 \approx 2.5 \times 10^{-5}$ to $1.4 \times 10^{-4}$. Even with $\lambda_{\text{cov}} = 100.0$, the diagonal addition is only $0.0025$ to $0.014$, compared to asset market variance of $0.040$. This provides a mild diversification effect without completely crippling high-alpha allocations.
3. **Lack of Directional Selectivity**:
   Because covariance inflation is quadratic, it penalizes positive deviations and negative deviations equally. It does not know whether a high-uncertainty asset will produce a large positive surprise or a large negative surprise. Consequently, it slightly improves drawdown ($-14.88\%$ vs $-15.12\%$) by shaving peak risk, but slightly lowers return ($41.99\%$ vs $43.80\%$).
