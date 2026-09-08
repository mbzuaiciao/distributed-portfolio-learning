# Tutorial 14 — The Economic Value of Bayesian Uncertainty

In quantitative finance and machine learning, a recurring thesis is that modern probabilistic models—such as Bayesian neural networks, Monte Carlo dropout, or Bayesian Low-Rank Adaptation (LoRA)—provide predictive uncertainty that should improve portfolio decisions.

However, standard mean-variance optimization treats point forecasts $\hat{\mu}$ as ground truth, leading to Michaud's well-known **"error maximization"** problem: the optimizer aggressively overweights assets whose expected returns happen to be overestimated.

Does epistemic predictive uncertainty $u_t = \operatorname{Var}(\mu_t \mid \mathcal{D})$ add genuine economic value when the point forecast $\hat{\mu}_t$ is held **strictly identical**?

In this tutorial, we present the methodology, mathematics, code, and empirical results of a 30-seed controlled backtest across 5 information calibration regimes.

---

## 1. Pedagogical Progression

This tutorial follows the standard scientific progression:
$$\text{Intuition} \longrightarrow \text{Controlled Setup} \longrightarrow \text{Mathematics} \longrightarrow \text{Code} \longrightarrow \text{Experiment} \longrightarrow \text{Results} \longrightarrow \text{Interpretation}$$

---

## 2. Intuition: Why Point Forecasts Fail

Consider an asset universe with true expected return vector $\mu^*$. An ML model produces an estimate $\hat{\mu} = \mu^* + \epsilon$, where $\epsilon$ represents forecast error.

When an optimizer solves:
$$\max_w \quad \hat{\mu}^T w - \frac{\gamma}{2} w^T Q w$$

The gradient $\nabla_w J = \hat{\mu} - \gamma Q w$ dictates that assets with large positive $\hat{\mu}_j$ receive the largest weights. If asset $j$ has a large error $\epsilon_j > 0$, the optimizer mistakes estimation error for economic alpha.

A Bayesian model provides not just $\hat{\mu}_j = \mathbb{E}[\mu_j \mid \mathcal{D}]$ but also an epistemic uncertainty measure:
$$u_j = \operatorname{Var}(\mu_j \mid \mathcal{D})$$

**The Crucial Question**: If we provide an uncertainty-aware policy with the **exact same** point forecast $\hat{\mu}_t$ as the deterministic policy, can incorporating $u_t$ prevent error maximization and improve out-of-sample realized utility?

---

## 3. The Controlled Experimental Setup

To isolate the marginal value of predictive uncertainty, our experimental design enforces strict counterfactual controls:

### 3.1 Identical Point Forecasts
Both policies observe the identical point return forecast $\hat{\mu}_t$ at each period $t$:
- **Deterministic Policy ($D$)**: Solves the allocation problem using $(\hat{\mu}_t, Q)$.
- **Uncertainty-Aware Policy ($U$)**: Solves the allocation problem using $(\hat{\mu}_t, Q)$ plus the predictive uncertainty vector $u_t$.
- **Omniscient Oracle Policy ($O$)**: Solves the allocation problem using the true latent DGP parameters $(\mu_t^*, Q^*)$.

### 3.2 Five Information Calibration Regimes
We test 5 distinct regimes to determine the boundary conditions under which uncertainty helps or harms:

1. **Well-Calibrated**: Epistemic uncertainty is strongly positively correlated with absolute forecast error:
   $$u_{t, j} \propto |\hat{\mu}_{t, j} - \mu_{t, j}^*| + \eta_t$$
2. **Weakly-Calibrated**: Uncertainty has a weak, noisy correlation with true error (signal-to-noise ratio $\approx 0.5$).
3. **Uninformative**: Uncertainty is independent white noise uncorrelated with error ($u_{t, j} \sim \operatorname{Uniform}$).
4. **Miscalibrated**: Uncertainty is negatively correlated with true error (the model is highly confident when its errors are largest).
5. **Regime Shift**: The market experiences structural breaks at unpredictable times, causing sudden volatility spikes and forecast error surges.

### 3.3 Realistic Economic Friction
- **Transaction Costs**: Linear turnover penalty $\kappa = 10 \text{ bps}$ ($0.0010$) per unit of portfolio rebalancing.
- **Risk Aversion**: Parameter $\gamma = 1.0$.
- **Realized Single-Period Utility**:
  $$U_t = r_t - \frac{\gamma}{2} \cdot \text{risk}_t - \kappa \cdot \|\Delta w_t\|_1$$
  where $r_t = w_t^T R_t$ is realized return, $\text{risk}_t = w_t^T Q_t^* w_t$ is true realized variance, and $\Delta w_t = w_t - w_{t-1}^+$ is turnover.

---

## 4. Mathematical Formulation

### 4.1 Uncertainty Shrinkage Model
We implement uncertainty awareness via Bayesian shrinkage:
$$\tilde{\mu}_{t, j} = \frac{\hat{\mu}_{t, j}}{1 + \lambda_u u_{t, j}}$$

where $\lambda_u \ge 0$ is the uncertainty aversion parameter.

- When asset $j$ has zero uncertainty ($u_{t, j} \to 0$), $\tilde{\mu}_{t, j} \to \hat{\mu}_{t, j}$.
- When asset $j$ has severe uncertainty ($u_{t, j} \gg 0$), $\tilde{\mu}_{t, j}$ is shrunk toward zero, preventing the optimizer from concentrating capital in spurious high-return forecasts.

Equivalently, this is dual to diagonal risk augmentation:
$$\min_w \quad w^T \left( Q + \lambda_u \operatorname{diag}(u_t) \right) w - \hat{\mu}_t^T w \quad \text{s.t.} \quad \mathbf{1}^T w = 1, \quad w \ge 0$$

### 4.2 Economic Regret vs Omniscient Oracle
At each step $t$, the omniscient oracle constructs the optimal benchmark weight:
$$w_t^* = \arg\min_w \left\{ w^T Q^* w - (\mu_t^*)^T w : \mathbf{1}^T w = 1, w \ge 0 \right\}$$

The single-period regret for any policy $w_t$ is defined as the utility deficit relative to the oracle:
$$\text{Regret}_t(w_t) = U_t(w_t^*) - U_t(w_t) \ge 0$$

---

## 5. Python Implementation

The core simulation engine is implemented in `src/hypergraph_portfolio/uncertainty.py`:

```python
from hypergraph_portfolio.uncertainty import (
    SyntheticMarketEnvironment,
    apply_uncertainty_shrinkage,
    run_sequential_portfolio_simulation,
)

# 1. Initialize synthetic market DGP
env = SyntheticMarketEnvironment(
    n_assets=5,
    n_steps=250,
    regime="well_calibrated",
    seed=42,
)

# 2. Run sequential portfolio backtest
results = run_sequential_portfolio_simulation(
    env=env,
    lambda_u=1.0,
    risk_aversion=1.0,
    turnover_penalty=0.0010,  # 10 bps
)

print(f"Deterministic Sharpe: {results['deterministic']['sharpe']:.3f}")
print(f"Uncertainty Sharpe:   {results['uncertainty']['sharpe']:.3f}")
print(f"Oracle Sharpe:        {results['oracle']['sharpe']:.3f}")
```

---

## 6. Multi-Seed Experimental Protocol

To eliminate sample bias and verify statistical significance, we executed:
- **30 independent random seeds** ($0, 1, \dots, 29$) per regime.
- **5 regimes**: `well_calibrated`, `weakly_calibrated`, `uninformative`, `miscalibrated`, `regime_shift`.
- **$T = 250$ trading days** per simulation run.
- **Paired difference testing**: For each seed $s$, we compute:
  $$\Delta \text{Sharpe}_s = \text{Sharpe}_{U, s} - \text{Sharpe}_{D, s}$$
  $$\Delta \text{Utility}_s = \bar{U}_{U, s} - \bar{U}_{D, s}$$
  and evaluate the paired Student's $t$-statistic $t = \frac{\bar{\Delta}}{\text{SE}(\Delta)}$.

The full experiment is reproducible via:
```bash
.venv/bin/python experiments/run_bayesian_allocation_value.py
```

---

## 7. Empirical Results

The multi-seed experiment produced the following standardized metrics across 150 independent simulations:

### Table 1: Multi-Seed Statistical Summary (30 Seeds per Regime)

| Regime | Det Sharpe | Unc Sharpe | Oracle Sharpe | Paired $\Delta$ Sharpe | $t$-stat | Paired $\Delta$ Utility | $t$-stat | Max Drawdown Imp. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Well-Calibrated** | $0.797 \pm 0.300$ | **$0.891 \pm 0.376$** | $1.169$ | **$+0.094 \pm 0.033$** | **$+2.85^{**}$** | **$+0.00114 \pm 0.00037$** | **$+3.08^{**}$** | **$+1.71\%$** ($-24.7\%$ vs $-23.0\%$) |
| **Regime Shift** | $0.808 \pm 0.413$ | **$0.903 \pm 0.418$** | $1.172$ | **$+0.095 \pm 0.031$** | **$+3.06^{**}$** | **$+0.00116 \pm 0.00037$** | **$+3.14^{**}$** | **$+3.26\%$** ($-25.6\%$ vs $-22.3\%$) |
| **Weakly-Calibrated** | $0.797 \pm 0.300$ | $0.768 \pm 0.329$ | $1.169$ | $-0.030 \pm 0.024$ | $-1.22$ | $-0.00041 \pm 0.00028$ | $-1.46$ | $-0.47\%$ ($-24.7\%$ vs $-25.2\%$) |
| **Uninformative** | $0.797 \pm 0.300$ | $0.707 \pm 0.306$ | $1.169$ | **$-0.091 \pm 0.023$** | **$-4.00^{***}$** | **$-0.00115 \pm 0.00028$** | **$-4.05^{***}$** | **$-1.60\%$** ($-24.7\%$ vs $-26.3\%$) |
| **Miscalibrated** | $0.786 \pm 0.338$ | $0.679 \pm 0.329$ | $1.263$ | **$-0.107 \pm 0.024$** | **$-4.47^{***}$** | **$-0.00149 \pm 0.00032$** | **$-4.63^{***}$** | **$-1.19\%$** ($-27.5\%$ vs $-28.7\%$) |

*Significance: $^{**} p < 0.01$, $^{***} p < 0.001$.*

---

## 8. Visual Evidence

The experiment automatically saves high-resolution diagnostic charts to `artifacts/bayesian_allocation_value/`:

### 8.1 Cumulative Utility Over Time
Under both **Well-Calibrated** and **Regime Shift** conditions, the uncertainty-aware portfolio consistently outperforms the deterministic baseline, accumulating higher net economic utility after transaction costs:

- `cumulative_utility_well_calibrated.png`: Shows steady outperformance of the uncertainty-aware trajectory, pulling away from deterministic allocation while tracking the oracle trajectory.
- `cumulative_utility_regime_shift.png`: Demonstrates that during market dislocations (periods of structural break), the deterministic policy suffers severe utility drawdowns due to erroneous bets, whereas uncertainty shrinkage dynamically dampens exposure.

### 8.2 Sharpe Ratio Distribution Across All Regimes
- `sharpe_comparison_all_regimes.png`: Visualizes box plots for all 5 regimes side by side.
  - In `well_calibrated` and `regime_shift`, the uncertainty distribution is shifted decisively higher.
  - In `weakly_calibrated`, distributions overlap substantially.
  - In `uninformative` and `miscalibrated`, the uncertainty distribution collapses below deterministic.

### 8.3 Drawdown & Tail Risk Protection
- `drawdown_comparison_all_regimes.png`: Compares worst-case maximum drawdown across regimes. In `regime_shift`, average maximum drawdown is curtailed from $-25.66\%$ down to $-22.30\%$, representing a $+3.26\%$ risk reduction.

### 8.4 Regret vs Omniscient Oracle
- `regret_distribution_well_calibrated.png`: The distribution of single-period regret $\text{Regret}_t(w)$ for the uncertainty-aware policy has a visibly smaller mean and lighter right tail than the deterministic policy, confirming closer adherence to the oracle benchmark.

### 8.5 Turnover vs Utility Frontier
- `turnover_vs_utility_frontier.png`: Evaluates the efficiency trade-off. Shrinkage acts as a natural stabilizer, preventing erratic rebalancing between noisy forecast steps and conserving transaction costs.

---

## 9. Critical Scientific Interpretation

### 9.1 When Does Uncertainty Add Value?
1. **Error Mitigation (Well-Calibrated)**: When uncertainty accurately tags volatile or noisy assets, shrinkage reduces portfolio variance by preventing extreme long positions on estimation errors. This yields a statistically significant Sharpe boost of $+0.094$ ($p = 0.008$) and utility boost of $+0.00114$ ($p = 0.004$).
2. **Breakout Protection (Regime Shifts)**: During structural transitions, models that register high predictive uncertainty (e.g., ensemble divergence or epistemic variance spikes) automatically pull back risk, cutting maximum drawdowns by $+3.26\%$.

### 9.2 When Does Uncertainty Destroy Value?
1. **Uninformative Uncertainty ($\Delta \text{Sharpe} = -0.091, t = -4.00$)**: If predictive uncertainty is merely uncalibrated white noise, shrinking by $u_t$ penalizes valid alpha signals at random. It distorts the true risk-return tradeoff, increases unnecessary turnover, and hurts net utility.
2. **Miscalibrated Uncertainty ($\Delta \text{Sharpe} = -0.107, t = -4.47$)**: When the model is overconfident on its worst predictions, the uncertainty penalty operates in reverse—under-penalizing the most dangerous assets while penalizing reliable assets.

### 9.3 Practical Takeaways for Quantitative Portfolio Managers
- **Uncertainty is Not Alpha**: Incorporating epistemic predictive uncertainty does not create new returns; it acts as an **adaptive Bayesian regularizer** against error maximization.
- **Calibration is a Mandatory Prerequisite**: Deploying uncertainty-aware allocation without first proving out-of-sample calibration ($u_t \propto \text{Error}_t$) is actively harmful to portfolio performance.
- **Seamless Integration with Decentralization**: Because the shrinkage $\tilde{\mu}_{t, j} = \hat{\mu}_{t, j} / (1 + \lambda_u u_{t, j})$ or risk augmentation $\tilde{Q}_i = Q_i + \lambda_u \operatorname{diag}(u_i)$ modifies local desk parameters prior to hypergraph consensus, this mechanism integrates natively into the paper's continuous-time primal-dual dynamics with zero communication overhead.
