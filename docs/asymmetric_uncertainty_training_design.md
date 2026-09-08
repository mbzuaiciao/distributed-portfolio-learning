# Research Design: Asymmetric Uncertainty Training & Directional Loss Formulation

## 1. Executive Problem Statement

Across Phases 3, 4, and 5 of this laboratory, empirical investigations have revealed a profound structural limitation in standard Bayesian deep learning applied to financial asset allocation:

1. **Informational Validity**: Bayesian-LoRA epistemic uncertainty contains genuine, high-fidelity signal about the model's future absolute forecast error ($\text{Spearman } \rho \approx 0.55$, $\text{AUROC } \approx 0.82$).
2. **Economic Neutrality / Failure in Selection**: In cross-sectional selection (Phases 3 & 4), symmetric confidence shrinkage and covariance inflation fail to improve out-of-sample portfolio utility or Sharpe ratio.
3. **Structural De-risking in Exposure Timing**: In portfolio-level exposure scaling (Phase 5), aggregate epistemic uncertainty dampens portfolio volatility and slashes drawdowns from $-15.1\%$ to $-4.5\%$, but fails to produce statistically significant Sharpe ratio expansion (Outcome D).
4. **Root Cause**: The underlying neural network was trained under a **symmetric regression loss** (Mean Squared Error / Gaussian Negative Log-Likelihood):
   $$\mathcal{L}_{\text{sym}}(\theta) = \frac{1}{N} \sum_{i=1}^N \left(y_i - f_\theta(x_i)\right)^2$$
   Under symmetric training, the epistemic posterior $p(w | \mathcal{D})$ reflects dispersion around the central conditional expectation $\mathbb{E}[y | x]$. A large epistemic standard deviation $\sigma_{\text{epi}}$ signals that the model is uncertain about the *magnitude* of the return. It is equally likely that the true return is a massive positive surprise ($y \gg \hat{\mu}$) or a severe crash ($y \ll \hat{\mu}$).

To convert epistemic uncertainty into genuine economic value, the model itself must be trained with **asymmetric, decision-relevant risk formulations**.

---

## 2. Mathematical Formulations for Asymmetric Bayesian Training

Rather than estimating a symmetric Gaussian posterior over conditional means, we formulate three alternative loss objectives for Bayesian-LoRA training:

### A. Pinball / Quantile Loss (Value at Risk)
For a specified lower risk quantile $\tau \in (0, 0.5)$ (e.g., $\tau = 0.05$ or $\tau = 0.10$):
$$\mathcal{L}_{\text{pinball}}(y, \hat{q}_\tau) = \max\left(\tau (y - \hat{q}_\tau), \, (\tau - 1)(y - \hat{q}_\tau)\right)$$

- If $y < \hat{q}_\tau$ (a downside violation), the penalty is $(1 - \tau) |y - \hat{q}_\tau|$, heavily penalizing severe underestimation of downside risks.
- If $y \ge \hat{q}_\tau$, the penalty is $\tau |y - \hat{q}_\tau|$.
- **Bayesian Posterior Interpretation**: The epistemic uncertainty $\sigma_{\text{epi}, \tau}$ of the quantile prediction measures uncertainty *about the tail boundary itself*, directly informing tail risk.

### B. Conditional Value at Risk (Expected Shortfall) Joint Loss
Following Fissler and Ziegel (2016) and Patton et al. (2019), Expected Shortfall (ES) at level $\alpha$ and Value at Risk (VaR) $v$ can be jointly estimated via a strictly consistent scoring function:
$$\mathcal{L}_{\text{FZ}}(y, v, e) = -\frac{1}{\alpha e} \mathbf{1}_{\{y \le v\}} (v - y) + \frac{v}{e} + \log(-e) - 1$$
where $v = \operatorname{VaR}_\alpha(Y)$ and $e = \operatorname{ES}_\alpha(Y) < 0$.

- Optimizing $\mathcal{L}_{\text{FZ}}$ forces the neural network to explicitly predict downside tail expectation.
- Epistemic uncertainty derived from Bayesian-LoRA ensembles under $\mathcal{L}_{\text{FZ}}$ directly isolates **epistemic uncertainty over extreme crashes**.

### C. Asymmetric Huber / Quadratic Utility Loss
To align training with portfolio utility $U = r - \frac{\gamma}{2} r^2$:
$$\mathcal{L}_{\text{asym}}(y, \hat{y}) = \begin{cases} (1 + \kappa) (y - \hat{y})^2 & \text{if } y < \hat{y} \text{ (over-optimistic forecast)} \\ (1 - \kappa) (y - \hat{y})^2 & \text{if } y \ge \hat{y} \text{ (conservative forecast)} \end{cases}$$
where $\kappa \in [0, 1)$ penalizes downside forecast errors more heavily than missed upside.

---

## 3. Architecture & LoRA Adaptation Strategy

1. **Base Model**: Frozen financial foundation model or tabular encoder.
2. **LoRA Adapters**: Low-rank decomposition matrices $A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{k \times r}$ applied to projection layers.
3. **Stochastic Weights**:
   - Apply Laplace approximation or SWAG (Stochastic Weight Averaging Denser) over the LoRA parameters $\theta_{\text{LoRA}} \sim \mathcal{N}(\mu_{\text{LoRA}}, \Sigma_{\text{LoRA}})$.
4. **Dual Output Head**:
   - Head 1: Conditional mean $\hat{\mu}(x)$ or median $\hat{q}_{0.5}(x)$.
   - Head 2: Downside risk metric $\hat{e}_\tau(x)$ (Expected Shortfall).
   - Head 3: Asymmetric epistemic variance $\sigma_{\text{epi, down}}^2(x)$.

---

## 4. Proposed Validation Protocol

To verify whether asymmetric uncertainty training resolves the Phase 5 limitation:
1. **Downside Selectivity Test**:
   - Measure the AUROC of $\sigma_{\text{epi, down}}$ for detecting the worst 10% weekly return drawdowns. Target: $\text{AUROC} > 0.70$ (vs current $0.54 - 0.60$).
2. **Asymmetric Exposure Attribution**:
   - In holdout simulations, the exposure scale $s_t$ must drop sharply before the worst 10% weeks while remaining elevated ($> 80\%$) during the best 10% weeks.
3. **Sharpe Expansion**:
   - Paired block bootstrap test must show that the 95% confidence interval for $\Delta \text{Sharpe} = \text{Sharpe}_{\text{asym}} - \text{Sharpe}_{M0}$ strictly excludes zero.

This research design bridges the gap between statistical model uncertainty and financial decision utility.
