# Tutorial 41: Temporal Regularization & Optimal Holding Horizon

In quantitative finance, portfolios are typically formed by ranking assets on a predictive signal and allocating capital accordingly. When new signals arrive every period, managers face a dilemma:
- **Trade immediately** on the fresh forecast: captures peak predictive accuracy, but triggers heavy portfolio turnover, concentrated transaction costs, and exposure to transient estimation noise.
- **Hold positions longer**: amortizes trading costs and diversifies noise across time, but suffers information decay (alpha dilution) as older signals grow stale.

**Cohort stacking** (forming overlapping sub-portfolios rolled across horizon $H$) is the natural architectural bridge between these extremes. In this tutorial, we demonstrate that cohort stacking acts as a formal **temporal regularization mechanism** and derive the optimal holding horizon frontier:
$$H^* = f(\tau_{\text{signal}},\ c,\ \sigma_{\text{noise}},\ \gamma,\ \beta)$$

---

## 1. The Core Tradeoff & Intuition

Overlapping cohort portfolios balance three opposing forces:

```
                  ┌────────────────────────────────────────────────────────┐
                  │               TEMPORAL REGULARIZATION                  │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
         ┌───────────────────────────────────┼───────────────────────────────────┐
         ▼                                   ▼                                   ▼
┌──────────────────┐               ┌──────────────────┐                ┌──────────────────┐
│  ALPHA DILUTION  │               │COST AMORTIZATION │                │NOISE DIVERSIFIC. │
│                  │               │                  │                │                  │
│ Holding stale    │               │ Staggered rolling│                │ Overlapping      │
│ cohorts dilutes  │               │ cuts turnover by │                │ cohorts average  │
│ decaying alpha   │               │ factor of 1/H    │                │ estimation noise │
│ (Favors H=1)     │               │ (Favors H ≥ 4)   │                │ (Favors H ≥ 4)   │
└──────────────────┘               └──────────────────┘                └──────────────────┘
```

1. **Fast signals favor fast trading**: If a signal decays rapidly ($\tau_{1/2} < 1\text{ week}$), holding cohorts past week 1 inflicts severe alpha dilution that penalizes utility.
2. **Noisy signals favor temporal smoothing**: If a signal is contaminated by cross-sectional observation noise ($\sigma_{\text{noise}} > 0$), averaging rankings across successive entry cohorts cancels idiosyncratic ranking errors.
3. **Transaction friction rewards patient holding**: If rebalancing incurs transaction costs ($c > 0$), rolling $1/H$ of the portfolio each week amortizes turnover friction proportionally to $1/H$.
4. **Overlapping cohorts diversify timing risk**: Staggered entry schedules smooth out entry-timing variance, reducing aggregate portfolio volatility by $30\text{--}65\%$.

---

## 2. Stylized Analytical Utility Model

Consider an asset universe where a predictive signal generated at entry age $k=0$ decays exponentially with holding age $k \ge 0$:
$$\mu_k = \mu_0 \rho^k = \mu_0 e^{-k / \tau}$$
where $\mu_0 \propto \beta$ is entry signal strength and $\rho \in [0, 1)$ is weekly persistence.

For an overlapping portfolio with holding horizon $H$, aggregate weights are the uniform average across active cohorts:
$$w_t^H = \frac{1}{H} \sum_{k=0}^{H-1} w_{t-k}^{\text{base}}$$

### 2.1 Component Formulations
1. **Expected Gross Alpha**:
   $$\bar{r}_{\text{gross}}(H) = \frac{1}{H} \sum_{k=0}^{H-1} \mu_k = \mu_0 \frac{1 - \rho^H}{H(1 - \rho)}$$
2. **Turnover & Transaction Costs**:
   With base rebalancing turnover $\text{TO}_0$, cohort stacking reduces aggregate traded turnover to:
   $$\text{TO}(H) \approx \frac{\text{TO}_0}{H} \implies \text{Cost}(H) = c \frac{\text{TO}_0}{H}$$
3. **Variance & Risk Penalty**:
   The variance of the overlapping portfolio decomposes into common market risk and diversified cohort-specific tracking error:
   $$\sigma^2(H) \approx \sigma_0^2 + \frac{\sigma_1^2}{H} \implies \text{Risk Penalty} = \frac{1}{2}\gamma \left(\sigma_0^2 + \frac{\sigma_1^2}{H}\right)$$

### 2.2 Net Utility Objective
Combining these components yields the stylized expected utility:
$$U(H) = \mu_0 \frac{1 - \rho^H}{H(1 - \rho)} - c \frac{\text{TO}_0}{H} - \frac{1}{2}\gamma \left(\sigma_0^2 + \frac{\sigma_1^2}{H}\right)$$

---

## 3. Monotonic Comparative Statics

Evaluating the marginal utility of extending horizon $H$ relative to $H=1$:
$$\Delta U(H, 1) = U(H) - U(1) = \mu_0 \left[ \frac{1 - \rho^H}{H(1 - \rho)} - 1 \right] + c \cdot \text{TO}_0 \left(1 - \frac{1}{H}\right) + \frac{1}{2}\gamma \sigma_1^2 \left(1 - \frac{1}{H}\right)$$

From this formulation, we analytically derive five comparative statics:

| Parameter | Economic Factor | Comparative Static | Mechanism |
| :--- | :--- | :---: | :--- |
| **Transaction Cost ($c$)** | Friction | $\frac{\partial \Delta U}{\partial c} = \text{TO}_0 (1 - 1/H) > 0$ | Higher costs reward turnover amortization $\implies \partial H^* / \partial c \ge 0$. |
| **Signal Persistence ($\rho$)** | Information Life | $\frac{\partial \Delta U}{\partial \rho} > 0$ (for $\beta > 0$) | Slower decay reduces alpha dilution penalty $\implies \partial H^* / \partial \rho \ge 0$. |
| **Signal Strength ($\beta$)** | Alpha Value | $\frac{\partial \Delta U}{\partial \beta} \le 0$ (for $\rho < 1$) | Valuable perishable alpha penalizes stale cohorts $\implies \partial H^* / \partial \beta \le 0$. |
| **Signal Noise ($\sigma_{\text{noise}}$)** | Observation Error | $\frac{\partial \Delta U}{\partial \sigma_{\text{noise}}} \ge 0$ | Multi-cohort averaging cancels transient noise $\implies \partial H^* / \partial \sigma_{\text{noise}} \ge 0$. |
| **Risk Aversion ($\gamma$)** | Volatility Penalty | $\frac{\partial \Delta U}{\partial \gamma} = \frac{1}{2} \sigma_1^2 (1 - 1/H) > 0$ | Cross-cohort diversification dampens variance $\implies \partial H^* / \partial \gamma \ge 0$. |

---

## 4. Controlled Synthetic Market Simulation (Phase 20)

To verify whether these analytical frontiers hold under discrete-time market dynamics without historical empirical artifacts, Phase 20 implemented a controlled simulation:

### 4.1 Data-Generating Process (DGP)
1. **Latent Predictive Signal**:
   At each week $t$, asset $i$ receives a latent persistent factor $s_{i,t} \sim \mathcal{N}(0, 1)$.
2. **Single Calendar Realized Return**:
   Crucially, each asset has **exactly one realized market return per calendar week**, shared across all active cohorts holding that asset:
   $$R_{u, i} = \sum_{k=0}^{K} \beta \rho^k s_{i, u-k} + \epsilon_{u, i}, \quad \epsilon_{u, i} \sim \mathcal{N}(0, \sigma_{\epsilon}^2)$$
   This eliminates cohort-specific return noise and prevents oracle information leakage.
3. **Noisy Ranking Signals**:
   Observed signals include additive observation noise: $\hat{s}_{i,t} = s_{i,t} + \sigma_{\text{noise}} \eta_{i,t}$.
4. **Preregistered Evaluation Matrix**:
   - Horizons: $H \in \{1, 2, 4, 8, 13\}$
   - Signal Strength: $\beta \in \{0.000, 0.005, 0.010, 0.020\}$
   - Persistence: $\rho \in \{0.0, 0.25, 0.50, 0.75, 0.90, 0.97\}$ ($\tau_{1/2} \in [0.0, 22.8]\text{ wks}$)
   - Noise: $\sigma_{\text{noise}} \in \{0.0, 0.5, 1.0, 2.0\}$
   - Transaction Costs: $c \in \{0, 5, 10, 25, 50\}\text{ bps}$
   - Risk Aversion: $\gamma \in \{0.0, 0.5, 1.0, 2.5, 5.0\}$
   - 30 independent random seeds $\implies 360{,}000$ evaluations.

---

## 5. Horizon Phase Diagram & Empirical Findings

The synthetic simulation confirmed all five preregistered comparative-static hypotheses:

| Hypothesis | Description | Predicted Direction | Empirical Consistency | Status |
| :--- | :--- | :---: | :---: | :---: |
| **H1** | Cost Monotonicity | $\partial H^* / \partial c \ge 0$ | 100.0% | **CONFIRMED** |
| **H2** | Persistence Monotonicity | $\partial H^* / \partial \rho \ge 0$ (for $\beta > 0$) | 90.0% | **CONFIRMED** |
| **H3** | Short-Lived Alpha Reversal | $H^*=1$ at $\rho=0, c=0$; $\partial H^* / \partial \beta \le 0$ under fast decay | 100.0% | **CONFIRMED** |
| **H4** | Noise Regularization | $\partial H^* / \partial \sigma_{\text{noise}} \ge 0$ | 100.0% | **CONFIRMED** |
| **H5** | Risk Aversion Preference | $\partial H^* / \partial \gamma \ge 0$ | 100.0% | **CONFIRMED** |

### The Four Primary Horizon Regimes:
1. **Short-Horizon Dominance ($H=1$)**:
   - Emerges exclusively when costs are negligible ($c \le 5\text{ bps}$), signals are strong ($\beta \ge 0.01$), and decay is fast ($\tau_{1/2} \le 0.5\text{ wks}$).
2. **Intermediate Regularization ($H=2\text{--}4$)**:
   - The broad steady-state attractor for institutional parameters ($c \approx 10\text{ bps}$, moderate noise, half-life $1\text{--}6\text{ weeks}$).
   - Cuts turnover by 50–75% and volatility by 30–45% while preserving predictive rank order.
3. **Extended Stacking ($H=8\text{--}13$)**:
   - Dominates when transaction friction is high ($c \ge 25\text{ bps}$), signals are long-lived ($\tau_{1/2} > 6\text{ wks}$), or noise is severe ($\sigma_{\text{noise}} \ge 2.0$).
4. **Exact Null Baseline ($\beta = 0.000$)**:
   - Under constant uniform weighting ($w_t = 1/N$), rebalancing turnover is identically zero ($\text{TO} \equiv 0$).
   - All horizons produce identical gross returns, zero costs, and identical net utility ($U(H) \equiv U(1)$), resulting in an exact mathematical tie ($100\%$ classified as `Tie_1-2-4-8-13`).

---

## 6. Connection to Earlier Empirical Results (Phases 18 & 19)

This temporal regularization framework explains the trajectory of our empirical findings:
1. **Phase 18 Reversal**: When evaluated on clean metadata, post-earnings predictive alpha did not persist across weeks 2–4 ($p=0.105$). Yet $H=4$ survived. Why? Because $H=4$ provided mechanical turnover reduction ($-74\%$) and volatility reduction ($-66\%$).
2. **Phase 19 Invariance**: Stress-testing across empirical, random, constant, and synthetic signals confirmed that these turnover and risk savings were architecture-general. Crucially, Phase 19 showed that stacking was most advantageous when signal strength was weak—directly mirroring the $\partial H^* / \partial \beta \le 0$ reversal confirmed in Phase 20.
3. **Parameter Mapping**: Mapping empirical parameters ($c = 10\text{ bps}$, Rank IC $\approx 0.05\text{--}0.06$, $\tau_{1/2} \approx 1\text{--}2.4\text{ wks}$) into the Phase 20 synthetic phase diagram places the empirical strategy directly in the intermediate regularization regime ($H^* \ge 2\text{--}4$). The survival of $H=4$ was the direct mathematical consequence of operating a noisy signal under execution friction.

---

## 7. Practical Implementation Principles

When designing multi-cohort investment strategies:
1. **Measure Signal Half-Life First**: Never choose holding horizons based on gross backtests alone. If $\tau_{1/2} < 1\text{ week}$, stacking beyond $H=2$ will dilute alpha unless transaction costs exceed $25\text{ bps}$.
2. **Audit Base Turnover Churn**: Compute holding overlap between successive rebalancing baskets. If turnover exceeds $50\%$ per period, cohort stacking is an immediate structural dampener.
3. **Recognize Regularization Limits**: Stacking cannot manufacture predictive alpha from zero; under time-invariant equal-weighting, stacking yields no diversification gain. Stacking provides value when there is real rebalancing churn or observation noise to smooth.
