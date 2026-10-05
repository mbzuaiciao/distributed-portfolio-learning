# Phase 20A: Analytical Derivation of the Optimal Cohort Holding Horizon

## 1. Executive Summary

This document formalizes the mathematical theory of cohort stacking as a **temporal regularization mechanism**. We derive a minimal analytical model of the fundamental tradeoff governing cohort horizon selection:

$$\max_{H \ge 1} U(H) = \text{Alpha}(H) - \text{Cost}(H) - \gamma \cdot \text{Risk}(H)$$

where:
- **Alpha Dilution**: Holding stale cohorts dilutes fast-decaying predictive information ($\partial \text{Alpha}/\partial H \le 0$).
- **Cost Amortization**: Rolling cohorts over horizon $H$ reduces portfolio turnover proportionally to $1/H$ ($\partial \text{Cost}/\partial H < 0$).
- **Temporal Diversification**: Stacking overlapping entry cohorts averages independent estimation noise and idiosyncratic return shocks ($\partial \text{Risk}/\partial H \le 0$).

This framework explains the Phase 19 empirical reversal mechanistically: under weak signals with rebalancing churn ($\mu_0 \to 0, \text{TO}_0 > 0$), alpha dilution vanishes, allowing mechanical cost amortization and temporal diversification to dominate and make longer horizons ($H=4$) optimal. (Under an exact time-invariant uniform null where $\text{TO}_0 = 0$, all horizons tie exactly with $\Delta U \equiv 0$). Conversely, under strong, rapidly decaying signals, alpha preservation forces the optimal horizon to $H=1$.

---

## 2. Model Setup

Consider a discrete-time rolling cohort portfolio where at each week $t$, a new cohort is initiated with target holding horizon $H \in \{1, 2, \dots\}$.

### 2.1 Signal Generation and Decay
At cohort entry ($k=0$), the portfolio model assigns asset weights based on an observed predictive ranking signal. The expected return conditional on the entry signal decays exponentially with holding age $k \ge 0$:

$$\mu_k = \mu_0 \rho^k = \mu_0 e^{-k / \tau}$$

where:
- $\mu_0 > 0$ is the initial conditional expected return (alpha) at entry age $k=0$;
- $\rho \in [0, 1)$ is the weekly persistence coefficient;
- $\tau = -1 / \ln(\rho)$ is the continuous-time characteristic decay time;
- $\tau_{1/2} = \frac{\ln(0.5)}{\ln(\rho)}$ is the signal half-life in weeks.

When $\rho = 0$ ($\tau_{1/2} = 0$), the signal is purely transient (1-week impulse). As $\rho \to 1$, the signal is permanent.

### 2.2 Expected Signal Contribution over Horizon $H$
In steady state, the cohort-stacked portfolio allocates equal capital fraction $1/H$ across the $H$ active cohorts of ages $k \in \{0, 1, \dots, H-1\}$. The aggregate expected gross return is:

$$\text{Alpha}(H) = \frac{1}{H} \sum_{k=0}^{H-1} \mu_k = \frac{\mu_0}{H} \sum_{k=0}^{H-1} \rho^k = \begin{cases} \mu_0 \cdot \frac{1 - \rho^H}{H(1 - \rho)}, & \text{if } \rho < 1 \\ \mu_0, & \text{if } \rho = 1 \end{cases}$$

**Proposition 1 (Alpha Dilution)**: For any fixed $\rho \in [0, 1)$, $\text{Alpha}(H)$ is strictly decreasing in $H$.
*Proof*: Let $f(H) = \frac{1 - \rho^H}{H}$. The continuous derivative with respect to $H$ is:
$$\frac{df}{dH} = \frac{-H \rho^H \ln(\rho) - (1 - \rho^H)}{H^2} = \frac{\rho^H ( -H \ln\rho + 1) - 1}{H^2}$$
Since $\ln(1/\rho) > 0$, let $x = H \ln(1/\rho) > 0$. The numerator is $e^{-x}(x + 1) - 1$. Since $e^x > 1 + x$ for all $x > 0$, $e^{-x}(x+1) < 1$, so the numerator is strictly negative for all $H \ge 1, \rho < 1$. Hence $\frac{\partial \text{Alpha}}{\partial H} < 0$. $\blacksquare$

### 2.3 Turnover and Transaction Cost
At each period, the oldest cohort of age $H-1$ is closed and a new cohort is entered. With $H$ active cohorts, each cohort represents $1/H$ of the portfolio. If the baseline one-way turnover of an un-stacked single cohort is $\text{TO}_0$, the steady-state portfolio turnover scales inversely with horizon:

$$\text{Turnover}(H) \approx \frac{\text{TO}_0}{H}$$

With linear transaction cost rate $c$ per unit of traded turnover:

$$\text{Cost}(H) = c \cdot \text{Turnover}(H) = \frac{c \cdot \text{TO}_0}{H}$$

**Proposition 2 (Cost Amortization)**: $\text{Cost}(H)$ is strictly decreasing and convex in $H$, with marginal cost savings $\Delta \text{Cost} = c \cdot \text{TO}_0 \left(\frac{1}{H_1} - \frac{1}{H_2}\right) > 0$ for $H_2 > H_1$.

### 2.4 Variance and Cohort Diversification Under Shared Calendar Asset Returns
In accordance with realistic financial markets, each asset $i$ has **exactly one realized return per calendar week $u$**:
$$R_{i, u} = \bar{\mu}_{i, u} + \epsilon_{\text{sys}, u} + \xi_{i, u}$$
where $\epsilon_{\text{sys}, u} \sim \mathcal{N}(0, \sigma_{\text{sys}}^2)$ is a market-wide factor, and $\xi_{i, u} \sim \mathcal{N}(0, \sigma_{\text{idio}}^2)$ is the asset-specific calendar return shock. Crucially, there are **no cohort-specific idiosyncratic shocks**: any active cohort holding asset $i$ at week $u$ receives the identical return $R_{i, u}$.

The realized portfolio return at calendar week $u$ is the inner product of the stacked weights and the market return vector:
$$R_u = \sum_{i=1}^N w_{u, i}^{\text{stacked}} R_{i, u} = (w_u^{\text{stacked}})^\top R_u = \text{Alpha}(H) + \epsilon_{\text{sys}, u} + \sum_{i=1}^N w_{u, i}^{\text{stacked}} \xi_{i, u}$$

The portfolio variance conditional on weights is:
$$\text{Var}(R_u; H) = \sigma_{\text{sys}}^2 + \sigma_{\text{idio}}^2 \sum_{i=1}^N (w_{u, i}^{\text{stacked}})^2 = \sigma_{\text{sys}}^2 + \frac{\sigma_{\text{idio}}^2}{N_{\text{eff}}(H)}$$
Because stacked weights $w_u^{\text{stacked}} = \frac{1}{H} \sum_{k=0}^{H-1} w_{u-k}$ average asset allocations across $H$ independent entry cohorts with pairwise asset overlap $\psi \in [0, 1)$, the inverse Herfindahl index scales as:
$$\sum_{i=1}^N (w_{u, i}^{\text{stacked}})^2 \approx \frac{1}{N_0} \left[ \psi + \frac{1 - \psi}{H} \right]$$
where $N_0$ is the effective holding count of a single cohort. Thus:
$$\text{Var}(R_u; H) = \sigma_0^2 + \frac{\sigma_1^2}{H}$$
where $\sigma_0^2 = \sigma_{\text{sys}}^2 + \frac{\psi \sigma_{\text{idio}}^2}{N_0}$ is the systematic and core overlap variance floor, and $\sigma_1^2 = \frac{(1 - \psi)\sigma_{\text{idio}}^2}{N_0} > 0$ is the temporal diversification benefit achieved by averaging entry weights across cohorts.

### 2.5 Risk-Adjusted Expected Utility
Under standard mean-variance utility with risk-aversion coefficient $\gamma \ge 0$:
$$U(H) = \text{Alpha}(H) - \text{Cost}(H) - \frac{1}{2}\gamma \cdot \text{Var}(R_t; H)$$
Substituting our expressions:
$$U(H) = \mu_0 \frac{1 - \rho^H}{H(1 - \rho)} - \frac{c \cdot \text{TO}_0}{H} - \frac{1}{2}\gamma \left(\sigma_0^2 + \frac{\sigma_1^2}{H}\right)$$
Grouping by $1/H$:
$$U(H) = -\frac{1}{2}\gamma \sigma_0^2 + \frac{1}{H} \left[ \mu_0 \frac{1 - \rho^H}{1 - \rho} - c \cdot \text{TO}_0 - \frac{1}{2}\gamma \sigma_1^2 \right]$$

---

## 3. Comparative Statics and Optimality Conditions

### 3.1 Pairwise Horizon Comparison: $H$ vs $H=1$
To determine when holding horizon $H > 1$ is preferred to single-week execution $H=1$, consider the utility difference:

$$\Delta U(H, 1) = U(H) - U(1) = \underbrace{\mu_0 \left[ \frac{1 - \rho^H}{H(1 - \rho)} - 1 \right]}_{\le 0 \text{ (Alpha Dilution)}} + \underbrace{\left(1 - \frac{1}{H}\right) \left[ c \cdot \text{TO}_0 + \frac{1}{2}\gamma \sigma_1^2 \right]}_{\ge 0 \text{ (Mechanical Gains)}}$$

Factoring out $(1 - 1/H)$:
$$\Delta U(H, 1) = \left(1 - \frac{1}{H}\right) \left\{ c \cdot \text{TO}_0 + \frac{1}{2}\gamma \sigma_1^2 - \mu_0 \left[ \frac{1 - \frac{1 - \rho^H}{H(1 - \rho)}}{1 - 1/H} \right] \right\}$$

Define the **Effective Information Decay Factor**:
$$\Lambda(H, \rho) \equiv \frac{1 - \frac{1 - \rho^H}{H(1 - \rho)}}{1 - 1/H}$$

Then $H$ is strictly preferred to $H=1$ if and only if:
$$\boxed{c \cdot \text{TO}_0 + \frac{1}{2}\gamma \sigma_1^2 > \mu_0 \Lambda(H, \rho)}$$

### 3.2 Monotonic Comparative Statics

1. **Transaction Cost ($c$)**:
   $$\frac{\partial \Delta U(H, 1)}{\partial c} = \left(1 - \frac{1}{H}\right) \text{TO}_0 > 0$$
   As transaction costs increase, mechanical friction penalizes fast rebalancing. **The optimal horizon $H^*$ weakly increases with $c$.**

2. **Signal Half-Life / Persistence ($\rho$ or $\tau$):**
   $$\frac{\partial \Lambda(H, \rho)}{\partial \rho} < 0 \implies \frac{\partial \Delta U(H, 1)}{\partial \rho} > 0$$
   When $\rho \to 1$, $\Lambda(H, \rho) \to 0$. In the limit of permanent alpha, there is zero dilution penalty: $\Delta U(H, 1) = (1 - 1/H)[c \text{TO}_0 + \frac{1}{2}\gamma \sigma_1^2] > 0$, so $H^*$ takes the maximum available horizon. **The optimal horizon $H^*$ weakly increases with signal half-life.**

3. **Short-Lived Alpha ($\mu_0$ high, $\rho \approx 0$):**
   When $\rho = 0$, $\frac{1 - \rho^H}{H(1-\rho)} = \frac{1}{H}$, so $\Lambda(H, 0) = \frac{1 - 1/H}{1 - 1/H} = 1.0$.
   The break-even condition becomes:
   $$c \cdot \text{TO}_0 + \frac{1}{2}\gamma \sigma_1^2 > \mu_0$$
   If initial alpha is strong ($\mu_0 > c \text{TO}_0 + \frac{1}{2}\gamma \sigma_1^2$), then $\Delta U(H, 1) < 0$ for all $H > 1$. **$H=1$ strictly dominates when fresh alpha exceeds total rebalancing friction.**

4. **Signal Noise ($\sigma_{\text{noise}}$):**
   Let the observed entry signal be $\hat{s} = s + \sigma_{\text{noise}}\eta$. The portfolio expected alpha scales with signal-to-noise ratio:
   $$\mu_0(\sigma_{\text{noise}}) = \frac{\mu_{\text{clean}}}{\sqrt{1 + \sigma_{\text{noise}}^2}}$$
   As $\sigma_{\text{noise}} \to \infty$, $\mu_0 \to 0$. In this limit:
   $$\lim_{\sigma_{\text{noise}} \to \infty} \Delta U(H, 1) = \left(1 - \frac{1}{H}\right) \left[ c \cdot \text{TO}_0 + \frac{1}{2}\gamma \sigma_1^2 \right] > 0$$
   **As signal noise increases, the alpha penalty vanishes, and longer horizons dominate through turnover reduction and variance suppression.**

5. **Risk Aversion ($\gamma$):**
   $$\frac{\partial \Delta U(H, 1)}{\partial \gamma} = \frac{1}{2}\left(1 - \frac{1}{H}\right)\sigma_1^2 > 0$$
   Increasing risk aversion places greater weight on the variance reduction offered by overlapping entry cohorts. **The optimal horizon $H^*$ weakly increases with risk aversion $\gamma$.**

6. **Signal Strength ($\beta$):**
   Since $\mu_0 \propto \beta$, the derivative of the horizon utility delta with respect to signal strength is:
   $$\frac{\partial \Delta U(H, 1)}{\partial \beta} = \left[ \frac{1 - \rho^H}{H(1 - \rho)} - 1 \right] \frac{\partial \mu_0}{\partial \beta} \le 0 \quad (\text{for } \rho < 1)$$
   As predictive signal strength $\beta$ increases when persistence is finite ($\rho < 1$), the absolute penalty of holding stale cohorts scales up linearly. **Higher signal strength $\beta$ shifts preference toward shorter horizons ($H=1$ or $H=2$).**
   Conversely, when predictive alpha is weak but rebalancing churn exists ($\beta \to 0, \text{TO}_0 > 0$, as in Phase 19 random permutations), the alpha dilution penalty is vanishingly small, so regularizer savings ($c \cdot \text{TO}_0 + \frac{1}{2}\gamma \sigma_1^2$) dominate, selecting $H \ge 4$. If the null regime is implemented as time-invariant equal weighting ($w_t \equiv 1/N$, as in Phase 20's $\beta = 0$), rebalancing turnover is identically zero ($\text{TO}_0 = 0$), causing all horizons to tie mathematically with $\Delta U \equiv 0$.

---

## 4. Break-Even Boundaries

Setting $\Delta U(H, 1) = 0$ yields closed-form break-even frontiers:

1. **Break-Even Cost Boundary**:
   $$c^*(H; \rho, \mu_0, \gamma) = \frac{\mu_0 \Lambda(H, \rho) - \frac{1}{2}\gamma \sigma_1^2}{\text{TO}_0}$$
   If actual transaction cost $c > c^*$, horizon $H$ outperforms $H=1$.

2. **Break-Even Alpha Boundary**:
   $$\mu_0^*(H; \rho, c, \gamma) = \frac{c \cdot \text{TO}_0 + \frac{1}{2}\gamma \sigma_1^2}{\Lambda(H, \rho)}$$
   For any signal with initial alpha $\mu_0 < \mu_0^*$, horizon $H$ outperforms $H=1$.
