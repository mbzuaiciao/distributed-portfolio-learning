# Temporal Regularization in Portfolio Execution: Optimal Holding Horizons Under Signal Decay and Trading Friction

**Working Paper**  
**October 2026**

---

## Abstract

Quantitative investment strategies that translate periodic predictive signals into target asset weights face a fundamental execution tradeoff: continuously rebalancing to follow fresh forecasts captures peak predictive accuracy, but generates heavy portfolio turnover, execution transaction costs, and concentration in transient estimation noise. Overlapping cohort portfolios—in which capital is partitioned across multiple sub-portfolios staggered over holding horizon $H$—offer a structural smoothing mechanism. In this paper, we formalize cohort stacking as an architectural **temporal regularization mechanism** that balances responsiveness to fresh predictive information against execution friction and estimation noise. We derive a stylized analytical model of expected utility over holding horizon $H$, establishing closed-form comparative statics and break-even boundaries for the utility-maximizing horizon within a discrete candidate set $H \in \{1, 2, 4, 8, 13\}$. To evaluate these frontiers without historical data distortions, we execute a preregistered synthetic market simulation across 360,000 parameter-seed evaluations under a controlled data-generating process (DGP) where all active cohorts share the single realized market return of each asset per calendar week. All five preregistered comparative static hypotheses are confirmed: the simulated utility-maximizing horizon $H^*$ increases monotonically with transaction costs $c$ (100% simulation consistency), signal persistence $\rho$ (90% consistency), observation noise $\sigma_{\text{noise}}$ (100% consistency), and risk aversion $\gamma$ (100% consistency), while strong, short-lived alpha favors shorter, more responsive horizons ($H^*=1$; 100% consistency). Accounting closure holds to machine precision ($|\text{error}| \le 3.14\times 10^{-17}$), and an audit of the exact-null regime ($\beta=0$) confirms that all horizons tie identically under time-invariant uniform weights ($0.00\text{e}+00$ discrepancy across seven invariants). Finally, we compare earlier retrospective post-earnings results with the synthetic frontier. The historical behavior is consistent with a regime in which execution friction and noisy signals make temporal smoothing valuable, but this mapping is descriptive rather than an identified causal or structural estimate. We emphasize that Phase 20 is a controlled synthetic investigation and Phases 18–19 provide retrospective empirical motivation; this study does not establish independent prospective or live-trading performance.

---

## 1. Introduction

A central dilemma in systematic asset management is managing the tension between forecast freshness and portfolio implementation drag. Machine learning models, factor scoring systems, and fundamental analysts frequently generate refreshed cross-sectional forecasts at high frequencies (e.g., daily or weekly). When a Markowitz-style mean-variance (Markowitz, 1952) or rank-based optimizer translates these signals into target portfolios, the resulting asset weights can exhibit substantial instability. Rebalancing to track the latest forecast incurs heavy portfolio turnover, elevated transaction costs, and vulnerability to transient estimation noise.

To dampen this churn, quantitative practitioners often employ heuristic execution rules, including trade bands, turnover penalties, or holding-period constraints. One practical execution rule is the **overlapping cohort portfolio** (or "cohort stacking"), in which staggered sub-portfolios are held concurrently. Under cohort stacking with holding horizon $H$, capital is partitioned into $H$ independent sub-portfolios. At each rebalancing date $t$, the oldest sub-portfolio (initiated at date $t-H$) is liquidated, and a fresh sub-portfolio is formed based on the latest signal. The aggregate portfolio is the uniform average of all $H$ currently active cohorts.

In event-driven applications, improvements from longer holding periods can be tempting to interpret as evidence of persistent predictive alpha. That interpretation is not automatic, because the same holding rule also changes turnover, transaction costs, and diversification across entry dates.

In this paper, we study cohort stacking as an **architectural temporal regularizer**. Even when underlying predictive persistence is negligible or absent, overlapping cohorts mechanically reduce rebalancing turnover by a factor of approximately $1/H$ and diversify idiosyncratic entry-timing variance across active cohorts. However, extending the holding horizon also incurs an inevitable cost: **alpha dilution**. As cohorts age, decaying predictive signals lose forecasting power, dragging down aggregate portfolio gross returns.

We define the **optimal holding horizon** $H^*$ precisely as:
> **The utility-maximizing horizon within the discrete candidate set $H \in \{1, 2, 4, 8, 13\}$, under the specified portfolio formation rule, linear transaction-cost model, and mean-variance objective.**

The central research question we address is:
> **How does the utility-maximizing cohort holding horizon $H^*$ depend on signal persistence ($\rho$), observation noise ($\sigma_{\text{noise}}$), signal strength ($\beta$), transaction friction ($c$), and portfolio risk aversion ($\gamma$)?**

We address this question through a unified three-part contribution:
1. **Analytical Utility Model**: We construct a stylized expected utility framework formalizing the triad of economic forces—**alpha dilution**, **cost amortization**, and **cross-cohort temporal diversification**—deriving closed-form comparative statics and break-even boundaries.
2. **Controlled Synthetic Simulation (Phase 20)**: Under a preregistered simulation design enforcing a single realized asset return per calendar week shared across active cohorts, we evaluate 360,000 parameter-seed instances. We confirm all five primary comparative statics, establish 2D horizon phase diagrams, verify accounting closure to machine precision ($3.14\times 10^{-17}$), and confirm that time-invariant uniform weights produce an exact mathematical tie across all horizons ($0.00\text{e}+00$ discrepancy).
3. **Retrospective Empirical Reconciliation (Phases 18 & 19)**: We reconcile earlier findings from quarterly corporate earnings announcement portfolios with the controlled model. The observed $H=4$ behavior is consistent with an architectural regularization mechanism driven by turnover reduction and temporal smoothing; the historical evidence does not independently identify that mechanism.

We explicitly delineate the boundaries of our investigation: Phase 20 is a controlled synthetic simulation under stylized linear-Gaussian assumptions; Phases 18 and 19 provide retrospective empirical motivation and sanity checks on historical holdout data (terminating May 15, 2025); and no independent prospective live trading execution is evaluated.

---

## 2. Related Setting and Evolutionary Motivation

This investigation arose from a broader multi-phase empirical laboratory studying distributed portfolio optimization, execution stability, and machine learning uncertainty quantification (Papastaikoudis et al., 2025; Blundell et al., 2015; Hu et al., 2022). 

Initially, our research evaluated whether Bayesian Low-Rank Adaptation (LoRA) could provide actionable epistemic uncertainty estimates to guide portfolio risk management in financial event forecasting. While preliminary experiments appeared to show strong error-tracking correlations ($\rho \approx 0.60$) and promising selective-risk retention curves, comprehensive forensic audits revealed that these early results were driven by methodological artifacts:
- **Target Outcome Leakage**: Historical text serialization pipelines had inadvertently concatenated continuous post-event price realizations (`move_1d`) into input text, allowing models to extract labels during training.
- **Reliability Metric Distortion**: The apparent rank correlation between predictive entropy and absolute prediction error was mathematically proven to be an algebraic artifact of 80/20 class imbalance under probability shrinkage, mechanically forcing within-class correlations to $\pm 1.0000$.
- **Baseline Equivalence**: When retrained on clean metadata, Bayesian LoRA, clean deterministic LoRA, and a trivial constant base-rate null achieved practically equivalent scoring-rule performance, while 32-draw Monte Carlo sampling imposed a $30\times$ latency penalty over a single deterministic pass of the posterior mean.
- **Collaborative LoRA**: A pilot testing heterogeneous sector specialists failed continuation criteria as learned weights degenerated to a single specialist.

Concurrently, portfolio-level diagnostics revealed that underlying event-driven portfolios turned over at an extreme rate (~72x/year traded turnover, 104x/year base turnover) due to weekly earnings panel disjointness and optimizer corner solutions. Implementing multi-week cohort stacking ($H=4$) slashed turnover by $74\%$ and volatility by $66\%$, delivering substantial Sharpe ratio improvements.

However, when this cohort architecture was subjected to clean revalidation on uncompromised metadata (Phase 18), residual predictive signal across weeks 2–4 was statistically indistinguishable from zero (residual IC $= 0.0060, p=0.105$). Furthermore, stress-testing across diverse signal regimes (Phase 19) established that turnover and volatility reductions were **architecture-general**, persisting across empirical, random, constant, and synthetic signals. Crucially, Phase 19 revealed a directional reversal: cohort stacking delivered maximum relative utility under weak or null signals, but lost relative advantage as synthetic signal strength increased.

A particularly close theoretical benchmark is Gârleanu and Pedersen (2013), who derive optimal dynamic trading policies when returns are predictable and trading is costly, with slower-decaying predictors receiving greater weight in the dynamic target. Our question is narrower and structurally different: we study a discrete overlapping-cohort execution rule and characterize how its utility-maximizing horizon changes with signal decay, noise, costs, and risk aversion. We therefore interpret the cohort frontier as a complementary execution-design result rather than a replacement for the general dynamic-control solution.

These findings shifted our scientific focus from predictive model complexity to execution architecture: **why does cohort stacking function so effectively under weak signals, and how does the optimal horizon respond to market friction and signal decay?**

---

## 3. Temporal Regularization Framework

### 3.1 Mechanics of Overlapping Cohort Portfolios

Consider a discrete-time investment setting where at each period $t$, a quantitative forecasting model generates an expected return ranking over an asset universe of size $N$. 

Under standard single-horizon rebalancing ($H=1$), the manager constructs a base portfolio weight vector $w_t^{\text{base}} \in \Delta^{N-1}$ and rebalances the entire portfolio each period. The one-period rebalancing turnover is:
$$\text{TO}_t^{H=1} = \frac{1}{2} \| w_t^{\text{base}} - w_{t-1}^{\text{base}} \|_1$$
Under overlapping cohort stacking with horizon $H \in \{1, 2, \dots\}$, capital is partitioned equally across $H$ sub-portfolios. At period $t$, the cohort entered at $t-H$ is liquidated, and a new cohort with target weights $w_t^{\text{base}}$ is initiated. The aggregate portfolio weight vector $w_t^H$ is:
$$w_t^H = \frac{1}{H} \sum_{k=0}^{H-1} w_{t-k}^{\text{base}}$$
Assuming stationary rebalancing dynamics, the turnover of the aggregate portfolio scales inversely with the horizon:
$$\text{TO}_t^H = \frac{1}{2} \| w_t^H - w_{t-1}^H \|_1 = \frac{1}{2H} \| w_t^{\text{base}} - w_{t-H}^{\text{base}} \|_1 \approx \frac{\text{TO}_0}{H}$$
where $\text{TO}_0$ denotes the characteristic one-period base turnover.

### 3.2 Analytical Stylized Utility Model

Let predictive signal quality decay geometrically with holding age $k \ge 0$:
$$\mu_k = \mu_0 \rho^k = \mu_0 e^{-k / \tau}$$
where $\mu_0 \ge 0$ is initial gross alpha at entry age $k=0$, $\rho \in [0, 1)$ is the one-period signal persistence parameter, $\tau = -1/\ln(\rho)$ is the characteristic decay time, and $\tau_{1/2} = \ln(0.5)/\ln(\rho)$ is the signal half-life.

The expected return captured by a cohort of age $k$ is $\mu_k$. The expected gross alpha captured by an aggregate portfolio of horizon $H$ is the average over all active cohort ages:
$$\bar{r}_{\text{gross}}(H) = \frac{1}{H} \sum_{k=0}^{H-1} \mu_0 \rho^k = \begin{cases} \mu_0 \frac{1 - \rho^H}{H(1 - \rho)}, & \text{if } \rho < 1 \\ \mu_0, & \text{if } \rho = 1 \end{cases}$$
The portfolio incurs linear transaction friction with cost rate $c$ per unit of traded turnover:
$$\text{Cost}(H) = c \cdot \text{TO}(H) \approx \frac{c \cdot \text{TO}_0}{H}$$
The aggregate portfolio variance decomposes into market-wide systematic variance $\sigma_0^2$ and cohort-specific estimation/timing variance $\sigma_1^2$. Because active cohorts are initiated at staggered dates with independent estimation noise, cross-cohort averaging diversifies cohort-specific variance:
$$\sigma^2(H) \approx \sigma_0^2 + \frac{\sigma_1^2}{H}$$
Combining these components under a mean-variance utility objective with risk aversion parameter $\gamma \ge 0$, we define the stylized expected utility:
$$U(H) = \mu_0 \frac{1 - \rho^H}{H(1 - \rho)} - \frac{c \cdot \text{TO}_0}{H} - \frac{1}{2}\gamma \left(\sigma_0^2 + \frac{\sigma_1^2}{H}\right)$$

### 3.3 The Three Core Economic Forces

The derivative of utility with respect to horizon $H$ highlights the three fundamental forces of temporal regularization:
1. **Alpha Dilution ($\partial \text{Alpha}/\partial H \le 0$)**: As $H$ increases, older cohorts with decayed predictive signals dilute the fresh alpha of newly entered cohorts. For any $\rho < 1$, alpha dilution strictly penalizes longer horizons and favors $H=1$.
2. **Cost Amortization ($\partial \text{Cost}/\partial H < 0$)**: Rolling $1/H$ of the portfolio amortizes turnover friction proportionally to $1/H$. This force strictly rewards longer horizons whenever $\text{TO}_0 > 0$ and $c > 0$.
3. **Temporal Diversification ($\partial \text{Risk}/\partial H \le 0$)**: Overlapping cohorts average transient estimation errors and entry-timing shocks, reducing aggregate portfolio variance. This force strictly rewards longer horizons whenever $\sigma_1^2 > 0$ and $\gamma > 0$.

### 3.4 Boundary Cases
- **Transient Signals ($\rho = 0$)**: $\bar{r}_{\text{gross}}(H) = \mu_0 / H$. Gross alpha drops as $1/H$.
- **Permanent Signals ($\rho \to 1$)**: $\bar{r}_{\text{gross}}(H) = \mu_0$. Alpha dilution vanishes completely; longer horizons strictly dominate whenever $c > 0$ or $\gamma > 0$.
- **Zero Rebalancing Churn ($\text{TO}_0 = 0, \sigma_1^2 = 0$)**: When base weights are time-invariant (e.g. constant uniform weighting), $U(H) = -\frac{1}{2}\gamma \sigma_0^2$ for all $H$. In this exact-null state, all horizons tie identically. Stacking benefits emerge only when base allocations churn across periods.

---

## 4. Analytical Comparative Statics & Break-Even Frontiers

To characterize utility-maximizing horizon selection, we evaluate the horizon utility differential relative to the single-period baseline $H=1$:
$$\Delta U(H, 1) = U(H) - U(1) = \mu_0 \left[ \frac{1 - \rho^H}{H(1 - \rho)} - 1 \right] + c \cdot \text{TO}_0 \left(1 - \frac{1}{H}\right) + \frac{1}{2}\gamma \sigma_1^2 \left(1 - \frac{1}{H}\right)$$

Factoring out $(1 - 1/H)$:
$$\Delta U(H, 1) = \left(1 - \frac{1}{H}\right) \left\{ c \cdot \text{TO}_0 + \frac{1}{2}\gamma \sigma_1^2 - \mu_0 \Lambda(H, \rho) \right\}$$
where the effective information decay factor is defined as:
$$\Lambda(H, \rho) \equiv \frac{1 - \frac{1 - \rho^H}{H(1 - \rho)}}{1 - 1/H}$$

From this formulation, we analytically derive five primary comparative statics:

### 4.1 Cost Monotonicity (H1)
$$\frac{\partial \Delta U(H, 1)}{\partial c} = \text{TO}_0 \left(1 - \frac{1}{H}\right) > 0 \quad (\forall H > 1, \text{TO}_0 > 0)$$
Higher transaction friction increases the marginal economic value of turnover reduction. Within the candidate set, the utility-maximizing holding horizon $H^*$ is weakly increasing in transaction cost: $\partial H^* / \partial c \ge 0$.

### 4.2 Persistence Monotonicity (H2)
For any active signal ($\mu_0 > 0$), differentiating alpha capture with respect to persistence $\rho$ yields:
$$\frac{\partial}{\partial \rho}\left[ \frac{1 - \rho^H}{1 - \rho} \right] = \frac{1 - H\rho^{H-1} + (H-1)\rho^H}{(1 - \rho)^2} > 0 \quad (\forall H > 1, \rho \in [0, 1))$$
Because $\frac{\partial \Lambda(H, \rho)}{\partial \rho} < 0$, higher persistence strictly reduces the alpha dilution penalty. Thus, the range of horizons over which cost and variance savings dominate expands, establishing $\partial H^* / \partial \rho \ge 0$.

### 4.3 Short-Lived Alpha Reversal (H3)
In the frictionless limit ($c = 0$) under transient signals ($\rho = 0$):
$$\Delta U(H, 1)\big|_{\rho=0, c=0} = \left(1 - \frac{1}{H}\right) \left[ \frac{1}{2}\gamma \sigma_1^2 - \mu_0 \right]$$
For unconstrained alpha ($\mu_0 > \frac{1}{2}\gamma \sigma_1^2$), this differential is strictly negative for all $H > 1$, establishing $H^* = 1$. Furthermore, for any $\rho < 1$:
$$\frac{\partial \Delta U(H, 1)}{\partial \mu_0} = \frac{1 - \rho^H}{H(1 - \rho)} - 1 \le 0$$
As signal strength $\mu_0$ increases, the absolute cost of alpha dilution scales linearly, shifting preference toward shorter, more responsive horizons.

### 4.4 Noise Regularization (H4)
When observed signals contain additive ranking error $\hat{s} = s + \sigma_{\text{noise}}\eta$, initial alpha scales as $\mu_0 \propto 1 / \sqrt{1 + \sigma_{\text{noise}}^2}$. As noise increases, alpha dilution weakens while cross-cohort noise cancellation retains value. Within the controlled model, increasing signal noise shifts the utility-maximizing horizon upward: $\partial H^* / \partial \sigma_{\text{noise}} \ge 0$.

### 4.5 Risk Aversion Preference (H5)
$$\frac{\partial \Delta U(H, 1)}{\partial \gamma} = \frac{1}{2}\sigma_1^2 \left(1 - \frac{1}{H}\right) > 0 \quad (\forall H > 1, \sigma_1^2 > 0)$$
Higher risk aversion increases the utility penalty assigned to portfolio variance, expanding preference for the diversification offered by overlapping cohorts: $\partial H^* / \partial \gamma \ge 0$.

### 4.6 Closed-Form Break-Even Frontiers
Setting $\Delta U(H, 1) = 0$ yields closed-form break-even boundaries:
- **Break-Even Cost Boundary**:
  $$c^*(H) = \frac{\mu_0 \Lambda(H, \rho) - \frac{1}{2}\gamma \sigma_1^2}{\text{TO}_0}$$
  If actual execution cost $c > c^*(H)$, horizon $H$ delivers strictly higher expected net utility than $H=1$.
- **Break-Even Alpha Boundary**:
  $$\mu_0^*(H) = \frac{c \cdot \text{TO}_0 + \frac{1}{2}\gamma \sigma_1^2}{\Lambda(H, \rho)}$$
  For any signal with initial alpha $\mu_0 < \mu_0^*(H)$, horizon $H$ delivers strictly higher net utility than $H=1$.

---

## 5. Controlled Synthetic Experiment Design (Phase 20)

To test these comparative statics rigorously without reliance on historical market anomalies, we implemented a controlled synthetic market simulation.

### 5.1 The Data-Generating Process (DGP)
1. **Asset Universe & Horizon Grid**: $N = 30$ assets evaluated across 156 calendar weeks, with a 52-week latent-signal history used for the finite-memory return construction. Horizons were frozen at $H \in \{1, 2, 4, 8, 13\}$.
2. **Latent Information Process**: At each calendar week $t$, each asset $i$ receives an independent latent predictive factor $s_{i,t} \sim \mathcal{N}(0, 1)$.
3. **Single Realized Market Return Invariance**:
   Crucially, to eliminate cohort-specific return noise or oracle leaks, each asset has **exactly one realized market return per calendar week**, shared by every active cohort holding that asset:
   $$R_{u, i} = \sum_{k=0}^{K} \beta \rho^k s_{i, u-k} + \epsilon_{\text{mkt}, u} + \xi_{u, i}$$
   where $K = 52$ truncates the latent history, $\beta$ governs signal magnitude, $\epsilon_{\text{mkt}, u} \sim \mathcal{N}(0, \sigma_{\text{mkt}}^2)$ with $\sigma_{\text{mkt}} = 0.02$ is a shared market factor shock, and $\xi_{u, i} \sim \mathcal{N}(0, \sigma_{\text{idio}}^2)$ with $\sigma_{\text{idio}} = 0.04$ is an asset-specific calendar return shock.
4. **Finite-Memory Truncation Disclosure**: For the highest persistence level ($\rho = 0.97$), $0.97^{52} \approx 0.2032$. Thus, the 52-week truncation leaves approximately $20\%$ of latent signal weight unmodeled. The $\rho = 0.97$ case is therefore an explicitly finite-memory approximation rather than an asymptotic infinite-history AR(1) process.
5. **Noisy Ranking Signals**: At entry week $t$, portfolio weights are formed by ranking assets on noisy observed signals:
   $$\hat{s}_{i,t} = s_{i,t} + \sigma_{\text{noise}} \eta_{i,t}, \quad \eta_{i,t} \sim \mathcal{N}(0, 1)$$
6. **Exact-Null Baseline**: At $\beta = 0.000$, ranking signals are deactivated, and base portfolio weights are set to uniform equal weighting ($w_{i,t} = 1/N$).

### 5.2 Preregistered Simulation Matrix
The simulation evaluates a complete factorial design:
- Signal Strength: $\beta \in \{0.000, 0.005, 0.010, 0.020\}$
- Persistence: $\rho \in \{0.0, 0.25, 0.50, 0.75, 0.90, 0.97\}$ ($\tau_{1/2} \in [0.0, 22.8]\text{ weeks}$)
- Observation Noise: $\sigma_{\text{noise}} \in \{0.0, 0.5, 1.0, 2.0\}$
- Transaction Costs: $c \in \{0, 5, 10, 25, 50\}\text{ bps}$
- Risk Aversion: $\gamma \in \{0.0, 0.5, 1.0, 2.5, 5.0\}$
- Random Seeds: 30 independent seeds ($S \in \{42, \dots, 71\}$)
- **Total Evaluations**: $4 \times 6 \times 4 \times 5 \times 5 \times 5 \times 30 = 360{,}000$ portfolio-horizon evaluations.

### 5.3 Preregistered Invariants & Tie Handling
For every evaluation, numerical closure was audited against the identity:
$$\Delta U(H_A, H_B) = (\bar{r}_{\text{gross}, A} - \bar{r}_{\text{gross}, B}) - c(\overline{\text{TO}}_A - \overline{\text{TO}}_B) - \frac{1}{2}\gamma(\text{Var}(r_{\text{net}, A}) - \text{Var}(r_{\text{net}, B}))$$
with required tolerance $|\text{error}| < 10^{-12}$.

Statistical ties across seeds were evaluated using two-sided paired $t$-tests ($\alpha = 0.05$). If no horizon significantly dominated all others, the cell was designated an Indeterminate/Tie Zone.

---

## 6. Simulation Results

### 6.1 Confirmation of Comparative Static Hypotheses

Across all 360,000 evaluations, the synthetic simulation confirmed all five preregistered comparative static hypotheses:

| Hypothesis | Description | Predicted Direction | Simulation Consistency | Confirmed? |
| :--- | :--- | :---: | :---: | :---: |
| **H1** | Cost Monotonicity | $\partial H^* / \partial c \ge 0$ | **100.0%** | **YES** |
| **H2** | Persistence Monotonicity | $\partial H^* / \partial \rho \ge 0$ (for $\beta > 0$) | **90.0%** | **YES** |
| **H3** | Short-Lived Alpha Reversal | $H^*=1$ at $\rho=0, c=0$; $\partial H^* / \partial \beta \le 0$ under fast decay | **100.0%** | **YES** |
| **H4** | Noise Regularization | $\partial H^* / \partial \sigma_{\text{noise}} \ge 0$ | **100.0%** | **YES** |
| **H5** | Risk Aversion Preference | $\partial H^* / \partial \gamma \ge 0$ | **100.0%** | **YES** |

### 6.2 Accounting Closure & Numerical Integrity
Across all 288,000 pairwise horizon evaluations ($H \in \{2, 4, 8, 13\}$ vs. $H=1$), numerical closure held to machine precision:
$$\max |\text{error}| = 3.144 \times 10^{-17} \ll 10^{-12}$$
confirming exact algebraic consistency in turnover accounting and utility decomposition.

### 6.3 Exact-Null Baseline Audit ($\beta = 0.000$)
A post-run audit of the exact-null regime verified seven invariants across all 72,000 comparisons against $H=1$:
- Aggregate portfolio weight vector: $\max |\Delta| = 0.00\text{e}+00$
- Weekly gross return: $\max |\Delta| = 0.00\text{e}+00$
- Weekly turnover: $\max |\Delta| = 0.00\text{e}+00$
- Weekly transaction cost: $\max |\Delta| = 0.00\text{e}+00$
- Weekly net return: $\max |\Delta| = 0.00\text{e}+00$
- Mean utility: $\max |\Delta| = 0.00\text{e}+00$
- Variance term: $\max |\Delta| = 0.00\text{e}+00$

Under constant uniform weighting ($w_t = 1/N$), rebalancing turnover is identically zero ($\text{TO} \equiv 0$). Consequently, all active cohorts hold identical weights, and all horizons tie identically ($100\%$ of cells classified as `Tie_1-2-4-8-13`). This audit confirmed that stacking dominance under friction emerges as soon as the portfolio has rebalancing churn (weak alpha or ranking noise), but does not operate on time-invariant allocations.

### 6.4 Four Principal Horizon Regimes

The resulting 2D synthetic horizon phase diagrams reveal four distinct operational regimes:
1. **Short-Horizon Dominance ($H=1$)**:
   - Confined to low-friction ($c \le 5\text{ bps}$), short-lived ($\tau_{1/2} \le 0.5\text{ wks}$), high-alpha ($\beta \ge 0.01$) environments.
   - Fresh alpha decays too quickly to justify holding cohorts past week 1.
2. **Intermediate Regularization ($H=2\text{ to }4$)**:
   - The broad steady-state attractor for institutional parameters ($c \approx 10\text{ bps}$, moderate noise, half-life $1\text{--}6\text{ weeks}$).
   - Cuts turnover by 50–75% and volatility by 30–45% while preserving predictive rank order.
3. **Extended Stacking ($H=8\text{ to }13$)**:
   - Dominates when transaction friction is high ($c \ge 25\text{ bps}$), signals are highly persistent ($\tau_{1/2} > 6\text{ wks}$), or observation noise is severe ($\sigma_{\text{noise}} \ge 2.0$).
4. **Exact-Null Tie Regime ($\beta = 0.000$)**:
   - Zero rebalancing churn yields identical performance across all horizons ($\text{TO} \equiv 0, \Delta U \equiv 0$).

---

## 7. Retrospective Empirical Reconciliation (Phases 18 & 19)

We connect these synthetic findings back to the retrospective event-risk setting that motivated this research. The empirical motivation is related to the long post-earnings-announcement-drift literature (e.g., Bernard and Thomas, 1989), but our clean revalidation does not establish persistent multi-week drift:

In Phase 18, clean revalidation of quarterly post-earnings portfolios on uncompromised metadata exhibited:
- Transaction cost assumption: $c = 10\text{ bps}$
- Empirical Rank IC $\approx 0.05\text{--}0.06$ (used only as a descriptive guide to nearby weak/noisy synthetic regimes; it does not identify $\beta$ or $\sigma_{\text{noise}}$ structurally)
- Estimated weekly persistence: $\rho \approx 0.50\text{--}0.75$ ($\tau_{1/2} \approx 1.0\text{--}2.4\text{ weeks}$)
- Annualized net return: $9.24\%$ ($H=1$) vs $14.71\%$ ($H=4$)
- Annualized net volatility: $19.59\%$ ($H=1$) vs $6.62\%$ ($H=4$) ($-66.2\%$ reduction)
- Annualized traded risky turnover: $73.03$ ($H=1$) vs $19.01$ ($H=4$) ($-73.97\%$ reduction)
- Clean annualized Sharpe ratio: **$0.469$** ($H=1$) to **$2.223$** ($H=4$)
- Mean net weekly utility delta: $+13.804\text{ bp/week}$ (paired 95% CI: $[-54.68, +84.33]\text{ bp/week}$)
- Incremental weeks 2–4 predictive persistence: Rank IC $= 0.0060$, $p = 0.105$ (statistically insignificant)

Using nearby weak/noisy Phase 20 grid cells as a descriptive comparison—not as an estimated structural mapping—the simulated utility-maximizing horizon is typically $H^* \in \{1, 2\}$ at $c = 10\text{ bps}$ and shifts toward $H=4$ or $H=8$ as transaction costs rise to $25\text{--}50\text{ bps}$ or observation noise increases.

Why, then, did the empirical earnings strategy strongly favor $H=4$?
One quantitatively important explanation is **panel churn**: because corporate earnings announcements occur quarterly, the weekly cross-section of reporting companies was 100% disjoint from week to week. Under un-stacked rebalancing ($H=1$), this disjointness forced an extreme baseline turnover of $\sim 72\text{--}104\text{x/year}$. In the analytical utility model, larger base turnover $\text{TO}_0$ increases the marginal value of cost amortization and therefore shifts preference toward longer horizons. This mechanism is consistent with the historical $H=4$ result, but the retrospective data do not identify a unique structural $H^*$.

This resolves the empirical paradox:
- In Phase 18, persistent post-earnings alpha failed to achieve significance ($p = 0.105$), yet $H=4$ delivered substantial Sharpe gains ($0.469 \to 2.223$) and turnover reductions ($-73.97\%$).
- In Phase 19, stress testing proved that turnover and volatility savings occurred even under random permutations and null signals, but relative utility gains vanished under strong synthetic signals.

The synthetic simulation provides a controlled mechanistic account consistent with the historical pattern: **multi-week stacking can act as an architectural regularizer when signal quality is weak and portfolio churn is high**. In noisy, low-alpha, high-churn environments, the turnover and variance savings of overlapping cohorts overwhelm alpha dilution, making multi-week cohort stacking optimal even when underlying predictive signals are weak.

---

## 8. Evolutionary Provenance & Falsification Summary

For scientific transparency, we document how earlier candidate hypotheses were systematically investigated and falsified:

1. **Bayesian Epistemic Uncertainty (Phases 11–16.1)**:
   - Evaluated as an asset-selection and timing signal.
   - Leakage audit revealed training text contaminated with `move_1d`.
   - Entropy-error metric audit showed $\rho \approx 0.60$ was a mechanical artifact of 80/20 class imbalance under probability shrinkage.
   - Clean retraining showed Bayesian LoRA ($\text{Brier} = 0.15987$) was matched by deterministic LoRA ($0.16079$) and a constant base-rate null ($0.16000$).
   - Downstream portfolio mappings established no robust clean economic advantage.
2. **Collaborative LoRA Specialist Weighting (Pilot)**:
   - Evaluated cross-sector specialization with learned coordination weights.
   - Failed continuation criteria as weights collapsed to banking-only selection, failing to outperform global pooled training. Closed as a negative result.
3. **Prospective 2026 Confirmation (Phase 10)**:
   - Exhaustive data audit proved all market returns terminated on May 15, 2025.
   - Adhering to non-manufacturing protocols, prospective confirmation claims were retracted and classified as Outcome B (retrospective only).

These negative results redirected our inquiry away from predictive model complexity and toward portfolio execution mechanics, ultimately yielding the temporal regularization framework.

---

## 9. Limitations & Boundary Conditions

1. **Controlled Synthetic DGP Idealizations**: The Phase 20 simulation assumes linear Gaussian signal decay, stationary return volatility, and uncorrelated asset residual shocks. Real financial markets exhibit volatility clustering, fat tails, jump processes, and correlation breakdown during crises.
2. **Linear Transaction Cost Model**: Costs are modeled as proportional linear frictions ($c \cdot \text{TO}$). In institutional asset management, large orders face nonlinear square-root price impact and temporary liquidity depletion, which would further penalize rapid single-period turnover.
3. **No Liquidity or Capacity Constraints**: The model assumes unconstrained capital allocation at cost rate $c$; in live trading, small-cap asset capacity constraints limit position sizing.
4. **Static Holding Horizons**: The framework evaluates constant holding horizons $H$. In live markets, signal decay rates and market volatility fluctuate dynamically, suggesting that time-varying adaptive horizons could yield additional efficiency.
5. **Discrete Candidate Grid**: Optimization is performed over the frozen grid $H \in \{1, 2, 4, 8, 13\}$ rather than continuous holding durations.
6. **Finite-Memory Truncation**: Truncating latent history at $K = 52$ weeks captures $\approx 80\%$ of asymptotic signal life for $\rho = 0.97$ ($0.97^{52} \approx 0.203$), representing an explicitly finite-memory approximation.
7. **Reconstructed Historical Returns in Empirical Motivation**: Phase 18 used geometrically interpolated event-endpoint returns rather than continuous intraweek market marks, which mechanically smooths volatility between reporting events.
8. **Retrospective Holdout Scope**: All empirical data terminated on May 15, 2025; no independent prospective market data was available.
9. **Absence of Live Execution**: Backtests omit borrow availability, short rebate rates, exchange fee tiers, and real-time execution slippage.
10. **Heuristic Execution Context**: Overlapping cohorts are evaluated as an execution smoothing heuristic; this paper does not claim to solve the dynamic multi-period stochastic control problem.

---

## 10. Conclusion & Design Principles

Overlapping cohort portfolios are not merely an empirical heuristic for capturing prolonged price drift; they are a formal **temporal regularization mechanism**. By pooling staggered sub-portfolios across time, cohort stacking trades responsiveness to fresh signals against execution turnover and estimation noise.

For quantitative practitioners, our findings offer clear architectural design principles:
- **Measure Signal Half-Life Before Setting Horizons**: For transient signals ($\tau_{1/2} < 0.5\text{ weeks}$), stacking beyond $H=1$ severely penalizes performance unless transaction costs exceed $25\text{ bps}$.
- **Use Stacking to Dampen High-Frequency Panel Churn**: In event-driven strategies with highly disjoint cross-sections, cohort stacking can materially reduce turnover; in the retrospective earnings setting studied here, the reduction was about $74\%$ for $H=4$.
- **Recognize the Friction-Decay Frontier**: Within the controlled model and candidate horizon grid, the utility-maximizing horizon generally shifts upward with transaction costs, signal persistence, observation noise, and risk aversion. When alpha is weak and noisy, temporal regularization can be an important implementation lever.


---

## References

Bernard, V. L., & Thomas, J. K. (1989). Post-earnings-announcement drift: Delayed price response or risk premium? *Journal of Accounting Research, 27*, 1–36. https://doi.org/10.2307/2491062

Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty in neural networks. In *Proceedings of the 32nd International Conference on Machine Learning* (PMLR 37, pp. 1613–1622).

Gârleanu, N., & Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *The Journal of Finance, 68*(6), 2309–2340. https://doi.org/10.1111/jofi.12080

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations (ICLR)*.

Markowitz, H. (1952). Portfolio selection. *The Journal of Finance, 7*(1), 77–91. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x

Papastaikoudis, I., Watson, J., & Lestas, I. (2025). Distributed portfolio optimization & decentralized pricing. *Cambridge Working Papers in Economics*, No. 2531, Faculty of Economics, University of Cambridge. https://doi.org/10.17863/CAM.119662
