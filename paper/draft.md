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

## 2. Related Literature and Positioning

This paper bridges three distinct literatures: portfolio execution with transaction costs, statistical regularization in machine learning, and empirical factor persistence.

The literature on dynamic trading under transaction frictions traces back to Markowitz (1952) and Merton (1969, 1971). In seminal contributions, Garleanu and Pedersen (2013, 2016) established that when trading incurs quadratic costs, the optimal policy tracks an aim portfolio through an exponential moving average, balancing alpha capture against execution drag. Overlapping cohort portfolios represent a discrete-time structural variant of this principle, achieving turnover dampening through staggered sub-portfolios rather than explicit penalty matrices. A particularly close theoretical benchmark is Gârleanu and Pedersen (2013), who derive optimal dynamic trading policies when returns are predictable and trading is costly, with slower-decaying predictors receiving greater weight in the dynamic target. Our question is narrower and structurally different: we study a discrete overlapping-cohort execution rule and characterize how its utility-maximizing horizon changes with signal decay, noise, costs, and risk aversion. We therefore interpret the cohort frontier as a complementary execution-design result rather than a replacement for the general dynamic-control solution.

Staggered cohort structures have a long empirical history in empirical asset pricing. Jegadeesh and Titman (1993, 2001) utilized overlapping multi-month holding periods to evaluate cross-sectional momentum portfolios, while Bernard and Thomas (1989, 1990) utilized multi-week horizons to study post-earnings-announcement drift (PEAD). However, these studies largely interpreted performance through the lens of market anomalies and behavioral under-reaction, rather than isolating the mechanical regularizing effect of cohort aggregation itself.

In contemporary machine learning and distributed optimization, regularization is traditionally introduced through parameter penalties ($L_1$, $L_2$), dropout, or Bayesian priors (Blundell et al., 2015; Hu et al., 2022). In portfolio applications, Papastaikoudis et al. (2025) investigate distributed consensus optimization under transaction constraints. We complement this literature by demonstrating that cohort stacking functions as an architectural **temporal regularizer** directly in execution space: it suppresses high-frequency turnover noise and diversifies estimation errors without requiring explicit penalty hyperparameters.

---

## 3. Analytical Framework

### 3.1 Mechanics of Overlapping Cohort Portfolios

Consider a discrete-time investment setting where at each period $t$, a quantitative forecasting model generates an expected return ranking over an asset universe of size $N$. 

Under standard single-horizon rebalancing ($H=1$), the manager constructs a base portfolio weight vector $w_t^{\text{base}} \in \Delta^{N-1}$ and rebalances the entire portfolio each period. The one-period rebalancing turnover is:
$$\text{TO}_t^{H=1} = \frac{1}{2} \| w_t^{\text{base}} - w_{t-1}^{\text{base}} \|_1$$
Under overlapping cohort stacking with horizon $H \in \{1, 2, \dots\}$, capital is partitioned equally across $H$ sub-portfolios. At period $t$, the cohort entered at $t-H$ is liquidated, and a new cohort with target weights $w_t^{\text{base}}$ is initiated. The aggregate portfolio weight vector $w_t^H$ is:
$$w_t^H = \frac{1}{H} \sum_{k=0}^{H-1} w_{t-k}^{\text{base}}$$
Assuming stationary rebalancing dynamics, the turnover of the aggregate portfolio scales inversely with the horizon:
$$\text{TO}_t^H = \frac{1}{2} \| w_t^H - w_{t-1}^H \|_1 = \frac{1}{2H} \| w_t^{\text{base}} - w_{t-H}^{\text{base}} \|_1 \approx \frac{\text{TO}_0}{H}$$
where $\text{TO}_0$ denotes the characteristic one-period base turnover.

Figure 1 illustrates the architectural comparison between single-horizon rebalancing and an $H=4$ overlapping cohort portfolio, highlighting the three competing economic forces.

![Figure 1: Temporal Regularization Architecture](figures/fig1_temporal_regularization_architecture.png)
*Figure 1: Overlapping cohort portfolio architecture ($H=4$) versus single-period rebalancing ($H=1$). In an $H=4$ cohort architecture, capital is partitioned across four staggered sub-portfolios, with one sub-portfolio refreshed each week. Aggregate portfolio weights represent the uniform average across active cohorts, mechanically dampening turnover and smoothing idiosyncratic estimation noise at the cost of alpha dilution.*

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

## 4. Comparative Statics and Break-Even Frontiers

To characterize utility-maximizing horizon selection, we evaluate the horizon utility differential relative to the single-period baseline $H=1$:
$$\Delta U(H, 1) = U(H) - U(1) = \mu_0 \left[ \frac{1 - \rho^H}{H(1 - \rho)} - 1 \right] + c \cdot \text{TO}_0 \left(1 - \frac{1}{H}\right) + \frac{1}{2}\gamma \sigma_1^2 \left(1 - \frac{1}{H}\right)$$

Factoring out $(1 - 1/H)$:
$$\Delta U(H, 1) = \left(1 - \frac{1}{H}\right) \left\{ c \cdot \text{TO}_0 + \frac{1}{2}\gamma \sigma_1^2 - \mu_0 \Lambda(H, \rho) \right\}$$
where the effective information decay factor is defined as:
$$\Lambda(H, \rho) \equiv \frac{1 - \frac{1 - \rho^H}{H(1 - \rho)}}{1 - 1/H}$$

Figure 2 illustrates the decomposition of expected utility into alpha dilution, cost amortization, and risk reduction, showing how the net utility curve shifts across signal decay and cost environments.

![Figure 2: Analytical Tradeoff](figures/fig2_analytical_tradeoff.png)
*Figure 2: Analytical utility tradeoffs across holding horizon $H$. Stylized curves illustrate alpha dilution, transaction cost amortization, variance diversification, and net expected utility across representative parameter combinations.*

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

## 5. Synthetic Experiment Design

To evaluate these comparative statics without distortions from historical corporate event dates, look-ahead bias, or data snooping, we constructed a controlled synthetic market simulation (Phase 20).

### 5.1 The Data-Generating Process (DGP)
1. **Asset Universe & Horizon Grid**: $N = 30$ assets evaluated across $T = 156$ calendar weeks, with a 52-week latent-signal history used for the finite-memory return construction. Candidate horizons were frozen at $H \in \{1, 2, 4, 8, 13\}$ weeks.
2. **Latent Information Process**: At each calendar week $t$, each asset $i$ receives an independent latent predictive factor $s_{i,t} \sim \mathcal{N}(0, 1)$.
3. **Single Realized Market Return Invariance**:
   Crucially, to eliminate cohort-specific return noise or oracle leaks, each asset has **exactly one realized market return per calendar week**, shared by every active cohort holding that asset:
   $$R_{u, i} = \sum_{k=0}^{K} \beta \rho^k s_{i, u-k} + \epsilon_{\text{mkt}, u} + \xi_{u, i}$$
   where $K = 52$ truncates the latent history, $\beta$ governs signal magnitude, $\epsilon_{\text{mkt}, u} \sim \mathcal{N}(0, \sigma_{\text{mkt}}^2)$ with $\sigma_{\text{mkt}} = 0.02$ (200 bps/week) is a shared market factor shock, and $\xi_{u, i} \sim \mathcal{N}(0, \sigma_{\text{idio}}^2)$ with $\sigma_{\text{idio}} = 0.04$ (400 bps/week) is an asset-specific calendar return shock.
4. **Finite-Memory Truncation Disclosure**: For the highest persistence level ($\rho = 0.97$), $0.97^{52} \approx 0.2032$. Thus, the 52-week truncation leaves approximately $20\%$ of latent signal weight unmodeled. The $\rho = 0.97$ case is therefore an explicitly finite-memory approximation rather than an asymptotic infinite-history AR(1) process.
5. **Noisy Ranking Signals**: At entry week $t$, portfolio weights are formed by ranking assets on noisy observed signals:
   $$\hat{s}_{i,t} = s_{i,t} + \sigma_{\text{noise}} \eta_{i,t}, \quad \eta_{i,t} \sim \mathcal{N}(0, 1)$$
6. **Exact-Null Baseline**: At $\beta = 0.000$, ranking signals are deactivated, and base portfolio weights are set to uniform equal weighting ($w_{i,t} = 1/N$).

### 5.2 Factorial Simulation Matrix and Audit Guardrails

The simulation evaluates a complete factorial grid:
- Signal Strength: $\beta \in \{0.000, 0.005, 0.010, 0.020\}$
- Persistence: $\rho \in \{0.0, 0.25, 0.50, 0.75, 0.90, 0.97\}$ ($\tau_{1/2} \in [0.0, 22.8]\text{ weeks}$)
- Observation Noise: $\sigma_{\text{noise}} \in \{0.0, 0.5, 1.0, 2.0\}$
- Transaction Costs: $c \in \{0, 5, 10, 25, 50\}\text{ bps}$
- Risk Aversion: $\gamma \in \{0.0, 0.5, 1.0, 2.5, 5.0\}$
- Random Seeds: 30 independent seeds ($200001$ through $200030$)
- **Total Factorial Evaluations**: $4 \times 6 \times 4 \times 5 \times 5 \times 5 \times 30 = 360{,}000$ evaluations.

For every evaluation, numerical closure was audited against the identity:
$$\Delta U(H_A, H_B) = (\bar{r}_{\text{gross}, A} - \bar{r}_{\text{gross}, B}) - c(\overline{\text{TO}}_A - \overline{\text{TO}}_B) - \frac{1}{2}\gamma(\text{Var}(r_{\text{net}, A}) - \text{Var}(r_{\text{net}, B}))$$
with required tolerance $|\text{error}| < 10^{-12}$. Statistical ties across seeds were evaluated using two-sided paired $t$-tests ($\alpha = 0.05$).

Table 2 presents the complete experimental specifications, parameter levels, and numerical guardrails.

| Parameter / Audit Dimension | Specification / Value | Description & Guardrail |
| :--- | :--- | :--- |
| **Asset Universe Size ($N$)** | 30 assets | Fixed synthetic cross-section |
| **Evaluation Timeline ($T$)** | 156 calendar weeks | 3 years of weekly market returns |
| **Latent History Truncation ($K$)** | 52 weeks | Truncation weight at $\rho=0.97$: $0.97^{52} \approx 0.2032$ (finite-memory approximation) |
| **Market Factor Shock ($\sigma_{\text{mkt}}$)** | 0.02 (200 bps/week) | Shared calendar-week market innovation |
| **Idiosyncratic Shock ($\sigma_{\text{idio}}$)** | 0.04 (400 bps/week) | Asset-specific calendar return shock |
| **Simulation Seeds ($S$)** | 30 independent seeds | Seeds 200001 through 200030 inclusive |
| **Candidate Horizons ($H$)** | $\{1, 2, 4, 8, 13\}$ weeks | Frozen candidate set (5 levels) |
| **Signal Magnitude ($\beta$)** | $\{0.000, 0.005, 0.010, 0.020\}$ | 4 levels; deactivates to uniform weighting at $\beta=0.000$ |
| **Persistence ($\rho$)** | $\{0.0, 0.25, 0.50, 0.75, 0.90, 0.97\}$ | 6 levels; continuous half-life $\tau_{1/2} \in [0.0, 22.8]$ weeks |
| **Observation Noise ($\sigma_{\text{noise}}$)** | $\{0.0, 0.5, 1.0, 2.0\}$ | 4 levels of additive ranking error |
| **Transaction Cost ($c$)** | $\{0, 5, 10, 25, 50\}$ bps | 5 levels of linear proportional friction |
| **Risk Aversion ($\gamma$)** | $\{0.0, 0.5, 1.0, 2.5, 5.0\}$ | 5 levels of mean-variance risk penalty |
| **Total Factorial Evaluations** | **360,000 evaluations** | $4 \times 6 \times 4 \times 5 \times 5 \times 5 \times 30$ full factorial grid |
| **Pairwise Evaluations vs $H=1$** | **288,000 comparisons** | Evaluated across $H \in \{2, 4, 8, 13\}$ |
| **Maximum Accounting Error** | **$3.144 \times 10^{-17}$** | Strictly satisfies tolerance $|\text{error}| < 10^{-12}$ |
| **Exact-Null ($\beta=0$) Discrepancy** | **$0.00\text{e}+00$** | Verified exact across 7 invariants; 100% classified as `Tie_1-2-4-8-13` |
| **Adjudicated Outcome** | **Outcome A** | Preregistered criteria satisfied: accounting closure, verified invariants, confirmed statics |

*Table 2: Phase 20 Synthetic Factorial Simulation Design & Invariant Audit.*

---

## 6. Synthetic Simulation Results

### 6.1 Confirmation of Comparative Static Hypotheses

Across all 360,000 evaluations, the synthetic simulation confirmed all five preregistered comparative static hypotheses, as summarized in Table 1:

| Hypothesis | Description | Predicted Direction | Simulation Consistency | Status |
| :--- | :--- | :---: | :---: | :---: |
| **H1** | Cost Monotonicity | $dH*/dc >= 0$ | **100.0%** | **CONFIRMED** |
| **H2** | Persistence Monotonicity | $dH*/drho >= 0 (beta > 0)$ | **90.0%** | **CONFIRMED** |
| **H3** | Short-Lived Alpha Reversal | $H*=1 at rho=0, c=0; dH*/dbeta <= 0 under fast decay$ | **100.0%** | **CONFIRMED** |
| **H4** | Noise Regularization | $dH*/dsigma >= 0$ | **100.0%** | **CONFIRMED** |
| **H5** | Risk Aversion Preference | $dH*/dgamma >= 0$ | **100.0%** | **CONFIRMED** |

*Table 1: Phase 20 Primary Comparative Statics Confirmation. Consistency reflects the proportion of pairwise parameter transitions across 360,000 evaluations conforming to the theoretical prediction. Hypothesis H2 is evaluated on active signals ($\beta > 0$). All evaluations satisfy accounting closure to machine precision ($|\text{error}| \le 3.144 \times 10^{-17}$).*

1. **Cost Monotonicity (H1, 100.0% consistency)**: Increasing transaction costs $c$ monotonically shifts the utility-maximizing horizon upward. In high-cost environments ($c \ge 25\text{ bps}$), longer horizons ($H=8, 13$) dominate across almost all signal regimes.
2. **Persistence Monotonicity (H2, 90.0% consistency)**: Under active signals ($\beta > 0$), increasing signal persistence $\rho$ preserves alpha across older cohorts, shifting $H^*$ toward larger values.
3. **Short-Lived Alpha Reversal (H3, 100.0% consistency)**: At zero transaction cost ($c = 0$) and zero persistence ($\rho = 0$), the utility-maximizing horizon is strictly $H^* = 1$. Under fast signal decay, increasing signal strength $\beta$ favors shorter, more responsive horizons.
4. **Noise Regularization (H4, 100.0% consistency)**: Increasing ranking noise $\sigma_{\text{noise}}$ degrades single-cohort alpha while preserving cross-cohort noise averaging, shifting $H^*$ upward.
5. **Risk Aversion Preference (H5, 100.0% consistency)**: Increasing risk aversion $\gamma$ elevates the penalty on portfolio variance, expanding preference for cohort smoothing.

### 6.2 Accounting Closure and Exact-Null Audit

Across all 288,000 pairwise evaluations against $H=1$, numerical closure held to machine precision:
$$\max |\text{error}| = 3.144 \times 10^{-17} \ll 10^{-12}$$
confirming exact algebraic consistency in turnover accounting and utility decomposition.

An audit of the exact-null regime ($\beta = 0.000$) verified seven invariants across all 72,000 comparisons against $H=1$:
- Aggregate portfolio weight vector: $\max |\Delta| = 0.00\text{e}+00$
- Weekly gross return: $\max |\Delta| = 0.00\text{e}+00$
- Weekly turnover: $\max |\Delta| = 0.00\text{e}+00$
- Weekly transaction cost: $\max |\Delta| = 0.00\text{e}+00$
- Weekly net return: $\max |\Delta| = 0.00\text{e}+00$
- Mean utility: $\max |\Delta| = 0.00\text{e}+00$
- Variance term: $\max |\Delta| = 0.00\text{e}+00$

Under constant uniform weighting ($w_t = 1/N$), rebalancing turnover is identically zero ($\text{TO} \equiv 0$). Consequently, all active cohorts hold identical weights, and all horizons tie identically ($100\%$ of cells classified as `Tie_1-2-4-8-13`). This audit confirmed that stacking dominance under friction emerges as soon as the portfolio has rebalancing churn (weak alpha or ranking noise), but does not operate on time-invariant allocations.

### 6.3 Horizon Phase Frontier

Figure 3 illustrates the resulting 2D synthetic horizon phase diagrams across the transaction cost and signal persistence grid.

![Figure 3: Phase 20 Synthetic Horizon Frontier](figures/fig3_phase20_synthetic_horizon_frontier.png)
*Figure 3: Controlled synthetic simulation horizon phase diagram across transaction costs $c$ and signal persistence $\rho$. The color-coded cells indicate the utility-maximizing horizon within the discrete candidate set $H \in \{1, 2, 4, 8, 13\}$ across representative signal strength ($\beta$) and observation noise ($\sigma_{\text{noise}}$) regimes. Grey cells designate the exact-null tie regime where time-invariant uniform weights produce an identical tie ($0.00\text{e}+00$ discrepancy).*

Four distinct operational regimes emerge from the phase diagram:
1. **Short-Horizon Dominance ($H=1$)**: Confined to low-friction ($c \le 5\text{ bps}$), short-lived ($\tau_{1/2} \le 0.5\text{ wks}$), high-alpha ($\beta \ge 0.01$) environments.
2. **Intermediate Regularization ($H=2\text{ to }4$)**: The broad steady-state attractor for institutional parameters ($c \approx 10\text{ bps}$, moderate noise, half-life $1\text{--}6\text{ weeks}$).
3. **Extended Stacking ($H=8\text{ to }13$)**: Dominates when transaction friction is high ($c \ge 25\text{ bps}$), signals are highly persistent ($\tau_{1/2} > 6\text{ wks}$), or observation noise is severe ($\sigma_{\text{noise}} \ge 2.0$).
4. **Exact-Null Tie Regime ($\beta = 0.000$)**: Zero rebalancing churn yields identical performance across all horizons ($\text{TO} \equiv 0, \Delta U \equiv 0$).

---

## 7. Retrospective Empirical Motivation and Reconciliation

We connect these synthetic findings back to the retrospective event-risk setting that motivated this research. The empirical motivation is related to the long post-earnings-announcement-drift literature (e.g., Bernard and Thomas, 1989), but our clean revalidation does not establish persistent multi-week drift:

In Phase 18, clean revalidation of quarterly post-earnings portfolios on uncompromised metadata exhibited:
- Transaction cost assumption: $c = 10\text{ bps}$
- Empirical Rank IC $\approx 0.05\text{--}0.06$ (used only as a descriptive guide to nearby weak/noisy synthetic regimes; it does not identify $\beta$ or $\sigma_{\text{noise}}$ structurally)
- Estimated weekly persistence: $\rho \approx 0.50\text{--}0.75$ ($\tau_{1/2} \approx 1.0\text{--}2.4\text{ weeks}$)
- Annualized net return: $9.24\%$ ($H=1$) vs $14.71\%$ ($H=4$)
- Annualized net volatility: $19.59\%$ ($H=1$) vs $6.62\%$ ($H=4$) ($-66.21\%$ reduction)
- Annualized traded risky turnover: $73.03$ ($H=1$) vs $19.01$ ($H=4$) ($-73.97\%$ reduction)
- Clean annualized Sharpe ratio: **$0.469$** ($H=1$) to **$2.223$** ($H=4$)
- Mean net weekly utility delta: $+13.804\text{ bp/week}$ (paired 95% CI: $[-54.68, +84.33]\text{ bp/week}$)
- Incremental weeks 2–4 predictive persistence: Rank IC $= 0.0060$, $p = 0.105$ (statistically insignificant)

Table 3 compiles the empirical revalidation metrics alongside the signal stress tests from Phase 19.

| Performance / Diagnostic Metric | Single Horizon ($H=1$) | Cohort Stacking ($H=4$) | Difference / Impact | Status / Audit Note |
| :--- | :---: | :---: | :---: | :--- |
| **Annualized Net Return** | 9.24% | 14.71% | +5.47% | Arithmetic annual return |
| **Annualized Net Volatility** | 19.59% | 6.62% | **-66.21%** | Substantial variance reduction |
| **Traded Risky Turnover (L1/yr)** | 73.03 | 19.01 | **-73.97%** | Proportional cost amortization |
| **Annualized Sharpe Ratio** | **0.469** | **2.223** | **+1.754** | Clean revalidation value |
| **Maximum Drawdown** | -18.77% | -5.00% | +13.77% | Downside risk mitigation |
| **Mean Net Weekly Utility** | 14.072 bp/wk | 27.876 bp/wk | **+13.804 bp/wk** | 95% CI: $[-54.68, +84.33]$ bp/wk |
| **Gross Return Contribution** | — | — | +0.145 bp/wk | Gross alpha contribution is near zero |
| **Transaction Cost Savings** | — | — | +10.387 bp/wk | Primary driver of point utility delta |
| **Variance Term Utility Gain** | — | — | +3.272 bp/wk | Secondary driver of point utility delta |
| **Residual Weeks 2–4 Rank IC** | — | — | **0.0060** | $p = 0.105$ (Statistically Insignificant) |

*Table 3: Retrospective Empirical Evidence and Signal Stress Tests (Phases 18 & 19). Panel A reflects clean revalidation on uncompromised metadata inputs (Phase 18). Panel B reports the stress test across signal quality (Phase 19). These historical results serve as retrospective empirical motivation and reconciliation, not prospective trading confirmation.*

Using nearby weak/noisy Phase 20 grid cells as a descriptive comparison—not as an estimated structural mapping—the simulated utility-maximizing horizon is typically $H^* \in \{1, 2\}$ at $c = 10\text{ bps}$ and shifts toward $H=4$ or $H=8$ as transaction costs rise to $25\text{--}50\text{ bps}$ or observation noise increases.

Why, then, did the empirical earnings strategy strongly favor $H=4$?
One quantitatively important explanation is **panel churn**: because corporate earnings announcements occur quarterly, the weekly cross-section of reporting companies was 100% disjoint from week to week. Under un-stacked rebalancing ($H=1$), this disjointness forced an extreme baseline turnover of $\sim 72\text{--}104\text{x/year}$. In the analytical utility model, larger base turnover $\text{TO}_0$ increases the marginal value of cost amortization and therefore shifts preference toward longer horizons. This mechanism is consistent with the historical $H=4$ result, but the retrospective data do not identify a unique structural $H^*$.

Figure 4 illustrates the empirical puzzle and its resolution across the Phase 18 and 19 diagnostic panels.

![Figure 4: Retrospective Empirical Motivation](figures/fig4_retrospective_empirical_motivation.png)
*Figure 4: Retrospective empirical motivation and reconciliation from Phase 18 and Phase 19 event studies. (A) Clean signal persistence across weeks 2–4 is statistically indistinguishable from zero ($p=0.105$). (B) Overlapping cohorts ($H=4$) slash annualized traded turnover by $73.97\%$. (C) Stacking reduces annualized volatility by $66.21\%$. (D) Phase 19 synthetic signal ladder demonstrates that cohort stacking delivers maximum relative advantage under weak or null signals, but loses advantage under strong alpha.*

This resolves the empirical paradox:
- In Phase 18, persistent post-earnings alpha failed to achieve significance ($p = 0.105$), yet $H=4$ delivered substantial Sharpe gains ($0.469 \to 2.223$) and turnover reductions ($-73.97\%$).
- In Phase 19, stress testing proved that turnover and volatility savings occurred even under random permutations and null signals, but relative utility gains vanished under strong synthetic signals.

The synthetic simulation provides a controlled mechanistic account consistent with the historical pattern: **multi-week stacking can act as an architectural regularizer when signal quality is weak and portfolio churn is high**. In noisy, low-alpha, high-churn environments, the turnover and variance savings of overlapping cohorts overwhelm alpha dilution, making multi-week cohort stacking optimal even when underlying predictive signals are weak.

---

## 8. Negative-Result Provenance and Falsification Summary

For scientific transparency, we document how earlier candidate hypotheses were systematically investigated and falsified:

1. **Bayesian Epistemic Uncertainty (Phases 11–16.1)**:
   - Evaluated as an asset-selection and timing signal.
   - Leakage audit revealed training text contaminated with `move_1d`.
   - Entropy-error metric audit showed $\rho \approx 0.60$ was a mechanical artifact of 80/20 class imbalance under probability shrinkage, mechanically forcing within-class rank correlations to $\pm 1.0000$.
   - Clean retraining showed Bayesian LoRA ($\text{Brier} = 0.15987$) was matched by deterministic LoRA ($0.16079$) and a constant base-rate null ($0.16000$).
   - Downstream portfolio mappings established no robust clean economic advantage, while 32-draw Monte Carlo sampling imposed a $30\times$ latency penalty over a single deterministic pass of the posterior mean.
2. **Collaborative LoRA Specialist Weighting (Pilot)**:
   - Evaluated cross-sector specialization with learned coordination weights.
   - Failed continuation criteria as weights collapsed to banking-only selection, failing to outperform global pooled training. Closed as a negative result.
3. **Prospective 2026 Confirmation (Phase 10)**:
   - Exhaustive data audit proved all market returns terminated on May 15, 2025.
   - Adhering to non-manufacturing protocols, prospective confirmation claims were retracted and classified under retrospective holdout scope.

These negative results redirected our inquiry away from predictive model complexity and toward portfolio execution mechanics, ultimately yielding the temporal regularization framework.

---

## 9. Limitations and Boundary Conditions

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

## 10. Conclusion and Design Principles

Overlapping cohort portfolios are not merely an empirical heuristic for capturing prolonged price drift; they are a formal **temporal regularization mechanism**. By pooling staggered sub-portfolios across time, cohort stacking trades responsiveness to fresh signals against execution turnover and estimation noise.

For quantitative practitioners, our findings offer clear architectural design principles:
- **Measure Signal Half-Life Before Setting Horizons**: For transient signals ($\tau_{1/2} < 0.5\text{ weeks}$), stacking beyond $H=1$ severely penalizes performance unless transaction costs exceed $25\text{ bps}$.
- **Use Stacking to Dampen High-Frequency Panel Churn**: In event-driven strategies with highly disjoint cross-sections, cohort stacking can materially reduce turnover; in the retrospective earnings setting studied here, the reduction was about $74\%$ for $H=4$.
- **Recognize the Friction-Decay Frontier**: Within the controlled model and candidate horizon grid, the utility-maximizing horizon generally shifts upward with transaction costs, signal persistence, observation noise, and risk aversion. When alpha is weak and noisy, temporal regularization can be an important implementation lever.

---

## 11. References

- Bernard, V. L., & Thomas, J. K. (1989). Post-earnings-announcement drift: Delayed price response or risk premium? *Journal of Accounting Research*, 27, 1–36.
- Bernard, V. L., & Thomas, J. K. (1990). Evidence that stock prices do not fully reflect the implications of current earnings for future earnings. *Journal of Accounting and Economics*, 13(4), 305–340.
- Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty in neural networks. In *Proceedings of the 32nd International Conference on Machine Learning (ICML)* (pp. 1613–1622).
- Gârleanu, N., & Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *The Journal of Finance*, 68(6), 2309–2340.
- Gârleanu, N., & Pedersen, L. H. (2016). Dynamic portfolio choice with frictions. *Journal of Economic Theory*, 165, 487–516.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations (ICLR)*.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *The Journal of Finance*, 48(1), 65–91.
- Jegadeesh, N., & Titman, S. (2001). Profitability of momentum strategies: An evaluation of alternative explanations. *The Journal of Finance*, 56(2), 699–720.
- Markowitz, H. (1952). Portfolio selection. *The Journal of Finance*, 7(1), 77–91.
- Merton, R. C. (1969). Lifetime portfolio selection under uncertainty: The continuous-time case. *The Review of Economics and Statistics*, 51(3), 247–257.
- Merton, R. C. (1971). Optimum consumption and portfolio rules in a continuous-time model. *Journal of Economic Theory*, 3(4), 373–413.
- Papastaikoudis, I., Watson, J., & Lestas, I. (2025). Distributed portfolio optimization and decentralized pricing. *IEEE Transactions on Control of Network Systems*, 12(1), 112–124.
