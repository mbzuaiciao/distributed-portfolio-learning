# Phase 20 Preregistration: Optimal Cohort Holding Horizon (Amended DGP)

## Scientific Question

**What cohort holding horizon $H^*$ is optimal as a function of signal half-life ($\tau_{\text{signal}}$), transaction cost ($c$), signal noise ($\sigma_{\text{noise}}$), and portfolio risk aversion ($\gamma$)?**

Formally:
$$H^* = f(\tau_{\text{signal}},\ c,\ \sigma_{\text{noise}},\ \gamma)$$

This investigation explains the Phase 19 reversal mechanistically under a fully controlled synthetic Data Generating Process (DGP) free of empirical LoRA confounding.

---

## 1. Mathematical Data Generating Process (DGP)

The DGP generates a stationary universe of $N = 30$ assets across $T = 156$ weeks (3 full years):

1. **Latent Predictive Signal**:
   At each entry week $t \in \{0, \dots, T-1\}$, asset $i \in \{1, \dots, N\}$ receives:
   $$s_{i, t} \sim \mathcal{N}(0, 1)$$

2. **Observed Ranking Signal (with Measurement Noise)**:
   Portfolio entry weights observe only the noisy proxy:
   $$\hat{s}_{i, t} = s_{i, t} + \sigma_{\text{noise}} \eta_{i, t}, \quad \eta_{i, t} \sim \mathcal{N}(0, 1)$$
   The true latent signal $s_{i,t}$ remains strictly unobserved at portfolio formation.

3. **Single Realized Calendar Return Per Asset (Shared Across Cohorts)**:
   In accordance with actual market mechanics, each asset $i$ has **exactly one realized return per calendar week $u$**, shared by every active cohort holding that asset during week $u$. Cohort-specific realized idiosyncratic shocks are strictly eliminated.
   For calendar week $u \in \{0, \dots, T-1\}$, asset $i$ realizes return:
   $$R_{i, u} = \beta \sum_{k=0}^{K_{\text{burn}}} \rho^k s_{i, u-k} + \epsilon_{\text{mkt}, u} + \xi_{i, u}$$
   where:
   - $\beta = 0.01$ (signal predictive strength coefficient);
   - $\rho \in [0, 1)$ governs weekly signal persistence;
   - $K_{\text{burn}} = 20$ weeks of historical signal superposition;
   - $\epsilon_{\text{mkt}, u} \sim \mathcal{N}(0, \sigma_{\text{mkt}}^2)$ is the systematic market factor in calendar week $u$ ($\sigma_{\text{mkt}} = 0.02$);
   - $\xi_{i, u} \sim \mathcal{N}(0, \sigma_{\text{idio}}^2)$ is the asset-specific calendar return shock in week $u$ ($\sigma_{\text{idio}} = 0.04$).
   
   **Invariance Property**: For any two cohorts $t_1, t_2 \le u$ holding asset $i$ in calendar week $u$, the realized return on asset $i$ is identically equal to $R_{i, u}$.

4. **Cohort Portfolio Formation**:
   Each week $t$, cohort base weights $w_t \in \Delta^N$ are solved using canonical simplex-projected Markowitz optimization matching the repo's equicorrelated covariance structure:
   $$w_t = \arg\min_{w \ge 0, \sum_i w_i = 1} w^\top Q w - \hat{\mu}_t^\top w \equiv \Pi_{\Delta^N}\left( \frac{\beta \hat{s}_t}{2 \times 0.04} \right)$$

5. **Cohort Stacking & Horizon Expiration**:
   For horizon $H$, the portfolio in week $u$ aggregates $K = \min(u+1, H)$ active cohorts with equal capital allocation $1/K$:
   $$w_u^{\text{stacked}} = \frac{1}{K} \sum_{k=0}^{K-1} w_{u-k}$$
   Active cohorts expire and are removed from the portfolio after age $H-1$.
   The realized gross return of the portfolio in calendar week $u$ is:
   $$R_{\text{gross}, u} = \sum_{i=1}^N w_{u, i}^{\text{stacked}} R_{i, u} = (w_u^{\text{stacked}})^\top R_u$$

---

## 2. Frozen Parameter Grids

1. **Horizon Grid ($H$)**:
   $$H \in \{1, 2, 4, 8, 13\}$$

2. **Signal Persistence Grid ($\rho$) & Implied Half-Life ($\tau_{1/2}$)**:
   $$\rho \in \{0.0, 0.25, 0.50, 0.75, 0.90, 0.97\}$$
   Implied half-lives:
   - $\rho = 0.00 \implies \tau_{1/2} = 0.00\text{ weeks}$ (transient 1-week impulse)
   - $\rho = 0.25 \implies \tau_{1/2} = 0.50\text{ weeks}$
   - $\rho = 0.50 \implies \tau_{1/2} = 1.00\text{ weeks}$
   - $\rho = 0.75 \implies \tau_{1/2} = 2.41\text{ weeks}$
   - $\rho = 0.90 \implies \tau_{1/2} = 6.58\text{ weeks}$
   - $\rho = 0.97 \implies \tau_{1/2} = 22.76\text{ weeks}$

3. **Transaction Cost Grid ($c$)**:
   $$c \in \{0, 5, 10, 25, 50\}\text{ bps} \equiv \{0.0000, 0.0005, 0.0010, 0.0025, 0.0050\}$$

4. **Signal Noise Grid ($\sigma_{\text{noise}}$)**:
   $$\sigma_{\text{noise}} \in \{0.0, 0.5, 1.0, 2.0\}$$

5. **Risk Aversion Grid ($\gamma$)**:
   $$\gamma \in \{0.0, 0.5, 1.0, 2.5, 5.0\}$$

6. **Random Simulation Seeds**:
   30 predeclared seeds:
   $$\text{seeds} = [200001, 200002, \dots, 200030]$$

---

## 3. Total Experiment Count & Computational Strategy

- Parameter cells: $6 (\rho) \times 4 (\sigma_{\text{noise}}) \times 5 (c) \times 5 (\gamma) \times 5 (H) = 3,000$ cells per seed.
- Total matrix size: $3,000 \times 30 = 90,000$ evaluations.
- Simulated cohort rollouts: $6 (\rho) \times 4 (\sigma_{\text{noise}}) \times 30 (\text{seeds}) \times 5 (H) = 3,600$ rollouts.
- Grid transformations over $(c, \gamma)$: vectorized linear array operations.
- Projected runtime: $\approx 20 - 30\text{ seconds}$ CPU-local.

---

## 4. Frozen Utility Definition & Primary Optimization Endpoint

For weekly net return sequence $r_{\text{net}, t} = r_{\text{gross}, t} - c \cdot \text{TO}_t$:

$$U(H) = \bar{r}_{\text{net}}(H) - \frac{1}{2}\gamma \text{Var}(r_{\text{net}}(H), \text{ddof}=1)$$

The primary optimization endpoint is:
$$H^*(\rho, c, \sigma_{\text{noise}}, \gamma) = \arg\max_{H \in \{1, 2, 4, 8, 13\}} \bar{U}(H)$$

### Statistical Uncertainty and Indeterminate / Tie Zones
For parameter cell $(\rho, c, \sigma, \gamma)$, let $H^*$ denote the horizon with maximum sample mean utility. For any competing horizon $H'$, we test the paired seed difference $d_s = U_s(H^*) - U_s(H')$:
- If a two-sided paired t-test yields $p > 0.05$ (or 95% CI includes 0), $H'$ is statistically indistinguishable from $H^*$.
- If one or more $H' \ne H^*$ are indistinguishable from $H^*$, the cell is designated an **Indeterminate / Tie Zone** between $\{H^*, H'\}$.
- If $H^*$ significantly dominates all $H' \ne H^*$ ($p < 0.05$), $H^*$ is designated a **Dominant Winner**.

---

## 5. Frozen Primary Scientific Hypotheses

- **H1 (Cost Comparative Static)**: Optimal $H^*$ is weakly increasing in transaction cost $c$ ($\partial H^* / \partial c \ge 0$).
- **H2 (Persistence Comparative Static)**: Optimal $H^*$ is weakly increasing in signal persistence $\rho$ ($\partial H^* / \partial \rho \ge 0$).
- **H3 (Short-Lived Alpha)**: At zero persistence ($\rho = 0.0$) and frictionless cost ($c = 0$), $H^* = 1$ under clean signal ($\sigma_{\text{noise}} = 0.0$) and low risk aversion ($\gamma \le 1.0$).
- **H4 (Noise Regularization)**: As signal noise $\sigma_{\text{noise}}$ increases, $H^*$ weakly increases (longer cohorts average transient ranking error).
- **H5 (Risk Aversion Effect)**: Increasing risk aversion $\gamma$ shifts preference toward larger $H$ ($\partial H^* / \partial \gamma \ge 0$).

---

## 6. Accounting Closure & Invariants

Across every simulation seed and evaluated cell:
$$\Delta U(H_A, H_B) = \Delta \bar{r}_{\text{gross}} + \Delta \text{cost effect} + \Delta \text{risk effect}$$
where:
- $\Delta \bar{r}_{\text{gross}} = \bar{r}_{\text{gross}, A} - \bar{r}_{\text{gross}, B}$
- $\Delta \text{cost effect} = -c (\overline{\text{TO}}_A - \overline{\text{TO}}_B)$
- $\Delta \text{risk effect} = -\frac{1}{2}\gamma (\text{Var}(r_{\text{net}, A}) - \text{Var}(r_{\text{net}, B}))$

**Invariant**: Numerical closure error $|\Delta U - (\Delta \bar{r}_{\text{gross}} + \Delta \text{cost effect} + \Delta \text{risk effect})| < 10^{-12}$.

---

## 7. Frozen Outcome Classification Tree

- **Outcome A (Coherent Horizon Frontier)**:
  - All simulator audits pass (including shared calendar return invariant);
  - Accounting closure holds with error $< 10^{-12}$;
  - At least 4 of 5 primary comparative-static hypotheses (H1-H5) hold with statistical significance ($\ge 80\%$ monotonic slice consistency);
  - Horizon phase diagram exhibits contiguous, non-degenerate frontiers.
  - *Interpretation*: Temporal regularization exhibits a coherent, predictable general design law.

- **Outcome B (Partial Frontier)**:
  - Audits pass and accounting closure holds;
  - 2 or 3 of the 5 primary hypotheses hold, or strong interaction terms create non-monotonic regimes.
  - *Interpretation*: Horizon selection is structured but requires multi-variable interaction terms.

- **Outcome C (No Stable Frontier)**:
  - Audits pass and accounting closure holds;
  - Fewer than 2 primary hypotheses hold, or $H^*$ varies erratically across seeds without statistical separation.
  - *Interpretation*: The Phase 19 temporal-regularizer story fails to generalize to a controlled setting.

- **Outcome D (Model Artifact / Invariant Failure)**:
  - Any simulator audit fails, or accounting closure fails ($|\text{error}| \ge 10^{-12}$).
