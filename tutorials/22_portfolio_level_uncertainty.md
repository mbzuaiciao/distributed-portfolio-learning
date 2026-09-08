# Tutorial 22: Portfolio-Level Uncertainty Aggregation & Exposure Dynamics

In Phases 3 and 4, we established a definitive negative result for cross-sectional portfolio construction:
> **Negative Finding (Phases 3 & 4)**: Bayesian-LoRA epistemic uncertainty contains strong predictive information about *forecast error magnitude* ($\text{Spearman } \rho \approx 0.55$, $\text{AUROC } \approx 0.82$), but it does **not** indicate *directional downside*. Applying confidence shrinkage or covariance inflation to individual assets hurts or leaves neutral out-of-sample portfolio utility because it suppresses high-alpha bets without improving asset selectivity.

Phase 5 investigates a fundamentally different hypothesis:
> **The Exposure Hypothesis**:
> *Epistemic uncertainty may fail at choosing WHICH assets to own (cross-sectional selection), but may be valuable for deciding HOW MUCH aggregate portfolio risk to take (portfolio-level exposure timing).*

In this tutorial, we define how to aggregate asset-level epistemic uncertainties into a single portfolio-level metric $u_t$, map $u_t$ into an aggregate risky exposure scale $s_t \in [0, 1]$, and allocate the remainder to uninvested cash without distorting relative asset selections.

---

## 1. Aggregating Epistemic Uncertainty

Given a universe of $m$ assets at weekly decision epoch $t$, the Bayesian-LoRA ensemble produces an epistemic standard deviation $\sigma_{\text{epi}, j, t} \ge 0$ for each asset $j \in \{1, \dots, m\}$.

We consider five candidate scalar aggregation operators:

1. **Unweighted Mean**:
   $$u_t^{\text{mean}} = \frac{1}{m} \sum_{j=1}^m \sigma_{\text{epi}, j, t}$$
2. **Median**:
   $$u_t^{\text{median}} = \operatorname{median}\left(\{\sigma_{\text{epi}, j, t}\}_{j=1}^m\right)$$
3. **Portfolio-Weighted Uncertainty**:
   $$u_t^{\text{weighted}} = \sum_{j=1}^m w_{\text{base}, j, t} \, \sigma_{\text{epi}, j, t}$$
   where $w_{\text{base}, t}$ is the normalized baseline risky portfolio weight vector.
4. **Maximum Uncertainty**:
   $$u_t^{\text{max}} = \max_{j} \sigma_{\text{epi}, j, t}$$
5. **75th Percentile ($q_{75}$)**:
   $$u_t^{q75} = \operatorname{percentile}_{75}\left(\{\sigma_{\text{epi}, j, t}\}_{j=1}^m\right)$$

### Development Set Tuning (2022–2023)
On the 99-week development dataset, tuning over $\lambda_{\text{epi}} \in [0, 300]$ revealed that the **75th percentile ($q_{75}$)** achieved the highest out-of-sample utility ($+0.000919$ weekly net utility). This metric captures aggregate model stress without being overly distorted by a single extreme outlier asset.

---

## 2. Epistemic Exposure Scaling Function

The scalar exposure scale $s_{\text{epi}, t}$ governs the total fraction of wealth allocated to risky assets:
$$s_{\text{epi}, t} = g(u_t; \lambda_{\text{epi}})$$

We utilize the rational decay mapping:
$$s_{\text{epi}, t} = \frac{1}{1 + \lambda_{\text{epi}} u_t}$$

### Mathematical Guarantees:
1. **Bounded Exposure**: $s_{\text{epi}, t} \in [0, 1]$ for all $u_t \ge 0, \lambda_{\text{epi}} \ge 0$.
2. **Monotonic De-risking**:
   $$\frac{\partial s_{\text{epi}, t}}{\partial u_t} = -\frac{\lambda_{\text{epi}}}{(1 + \lambda_{\text{epi}} u_t)^2} \le 0$$
   Higher epistemic uncertainty strictly reduces portfolio leverage.
3. **Deterministic Reproduction**:
   $$\lambda_{\text{epi}} = 0 \implies s_{\text{epi}, t} = 1.0 \quad \forall u_t$$
   When $\lambda_{\text{epi}} = 0$, the strategy identically reproduces the base deterministic portfolio $M_0$.

---

## 3. Preserving Relative Cross-Sectional Proportions

A critical research requirement of Phase 5 is that portfolio-level exposure scaling **must not alter the relative rankings or proportions of the risky assets**.

Given base risky portfolio weights $w_{\text{base}, t} \in \mathbb{R}^m$ such that $\sum_{j=1}^m w_{\text{base}, j, t} = 1.0$ and $w_{\text{base}, j, t} \ge 0$:
$$w_{\text{risky}, j, t} = s_t \cdot w_{\text{base}, j, t}$$
$$w_{\text{cash}, t} = 1.0 - s_t$$

### Invariant Relative Bets:
For any pair of invested risky assets $i, j$ with $w_{\text{base}, j, t} > 0$:
$$\frac{w_{\text{risky}, i, t}}{w_{\text{risky}, j, t}} = \frac{s_t \cdot w_{\text{base}, i, t}}{s_t \cdot w_{\text{base}, j, t}} = \frac{w_{\text{base}, i, t}}{w_{\text{base}, j, t}}$$
The cross-sectional bet structure is 100% invariant to the exposure scale $s_t$.

---

## 4. Cash Return & Transaction Turnover Accounting

To ensure an economically realistic backtest:
1. **Cash Return**: Uninvested capital earns strictly zero return ($r_{\text{cash}} = 0.0$).
2. **Portfolio Return**:
   $$r_{\text{port}, t} = w_{\text{risky}, t}^T r_{t} + w_{\text{cash}, t} \cdot 0 = s_t (w_{\text{base}, t}^T r_t)$$
3. **Exact Cash Turnover**:
   Transitions into and out of cash require rebalancing liquidity. Turnover at epoch $t$ is calculated across both risky assets and the cash sleeve:
   $$\text{Turnover}_t = \sum_{j=1}^m |w_{\text{risky}, j, t} - w_{\text{risky}, j, t-1}| + |w_{\text{cash}, t} - w_{\text{cash}, t-1}|$$
   At 10 bps transaction cost, liquidating $50\%$ into cash and later re-entering incurs explicit turnover drag on both legs.

In Tutorial 23, we contrast this epistemic exposure mechanism against traditional realized-volatility targeting.
