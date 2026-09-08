# Tutorial 26: Matched-Exposure Timing & Risk-Budget Controls

A central methodological flaw in evaluating timing strategies is **confounding dynamic timing with unconditional risk-budget shifts**:
> *If strategy A averages $30\%$ equity exposure and strategy B averages $100\%$ equity exposure, strategy A will naturally have lower volatility and shallower drawdowns in almost any volatile asset class. Comparing their drawdowns directly without matching average exposure tells us nothing about whether strategy A had good timing.*

To isolate true timing skill from risk-budget scaling, Phase 6 enforces **exact mean-exposure matching**.

---

## 1. The Five Matched Policies

For any given target exposure budget $\bar{s} \in [0.30, 0.50, 0.70, 0.90]$:

1. **$M_0$ — Full-Risk Baseline**: Fixed 100% investment in base risky portfolio ($s_t = 1.0$).
2. **$M_8$ — Constant Exposure Baseline**:
   $$s_{8, t} = \bar{s} \quad \forall t$$
   Allocates a fixed fraction $\bar{s}$ to the base risky portfolio and $1 - \bar{s}$ to cash ($r_{\text{cash}} = 0.0$). $M_8$ has **zero timing skill** by definition, serving as the pure risk-budget control.
3. **$M_9$ — Relative Epistemic Timing**:
   $$s_{9, t} = \operatorname{clip}\left(s_{\text{base}} - \lambda_{\text{rel}} \max(0, z_{\text{epi}, t} - \tau), \, 0.0, \, 1.0\right)$$
   Where $z_{\text{epi}, t}$ is the robust trailing z-score. Parameters $(s_{\text{base}}, \lambda_{\text{rel}})$ are calibrated on development data to achieve average exposure equal to $\bar{s}$.
4. **$M_{10}$ — Matched Volatility Timing**:
   $$s_{10, t} = \min\left(1.0, \, \frac{\sigma_{\text{target}}}{\hat{\sigma}_{\text{hist}, t}}\right)$$
   Target volatility $\sigma_{\text{target}}$ is calibrated so that the mean exposure of $M_{10}$ matches $\bar{s}$ on holdout.
5. **$M_{11}$ — Placebo / Permuted Timing Control**:
   1,000 randomized exposure sequences that preserve the exact empirical distribution of $M_9$ while scrambling its temporal alignment.
6. **$M_{12}$ — Combined Relative Epistemic + Volatility Timing**:
   $$s_{12, t} = \min(s_{9, t}, s_{10, t}) \cdot \alpha_{\text{match}}$$
   Calibrated to match the same average exposure $\bar{s}$.

---

## 2. Invariance of Relative Cross-Sectional Asset Weights

Across all policies ($M_0, M_8, M_9, M_{10}, M_{11}, M_{12}$), the underlying cross-sectional relative weights are **100% identical**:
$$w_{\text{risky}, j, t} = s_t \cdot w_{\text{base}, j, t}$$
$$w_{\text{cash}, t} = 1.0 - s_t$$

Where $w_{\text{base}, t}$ is the normalized Markowitz mean-variance portfolio solving:
$$\min_w w^T Q w - \hat{\mu}^T w \quad \text{s.t.} \quad \sum_j w_j = 1, \, w_j \ge 0$$

All divergence in performance arises strictly from **when and how total portfolio leverage $s_t$ fluctuates**.

---

## 3. The Decisive Comparison: $M_9$ vs $M_8$

The primary hypothesis test of Phase 6 is:
$$\Delta \text{Sharpe} = \text{Sharpe}(M_9) - \text{Sharpe}(M_8)$$
$$\Delta \text{Utility} = U(M_9) - U(M_8)$$

- If $\Delta \text{Sharpe} \le 0$ and $\Delta U \le 0$, then epistemic uncertainty adds **no timing value beyond average de-risking**.
- If $\Delta \text{Sharpe} > 0$ and $\Delta U > 0$ with statistical significance over $M_8$ and $M_{10}$, then relative epistemic uncertainty possesses genuine dynamic market timing skill.

In Tutorial 27, we present the circular block placebo test and the empirical holdout findings.
