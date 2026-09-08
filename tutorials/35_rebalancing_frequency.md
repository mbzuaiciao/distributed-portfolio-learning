# Tutorial 35: Rebalancing Frequency & Inertia Bands: The Responsiveness Tradeoff

Every market-timing system faces an unavoidable engineering tradeoff:
> **The Responsiveness vs Friction Dilemma**:
> If the system updates exposure rapidly, it incurs ruinous transaction turnover.
> If the system updates exposure too sluggishly, it fails to de-risk before sharp, fast market crashes.

In Phase 8, we systematically analyzed three architectural mechanisms designed to manage this tradeoff:
1. **Inactivity Rebalancing Bands ($\delta, \eta$)**
2. **Periodic Rebalancing (Biweekly vs Monthly) with Emergency Overrides**
3. **Minimum Holding Durations (Lock-in Windows)**

---

## 1. Inertia Rebalancing Bands

Instead of updating the portfolio whenever desired exposure $s^*_t$ changes, an **inertia band** defines a tolerance half-width $\delta$:
$$s_t = \begin{cases} s_{t-1} & \text{if } |s^*_t - s_{t-1}| < \delta \\ s_{t-1} + \eta (s^*_t - s_{t-1}) & \text{if } |s^*_t - s_{t-1}| \ge \delta \end{cases}$$

```
Desired Target vs Executed Exposure:
  Target s*:      0.70  -->  0.73  -->  0.71  -->  0.85  -->  0.83
  Band (delta=0.05):
  Executed s:     0.70  -->  0.70  -->  0.70  -->  0.85  -->  0.85
  Trades Saved:             [SKIP]       [SKIP]                [SKIP]
```

By filtering out minor fluctuations (changes $< \delta$), the portfolio remains stable during quiet market conditions while responding decisively to large shocks.

---

## 2. Periodic Rebalancing with Emergency Overrides

Rather than checking and trading every week, an investor can rebalance on a fixed schedule (e.g., biweekly or monthly).

However, pure low-frequency rebalancing is dangerous in financial markets: a severe geopolitical or macroeconomic crisis could occur in Week 1 of a monthly cycle, and the portfolio would remain trapped in full exposure for three more weeks.

To resolve this, Phase 8 introduced the **Severe-Spike Emergency Override**:
$$s_t = \begin{cases} s_{\text{emergency}} & \text{if } z_{\text{epi}, t} > \tau_{\text{severe}} \quad (\text{Emergency Override}) \\ s^*_t & \text{if } t \pmod{\text{frequency}} = 0 \quad (\text{Regular Scheduled Date}) \\ s_{t-1} & \text{otherwise} \end{cases}$$

This guarantees that extreme tail-risk signals ($\tau_{\text{severe}} = 2.0$) trigger immediate de-risking regardless of the calendar schedule, while ordinary adjustments wait for regular rebalance dates.

---

## 3. The Danger of Over-Smoothing: Why Did $M_{17}$ Falter?

In Phase 8, we tested a two-stage combined policy ($M_{17}$):
$$\text{Raw Signal} \longrightarrow \text{Hysteresis State Machine} \longrightarrow \text{Rebalancing Band } (\delta = 0.20)$$

On development data, this combination appeared promising. However, on the 2024–2025 out-of-sample holdout, $M_{17}$ delivered inferior performance:
- Sharpe: $1.492$ (vs $1.702$ for pure Hysteresis $M_{13}$ and $1.508$ for constant $M_8$).
- Net Economic Benefit: $-3.23\%$ (underperforming constant exposure).

### Why Did Chaining Band on Top of Hysteresis Fail?
1. **Double Lags**: Hysteresis already creates stickiness. Applying an additional wide dead-band ($\delta = 0.20$) meant that once the portfolio de-risked, the re-risking signal could not jump over the $\delta$ threshold when the market began its sharp recovery.
2. **Upside Lock-Out**: The strategy stayed defensive for too long, completely missing the post-August 2024 tech rebound.
3. **The Principle of Parsimony**:
   > *A single, well-calibrated execution rule (Hysteresis $M_{13}$) is vastly superior to stacking multiple layers of smoothing.*
   Over-filtering destroys the timing value by turning dynamic risk management into an unresponsive, lagging anchor.
