# Tutorial 34: Transaction-Cost-Aware Timing & Friction Hurdle Rates

A classic trap in quantitative strategy design is **optimizing on gross metrics**. A market timing strategy may produce an impressive gross Sharpe ratio of $2.10$, only to deliver negative alpha once subjected to realistic trading frictions.

In Phase 8, we formalize the **turnover tax** and implement a strictly cost-aware evaluation protocol.

---

## 1. The Turnover Tax on Dynamic Exposure

When an investor holds a constant matched exposure ($\bar{s} = 0.70$), portfolio turnover arises only from internal cross-sectional rebalancing among the risky assets.

When an investor dynamically adjusts cash exposure $s_t$:
$$\Delta s_t = |s_t - s_{t-1}|$$
they must execute trades in both the risky sleeve and the cash sleeve:
$$\text{Cash-Sleeve Turnover}_t = 2 \cdot |\Delta s_t|$$

For a proportional transaction fee $c_{\text{bps}}$ (e.g., $10\text{ bps} = 0.0010$ per unit traded), the cumulative friction drag over $T$ weeks is:
$$\text{Cost Drag} = \sum_{t=1}^T c_{\text{bps}} \cdot 2 |\Delta s_t|$$

If a strategy adjusts exposure by an average of $15\%$ every week:
$$\text{Annual Turnover} \approx 52 \times 2 \times 0.15 = 15.6 \times \text{ Portfolio Value}$$
At $10$ bps, this imposes an unconditional drag of:
$$15.6 \times 0.0010 = 1.56\% \text{ per year}$$

To create economic value, the **gross benefit of avoided losses must exceed this $156$ bps hurdle rate**.

---

## 2. Tuning on Net Utility ($U_{\text{net}}$) vs Gross Sharpe

In Phase 8, all hyperparameter searches on development data optimize for **Net Realized Quadratic Utility**:
$$U_{\text{net}} = r_t - \frac{\gamma}{2} \sigma_t^2 - \text{tc}_t$$
with $\gamma = 1.0$ and $c = 10\text{ bps}$.

### Why Tuning on Gross Sharpe Fails
- **Gross Sharpe** rewards strategies that make tiny, high-frequency tweaks to volatility, even if each tweak trades only $1\%$ of the portfolio.
- Over 52 weeks, 52 tiny trades accumulate transaction drag that destroys net wealth.
- **Net Utility** penalizes turnover directly, forcing the optimization algorithm to select wider hysteresis bands and higher persistence thresholds that only trade when high conviction exists.

---

## 3. Empirical Cost Sensitivity Surface (0 to 50 bps)

Evaluating across five transaction cost assumptions on the 2024–2025 holdout:

| Cost Level | $M_8$ Constant Sharpe | $M_9$ Raw Timing Sharpe | $M_{13}$ Hysteresis Sharpe | $M_9$ Net Utility | $M_{13}$ Net Utility | Winning Policy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 bps (Zero Cost)** | $1.860$ | $2.075$ | $2.074$ | $+0.001909$ | **$+0.002054$** | **$M_{13}$ Hysteresis** |
| **5 bps (Liquid)** | $1.684$ | $1.881$ | **$1.888$** | $+0.000330$ | **$+0.000556$** | **$M_{13}$ Hysteresis** |
| **10 bps (Baseline)**| $1.508$ | $1.687$ | **$1.702$** | $-0.001250$ | **$-0.000941$** | **$M_{13}$ Hysteresis** |
| **25 bps (Mid-Cap)** | $0.978$ | $1.103$ | **$1.141$** | $-0.005990$ | **$-0.005434$** | **$M_{13}$ Hysteresis** |
| **50 bps (Illiquid)**| $0.096$ | $0.126$ | **$0.201$** | $-0.013890$ | **$-0.012923$** | **$M_{13}$ Hysteresis** |

```
Sharpe Ratio Decay Across Costs:
  M13 Hysteresis:  2.074 (0 bps) --> 1.702 (10 bps) --> 1.141 (25 bps) --> 0.201 (50 bps)
  M9 Raw Timing:   2.075 (0 bps) --> 1.687 (10 bps) --> 1.103 (25 bps) --> 0.126 (50 bps)
  M8 Constant:     1.860 (0 bps) --> 1.508 (10 bps) --> 0.978 (25 bps) --> 0.096 (50 bps)
```

### Critical Observations
1. **Raw $M_9$ Degrades Faster Than Constant Baseline**:
   At $25$ bps, Raw $M_9$'s net utility falls below $M_8$ ($-0.00599$ vs $-0.00583$), proving that unconstrained high-turnover timing is economically unviable in illiquid assets.
2. **Hysteresis ($M_{13}$) Maintains Its Edge Up to 50 bps**:
   Because $M_{13}$ cuts exposure adjustments by over $40\%$, it preserves positive net delta Sharpe over $M_8$ all the way to 50 bps ($0.201$ vs $0.096$).
3. **Turnover Cost Savings**:
   $M_{13}$ saves **$+121.4$ basis points of trading costs** relative to $M_9$ while actually *increasing* gross benefit retained ($113.5\%$) by dodging whipsaw traps.
