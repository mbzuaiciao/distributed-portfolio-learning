# Tutorial 38: Auditing Economic Attribution & Exact Closure Identity

In portfolio evaluation, decomposing return differences between an active strategy and a benchmark reveals *why* a policy outperformed or underperformed. However, if the attribution components are not mutually exclusive and exhaustive, attribution numbers become misleading.

Phase 9 audits the attribution mechanics and reconstructs an exact, non-overlapping 4-quadrant decomposition of timing P&L.

---

## 1. The Phase 8 Attribution Flaw

In Phase 8, timing attribution was computed via:
```python
down_mask = r_arr < 0.0
avoided_losses = np.sum((c - s_arr[down_mask]) * (-r_arr[down_mask]))

up_mask = r_arr > 0.0
sacrificed_upside = np.sum((c - s_arr[up_mask]) * r_arr[up_mask])
bonus_upside = np.sum((s_arr[up_mask] - c) * r_arr[up_mask])
```

Notice the algebraic identity between `sacrificed_upside` and `bonus_upside`:
$$\text{bonus\_upside} = \sum_{r > 0} (s_t - c) r_t = - \sum_{r > 0} (c - s_t) r_t = - \text{sacrificed\_upside}$$

Because neither formula conditioned on whether exposure was greater or less than the benchmark $c$, `bonus_upside` was mathematically identical to `-sacrificed_upside`! Furthermore, when $s_t > c$ during down markets ($r < 0$), `(c - s_t) * (-r_t)` was negative, conflating extra downside suffered with a reduction in avoided losses.

---

## 2. Rebuilding the 4-Quadrant Mutually Exclusive Decomposition

Let $r_{\text{base}, t}$ be the return of the base risky portfolio, $s_t$ be the dynamic exposure of the active policy, and $c_{\text{const}}$ be the constant exposure benchmark.

The difference in gross return for period $t$ is:
$$\Delta r_t = (s_t - c_{\text{const}}) r_{\text{base}, t}$$

We define four non-overlapping, strictly positive components based on the joint signs of $\Delta s_t = s_t - c_{\text{const}}$ and $r_{\text{base}, t}$:

```
                              Market Return (r_base, t)
                           Negative (< 0)        Positive (>= 0)
                     +-----------------------+-----------------------+
   Exposure Relative | (1) Losses Avoided    | (2) Upside Sacrificed |
   to Benchmark      |     (L_avoid > 0)     |     (U_sac > 0)       |
   (Delta s_t < 0)   |  (c - s_t) * (-r_t)   |   (c - s_t) * r_t     |
                     +-----------------------+-----------------------+
   Exposure Relative | (4) Extra Downside    | (3) Extra Upside      |
   to Benchmark      |     (D_suf > 0)       |     (U_cap > 0)       |
   (Delta s_t > 0)   |  (s_t - c) * (-r_t)   |   (s_t - c) * r_t     |
                     +-----------------------+-----------------------+
```

### Component Definitions:
1. **Losses Avoided ($L_{\text{avoid}, t}$)**:
   $$L_{\text{avoid}, t} = \mathbb{I}_{\{\Delta s_t < 0, r_t < 0\}} \cdot (c - s_t)(-r_t) \ge 0$$
   Capital preserved by holding reduced exposure during a market decline.
2. **Upside Sacrificed ($U_{\text{sac}, t}$)**:
   $$U_{\text{sac}, t} = \mathbb{I}_{\{\Delta s_t < 0, r_t \ge 0\}} \cdot (c - s_t)(r_t) \ge 0$$
   Rally gains missed by holding reduced exposure when the market surged.
3. **Extra Upside Captured ($U_{\text{cap}, t}$)**:
   $$U_{\text{cap}, t} = \mathbb{I}_{\{\Delta s_t > 0, r_t \ge 0\}} \cdot (s_t - c)(r_t) \ge 0$$
   Additional gains earned when scaled exposure exceeded the constant benchmark during an up market.
4. **Extra Downside Suffered ($D_{\text{suf}, t}$)**:
   $$D_{\text{suf}, t} = \mathbb{I}_{\{\Delta s_t > 0, r_t < 0\}} \cdot (s_t - c)(-r_t) \ge 0$$
   Additional losses incurred when scaled exposure exceeded the constant benchmark during a down market.

---

## 3. Proof of Exact Algebraic Closure

In every period $t$, exactly one quadrant is active (or none if $\Delta s_t = 0$ or $r_{\text{base}, t} = 0$):
$$\Delta r_t = (s_t - c) r_t \equiv L_{\text{avoid}, t} - U_{\text{sac}, t} + U_{\text{cap}, t} - D_{\text{suf}, t}$$

Subtracting incremental transaction costs $\Delta \text{TC}_t = \text{TC}_{\text{policy}, t} - \text{TC}_{\text{constant}, t}$:
$$\Delta \text{PnL}_t = L_{\text{avoid}, t} - U_{\text{sac}, t} + U_{\text{cap}, t} - D_{\text{suf}, t} - \Delta \text{TC}_t$$

Summing over all $T$ periods:
$$\text{Net Economic Benefit} \equiv \sum_{t=1}^T L_{\text{avoid}, t} - \sum_{t=1}^T U_{\text{sac}, t} + \sum_{t=1}^T U_{\text{cap}, t} - \sum_{t=1}^T D_{\text{suf}, t} - \sum_{t=1}^T \Delta \text{TC}_t$$

This identity closes to machine precision ($< 10^{-16}$).

---

## 4. Empirical Rebuilt Attribution Results (69-Week Holdout)

| Policy | Losses Avoided | Upside Sacrificed | Extra Upside Captured | Extra Downside Suffered | Gross Return Diff | Incremental TC | Net Economic Benefit | Closure Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$M_8$ Constant** | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.0$ |
| **$M_9$ Raw Timing** | $+11.29\%$ | $-14.43\%$ | $+12.25\%$ | $-4.93\%$ | $+4.18\%$ | $-0.03\%$ | **$+4.21\%$** | $6.2 \times 10^{-17}$ |
| **$M_{10}$ Vol Timing** | $+4.90\%$ | $-11.31\%$ | $+3.54\%$ | $-2.62\%$ | $-5.50\%$ | $+0.02\%$ | **$-5.52\%$** | $0.0$ |
| **$M_{13}$ Hysteresis** | **$+10.28\%$** | **$-12.41\%$** | **$+12.24\%$** | **$-5.36\%$** | **$+4.74\%$** | **$-0.03\%$** | **$+4.77\%$** | $1.4 \times 10^{-17}$ |

### Mechanistic Takeaways:
1. **$M_{13}$ Slashes Sacrificed Upside**: By requiring uncertainty to drop below $\tau_{\text{exit}}=0.2$ before restoring exposure, $M_{13}$ exits cleanly and avoids staying depressed during market rallies, reducing sacrificed upside from $-14.43\%$ ($M_9$) to **$-12.41\%$** ($M_{13}$).
2. **Net Economic Advantage**: $M_{13}$ delivers a higher net economic benefit (**$+4.77\%$** vs $+4.21\%$ for $M_9$) and completely avoids the devastating drag of volatility timing ($-5.52\%$).
3. **Flawless Accounting Closure**: All closure errors are on the order of $10^{-17}$, confirming that the rebuilt decomposition is mathematically airtight.
