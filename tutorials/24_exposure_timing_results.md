# Tutorial 24: Empirical Holdout Results & Scientific Verdict on Exposure Timing

Phase 5 investigated whether Bayesian-LoRA epistemic uncertainty can generate incremental economic value through portfolio-level exposure timing ($s_t \in [0, 1]$ into cash).

We evaluated four strategies across three random seeds (`42`, `123`, `777`) on the **2024–2025 frozen holdout** (69 weekly rebalancing decision epochs):
- **$M_0$ (Deterministic)**: $s_t = 1.0$ (fixed full exposure).
- **$M_5$ (Epistemic Scaling)**: $s_{\text{epi}, t} = 1 / (1 + \lambda_{\text{epi}}^* u_t)$ with $u_t = q_{75}(\sigma_{\text{epi}})$ and $\lambda_{\text{epi}}^* = 300.0$ frozen from dev tuning.
- **$M_6$ (Volatility Targeting)**: $s_{\text{vol}, t} = \min(1.0, \sigma_{\text{target}}^* / \hat{\sigma}_{\text{hist}, t})$ with $\sigma_{\text{target}}^* = 0.15$ frozen from dev tuning.
- **$M_7$ (Combined Scaling)**: $s_{\text{combined}, t} = \min(s_{\text{epi}, t}, s_{\text{vol}, t})$.

---

## 1. Out-of-Sample Performance Summary (3 Seeds Mean $\pm$ Std)

| Metric | $M_0$ (Deterministic) | $M_5$ (Epistemic) | $M_6$ (Vol Target) | $M_7$ (Combined) |
| :--- | :---: | :---: | :---: | :---: |
| **Annualized Return** | $\mathbf{43.80\% \pm 6.59\%}$ | $14.10\% \pm 3.24\%$ | $26.03\% \pm 5.70\%$ | $14.07\% \pm 3.22\%$ |
| **Annualized Volatility** | $29.08\% \pm 1.13\%$ | $\mathbf{8.66\% \pm 0.77\%}$ | $17.99\% \pm 0.27\%$ | $\mathbf{8.66\% \pm 0.77\%}$ |
| **Sharpe Ratio** | $1.508 \pm 0.225$ | $\mathbf{1.606 \pm 0.248}$ | $1.444 \pm 0.306$ | $1.603 \pm 0.246$ |
| **Sortino Ratio** | $2.843 \pm 0.479$ | $\mathbf{3.076 \pm 0.579}$ | $2.587 \pm 0.658$ | $3.070 \pm 0.574$ |
| **Calmar Ratio** | $2.896 \pm 0.357$ | $\mathbf{3.122 \pm 0.618}$ | $2.473 \pm 0.510$ | $3.116 \pm 0.613$ |
| **Max Drawdown** | $-15.12\%$ | $\mathbf{-4.49\%}$ | $-10.49\%$ | $\mathbf{-4.49\%}$ |
| **Mean Risky Exposure** | $100.0\%$ | $30.08\%$ | $57.89\%$ | $30.05\%$ |
| **Mean Net Utility** | $-0.00621 \pm 0.00140$ | $\mathbf{+0.00090 \pm 0.00039}$ | $-0.00068 \pm 0.00100$ | $\mathbf{+0.00090 \pm 0.00039}$ |

---

## 2. Statistical Significance via Circular Block Bootstrap

To account for serial correlation and regime clustering, we conducted paired circular block bootstrap evaluations ($B = 4$ weeks, 1,000 resamples) measuring $\Delta \text{Sharpe} = \text{Sharpe}_{\text{policy}} - \text{Sharpe}_{M_0}$ and $\Delta \text{Utility} = U_{\text{policy}} - U_{M_0}$:

| Seed | Policy | $\Delta \text{Sharpe}$ Point | 95% Bootstrap CI | $\Delta \text{Utility}$ Point | 95% Bootstrap CI | $p$-value ($U$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **42** | $M_5$ (Epistemic) | $+0.0639$ | $[-0.1497, +0.2912]$ | $+0.008554$ | $[+0.000832, +0.015472]$ | $\mathbf{0.036}$ |
| | $M_6$ (Vol Target) | $-0.1742$ | $[-0.4526, +0.1471]$ | $+0.006186$ | $[+0.002553, +0.009637]$ | $\mathbf{0.006}$ |
| | $M_7$ (Combined) | $+0.0639$ | $[-0.1316, +0.2994]$ | $+0.008554$ | $[+0.001081, +0.015322]$ | $\mathbf{0.016}$ |
| **123**| $M_5$ (Epistemic) | $+0.0789$ | $[-0.0689, +0.2737]$ | $+0.006793$ | $[+0.001089, +0.012594]$ | $\mathbf{0.014}$ |
| | $M_6$ (Vol Target) | $+0.0171$ | $[-0.2403, +0.3370]$ | $+0.005908$ | $[+0.001720, +0.009759]$ | $\mathbf{0.000}$ |
| | $M_7$ (Combined) | $+0.0762$ | $[-0.0773, +0.2368]$ | $+0.006793$ | $[+0.000726, +0.013001]$ | $\mathbf{0.030}$ |
| **777**| $M_5$ (Epistemic) | $+0.1521$ | $[-0.0524, +0.3812]$ | $+0.005987$ | $[-0.000422, +0.012103]$ | $0.066$ |
| | $M_6$ (Vol Target) | $-0.0342$ | $[-0.3903, +0.3847]$ | $+0.004484$ | $[+0.000965, +0.007599]$ | $\mathbf{0.002}$ |
| | $M_7$ (Combined) | $+0.1473$ | $[-0.0605, +0.4314]$ | $+0.005981$ | $[-0.000208, +0.011833]$ | $0.068$ |

### Interpretation of Statistical Results:
1. **Utility Improvement**: In Seeds 42 and 123, $M_5$ achieves a statistically significant improvement in mean net utility ($p = 0.036$ and $p = 0.014$). In Seed 777, it is marginally significant ($p = 0.066$).
2. **Sharpe Ratio CI Spans Zero**: While the point estimate for $\Delta \text{Sharpe}$ is positive across all three seeds ($+0.064$ to $+0.152$), the 95% bootstrap confidence intervals for $\Delta \text{Sharpe}$ span zero in every seed (e.g. $[-0.15, +0.29]$).
3. **Volatility Reduction vs Absolute Return**: $M_5$ slashes maximum drawdown by $70\%$ (from $-15.1\%$ down to $-4.5\%$) and reduces annualized volatility from $29.1\%$ to $8.7\%$, but it sacrifices annualized nominal return ($14.1\%$ vs $43.8\%$).

---

## 3. Information Value & Exposure Attribution

Why does $M_5$ improve utility, and is it genuinely timing the market?

### Predictive Information (Part J & K)
- **Forecast Error Magnitude**: Aggregate uncertainty $u_t$ significantly predicts next-week mean absolute forecast error ($\text{Spearman } \rho \in [0.153, 0.299]$, $p < 0.05$ in 2 of 3 seeds).
- **Directional Loss**: The AUROC for predicting negative return weeks using $u_t$ is weak: $\text{AUROC} \in [0.54, 0.60]$.
- **Downside Tail Loss**: The AUROC for predicting tail loss weeks ($r < -2\%$) using $u_t$ ranges between $0.45$ and $0.63$.
- **Incremental Value Regression**: In the bivariate model $\text{Future Loss} \sim \beta_{\text{vol}} \hat{\sigma}_{\text{hist}} + \beta_u u_t$, the coefficient $\beta_u$ is positive across all seeds ($+0.0013$ to $+0.0027$).

### Exposure Attribution (Part M)
Comparing average risky exposure in the worst 10% weeks vs best 10% weeks reveals:
- **Worst 10% Weeks (Downside Defense)**: $M_5$ exposure was $\mathbf{28.8\%}$.
- **Best 10% Weeks (Upside Capture)**: $M_5$ exposure was $\mathbf{31.7\%}$.
- **Full Period Average**: $M_5$ exposure was $\mathbf{30.1\%}$.

Notice that $M_5$ holds almost identical exposure in both the worst weeks ($28.8\%$) and best weeks ($31.7\%$). The massive reduction in volatility and drawdown is largely driven by **structural de-risking** (holding $\sim 70\%$ cash) rather than acute, highly selective market timing.

---

## 4. The Scientific Decision Tree Verdict

We evaluate the experimental results against the predefined Decision Tree:

- **Outcome A (Pure Epistemic Dominance)**: $M_5$ significantly outperforms $M_6$ and $M_0$ across Sharpe, drawdown, and utility with non-zero CI on Sharpe. *(Rejected: $\Delta \text{Sharpe}$ 95% CI spans zero).*
- **Outcome B (Complementary Value)**: $M_7$ strictly outperforms both $M_5$ and $M_6$. *(Rejected: $M_7$ performance is virtually identical to $M_5$ because $s_{\text{epi}} \approx 0.30 \le s_{\text{vol}} \approx 0.58$).*
- **Outcome C (Risk-Targeting Equivalence / Proxy)**: Epistemic scaling matches volatility targeting because $u_t$ is collinear with $\sigma_{\text{hist}}$. *(Rejected: $\rho(u_t, \sigma_{\text{hist}}) < 0.28$, and $M_5$ achieves higher Sharpe $1.61$ vs $M_6$'s $1.44$).*
- **Outcome D (Structural De-risking without True Timing)**: Epistemic scaling reduces volatility and drawdown, producing positive net utility gains due to risk aversion, but does not achieve statistically significant Sharpe expansion over $M_0$ because it scales down upside and downside equally.
- **Outcome E (Epistemic Failure)**: Exposure scaling hurts performance. *(Rejected: $M_5$ achieves higher Sharpe, Sortino, Calmar, and Utility).*

### Final Verdict: **Outcome D (Structural De-risking without Dynamic Market Timing)**
Bayesian-LoRA epistemic uncertainty successfully regulates portfolio risk and dramatically dampens crash drawdowns, improving net quadratic utility. However, because $u_t$ predicts forecast error *magnitude* rather than *directional drops*, epistemic scaling de-risks uniformly across market regimes, dampening bull run upside along with bear crashes.

Consequently, epistemic exposure scaling provides a valid conservative risk dampener, but does **not** deliver a statistically significant Sharpe expansion over buy-and-hold base Markowitz.
