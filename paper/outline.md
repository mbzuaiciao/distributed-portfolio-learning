# Paper Outline: Temporal Regularization in Portfolio Execution

**Title**: *Temporal Regularization in Portfolio Execution: Optimal Holding Horizons Under Signal Decay and Trading Friction*  
**Date**: October 2026

---

## Section-by-Section Structural Specification

### 1. Introduction
- Motivation: The quantitative portfolio dilemma between forecast freshness and implementation friction.
- The Mechanics of Overlapping Cohort Portfolios ("Cohort Stacking"): Partitioning capital across $H$ staggered sub-portfolios.
- The Core Question: What determines the optimal cohort holding horizon $H^*$ as a function of signal persistence ($\tau$), observation noise ($\sigma_{\text{noise}}$), signal strength ($\beta$), transaction costs ($c$), and risk aversion ($\gamma$)?
- Summary of Contributions:
  1. Analytical utility model formalizing alpha dilution, cost amortization, and temporal noise diversification.
  2. Controlled synthetic simulation across 360,000 evaluations confirming 5/5 comparative static hypotheses.
  3. Machine-precision accounting closure ($3.14\times 10^{-17}$) and exact-null invariant audit.
  4. Reconciliation of empirical post-earnings findings: showing $H=4$ survival was an architectural regularizer effect rather than persistent predictive alpha.

### 2. Related Setting & Evolutionary Motivation
- Origin within distributed portfolio optimization, hypergraphs, and execution stability.
- How rigorous forensic audits of earlier ML uncertainty approaches (target leakage, entropy-error metric artifacts) shifted the research focus from predictive uncertainty modeling to execution architecture.

### 3. Temporal Regularization Framework
- Formal mechanics of discrete-time rolling cohort portfolios.
- Mathematical derivation of stylized expected utility:
  $$U(H) = \mu_0 \frac{1 - \rho^H}{H(1 - \rho)} - \frac{c \cdot \text{TO}_0}{H} - \frac{1}{2}\gamma \left(\sigma_0^2 + \frac{\sigma_1^2}{H}\right)$$
- Detailed exposition of the three fundamental economic forces:
  1. Alpha Dilution ($\partial \text{Alpha}/\partial H \le 0$)
  2. Cost Amortization ($\partial \text{Cost}/\partial H < 0$)
  3. Temporal Diversification ($\partial \text{Risk}/\partial H \le 0$)

### 4. Analytical Comparative Statics & Break-Even Frontiers
- Derivation of five primary comparative statics:
  - Cost monotonicity: $\partial H^* / \partial c \ge 0$
  - Persistence monotonicity: $\partial H^* / \partial \rho \ge 0$
  - Short-lived alpha reversal: $H^* = 1$ at $\rho=0, c=0$; $\partial H^* / \partial \beta \le 0$ under fast decay
  - Noise regularization: $\partial H^* / \partial \sigma_{\text{noise}} \ge 0$
  - Risk aversion preference: $\partial H^* / \partial \gamma \ge 0$
- Derivation of closed-form break-even cost $c^*(H)$ and break-even alpha $\mu_0^*(H)$ boundaries.

### 5. Controlled Synthetic Experiment Design (Phase 20)
- Motivation for a controlled data-generating process (DGP) independent of historical empirical quirks.
- Preregistered design invariances:
  - Single realized market return per asset per calendar week shared across all cohorts (preventing cohort-specific idiosyncratic shocks or oracle leaks).
  - Controlled latent predictive signal with geometric decay factor $\rho$ and continuous half-life $\tau_{1/2}$.
  - Additive ranking noise $\sigma_{\text{noise}}$ and signal magnitude $\beta$.
  - 52-week finite signal history truncation.
  - Frozen factorial grid: $H \in \{1, 2, 4, 8, 13\}$, $c \in \{0, 5, 10, 25, 50\}\text{ bps}$, $\gamma \in \{0, 0.5, 1.0, 2.5, 5.0\}$, 30 independent seeds ($N = 360{,}000$ evaluations).
  - Preregistered numerical closure tolerance ($|\text{error}| < 10^{-12}$) and paired $t$-test tie-handling protocol.

### 6. Simulation Results
- Empirical confirmation of all five comparative static hypotheses (H1: 100%, H2: 90%, H3: 100%, H4: 100%, H5: 100%).
- Accounting closure verified to machine precision ($\max |\text{error}| = 3.14\times 10^{-17}$).
- Invariant audit of the exact-null regime ($\beta = 0.000$): exact 5-way mathematical tie ($TO \equiv 0, \Delta U \equiv 0$) with zero discrepancy ($0.00\text{e}+00$).
- Identification of four principal horizon regimes: Short-Horizon Dominance ($H=1$), Intermediate Regularization ($H=2\text{--}4$), Extended Stacking ($H=8\text{--}13$), and the Exact-Null Tie Regime.
- Formal classification: **Outcome A**.

### 7. Empirical Reconciliation (Phases 18 & 19)
- Historical context: post-earnings announcement portfolios with quarterly earnings events.
- Clean revalidation (Phase 18): residual alpha across weeks 2–4 failed significance ($p=0.105$), yet $H=4$ delivered large Sharpe gains.
- Architecture-general stress test (Phase 19): turnover and volatility savings occurred across random, constant, and synthetic signals, while utility gains reversed under strong alpha.
- Parameter mapping: institutional parameters ($c = 10\text{ bps}$, Rank IC $\approx 0.05$, $\tau_{1/2} \approx 1\text{--}2.4\text{ wks}$) map directly into the synthetic intermediate regularization regime ($H^* \ge 2\text{--}4$).
- Resolving the empirical paradox: cohort stacking survived as an architectural regularizer, not as an alpha-capture mechanism.

### 8. Evolutionary Provenance & Falsification Summary
- Transparent documentation of prior negative branches:
  - Bayesian LoRA epistemic uncertainty audits (target leakage, entropy-error metric artifact, single-pass equivalence).
  - Collaborative LoRA branch closure (failure of heterogeneous specialist weighting).
  - Absence of independent prospective 2026 data.

### 9. Limitations
- Synthetic DGP idealizations (linear Gaussian decay, stationary volatility, uncorrelated asset residuals).
- Simple proportional transaction cost model (omits nonlinear price impact).
- Static horizon selection rather than dynamic time-varying horizon estimation.
- Finite memory truncation at $\rho = 0.97$.
- Retrospective empirical mapping; lack of real-time prospective live trading execution.

### 10. Conclusion & Design Principles
- Overlapping cohorts as a general temporal regularization mechanism.
- Practical rules of thumb for quantitative portfolio managers: measuring half-life before setting horizons, using stacking to dampen panel churn, and recognizing the friction-decay frontier.
