# Study Limitations & Boundary Conditions

This document outlines the explicit boundary conditions and limitations of the findings presented in the consolidated paper.

---

## 1. Synthetic Simulation Assumptions (Phase 20)

1. **Controlled Synthetic DGP Idealizations**:
   - The data-generating process assumes stationary return volatility, homoskedastic asset shocks ($\sigma_{\text{mkt}} = 0.02, \sigma_{\text{idio}} = 0.04$), and zero cross-asset correlation spikes. Real financial markets exhibit volatility clustering (GARCH effects), fat-tailed return distributions, structural breaks, and jump processes.
2. **Linear-Gaussian Signal Decay**:
   - Latent information follows a linear AR(1) process with geometric decay ($s_{t+k} = \rho^k s_t$). In financial markets, information dissipation can be non-exponential, regime-dependent, or discrete (e.g., following earnings revisions, macro announcements, or analyst downgrades).
3. **Proportional Linear Transaction Costs**:
   - Trading costs are modeled as linear frictions ($c \cdot \text{TO}$). In institutional asset management, large orders incur non-linear square-root market impact costs and execution slippage, which would further penalize rapid single-period turnover and increase the relative advantage of longer horizons.
4. **Absence of Liquidity and Capacity Constraints**:
   - The simulation assumes infinite liquidity and zero market depth constraints at cost rate $c$. In live trading, market capacity limits position sizes and restricts rapid rotation, particularly in small- and mid-cap equities.
5. **Static Holding Horizons**:
   - The framework evaluates constant holding horizons $H \in \{1, 2, 4, 8, 13\}$. In practice, adaptive strategies that dynamically expand horizons during low-signal or high-volatility regimes could provide further economic efficiencies.
6. **Discrete Candidate Grid**:
   - Optimal horizons are selected over a discrete grid ($H \in \{1, 2, 4, 8, 13\}$) rather than continuous or fractional holding periods.
7. **Finite-Memory Truncation**:
   - For high-persistence signals ($\rho = 0.97$), truncating latent history at $K = 52$ calendar weeks captures $\approx 80\%$ of asymptotic signal weight ($0.97^{52} \approx 0.2032$). While sufficient for asset ranking, infinite-memory AR(1) dynamics would show slightly higher persistence.

---

## 2. Empirical Scope & Revalidation Boundaries (Phases 18 & 19)

8. **Reconstructed Historical Weekly Returns**:
   - Phase 18 evaluated geometrically interpolated event-endpoint returns rather than continuous observed weekly market marks. Weeks 2–4 share the same per-event drift, meaning that multi-week returns reflect endpoint interpolation. True daily execution marks are required to confirm intraweek liquidity and price dynamics.
9. **Retrospective Holdout Scope**:
   - All empirical analyses are evaluated on retrospective historical data terminating on May 15, 2025. As established in the Phase 10 provenance audit, no independent post-2025 prospective market data was available.
10. **Absence of Live Execution**:
    - Backtests incorporate historical pricing, execution delays, and proportional fee assumptions, but do not reflect live order routing, order book queue priority, short-selling borrow availability, or borrow fees.

---

## 3. Negative & Falsified Results Scope (Phases 11–17)

1. **Language Model Representation**:
   - Bayesian LoRA and deterministic LoRA were evaluated on minimal metadata inputs (company, ticker, date) rather than full earnings call transcripts or dense fundamental filings.
2. **Task Scope**:
   - Evaluated on binary downside classification ($20\%$ tail-risk threshold). The failure of Bayesian LoRA to beat a constant base-rate null does not imply that language models cannot predict continuous returns or alternative targets.
3. **Collaborative LoRA**:
   - Evaluated in a 3-specialist sector pilot. The failure of learned weighting does not rule out collaborative architectures in domains with demonstrably complementary data distributions.
