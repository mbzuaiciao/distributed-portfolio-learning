# Study Limitations & Boundary Conditions

This document outlines the explicit boundary conditions and limitations of the findings presented in the consolidated paper.

---

## 1. Synthetic Simulation Assumptions (Phase 20)

1. **Linear Gaussian Decay**:
   - The data-generating process assumes geometric decay of the latent signal ($s_{t+k} = \rho^k s_t$). In real financial markets, signal decay can be non-exponential, regime-dependent, or abrupt (e.g., following earnings revisions or macro surprises).
2. **Stationary Volatility & Uncorrelated Residuals**:
   - The simulation assumes homoskedastic asset return shocks ($\sigma_{\epsilon} = 0.05$) without volatility clustering (GARCH effects), fat tails, or dynamic cross-asset correlation spikes during market drawdowns.
3. **Proportional Linear Transaction Costs**:
   - Trading costs are modeled as linear frictions ($c \cdot \text{TO}$). In institutional asset management, large orders incur non-linear square-root market impact costs and execution slippage, which would further penalize rapid single-period turnover.
4. **Static Holding Horizons**:
   - The framework evaluates constant holding horizons $H \in \{1, 2, 4, 8, 13\}$. In practice, adaptive strategies that dynamically expand horizons during low-signal or high-volatility regimes could provide further economic efficiencies.
5. **Finite-Memory Truncation**:
   - For high-persistence signals ($\rho = 0.97$), truncating latent history at $K = 52$ weeks captures $\approx 80\%$ of asymptotic signal weight. While sufficient for ranking assets, infinite-memory AR(1) dynamics would show slightly extended half-lives.

---

## 2. Empirical Scope & Revalidation Boundaries (Phases 18 & 19)

1. **Retrospective Holdout Only**:
   - All empirical analyses are evaluated on retrospective historical data terminating on May 15, 2025. As established in the Phase 10 provenance audit, no independent post-2025 prospective market data was available.
2. **Specific Event Universe**:
   - Empirical findings are situated within a quarterly corporate earnings announcement universe of U.S. equities. The extreme turnover (~72x/year traded) was partly driven by the disjoint weekly cross-section of reporting companies. While representative of event-driven factor strategies, dense factor universes (e.g., S&P 500 momentum) exhibit higher baseline holding overlap.
3. **Absence of Live Execution**:
   - Reconstructed backtests incorporate historical pricing, execution delays, and proportional fee assumptions, but do not reflect live order routing, short-selling borrow availability, or borrow fees.

---

## 3. Negative & Falsified Results Scope (Phases 11–17)

1. **Language Model Representation**:
   - Bayesian LoRA and deterministic LoRA were evaluated on minimal metadata inputs (company, ticker, date) rather than full earnings call transcripts or dense fundamental filings.
2. **Task Scope**:
   - Evaluated on binary downside classification ($20\%$ tail-risk threshold). The failure of Bayesian LoRA to beat a constant base-rate null does not imply that language models cannot predict continuous returns or alternative targets.
3. **Collaborative LoRA**:
   - Evaluated in a 3-specialist sector pilot. The failure of learned weighting does not rule out collaborative architectures in domains with demonstrably complementary data distributions.
