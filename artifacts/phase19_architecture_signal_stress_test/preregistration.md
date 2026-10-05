# Phase 19 Preregistration: Architecture Signal Stress Test

## Scientific Question

**Is the H=4 cohort-stacking benefit a model-agnostic portfolio-construction effect that persists across signal quality, including weak or null ranking signals?**

This is an architecture stress test, not a model development exercise. No new LoRA models are trained. The portfolio machinery is held strictly fixed to evaluate performance exclusively across ranking regimes.

---

## 1. Frozen Signal Regimes

1. **Clean Deterministic LoRA**:
   - Preserves exact Phase 18 clean predictions (`clean_predictions_seed_{42,123,777}.csv`).
   - Signal score: $\hat{\mu} = \text{center} - p_{\text{clean}}$.

2. **Ticker-Risk Baseline**:
   - Training-only historical downside frequency per ticker with 20 pseudo-event shrinkage toward training mean (`center`).
   - Signal score: $\hat{\mu} = \text{center} - \text{risk}_{\text{ticker}}$.

3. **Constant / Equal Ranking**:
   - `constant_ranking`: Constant zero score ($\hat{\mu} = 0.0$) passed into the Markowitz optimizer.
   - `equal_weight`: Exact uniform allocation within cohort ($w_i = 1/m$).

4. **Random Ranking Null**:
   - Cross-sectional permutation of predictions within each weekly cohort across 50 predeclared seeds:
     `seeds = [190001, ..., 190050]`.
   - Applied across each of the 3 base model prediction profiles (total 150 evaluations per horizon).
   - Preserves empirical probability amplitude and weekly cohort structure while destroying cross-sectional rank information.

5. **Synthetic Weak-Signal Ladder (Diagnostic Simulation)**:
   - Controlled diagnostic score family:
     $$\text{score}(\alpha) = \alpha \cdot z_{\text{oracle}} + \sqrt{1 - \alpha^2} \cdot z_{\text{noise}}$$
   - $\alpha \in \{0.00, 0.05, 0.10, 0.20, 0.40\}$.
   - 20 independent noise seeds per $\alpha$: `seeds = [195001, ..., 195020]`.
   - $z_{\text{oracle}}$ is the standardized reconstructed cumulative 4-week return.
   - Strictly a diagnostic simulation to map architecture behavior across signal strength; never an empirical strategy claim.

---

## 2. Frozen Hypotheses

- **H1 (Mechanical Invariance)**: $H=4$ reduces traded turnover by at least 50% relative to $H=1$ across all empirical and null ranking regimes.
- **H2 (Diversification Invariance)**: $H=4$ reduces annualized volatility by at least 30% relative to $H=1$ across all empirical and null ranking regimes.
- **H3 (Utility Dependence)**: The $H=4$ utility delta $\Delta U$ shrinks toward zero as ranking signal quality falls toward zero, even while mechanical turnover and volatility gains remain stable.
- **H4 (No Alpha Interpretation Under Null Rankings)**: If $H=4$ improves Sharpe or utility under random or constant ranking, all gains are classified as mechanical portfolio diversification and cost-amortization effects, not predictive alpha.

---

## 3. Frozen Outcome Classification

- **Outcome A (Architecture-General)**: Turnover reduction $\ge 50\%$ and volatility reduction $\ge 30\%$ across all empirical, constant, and random ranking regimes, with $\Delta U \ge 0$.
- **Outcome B (Signal-Dependent Architecture)**: Mechanical turnover and volatility gains hold, but net economic utility $\Delta U < 0$ when signal quality is weak or null.
- **Outcome C (Phase 18-Specific)**: Mechanical turnover or volatility reductions fail to achieve thresholds (<50% turnover or <30% volatility reduction).
- **Outcome D (Mixed / Inconclusive)**: Results yield inconsistent criteria across seeds or accounting closure fails.

---

## 4. Accounting Closure & Statistical Invariants

- Exact identity checked per seed and regime:
  $$\Delta U = \Delta \bar{r}_{\text{gross}} + \Delta \text{CostSaved} + \Delta U_{\text{var}}$$
  Tolerance: $|\text{error}| < 10^{-12}$.
- Paired moving block bootstrap (1000 draws, 13-week blocks, seed 190042) preserving shared market history across seeds.
- Within-entry-week permutation null (199 draws, seed 190123).
