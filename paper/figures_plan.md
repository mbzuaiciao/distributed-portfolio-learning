# Figures Plan & Visual Specifications

This document outlines the minimal, publication-grade figure set for the paper. Each figure is designed to convey a distinct methodological or empirical finding without redundant visual clutter.

---

## Figure 1: Research & Audit Trajectory Timeline

- **Type**: Conceptual Flow / Methodology Diagram (Mermaid / TikZ / Matplotlib block diagram).
- **Caption**: *"Figure 1: Chronology of the research investigation. Initial experiments showed large apparent Bayesian uncertainty and reliability advantages (Phases 11–15). Systematic forensic auditing (Phases 16.1) revealed target outcome leakage in historical training text and a mechanical artifact in pooled reliability correlation. Clean retraining from scratch under preregistered protocols (Phase 17) demonstrated practical equivalence between Bayesian and deterministic LoRA, redundancy of test-time Monte Carlo sampling, and the absence of robust portfolio improvements."*
- **Visual Structure**:
  ```
  [1. Initial Empirical Promise]
      │  • Apparent high validation AUROC (>0.999)
      │  • Apparent entropy-error correlation (ρ ≈ 0.60)
      │  • Promising selective risk abstention
      ▼
  [2. Mechanism Probing & Portfolio Evaluation]
      │  • Portfolio risk inflation (M16) volatile across regimes
      │  • Diagnostic tests show inference redundancy (Phase 15.1)
      ▼
  [3. Forensic Audit Discoveries]
      │  • Discovery of target leakage: ". 1-day move was " + move_1d
      │  • Mathematical proof of mechanical correlation under class imbalance
      │  • Constant base-rate null beats leaked models (Brier = 0.1600 vs 0.1924)
      ▼
  [4. Clean Retraining & Preregistered Protocol]
      │  • Canonical clean text pipeline (zero outcome tokens)
      │  • Pre-committed equivalence margins & universe hashes
      │  • Matched training across seeds 42, 123, 777
      ▼
  [5. Final Clean Scientific Findings]
      │  • Bayesian posterior mean and deterministic LoRA practically equivalent
      │  • MC32 sampling 100% computationally redundant (30x latency penalty)
      │  • No robust downstream portfolio benefit established
  ```

---

## Figure 2: Out-of-Sample Discrimination & Proper Scoring Performance

- **Type**: Multi-panel Bar / Interval Plot with 95% Clustered Bootstrap Confidence Intervals.
- **Data Source**: `artifacts/phase17_clean_bayesian/forecast_metrics.csv` & `bootstrap_intervals.csv`.
- **Panels**:
  - **Panel A: Discriminative Ability (AUROC)**:
    - Base-Rate Null ($0.5000$)
    - Clean Deterministic LoRA ($0.5131$, $95\%\text{ CI: } [0.503, 0.523]$)
    - Clean Bayesian Posterior Mean ($0.5253$, $95\%\text{ CI: } [0.516, 0.535]$)
    - Clean Bayesian MC32 ($0.5250$, $95\%\text{ CI: } [0.516, 0.535]$)
  - **Panel B: Proper Scoring (Brier Score)**:
    - Clean Deterministic LoRA ($0.16079$)
    - Base-Rate Null ($0.16000$)
    - Clean Bayesian Posterior Mean ($0.15987$)
    - Clean Bayesian MC32 ($0.15987$)
  - **Panel C: Expected Calibration Error (ECE)**:
    - Base-Rate Null ($0.00000$)
    - Clean Bayesian Posterior Mean ($0.00686$)
    - Clean Bayesian MC32 ($0.00695$)
    - Clean Deterministic LoRA ($0.01785$)
- **Caption**: *"Figure 2: Clean out-of-sample forecast performance across models ($N=12,300$, pooled 2022–2025). Shaded error bars indicate 95% calendar-week clustered bootstrap confidence intervals. Both architectures achieve AUROC barely above random chance. In Brier score, all models cluster around the trivial constant base-rate predictor ($p \equiv 0.20$, dashed line), with Bayesian posterior mean achieving a negligible $0.00013$ improvement over the null."*

---

## Figure 3: Bayesian Posterior Mean vs. MC32 Prediction Agreement

- **Type**: Scatter / Density Correlation Plot ($N=12,300$).
- **Data Source**: `artifacts/phase17_clean_bayesian/clean_oos_predictions.csv`.
- **Axes**:
  - X-axis: $p_{\text{bayes\_mean}}$ (single forward pass using posterior mean weights $\mu$).
  - Y-axis: $p_{\text{bayes\_mc32}}$ (posterior predictive mean across 32 stochastic draws).
- **Key Visual Elements**:
  - 45-degree identity line ($y = x$).
  - Inset histogram of pairwise absolute differences: $|p_{\text{bayes\_mc32}} - p_{\text{bayes\_mean}}|$, showing that $99.9\%$ of points differ by less than $0.001$.
  - Inset box highlighting compute comparison ($7.89\text{ ms}$ vs. $241.02\text{ ms}$).
- **Caption**: *"Figure 3: Empirical equivalence of Monte Carlo weight sampling and single-pass posterior-mean inference ($N=12,300$). Predicted probabilities from 32 stochastic Monte Carlo draws are plotted against a single forward pass evaluated at the variational posterior mean $\mu$. The Pearson correlation is $r > 0.9999$, with maximum absolute difference $< 0.002$. Running 32 stochastic passes multiplies latency by $30.6\times$ without altering point predictions."*

---

## Figure 4: Regime-Level Stability & Seed Variance

- **Type**: Grouped Bar Chart by Historical Regime (2022, 2023, 2024–2025) with Seed Dots.
- **Data Source**: `artifacts/phase17_clean_bayesian/regime_metrics.csv` & `seed_metrics.csv`.
- **Panels**:
  - **Panel A: AUROC Across Regimes**:
    - 2022: Det $0.509$ vs. Bayes $0.515$
    - 2023: Det $0.513$ vs. Bayes $0.517$
    - 2024–2025: Det $0.517$ vs. Bayes $0.538$
  - **Panel B: Brier Score Across Regimes**:
    - 2022: Det $0.1577$ vs. Bayes $0.1569$ vs. Null $0.1569$
    - 2023: Det $0.1565$ vs. Bayes $0.1557$ vs. Null $0.1558$
    - 2024–2025: Det $0.1659$ vs. Bayes $0.1648$ vs. Null $0.1650$
- **Caption**: *"Figure 4: Forecast accuracy across three clean post-training historical regimes. Across all market environments—including the 2022 inflation shock, the 2023 recovery, and 2024–2025 tech rally—model Brier scores remain tied with the constant base-rate null, and AUROC differences remain within a narrow band of near-chance discriminative power."*
