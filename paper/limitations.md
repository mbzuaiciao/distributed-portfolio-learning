# Limitations & Boundary Conditions

This document explicitly delineates the methodological, empirical, and architectural boundary conditions of the study. Acknowledging these limitations is essential to prevent over-generalization and preserve the scientific integrity of the paper.

---

## 1. Domain & Task Specificity
- **Single Downside Target**: The empirical investigation focuses on binary earnings-announcement downside prediction ($y_i = \mathbb{I}(\text{move\_1d}_i \le -4.69\%)$). The findings cannot be assumed to apply automatically to continuous return forecasting, multi-horizon volatility prediction, or broader financial NLP tasks (e.g., ESG extraction, credit rating downgrades, or filing sentiment analysis).
- **Extreme Noise-to-Signal Ratio**: Post-earnings price reactions are notoriously noisy and driven by complex macroeconomic factors, order flow, whisper numbers, and management guidance that are not captured in quarterly announcement dates or company names. The low out-of-sample discriminative ability ($\text{AUROC} \approx 0.51\text{--}0.53$) reflects the inherent difficulty of the market prediction task.

---

## 2. Text Representation Simplicity
- **Metadata-Only Clean Text**: To maintain strict experimental control and eliminate any possibility of post-event information leakage, the clean text representation was restricted to structured announcement metadata:
  $$\text{clean\_text} = \text{company} + \text{" ("} + \text{ticker} + \text{") quarterly earnings announcement on "} + \text{date} + \text{"."}$$
  While this strictly eliminated outcome leakage, it provided minimal linguistic signal. The findings do not imply that full earnings transcripts, 10-K filings, or news feeds lack predictive content. Rather, they show that on metadata-derived inputs, fine-tuned transformer adapters do not extract meaningful directionality beyond the unconditional empirical base rate.

---

## 3. Base-Rate Competitiveness
- **Class Imbalance & Shrinkage**: The empirical downside threshold isolates the lowest 20th percentile of moves ($p \approx 0.20$). In this regime, an unconditional constant predictor ($p_i \equiv 0.20$) achieves an out-of-sample Brier score of $0.16000$ and zero calibration error. Beating this simple null hypothesis requires substantial genuine predictive signal, which neither deterministic nor Bayesian LoRA models could consistently muster.

---

## 4. Architectural & Optimization Scope
- **Single Pre-trained Backbone**: All experiments utilized FinBERT (`ProsusAI/finbert`), a BERT-base architecture fine-tuned on financial sentiment. Other pre-trained architectures (e.g., RoBERTa, DeBERTa, or modern decoder-only LLMs) may exhibit different adaptation dynamics.
- **LoRA Parameterization**: Adaptation was restricted to rank-16 low-rank adapters on self-attention `query` and `value` projections. Full model fine-tuning or alternative adapter topologies (e.g., parallel adapters, IA3) were not evaluated.
- **Bayesian Variational Recipe**: The Bayesian-LoRA implementation used a mean-field Gaussian variational posterior with softplus parameterization and a fixed prior standard deviation ($\sigma_0 = 1.0$) and KL penalty weight ($\beta = 0.003$). While these hyperparameters followed the established literature and were evaluated on pre-2022 validation data, the Bayesian hyperparameter space was not exhaustively re-tuned after leakage elimination to avoid post-hoc data snooping.

---

## 5. Sample & Market Regime Breadth
- **Temporal Regime Window**: Out-of-sample evaluation was conducted across three historical regimes totaling 3.5 years (2022, 2023, 2024–May 2025; $N=4,100$ unique earnings events). While these regimes span sharply contrasting market conditions (a severe bear market/tightening cycle in 2022, a cyclical recovery in 2023, and a tech-concentrated bull market in 2024–2025), a 3.5-year window remains a limited sample in empirical finance.
- **Shared Economic Events Across Seeds**: Because all three random seeds (42, 123, 777) evaluate the identical historical timeline of earnings announcements, their prediction errors are subject to common macroeconomic shocks. While our analysis accounts for this using calendar-week clustered block bootstrapping, the effective number of independent macroeconomic market regimes remains small.

---

## 6. Portfolio Simulation Assumptions
- **Backbone Assumptions**: Downstream portfolio evaluations relied on simplified unadjusted Markowitz optimization ($M_0$) with equal cohort capital, 4-week holding periods, fixed risk aversion ($\gamma=1.0$), and a stylized $10\text{ bps}$ turnover cost.
- **Sensitivity to Implementation Choices**: Portfolio returns in academic backtests are sensitive to execution timing, slippage models, liquidity filtering, and rebalancing conventions. Portfolio findings should be viewed as illustrative sanity checks rather than production-grade trading strategies.
