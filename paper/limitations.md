# Limitations & Boundary Conditions

This document explicitly delineates the methodological, empirical, and architectural boundary conditions of the study. Acknowledging these limitations is essential to prevent over-generalization and ensure the conclusions remain scientifically defensible.

---

## 1. Primary Limitation: Minimal Metadata-Based Clean Input Representation
The central limitation of the clean experiment is that model inputs were restricted to structured corporate metadata rather than substantive textual disclosures:
$$\text{clean\_text} = \text{company} + \text{" ("} + \text{ticker} + \text{") quarterly earnings announcement on "} + \text{date} + \text{"."}$$
This minimal representation was deliberately adopted to provide an unassailable baseline that guaranteed zero possibility of post-event target outcome leakage. However, it contains minimal linguistic or economic narrative. 

**Scope Boundary**: Our findings demonstrate that on minimal announcement metadata, fine-tuned transformer adapters extract almost zero directional signal beyond the unconditional empirical base rate. Our study does **not** evaluate whether full earnings call transcripts, 10-K/10-Q filings, or real-time news feeds contain predictive signal or whether Bayesian LoRA would perform differently on rich semantic text.

---

## 2. Weak Overall Discriminative Performance & Noisy Market Labels
- **Low Signal-to-Noise Ratio**: Across all clean out-of-sample evaluations, both deterministic and Bayesian models achieved discriminative performance barely exceeding random chance ($\text{AUROC} \approx 0.513\text{--}0.525$). 
- **Noisy Market-Outcome Targets**: 1-day post-earnings returns are driven by complex market microstructure, pre-existing whisper numbers, macroeconomic news, and liquidity flows that cannot be captured by corporate identity and announcement dates alone.
- **Competitiveness of the Base-Rate Null**: Because labels are imbalanced ($80\%$ non-downside, $20\%$ downside), a zero-parameter constant predictor ($p \equiv 0.20$) achieves an out-of-sample Brier score of $0.16000$ and zero calibration error. Fine-tuned models output probabilities that cluster tightly around this base rate ($\bar{p} \approx 0.187\text{--}0.193$, $\sigma_p \approx 0.02\text{--}0.03$).

---

## 3. Scope of Downstream Portfolio Conclusions
- **Secondary Evidence Only**: Portfolio simulations must be interpreted strictly as secondary evidence. Because the underlying clean point forecasts possess minimal discriminative edge, portfolio optimizations are operating on extremely weak signals.
- **Stylized Assumptions**: The portfolio backtest employed stylized assumptions: 4-week holding periods, equal cohort weighting, fixed risk aversion ($\gamma=1.0$), and a simplified $10\text{ bps}$ turnover cost. The findings show that uncertainty mappings cannot rescue a weak point predictor in this specific setup, but they do not prove that uncertainty adjustments are ineffective in systems with strong underlying alpha.

---

## 4. Architectural & Optimization Scope
- **Single Pre-trained Backbone**: All experiments utilized FinBERT (`ProsusAI/finbert`), a BERT-base architecture. Larger generative decoder-only architectures (e.g., LLaMA, Mistral) or alternative encoder models (e.g., RoBERTa, DeBERTa) were not evaluated.
- **LoRA Parameterization**: Adaptation was restricted to rank-16 low-rank adapters on self-attention `query` and `value` projections ($591,362$ trainable parameters). Alternative parameter-efficient tuning methods (e.g., prefix tuning, prompt tuning, IA3) or full fine-tuning were not tested.
- **Fixed Bayesian Hyperparameters**: To avoid post-hoc data snooping, Bayesian hyperparameters ($\sigma_0 = 1.0, \beta = 0.003, \sigma_{\text{init}} = 0.01$) were set based on pre-2022 validation data and standard literature conventions, rather than exhaustively re-tuned after leakage elimination. It remains possible that alternative prior specifications or KL schedules could alter optimization dynamics.

---

## 5. Market Regime Breadth & Shared Economic Shocks
- **Temporal Breadth**: Out-of-sample evaluation was restricted to three historical regimes spanning 3.5 years (2022 through May 2025; $N_{\text{unique}}=4,100$ events). While these regimes include contrasting market environments (the 2022 inflationary bear market, the 2023 recovery, and the 2024–2025 bull market), 3.5 years represents a finite window in financial history.
- **Shared Economic Realizations Across Seeds**: All random seeds (42, 123, 777) evaluate the identical historical timeline of corporate announcements. While our statistical tests explicitly account for weekly cross-sectional co-occurrences via calendar-week clustered bootstrapping, the effective number of independent macroeconomic market regimes is limited.
