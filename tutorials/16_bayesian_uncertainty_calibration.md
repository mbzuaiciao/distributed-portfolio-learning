# Tutorial 16 — Informational Value of Bayesian-LoRA Epistemic Uncertainty (Hypothesis H1)

In [Tutorial 15](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/tutorials/15_synthetic_uncertainty_audit.md), we proved in a controlled synthetic environment that *if* epistemic predictive uncertainty is well-calibrated with forecast error, it adds statistically significant economic value.

However, in financial machine learning, predictive uncertainty estimates produced by deep neural networks are often uncalibrated or merely reflect aleatoric market noise.

Before using Bayesian-LoRA uncertainty in portfolio allocation, we must rigorously test **Hypothesis H1**:
> **Hypothesis H1**: Higher Bayesian-LoRA epistemic uncertainty predicts larger future forecast error on out-of-sample financial data.

In this tutorial, we test H1 on the untouched 2024–2025 holdout dataset across multiple independent training seeds.

---

## 1. Pedagogical Progression

$$\text{Intuition} \longrightarrow \text{Hypothesis} \longrightarrow \text{Mathematical Setup} \longrightarrow \text{Implementation} \longrightarrow \text{Experiment} \longrightarrow \text{Results} \longrightarrow \text{Limitations} \longrightarrow \text{What We Learned}$$

---

## 2. Intuition: Why Measure Epistemic Uncertainty?

When an enterprise LLM or neural network adapts to earnings reports, news, or macroeconomic text using Low-Rank Adaptation (LoRA):
- Standard deterministic LoRA yields a single point forecast $\hat{\mu}_{j, t}$.
- **Bayesian-LoRA** places a variational posterior distribution over the low-rank adapter weights $W = W_0 + B A$, where $A, B \sim q_\theta(W)$.
- By drawing $S$ Monte Carlo posterior samples, the model computes:
  $$\hat{\mu}_{j, t} = \frac{1}{S} \sum_{s=1}^S f(x_{j, t}; W^{(s)})$$
  $$\sigma_{\text{epi}, j, t}^2 = \frac{1}{S - 1} \sum_{s=1}^S \left( f(x_{j, t}; W^{(s)}) - \hat{\mu}_{j, t} \right)^2$$

**The Core Question**: Does this sample variance $\sigma_{\text{epi}, j, t}$ merely reflect random weight fluctuations, or does it genuinely identify episodes where the model is prone to severe forecast errors?

---

## 3. Mathematical Setup & Evaluation Protocol

### 3.1 Forecast Error Definitions
Let $r_{j, t+1}$ be the realized return following the prediction at timestamp $t$. The forecast error is:

$$e_{j, t+1} = r_{j, t+1} - \hat{\mu}_{j, t}$$

We evaluate three error metrics against decision-time epistemic standard deviation $\sigma_{\text{epi}, j, t}$:
1. **Absolute Error**: $|e_{j, t+1}|$
2. **Squared Error**: $e_{j, t+1}^2$
3. **Large-Error Indicator**: $\mathbb{I}\left[ |e_{j, t+1}| \ge q_{0.80}(|e|) \right]$, indicating the worst 20% of forecast errors.

### 3.2 Statistical Diagnostics
1. **Rank Correlation (Spearman $\rho_s$)**: Evaluates monotonic ordering between $\sigma_{\text{epi}}$ and $|e|$, robust to heavy-tailed return outliers.
2. **Linear Correlation (Pearson $r$)**: Evaluates linear covariance.
3. **Bootstrap 95% Confidence Intervals**: Computed using 1,000 empirical resamples with replacement.
4. **Quantile Error Discrepancy**: Divides observations into quartiles based on $\sigma_{\text{epi}}$:
   $$\text{Ratio}_{Q4 / Q1} = \frac{\mathbb{E}[|e| \mid \sigma_{\text{epi}} \in Q_4]}{\mathbb{E}[|e| \mid \sigma_{\text{epi}} \in Q_1]}$$
5. **AUROC & Average Precision**: Area under the ROC curve for discriminating whether an observation falls into the top-20% largest error bucket.

---

## 4. Implementation

The evaluation pipeline is implemented in [`experiments/run_bayesian_uncertainty_calibration.py`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/experiments/run_bayesian_uncertainty_calibration.py):

```python
from hypergraph_portfolio.bayesian_lora_adapter import BayesianDatasetAdapter

# 1. Load validated records
records = BayesianDatasetAdapter.load_from_csv(
    predictions_path="data/bayesian_lora/holdout/beta_0p003_seed_42_mc32_predictions.csv",
    reconstructed_meta_path="data/bayesian_lora/reconstructed_returns.csv",
)

# 2. Extract signals and realizations
sigma = np.array([r.epistemic_std for r in records])
realized = np.array([r.realized_return for r in records])
forecast = np.array([r.mu_hat for r in records])
abs_err = np.abs(realized - forecast)

# 3. Compute Spearman rank correlation with bootstrap CI
rho, p_value = stats.spearmanr(sigma, abs_err)
```

---

## 5. Empirical Results (Holdout 2024–2025)

The script evaluates the frozen Bayesian-LoRA specification across three independent training seeds on 1,728 untouched holdout observations:

```bash
.venv/bin/python experiments/run_bayesian_uncertainty_calibration.py
```

### Table 1: Informational Value Summary (1,728 Holdout Events)

| Seed | Spearman $\rho_s$ (95% CI) | Pearson $r$ (95% CI) | AUROC (Top 20% Errors) | Q1 Mean \|e\| | Q4 Mean \|e\| | Top/Bottom Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | **$+0.5961$** $[0.563, 0.628]$ | **$+0.6111$** $[0.583, 0.641]$ | **$0.8395$** | $12.47\%$ | $27.52\%$ | **$2.208\times$** |
| **Seed 123** | **$+0.5768$** $[0.547, 0.607]$ | **$+0.5680$** $[0.541, 0.598]$ | **$0.8169$** | $13.16\%$ | $27.32\%$ | **$2.076\times$** |
| **Seed 777** | **$+0.4777$** $[0.441, 0.517]$ | **$+0.5331$** $[0.496, 0.570]$ | **$0.7983$** | $9.92\%$ | $20.63\%$ | **$2.081\times$** |
| **Mean** | **$+0.5502 \pm 0.0519$** | **$+0.5707 \pm 0.0319$** | **$0.8182$** | — | — | **$2.122\times$** |

*All $p$-values $< 10^{-100}$.*

---

## 6. Visual Diagnostics

Diagnostic artifacts are saved in [`artifacts/bayesian_uncertainty_calibration/`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/artifacts/bayesian_uncertainty_calibration/):

### 6.1 Error by Uncertainty Quartile
`error_by_uncertainty_quantile.png` shows monotonic progression:
- **Q1 (Lowest Uncertainty)**: Mean error is $12.47\%$.
- **Q2**: Mean error rises to $15.82\%$.
- **Q3**: Mean error rises to $20.14\%$.
- **Q4 (Highest Uncertainty)**: Mean error surges to $27.52\%$.
Assets in the highest epistemic uncertainty quartile suffer **more than double ($2.12\times$)** the forecast error of assets in the lowest quartile.

### 6.2 Scatter & Binned Mean Trend
`error_vs_uncertainty_scatter.png` confirms that the relationship is not driven by a handful of outliers: binned mean error increases steadily across the full range of epistemic standard deviations.

### 6.3 Large Error Detection (AUROC = 0.839)
`roc_curve_large_errors.png` demonstrates that Bayesian-LoRA epistemic uncertainty provides strong discriminatory power (AUROC $= 0.8395$, Average Precision $= 0.536$ vs $0.200$ base rate) for flagging outsized forecast failures.

### 6.4 Temporal Stability (2024 vs 2025)
- **2024 Slice ($N = 1,184$)**: Spearman $\rho_s = +0.6155$.
- **2025 Slice ($N = 544$)**: Spearman $\rho_s = +0.5484$.
The predictive relationship persists across both years without degradation.

---

## 7. Limitations & Honest Scientific Assessment

1. **Correlation is Not Alpha**: A strong correlation between uncertainty and error does not automatically guarantee profitable portfolio performance. If an asset has a huge expected positive return, shrinking its weight could forfeit excess return.
2. **Event-Driven Structure**: The dataset comprises corporate earnings episodes. In continuous market regimes without discrete catalysts, signal-to-noise ratios are typically lower.
3. **Downside Sensitivity**: Much of the uncertainty spike occurs around high-volatility events, meaning epistemic uncertainty partially correlates with event-induced jump risk.

---

## 8. What We Learned

- **Hypothesis H1 is verified**: Bayesian-LoRA epistemic uncertainty contains statistically significant, decision-relevant information about future forecast error ($\rho_s = +0.550$, AUROC $= 0.818$).
- Next, in [Tutorial 17](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/tutorials/17_bayesian_decision_value.md), we test **Hypothesis H2**: Does this informational advantage translate into superior portfolio decisions when holding point forecasts strictly identical?
