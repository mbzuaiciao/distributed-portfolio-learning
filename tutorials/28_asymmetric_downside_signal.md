# Tutorial 28: Asymmetric Downside Signal vs Epistemic Dispersion

Across Phases 3 through 5, a recurring question has been whether neural network epistemic uncertainty failed in cross-sectional asset selection because of **target misalignment**:
> *Does a Bayesian neural network trained on symmetric regression (predicting returns under squared error) produce symmetric dispersion $\sigma_{\text{epi}}$ that fails to distinguish upside surprises from catastrophic drops?*
> *Would an explicitly directional downside target (predicting $P(r_{t+1} < 0)$ or earnings crash classification) materially outperform epistemic dispersion for risk forecasting?*

In Phase 6 (Secondary Track), we conducted a controlled empirical comparison on the identical 2024–2025 frozen holdout (1,728 event predictions).

---

## 1. Comparing the Competing Predictors

We benchmark three predictive signals:

1. **Symmetric Epistemic Dispersion ($\sigma_{\text{epi}}$)**:
   Latent parameter variance derived from the Bayesian-LoRA Monte Carlo ensemble. Large $\sigma_{\text{epi}}$ indicates high disagreement among sampled models.
2. **Directional Downside Target ($p_{\text{down}}$)**:
   The model's posterior predictive probability of an earnings drop ($P(\text{label}=1)$). High $p_{\text{down}}$ explicitly indicates an expected negative return shock.
3. **Historical Volatility ($\hat{\sigma}_{\text{hist}}$)**:
   Rolling 12-week sample standard deviation of market returns.

---

## 2. Empirical Holdout Performance (3 Seeds Comparison)

We evaluate discriminative power across two objective downside definitions:
- **Task A (Earnings Downside Event, $\text{label} = 1$)**
- **Task B (Next-Day Negative Realized Return, $r_{1d} < 0$)**

| Model Seed | Predictor | Task A: AUROC | Task A: Avg Prec (AP) | Task B: AUROC | Brier Score | Recall @ Top 20% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | **Directional $p_{\text{down}}$** | $\mathbf{0.5982}$ | $\mathbf{0.3728}$ | $\mathbf{0.5655}$ | $\mathbf{0.1939}$ | $27.31\%$ |
| | Symmetric $\sigma_{\text{epi}}$ | $0.5945$ | $0.3458$ | $0.5664$ | $0.3120$ | $\mathbf{29.07\%}$ |
| **Seed 123**| **Directional $p_{\text{down}}$** | $\mathbf{0.5972}$ | $\mathbf{0.3756}$ | $\mathbf{0.5691}$ | $\mathbf{0.1926}$ | $\mathbf{27.75\%}$ |
| | Symmetric $\sigma_{\text{epi}}$ | $0.5865$ | $0.3188$ | $0.5613$ | $0.2985$ | $27.09\%$ |
| **Seed 777**| **Directional $p_{\text{down}}$** | $\mathbf{0.5954}$ | $\mathbf{0.3733}$ | $0.5543$ | $\mathbf{0.2038}$ | $\mathbf{27.97\%}$ |
| | Symmetric $\sigma_{\text{epi}}$ | $0.5931$ | $0.3643$ | $\mathbf{0.5559}$ | $0.3087$ | $27.31\%$ |

---

## 3. Scientific Findings & Interpretation

1. **Modest Edge for Directional Probability**:
   Directional $p_{\text{down}}$ achieves slightly higher Average Precision ($0.374$ vs $0.343$) and lower Brier score ($0.197$ vs $0.306$) than symmetric $\sigma_{\text{epi}}$.
2. **Persistent Ceiling in Downside Discrimination ($\text{AUROC} \approx 0.60$)**:
   Crucially, **neither signal achieves high discriminative power for market drops**. Both $p_{\text{down}}$ and $\sigma_{\text{epi}}$ cluster tightly around $\text{AUROC} \approx 0.59 - 0.60$ for earnings drops and $\text{AUROC} \approx 0.56$ for negative return days.
3. **Contrast with Absolute Forecast Error ($\text{AUROC} \approx 0.82$)**:
   Epistemic dispersion $\sigma_{\text{epi}}$ achieves an astonishing $\text{AUROC} = 0.818$ and $\rho_s = +0.55$ when predicting *how wrong the model's forecast will be*. In contrast, predicting whether an asset will go down tomorrow remains dominated by low signal-to-noise ratios in financial markets.
4. **Conclusion on Secondary Track (Outcome F / Weak Target Misalignment)**:
   Switching from symmetric dispersion to an explicitly directional downside probability provides a small calibration improvement, but does not fundamentally unlock directional predictability on this dataset. The high efficacy of **relative epistemic timing ($M_9$)** in Tutorial 27 succeeds because it times *forecast failure regimes* (model fragility) rather than attempting noisy directional market-drop prediction.
