# When Uncertainty Looks Useful: Target Leakage, Reliability Metrics, and Bayesian LoRA in Financial Event Risk

**Working Paper**  
**September 2026**

---

## Abstract

Parameter-efficient fine-tuning via Low-Rank Adaptation (LoRA) is widely explored for financial risk modeling, with Bayesian variational extensions proposed to quantify epistemic uncertainty for portfolio construction. In initial experiments predicting earnings-announcement downside risk, Bayesian LoRA appeared to achieve near-perfect validation accuracy and strong error-tracking correlations ($\rho \approx 0.60$ between predictive entropy and absolute error). However, a forensic audit revealed that these apparent advantages were driven by experimental vulnerabilities: inadvertent target outcome leakage in historical training text (`move_1d` tokens concatenated into inputs) and a mechanical artifact where class imbalance ($80\%$ majority class) forces within-class entropy-error correlations to $\pm 1.0000$ under probability shrinkage. Retraining both regularized deterministic LoRA and Bayesian LoRA on strictly clean, metadata-based inputs ($N_{\text{train}}=11,985$, $N_{\text{val}}=2,116$, $N_{\text{test}}=12,300$ across 2022–2025) yielded modest discriminative improvement for Bayesian LoRA ($\text{AUROC} = 0.5253$ vs. $0.5131$), but proper scoring performance remained practically equivalent ($\Delta \text{Brier} = -0.00092$, $95\%\text{ CI: } [-0.00115, -0.00070]$). Crucially, a constant base-rate predictor ($p \equiv 0.20$) was highly competitive ($\text{Brier} = 0.16000$), matching Bayesian LoRA ($0.15987$) and exceeding deterministic LoRA ($0.16079$). Furthermore, 32-draw Monte Carlo sampling was practically equivalent to single-pass posterior-mean inference under preregistered margins ($\Delta \text{Brier} = +0.00001, \Delta \text{AUROC} = -0.00028$) while incurring $\sim 30\times$ higher latency, and uncertainty-aware portfolio adjustments established no robust clean economic advantage. These findings highlight the critical importance of baseline discipline, class-conditional metric auditing, and leakage prevention when evaluating uncertainty in financial event forecasting.

---

## 1. Introduction

Accurate assessment of forecast uncertainty is widely considered a prerequisite for deploying deep learning systems in high-stakes financial environments. In quantitative portfolio management, point forecasts of asset returns or tail-risk events are notoriously noisy; uncalibrated point predictions can induce extreme leverage, concentrated turnover, and severe drawdown risk when fed directly into mean-variance or risk-parity optimizers. Consequently, the financial machine learning literature has devoted substantial attention to uncertainty quantification, seeking models that can reliably distinguish between aleatoric uncertainty (inherent market randomness) and epistemic uncertainty (model parameter ignorance).

With the ascendancy of pre-trained transformer representations, Parameter-Efficient Fine-Tuning (PEFT) has emerged as the dominant paradigm for financial Natural Language Processing (NLP). Among PEFT methodologies, Low-Rank Adaptation (LoRA; Hu et al., 2021) has achieved preeminence by freezing pre-trained backbone weights and injecting low-rank trainable decomposition matrices into self-attention projections. To endow LoRA with principled uncertainty estimates, variational Bayesian extensions have been introduced. By placing variational Gaussian distributions over the low-rank adapter weights and optimizing a variational free-energy objective (Bayes by Backprop; Blundell et al., 2015), Bayesian LoRA aims to capture parameter uncertainty efficiently while updating less than one percent of the base model's parameters.

In this paper, we document an empirical investigation into whether Bayesian LoRA delivers genuine, actionable value for financial event-risk forecasting and quantitative portfolio construction. Under our earnings-event prediction setup, dataset, label construction, FinBERT/LoRA architecture, and minimal clean input representation, the research arc began with apparent empirical promise: in preliminary experiments predicting corporate earnings downside shocks, Bayesian LoRA appeared to achieve high predictive accuracy, strong rank correlations between predictive entropy and realized prediction errors, and promising selective-risk retention curves. These encouraging signs motivated an extensive effort to translate Bayesian uncertainty into downstream portfolio enhancements, including dynamic cash-exposure timing, signal shrinkage, selective asset abstention, and uncertainty-aware risk inflation.

However, as our investigation proceeded to rigorous out-of-sample stress testing across contrasting market regimes (including the 2022 inflationary bear market, the 2023 cyclical rebound, and the 2024–2025 continuation), the apparent advantages of Bayesian uncertainty exhibited severe instability. Downstream portfolio mappings that delivered high Sharpe ratios in one historical regime caused substantial underperformance in adjacent periods. This instability prompted a comprehensive forensic audit of the entire experimental pipeline.

The audit revealed two foundational methodological issues that completely recast the interpretation of earlier results:
1. **Target Outcome Leakage in Training Pipelines**: The text serialization module used during historical training inadvertently concatenated post-event market realizations (`. 1-day move was ` + `move_1d`) directly into the input text string. Because the binary classification target was deterministically defined by thresholding `move_1d`, the model was provided with ground-truth outcome tokens during training and pre-2022 validation, achieving an artificial validation AUROC of $0.9998\text{--}1.0000$. When evaluated on clean text out of sample, the model suffered distribution collapse toward near-chance discrimination ($\text{AUROC} \approx 0.50$).
2. **Mechanical Artifacts in Reliability Metrics**: The prominent evaluation metric used to assert Bayesian error awareness—the Spearman rank correlation between predictive entropy and absolute prediction error ($\rho \approx 0.60$)—was revealed to be largely an artifact of class imbalance under probability shrinkage. In binary classification with an $80\%$ majority class and probabilities concentrated below $0.50$, the rank correlation within the majority class is mechanically forced to $+1.0000$, and within the minority class to $-1.0000$. The apparent positive pooled correlation reflects unconditional probability shrinkage toward zero rather than learned epistemic awareness of error.

To resolve the core scientific question—*When evaluated on matched, clean text without outcome leakage, does Bayesian-LoRA training provide measurable value over regularized deterministic LoRA?*—we executed a strictly controlled, preregistered re-evaluation. We retrained both regularized deterministic LoRA and variational Bayesian LoRA from scratch on strictly clean, metadata-based inputs across a partitioned universe of 18,201 earnings events ($N_{\text{train}} = 11,985$, $N_{\text{val}} = 2,116$), followed by out-of-sample evaluation across 12,300 clean observations (4,100 unique events across 3 seeds) spanning three historical regimes.

Our primary contributions follow a strict methodological hierarchy:
1. **Target-Leakage Forensic Audit**: We document the subtle mechanics of target outcome leakage in financial text serialization and demonstrate why high validation scores can mask test-time collapse.
2. **Reliability-Metric Audit**: We provide a formal derivation and empirical demonstration showing how standard rank-correlation metrics between entropy and error are mechanically distorted by class imbalance and probability shrinkage.
3. **Clean Matched Retraining**: Under preregistered equivalence margins and calendar-week clustered bootstrapping, we establish that Bayesian training yields a modest gain in discriminative ability ($\text{AUROC} = 0.5253$ vs. $0.5131$) but is practically equivalent to regularized deterministic LoRA in proper scoring rules ($\Delta \text{Brier} = -0.00092$), with both models remaining weak predictors on minimal metadata inputs.
4. **Strong Base-Rate Baseline**: We demonstrate that a trivial constant base-rate predictor ($p \equiv 0.20$) achieves an out-of-sample Brier score of $0.16000$ and zero calibration error, outperforming deterministic LoRA ($0.16079$) and matching Bayesian LoRA ($0.15987$).
5. **Posterior-Mean vs. MC32 Equivalence**: We demonstrate that 32-draw Monte Carlo weight sampling was practically equivalent to a single deterministic forward pass evaluating the posterior mean parameters under preregistered margins ($\Delta \text{Brier} = +0.00001, \Delta \text{AUROC} = -0.00028$), while incurring a $30.6\times$ latency penalty in this experiment.

As secondary evidence, downstream portfolio simulations demonstrate that uncertainty-aware asset adjustments do not translate into robust economic improvements out of sample, largely because the underlying clean point-prediction signal is inherently weak.

The remainder of this paper is organized as follows. Section 2 outlines the problem setup and model architectures. Section 3 reviews the initial historical findings and explains why they appeared promising. Section 4 presents the forensic audit findings. Section 5 describes the clean experimental protocol. Section 6 presents the clean out-of-sample forecasting results. Section 7 analyzes the computational cost of Bayesian inference. Section 8 evaluates downstream portfolio implications as secondary evidence. Section 9 discusses broader methodological takeaways, Section 10 outlines study limitations, and Section 11 concludes.

---

## 2. Problem Setup & Methodological Framework

### 2.1 The Downside Earnings Prediction Task
We consider the financial event-study problem of predicting tail-risk price reactions following quarterly corporate earnings announcements. Let $\mathcal{E}_i = (c_i, \tau_i, t_i)$ denote an earnings event for company $c_i$ with ticker symbol $\tau_i$ announced at timestamp $t_i$. Let $r_{i, \Delta t}$ denote the realized 1-day post-earnings cumulative abnormal return:
$$r_{i, \Delta t} = \frac{P_{i, t_i + 1\text{d}} - P_{i, t_i}}{P_{i, t_i}} - r_{m, \Delta t}$$
where $r_{m, \Delta t}$ is the market benchmark return.

To isolate acute downside tail risk, the downstream classification label $y_i \in \{0, 1\}$ is defined by the lower quintile ($20\text{th}$ percentile) of the historical pre-2022 return distribution:
$$y_i = \mathbb{I}(r_{i, \Delta t} \le \gamma_{0.20})$$
where $\gamma_{0.20} = -4.692\%$ is the empirical cutoff. An event where the asset drops by more than $4.692\%$ is labeled a downside event ($y_i = 1$); all other events are labeled non-downside ($y_i = 0$). By construction, the target exhibits severe class imbalance ($80\%$ non-downside, $20\%$ downside).

### 2.2 Base Language Model Architecture
All models employ FinBERT (`ProsusAI/finbert`; Araci, 2019), a 12-layer, 768-hidden-dimension Bidirectional Encoder Representations from Transformers (BERT) model pre-trained on the Financial PhraseBank and corporate filings. The base model parameters $\mathbf{W}_0$ are frozen throughout all experiments, preserving pre-trained linguistic features and preventing catastrophic forgetting.

### 2.3 Deterministic Low-Rank Adaptation (LoRA)
In standard deterministic LoRA (Hu et al., 2021), weight updates to selected linear projections in the self-attention mechanism are parameterized by low-rank factorizations. For a frozen projection $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, the modified forward pass is:
$$\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \Delta \mathbf{W} \mathbf{x} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B} \mathbf{A} \mathbf{x}$$
where $\mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}$ and $\mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}$ are trainable matrices with rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$, and $\alpha$ is a constant scaling hyperparameter. Adapters are injected exclusively into the self-attention `query` and `value` projections across all 12 transformer layers. With $r=16$ and $\alpha=32$, the total number of trainable adapter parameters plus the binary classification head is $591,362$ ($0.54\%$ of the 109 million base parameters). The model is optimized using cross-entropy loss with AdamW and $L_2$ weight decay regularization.

### 2.4 Variational Bayesian LoRA
Variational Bayesian LoRA replaces deterministic adapter weights with probability distributions. Following the mean-field variational inference framework (Blundell et al., 2015), every entry in the low-rank matrices $\mathbf{A}$ and $\mathbf{B}$ is modeled as an independent Gaussian distribution:
$$q_\theta(\mathbf{A}) = \prod_{j=1}^r \prod_{k=1}^{d_{\text{in}}} \mathcal{N}\left(A_{jk}; \, \mu_{A, jk}, \, \sigma_{A, jk}^2\right), \quad q_\theta(\mathbf{B}) = \prod_{j=1}^{d_{\text{out}}} \prod_{k=1}^r \mathcal{N}\left(B_{jk}; \, \mu_{B, jk}, \, \sigma_{B, jk}^2\right)$$
To ensure strictly positive standard deviations during gradient descent, variances are reparameterized using the softplus transformation:
$$\sigma = \text{softplus}(\rho) + \epsilon = \log(1 + e^\rho) + \epsilon$$
The variational parameter set $\theta = \{\mu_A, \rho_A, \mu_B, \rho_B, \mathbf{w}_{\text{clf}}\}$ is trained by minimizing the negative Evidence Lower Bound (ELBO):
$$\mathcal{L}_{\text{ELBO}}(\theta) = -\mathbb{E}_{q_\theta(\mathbf{A}, \mathbf{B})}\left[ \sum_{i=1}^{B} \log p(y_i \mid \mathbf{x}_i, \mathbf{A}, \mathbf{B}) \right] + \frac{\beta}{N_{\text{train}}} D_{\text{KL}}\left( q_\theta(\mathbf{A}, \mathbf{B}) \,\|\, p(\mathbf{A}, \mathbf{B}) \right)$$
where $p(\mathbf{A}, \mathbf{B}) = \mathcal{N}(0, \sigma_0^2 \mathbf{I})$ is an isotropic Gaussian prior with $\sigma_0 = 1.0$, and $\beta$ is a KL weighting factor scaled by the dataset size $N_{\text{train}} = 11,985$.

At inference time, predictions can be evaluated in two configurations:
1. **Monte Carlo Sampling (MC-$M$)**: Drawing $M$ stochastic weight realizations $\mathbf{A}^{(m)}, \mathbf{B}^{(m)} \sim q_\theta$ and computing the posterior predictive mean probability:
   $$\bar{p}_i = \frac{1}{M} \sum_{m=1}^M p(y_i = 1 \mid \mathbf{x}_i, \mathbf{A}^{(m)}, \mathbf{B}^{(m)})$$
   and predictive variance $\sigma_{\text{epi}, i}^2 = \frac{1}{M} \sum_{m=1}^M (p_i^{(m)} - \bar{p}_i)^2$.
2. **Posterior Mean Forward Pass**: Disabling stochastic perturbation and evaluating the network deterministically at the variational mean weights $\mu_A, \mu_B$ in a single forward pass ($M=1$).

---

## 3. Historical Uncertainty Findings & Initial Appearances

To understand how the research trajectory evolved, it is necessary to examine why Bayesian uncertainty initially appeared so promising in early experimental phases (Phases 11–15).

> [!WARNING]
> **Scientific Status Notice**: The empirical results described in this section reflect historical experimental stages that were subsequently discovered to be compromised by target outcome leakage. They are documented here strictly for historical and forensic context and must not be interpreted as valid evidence of model performance.

### 3.1 Apparent Discriminative Strength and Error Correlation
During Phase 11 replication, Bayesian LoRA models trained across three random seeds (42, 123, 777) achieved near-perfect validation accuracy on pre-2022 data, with validation AUROCs ranging between $0.9998$ and $1.0000$. When deployed out of sample, the models appeared to exhibit high calibration and reliability:
- **Proper Scoring Superiority**: Bayesian LoRA appeared to achieve a Brier score of $0.1924$ on post-training holdouts, substantially lower than an initial unregularized deterministic LoRA baseline ($0.2557$).
- **Apparent Error-Tracking Rank Correlation**: Evaluating the Spearman correlation between predictive Shannon entropy:
  $$H(p) = -p \log_2(p) - (1-p) \log_2(1-p)$$
  and the absolute realization error $|p - y|$ yielded a correlation of $\rho = +0.5990$ ($p < 0.0001$). This was initially taken as evidence that the model possessed fine-grained epistemic self-awareness, assigning high entropy specifically when its forecasts were prone to large errors.
- **Selective-Risk Curves**: Abstaining on the $20\%$ of events with highest predictive entropy appeared to monotonically reduce classification error and improve downside precision.

### 3.2 Downstream Portfolio Policy Attempts
Encouraged by these apparent reliability metrics, Phases 13 through 15 explored multiple portfolio mechanisms to translate epistemic uncertainty into economic gains:
1. **Uncertainty-Aware Signal Shrinkage ($M_{14}$)**: Shrinking expected return forecasts toward the universe mean as a linear function of predictive entropy: $\tilde{\mu}_i = (1 - \lambda \bar{H}_i) \hat{\mu}_i$.
2. **Selective Abstention ($M_{15}$)**: Forcing asset weights to zero for the top $30\%$ most uncertain assets prior to Markowitz optimization.
3. **Asset-Level Risk Inflation ($M_{16}$)**: Inflating diagonal elements of the asset covariance matrix proportionally to epistemic variance: $\tilde{\Sigma}_{ii} = \Sigma_{ii} (1 + \lambda_u \sigma_{\text{epi}, i}^2)$.
4. **Hysteresis Portfolio Exposure Timing ($M_{13}$)**: Adjusting total equity market exposure dynamically between $45\%$ and $85\%$ based on trailing aggregate epistemic regime shifts.

### 3.3 The Out-of-Sample Regime Breakdown
When these frozen policies were adjudicated on clean, post-training market regimes (2022, 2023, 2024–2025), a sharp empirical contradiction emerged. In the 2023 cyclical recovery, risk inflation ($M_{16}$) delivered substantial Sharpe ratio improvements ($2.186$ vs. $1.630$ for the unadjusted base portfolio $M_0$). However, when evaluated on the 2022 inflation shock, the exact same frozen mapping caused severe portfolio degradation, lagging the baseline by over $500\text{ basis points}$. In 2024–2025, performance degraded again. No ex-ante rule could reliably predict when the uncertainty adjustment would help or harm.

This regime divergence prompted an unsparing forensic audit of the data generation and evaluation pipelines.

---

## 4. Forensic Audit Findings

The forensic audit (Phases 15.1 and 16.1) systematically inspected every step of data preparation, tokenization, and metric evaluation.

### 4.1 Target Outcome Leakage in Training Pipelines
The primary discovery of the audit was target outcome leakage embedded in the text serialization code. In `experiments/run_phase11_replication.py` (line 132) and inherited by `experiments/run_phase16_deterministic_control.py` (line 427), the input string for each earnings event was constructed as:
```python
pretest["earnings_transcript"] = (
    pretest["company_name"].fillna("")
    + " reported earnings on "
    + pretest["date"].astype(str)
    + ". 1-day move was "
    + pretest["move_1d"].astype(str)
)
```
Because the downstream classification label was defined as $y_i = \mathbb{I}(\text{move\_1d}_i \le -4.692\%)$, concatenating the continuous value of `move_1d` directly into the input text string provided the model with the exact continuous realization of the prediction target.

#### Mechanics of the Failure:
1. **Trivial Optimization**: During fine-tuning, the transformer's attention heads rapidly learned to attend to the numeric string following `"1-day move was"`. The classification head simply learned a thresholding rule on the tokenized digits, driving training cross-entropy loss to near zero and validation AUROC to $0.9998\text{--}1.0000$.
2. **Evaluation Distribution Mismatch**: In contrast to training, clean out-of-sample evaluation pipelines in later phases used clean metadata strings without the leaked return:
   ```python
   f"{company} ({ticker}) quarterly earnings announcement on {date_str}."
   ```
3. **Distribution Collapse**: Lacking the leaked outcome tokens at test time, the model encountered an out-of-distribution input shift. Its discriminative ability collapsed immediately to near random chance ($\text{AUROC} \approx 0.50$).
4. **Probability Shrinkage Toward Base Rate**: Deprived of its primary feature, the model's output logits defaulted to the classification head's bias term, producing probabilities tightly concentrated near the training prevalence ($\bar{p} \approx 0.18$).

### 4.2 Reliability Metric Artifact: Within-Class Decomposition
The second revelation of the audit concerned the reliability metric itself. The apparent rank correlation of $\rho \approx 0.60$ between predictive entropy $H(p)$ and absolute error $|p - y|$ had been accepted as proof that the model's confidence tracked forecast accuracy.

We audited this relationship by disaggregating observations conditional on the true realization $y \in \{0, 1\}$. In binary classification, the absolute prediction error is:
$$e(p, y) = |p - y| = \begin{cases} p & \text{if } y = 0 \\ 1 - p & \text{if } y = 1 \end{cases}$$
Shannon entropy $H(p) = -p \log_2 p - (1-p) \log_2(1-p)$ is strictly concave on $p \in [0, 1]$, with a unique maximum at $p = 0.5$. On the half-interval $p \in [0, 0.5)$, $H(p)$ is a strictly monotonically increasing function of $p$.

Because the model's predicted probabilities suffered shrinkage and were almost universally concentrated below $0.50$ ($p \in [0.10, 0.30]$):
1. **For non-downside events ($y = 0$)**:
   $$e(p, 0) = p \implies \frac{\partial e}{\partial p} > 0, \quad \frac{\partial H}{\partial p} > 0 \quad (\forall p < 0.5)$$
   Because both error $e$ and entropy $H$ are strictly monotonic increasing functions of $p$, their rank order is identical. Thus, the within-class Spearman correlation is mathematically forced to:
   $$\rho(H(p), e(p, 0)) \equiv \mathbf{+1.0000}$$
2. **For downside events ($y = 1$)**:
   $$e(p, 1) = 1 - p \implies \frac{\partial e}{\partial p} < 0, \quad \frac{\partial H}{\partial p} > 0 \quad (\forall p < 0.5)$$
   Here, error is strictly decreasing in $p$ while entropy is strictly increasing. Thus, their rank order is exactly inverted, forcing the within-class Spearman correlation to:
   $$\rho(H(p), e(p, 1)) \equiv \mathbf{-1.0000}$$

#### Empirical Verification:
When evaluated on the clean out-of-sample holdout ($N=12,300$), this exact algebraic relationship was confirmed:
- For $y=0$ ($80\%$ of sample): $\rho = \mathbf{+1.00000}$
- For $y=1$ ($20\%$ of sample): $\rho = \mathbf{-1.00000}$

When pooled across all observations, the $80\%$ majority class dominates the rank ordering, mechanically generating a positive pooled correlation ($\rho \approx +0.61\text{--}0.62$). **The apparent pooled reliability relationship is largely mechanically induced by the shared algebraic dependence of both quantities on $p$ and by class composition, rather than independent learned error awareness.**

### 4.3 The Constant Base-Rate Null Control
To establish an unyielding lower bound on forecast performance, we constructed a zero-parameter null predictor that outputs the training sample prevalence ($p_i \equiv 0.20$) unconditionally for every event.

Under proper scoring rules on the clean holdout ($N=12,300$, actual downside rate $20.0\%$):
- The constant null predictor achieves a Brier score of:
  $$\text{Brier}_{\text{null}} = \frac{1}{N} \sum_{i=1}^N (0.20 - y_i)^2 = 0.80(0.20 - 0)^2 + 0.20(0.20 - 1)^2 = 0.032 + 0.128 = \mathbf{0.16000}$$
- The constant null achieves an Expected Calibration Error (ECE) of identically $\mathbf{0.00000}$.
- In contrast, the historical leaked Bayesian model achieved a Brier score of $0.1924$—substantially worse than predicting the constant base rate.

---

## 5. Clean Experimental Protocol & Preregistration

To establish an uncompromised comparison between Bayesian and deterministic adaptation, Phase 17 executed a complete re-implementation under strict preregistration.

### 5.1 Canonical Clean Text Pipeline
All training, validation, and test text inputs were generated using a minimal metadata-based clean serialization template containing zero post-event market variables:
```python
clean_text = f"{company} ({ticker}) quarterly earnings announcement on {date_str}."
```
Every text string across all 18,201 events in the partitioned universe was verified against automated assertion checks ensuring the complete absence of forbidden tokens (`move_1d`, `move_30d`, `1-day move was`, `30-day move was`, `-4.69`, and `downside`).

### 5.2 Chronological Partitioning & Data Hashes
Data was partitioned chronologically to eliminate lookahead bias:
- **Clean Training Universe**: Pre-2022 ($N_{\text{train}}=11,985$ events; SHA256: `5a6e316546c59eeb98b42df3c8a4459f8bfb291e9216d2d8109355f45c4c601b`).
- **Clean Validation Universe**: Pre-2022 ($N_{\text{val}}=2,116$ events; SHA256: `9c998f01ca784b7200103e698ca316b8f2678ad167dc1643d88736590cae7a2e`).
- **Clean Out-of-Sample Universe**: 2022–May 2025 ($N_{\text{unique}}=4,100$ events; SHA256: `cf89dbe5785c8cbd5ccae1e2002eaa4626242fdaa8f3ef68af54e5bd44cd7c8d`).

### 5.3 Preregistration and Pre-Commit
Prior to training or evaluating on out-of-sample data, the experimental plan was formalized in `artifacts/phase17_clean_bayesian/phase17_preregistration.json` and committed to version control at commit `bd84126`. The preregistration locked:
- Model architectures and parameter counts ($591,362$ trainable weights).
- Optimizer recipes: AdamW, batch size 32, 2 epochs, learning rate $2\times 10^{-4}$, weight decay $0.0$ for Deterministic and $0.01$ for Bayesian, KL beta $0.003$.
- Predeclared equivalence margins for Two One-Sided Tests (TOST):
  - AUROC difference: $\pm 0.005$
  - Brier score difference: $\pm 0.002$
  - Log loss difference: $\pm 0.010$
  - ECE difference: $\pm 0.010$

### 5.4 Clustered Block Bootstrap Protocol
In financial panel data, earnings announcements cluster heavily on specific calendar dates, creating cross-sectional return correlations driven by common macroeconomic factors. Treating $N=12,300$ evaluations as independent would artificially deflate standard errors. 

To ensure valid statistical inference, all confidence intervals and hypothesis tests were conducted using a **calendar-week clustered block bootstrap** ($B=1,000$ iterations). All observations occurring in the same calendar week were sampled together with replacement across the 168 historical weeks, preserving simultaneous cross-asset correlation structures.

---

## 6. Clean Forecast Results

Table 1 presents the primary out-of-sample forecasting performance pooled across all 12,300 evaluations (3 seeds $\times$ 4,100 clean events) spanning 2022 through 2025.

### Table 1: Clean Out-of-Sample Forecast Performance ($N=12,300$, Pooled 2022–2025)

| System | Forward Passes | AUROC | Average Precision | Brier Score | Log Loss | ECE | Mean $p$ | Std $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base-Rate Null ($p \equiv 0.20$)** | 0 | 0.50000 | 0.20000 | 0.16000 | 0.50040 | 0.00000 | 0.20000 | 0.00000 |
| **Clean Deterministic LoRA** | 1 | 0.51312<br>`[0.503, 0.523]` | 0.21049 | 0.16079<br>`[0.159, 0.162]` | 0.50332<br>`[0.500, 0.507]` | 0.01785 | 0.18747 | 0.03083 |
| **Clean Bayesian Posterior Mean** | 1 | 0.52531<br>`[0.516, 0.535]` | 0.22358 | 0.15987<br>`[0.158, 0.161]` | 0.50007<br>`[0.497, 0.503]` | 0.00686 | 0.19314 | 0.02049 |
| **Clean Bayesian MC32** | 32 | 0.52504<br>`[0.516, 0.535]` | 0.22315 | 0.15987<br>`[0.158, 0.161]` | 0.50009<br>`[0.497, 0.503]` | 0.00695 | 0.19331 | 0.02053 |

*Note: Brackets report 95% confidence intervals from B=1,000 calendar-week clustered block bootstrap iterations.*

### 6.1 Discrimination & Scoring Analysis
1. **Modest Discrimination Gain for Bayesian LoRA**: Clean Bayesian Posterior Mean achieves an AUROC of $0.52531$, compared to $0.51312$ for Clean Deterministic LoRA (point difference $+0.01195$, $95\%\text{ CI: } [+0.00306, +0.02126]$). While statistically positive, an AUROC of $0.525$ indicates weak discriminative capability, barely exceeding a random coin flip ($0.500$).
2. **Proper Scoring Rule Equivalence**: In proper scoring rules, the difference between architectures is minimal:
   - **Brier Score**: $\Delta \text{Brier} = -0.00092$ ($95\%\text{ CI: } [-0.00115, -0.00070]$). The entire confidence interval falls strictly within the predeclared equivalence margin of $\pm 0.002$. Under standard equivalence testing (TOST), the two models are declared **statistically equivalent**.
   - **Log Loss**: $\Delta \text{LogLoss} = -0.00324$ ($95\%\text{ CI: } [-0.00411, -0.00246]$), falling strictly within the $\pm 0.010$ equivalence margin.
3. **Comparison Against the Base-Rate Null**: The constant null predictor ($p \equiv 0.20$) achieves a Brier score of $0.16000$. Clean Deterministic LoRA ($0.16079$) performs slightly worse than the null, while Clean Bayesian LoRA ($0.15987$) improves upon the null by a negligible $0.00013$ points.

### 6.2 Regime and Seed Disaggregation
Table 2 breaks down out-of-sample performance across the three clean historical regimes.

### Table 2: Performance Across Historical Market Regimes

| Historical Regime | Calendar Period | Clean Events ($N$) | Det LoRA AUROC | Bayes Mean AUROC | Det LoRA Brier | Bayes Mean Brier | Null Base-Rate Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2022 Regime** | 2022-01-01 to 2022-12-31 | 3,573 | 0.50925 | 0.51501 | 0.15767 | 0.15688 | 0.15688 |
| **2023 Regime** | 2023-01-01 to 2023-12-31 | 3,543 | 0.51303 | 0.51661 | 0.15646 | 0.15570 | 0.15583 |
| **2024–2025 Regime** | 2024-01-01 to 2025-05-15 | 5,184 | 0.51692 | 0.53786 | 0.16589 | 0.16477 | 0.16500 |

Across all regimes, model Brier scores track the base-rate null within $0.001$. Furthermore, seed-level disaggregation reveals initialization sensitivity:
- In **Seed 123**, Clean Deterministic LoRA achieves higher discrimination than Bayesian LoRA ($\text{AUROC} = 0.5227$ vs. $0.5098$).
- In **Seeds 42 and 777**, Bayesian LoRA achieves higher discrimination ($\text{AUROC} = 0.5577$ and $0.5581$ vs. $0.5345$ and $0.5191$).

Neither architecture eliminates seed instability, and neither demonstrates consistent superiority across all evaluation slices.

---

## 7. Bayesian Inference Cost & Monte Carlo Redundancy

A central justification for Bayesian neural networks is that drawing multiple stochastic parameter samples at test time captures epistemic variance and improves predictive calibration. We directly evaluated this hypothesis by comparing 32-draw Monte Carlo sampling (MC32) against a single forward pass evaluating the posterior mean parameters $\mu_A, \mu_B$.

Table 3 details the computational latency and statistical equivalence between the two inference configurations.

### Table 3: Posterior Mean vs. MC32 Computational & Statistical Comparison

| Metric / Attribute | Posterior Mean ($\mu$) | MC32 Sampling | Pairwise Difference (MC32 - Mean) | 95% Clustered CI | Predeclared Equivalence Margin | Equivalence Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Forward Passes / Event** | 1 | 32 | $+31$ | — | — | — |
| **Inference Latency** | 7.89 ms | 241.02 ms | $+233.13\text{ ms } (+30.6\times)$ | — | — | Compute Inefficient |
| **AUROC** | 0.52531 | 0.52504 | -0.00028 | `[-0.00080, +0.00027]` | $\pm 0.005$ | **EQUIVALENT** |
| **Average Precision** | 0.22358 | 0.22315 | -0.00043 | `[-0.00120, +0.00035]` | — | **EQUIVALENT** |
| **Brier Score** | 0.15987 | 0.15987 | +0.00001 | `[+0.00000, +0.00002]` | $\pm 0.002$ | **EQUIVALENT** |
| **Log Loss** | 0.50007 | 0.50009 | +0.00003 | `[+0.00000, +0.00005]` | $\pm 0.010$ | **EQUIVALENT** |
| **Expected Calibration Error** | 0.00686 | 0.00695 | +0.00022 | `[-0.00064, +0.00186]` | $\pm 0.010$ | **EQUIVALENT** |

### Findings on Inference Redundancy:
1. **Practical Identity of Predictions**: The predicted probabilities from MC32 and the single-pass Posterior Mean correlate at $r > 0.9999$. Across all 12,300 evaluations, the maximum absolute difference between $\bar{p}_{\text{MC32}}$ and $p_\mu$ was less than $0.002$.
2. **Strict Equivalence Across All Dimensions**: As shown in Table 3, all pairwise differences in AUROC, Brier score, log loss, and ECE fall strictly within the preregistered equivalence margins.
3. **Substantial Latency Penalty**: Evaluating 32 stochastic draws requires $241.02\text{ ms}$ per event, compared to $7.89\text{ ms}$ for the posterior mean. In an intraday production environment processing thousands of corporate events, this represents a $30.6\times$ latency penalty with zero statistical benefit.

Under this experimental setup, **Monte Carlo posterior weight sampling was computationally redundant**. Practitioners deploying Bayesian LoRA can evaluate the variational posterior mean weights in a single deterministic pass without loss of accuracy.

---

## 8. Downstream Portfolio Evaluation (Secondary Evidence)

Downstream portfolio backtests are treated strictly as secondary evidence because the underlying clean point-prediction signal is inherently weak. To verify whether small differences in forecast distributions translate into economic outcomes, we executed a clean portfolio sanity check under unadjusted Markowitz optimization ($M_0$).

### 8.1 Portfolio Simulation Structure
- **Universe & Cohorts**: All active stocks reporting earnings in each calendar week.
- **Holding Period**: $H=4$ weeks (staggered weekly cohorts with equal capital allocation).
- **Signal Construction**: Assets ranked by downside aversion signal $s_i = -p_i$ (preferring assets with lowest predicted downside probability).
- **Portfolio Selection**: Equal-weighted long position in top $20\%$ quintile of signal-ranked assets.
- **Transaction Costs**: Fixed $10\text{ basis points}$ per unit turnover.
- **Risk Aversion**: Mean-variance utility $U = \mu_p - \frac{1}{2} \gamma \sigma_p^2$ with $\gamma = 1.0$.

### 8.2 Clean Economic Results
Table 4 presents the annualized portfolio metrics across signals.

### Table 4: Portfolio Sanity Check ($M_0$ Unadjusted Markowitz, Secondary Evidence)

| System Name | Signal Used | Annualized Return | Annualized Volatility | Sharpe Ratio | Max Drawdown | Utility ($\gamma=1.0$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **System A: Clean Deterministic LoRA** | $p_{\text{det\_clean}}$ | +21.00% | 37.64% | 0.5580 | -38.11% | +0.1392 |
| **System B: Clean Bayesian Posterior Mean** | $p_{\text{bayes\_mean}}$ | +12.41% | 36.93% | 0.3360 | -40.20% | +0.0559 |
| **System C: Clean Bayesian MC32** | $p_{\text{bayes\_mc32}}$ | +10.83% | 36.94% | 0.2932 | -43.71% | +0.0401 |
| **Null Base-Rate ($p \equiv 0.20$)** | $p_{\text{null}}$ | +7.85% | 80.66% | 0.0974 | -25.47% | -0.2468 |

### Portfolio Takeaways:
1. **Clean Deterministic LoRA Outperforms Bayesian LoRA**: In this stylized allocation, Clean Deterministic LoRA achieves a Sharpe ratio of $0.5580$ and utility of $+0.1392$, compared to Sharpe $0.3360$ and utility $+0.0559$ for the Bayesian Posterior Mean.
2. **MC32 Decreases Economic Performance**: Aligning with our inference redundancy findings, MC32 sampling produces slightly noisier rankings, reducing the Sharpe ratio to $0.2932$.
3. **Absence of Robust Uncertainty Gains**: Across the broader research arc, complex uncertainty mappings ($M_{13}$ hysteresis timing, $M_{16}$ risk inflation) failed to deliver consistent, regime-invariant gains. Adjusting risk budgets or asset covariances by model uncertainty cannot compensate for underlying forecast weakness.

---

## 9. Discussion & Methodological Lessons

The findings of this investigation provide several critical methodological lessons for financial machine learning and applied uncertainty quantification. Table 5 summarizes the vulnerabilities identified during our audit.

### Table 5: Summary of Experimental Vulnerabilities and Methodological Lessons

| Vulnerability / Issue | What Initially Appeared | Forensic Audit Finding | Final Scientific Consequence |
| :--- | :--- | :--- | :--- |
| **Target Outcome Leakage** | Model achieved near-perfect validation AUROC ($>0.999$) and apparent high confidence. | Continuous label realization `move_1d` was concatenated into training text strings (`". 1-day move was " + move_1d`). | Historical checkpoints were compromised; clean test-time evaluation suffered distribution collapse to near-chance ($\text{AUROC} \approx 0.50$). |
| **Reliability Metric Artifact** | Predictive entropy strongly tracked prediction error ($\rho \approx 0.60$), suggesting error-awareness. | In binary classification under class imbalance, rank correlation within $y=0$ is mechanically $+1.0$ and within $y=1$ is $-1.0$ for $p < 0.5$. | Pooled correlation reflects probability shrinkage toward zero rather than learned epistemic error awareness. |
| **Base-Rate / Null Baseline** | Bayesian training appeared to provide superior calibration compared to unregularized models. | A zero-parameter constant base-rate predictor ($p \equiv 0.20$) achieves Brier score of $0.16000$ and zero calibration error. | Fine-tuned models barely move off the base rate ($\bar{p} \approx 0.19$, $\sigma_p \approx 0.02$), showing negligible text signal. |
| **Monte Carlo Redundancy** | 32-draw posterior sampling was presumed necessary to capture weight uncertainty. | Posterior mean parameters $\mu$ produce predictions that correlate with MC32 at $r > 0.9999$, with all metrics within equivalence bounds. | MC sampling introduces a $30.6\times$ latency penalty with zero measurable forecast gain; single-pass inference is strictly preferred. |
| **Portfolio Instability** | Uncertainty-aware risk inflation ($M_{16}$) appeared to deliver substantial Sharpe gains in 2023. | $M_{16}$ acted as an unconditional bear-market de-risking multiplier, causing severe underperformance in 2022 and 2024–2025. | No robust ex-ante portfolio mapping survived clean adjudication; under unadjusted Markowitz ($M_0$), deterministic LoRA achieved higher utility. |

### 9.1 The Subtle Hazards of Serialization Leakage
In multimodal and NLP architectures applied to tabular financial data, data serialization often involves assembling text prompts from disparate database columns. If post-event price realizations, analyst revisions, or label-derived continuous values enter the prompt template, pre-trained transformer attention heads will exploit them. Because transformers excel at extracting numeric tokens, even indirect outcome hints will induce leakage. Researchers must enforce automated token-level assertion tests on all serialized inputs.

### 9.2 The Necessity of Disaggregating Reliability Metrics
Evaluating uncertainty via rank correlations between predictive entropy and prediction error ($\text{Spearman}(\text{entropy}, |p - y|)$) is widespread in the deep learning literature. We have shown that on class-imbalanced datasets where predictions undergo shrinkage toward the majority class, this metric is mathematically degenerate. Researchers must report class-conditional correlations:
$$\rho_{y=0} = \text{Spearman}(H(p), |p - 0|), \quad \rho_{y=1} = \text{Spearman}(H(p), |p - 1|)$$
A model that genuinely anticipates errors should exhibit positive rank correlation within both classes. When $\rho_{y=0} = +1.0$ and $\rho_{y=1} = -1.0$, the pooled correlation is largely mechanically induced by the shared algebraic dependence of both quantities on $p$ and by class composition.

### 9.3 The Primacy of the Constant Base-Rate Null
In low-signal financial environments, comparing complex models solely against unregularized neural baselines creates an illusion of progress. An unregularized model may overfit and produce erratic probabilities, making a regularized or Bayesian model appear artificially superior. Benchmarking against a zero-parameter constant base-rate predictor ($p_i \equiv \bar{y}_{\text{train}}$) provides an essential reality check. If a complex neural model cannot decisively beat the base rate under proper scoring rules, its practical utility remains unproven.

---

## 10. Limitations

We emphasize the following boundary conditions:
1. **Minimal Metadata Input Representation**: The primary limitation of our clean rerun is that inputs were restricted to structured corporate metadata (`f"{company} ({ticker}) quarterly earnings announcement on {date_str}."`) rather than rich earnings call transcripts, 10-K/10-Q filings, or news narrative. While this guaranteed zero target leakage, it contained minimal linguistic content. Our findings do not rule out the possibility that richer textual disclosures contain exploitable predictive signals.
2. **Weak Overall Discrimination**: Across all clean out-of-sample evaluations, both deterministic and Bayesian models achieved discriminative performance barely exceeding random chance ($\text{AUROC} \approx 0.513\text{--}0.525$), reflecting the high noise-to-signal ratio of short-horizon post-earnings returns.
3. **Competitiveness of the Base-Rate Null**: In an imbalanced setting ($20\%$ downside), predicting the constant base rate ($p \equiv 0.20$) achieves $\text{Brier} = 0.16000$, matching or slightly outperforming fine-tuned models.
4. **Scope of Portfolio Conclusions**: Portfolio backtests are secondary evidence operating on weak underlying point forecasts; they should not be generalized to asset allocation settings with strong underlying return signals.
5. **Architectural & Hyperparameter Specificity**: Experiments were conducted using FinBERT (BERT-base) with rank-16 LoRA on self-attention projections. Bayesian hyperparameters followed standard literature conventions rather than an exhaustive post-hoc search after leakage correction.
6. **Temporal Sample Breadth**: Out-of-sample evaluation was restricted to three historical regimes (2022 through May 2025; $N_{\text{unique}}=4,100$ events). While these regimes span contrasting market cycles, 3.5 years represents a finite sample where all seeds share common macroeconomic histories.

---

## 11. Conclusion

After removing target leakage and auditing reliability metrics, Bayesian LoRA provided a modest discrimination improvement over matched deterministic LoRA in this specific earnings-event setting ($\text{AUROC} = 0.5253$ vs. $0.5131$), while proper scoring performance remained close ($\Delta \text{Brier} = -0.00092$) and a constant prevalence predictor ($p \equiv 0.20$) was highly competitive ($\text{Brier} = 0.16000$). Monte Carlo posterior sampling provided no practical advantage over a single posterior-mean forward pass while incurring roughly $30\times$ higher inference latency, and uncertainty-aware portfolio adjustments established no robust clean economic advantage. More broadly, the study illustrates how leakage, algebraically dependent reliability metrics, and weak baseline discipline can create misleading evidence of uncertainty quality in financial event forecasting.

---

## References

- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv preprint arXiv:1908.10063*.
- Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight Uncertainty in Neural Networks. *International Conference on Machine Learning (ICML)*.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *International Conference on Machine Learning (ICML)*.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations (ICLR)*.
- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
