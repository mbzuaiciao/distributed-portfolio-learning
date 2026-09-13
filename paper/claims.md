# Claims Ledger & Rhetorical Guardrails

This ledger records every empirical and methodological claim in the paper, specifying its verification status, supporting artifacts, provenance classification, allowed scientific wording, forbidden overclaims, and explicit scope boundaries.

---

## 1. Methodological & Forensic Claims

### Claim 1.1: Target Outcome Leakage in Historical Training Text
- **Status**: **SUPPORTED (CONFIRMED BY CODE AUDIT)**
- **Supporting Artifacts**:
  - `artifacts/phase16_1_audit/input_leakage_audit.json`
  - `artifacts/phase17_clean_bayesian/legacy_checkpoint_status.json`
  - `experiments/run_phase11_replication.py:L132`
  - `experiments/run_phase16_deterministic_control.py:L427`
- **Data Provenance**: Forensic Code Audit
- **Scope Limitation**: Applies strictly to the historical Phase 11 and Phase 16 training scripts within this codebase where text serialization inadvertently included `move_1d`.
- **Allowed Wording**:
  - *"The historical training and validation data pipeline inadvertently concatenated continuous post-event price realizations (`. 1-day move was ` + `move_1d`) into the model input text."*
  - *"Because the downstream binary target $y_i$ was deterministically defined by thresholding `move_1d`, the model had access to ground-truth outcome tokens during training, producing artificial validation AUROC ($> 0.999$) followed by out-of-sample distribution collapse when evaluated on clean text."*
- **Forbidden Overclaims**:
  - *"Target leakage was the sole reason why Bayesian uncertainty failed to deliver portfolio gains."*
  - *"All past findings in the broader literature are invalidated by target leakage."*
  - *"The leak was intentional."*

---

### Claim 1.2: Mechanical Distortion in Spearman Reliability Correlation
- **Status**: **SUPPORTED (ANALYTICALLY DERIVED & EMPIRICALLY CONFIRMED)**
- **Supporting Artifacts**:
  - `artifacts/phase16_1_audit/within_class_reliability.csv`
  - `artifacts/phase17_clean_bayesian/forecast_metrics.csv`
- **Data Provenance**: Valid Clean Primary & Analytical Derivation
- **Scope Limitation**: Applies to binary classification where labels are imbalanced and predicted probabilities undergo shrinkage toward the majority class (e.g., $p < 0.5$).
- **Allowed Wording**:
  - *"The apparent pooled Spearman rank correlation between predictive entropy and absolute prediction error ($\rho \approx 0.60$) is largely mechanically induced by the shared algebraic dependence of both quantities on $p$ and by class composition."*
  - *"Because entropy and absolute error are both deterministic functions of $p$, within-class correlations for probabilities below $0.5$ are mechanically forced to $+1.0000$ for the non-downside class ($y=0$, representing $80\%$ of events) and $-1.0000$ for the downside class ($y=1$). The pooled metric is therefore not independent evidence of learned error awareness."*
- **Forbidden Overclaims**:
  - *"The correlation is entirely a meaningless artifact in all settings."*
  - *"Predictive entropy has zero relationship to model accuracy."*

---

### Claim 1.3: Baseline Competitiveness of the Constant Base-Rate Null
- **Status**: **SUPPORTED (EMPIRICALLY CONFIRMED)**
- **Supporting Artifacts**:
  - `artifacts/phase16_1_audit/base_rate_control.csv`
  - `artifacts/phase17_clean_bayesian/forecast_metrics.csv`
- **Data Provenance**: Valid Clean Primary
- **Scope Limitation**: Evaluated on binary 1-day earnings downside classification ($20\%$ base rate) using minimal clean metadata inputs across 2022–2025.
- **Allowed Wording**:
  - *"A zero-parameter constant base-rate predictor ($p_i \equiv 0.20$) achieves an out-of-sample Brier score of $0.16000$ and zero calibration error across the clean 2022–2025 regimes."*
  - *"Neither regularized deterministic LoRA ($\text{Brier} = 0.16079$) nor Bayesian LoRA ($\text{Brier} = 0.15987$) demonstrates substantial scoring-rule improvement over predicting the training prevalence unconditionally on this minimal input representation."*
- **Forbidden Overclaims**:
  - *"Machine learning is universally incapable of beating the base rate in finance."*
  - *"Pre-trained language models contain zero financial signal."*

---

## 2. Model Comparison & Forecasting Claims

### Claim 2.1: Clean Bayesian vs. Deterministic Out-of-Sample Performance
- **Status**: **SUPPORTED (STATISTICALLY BOUNDED)**
- **Supporting Artifacts**:
  - `artifacts/phase17_clean_bayesian/forecast_metrics.csv`
  - `artifacts/phase17_clean_bayesian/bootstrap_intervals.csv`
  - `artifacts/phase17_clean_bayesian/equivalence_analysis.csv`
- **Data Provenance**: Valid Clean Primary ($N_{\text{train}}=11,985$, $N_{\text{val}}=2,116$, $N_{\text{test}}=12,300$)
- **Scope Limitation**: Specifically bounded to FinBERT with rank-16 LoRA on query/value projections predicting 1-day earnings downside from minimal metadata inputs.
- **Allowed Wording**:
  - *"Under this earnings-event prediction setup, dataset, label construction, FinBERT/LoRA architecture, and minimal clean input representation, Bayesian LoRA achieves a modest improvement in discriminative ability over regularized deterministic LoRA ($\text{AUROC} = 0.5253$ vs. $0.5131$, mean difference $+0.012$, $95\%\text{ CI: } [+0.003, +0.021]$)."*
  - *"Proper scoring rules (Brier score and log loss) were practically equivalent between Bayesian posterior-mean LoRA and regularized deterministic LoRA under preregistered equivalence margins ($\Delta \text{Brier} = -0.00092$, $95\%\text{ CI: } [-0.00115, -0.00070]$; $\Delta \text{LogLoss} = -0.00324$, $95\%\text{ CI: } [-0.00411, -0.00246]$)."*
  - *"Both architectures remain weak predictors of 1-day post-earnings price return direction on this clean metadata representation."*
- **Forbidden Overclaims**:
  - *"Bayesian LoRA clearly dominates deterministic LoRA."*
  - *"Deterministic LoRA universally outperforms Bayesian LoRA."*
  - *"Bayesian LoRA has zero value for financial NLP."*
  - *"Bayesian LoRA is proven to be useless."*

---

### Claim 2.2: Computational Redundancy of Monte Carlo Posterior Sampling (MC32)
- **Status**: **SUPPORTED (STATISTICALLY EQUIVALENT & EXPERIMENTALLY BOUNDED)**
- **Supporting Artifacts**:
  - `artifacts/phase17_clean_bayesian/equivalence_analysis.csv`
  - `artifacts/phase17_clean_bayesian/compute_cost.csv`
  - `artifacts/phase17_clean_bayesian/bootstrap_intervals.csv`
- **Data Provenance**: Valid Clean Primary
- **Scope Limitation**: Applies to inference-time evaluation of the trained variational Bayesian LoRA model on this dataset.
- **Allowed Wording**:
  - *"For this task, dataset, and trained model, 32-draw Monte Carlo posterior weight sampling was practically equivalent to single-pass posterior-mean inference under preregistered equivalence margins ($\Delta \text{AUROC} = -0.00028$, $\Delta \text{Brier} = +0.00001$, $\Delta \text{LogLoss} = +0.00003$, $\Delta \text{ECE} = +0.00022$)."*
  - *"Because MC32 incurs an inference latency penalty of roughly $30\times$ ($241.0\text{ ms/event}$ vs. $7.9\text{ ms/event}$) with no measurable improvement in calibration or scoring accuracy, test-time MC sampling was computationally redundant in this experiment."*
- **Forbidden Overclaims**:
  - *"We prove Monte Carlo sampling is identical to the posterior mean."*
  - *"Monte Carlo sampling is always useless in deep learning."*
  - *"Bayesian posterior sampling can never provide value in any domain."*

---

## 3. Economic & Portfolio Claims (Secondary Evidence)

### Claim 3.1: Downstream Portfolio Economic Value of Uncertainty Mappings
- **Status**: **NOT SUPPORTED AS PRIMARY / SECONDARY EVIDENCE ONLY**
- **Supporting Artifacts**:
  - `artifacts/phase17_clean_bayesian/portfolio_sanity_check.csv`
  - `artifacts/phase13c_regime_evaluation/regime_adjudication_summary.csv`
  - `artifacts/phase14_m16_mechanism/mechanism_decomposition_summary.json`
- **Data Provenance**: Valid Clean Secondary & Primary Sanity Check
- **Scope Limitation**: Simplified Markowitz optimization ($H=4$, $\gamma=1.0$, $10\text{ bps}$ turnover) evaluated on clean post-training regimes.
- **Allowed Wording**:
  - *"Downstream portfolio experiments are treated as secondary evidence because the underlying clean point-prediction signal is inherently weak."*
  - *"Across multiple historical regimes, no robust, ex-ante beneficial portfolio policy based on asset-level or portfolio-level Bayesian uncertainty adjustments was established."*
  - *"In clean unadjusted Markowitz portfolio simulation ($M_0$), Clean Deterministic LoRA achieved higher Sharpe ($0.5580$ vs. $0.3360$) and higher utility ($+0.1392$ vs. $+0.0559$) than Bayesian Posterior Mean, demonstrating that uncertainty-aware adjustments did not translate into clean economic gains in this setting."*
- **Forbidden Overclaims**:
  - *"Uncertainty-aware portfolio optimization is fundamentally flawed."*
  - *"Deterministic models are universally superior portfolio engines."*
  - *"Bayesian methods can never improve financial utility."*
