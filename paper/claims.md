# Claims Ledger

This ledger records every scientific and empirical claim in the paper, its verification status, supporting artifacts, provenance classification, allowed wording, and explicitly forbidden overclaims.

---

## 1. Methodological & Forensic Claims

### Claim 1.1: Target Outcome Leakage in Historical Training Text
- **Status**: **SUPPORTED (CONFIRMED BY FORENSIC AUDIT)**
- **Supporting Artifacts**:
  - `artifacts/phase16_1_audit/input_leakage_audit.json`
  - `artifacts/phase17_clean_bayesian/legacy_checkpoint_status.json`
  - `experiments/run_phase11_replication.py:L132`
  - `experiments/run_phase16_deterministic_control.py:L427`
- **Data Provenance**: Forensic Audit (Historical Codebase Inspection)
- **Allowed Wording**:
  - *"The historical training and validation data pipeline inadvertently concatenated continuous post-event price realizations (`. 1-day move was ` + `move_1d`) into the model input text."*
  - *"Because the downstream binary target $y_i$ was deterministically defined by thresholding `move_1d`, the model had access to ground truth during training, achieving near-perfect validation AUROC ($> 0.999$) followed by severe out-of-sample distribution collapse on clean text."*
  - *"Target leakage in training text was discovered through systematic forensic auditing of input pipelines."*
- **Forbidden Overclaims**:
  - *"Target leakage was the sole reason why Bayesian uncertainty failed to deliver portfolio gains."*
  - *"The original researchers intentionally introduced target leakage."*
  - *"All past findings in the entire field are invalidated by target leakage."*

---

### Claim 1.2: Mechanical Artifact in Spearman Reliability Correlation
- **Status**: **SUPPORTED (MATHEMATICALLY PROVED & EMPIRICALLY CONFIRMED)**
- **Supporting Artifacts**:
  - `artifacts/phase16_1_audit/within_class_reliability.csv`
  - `artifacts/phase17_clean_bayesian/forecast_metrics.csv`
- **Data Provenance**: Valid Clean Primary & Analytical Proof
- **Allowed Wording**:
  - *"The strong pooled Spearman rank correlation between predictive entropy and absolute prediction error ($\rho \approx 0.60$) is largely a mathematical artifact of class imbalance under probability shrinkage below $0.50$."*
  - *"When disaggregated conditional on the true class, the rank correlation is identically $+1.0000$ for non-downside events ($y=0$, representing $80\%$ of observations) and $-1.0000$ for downside events ($y=1$), demonstrating that pooled correlation reflects probability shrinkage rather than epistemic error awareness."*
- **Forbidden Overclaims**:
  - *"Predictive entropy has zero relationship with error in any setting."*
  - *"Calibration metrics are always invalid."*

---

### Claim 1.3: Baseline Competitiveness of the Constant Base-Rate Null
- **Status**: **SUPPORTED (EMPIRICALLY CONFIRMED)**
- **Supporting Artifacts**:
  - `artifacts/phase16_1_audit/base_rate_control.csv`
  - `artifacts/phase17_clean_bayesian/forecast_metrics.csv`
- **Data Provenance**: Valid Clean Primary
- **Allowed Wording**:
  - *"A zero-parameter constant base-rate predictor ($p_i \equiv 0.20$) achieves an out-of-sample Brier score of $0.16000$ and zero calibration error across the clean 2022–2025 regimes."*
  - *"Neither regularized deterministic LoRA ($\text{Brier} = 0.16079$) nor Bayesian LoRA ($\text{Brier} = 0.15987$) demonstrates substantial scoring-rule improvement over predicting the training prevalence unconditionally."*
- **Forbidden Overclaims**:
  - *"Machine learning is completely incapable of beating the base rate."*
  - *"FinBERT contains zero predictive information about equities."*

---

## 2. Model Comparison & Forecasting Claims

### Claim 2.1: Clean Bayesian vs. Deterministic Out-of-Sample Performance
- **Status**: **SUPPORTED (STATISTICALLY BOUNDED)**
- **Supporting Artifacts**:
  - `artifacts/phase17_clean_bayesian/forecast_metrics.csv`
  - `artifacts/phase17_clean_bayesian/bootstrap_intervals.csv`
  - `artifacts/phase17_clean_bayesian/equivalence_analysis.csv`
- **Data Provenance**: Valid Clean Primary ($N=12,300$ clean OOS evaluations)
- **Allowed Wording**:
  - *"When trained on strictly clean text without target leakage, Bayesian LoRA achieves a small improvement in discriminative ability over regularized deterministic LoRA ($\text{AUROC} = 0.5253$ vs. $0.5131$, point difference $+0.012$)."*
  - *"Proper scoring rules (Brier score and log loss) are statistically and practically equivalent between Bayesian posterior-mean LoRA and regularized deterministic LoRA under preregistered equivalence margins ($\Delta \text{Brier} = -0.00092$, $95\%\text{ CI: } [-0.00115, -0.00070]$; $\Delta \text{LogLoss} = -0.00324$, $95\%\text{ CI: } [-0.00411, -0.00246]$)."*
  - *"Both architectures remain weak, near-chance predictors of 1-day post-earnings price return direction on this clean text representation."*
- **Forbidden Overclaims**:
  - *"Bayesian LoRA clearly dominates deterministic LoRA."*
  - *"Deterministic LoRA universally outperforms Bayesian LoRA."*
  - *"Bayesian LoRA has zero value for financial NLP."*
  - *"Bayesian LoRA is proven to be useless."*

---

### Claim 2.2: Computational Redundancy of Monte Carlo Posterior Sampling (MC32)
- **Status**: **SUPPORTED (STATISTICALLY EQUIVALENT & STRICTLY VERIFIED)**
- **Supporting Artifacts**:
  - `artifacts/phase17_clean_bayesian/equivalence_analysis.csv`
  - `artifacts/phase17_clean_bayesian/compute_cost.csv`
  - `artifacts/phase17_clean_bayesian/bootstrap_intervals.csv`
- **Data Provenance**: Valid Clean Primary
- **Allowed Wording**:
  - *"For this task and trained model, 32-draw Monte Carlo posterior weight sampling (MC32) was practically and statistically indistinguishable from a single deterministic forward pass using the variational posterior mean weights."*
  - *"All pairwise differences between MC32 and Posterior Mean fall strictly within preregistered equivalence margins ($\Delta \text{AUROC} = -0.00028$, $\Delta \text{Brier} = +0.00001$, $\Delta \text{LogLoss} = +0.00003$, $\Delta \text{ECE} = +0.00022$)."*
  - *"Because MC32 incurs a roughly $30\times$ inference latency penalty ($241.0\text{ ms/event}$ vs. $7.9\text{ ms/event}$) with zero measurable gain in calibration or discrimination, test-time MC sampling was computationally redundant for this experiment."*
- **Forbidden Overclaims**:
  - *"Monte Carlo sampling is always useless in deep learning."*
  - *"Bayesian posterior sampling can never provide value in any domain."*

---

### Claim 2.3: Selective Risk & Abstention Efficacy
- **Status**: **CONDITIONALLY SUPPORTED (MODEST IDENTICAL GAIN)**
- **Supporting Artifacts**:
  - `artifacts/phase17_clean_bayesian/selective_risk.csv`
- **Data Provenance**: Valid Clean Primary
- **Allowed Wording**:
  - *"Ranking predictions by confidence margin and abstaining on the most uncertain events yields modest, virtually identical reductions in realized downside risk for both deterministic and Bayesian models (dropping from $20.0\%$ at $100\%$ retention to $\approx 19.3\text{--}19.5\%$ at $80\%$ retention)."*
  - *"Bayesian training does not provide superior selective-risk curves compared to regularized deterministic LoRA."*
- **Forbidden Overclaims**:
  - *"Selective classification completely fails."*
  - *"Bayesian uncertainty provides exceptional selective abstention benefits."*

---

## 3. Economic & Portfolio Claims

### Claim 3.1: Downstream Portfolio Economic Value of Uncertainty Mappings
- **Status**: **NOT SUPPORTED (NO ROBUST ECONOMIC ADVANTAGE ESTABLISHED)**
- **Supporting Artifacts**:
  - `artifacts/phase17_clean_bayesian/portfolio_sanity_check.csv`
  - `artifacts/phase13c_regime_evaluation/regime_adjudication_summary.csv`
  - `artifacts/phase14_m16_mechanism/mechanism_decomposition_summary.json`
- **Data Provenance**: Valid Clean Primary & Secondary
- **Allowed Wording**:
  - *"Across multiple historical regimes (2022, 2023, 2024–2025), no robust, ex-ante beneficial portfolio policy based on asset-level or portfolio-level Bayesian uncertainty adjustments was established."*
  - *"In clean unadjusted Markowitz portfolio simulation ($M_0$), Clean Deterministic LoRA achieved higher Sharpe ($0.5580$ vs. $0.3360$) and higher utility ($+0.1392$ vs. $+0.0559$) than Bayesian Posterior Mean."*
  - *"Uncertainty-to-portfolio mappings cannot overcome the low signal-to-noise ratio of the underlying point forecasts."*
- **Forbidden Overclaims**:
  - *"Uncertainty-aware portfolio optimization is mathematically flawed."*
  - *"Bayesian methods can never help any portfolio."*
  - *"Risk inflation is universally destructive."*
