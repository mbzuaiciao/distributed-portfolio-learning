# Paper Outline: When Uncertainty Looks Useful

**Working Title**: *When Uncertainty Looks Useful: Target Leakage, Reliability Metrics, and Bayesian LoRA in Financial Event Risk*

---

## Section-by-Section Structural Specification

### 1. Introduction
- Motivation: The growing adoption of Parameter-Efficient Fine-Tuning (PEFT) and LoRA in financial NLP, coupled with the desire for well-calibrated epistemic uncertainty estimates to guide portfolio risk management.
- The Empirical Puzzle: Initial experiments indicated that Bayesian-LoRA adapters yielded remarkable predictive confidence, high error-tracking correlations, and promising selective-risk abstention.
- The Forensic Reality: Methodological audits revealed that these apparent advantages were driven by experimental artifacts: subtle target outcome leakage in training text and a mathematical artifact in the evaluation of reliability metrics under class imbalance.
- The Core Question: When evaluated under matched, clean inputs without leakage and with appropriate statistical controls, does Bayesian-LoRA training provide measurable value over regularized deterministic LoRA in financial event risk?
- Summary of Contributions:
  1. Detailed dissection of target leakage mechanics in sequence classification pipelines.
  2. Proof of mechanical correlation between predictive entropy and absolute error under probability shrinkage.
  3. Strict preregistered evaluation of clean Bayesian vs. deterministic LoRA across 12,300 clean OOS observations.
  4. Demonstration of practical equivalence between 32-draw Monte Carlo sampling and single-pass posterior-mean inference under preregistered margins.
  5. Downstream portfolio evaluation demonstrating lack of robust economic gains from Bayesian uncertainty adjustments.

---

### 2. Problem Setup & Methodological Framework
- **Downside Earnings Prediction Task**: Formulating the classification of 1-day post-earnings price shocks ($y_i = \mathbb{I}(\text{move\_1d}_i \le -4.69\%)$).
- **Base Architecture**: FinBERT (`ProsusAI/finbert`) with pre-trained sentiment representations.
- **Deterministic LoRA**: Low-rank linear adapters ($\mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B}\mathbf{A}$) injected into self-attention query and value projections ($r=16, \alpha=32$).
- **Bayesian LoRA**: Variational Gaussian posterior over low-rank weights $q(\theta) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$, optimized via Bayes by Backprop with scaled KL regularization.
- **Portfolio Motivation**: The economic hypothesis that epistemic uncertainty can guide Markowitz portfolio construction via asset-level risk inflation or dynamic cash-exposure timing.

---

### 3. Historical Uncertainty Findings & Initial Appearances
- Brief historical overview of early experimental results (Phases 11–16).
- Why Bayesian uncertainty initially appeared highly effective:
  - Strong rank correlation ($\rho \approx 0.60$) between predictive entropy and prediction error.
  - Apparent superiority over unregularized deterministic baselines in Brier score ($0.192$ vs. $0.256$).
  - Promising selective-risk retention curves suggesting effective abstention.
- **Explicit Invalidation Warning**: Formal documentation that these findings were historical artifacts of target outcome leakage and metric design flaws, quarantined from primary scientific evidence.

---

### 4. Forensic Audit Findings
- **Target Outcome Leakage Discovery**:
  - The inadvertent inclusion of `". 1-day move was " + move_1d` in training and validation text.
  - The mechanism of leakage: continuous label extraction yielding artificial training/validation AUROC $> 0.999$.
  - Extreme test-time distribution shift upon evaluating on clean text without the leaked phrase, causing discriminative collapse ($\text{AUROC} \approx 0.50$).
- **Reliability Metric Decomposition**:
  - Mathematical proof that $\text{Spearman}(\text{entropy}, |p - y|)$ is mechanically determined by class imbalance.
  - Within $y=0$: error $|p - 0| = p$, while entropy is strictly increasing in $p \in [0, 0.5]$, forcing $\rho \equiv +1.0$.
  - Within $y=1$: error $|p - 1| = 1 - p$, strictly decreasing in $p \in [0, 0.5]$, forcing $\rho \equiv -1.0$.
  - Demonstrating that pooled correlation is a mathematical consequence of 80% class imbalance, not epistemic error awareness.
- **The Constant Base-Rate Null Control**:
  - Evaluating a zero-parameter constant predictor ($p_i \equiv 0.20$).
  - Showing that the trivial null achieves a Brier score of $0.16000$, beating the leaked Bayesian model ($0.1932$) and matching clean models.

---

### 5. Clean Experimental Protocol & Preregistration
- **Canonical Clean Text Pipeline**: Constructing strictly clean text inputs: `f"{company} ({ticker}) quarterly earnings announcement on {date_str}."`
- **Data Universes & Temporal Split**: Pre-2022 training ($N=11,985$), pre-2022 validation ($N=2,116$), and clean post-training test regimes (2022, 2023, 2024–2025; $N=4,100$ unique events).
- **Matched Model Training**: Identical batch sizes, learning rates, epochs, and parameter counts across seeds 42, 123, 777.
- **Statistical Evaluation Protocol**:
  - Weekly clustered block bootstrap ($B=1,000$) preserving simultaneous cross-sectional co-occurrences.
  - Predeclared equivalence margins: AUROC $\pm 0.005$, Brier $\pm 0.002$, Log Loss $\pm 0.010$, ECE $\pm 0.010$.
  - Preregistration committed before clean OOS evaluation.

---

### 6. Clean Forecast Results
- **Discrimination**: Clean Bayesian Posterior Mean ($\text{AUROC} = 0.5253$) vs. Clean Deterministic LoRA ($\text{AUROC} = 0.5131$). Both remain weak, near-chance predictors.
- **Proper Scoring Rules**:
  - Brier score: Bayesian $0.15987$ vs. Deterministic $0.16079$ ($\Delta = -0.00092$, strictly equivalent within $\pm 0.002$).
  - Log loss: Bayesian $0.50007$ vs. Deterministic $0.50332$ ($\Delta = -0.00324$, strictly equivalent within $\pm 0.010$).
- **Calibration & Base-Rate Comparison**: Both models cluster near $\bar{p} \approx 0.187\text{--}0.193$. Trivial null ($p \equiv 0.20$) achieves $\text{Brier} = 0.16000$ and $\text{ECE} = 0.00000$.
- **Seed & Regime Stability**: Demonstrating substantial seed variance (e.g., Deterministic LoRA beats Bayesian in Seed 123; Bayesian beats Deterministic in Seed 42 and 777).

---

### 7. Bayesian Inference Cost & Monte Carlo Redundancy
- **Empirical Comparison**: 32 stochastic Monte Carlo draws (MC32) vs. a single forward pass of posterior mean weights.
- **Statistical Equivalence**: All pairwise differences between MC32 and Posterior Mean fall well within equivalence margins:
  - $\Delta \text{AUROC} = -0.00028$, $\Delta \text{Brier} = +0.00001$, $\Delta \text{LogLoss} = +0.00003$, $\Delta \text{ECE} = +0.00022$.
- **Computational Profile**: MC32 incurs a $30.6\times$ latency penalty ($241.02\text{ ms/event}$ vs. $7.89\text{ ms/event}$), rendering MC sampling computationally redundant.

---

### 8. Downstream Portfolio Evaluation
- Overview of portfolio experiments (Phases 13–15): signal shrinkage ($M_{14}$), selective abstention ($M_{15}$), risk inflation ($M_{16}$), and hysteresis exposure timing ($M_{13}$).
- Lack of robust economic value: Mappings that showed apparent benefits in 2023 caused severe underperformance in 2022 and 2024–2025.
- Clean Sanity Check ($M_0$ Unadjusted Markowitz): Clean Deterministic LoRA achieves higher Sharpe ($0.5580$ vs. $0.3360$) and higher quadratic utility ($+0.1392$ vs. $+0.0559$) than Bayesian LoRA.
- Framing: Portfolio adjustments cannot compensate for underlying forecast weakness.

---

### 9. Discussion & Methodological Recommendations
- The subtle hazards of target leakage in structured text representation.
- The necessity of evaluating against trivial base-rate null predictors in imbalanced financial data.
- The danger of pooled rank correlation metrics that conflate probability shrinkage with error prediction.
- Practical guidelines for evaluating Bayesian and uncertainty-aware methods in computational finance.

---

### 10. Limitations
- Single base model (FinBERT) and single adaptation topology (LoRA on query/value).
- Specificity to earnings announcement downside prediction.
- Text representation limited to structured company/date metadata.
- Low signal-to-noise ratio inherent in post-earnings price returns.
- Scope limited to three clean post-training market regimes (2022, 2023, 2024–2025).

---

### 11. Conclusion
- Synthesis of findings: Bayesian LoRA provides a modest discrimination gain but no practically meaningful scoring advantage or economic portfolio gain over regularized deterministic LoRA.
- Monte Carlo posterior sampling is redundant relative to single-pass posterior-mean evaluation.
- Broader moral: Rigorous forensic auditing and trivial null baselines are indispensable when evaluating uncertainty in financial machine learning.
