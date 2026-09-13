# Phase 18.1 Submission-Readiness Scientific Audit

This audit systematically checks every quantitative claim, sample size, input representation description, and architectural statement in `paper/draft.md` and supporting documents against canonical clean artifacts (`artifacts/phase17_clean_bayesian/` and `artifacts/phase16_1_audit/`).

---

## 1. Dataset Counts & Universe Verification

| Split / Universe | Stated in Draft | Canonical Artifact | Verified Count | Status | Required Edit |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Training Observations ($N_{\text{train}}$)** | "trained on 18,201 events" / "11,985" | `clean_text_manifest.csv` (`split == 'train'`) | **11,985** | **NEEDS_REWORDING** | Correct all instances stating models were "trained on 18,201 events". Total universe is 18,201; training set is strictly 11,985. |
| **Validation Observations ($N_{\text{val}}$)** | 2,116 | `clean_text_manifest.csv` (`split == 'validation'`) | **2,116** | **VERIFIED** | Pre-2022 validation universe strictly equals 2,116. |
| **Clean OOS Unique Events ($N_{\text{test, unique}}$)** | 4,100 | `clean_text_manifest.csv` (`split == 'clean_oos'`) | **4,100** | **VERIFIED** | 4,100 unique calendar events across 2022–2025. |
| **Clean OOS Seed Observations ($N_{\text{test, total}}$)** | 12,300 | `clean_oos_predictions.csv` (12,300 rows) | **12,300** | **VERIFIED** | Exactly 4,100 unique events $\times$ 3 seeds (42, 123, 777). |
| **Total Universe Events** | 18,201 | `clean_text_manifest.csv` (total rows) | **18,201** | **VERIFIED** | $11,985 + 2,116 + 4,100 = 18,201$. Clearly label as the total partitioned universe, not the training set size. |

---

## 2. Input Representation & Linguistic Scope Audit

| Component | Draft Statement | Exact Code / Artifact | Status | Required Edit |
| :--- | :--- | :--- | :---: | :--- |
| **Clean Input Text Template** | Implied general "financial text" / "earnings text" | `experiments/run_phase17_clean_bayesian.py:L268`: `f"{company} ({ticker}) quarterly earnings announcement on {date_str}."` | **AMBIGUOUS** | Explicitly qualify that clean experiments evaluated a **minimal metadata-based input representation** consisting of company name, ticker, and announcement date. Disclaim that full earnings call transcripts or 10-Q narrative text were not evaluated. |
| **Target Leakage String** | Concatenated `. 1-day move was ` + `move_1d` | `experiments/run_phase11_replication.py:L132` | **VERIFIED** | Accurately describes historical pipeline bug where label derivation variable was concatenated to inputs. |
| **Forbidden Token Assertions** | Zero outcome tokens in clean manifest | Automated assertions in `test_phase17_clean_bayesian.py` checking `move_1d`, `move_30d`, `1-day move was`, `30-day move was`, `-4.69`, `downside` | **VERIFIED** | Retain verified assertion checks. |

---

## 3. Quantitative & Statistical Performance Audit

| Metric / Result | Draft Location | Canonical Artifact | Verified Value | Status | Required Edit |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Clean Det LoRA AUROC** | Table 1, Sec 6.1 | `forecast_metrics.csv:L2` | **0.51312** | **VERIFIED** | Exact match. |
| **Clean Bayes Mean AUROC** | Table 1, Sec 6.1 | `forecast_metrics.csv:L3` | **0.52531** | **VERIFIED** | Exact match. |
| **Clean Bayes MC32 AUROC** | Table 1, Sec 6.1 | `forecast_metrics.csv:L4` | **0.52504** | **VERIFIED** | Exact match. |
| **Base-Rate Null AUROC** | Table 1, Sec 6.1 | `forecast_metrics.csv:L5` | **0.50000** | **VERIFIED** | Exact match. |
| **Clean Det LoRA Brier** | Table 1, Sec 6.1 | `forecast_metrics.csv:L2` | **0.16079** | **VERIFIED** | Exact match. |
| **Clean Bayes Mean Brier** | Table 1, Sec 6.1 | `forecast_metrics.csv:L3` | **0.15987** | **VERIFIED** | Exact match. |
| **Clean Bayes MC32 Brier** | Table 1, Sec 6.1 | `forecast_metrics.csv:L4` | **0.15987** | **VERIFIED** | Exact match. |
| **Base-Rate Null Brier** | Table 1, Sec 6.1 | `forecast_metrics.csv:L5` | **0.16000** | **VERIFIED** | Exact match. |
| **Clean Det LoRA Log Loss** | Table 1, Sec 6.1 | `forecast_metrics.csv:L2` | **0.50332** | **VERIFIED** | Exact match. |
| **Clean Bayes Mean Log Loss**| Table 1, Sec 6.1 | `forecast_metrics.csv:L3` | **0.50007** | **VERIFIED** | Exact match. |
| **Clean Bayes MC32 Log Loss**| Table 1, Sec 6.1 | `forecast_metrics.csv:L4` | **0.50009** | **VERIFIED** | Exact match. |
| **Base-Rate Null Log Loss** | Table 1, Sec 6.1 | `forecast_metrics.csv:L5` | **0.50040** | **VERIFIED** | Exact match. |
| **Clean Det LoRA ECE** | Table 1, Sec 6.1 | `forecast_metrics.csv:L2` | **0.01785** | **VERIFIED** | Exact match. |
| **Clean Bayes Mean ECE** | Table 1, Sec 6.1 | `forecast_metrics.csv:L3` | **0.00686** | **VERIFIED** | Exact match. |
| **Clean Bayes MC32 ECE** | Table 1, Sec 6.1 | `forecast_metrics.csv:L4` | **0.00695** | **VERIFIED** | Exact match. |
| **Base-Rate Null ECE** | Table 1, Sec 6.1 | `forecast_metrics.csv:L5` | **0.00000** | **VERIFIED** | Exact match. |
| **$\Delta$ Brier (Bayes - Det)** | Abstract, Sec 6.1 | `bootstrap_intervals.csv:L13` | **-0.00092** | **VERIFIED** | $95\%\text{ CI: } [-0.00115, -0.00070]$, within $\pm 0.002$ margin. |
| **$\Delta$ LogLoss (Bayes - Det)**| Abstract, Sec 6.1 | `bootstrap_intervals.csv:L14` | **-0.00324** | **VERIFIED** | $95\%\text{ CI: } [-0.00411, -0.00246]$, within $\pm 0.010$ margin. |
| **$\Delta$ AUROC (Bayes - Det)** | Abstract, Sec 6.1 | `bootstrap_intervals.csv:L12` | **+0.01195** | **VERIFIED** | $95\%\text{ CI: } [+0.00306, +0.02126]$, mean difference $+0.012$. |
| **$\Delta$ Brier (MC32 - Mean)** | Table 3, Sec 7 | `bootstrap_intervals.csv:L17` | **+0.00001** | **VERIFIED** | $95\%\text{ CI: } [0.00000, +0.00002]$, within $\pm 0.002$ margin. |
| **$\Delta$ AUROC (MC32 - Mean)**| Table 3, Sec 7 | `bootstrap_intervals.csv:L16` | **-0.00028** | **VERIFIED** | $95\%\text{ CI: } [-0.00080, +0.00027]$, within $\pm 0.005$ margin. |
| **Inference Latency (Mean)** | Table 3, Sec 7 | `compute_cost.csv:L3` | **7.887 ms** | **VERIFIED** | Forward pass time per event. |
| **Inference Latency (MC32)** | Table 3, Sec 7 | `compute_cost.csv:L4` | **241.015 ms** | **VERIFIED** | 32 forward passes per event. |
| **Latency Ratio** | Abstract, Sec 7 | $241.015 / 7.887$ | **30.56x** | **VERIFIED** | Stated as $\sim 30.6\times$ or roughly $30\times$. |
| **Within-Class $\rho$ ($y=0$)** | Sec 4.2 | `forecast_metrics.csv:L2-4` | **+1.00000** | **VERIFIED** | Mechanically forced identity for $p < 0.5$. |
| **Within-Class $\rho$ ($y=1$)** | Sec 4.2 | `forecast_metrics.csv:L2-4` | **-1.00000** | **VERIFIED** | Mechanically forced identity for $p < 0.5$. |

---

## 4. Methodological Tone & Rhetorical Claim Audit

| Category / Passage | Current Wording | Issue Identified | Required Modification |
| :--- | :--- | :--- | :--- |
| **Inference Equivalence** | "we prove MC32 is identical" / "mathematical identity" | Implies exact bitwise floating-point identity across random samples. | Replace with: *"MC32 was practically equivalent to posterior-mean inference under the preregistered equivalence margins."* |
| **Reliability Correlation** | "the pooled correlation is entirely an artifact" | Overstates the conclusion beyond the analytical derivation. | Replace with: *"The apparent pooled reliability relationship is largely mechanically induced by the shared algebraic dependence of both quantities on $p$ and by class composition."* |
| **Promotional / Superlatives**| "remarkable", "compelling", "decisive", "catastrophic collapse" | Non-neutral, sensational tone. | Replace with neutral scientific phrasing: "apparent initial signal", "rapid test-time degradation", "near-chance discrimination". |
| **Generalization Scope** | "Bayesian LoRA provides no value in financial NLP" | Overgeneralizes across all financial NLP tasks and architectures. | Strictly scope to: *"Under this earnings-event downside prediction setup, dataset, label thresholding, FinBERT/LoRA architecture, and metadata-based input representation..."* |
| **Portfolio Section Framing** | Headlined as primary finding with detailed return numbers. | Confuses the primary contribution hierarchy; clean point forecast is inherently weak. | Demote portfolio section to secondary empirical sanity check. Emphasize that portfolio adjustments cannot compensate for weak underlying point forecasts. |
| **Base-Rate Baseline** | "Machine learning cannot beat the base rate" | Broad overclaim. | State neutrally: *"A constant prevalence predictor ($p \equiv 0.20$) was highly competitive under proper scoring rules, matching or slightly exceeding fine-tuned models on metadata inputs."* |
