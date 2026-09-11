# Phase 13A: Historical Data Scope & Development/Evaluation Split Report

**Date:** 2026-09-11  
**Repository:** `distributed-portfolio-learning`  
**Git Commit SHA:** `c688ab398a4ece311fe53b51e7456d2874f0402d`  
**Status:** Canonical & Profiled  
**Primary Artifacts:**
- Inventory: [`artifacts/phase13_uncertainty_mapping/historical_data_inventory.csv`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/artifacts/phase13_uncertainty_mapping/historical_data_inventory.csv)
- Data Scope: [`artifacts/phase13_uncertainty_mapping/data_scope.json`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/artifacts/phase13_uncertainty_mapping/data_scope.json)
- Compute Decision: [`artifacts/phase13_uncertainty_mapping/compute_decision.json`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/artifacts/phase13_uncertainty_mapping/compute_decision.json)

---

## 1. Purpose & Scope Boundary

In accordance with the Phase 13 research protocol, **Phase 13A strictly focuses on data provenance, temporal regime segmentation, and compute sizing prior to any strategy evaluation**.

> **Scientific Boundary:**
> - **NO** strategy evaluations ($M_{14}, M_{15}, M_{16}$) are implemented or scored in this phase.
> - **NO** comparison of uncertainty-to-allocation mappings is performed.
> - The sole purpose is establishing the exact historical universe where genuine Bayesian-LoRA inference is supported and defining the non-overlapping development/evaluation split.

---

## 2. Historical Data Inventory

The underlying historical corpus [`data/bayesian_lora/reconstructed_returns.csv`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/data/bayesian_lora/reconstructed_returns.csv) was exhaustively inventoried across all 20,000 corporate earnings events spanning nearly two decades (**2005-10-13 to 2025-05-15**).

### Model-Inference Coverage Taxonomy:
Every event is categorized into one of five mutually exclusive classes:
- **Category A (Actual Phase-11 Checkpoint Inference Possible):** Events with valid ticker, announcement date, model text prompt, observed 1-day return ($move\_1d$), and complete 4-week holding path ($move\_30d$ and $d\_plus\_30\_close$).
- **Category B (Archived Old-Model Prediction Only):** Events that rely exclusively on legacy Stage-2 LoRA predictions where genuine Phase-11 checkpoint inference is infeasible. (In our clean dataset, all legacy dev events map 1-to-1 with the return database, so actual Phase-11 inference is possible for all of them; no events are restricted to Category B).
- **Category C (Missing Usable Text):** Events with null ticker or unparseable text format ($N=0$).
- **Category D (Missing Return Data):** Events where stock price history was missing in FNSPID, resulting in null $move\_1d$ ($N=1,799$ overall; $N=557$ in 2018–2025).
- **Category E (Incomplete $H=4$ Path):** Events where $move\_1d$ is observed but subsequent holding path is truncated ($N=0$ in the audited historical database).

### Full Corpus Coverage Summary (2005–2025):
| Coverage Category | Description | Count ($N$) | Percentage |
| :--- | :--- | :---: | :---: |
| **Category A** | Actual Phase-11 Checkpoint Inference Possible | **18,201** | 91.01% |
| **Category B** | Archived Old-Model Prediction Only | **0** | 0.00% |
| **Category C** | Missing Usable Text | **0** | 0.00% |
| **Category D** | Missing Return Data ($move\_1d$ NaN) | **1,799** | 8.99% |
| **Category E** | Incomplete $H=4$ Path | **0** | 0.00% |
| **Total** | Full Historical Corpus | **20,000** | 100.00% |

---

## 3. Temporal Regimes Definition (2018–2025 Primary Scope)

Pre-2018 data (2005–2017, 10,751 rows, 9,509 valid) is characterized by sparse cross-sectional breadth in early years and less standardized reporting times. We therefore focus the primary experimental universe on **2018–2025** ($N = 8,692$ valid events across 9,249 rows).

Every preferred regime is genuinely supported by abundant data, containing between **1,180 and 2,203 valid events**. Because no regime has fewer than 1,000 events, **no adjacent periods require mechanical merging**.

### Canonical Temporal Regimes:

| Regime ID | Economic Characterization | Temporal Bounds | Total Rows | Valid Events ($N$) | Missing Returns | Split Role |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`2018-2019`** | Pre-COVID Macro Baseline | 2018-01-04 to 2019-12-20 | 2,420 | **2,203** | 217 | **Development** |
| **`2020`** | COVID Shock & Unprecedented Liquidity Rebound | 2020-01-03 to 2020-12-22 | 1,303 | **1,198** | 105 | **Evaluation** |
| **`2021`** | Stimulus Boom & Low-Vol Tech Expansion | 2021-01-07 to 2021-12-22 | 1,278 | **1,191** | 87 | **Evaluation** |
| **`2022`** | Global Inflation Shock & Rapid Rate Hikes | 2022-01-06 to 2022-12-22 | 1,253 | **1,191** | 62 | **Evaluation** |
| **`2023`** | SVB Banking Stress & GenAI Market Breakout | 2023-01-05 to 2023-12-21 | 1,230 | **1,181** | 49 | **Evaluation** |
| **`2024-2025`** | Canonical Benchmark Holdout (Phase 11.3) | 2024-01-04 to 2025-05-15 | 1,765 | **1,728** | 37 | **Evaluation** |
| **Total** | **Primary Experimental Scope (2018–2025)** | **2018-01-04 to 2025-05-15** | **9,249** | **8,692** | **557** | **Dev + Eval** |

---

## 4. Development / Evaluation Split

To eliminate lookahead bias and parameter snooping:
1. **Development Split (`2018-2019`):**
   - **Sample Size:** $N = 2,203$ valid corporate earnings events.
   - **Date Range:** 2018-01-04 10:00:00 to 2019-12-20 10:00:00.
   - **Usage Rule:** Strictly reserved for parameter selection, threshold calibration ($\tau_{\text{enter}}, \tau_{\text{exit}}$), exposure scaling ($\lambda_{\text{rel}}$), and uncertainty-to-allocation mapping design.
2. **Evaluation Split (`2020-2025`):**
   - **Sample Size:** $N = 6,489$ valid corporate earnings events across 5 distinct macroeconomic regimes.
   - **Date Range:** 2020-01-03 10:00:00 to 2025-05-15 16:30:00.
   - **Usage Rule:** Evaluated out-of-sample with zero parameter adaptation or regime-specific tuning.
3. **Strict Causality Invariant:**
   $$\max(\text{Development Date}) = \text{2019-12-20} < \text{2020-01-03} = \min(\text{Evaluation Date})$$
   There is **zero overlap** between development and evaluation regimes.

---

## 5. Actual Inference Scale & Compute Decision

Using measured empirical throughput on local Apple Silicon (M2 MPS, $4.98 \text{ events/sec}$ on 32-draw Monte Carlo inference):

### Workload Scale:
- **Total Valid Events:** $8,692$
- **Seeds:** $3$ (Seeds 42, 123, 777)
- **Monte Carlo Draws ($S$):** $32$
- **Total Event-Inferences:** $8,692 \times 3 = \mathbf{26,076}$
- **Total Variational Forward Passes:** $26,076 \times 32 = \mathbf{834,432}$
- **Already Computed in Phase 11.3 (2024–2025):** $1,728 \times 3 = \mathbf{5,184}$ event-inferences ($165,888$ passes).
- **Remaining Events to Infer (2018–2023):** $6,964 \times 3 = \mathbf{20,892}$ event-inferences ($668,544$ passes).

### Local Runtime Projections:
- **Full Job (2018–2025):** $\frac{26,076}{4.98} \approx 5,236.1 \text{ s} \approx \mathbf{87.3} \text{ minutes}$.
- **Remaining Job (2018–2023):** $\frac{20,892}{4.98} \approx 4,195.2 \text{ s} \approx \mathbf{69.9} \text{ minutes}$.

### Compute Decision Rule Adjudication:
| Runtime Window | Protocol Action | Applicable? |
| :--- | :--- | :---: |
| $< 30$ minutes | Run locally | No |
| **30–120 minutes** | **Still prefer local unless inconvenient** | **YES (87.3 min / 69.9 min)** |
| $> 120$ minutes | Benchmark L4 cloud execution | No |

### Official Decision: `prefer_local`
- **Verdict:** Run the complete Phase 13 historical inference locally on Apple Silicon M2 MPS.
- **Rationale:** The entire job requires ~70–87 minutes, falling strictly within the 30–120 minute window where local execution is preferred. Running locally eliminates Google Cloud VM provisioning latency, network transfer overhead, SSH key handling, and unnecessary billing.
- **Cloud Infrastructure Status:** Preemption-safe sharding (`experiments/cloud/sharded_inference.py`) on NVIDIA L4 Spot (~8 minutes, \$0.04) remains fully validated and available in reserve.
