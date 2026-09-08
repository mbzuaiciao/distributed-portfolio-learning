# Tutorial 30: Nested Walk-Forward Validation & Non-Anticipation

A critical vulnerability in algorithmic trading research is **static split evaluation**. Even if an evaluation dataset is separated into "train" and "test" sets, researchers frequently:
1. Select hyperparameters on the test set after looking at the outcome;
2. Normalize features using test-set summary statistics (e.g., mean and standard deviation across the full test period);
3. Assume model performance remains uniform across different market regimes.

In Phase 7, we replaced static holdout evaluation with a strictly non-anticipating **nested walk-forward cross-validation protocol**.

---

## 1. The Nested Walk-Forward Architecture

The complete dataset spans 168 weekly decision epochs across 2022 to 2025:
- **Development Period (99 weeks)**: 2022-W01 to 2023-W51 (Fed rate hiking cycle, 2022 bear market, early 2023 rebound)
- **Holdout Period (69 weeks)**: 2024-W01 to 2025-W20 (AI infrastructure rally, summer 2024 tech correction, 2025 early volatility)

We partition the timeline into three expanding walk-forward folds:

```
Timeline (Weeks 0 to 168):
[-------------- Development (99w) --------------][------- 2024 H1 (26w) -------][------- 2024 H2 (22w) -------][--- 2025 (21w) ---]

Fold 1:
Train: [ Weeks 0 .. 99 (Dev) ] ------------------> Tune (W*, lambda*)
Test:                                            [ Weeks 99 .. 125 (2024 H1) ] ---> Evaluate M9 vs M8 Out-of-Sample

Fold 2:
Train: [ Weeks 0 .. 125 (Dev + 2024 H1) ] -------------------------------------> Tune (W*, lambda*)
Test:                                                                          [ Weeks 125 .. 147 (2024 H2) ] ---> Evaluate

Fold 3:
Train: [ Weeks 0 .. 147 (Dev + 2024) ] ------------------------------------------------------------------------> Tune (W*, lambda*)
Test:                                                                                                          [ Weeks 147 .. 168 (2025) ]
```

### Strict Non-Anticipation Rules
1. **Zero Future Information in Tuning**:
   For Fold $k$, hyperparameters $(W^*, \lambda^*)$ are selected from the grid $W \in [8, 13]$ and $\lambda_{\text{rel}} \in [0.15, 0.20, 0.25]$ strictly to maximize Sharpe ratio on the prior training period.
2. **Causal History for Normalization**:
   At test week $t$, the trailing window $W^*$ draws observations from $[t - W^*, t-1]$. Week $t=0$ of Holdout uses the final weeks of Development; no synthetic burn-in or test-period leakage occurs.
3. **Strictly Matched Out-of-Sample Exposure**:
   On each test block, the benchmark constant exposure is set to:
   $$s_{M8} = \frac{1}{|T_{\text{test}}|} \sum_{t \in T_{\text{test}}} s_{M9, t}$$
   ensuring that any Sharpe difference reflects purely the *timing of allocations* within that test period, not an unconditional risk differential.

---

## 2. Empirical Fold Results (Across 3 Seeds)

| Fold | Calendar Period | Test Weeks | Seed 42 $\Delta \text{Sharpe}$ | Seed 123 $\Delta \text{Sharpe}$ | Seed 777 $\Delta \text{Sharpe}$ | Fold Mean $\Delta \text{Sharpe}$ | Fold Mean $\Delta \text{Utility}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 2024 H1 (Jan–Jun) | 26 | $-0.4791$ | $+0.0387$ | $+0.1626$ | **$-0.0926$** | $-0.000223$ |
| **Fold 2** | 2024 H2 (Jul–Dec) | 22 | $+0.2310$ | $+0.3253$ | $-0.2325$ | **$+0.1079$** | $+0.000413$ |
| **Fold 3** | 2025 H1 (Jan–May) | 21 | $+0.7140$ | $-0.0477$ | $+0.1997$ | **$+0.2887$** | $+0.000958$ |
| **Overall**| **2024–2025 Full**| **69**| $+0.1553$ | $+0.1054$ | $+0.0433$ | **$+0.1013$** | $+0.000383$ |

---

## 3. Scientific Analysis: Why Did Fold 1 Struggle?

The nested walk-forward test exposes a vital behavioral feature of epistemic timing:

### The "Melt-Up" Drag in Fold 1 (2024 H1)
In the first half of 2024, mega-cap semiconductor and tech equities experienced a persistent, low-volatility upward rally driven by generative AI demand. During this period:
- Epistemic uncertainty occasionally spiked due to novel valuation multiples and divergence in LoRA weight adapters.
- Strategy $M_9$ dialed back exposure from $0.85$ to $0.65$.
- Because the market continued to rally without experiencing an immediate correction, de-risking sacrificed profitable upside.
- For Seed 42, this caused an out-of-sample Sharpe drag of $-0.479$.

### The Capital Protection Edge in Folds 2 and 3
In contrast, late 2024 and early 2025 experienced recurrent market corrections (the July/August 2024 tech pullback, rate volatility, and earnings revisions):
- Epistemic uncertainty spiked prior to or concurrent with sharp market drawdowns.
- De-risking protected capital effectively, producing positive $\Delta \text{Sharpe}$ up to $+0.714$ in Fold 3.

```
Regime Type:
- Sustained Low-Vol Bull Run (Fold 1): Relative Timing underperforms Constant Exposure (-0.09)
- Volatile / Corrective Market (Folds 2 & 3): Relative Timing strongly outperforms Constant (+0.11 to +0.29)
- Full Cycle Average: Positive net gain (+0.101 Sharpe)
```

## 4. Takeaways for System Design
Walk-forward testing confirms that relative epistemic timing is not an all-weather free lunch. It is an **asymmetric insurance policy**: it incurs small carry costs (sacrificed upside) during uncorrected bull runs, and recoups large benefits during choppy, uncertain corrections.
