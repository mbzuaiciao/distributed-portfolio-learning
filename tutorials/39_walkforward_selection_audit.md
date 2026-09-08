# Tutorial 39: Auditing Walk-Forward Model Selection (M13 vs M17)

A critical requirement of rigorous scientific auditing is resolving discrepancies between static backtests and out-of-sample walk-forward results. In Phase 8, static full-holdout benchmarks showed that **$M_{13}$ Hysteresis** was the standalone best policy, while the walk-forward validation table reported **$M_{17}$ Combined** as the selected candidate across all three expanding folds.

Phase 9 audits this discrepancy end-to-end to determine whether the divergence stemmed from genuine market regime changes or an implementation artifact.

---

## 1. The Five Audit Questions

### Question 1: Was $M_{17}$ genuinely selected using training data only?
**Finding**: No. In the Phase 8 walk-forward experiment script (`experiments/run_phase8_cost_aware_timing.py`), $M_{17}$ was hardcoded as one of four fixed evaluation models. The training fold only optimized the band width $\delta \in \{0.05, 0.10, 0.15\}$ for the band component; the composite model structure $M_{17}$ was never selected over a broader set of competitive candidates on training data.

### Question 2: Were its parameters different from the static $M_{17}$ parameters?
**Finding**: **Yes, substantially.**
- **Static $M_{17}$**: Used $\tau_{\text{enter}} = 1.2$, $\tau_{\text{exit}} = 0.2$, $\delta = 0.20$, and partial adjustment $\eta = 0.50$.
- **Walk-Forward $M_{17}$**: Used $\tau_{\text{enter}} = 1.0$, $\tau_{\text{exit}} = 0.2$, $\delta \in \{0.05, 0.10, 0.15\}$, and **full adjustment $\eta = 1.0$**.

### Question 3: Was $M_{13}$ included in the candidate set?
**Finding**: **No.** In `run_phase8_cost_aware_timing.py`, the walk-forward candidate loop was:
```python
for pname in ["M8_Constant", "M9_RawTiming", "M15_Band", "M17_Combined"]:
    ...
```
$M_{13}$ was inadvertently omitted from the walk-forward evaluation loop. Thus, $M_{17}$ appeared as the top performer simply because its primary competitor ($M_{13}$) was not present in the comparison.

### Question 4: Why did $M_{17}$ outperform in walk-forward but lag in static evaluation?
**Finding**: In the static full-holdout test, $M_{17}$ was burdened by a wide dead-band ($\delta = 0.20$) and sluggish $50\%$ partial adjustment ($\eta = 0.50$). During the sharp market recovery following August 2024, static $M_{17}$ took over three weeks to re-enter the market, severely curtailing its annual return to $29.99\%$ (vs $30.66\%$ for constant $M_8$).

In contrast, the walk-forward script specified $\eta = 1.0$ (immediate re-entry upon signal clearance) and selected tight dead-bands ($\delta \le 0.15$). This allowed walk-forward $M_{17}$ to behave almost identically to pure hysteresis without the sluggish recovery penalty!

### Question 5: What is the exact root cause of the discrepancy?
**Finding**: The discrepancy was caused by **parameter asymmetry** ($\eta = 1.0$ vs $\eta = 0.50$, $\delta \le 0.15$ vs $0.20$) combined with **candidate omission** ($M_{13}$ excluded from the loop).

---

## 2. Walk-Forward Selection Audit Table

The table below documents the exact configuration across all three walk-forward folds from `artifacts/phase9_audit/walkforward_selection_audit.csv`:

| Fold | Training Window | Evaluation Window | Evaluated Candidates | Tuned Parameters | Objective | Discrepancy Diagnosis |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fold 1** | 2022-W01 to 2024-W01 | 2024-W01 to 2024-W26 | $M_8, M_9, M_{15}, M_{17}$ | $\delta \in \{0.05, 0.10, 0.15\}$ | Trailing Net Utility | $M_{13}$ omitted; $M_{17}$ used $\eta=1.0$ |
| **Fold 2** | 2022-W01 to 2024-W28 | 2024-W28 to 2024-W49 | $M_8, M_9, M_{15}, M_{17}$ | $\delta \in \{0.05, 0.10, 0.15\}$ | Trailing Net Utility | $M_{13}$ omitted; $M_{17}$ used $\eta=1.0$ |
| **Fold 3** | 2022-W01 to 2024-W50 | 2024-W50 to 2025-W20 | $M_8, M_9, M_{15}, M_{17}$ | $\delta \in \{0.05, 0.10, 0.15\}$ | Trailing Net Utility | $M_{13}$ omitted; $M_{17}$ used $\eta=1.0$ |

---

## 3. Methodological Lesson

This audit reinforces the principle of **strict configuration hygiene** in quantitative research:
1. Every candidate policy evaluated in static holdouts must be evaluated in walk-forward pipelines using the exact same parameter definitions.
2. In walk-forward testing, model selection must choose between the candidate policies dynamically rather than reporting hardcoded configurations.
3. The finding that pure hysteresis ($M_{13}$) is optimal remains robust: adding complexity (dead-bands + partial adjustments) only introduces parameter fragility.
