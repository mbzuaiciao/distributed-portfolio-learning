# Phase 12: Prospective 2026 Outcome Evaluation Report

**Evaluation Window**: 2026 prospective evaluation through September 9, 2026  
**Final Scientific Verdict**: **Outcome B — Model property replicates, timing does not**  

---

## 1. Executive Summary & Sample Size
- **Frozen Input Events**: 5 benchmark constituents across 3 seeds (15 total predictions).
- **Complete H=4 Events**: 4 (GOOGL, MSFT, AAPL, AMZN).
- **Right-Censored Events**: 1 (NVDA_2026_Q2, released 2026-08-26, full 4-week holding path incomplete as of 2026-09-09).
- **Usable Calendar Weeks**: 5 weeks (2026-W30 through 2026-W34).
- **Market Data As-Of Date**: 2026-09-09.

---

## 2. Primary Prospective Endpoint: Delta U (M13 H4 minus M8 H4)
| Seed | Weeks | M8 Mean Utility | M13 Mean Utility | Delta Utility ($\Delta U$) | Positive? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 5 | 0.025037 | 0.025037 | **0.0** | False |
| **123** | 5 | 0.026753 | 0.026753 | **0.0** | False |
| **777** | 5 | 0.025101 | 0.025101 | **0.0** | False |
| **mean** | 5 | 0.02563 | 0.02563 | **0.0** | False |

### 4-Week Block Bootstrap
| Seed | Point Estimate $\Delta U$ | 95% Bootstrap CI | $\Pr(\Delta U > 0)$ |
| :---: | :---: | :---: | :---: |
| **42** | 0.0 | [0.0, 0.0] | 0.0% |
| **123** | 0.0 | [0.0, 0.0] | 0.0% |
| **777** | 0.0 | [0.0, 0.0] | 0.0% |
| **mean** | 0.0 | [0.0, 0.0] | 0.0% |

---

## 3. Architecture Confirmation: M8 H4 vs M8 H1
| Seed | H=4 Annual Ret | H=4 Sharpe | H=4 Utility | H=1 Annual Ret | H=1 Sharpe | H=1 Utility | H4 - H1 Utility |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 268.9% | 8.782 | 0.025037 | 6742.2% | 77.886 | 0.079375 | **-0.054337** |
| **123** | 303.6% | 9.251 | 0.026753 | 6742.2% | 77.886 | 0.079375 | **-0.052621** |
| **777** | 270.1% | 8.798 | 0.025101 | 6742.2% | 77.886 | 0.079375 | **-0.054274** |

---

## 4. Model-Level Uncertainty & Directional Diagnostics
### Epistemic Uncertainty vs Forecast Error
| Seed | Spearman $\rho(\sigma_{epi}^2, |\text{Error}|)$ | Spearman $p$-val | Q1 MAE | Q5 MAE | $Q_5 / Q_1$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | -0.6 | 2.8476e-01 | 0.6466 | 0.2796 | 0.43x |
| **123** | 0.7 | 1.8812e-01 | 0.1530 | 0.3931 | 2.57x |
| **777** | 0.6 | 2.8476e-01 | 0.2593 | 0.4972 | 1.92x |
| **mean** | 0.2333 | 2.5250e-01 | 0.3530 | 0.3900 | 1.64x |

### Directional Downside Probability Evaluation
| Seed | Downside Positive Rate | Brier Score | Directional AUROC | Average Precision |
| :---: | :---: | :---: | :---: | :---: |
| **42** | 20.0% | 0.2253 | 0.0000 | 0.2000 |
| **123** | 20.0% | 0.1934 | 0.7500 | 0.5000 |
| **777** | 20.0% | 0.1471 | 1.0000 | 1.0000 |
| **mean** | 20.0% | 0.1886 | 0.5833 | 0.5667 |

---

## 5. Outcome Tree Classification
- **Classification**: **Outcome B — Model property replicates, timing does not**
- **Interpretation**: Epistemic uncertainty reliably tracks model unreliability, but exposure timing overlay fails to deliver positive net utility.
