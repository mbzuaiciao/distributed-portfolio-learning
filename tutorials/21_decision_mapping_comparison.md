# Tutorial 21: Phase 4 Comparative Benchmark & Scientific Decision Tree

Phase 4 set out to resolve a fundamental question:
> *Can Bayesian-LoRA epistemic uncertainty be converted into out-of-sample portfolio value using decision-relevant risk mappings rather than naive symmetric mean shrinkage?*

We evaluated five policies across **three random initializations** (`Seed 42`, `Seed 123`, `Seed 777`) on the **2024–2025 frozen holdout** ($T = 432$ trading days, 1,728 asset-day events):

- **M0 — Deterministic**: Baseline using point forecast $\hat{\mu}$ and market covariance $Q$.
- **M1 — Symmetric Shrinkage**: Phase 3 benchmark $\tilde{\mu} = \rho(\sigma_{\text{epi}}) \hat{\mu}$.
- **M2 — Covariance Inflation**: Directional alpha intact, $Q_{\text{robust}} = Q + \lambda_{\text{cov}}^* \operatorname{diag}(\sigma_{\text{epi}}^2)$.
- **M3 — Downside-Adjusted Alpha**: $\tilde{\mu} = \hat{\mu} \cdot \max(0, 1 - \lambda_{\text{down}}^* p_{\text{down}})$.
- **M4 — Downside Covariance Inflation**: $Q_{\text{down}} = Q + \lambda_{\text{down}}^* \operatorname{diag}(p_{\text{down}} \cdot \sigma_{\text{epi}}^2)$.

All hyperparameters were tuned strictly on 2022–2023 development data and frozen prior to holdout evaluation.

---

## 1. Aggregate Holdout Performance (3 Seeds Mean $\pm$ Std)

| Policy | Annualized Return | Annualized Volatility | Sharpe Ratio | Sortino Ratio | Max Drawdown | Mean Net Utility | Bad/Good Weight Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0: Deterministic** | $\mathbf{43.80\% \pm 6.59\%}$ | $28.87\% \pm 1.12\%$ | $\mathbf{1.519 \pm 0.227}$ | $\mathbf{2.894 \pm 0.488}$ | $-15.12\%$ | $\mathbf{-0.00621 \pm 0.00140}$ | $\mathbf{0.651 \pm 0.056}$ |
| **M1: Symmetric Shrink** | $35.73\% \pm 8.72\%$ | $\mathbf{27.45\% \pm 0.66\%}$ | $1.301 \pm 0.310$ | $2.473 \pm 0.580$ | $-16.51\%$ | $-0.00659 \pm 0.00167$ | $0.680 \pm 0.056$ |
| **M2: Covariance Inflation**| $41.99\% \pm 6.31\%$ | $28.50\% \pm 1.07\%$ | $1.475 \pm 0.219$ | $2.831 \pm 0.461$ | $\mathbf{-14.88\%}$ | $-0.00636 \pm 0.00130$ | $0.658 \pm 0.056$ |
| **M3: Downside Alpha** | $27.53\% \pm 4.51\%$ | $28.19\% \pm 1.20\%$ | $0.974 \pm 0.131$ | $1.648 \pm 0.257$ | $-22.75\%$ | $-0.00791 \pm 0.00086$ | $0.709 \pm 0.034$ |
| **M4: Downside Cov** | $43.80\% \pm 6.59\%$ | $28.87\% \pm 1.12\%$ | $1.518 \pm 0.227$ | $2.893 \pm 0.488$ | $-15.12\%$ | $-0.00621 \pm 0.00140$ | $0.651 \pm 0.056$ |

---

## 2. Paired Circular Block Bootstrap Statistics (Part I)

To account for temporal autocorrelation, we conducted paired circular block bootstrap evaluations ($B = 4$ weeks, 1,000 resamples) evaluating $\Delta U = U_{\text{policy}} - U_{\text{M0}}$:

| Seed | Policy | $\Delta \text{Utility}$ Point | 95% Bootstrap CI | $t$-statistic | Empirical $p$-value | Sign Consistent? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **42** | M1 Symmetric Shrink | $-0.000773$ | $[-0.004280, +0.002082]$ | $-0.463$ | $0.678$ | No |
| | M2 Covariance Inflation | $-0.000011$ | $[-0.000372, +0.000374]$ | $-0.057$ | $0.968$ | No |
| | M3 Downside Alpha | $-0.000942$ | $[-0.006121, +0.003309]$ | $-0.383$ | $0.746$ | Negative |
| | M4 Downside Cov | $-0.000000$ | $[-0.000000, +0.000000]$ | $-0.345$ | $0.742$ | Neutral |
| **123**| M1 Symmetric Shrink | $+0.000235$ | $[-0.002398, +0.002602]$ | $+0.183$ | $0.816$ | No |
| | M2 Covariance Inflation | $-0.000142$ | $[-0.000795, +0.000318]$ | $-0.499$ | $0.704$ | No |
| | M3 Downside Alpha | $-0.001844$ | $[-0.006222, +0.001612]$ | $-0.888$ | $0.414$ | Negative |
| | M4 Downside Cov | $+0.000000$ | $[-0.000000, +0.000000]$ | $+0.224$ | $0.826$ | Neutral |
| **777**| M1 Symmetric Shrink | $-0.000600$ | $[-0.003211, +0.001322]$ | $-0.522$ | $0.602$ | No |
| | M2 Covariance Inflation | $-0.000302$ | $[-0.001011, +0.000158]$ | $-0.940$ | $0.482$ | No |
| | M3 Downside Alpha | $-0.002303$ | $[-0.006330, +0.001015]$ | $-1.195$ | $0.214$ | Negative |
| | M4 Downside Cov | $-0.000001$ | $[-0.000004, +0.000000]$ | $-1.005$ | $0.420$ | Neutral |

### Statistical Verdict
1. **None of the uncertainty-aware policies deliver a statistically significant improvement in net utility over the deterministic baseline.**
2. In all seeds, the 95% circular block bootstrap confidence intervals span zero for M1, M2, and M4.
3. M3 produces consistent negative point estimates in all three seeds.

---

## 3. Selectivity Analysis (Part E)

We classified holdout asset-day forecasts ex-post into Best 20%, Middle 60%, and Worst 20% realized return outcomes. We then measured the average capital pre-allocated to each bucket before outcomes were known:

| Policy | Avg Weight on Best 20% | Avg Weight on Worst 20% | Bad/Good Capital Ratio | Relative to M0 |
| :--- | :---: | :---: | :---: | :---: |
| **M0: Deterministic** | $\mathbf{5.27\%}$ | $\mathbf{3.43\%}$ | $\mathbf{0.651}$ | Baseline |
| **M1: Symmetric Shrink** | $5.21\%$ | $3.53\%$ | $0.680$ | $+4.5\%$ Worsened |
| **M2: Covariance Inflation**| $5.26\%$ | $3.45\%$ | $0.658$ | $+1.1\%$ Worsened |
| **M3: Downside Alpha** | $5.13\%$ | $3.63\%$ | $0.709$ | $+8.9\%$ Worsened |
| **M4: Downside Cov** | $5.27\%$ | $3.43\%$ | $0.651$ | Neutral |

> **Selectivity Takeaway**:  
> Not a single uncertainty-aware policy reduced capital on subsequently bad forecasts without equally or more heavily reducing capital on good forecasts. The deterministic baseline achieved the lowest Bad/Good ratio.

---

## 4. Scientific Decision Tree: Outcome Classification

At the outset of Phase 4, we pre-declared four mutually exclusive scientific outcomes:

```
                      ┌──────────────────────────────────────┐
                      │    Phase 4 Empirical Evaluation      │
                      └──────────────────┬───────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
         Does Covariance                                 Does Downside
         Inflation Work?                                 Probability Work?
                 │                                               │
        ┌────────┴────────┐                             ┌────────┴────────┐
        ▼                 ▼                             ▼                 ▼
     YES [A]           NO [D1]                       YES [B]           NO [D2]
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │       OUTCOME D: NEITHER WORKS       │
                      └──────────────────────────────────────┘
```

### Classification: OUTCOME D (Clean Negative Result)
- **Outcome A (Covariance Inflation works)**: Rejected. $Q_{\text{robust}}$ slightly reduced volatility ($-0.37\%$) and drawdown ($-14.88\%$ vs $-15.12\%$), but return dropped proportionally; paired $\Delta U$ 95% CIs span zero across all seeds.
- **Outcome B (Downside Probability works)**: Rejected. Downside-adjusted alpha softened negative forecasts on high-uncertainty assets, degrading return from $43.8\%$ to $27.5\%$ and increasing drawdown to $-22.75\%$.
- **Outcome C (Both work)**: Rejected.
- **Outcome D (Neither works)**: **CONFIRMED**.

### The Scientific Core
The Bayesian-LoRA uncertainty signal is **informative** about model error magnitude ($\rho_s = +0.55$, $\text{AUROC} = 0.82$), but under standard long-only mean-variance portfolio optimization, that information **cannot be converted into incremental economic value** using either symmetric mean shrinkage, covariance inflation, or heuristic downside alpha adjustments.

The optimizer is already an efficient information aggregator: its constrained quadratic objective naturally downweights assets with lower point alphas and relies on asset correlation diversification. Adding scalar penalties based on undirected epistemic dispersion dilutes the point signal without providing directional risk filtering.

---

## 5. Historical Architectural Proposals for Phase 5

Because centralized uncertainty decision mappings have not demonstrated economic improvement over deterministic point forecasts, **modifying the distributed consensus coordination mechanism (e.g., confidence-weighted dual penalties) is currently scientifically unjustified**.

At this stage of the historical research sequence, the following directions were
proposed to investigate the gap between uncertainty and directional alpha. They are
retained as historical proposals, not an active roadmap or evidence that combining
LoRA specialists improves prediction:
1. **Asymmetric Training Objectives**: Train the neural network to output directional downside quantiles (e.g., conditional Value-at-Risk) rather than undirected epistemic variance.
2. **Dynamic Leverage / Cash Buffering**: Use epistemic uncertainty not to rebalance between assets, but to scale aggregate portfolio beta (moving into cash when average universe uncertainty is high).
3. **Execution Cost Modeling**: Test whether high-uncertainty assets have higher transaction/slippage costs, where uncertainty acts as an execution friction penalty.


Later audits weakened the early reliability interpretation and must be considered when
reading the conclusions above. Independently, the
[collaborative-LoRA pilot failed its frozen continuation criteria](../docs/collaborative_lora_branch_closure.md):
learned weights collapsed to one specialist, CoLoRA was approximately tied with
pooled/global training, and the prevalence null was strongest on Brier/log loss.
No collaborative effect was established, so Bayesian/UQ and portfolio-style specialist
weighting were not escalated. Financial exposure or execution proposals are distinct
from neural specialist complementarity. That branch remains closed unless a materially
different task regime or data-generating setup demonstrates exploitable complementarity;
the pilot does not establish that collaboration is universally impossible.
