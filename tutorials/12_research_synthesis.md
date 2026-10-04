# Tutorial 12 — Final Research Synthesis

This synthesis began as the final tutorial of the original mathematical laboratory. It brings together the mathematical derivations, empirical reproductions, critical audits, and research extensions developed across this entire laboratory.

We summarize the answers to the six central research questions that framed our investigation of Papastaikoudis, Watson, and Lestas (2025).

Current research status: later Bayesian-LoRA audits weakened the early uncertainty
hypotheses, and the separate collaborative-LoRA pilot failed its frozen continuation
criteria. Its branch is closed; see the
[scientific closure](../docs/collaborative_lora_branch_closure.md). The mathematical
formulations below do not supply evidence of specialist complementarity.

---

## 1. Epistemic Status & Attribution

> [!NOTE]
> **Summary of Paper Claims**:
> - The paper claims that overlapping sub-portfolios can be modeled as a hypergraph whose normalized Laplacian $L = I - E(E^T E)^{-1}E^T$ enforces consensus.
> - It claims that a continuous-time primal-dual dynamical system provably converges to global equilibrium.
> - It claims that dual variables $v^*$ act as endogenous shadow prices that price structured financial products.
> - It claims that the decentralized formulation is informationally equivalent to Markowitz, yet achieves substantially lower risk (0.8573 vs 2.8408).

> [!IMPORTANT]
> **Summary of Our Critical Audit Findings**:
> - The continuous-time dynamical system and hypergraph projection algebra are **fully verified and correct**.
> - The claim of "informational equivalence" while achieving "substantially lower risk" is **mathematically invalid**. The reported objective drop is an artifact of downscaled sub-portfolio weights ($1/d_j^2$) and omitted cross-subportfolio covariances.
> - When evaluated on true market covariance $Q$, the distributed portfolio has slightly **higher** risk than the centralized Markowitz baseline (2.14 vs 2.04).
> - However, the distributed allocation achieves **far superior real-world diversification**, avoiding the extreme corner solutions of centralized Markowitz.

---

## 2. Synthesis: The Six Core Research Questions

### Question 1: What is the actual contribution of the paper?
The paper's real contribution is **not** a mathematical discovery of free diversification.
Its genuine contribution is:
1. **Interaction Geometry**: Formulating overlapping investment desks as a hypergraph, enabling multi-party asset sharing beyond pairwise graphs.
2. **Dynamical Consensus Solver**: Deriving a continuous-time primal-dual dynamical system with rigorous Lyapunov stability and LaSalle invariance proofs.
3. **Endogenous Valuation Architecture**: Establishing an explicit mathematical bridge from constraint multipliers ($v^*$) to decentralized shadow prices for multi-theme financial products.

### Question 2: What does the hypergraph buy us?
In traditional optimization, consensus is either bilateral (requiring pairwise graph edges) or centralized (requiring a single coordinator).
The hypergraph provides:
- A compact incidence matrix $E \in \mathbb{R}^{p \times |A|}$ where $|E| = |A|$.
- An exact orthogonal projector $P_E = E(E^T E)^{-1} E^T$ that computes the group average of each asset in a single matrix multiplication.
- A normalized Laplacian $L = I - P_E$ that allows fully decentralized, linear coordination across arbitrary combinations of overlapping mandates.

### Question 3: What do the dual variables mean economically?
Dual variables $v^*$ represent the **shadow cost of consensus tension**:
- If Desk 1 and Desk 2 both hold Asset 1, but Desk 1's local risk-return profile makes it want more Asset 1 than Desk 2, the consensus constraint $Lx = 0$ restrains Desk 1.
- The dual multiplier $v_1^1$ is the marginal penalty Desk 1 must absorb to remain aligned with the network.
- As shown in Tutorial 08, normalizing this shadow price produces an intuitive markup $\frac{v_i^j}{\sum_\ell v_\ell^j} \in [0, 1]$ that measures each desk's contribution to risk coupling.

### Question 4: Is the claimed distributed performance gain structural or a consequence of changing the problem?
Our controlled experiments in Tutorial 09 conclusively prove that it is **a consequence of changing the optimization problem**:
1. **Weight Downscaling**: Because $\sum x_i^j = 1$ is enforced across all local copies, the weight invested in each asset copy is divided by its degree $d_j$. Since variance scales quadratically, evaluating $w_j^2 = (z_j / d_j)^2$ reduces apparent quadratic risk by $\approx 1/d_j^2$.
2. **Missing Covariances**: Sub-portfolios only evaluate internal covariances. Cross-subportfolio covariances (such as between Desk 2's assets and Desk 3's assets) are omitted from the objective.
3. **True Performance on Market Covariance**: When the resulting distributed asset allocation $z^* = (0.4545, 0.2272, 0.3173)$ is evaluated on the true market covariance $Q$, its actual objective is **2.9197**, which is slightly worse than the centralized minimum of **2.8408**.

### Question 5: Where can predictive uncertainty enter economically?
Tutorial 11 formulates candidate mappings through which Bayesian predictive distributions $p(\mu \mid \mathcal{D}) \sim (\mathbb{E}[\mu], \operatorname{Var}(\mu))$ from Bayesian LoRA or neural ensembles could enter at three distinct levels:
1. **Asset Level (Return Shrinkage)**: $\tilde{\mu}_j = \frac{\mu_j}{1 + \kappa \operatorname{Var}(\mu_j)}$, penalizing assets with high epistemic uncertainty.
2. **Strategy Level (Risk Augmentation)**: $\tilde{Q}_i = Q_i + \gamma \operatorname{diag}(\operatorname{Var}(\mu_i))$, treating parameter uncertainty as unhedged variance.
3. **Firm Level (Mandate Re-weighting)**: $\tilde{a}_i \propto \frac{a_i}{1 + \beta \operatorname{Tr}(\operatorname{Var}(\mu_i))}$, shifting capital toward allocators with higher confidence.

### Question 6: What does the current evidence justify?
The original proposal was an out-of-sample rolling dynamic allocation study with
Bayesian LoRA feeding a hypergraph solver. That proposal is historical, not the current
next experiment. Subsequent leakage and reliability-metric audits substantially weakened
the early Bayesian uncertainty interpretation; clean metadata forecasts remained weak.

The separate frozen specialist pilot did not establish a collaborative LoRA effect.
Learned weights collapsed to the banking specialist, CoLoRA was approximately tied with
pooled/global training, and the target-training prevalence null was strongest on Brier
and log loss. Neither learned weighting nor CoLoRA passed the frozen continuation
criteria. Bayesian/UQ and portfolio-style specialist weighting were therefore not escalated.

The [closure document](../docs/collaborative_lora_branch_closure.md) distinguishes failure
to establish collaboration from weak task signal and possible lack of complementarity.
Revival requires a materially different task regime or data-generating setup with
independently demonstrated complementary predictive signal and fresh held-out evaluation.
Another weighting-rule search on this pilot is not justified. This scoped negative result
does not imply that all collaborative LoRA methods are impossible.

---

## 3. Laboratory Architecture Summary

The following tree records the original mathematical laboratory. It is historical, not a current inventory or production-readiness claim; see the README for the expanded repository:

```
hypergraph-portfolio-lab/
├── pyproject.toml                         # Packaging & dependencies
├── README.md                              # Complete lab overview & roadmap
├── src/hypergraph_portfolio/
│   ├── mean_variance.py                   # Centralized Markowitz solver & KKT
│   ├── hypergraph.py                      # Incidence E, degree D_E, Laplacian L
│   ├── consensus.py                       # Nullspace projectors & consistency audit
│   ├── primal_dual.py                     # Continuous-time ODE dynamics & Lyapunov
│   ├── pricing.py                         # Endogenous shadow price mechanism
│   └── uncertainty.py                     # 5 Bayesian uncertainty mechanisms
├── experiments/
│   ├── run_three_asset.py                 # Exact reproduction of Section 3
│   ├── run_audit_experiments.py           # Controlled audit suite (Tutorial 09)
│   ├── run_gics_etf.py                    # 11-sector GICS ETF experiment (Tutorial 10)
│   └── run_uncertainty_experiments.py     # Bayesian uncertainty comparison (Tutorial 11)
├── tests/                                 # 16 comprehensive unit tests (all passing)
└── tutorials/                             # 13 first-principles markdown tutorials (00–12)
```

---

## What You Should Now Understand

1. The exact mathematical strengths and limitations of the hypergraph portfolio framework.
2. Why claims of "free diversification" in distributed optimization must always be audited against true market covariance.
3. Why mathematical uncertainty mappings and portfolio analogies require separate empirical evidence before they justify specialist weighting.

---

## Questions Worth Investigating

- Can we design a hypergraph consensus algorithm where desks reach consensus not on static weights, but on dynamic hedging policies?
- How does the convergence speed of the continuous-time primal-dual ODE scale as the asset universe grows from 11 ETFs to 500 S&P stocks?
