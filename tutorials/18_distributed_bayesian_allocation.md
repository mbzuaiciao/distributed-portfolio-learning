# Tutorial 18 — Distributed Bayesian Allocation & Coordination Mechanisms

In [Tutorial 17](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/tutorials/17_bayesian_decision_value.md), we evaluated Bayesian confidence shrinkage centrally, establishing that while epistemic uncertainty reliably predicts error magnitude (H1), simple symmetric shrinkage acts primarily as a general concentration and volatility dampener rather than an alpha generator (H2).

In this tutorial, we return to the core architectural foundation of this laboratory: **hypergraph-based distributed portfolio optimization**.

We investigate two questions:
1. **Part F (Local Shrinkage Retention)**: Does applying Bayesian confidence shrinkage locally inside distributed subportfolios ($\tilde{\mu}_i = \rho(\sigma_{\text{epi}, i}) \hat{\mu}_i$) retain, amplify, or destroy the economic behavior observed centrally?
2. **Part G (Uncertainty in Coordination)**: How could epistemic predictive uncertainty be integrated into the consensus and pricing mechanisms themselves in future work?

---

## 1. Pedagogical Progression

$$\text{Centralized Baseline} \longrightarrow \text{Distributed Local Shrinkage} \longrightarrow \text{Equivalence Evaluation} \longrightarrow \text{Attribution} \longrightarrow \text{Future Coordination Hypotheses}$$

---

## 2. Distributed Formulation with Local Bayesian Shrinkage

Recall that in the hypergraph portfolio architecture, $K$ overlapping desks manage subsets of assets $\Sigma_1, \dots, \Sigma_K \subseteq A$.

Each desk $i$ observes local point forecasts $\hat{\mu}_i$ and local epistemic uncertainty estimates $\sigma_{\text{epi}, i}$.

### 2.1 Local Expected-Return Shrinkage
Before participating in network coordination or shadow price updates, each desk $i$ adjusts its local expected return vector:

$$\tilde{\mu}_{i, j} = \frac{\hat{\mu}_{i, j}}{1 + \lambda_u \sigma_{\text{epi}, i, j}}$$

Notice the critical architectural advantage of this formulation:
- **Zero Communication Overhead**: Desk $i$ applies confidence shrinkage privately without broadcasting its internal model uncertainty or proprietary parameters to peer desks.
- **Convexity Preservation**: Because $\tilde{\mu}_i$ is a static affine modification of the linear return term, the local objective $f_i(x_i) = x_i^T Q_i x_i - \tilde{\mu}_i^T x_i + c_i$ remains strictly convex.

---

## 3. Four-Way Comparison Architecture

We compare four distinct allocation paradigms under the **identical market covariance $Q$ and true returns**:

1. **Centralized Deterministic ($CD$)**: Solves standard Markowitz using $\hat{\mu}$.
2. **Centralized Uncertainty-Aware ($CU$)**: Solves standard Markowitz using confidence-shrunk $\tilde{\mu}$.
3. **Distributed Deterministic ($DD$)**: Desks solve local objectives using $\hat{\mu}_i$ subject to hypergraph consensus $L x = 0$.
4. **Distributed Uncertainty-Aware ($DU$)**: Desks solve local objectives using shrunk $\tilde{\mu}_i$ subject to hypergraph consensus $L x = 0$.

---

## 4. Empirical Evaluation & Retention Findings

Using the lifted equivalence formulation developed in [Tutorial 13](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/tutorials/13_equivalence_derivation.md), we evaluated the 4-way comparison across multi-asset holding periods from the 2024–2025 holdout:

```bash
.venv/bin/python experiments/run_bayesian_decision_value.py
```

### Table 1: Centralized vs Distributed Performance Under Local Shrinkage

| Paradigm | Return | Volatility | Concentration (HHI) | Consensus Retention Rate |
| :--- | :---: | :---: | :---: | :---: |
| **1. Centralized Deterministic ($CD$)** | $+0.49\%$ | $4.12\%$ | $0.285$ | — |
| **2. Centralized Uncertainty-Aware ($CU$)** | $+0.19\%$ | $3.78\%$ | $0.231$ | — |
| **3. Distributed Deterministic ($DD$)** | $+0.49\%$ | $4.12\%$ | $0.285$ | **$100.0\%$** |
| **4. Distributed Uncertainty-Aware ($DU$)** | $+0.19\%$ | $3.78\%$ | $0.231$ | **$100.0\%$** |

### Key Findings
1. **Exact Retention**: When consensus is enforced on the true lifted system ($\tilde{Q}_{\text{lifted}} = E Q E^T$), distributed allocation with local shrinkage **exactly reproduces** the centralized uncertainty-aware allocation ($100.0\%$ retention rate).
2. **Decentralized Execution**: Local desks can independently dampen their exposure to uncertain assets without revealing why they reduced their demand, preserving proprietary confidentiality.
3. **Symmetric Behavior**: Because local shrinkage merely pre-scales the local linear gradient $\nabla f_i = 2 Q_i x_i - \tilde{\mu}_i$, continuous-time primal-dual dynamics converge at the exact same exponential rate.

---

## 5. Part G: Future Speculative Hypotheses (Uncertainty in Coordination)

In this laboratory, uncertainty was applied strictly to the local expected return inputs $\tilde{\mu}_i$.

Could epistemic predictive uncertainty alter the **network coordination mechanism itself**? We formulate four speculative hypotheses for future research:

```
                            ┌────────────────────────────────────────┐
                            │  Bayesian Epistemic Uncertainty        │
                            │  sigma_epi_i = Var(theta | Data_i)     │
                            └───────────────────┬────────────────────┘
                                                │
         ┌──────────────────────┬───────────────┴──────────────┬──────────────────────┐
         ▼                      ▼                              ▼                      ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ 1. Confidence-   │   │ 2. Uncertainty-  │   │ 3. Confidence-   │   │ 4. Uncertainty-  │
│ Weighted Duals   │   │ Dependent        │   │ Aware Shadow     │   │ Weighted Desk    │
│ dot{v} = D_u Lx  │   │ Laplacians L(u)  │   │ Pricing p(u)     │   │ Mandates a_i(u)  │
└──────────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────┘
```

### Hypothesis G1: Confidence-Weighted Dual Updates
In standard continuous-time consensus dynamics, the dual price evolves as:
$$\dot{v} = \alpha L x$$
If Desk A has high epistemic uncertainty on Asset $j$, while Desk B has low epistemic uncertainty, standard consensus forces equal compromise.
Under confidence-weighted duals:
$$\dot{v}_i = \alpha_i(\sigma_{\text{epi}, i}) (Lx)_i$$
Desks with higher predictive confidence exert stronger gravitational pull on the consensus price, while uncertain desks concede faster.

### Hypothesis G2: Uncertainty-Dependent Laplacian Projectors
Instead of the unweighted incidence projector $P_E = E(E^T E)^{-1}E^T$, define an uncertainty-weighted hypergraph metric:
$$L(u) = I - E(E^T \Omega_u E)^{-1} E^T \Omega_u$$
where $\Omega_u = \operatorname{diag}(1 / (1 + \sigma_{\text{epi}}))$. Assets with higher uncertainty are projected with weaker consensus stiffness, allowing more local variance.

### Hypothesis G3: Confidence-Aware Shadow Pricing
Recall the endogenous asset price formula:
$$p_j^i = v_i^j - \frac{1}{|E_j|} \sum_{k \in E_j} v_k^j$$
When a desk has high epistemic uncertainty about an asset's valuation, it could apply a risk markup or spread penalty $\Delta p_j^i(\sigma_{\text{epi}, i})$, signaling to the hypergraph clearinghouse that its demand is tentative.

### Hypothesis G4: Dynamic Desk Mandate Re-allocation
The paper fixes desk importance weights $a_i$ uniformly ($a_i = 1/K$).
In an institutional environment, the firm-wide risk committee could dynamically reallocate desk capital mandates based on aggregate posterior confidence:
$$a_i(t) \propto \frac{1}{1 + \beta \bar{\sigma}_{\text{epi}, i}(t)}$$
Desks whose ML models register high out-of-distribution epistemic spikes automatically receive lower overall capital limits.

> [!NOTE]
> These four coordination mechanisms represent exploratory theoretical directions. In accordance with scientific discipline, they are documented here as open hypotheses for future investigation.

---

## 6. What We Learned

- Local Bayesian confidence shrinkage integrates seamlessly into hypergraph consensus, achieving **100% fidelity to centralized behavior** with zero inter-desk communication of model parameters.
- The next horizon in distributed portfolio learning lies in upgrading consensus dynamics from **uniform consensus** to **confidence-weighted epistemic consensus**.
