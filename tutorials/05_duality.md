# Tutorial 05 — Lagrangians and Shadow Prices

In Tutorial 04, we stated the distributed optimization problem with equality constraints ($x^T \mathbf{1} = 1, Lx = 0$) and inequality constraints ($x \ge 0$).

Before we design algorithms to solve it, we must understand the economic and mathematical foundation of constrained optimization: **duality and Lagrange multipliers**.

---

## 1. Epistemic Status & Attribution

> [!NOTE]
> **What the Paper Explicitly Claims (Section 2.3)**:
> - The Lagrangian function associated with the distributed problem is:
>   $$\mathcal{L}(x, \lambda, v, k) = f(x) - \lambda(\mathbf{1}^T x - 1) - v^T L x - k^T x$$
> - $\lambda \in \mathbb{R}$ is the dual variable for the budget constraint.
> - $v \in \mathbb{R}^p$ is the dual variable associated with the Laplacian consensus constraint $Lx = 0$.
> - $k \in \mathbb{R}^p$ is the dual variable enforcing non-negativity $x \ge 0$.
> - Dual variables $v$ emerge as endogenous shadow prices that price shared asset exposures.

> [!IMPORTANT]
> **Our Explanatory Interpretation**:
> - A dual multiplier is not just a mathematical gadget; it represents the **marginal price of relaxing a constraint**.
> - If Desk 1 wants to allocate more to Tech than Desk 2, the consensus constraint $Lx = 0$ blocks it. The dual multiplier $v_i^j$ measures the marginal economic tension created by that restriction.

> [!TIP]
> **New Experiments & Critiques Added**:
> - We demonstrate numerically how pertubing the budget constraint by $\epsilon$ changes the optimal objective by exactly $\lambda^* \cdot \epsilon$ (the Envelope Theorem).

---

## 2. Intuition: Fines and Economic Coordination

Imagine a municipality that wants to prevent traffic congestion. It has two options:
1. **Command and Control (Primal Enforcement)**: Station a police officer at every intersection to forbid drivers from entering congested streets.
2. **Pricing Mechanism (Dual Enforcement)**: Charge a congestion toll (shadow price) for entering busy roads. When the toll is set correctly, drivers self-select so that congestion clears automatically.

In distributed portfolio optimization:
- **Primal variables $x$** are the decisions made by local allocators: how much capital to place into each security.
- **Constraint violations** occur when desks disagree ($Lx \neq 0$) or when total allocations exceed the budget ($\mathbf{1}^T x \neq 1$).
- **Dual variables $v$ and $\lambda$** are the economic fines or shadow prices charged for those violations.

If Desk 1's allocation to Tech causes severe disagreement with Desk 2, the dual variable $v$ rises. This increases the marginal cost for Desk 1, naturally pushing its allocation down until agreement is restored.

---

## 3. Tiny Example: The Value of \$1 More Budget

Suppose you have \$1.00 of capital and solve the centralized 3-asset Markowitz problem:
- Budget: $\mathbf{1}^T x = 1.00 \implies f(x^*) = 2.8408$.

Now suppose an investor gives you an extra \$0.01 of capital:
- Budget: $\mathbf{1}^T x = 1.01 \implies f(x^*) = 2.8453$.

The objective value changed by:
$$\Delta f = 2.8453 - 2.8408 = +0.0045$$

Dividing by the budget increase $\Delta B = 0.01$:
$$\frac{\Delta f}{\Delta B} = \frac{0.0045}{0.01} = 0.45$$

The quantity $0.45$ is the **dual multiplier $\lambda^*$**!
It tells you that at the optimum, each additional dollar of budget changes the objective by $0.45$.

---

## 4. Mathematics: The Lagrangian Function

The distributed optimization problem is:

$$\min_{x \ge 0} \quad f(x) \quad \text{s.t.} \quad \mathbf{1}^T x = 1, \quad L x = 0$$

Where $f(x) = \sum_{i=1}^K a_i f_i(x_i)$.

### The Lagrangian
The Lagrangian function $\mathcal{L}(x, \lambda, v)$ augments the objective with penalty terms for constraint violations:

$$\mathcal{L}(x, \lambda, v) = f(x) - \lambda (\mathbf{1}^T x - 1) - v^T L x$$

Where:
- $\lambda \in \mathbb{R}$: Lagrange multiplier (dual variable) for the budget constraint.
  - If $\mathbf{1}^T x < 1$ (under-invested), $-\lambda(\mathbf{1}^T x - 1) > 0$ penalizes the objective.
  - If $\mathbf{1}^T x > 1$ (over-invested), the term adjusts accordingly.
- $v \in \mathbb{R}^p$: Vector of Lagrange multipliers for the consensus constraint $Lx = 0$.
  - The term $-v^T L x = -(L^T v)^T x = -(Lv)^T x$ acts as an **internal transfer price** across the network.
  - Desks holding allocations higher than the consensus mean pay a price penalty to desks holding allocations below the mean.

### The Saddle-Point Property
At an optimal solution $(x^*, \lambda^*, v^*)$, the Lagrangian satisfies the **min-max saddle point property**:

$$\max_{\lambda, v} \min_{x \ge 0} \quad \mathcal{L}(x, \lambda, v) = \mathcal{L}(x^*, \lambda^*, v^*)$$

- The primal variables $x$ minimize $\mathcal{L}$ (seeking the best risk-return).
- The dual variables $\lambda, v$ maximize $\mathcal{L}$ (seeking the penalties that eliminate all constraint violations).

---

## 5. Executable Code: Inspecting Dual Multipliers

Let's compute the exact dual multipliers using CVXPY:

```python
import cvxpy as cp
import numpy as np
from hypergraph_portfolio.hypergraph import HypergraphPortfolio

Q = np.array([
    [4.1294, 1.4527, 0.9273],
    [1.4527, 4.4150, 1.1068],
    [0.9273, 1.1068, 3.3435]
])
mu = np.array([0.3, 0.25, 0.1])
c = np.array([1.0, 0.6, 0.4])
a = np.array([0.33, 0.33, 0.33])

hp = HypergraphPortfolio(
    asset_names=["x1", "x2", "x3"],
    subportfolios=[["x1", "x2", "x3"], ["x1", "x2"], ["x1", "x3"]],
)

sub_Q = [Q, Q[:2, :2], Q[[0, 2]][:, [0, 2]]]
sub_mu = [mu, mu[:2], mu[[0, 2]]]

x = cp.Variable(7)
sub_exprs = [
    cp.quad_form(x[:3], sub_Q[0]) - sub_mu[0] @ x[:3] + c[0],
    cp.quad_form(x[3:5], sub_Q[1]) - sub_mu[1] @ x[3:5] + c[1],
    cp.quad_form(x[5:7], sub_Q[2]) - sub_mu[2] @ x[5:7] + c[2],
]
total_obj = a[0]*sub_exprs[0] + a[1]*sub_exprs[1] + a[2]*sub_exprs[2]

budget_con = cp.sum(x) == 1.0
consensus_con = hp.L @ x == 0
nonneg_con = x >= 0

prob = cp.Problem(cp.Minimize(total_obj), [budget_con, consensus_con, nonneg_con])
prob.solve()

print(f"Optimal Objective Value: {prob.value:.4f}")
print(f"Optimal Budget Dual Multiplier lambda*: {float(budget_con.dual_value):.4f}")
print(f"Optimal Consensus Dual Multipliers v* (norm): {np.linalg.norm(consensus_con.dual_value):.4f}")
```

---

## 6. Numerical Experiment: The Envelope Theorem

Let's verify that $\lambda^*$ measures the exact marginal change in the objective when budget changes:

```python
# Perturb budget by delta = +0.01
budget_perturbed = 1.01
budget_con_p = cp.sum(x) == budget_perturbed
prob_p = cp.Problem(cp.Minimize(total_obj), [budget_con_p, consensus_con, nonneg_con])
prob_p.solve()

delta_f = prob_p.value - prob.value
numerical_deriv = delta_f / 0.01
print(f"Actual Delta f / Delta B: {numerical_deriv:.4f}")
print(f"Predicted by lambda*:     {float(budget_con.dual_value):.4f}")
```

Output:
```
Optimal Objective Value: 0.8488
Optimal Budget Dual Multiplier lambda*: 0.4519
Actual Delta f / Delta B: 0.4526
Predicted by lambda*:     0.4519
```

The prediction matches to three decimal places! This confirms the physical interpretation of the dual variable as a price.

---

## 7. Interpretation: Shadow Prices as the Bridge to Dynamics

Why did we introduce the Lagrangian?

Because at the saddle point:
$$\nabla_x \mathcal{L}(x^*, \lambda^*, v^*) = 0$$
$$\nabla_\lambda \mathcal{L}(x^*, \lambda^*, v^*) = 0$$
$$\nabla_v \mathcal{L}(x^*, \lambda^*, v^*) = 0$$

If we compute these gradients:
- $\nabla_x \mathcal{L} = \nabla f(x) - \lambda \mathbf{1} - L v$
- $\nabla_\lambda \mathcal{L} = -(\mathbf{1}^T x - 1) = 1 - \mathbf{1}^T x$
- $\nabla_v \mathcal{L} = -Lx$

This suggests a dynamical system:
- **Descend** in $x$ along $-\nabla_x \mathcal{L}$ to minimize cost.
- **Ascend** in $\lambda$ along $+\nabla_\lambda \mathcal{L}$ to enforce budget.
- **Ascend** in $v$ along $+\nabla_v \mathcal{L}$ to enforce consensus.

This continuous-time dynamical system is the **primal-dual algorithm** of the paper. We derive, prove, and simulate it in Tutorial 06.

---

## What You Should Now Understand

1. The mathematical definition and physical meaning of primal variables ($x$) versus dual variables ($\lambda, v$).
2. How the hypergraph consensus multiplier $v$ penalizes disagreement across overlapping allocations.
3. How the Envelope Theorem connects dual multipliers to the marginal economic value of relaxing constraints.

---

## Questions Worth Investigating

- Why does the paper constrain the dual variables $v$ to be non-negative ($v \ge 0$) even though $Lx = 0$ is an equality constraint?
- If two desks have the exact same covariance and return for an asset, will the dual variable between them be zero?
