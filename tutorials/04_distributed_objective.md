# Tutorial 04 — The Distributed Mean-Variance Objective

In Tutorials 00–03, we built the necessary mathematical machinery:
1. Markowitz mean-variance optimization (Tutorial 00)
2. Modular sub-portfolios (Tutorial 01)
3. The hypergraph incidence matrix $E$ and Laplacian $L$ (Tutorial 02)
4. The consensus constraint $Lx = 0$ (Tutorial 03)

Now we assemble these pieces into the full **distributed mean-variance portfolio optimization problem** introduced in Section 2 of Papastaikoudis, Watson, and Lestas (2025).

---

## 1. Epistemic Status & Attribution

> [!NOTE]
> **What the Paper Explicitly Claims (Section 2, Equation 1)**:
> The distributed portfolio optimization problem is:
> $$\min_{x = [x_1, \dots, x_K]} \sum_{i=1}^K a_i f_i(x_i)$$
> subject to:
> - $x^T \mathbf{1} = 1$ (global budget constraint)
> - $L x = 0$ (consensus across shared assets)
> - $x \ge 0$ (no short-selling or leverage)
> Where $f_i(x_i) = x_i^T Q_i x_i - x_i^T \mu_i + c_i$, and $a_i$ are sub-portfolio weights summing to 1.

> [!IMPORTANT]
> **Our Explanatory Interpretation**:
> - Economically, $a_i$ represents the institutional priority or risk aversion weight of sub-portfolio $i$.
> - The fixed cost $c_i$ captures desk management fees, research overhead, or licensing costs for running strategy $i$.
> - Notice that the budget constraint $x^T \mathbf{1} = 1$ sums across **all local decision variables** $x = [x_1, \dots, x_K]^T$. This means total capital is split among the sub-portfolios, so each individual sub-portfolio invests strictly less than \$1.00!

> [!TIP]
> **New Experiments & Critiques Added**:
> - We show how the global objective decomposes into individual desk contributions $(f_1, f_2, f_3)$, and verify the exact formulation using CVXPY.

---

## 2. Intuition: The Multi-Desk Coordination Problem

Imagine a global multi-strategy fund with $K$ specialized investment desks.
The fund management team does not want to micromanage each desk's trading algorithm. Instead:
1. **Local Autonomy**: Each desk $i$ manages its own assets $x_i$, minimizing its own local variance $x_i^T Q_i x_i$, maximizing its own return $x_i^T \mu_i$, and paying its own operational fee $c_i$.
2. **Firm-Level Priority**: The fund CIO sets weights $a_i \ge 0$ ($\sum a_i = 1$) to decide how much importance to give to each desk's performance.
3. **Firm-Wide Solvency**: The firm has a single capital pool of \$1.00 ($x^T \mathbf{1} = 1$).
4. **Coordinated Positions**: Desks are not allowed to hold contradictory positions on shared assets ($Lx = 0$).

The goal is to find allocations that minimize the weighted sum of desk objectives while respecting global budget and consensus.

---

## 3. Tiny Example: Breakdown of Parameters

Let's examine the 3-subportfolio example from Section 3.2:
- Assets: $A = \{x_1, x_2, x_3\}$.
- Desks:
  - Desk 1 ($\Sigma_1 = \{x_1, x_2, x_3\}$): trades all 3 assets. Dimension $p_1 = 3$.
  - Desk 2 ($\Sigma_2 = \{x_1, x_2\}$): trades Tech and Industrials. Dimension $p_2 = 2$.
  - Desk 3 ($\Sigma_3 = \{x_1, x_3\}$): trades Tech and Treasuries. Dimension $p_3 = 2$.

### Economic Meaning of Each Matrix/Vector
1. **Local Covariances $Q_i$**:
   - $Q_1$ is the full $3 \times 3$ market covariance.
   - $Q_2 = \begin{pmatrix} 4.13 & 1.45 \\ 1.45 & 4.42 \end{pmatrix}$ evaluates risk only between Tech and Industrials. Desk 2 ignores Treasuries entirely!
   - $Q_3 = \begin{pmatrix} 4.13 & 0.93 \\ 0.93 & 3.34 \end{pmatrix}$ evaluates risk only between Tech and Treasuries.
2. **Local Expected Returns $\mu_i$**:
   - $\mu_1 = (0.30, 0.25, 0.10)^T$
   - $\mu_2 = (0.30, 0.25)^T$
   - $\mu_3 = (0.30, 0.10)^T$
3. **Fixed Operational Costs $c_i$**:
   - $c_1 = 1.0$ (running all 3 assets is expensive).
   - $c_2 = 0.6$ (medium desk overhead).
   - $c_3 = 0.4$ (lower overhead).
4. **Desk Importance Weights $a_i$**:
   - $a = (0.33, 0.33, 0.33)$ (equal priority across all three desks).

---

## 4. Mathematics: The Full Constrained Problem

We stack all local allocation vectors into a single global vector $x \in \mathbb{R}^p$ ($p = 7$):

$$x = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} = \begin{pmatrix} x_1^1 \\ x_2^1 \\ x_3^1 \\ x_1^2 \\ x_2^2 \\ x_1^3 \\ x_3^3 \end{pmatrix} \in \mathbb{R}^7$$

The optimization problem is:

$$\min_{x \in \mathbb{R}^7} \quad F(x) = \sum_{i=1}^3 a_i \left( x_i^T Q_i x_i - \mu_i^T x_i + c_i \right)$$

subject to:

$$\mathbf{1}^T x = 1 \quad (\text{global budget})$$
$$L x = 0 \quad (\text{hypergraph consensus})$$
$$x \ge 0 \quad (\text{long-only})$$

### Matrix Block Representation
We can write $F(x)$ compactly using block-diagonal matrices:

$$F(x) = x^T H x - g^T x + C$$

Where:
- $H = \operatorname{blkdiag}(a_1 Q_1, a_2 Q_2, a_3 Q_3) \in \mathbb{R}^{7 \times 7}$
- $g = (a_1 \mu_1^T, a_2 \mu_2^T, a_3 \mu_3^T)^T \in \mathbb{R}^7$
- $C = \sum_{i=1}^3 a_i c_i = 0.33(1.0 + 0.6 + 0.4) = 0.66$

Notice that $H$ is block-diagonal! 
**There are zero off-diagonal blocks connecting Desk 2 and Desk 3 in $H$.**
The only interaction between the desks comes through the linear constraint $Lx = 0$.

---

## 5. Executable Code

Let's solve this problem and inspect the components using CVXPY:

```python
import cvxpy as cp
import numpy as np
from hypergraph_portfolio.hypergraph import HypergraphPortfolio

# Base data
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

# CVXPY Formulation
x = cp.Variable(7)
x1 = x[0:3]
x2 = x[3:5]
x3 = x[5:7]

f1 = cp.quad_form(x1, sub_Q[0]) - sub_mu[0] @ x1 + c[0]
f2 = cp.quad_form(x2, sub_Q[1]) - sub_mu[1] @ x2 + c[1]
f3 = cp.quad_form(x3, sub_Q[2]) - sub_mu[2] @ x3 + c[2]

obj = cp.Minimize(a[0] * f1 + a[1] * f2 + a[2] * f3)
constraints = [
    cp.sum(x) == 1.0,
    hp.L @ x == 0,
    x >= 0,
]

prob = cp.Problem(obj, constraints)
prob.solve()

print("Optimal Local Allocation Vector x* (7 variables):")
print(np.round(x.value, 4))
print(f"\nOptimal Total Objective: {prob.value:.4f}")
print(f"Sub-objectives: f1={f1.value:.4f}, f2={f2.value:.4f}, f3={f3.value:.4f}")
```

---

## 6. Numerical Experiment & Verification

The output is:

```
Optimal Local Allocation Vector x* (7 variables):
[0.1517 0.1136 0.1589 0.1517 0.1136 0.1517 0.1589]

Optimal Total Objective: 0.8488
Sub-objectives: f1=1.2813, f2=0.7281, f3=0.5627
```

Let's check the values reported in the paper (page 12):
- Paper sub-objectives: $f_1 = 1.2813$, $f_2 = 0.7281$, $f_3 = 0.5627$.
  $$\frac{1.2813 + 0.7281 + 0.5627}{3} = \frac{2.5721}{3} = 0.857367 \approx 0.8573$$
- Our exact CVXPY values match $f_1 = 1.2813$, $f_2 = 0.7281$, $f_3 = 0.5627$!

Now look at the total capital deployed by each desk:
- Desk 1 ($x_1$): $0.1517 + 0.1136 + 0.1589 = \mathbf{0.4242}$
- Desk 2 ($x_2$): $0.1517 + 0.1136 = \mathbf{0.2653}$
- Desk 3 ($x_3$): $0.1517 + 0.1589 = \mathbf{0.3106}$
- Sum across all desks: $0.4242 + 0.2653 + 0.3106 = \mathbf{1.0000}$.

And what is the total firm-wide investment in each underlying asset?
$$z = E^T x^* = \begin{pmatrix} 3 \times 0.1517 \\ 2 \times 0.1136 \\ 2 \times 0.1589 \end{pmatrix} = \begin{pmatrix} \mathbf{0.4551} \\ \mathbf{0.2272} \\ \mathbf{0.3178} \end{pmatrix}$$
Sum of underlying holdings: $0.4551 + 0.2272 + 0.3178 = 1.0001 \approx 1.00$.

---

## 7. Interpretation: How Centralized Solvers Obscure the Economics

We just solved this problem using CVXPY in 5 lines of code.
However, using a centralized black-box interior-point solver hides the entire economic mechanism:
- How does Desk 1 communicate with Desk 2?
- What economic force penalizes them when their copies of Asset 1 disagree?
- Where do the shadow prices come from?

In Tutorial 05 and Tutorial 06, we move beyond black-box solvers. We derive the **Lagrangian** and continuous-time **primal-dual dynamics**, uncovering the endogenous shadow prices that coordinate this network.

---

## What You Should Now Understand

1. How the distributed objective aggregates local quadratic functions $f_i(x_i)$ with importance weights $a_i$.
2. Why the decision vector $x$ contains local copies of shared assets, and how $Lx = 0$ couples them.
3. How capital is divided among sub-portfolios under the global budget constraint $x^T \mathbf{1} = 1$.

---

## Questions Worth Investigating

- Why did the paper report $f(x^*) = 0.8573$ when the centralized Markowitz objective in Tutorial 00 was $2.8408$?
- Does the lower objective value mean the distributed portfolio is safer, or is it an artifact of evaluating the objective on downscaled sub-portfolio weights? (We audit this in Tutorial 09).
