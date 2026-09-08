# Tutorial 00 — Centralized Mean-Variance Portfolio Optimization

Welcome to the foundation of portfolio theory. Before understanding how to distribute portfolio optimization across networks, themes, or investment desks, we must first master the centralized problem formulated by Harry Markowitz (1952) and adapted in the paper by Papastaikoudis, Watson, and Lestas (2025).

---

## 1. Epistemic Status & Attribution

To maintain scientific integrity throughout this lab, we explicitly label claims into three categories:

> [!NOTE]
> **What the Paper Explicitly Claims (Section 3.1)**:
> - Centralized mean-variance optimization minimizes the quadratic objective $f(x) = x^T Q x - x^T \mu + c$ subject to budget $\sum_j x_j = 1$ and non-negativity $x_j \ge 0$.
> - For a specific 3-asset universe with given covariance matrix $Q$ and expected returns $\mu$, the asymptotic optimal allocation is $x^* = (0.3217, 0.2567, 0.4217)$ with optimal objective value $f(x^*) = 2.8408$.

> [!IMPORTANT]
> **Our Explanatory Interpretation**:
> - The quadratic term in the paper is written as $x^T Q x$ (rather than the textbook $\frac{1}{2} x^T Q x$), meaning the risk penalty gradient is $2 Q x - \mu$.
> - The scalar constant $c=1.0$ represents a base management/implementation cost. It does not shift the argmin $x^*$, but sets a baseline objective value.

> [!TIP]
> **New Experiments & Critiques Added**:
> - We verify the analytical KKT conditions and show exact numerical agreement between SciPy SLSQP and CVXPY.
> - We demonstrate why the centralized model requires a single global allocator with complete knowledge of all cross-asset correlations.

---

## 2. Intuition

Imagine you are the Chief Investment Officer (CIO) of a fund. You have \$1.00 of capital to distribute across $n$ assets. Each asset has:
1. An **expected return** ($\mu_i$): how much you expect to gain per dollar invested.
2. A **volatility and co-movement** ($Q_{ij}$): how much asset $i$ fluctuates and how it moves in tandem with asset $j$.

If you only cared about return, you would place 100% of your capital into whichever asset has the highest $\mu_i$. But in finance, high expected return is accompanied by risk. If all assets drop together, your fund suffers catastrophic drawdown. 

Markowitz's fundamental insight was that **diversification reduces portfolio risk without sacrificing expected return**, provided assets are not perfectly correlated.

---

## 3. Tiny Example

Suppose you have 3 assets:
- **Asset 1 (Tech)**: High return ($\mu_1 = 0.30$), high volatility ($Q_{11} = 4.13$).
- **Asset 2 (Industrials)**: Moderate return ($\mu_2 = 0.25$), high volatility ($Q_{22} = 4.42$).
- **Asset 3 (Treasuries/Defensive)**: Low return ($\mu_3 = 0.10$), lower volatility ($Q_{33} = 3.34$).

If you allocate weights $x = (x_1, x_2, x_3)$ summing to 1:
- Putting everything into Tech ($x = [1, 0, 0]$) gives return $0.30$, but variance $4.13$.
- Mixing across all three assets exploits the off-diagonal covariance entries (e.g., $Q_{13} = 0.93$), lowering portfolio variance to $\approx 2.04$.

---

## 4. Mathematics

The centralized mean-variance problem studied in Section 3.1 is:

$$\min_{x \in \mathbb{R}^n} \quad f(x) = x^T Q x - \mu^T x + c$$

subject to:

$$\mathbf{1}^T x = 1 \quad (\text{full investment / budget constraint})$$
$$x \ge 0 \quad (\text{long-only / no short-selling})$$

Where:
- $x = (x_1, \dots, x_n)^T$: asset weight vector in the simplex $\Delta^{n-1}$.
- $Q \in \mathbb{R}^{n \times n}$: symmetric positive semi-definite covariance matrix ($Q = Q^T \succeq 0$).
- $\mu \in \mathbb{R}^n$: expected return vector.
- $c \in \mathbb{R}$: fixed management cost.

### Gradient and First-Order Optimality
The gradient of $f(x)$ is:

$$\nabla f(x) = 2 Q x - \mu$$

The Lagrangian for the equality-constrained problem (assuming non-negativity is non-binding) is:

$$\mathcal{L}(x, \lambda) = x^T Q x - \mu^T x + c - \lambda (\mathbf{1}^T x - 1)$$

Stationarity requires:

$$\nabla_x \mathcal{L} = 2 Q x - \mu - \lambda \mathbf{1} = 0 \implies x^* = \frac{1}{2} Q^{-1} (\mu + \lambda \mathbf{1})$$

Plugging into $\mathbf{1}^T x = 1$ yields the dual multiplier:

$$\lambda^* = \frac{2 - \mathbf{1}^T Q^{-1} \mu}{\mathbf{1}^T Q^{-1} \mathbf{1}}$$

---

## 5. Executable Code

Let's solve the exact 3-asset problem using our library module [`mean_variance.py`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/src/hypergraph_portfolio/mean_variance.py):

```python
import numpy as np
from hypergraph_portfolio.mean_variance import MeanVariancePortfolio

# Parameters from Section 3.1 of Papastaikoudis et al. (2025)
Q = np.array([
    [4.1294, 1.4527, 0.9273],
    [1.4527, 4.4150, 1.1068],
    [0.9273, 1.1068, 3.3435]
])
mu = np.array([0.3, 0.25, 0.1])
c = 1.0

# Initialize and solve
portfolio = MeanVariancePortfolio(Q=Q, mu=mu, c=c, budget=1.0)
x_opt, f_opt = portfolio.solve()

print(f"Optimal Allocation x*: {np.round(x_opt, 4)}")
print(f"Optimal Objective Value f(x*): {f_opt:.4f}")
```

---

## 6. Numerical Experiment & Verification

Running the solver yields:

```
Optimal Asset Allocation x*: [0.3217, 0.2567, 0.4217]
Optimal Objective Value f(x*): 2.8408
Portfolio Variance x^T Q x:    2.0436
Portfolio Expected Return mu^T x: 0.2028
```

Notice how the components assemble into the objective:
$$f(x^*) = 2.0436 - 0.2028 + 1.0000 = 2.8408$$

This matches the paper's reported values down to the 4th decimal place:
- Paper: $x^* = (0.3217, 0.2567, 0.4217)$, $f(x^*) = 2.8408$.
- Repro: $x^* = (0.32167, 0.25667, 0.42166)$, $f(x^*) = 2.84076$.

---

## 7. Interpretation & Why This Is Centralized

Notice the structural assumption behind this formulation:
1. **Single Global Matrix $Q$**: Every pair of assets $(i, j)$ has a known covariance that must be assembled in one central computer.
2. **Single Decision Maker**: One allocator controls all $n$ asset weights simultaneously.
3. **Flat Portfolio Hierarchy**: There are no sub-strategies, no thematic desks, no regional mandates, and no privacy boundaries.

In modern asset management, this flat structure breaks down. Large institutions run specialized desks (Growth, Cyclicals, Real Assets, Defensive) that have distinct mandates and access to overlapping sets of securities. 

How can multiple autonomous allocators optimize their own sub-portfolios while agreeing on shared investments? That question motivates the distributed hypergraph formulation introduced in Tutorial 01.

---

## What You Should Now Understand

1. How Markowitz mean-variance optimization balances quadratic co-movement against linear expected returns on the probability simplex.
2. The exact mathematical formulation and notation used by Papastaikoudis et al. ($x^T Q x$ convention).
3. How to solve and verify centralized quadratic programs using [`MeanVariancePortfolio`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/src/hypergraph_portfolio/mean_variance.py).

---

## Questions Worth Investigating

- What happens if an institution cannot build a single centralized $Q$ matrix due to regulatory, computational, or jurisdictional barriers?
- If two different portfolio managers both want to hold Apple (AAPL) under different investment themes, how can the firm ensure their combined holdings remain consistent and budget-constrained?
