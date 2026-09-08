# Tutorial 09 — Critical Audit of the Central Claim

In Section 4 of Papastaikoudis, Watson, and Lestas (2025), the authors make two monumental assertions:

1. **Informational Equivalence**:
   > *"Both the decentralized and centralized cases are informationally equivalent and aggregation-consistent... The decentralized model can be viewed as a lifted representation that preserves the economic content of the centralized case."* (Section 4.1)
2. **Structural Superiority**:
   > *"In the previous example the centralized solution yields an optimal risk of $f(x^*) = 2.8408$, while the decentralized approaches yield significantly lower risk objectives, namely $f(x^*) = 0.8573$ and $f(x^*) = 0.7609$... This result underscores the risk-reducing power of structured decentralization."* (Section 4.2)

In this research tutorial, we do not prejudge the answer. We audit these two claims through rigorous mathematical deduction and controlled computational experiments.

---

## 1. Epistemic Status & Attribution

> [!NOTE]
> **What the Paper Explicitly Claims (Section 4)**:
> - Centralized Markowitz minimum: $f(x^*) = 2.8408$.
> - Specific partitions distributed minimum: $f(x^*) = 0.8573$.
> - Generalized powerset distributed minimum: $f(x^*) = 0.7609$.
> - The paper claims this 70%+ reduction in the objective occurs under an isomorphic, informationally equivalent formulation, proving the structural superiority of hypergraph decentralization.

> [!IMPORTANT]
> **Our Explanatory Interpretation & Mathematical Proof**:
> - If problem B is a true "lifted reparameterization" of problem A, their optimal objective values **must be mathematically identical**. A change of variables cannot change the minimum of a function.
> - The reported drop from $2.8408$ to $0.8573$ occurs because the distributed objective function is **not** the centralized objective function evaluated on lifted coordinates.
> - Instead, the distributed objective evaluates local variances on downscaled weights ($w_j = z_j / d_j$), shrinking quadratic variance terms by $\approx 1/d_j^2$, omitting cross-subportfolio covariances, and scaling down fixed costs.

> [!TIP]
> **New Experiments & Critiques Added**:
> - We construct the **true lifted centralized Markowitz problem** and prove it yields **exactly 2.8408**.
> - We take the distributed asset allocation $z^* = (0.4545, 0.2272, 0.3173)$ and evaluate its actual risk on the true market covariance $Q$. Its real objective is **2.9197**, which is strictly **worse** than the centralized minimum of 2.8408.

---

## 2. Intuition: Comparing Apples to Slices of Apples

Suppose a grocery store sells a fruit basket containing 1 apple, 1 orange, and 1 banana for **\$2.84**.

Now suppose three roommates decide to buy fruit together:
- Roommate 1 buys $\frac{1}{3}$ of an apple, $\frac{1}{2}$ of an orange, and $\frac{1}{2}$ of a banana for **\$1.28**.
- Roommate 2 buys $\frac{1}{3}$ of an apple and $\frac{1}{2}$ of an orange for **\$0.73**.
- Roommate 3 buys $\frac{1}{3}$ of an apple and $\frac{1}{2}$ of a banana for **\$0.56**.

Together, they have bought exactly 1 apple, 1 orange, and 1 banana.
The average cost per roommate is:

$$\frac{\$1.28 + \$0.73 + \$0.56}{3} = \mathbf{\$0.857}$$

Can the roommates claim: *"By using a decentralized buying structure, the price of fruit dropped from \$2.84 to \$0.857 — a 70% savings!"*?

Obviously not! Each roommate paid \$0.857 because they only bought a **fractional slice** of the fruit!

In the paper's distributed formulation, the exact same thing happens. The global budget constraint is $\sum x_i^j = 1$. When there are 7 local variables, each local variable is roughly $0.15$ instead of $0.35$. When you evaluate a quadratic function $x_i^T Q_i x_i$ on variables that are 3 times smaller, the quadratic value is **9 times smaller**!

---

## 3. Controlled Experiment 1: The True Risk of Distributed Allocations

If the distributed approach truly produces *"less risk for the investor"* as claimed in Section 3.2, then when we look at the **actual portfolio of assets** held by the investor in the real world:

$$z^* = E^T x^* = (0.4545, 0.2272, 0.3173)^T$$

its true financial variance and true return evaluated under the real covariance matrix $Q$ must be lower than the centralized portfolio $x^*_{\text{cent}} = (0.3217, 0.2567, 0.4217)^T$.

Let's test this in Python:

```python
import numpy as np

Q = np.array([
    [4.1294, 1.4527, 0.9273],
    [1.4527, 4.4150, 1.1068],
    [0.9273, 1.1068, 3.3435]
])
mu = np.array([0.3, 0.25, 0.1])
c = 1.0

# 1. Centralized Optimum
x_cent = np.array([0.3217, 0.2567, 0.4217])
true_var_cent = x_cent @ Q @ x_cent
true_ret_cent = mu @ x_cent
f_true_cent = true_var_cent - true_ret_cent + c

# 2. Distributed Allocations (Aggregated)
z_spec = np.array([0.4545, 0.2272, 0.3173])  # Section 3.2
z_pwr  = np.array([0.3572, 0.3040, 0.3388])  # Section 3.3

f_true_spec = z_spec @ Q @ z_spec - mu @ z_spec + c
f_true_pwr  = z_pwr  @ Q @ z_pwr  - mu @ z_pwr  + c

print(f"Centralized Markowitz Minimum:  f_true = {f_true_cent:.4f} (Var: {true_var_cent:.4f}, Ret: {true_ret_cent:.4f})")
print(f"Distributed Specific on Q:      f_true = {f_true_spec:.4f} (Var: {z_spec @ Q @ z_spec:.4f}, Ret: {mu @ z_spec:.4f})")
print(f"Distributed Powerset on Q:      f_true = {f_true_pwr:.4f} (Var: {z_pwr @ Q @ z_pwr:.4f}, Ret: {mu @ z_pwr:.4f})")
```

Output:
```
Centralized Markowitz Minimum:  f_true = 2.8408 (Var: 2.0436, Ret: 0.2028)
Distributed Specific on Q:      f_true = 2.9197 (Var: 2.1446, Ret: 0.2249)
Distributed Powerset on Q:      f_true = 2.8696 (Var: 2.0866, Ret: 0.2170)
```

### Finding 1
When evaluated fairly on the real-world asset covariance matrix $Q$:
- Centralized Markowitz objective: **2.8408**
- Distributed Specific allocation: **2.9197** (+0.0789 WORSE)
- Distributed Powerset allocation: **2.8696** (+0.0288 WORSE)

The distributed portfolio does **not** have less risk. It has **more risk** (variance 2.14 vs 2.04) and a higher overall objective value. 
By definition of convex optimization, the Markowitz solution is the global minimum over the simplex. No other allocation can beat it on the same objective.

---

## 4. Controlled Experiment 2: Dissecting the 0.8573 Gap

Why did the paper report $f(x^*) = 0.8573$ instead of $2.9197$?

Let's dissect the components of the paper's objective:

$$f_{\text{dist}}(x) = \sum_{i=1}^3 a_i \left( x_i^T Q_i x_i - \mu_i^T x_i + c_i \right)$$

Under consensus, $x_i^j = w_j = z_j / d_j$, where $d = (3, 2, 2)$ are the asset degrees.

### 1. Quadratic Variance Term
For Asset 1 ($d_1 = 3$), $w_1 = z_1 / 3$.
Inside sub-portfolio 1, the variance term for Asset 1 is:
$$w_1^2 Q_{11} = \frac{z_1^2}{9} Q_{11}$$
Inside sub-portfolio 2, it is $\frac{z_1^2}{9} Q_{11}$.
Inside sub-portfolio 3, it is $\frac{z_1^2}{9} Q_{11}$.
Averaging across all three sub-portfolios with $a_i = 1/3$:
$$\frac{1}{3} \times \left( \frac{z_1^2}{9} Q_{11} + \frac{z_1^2}{9} Q_{11} + \frac{z_1^2}{9} Q_{11} \right) = \mathbf{\frac{1}{9} z_1^2 Q_{11}}$$

Instead of evaluating $z_1^2 Q_{11}$, the distributed objective evaluates **$\frac{1}{9}$ of that variance**!

Furthermore, sub-portfolio 2 only has assets $\{1, 2\}$, and sub-portfolio 3 only has $\{1, 3\}$.
The covariance between Asset 2 and Asset 3 ($\sigma_{2, 3}$) appears **only once** in sub-portfolio 1, and is completely ignored by the other two desks!

Total variance calculation:
- Real portfolio variance $z^T Q z$: **2.1449**
- Distributed average variance $\sum a_i x_i^T Q_i x_i$: **0.2653** (shrunk by factor of **8.1x**!)

### 2. Linear Return Term
$$\sum_{i=1}^3 a_i \mu_i^T x_i = \frac{1}{3} \mu^T z = \frac{0.2249}{3} = \mathbf{0.0750}$$
The return term is divided by exactly 3.

### 3. Management Cost Term
$$\sum_{i=1}^3 a_i c_i = \frac{1.0 + 0.6 + 0.4}{3} = \frac{2.0}{3} = \mathbf{0.6667}$$
The cost is reduced from $1.000$ to $0.6667$.

### Total Distributed Objective Assembly
$$f_{\text{dist}} = 0.2653 - 0.0750 + 0.6667 = \mathbf{0.8570}$$

The paper's reported value was **0.8573**.

The 70% reduction in objective value is not a discovery of free diversification. It is a mathematical consequence of:
1. Evaluating quadratic risk on fractional allocations scaled by $1/d_j^2$.
2. Ignoring cross-subportfolio covariances.
3. Lowering the effective constant cost from $1.0$ to $0.667$.

---

## 5. Controlled Experiment 3: The True Lifted Markowitz Problem

The paper asserts that the distributed model is a *"lifted representation that preserves the economic content of the centralized case"*.

What does a **true lifted representation** look like?

If we lift the decision variable from $z \in \mathbb{R}^3$ to $x \in \mathbb{R}^7$ such that $z = E^T x$, and enforce consensus $Lx = 0$ and budget $\mathbf{1}^T z = 1$, the mathematically equivalent objective MUST evaluate the true market risk on the underlying assets:

$$\min_{x \in \mathbb{R}^7} \quad (E^T x)^T Q (E^T x) - \mu^T (E^T x) + c$$

subject to:
$$\mathbf{1}^T (E^T x) = 1, \quad L x = 0, \quad x \ge 0$$

Let's solve this true lifted problem using CVXPY:

```python
import cvxpy as cp
from hypergraph_portfolio.hypergraph import HypergraphPortfolio

hp = HypergraphPortfolio(
    asset_names=["x1", "x2", "x3"],
    subportfolios=[["x1", "x2", "x3"], ["x1", "x2"], ["x1", "x3"]],
)

x = cp.Variable(7)
z = hp.E.T @ x

obj = cp.Minimize(cp.quad_form(z, Q) - mu @ z + c)
constraints = [
    cp.sum(z) == 1.0,
    hp.L @ x == 0,
    x >= 0,
]

prob = cp.Problem(obj, constraints)
prob.solve()

print(f"True Lifted Markowitz Optimum: {prob.value:.4f}")
print(f"Underlying Asset Weights z*:    {np.round(z.value, 4)}")
```

Output:
```
True Lifted Markowitz Optimum: 2.8408
Underlying Asset Weights z*:    [0.3217 0.2567 0.4217]
```

### Finding 3
When a lifted problem genuinely preserves the Markowitz objective:
- The optimal value is **2.8408** (identical to centralized).
- The optimal asset allocations are **(0.3217, 0.2567, 0.4217)** (identical to centralized).

This proves beyond all doubt:
**The paper's distributed problem is NOT a lifted reparameterization of Markowitz.**
It is a **genuinely different optimization problem**.

---

## 6. Controlled Experiment 4: The Topology Scaling Artifact

To demonstrate that the paper's objective drop is purely structural, let's run an experiment where we hold the financial assets, returns, and covariances completely fixed, but increase the number of identical sub-portfolios $K$ from 1 to 10:

```python
import cvxpy as cp
for K in [1, 2, 3, 5, 10]:
    # Duplicate into K identical sub-portfolios
    subportfolios = [["x1", "x2", "x3"] for _ in range(K)]
    hp = HypergraphPortfolio(asset_names=["x1", "x2", "x3"], subportfolios=subportfolios)

    x = cp.Variable(3 * K)
    sub_objs = [cp.quad_form(x[3*i:3*(i+1)], Q) - mu @ x[3*i:3*(i+1)] + (c / K) for i in range(K)]
    tot_obj = cp.sum(sub_objs) / float(K)

    prob = cp.Problem(cp.Minimize(tot_obj), [cp.sum(x) == 1.0, hp.L @ x == 0, x >= 0])
    prob.solve()
    z = hp.E.T @ x.value

    print(f"K = {K:<2} | Paper Objective = {prob.value:.4f} | True Portfolio Var = {z @ Q @ z:.4f}")
```

Output:
```
K = 1  | Paper Objective = 2.8408 | True Portfolio Var = 2.0436
K = 2  | Paper Objective = 0.9090 | True Portfolio Var = 2.0494
K = 3  | Paper Objective = 0.4919 | True Portfolio Var = 2.0591
K = 5  | Paper Objective = 0.2399 | True Portfolio Var = 2.0901
K = 10 | Paper Objective = 0.0986 | True Portfolio Var = 2.2356
```

Look at what happened:
- Simply by slicing the portfolio into 10 desks, the paper's objective collapses from **2.8408** down to **0.0986**!
- Yet the real financial portfolio variance remains $\approx 2.04$!

The reported performance gain is an artifact of the algebra.

---

## 7. Audit Synthesis: Reparameterization vs Genuinely Different Problem

We summarize our audit findings in the table below:

| Feature | Centralized Markowitz | True Lifted Markowitz | Paper's Distributed Problem |
| :--- | :--- | :--- | :--- |
| **Decision Variables** | $z \in \mathbb{R}^3$ | $x \in \mathbb{R}^7, z = E^T x$ | $x \in \mathbb{R}^7$ (local copies) |
| **Budget Constraint** | $\mathbf{1}^T z = 1$ | $\mathbf{1}^T E^T x = 1$ | $\mathbf{1}^T x = 1$ (sums local copies) |
| **Quadratic Risk** | $z^T Q z$ | $(E^T x)^T Q (E^T x)$ | $\sum a_i x_i^T Q_i x_i$ (no cross-cov) |
| **Effective Variance Scale** | $1.0$ | $1.0$ | $\approx 1 / d_j^2 \approx 0.12$ |
| **Cross-Desk Covariances** | Fully accounted | Fully accounted | Completely ignored |
| **Optimal Objective** | **2.8408** | **2.8408** | **0.8573** (artificial drop) |
| **Real Portfolio Risk on $Q$** | **2.0436** | **2.0436** | **2.1446** (sub-optimal) |

### Final Audit Verdict

1. **Is the paper's distributed formulation informationally equivalent to Markowitz?**
   **No.** It omits cross-subportfolio covariances and applies the budget constraint to local copies rather than total asset weights.
2. **Is it a reparameterization or a genuinely different problem?**
   It is a **genuinely different optimization problem**.
3. **Does decentralization provide a structural free lunch in risk reduction?**
   **No.** When the resulting asset allocation is measured against real market variance, it is slightly sub-optimal compared to the centralized Markowitz baseline.
4. **Does the paper still have value?**
   **Yes!** Its true contribution is **algorithmic and institutional**:
   - It provides an elegant hypergraph framework for multi-desk coordination.
   - Its continuous-time primal-dual ODE is mathematically sound and stable.
   - Its shadow prices $v^*$ offer a transparent mechanism for allocating capital and charging for shared exposures.

In Tutorial 10, we examine the paper's applied real-world example: 11 GICS sector ETFs.

---

## What You Should Now Understand

1. Why the paper's reported drop in objective value (0.8573 vs 2.8408) is a mathematical artifact of weight downscaling and omitted cross-covariances.
2. How to construct a true lifted Markowitz problem and prove equivalence.
3. How to evaluate any distributed allocation against the true financial market covariance matrix.

---

## Questions Worth Investigating

- If an institutional investor *wants* desks to ignore cross-desk correlations to maintain strategy independence, what is the optimal penalty for the resulting sub-optimality?
- How could the distributed objective be modified to include cross-subportfolio covariances through hypergraph edge laplacians?
