# Tutorial 01 — Why Distributed Portfolio Optimization?

In Tutorial 00, we saw that standard Markowitz portfolio optimization assumes a single, centralized decision-maker who optimizes a single flat vector $x \in \mathbb{R}^n$ using a global covariance matrix $Q$.

In this tutorial, we explore the core institutional and mathematical motivation for replacing one flat portfolio with a network of **overlapping sub-portfolios**.

---

## 1. Epistemic Status & Attribution

> [!NOTE]
> **What the Paper Explicitly Claims (Sections 1 & 2)**:
> - Institutional asset allocation is inherently modular: investors construct portfolios around themes, sector strategies, or mandates.
> - By decomposing the asset universe into subsets of the powerset $\mathcal{P}(A)$, investors can operate specialized sub-portfolios.
> - Shared assets across overlapping sub-portfolios require a coupling and consensus mechanism to maintain consistency.

> [!IMPORTANT]
> **Our Explanatory Interpretation**:
> - Distributed optimization is not just a computational trick; it mirrors institutional reality. In a multi-manager hedge fund or asset management firm, the Tech desk, the Multi-Asset desk, and the Macro desk each run their own local optimization models.
> - The fundamental challenge is coordinating their overlapping holdings without forcing every desk to merge into a monolithic committee.

> [!TIP]
> **New Experiments & Critiques Added**:
> - We demonstrate numerically what happens when overlapping desks optimize independently without coordination: extreme disagreement and budget violations occur.

---

## 2. Intuition: The Multi-Desk Dilemma

Consider an investment institution with three specialized desks:
1. **Desk 1 (All-Cap Multi-Asset)**: Authorized to trade Assets $\{A, B, C\}$.
2. **Desk 2 (Growth & Innovation)**: Authorized to trade Assets $\{A, B\}$.
3. **Desk 3 (Value & Dividends)**: Authorized to trade Assets $\{A, C\}$.

Notice that **Asset A** appears on all three desks!
- The Multi-Asset manager might want 40% in Asset A based on macro trends.
- The Growth manager might want 70% in Asset A based on momentum.
- The Value manager might want 15% in Asset A based on valuation multiples.

If each manager optimizes their mandate independently:
- Who decides how many total shares of Asset A the firm actually owns?
- If Desk 2 buys Asset A while Desk 3 shorts Asset A, the firm pays double transaction fees to trade against itself!
- If the firm enforces a global risk budget, uncoordinated local allocations will inevitably violate the firm-wide capital limit.

---

## 3. Tiny Example: Overlapping Subsets

Let our asset universe be $A = \{A, B, C\}$.
We define $K = 3$ sub-portfolios:

$$\Sigma_1 = \{A, B, C\}$$
$$\Sigma_2 = \{A, B\}$$
$$\Sigma_3 = \{A, C\}$$

Let us write down the decision variables for each sub-portfolio:
- Sub-portfolio 1 chooses: $x_1 = (x_1^A, x_1^B, x_1^C)^T \in \mathbb{R}^3$
- Sub-portfolio 2 chooses: $x_2 = (x_2^A, x_2^B)^T \in \mathbb{R}^2$
- Sub-portfolio 3 chooses: $x_3 = (x_3^A, x_3^C)^T \in \mathbb{R}^2$

Notice what has happened to our variable space:
- Centralized universe: 3 assets $\implies$ 3 variables $(z_A, z_B, z_C)$.
- Distributed universe: 3 local vectors $\implies 3 + 2 + 2 = 7$ local variables!

```
Subportfolio 1 (Multi-Asset):   [ x_1^A,   x_1^B,   x_1^C ]
Subportfolio 2 (Growth):        [ x_2^A,   x_2^B ]
Subportfolio 3 (Value):         [ x_3^A,            x_3^C ]
                                    ▲        ▲        ▲
Shared Exposure Couplings:        All 3    SP1+SP2  SP1+SP3
```

Asset A is duplicated across 3 sub-portfolios. Asset B is duplicated across 2. Asset C is duplicated across 2.

---

## 4. Mathematics: Why Independent Optimization Fails

Suppose each sub-portfolio $i$ solves its own independent Markowitz problem:

$$\min_{x_i} \quad x_i^T Q_i x_i - \mu_i^T x_i \quad \text{s.t.} \quad \mathbf{1}^T x_i = 1, \quad x_i \ge 0$$

where $Q_i$ is the sub-matrix of $Q$ corresponding to the assets in $\Sigma_i$.

Because $Q_i$ contains only a subset of the assets and misses cross-covariances with excluded securities, each sub-portfolio faces a different localized risk landscape:
- In $\Sigma_2$, Asset A is paired only with B. If $\operatorname{Cov}(A, B)$ is high, Desk 2 will down-weight A.
- In $\Sigma_3$, Asset A is paired only with C. If $\operatorname{Cov}(A, C)$ is low, Desk 3 will load up on A to hedge C.

The resulting local allocations will have:

$$x_1^A \neq x_2^A \neq x_3^A$$

This produces **inconsistency**. The institution cannot determine the firm-wide cost of capital, cannot structure clean modular products, and cannot evaluate whether the overall portfolio is risk-efficient.

---

## 5. Executable Code: Demonstrating the Disagreement

Let's simulate three independent local optimizations using our 3-asset parameters:

```python
import numpy as np
from hypergraph_portfolio.mean_variance import MeanVariancePortfolio

Q = np.array([
    [4.1294, 1.4527, 0.9273],
    [1.4527, 4.4150, 1.1068],
    [0.9273, 1.1068, 3.3435]
])
mu = np.array([0.3, 0.25, 0.1])

# Subportfolio 1: {A, B, C}
sp1 = MeanVariancePortfolio(Q=Q, mu=mu, budget=1.0)
x1_opt, _ = sp1.solve()

# Subportfolio 2: {A, B}
sp2 = MeanVariancePortfolio(Q=Q[:2, :2], mu=mu[:2], budget=1.0)
x2_opt, _ = sp2.solve()

# Subportfolio 3: {A, C}
sp3 = MeanVariancePortfolio(Q=Q[[0, 2]][:, [0, 2]], mu=mu[[0, 2]], budget=1.0)
x3_opt, _ = sp3.solve()

print("Independent Allocations to Asset A:")
print(f"  Subportfolio 1: x_1^A = {x1_opt[0]:.4f}")
print(f"  Subportfolio 2: x_2^A = {x2_opt[0]:.4f}")
print(f"  Subportfolio 3: x_3^A = {x3_opt[0]:.4f}")
print(f"  Maximum Disagreement: {max(x1_opt[0], x2_opt[0], x3_opt[0]) - min(x1_opt[0], x2_opt[0], x3_opt[0]):.4f}")
```

---

## 6. Numerical Experiment & Verification

Running the script reveals the conflict:

```
Independent Allocations to Asset A:
  Subportfolio 1: x_1^A = 0.3217
  Subportfolio 2: x_2^A = 0.5190
  Subportfolio 3: x_3^A = 0.3703
  Maximum Disagreement: 0.1973
```

- Desk 2 wants **51.9%** in Asset A.
- Desk 1 wants **32.2%** in Asset A.
- Desk 3 wants **37.0%** in Asset A.

The desks disagree by nearly **20% of their total capital** on Asset A alone!

Furthermore, each desk allocated 100% of its own budget. If all three desks deploy capital, the firm has deployed 300% leverage without coordination.

---

## 7. Interpretation: The Need for an Interaction Geometry

To make modular sub-portfolios viable, we need two things:
1. **A Global Budget Constraint**: $\sum_{i=1}^K \sum_{l} x_i^l = \text{Budget}$ to control overall firm leverage.
2. **A Consistency / Consensus Mechanism**: A mathematical constraint forcing the local copies of Asset A to agree:
   $$x_1^A = x_2^A = x_3^A$$

How can we express this consistency requirement compactly for any arbitrary collection of overlapping sub-portfolios?

Ordinary graphs can only connect pairs of nodes (edges of size 2). But Asset A is shared by **three** sub-portfolios simultaneously! 

To model relationships where an asset is shared across three, four, or dozens of portfolios, ordinary graphs fail. We need a **hypergraph**. We construct this in Tutorial 02.

---

## What You Should Now Understand

1. Why financial institutions naturally operate with modular, overlapping sub-portfolios rather than a single flat portfolio.
2. How overlapping portfolios duplicate asset decision variables into local copies ($x_1^j, x_2^j, \dots$).
3. Why independent local optimization creates severe allocation disagreements and budget violations.

---

## Questions Worth Investigating

- If we force $x_1^A = x_2^A = x_3^A$, what is the economic cost of that constraint?
- If two desks have fundamentally different return views on Asset A ($\mu_1^A \neq \mu_2^A$), who bears the penalty when they are forced to agree?


# ==t01.py

Yes. The key purpose of this code is to demonstrate the **consistency problem caused by overlapping subportfolios**.

You have three assets:

* `A`
* `B`
* `C`

and three portfolio managers/subportfolios:

$$
S_1=\{A,B,C\},\qquad S_2=\{A,B\},\qquad S_3=\{A,C\}.
$$

Notice that **asset A appears in all three subportfolios**. The code lets each subportfolio optimize independently and then asks:

> Do they independently choose the same allocation for A?

Usually, they don't.

### 1. Define the data

```python
import numpy as np
from hypergraph_portfolio.mean_variance import MeanVariancePortfolio
```

`numpy` handles the vectors and matrices. `MeanVariancePortfolio` is your project's class that solves the portfolio optimization problem.

Then:

```python
Q = np.array([
    [4.1294, 1.4527, 0.9273],
    [1.4527, 4.4150, 1.1068],
    [0.9273, 1.1068, 3.3435]
])
```

`Q` is the **risk/covariance-like matrix** for assets \(A,B,C\).

You can read it as:

$$
Q=
\begin{bmatrix}
4.1294 & 1.4527 & 0.9273\\
1.4527 & 4.4150 & 1.1068\\
0.9273 & 1.1068 & 3.3435
\end{bmatrix}.
$$

The diagonal entries describe the individual risk contributions:

* `Q[0,0] = 4.1294` → A
* `Q[1,1] = 4.4150` → B
* `Q[2,2] = 3.3435` → C

The off-diagonal entries describe how the assets interact/covary.

Then:

```python
mu = np.array([0.3, 0.25, 0.1])
```

These are the expected returns:

$$
\mu =
\begin{bmatrix}
0.30\\
0.25\\
0.10
\end{bmatrix}
$$

so:

```text
A → 0.30
B → 0.25
C → 0.10
```

A has the highest expected return.

---

### 2. Subportfolio 1 gets all three assets

```python
sp1 = MeanVariancePortfolio(Q=Q, mu=mu, budget=1.0)
x1_opt, _ = sp1.solve()
```

This manager solves a portfolio optimization problem using:

$$
\{A,B,C\}.
$$

The budget is:

```python
budget=1.0
```

so, roughly speaking, the manager has one unit of capital to allocate.

If:

```python
x1_opt
```

were, for example,

```text
[0.40, 0.35, 0.25]
```

that would mean:

$$
x_1^A=0.40,\quad
x_1^B=0.35,\quad
x_1^C=0.25.
$$

The important quantity for this experiment is:

```python
x1_opt[0]
```

because index `0` corresponds to asset A.

---

### 3. Subportfolio 2 only gets A and B

```python
sp2 = MeanVariancePortfolio(
    Q=Q[:2, :2],
    mu=mu[:2],
    budget=1.0
)
```

This syntax is worth unpacking.

```python
Q[:2, :2]
```

means:

> take rows 0 and 1, and columns 0 and 1.

So:

$$
Q[:2,:2]
=
\begin{bmatrix}
4.1294 & 1.4527\\
1.4527 & 4.4150
\end{bmatrix}.
$$

This is just the part of the covariance/risk matrix concerning A and B.

Similarly:

```python
mu[:2]
```

gives:

```text
[0.3, 0.25]
```

so manager 2 only knows about:

$$
\{A,B\}.
$$

Then:

```python
x2_opt, _ = sp2.solve()
```

might produce something like:

```text
[0.55, 0.45]
```

and again,

```python
x2_opt[0]
```

means this manager's allocation to A.

---

### 4. Subportfolio 3 gets A and C

This line looks more complicated:

```python
Q[[0, 2]][:, [0, 2]]
```

because A and C are **not adjacent** in the original matrix.

Their indices are:

```text
A → 0
B → 1
C → 2
```

First:

```python
Q[[0, 2]]
```

selects rows `0` and `2`:

$$
\begin{bmatrix}
4.1294 & 1.4527 & 0.9273\\
0.9273 & 1.1068 & 3.3435
\end{bmatrix}.
$$

Then:

```python
[:, [0, 2]]
```

means:

> keep every selected row (`:`), but only columns 0 and 2.

Therefore you end up with:

$$
\begin{bmatrix}
4.1294 & 0.9273\\
0.9273 & 3.3435
\end{bmatrix}.
$$

That's precisely the A–C submatrix.

Likewise:

```python
mu[[0, 2]]
```

produces:

```text
[0.3, 0.1]
```

So:

```python
sp3 = MeanVariancePortfolio(...)
x3_opt, _ = sp3.solve()
```

optimizes only over:

$$
\{A,C\}.
$$

Again:

```python
x3_opt[0]
```

is A's allocation.

---

## 5. Now comes the point of the experiment

The code prints:

```python
print(f"  Subportfolio 1: x_1^A = {x1_opt[0]:.4f}")
print(f"  Subportfolio 2: x_2^A = {x2_opt[0]:.4f}")
print(f"  Subportfolio 3: x_3^A = {x3_opt[0]:.4f}")
```

Suppose you got:

```text
Subportfolio 1: x_1^A = 0.30
Subportfolio 2: x_2^A = 0.45
Subportfolio 3: x_3^A = 0.60
```

Then you have three different answers to the question:

> **How much A should we hold?**

Manager 1 says:

$$
x_1^A=0.30
$$

Manager 2 says:

$$
x_2^A=0.45
$$

Manager 3 says:

$$
x_3^A=0.60.
$$

That's the **overlap consistency problem**.

And this connects directly to what we were discussing about the paper.

---

## Why would they choose different amounts of A?

Because A is being evaluated in a **different local context** each time.

Manager 1 asks:

> How much A should I own when I can diversify against both B and C?

Manager 2 asks:

> How much A should I own when my only alternative is B?

Manager 3 asks:

> How much A should I own when my only alternative is C?

Those are different optimization problems.

So even though the underlying asset A is identical, its optimal allocation can change depending on which other assets are available.

That's exactly why overlap creates a problem.

---

## 6. Maximum disagreement

Finally:

```python
max(x1_opt[0], x2_opt[0], x3_opt[0]) -
min(x1_opt[0], x2_opt[0], x3_opt[0])
```

finds:

$$
\max(x_1^A,x_2^A,x_3^A)
-
\min(x_1^A,x_2^A,x_3^A).
$$

Using my made-up example:

$$
0.60-0.30=0.30.
$$

So the managers disagree by as much as **30 percentage points** on A.

The `:.4f` here:

```python
{x1_opt[0]:.4f}
```

just means:

> print the floating-point number with four decimal places.

---

### The deeper idea

Conceptually, your program is doing this:

```text
                  Global assets
                  {A, B, C}
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
     {A,B,C}        {A,B}         {A,C}
       SP1            SP2           SP3
        │              │             │
        ▼              ▼             ▼
      x₁ᴬ            x₂ᴬ           x₃ᴬ

         possibly all different
```

If these are supposed to describe **one underlying portfolio**, you don't really want three different versions of A.

You want a consistency condition such as:

$$
\boxed{x_1^A=x_2^A=x_3^A}
$$

That equality constraint is the thing the hypergraph/decentralized optimization machinery is ultimately trying to enforce.

So this little program is essentially a **counterexample** showing why you cannot just solve all overlapping subportfolios independently and assume they'll automatically agree.

