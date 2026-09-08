# Tutorial 03 — Consensus Constraints

In Tutorial 02, we constructed the hypergraph Laplacian matrix $L = I - E(E^T E)^{-1}E^T$.

Now we address the central constraint in the paper's distributed formulation:

$$L x = 0$$

Why does this single matrix-vector equation guarantee that all sub-portfolios agree on their shared investments? And what happens when they disagree?

---

## 1. Epistemic Status & Attribution

> [!NOTE]
> **What the Paper Explicitly Claims (Section 2.1)**:
> - The condition $Lx = 0$ enforces consistency across shared assets: $x_i^j = z_j$ for all sub-portfolios $i$ sharing asset $j$.
> - The consensus constraint couples the sub-portfolios through the hypergraph incidence structure.

> [!IMPORTANT]
> **Our Explanatory Interpretation**:
> - The null space of $L$ is exactly $\operatorname{Range}(E)$.
> - Since $E$ maps an underlying $|A|$-dimensional asset vector $z$ to stacked local copies, any vector in $\operatorname{Range}(E)$ must have identical values across all copies of the same asset.
> - The quadratic form $x^T L x = \frac{1}{2} \sum_{j=1}^{|A|} \frac{1}{|E_j|} \sum_{u, v \in E_j} (x_u - x_v)^2$ directly measures total pairwise disagreement across all shared asset copies!

> [!TIP]
> **New Experiments & Critiques Added**:
> - We implement a consensus audit tool [`check_consensus`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/src/hypergraph_portfolio/consensus.py) that decomposes the error vector $Lx$ by individual asset to pinpoint which desks disagree and by how much.

---

## 2. Intuition: The Mirror Room

Imagine three different portfolio managers standing in front of three mirrors, each holding a sign with their proposed allocation to Asset 1 (Tech):
- Manager 1 holds up: **0.15**
- Manager 2 holds up: **0.15**
- Manager 3 holds up: **0.15**

Because all three numbers match, there is zero disagreement.

Now suppose Manager 1 changes their sign to **0.50**:
- The group average becomes $\frac{0.50 + 0.15 + 0.15}{3} = 0.2667$.
- Discrepancy for Manager 1: $0.50 - 0.2667 = +0.2333$.
- Discrepancy for Manager 2: $0.15 - 0.2667 = -0.1167$.
- Discrepancy for Manager 3: $0.15 - 0.2667 = -0.1167$.

Notice that:
1. When all managers agree, every manager's difference from the group average is **zero**.
2. When any manager disagrees, the differences are **non-zero** and sum to zero.

This is precisely what $Lx$ calculates!

---

## 3. Tiny Example

Let's use our 7-node vector $x = [x_1^1, x_2^1, x_3^1, x_1^2, x_2^2, x_1^3, x_3^3]^T$.

### Case A: Perfect Agreement
Suppose the underlying asset weights are $w = (0.15, 0.10, 0.20)^T$.
Every copy of Asset 1 is $0.15$, every copy of Asset 2 is $0.10$, every copy of Asset 3 is $0.20$:

$$x_{\text{consistent}} = \begin{pmatrix} 0.15 \\ 0.10 \\ 0.20 \\ 0.15 \\ 0.10 \\ 0.15 \\ 0.20 \end{pmatrix}$$

Let's compute $L x_{\text{consistent}}$ using row 1 of $L$ (which has entries $\frac{2}{3}, 0, 0, -\frac{1}{3}, 0, -\frac{1}{3}, 0$):

$$(Lx)_1 = \frac{2}{3}(0.15) - \frac{1}{3}(0.15) - \frac{1}{3}(0.15) = 0.10 - 0.05 - 0.05 = 0$$

All 7 rows evaluate to 0! Thus $L x_{\text{consistent}} = \mathbf{0}$.

### Case B: Disagreement on Asset 1
Now change the first entry (Desk 1's copy of Asset 1) to $0.45$:

$$x_{\text{inconsistent}} = \begin{pmatrix} 0.45 \\ 0.10 \\ 0.20 \\ 0.15 \\ 0.10 \\ 0.15 \\ 0.20 \end{pmatrix}$$

Now compute $Lx$:
$$(Lx)_1 = \frac{2}{3}(0.45) - \frac{1}{3}(0.15) - \frac{1}{3}(0.15) = 0.30 - 0.05 - 0.05 = \mathbf{+0.20}$$
$$(Lx)_4 = -\frac{1}{3}(0.45) + \frac{2}{3}(0.15) - \frac{1}{3}(0.15) = -0.15 + 0.10 - 0.05 = \mathbf{-0.10}$$
$$(Lx)_6 = -\frac{1}{3}(0.45) - \frac{1}{3}(0.15) + \frac{2}{3}(0.15) = -0.15 - 0.05 + 0.10 = \mathbf{-0.10}$$

Notice:
$$(Lx) = \begin{pmatrix} +0.20 \\ 0 \\ 0 \\ -0.10 \\ 0 \\ -0.10 \\ 0 \end{pmatrix} \neq \mathbf{0}$$

The norm $\|Lx\|_2 = \sqrt{0.20^2 + (-0.10)^2 + (-0.10)^2} = \sqrt{0.06} \approx 0.245$.

---

## 4. Mathematics: Null Space and Consensus Subspace

### Theorem: Null Space Characterization
$$\operatorname{Null}(L) = \operatorname{Range}(E)$$

**Proof**:
Recall $L = I - E(E^T E)^{-1}E^T$.
1. Suppose $x \in \operatorname{Range}(E)$. Then $x = E z$ for some $z \in \mathbb{R}^{|A|}$.
   $$L x = (I - E(E^T E)^{-1}E^T) E z = E z - E (E^T E)^{-1} (E^T E) z = E z - E z = 0$$
   Thus $\operatorname{Range}(E) \subseteq \operatorname{Null}(L)$.
2. Suppose $L x = 0$. Then $(I - P_E) x = 0 \implies x = P_E x = E [(E^T E)^{-1} E^T x]$.
   Letting $z = (E^T E)^{-1} E^T x$, we have $x = E z$, so $x \in \operatorname{Range}(E)$.
   Thus $\operatorname{Null}(L) \subseteq \operatorname{Range}(E)$.
3. Therefore $\operatorname{Null}(L) = \operatorname{Range}(E)$. $\blacksquare$

### What Does $x = Ez$ Mean Component-Wise?
Since $E_{l, j} = 1$ if node $l$ represents asset $j$, the equation $x = E z$ states:

$$x_i^j = z_j \quad \text{for every sub-portfolio } i \text{ containing asset } j$$

In words: **every local copy of asset $j$ must equal the common scalar $z_j$**.
This is why $Lx = 0$ is the exact algebraic condition for consensus.

---

## 5. Executable Code

Let's test this in Python using [`consensus.py`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/src/hypergraph_portfolio/consensus.py):

```python
import numpy as np
from hypergraph_portfolio.hypergraph import HypergraphPortfolio
from hypergraph_portfolio.consensus import check_consensus, compute_consensus_discrepancies, project_onto_consensus

# Setup 3-subportfolio hypergraph
hp = HypergraphPortfolio(
    asset_names=["x1", "x2", "x3"],
    subportfolios=[["x1", "x2", "x3"], ["x1", "x2"], ["x1", "x3"]],
)

# 1. Consistent allocation
x_consistent = np.array([0.15, 0.10, 0.20, 0.15, 0.10, 0.15, 0.20])
is_cons, err = check_consensus(x_consistent, hp)
print(f"Consistent vector: is_consensus={is_cons}, ||Lx||={err:.6f}")

# 2. Inconsistent allocation (Desk 1 overweights x1)
x_inconsistent = np.array([0.45, 0.10, 0.20, 0.15, 0.10, 0.15, 0.20])
is_cons, err = check_consensus(x_inconsistent, hp)
print(f"Inconsistent vector: is_consensus={is_cons}, ||Lx||={err:.6f}")

# 3. Discrepancy analysis
disc = compute_consensus_discrepancies(x_inconsistent, hp)
print("\nDiscrepancy Breakdown:")
for asset, info in disc.items():
    print(f"  {asset}: values={info['values']}, max_spread={info['max_discrepancy']:.4f}")

# 4. Project onto consensus
x_projected = project_onto_consensus(x_inconsistent, hp)
print(f"\nProjected Vector: {np.round(x_projected, 4)}")
is_cons_proj, err_proj = check_consensus(x_projected, hp)
print(f"After projection: is_consensus={is_cons_proj}, ||Lx||={err_proj:.6f}")
```

---

## 6. Numerical Experiment & Output

Running the code yields:

```
Consistent vector: is_consensus=True, ||Lx||=0.000000
Inconsistent vector: is_consensus=False, ||Lx||=0.244949

Discrepancy Breakdown:
  x1: values=[0.45, 0.15, 0.15], max_spread=0.3000
  x2: values=[0.1, 0.1], max_spread=0.0000
  x3: values=[0.2, 0.2], max_spread=0.0000

Projected Vector: [0.25 0.1  0.2  0.25 0.1  0.25 0.2 ]
After projection: is_consensus=True, ||Lx||=0.000000
```

Notice:
- The projection replaces the conflicting copies of Asset 1 (`[0.45, 0.15, 0.15]`) with their exact average:
  $$\frac{0.45 + 0.15 + 0.15}{3} = 0.25$$
- Assets 2 and 3 had zero discrepancy and were left completely untouched!

---

## 7. Interpretation: Consensus as a Linear Subspace

In $\mathbb{R}^7$, the set of all possible local allocations forms a 7-dimensional space.
The consensus condition $Lx = 0$ defines a **3-dimensional linear subspace**:

$$\mathcal{C} = \{x \in \mathbb{R}^7 : Lx = 0\} = \operatorname{Range}(E)$$

Any point outside $\mathcal{C}$ represents an uncoordinated, inconsistent allocation. 
The vector $Lx$ points directly along the shortest path from the current allocation $x$ back to the consensus subspace $\mathcal{C}$.

In Tutorial 04, we combine this consensus constraint with the local sub-portfolio objectives to define the complete distributed optimization problem.

---

## What You Should Now Understand

1. Why $Lx = 0$ is mathematically equivalent to $x \in \operatorname{Range}(E)$.
2. How $Lx$ extracts the deviation of each local allocation copy from the mean across all sub-portfolios sharing that asset.
3. How to project any inconsistent vector onto the consensus subspace using $P_E = I - L$.

---

## Questions Worth Investigating

- If two desks have unequal capital weights ($a_1 \neq a_2$), should consensus force unweighted equality ($x_1^j = x_2^j$) or weighted equality?
- What happens if the consensus constraint is penalized in the objective ($\frac{\rho}{2} \|Lx\|^2$) rather than enforced as a hard equality constraint?
