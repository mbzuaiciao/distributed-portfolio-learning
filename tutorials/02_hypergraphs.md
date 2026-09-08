# Tutorial 02 — Overlapping Sub-Portfolios as a Hypergraph

In Tutorial 01, we saw that when sub-portfolios overlap, the same underlying security appears across multiple investment mandates simultaneously. 

In this tutorial, we introduce the mathematical framework used by Papastaikoudis, Watson, and Lestas (2025) to model these multi-way overlaps: **hypergraphs**.

---

## 1. Epistemic Status & Attribution

> [!NOTE]
> **What the Paper Explicitly Claims (Section 2.1)**:
> - The structural interdependencies across sub-portfolios are represented using a hypergraph $\mathcal{H} = (\mathcal{V}, \mathcal{E})$.
> - Node set $\mathcal{V}$ is partitioned as $\mathcal{V} = \{\mathcal{V}_1, \dots, \mathcal{V}_K\}$, where each node corresponds to a local variable component.
> - Hyperedge set $\mathcal{E}$ represents the underlying assets $|\mathcal{E}| = |A|$. Each hyperedge links all local copies of the same underlying asset.
> - Vertex degree matrix is $D_V = I_{p \times p}$.
> - Edge degree matrix is $D_E = E^T E = \operatorname{diag}(|E_1|, \dots, |E_{|A|}|)$.
> - The normalized hypergraph Laplacian is $L = D_V - E D_E^{-1} E^T = I - E(E^T E)^{-1} E^T$.

> [!IMPORTANT]
> **Our Explanatory Interpretation**:
> - Mathematically, $P_E = E(E^T E)^{-1} E^T$ is the **orthogonal projection matrix** onto the column space (range) of the incidence matrix $E$.
> - Consequently, $L = I - P_E$ is the orthogonal projection onto the **orthogonal complement** $\operatorname{Range}(E)^\perp$.
> - This provides an immediate, elegant geometric explanation for why $Lx = 0$ enforces consensus: a vector has zero projection onto $\operatorname{Range}(E)^\perp$ if and only if it lies entirely within $\operatorname{Range}(E)$.

> [!TIP]
> **New Experiments & Critiques Added**:
> - We prove and numerically verify all 4 algebraic properties: $L = L^T$, $L^2 = L$, eigenvalues $\in \{0, 1\}$, and $L E = 0$.

---

## 2. Intuition: Why Ordinary Graphs Are Insufficient

In an **ordinary graph** $G = (V, E)$, an edge is a pair of vertices $\{u, v\}$. It represents a binary relationship (dyad).

If an asset (say, Apple) is owned by Desk 1 and Desk 2, we can draw a standard edge between Desk 1 and Desk 2:

```
[Desk 1] --------- (Apple) --------- [Desk 2]
```

Now suppose **three** desks (Desk 1, Desk 2, Desk 3) own Apple. How do you represent this with ordinary edges?
- You could draw three pairwise edges: (1-2), (2-3), (1-3) forming a triangle (clique).
- But pairwise edges represent three separate bilateral agreements! If Desk 1 agrees with Desk 2, and Desk 2 agrees with Desk 3, you are modeling 3 pairwise constraints rather than **one shared underlying security**.
- In financial reality, Apple is not three bilateral agreements. Apple is a **single underlying asset** shared across a group of size 3.

A **hypergraph** generalizes graphs by allowing a "hyperedge" to connect **any number of vertices simultaneously**.

```
         Desk 1 (x_1^A)
              \
               \
    [ HYPEREDGE: Asset A ] ----- Desk 2 (x_2^A)
               /
              /
         Desk 3 (x_3^A)
```

A hyperedge is simply a subset of vertices. Here, the hyperedge for Asset A is $\{x_1^A, x_2^A, x_3^A\}$.

---

## 3. Tiny Hand-Verifiable Example

Let's return to our 3-asset, 3-subportfolio universe from Tutorial 01:
- Assets: $A = \{x_1, x_2, x_3\}$
- Sub-portfolios:
  - $\Sigma_1 = \{x_1, x_2, x_3\}$ (nodes 1, 2, 3)
  - $\Sigma_2 = \{x_1, x_2\}$ (nodes 4, 5)
  - $\Sigma_3 = \{x_1, x_3\}$ (nodes 6, 7)

### Vertex Set $\mathcal{V}$
There are $p = 3 + 2 + 2 = 7$ total local variable nodes:
- Node 1: $x_1^1$ (Asset 1 in SP1)
- Node 2: $x_2^1$ (Asset 2 in SP1)
- Node 3: $x_3^1$ (Asset 3 in SP1)
- Node 4: $x_1^2$ (Asset 1 in SP2)
- Node 5: $x_2^2$ (Asset 2 in SP2)
- Node 6: $x_1^3$ (Asset 1 in SP3)
- Node 7: $x_3^3$ (Asset 3 in SP3)

### Hyperedge Set $\mathcal{E}$
There are $|\mathcal{E}| = 3$ hyperedges (one for each underlying asset):
- Hyperedge 1 (Asset 1): connects nodes $\{1, 4, 6\}$ (degree = 3)
- Hyperedge 2 (Asset 2): connects nodes $\{2, 5\}$ (degree = 2)
- Hyperedge 3 (Asset 3): connects nodes $\{3, 7\}$ (degree = 2)

---

## 4. Mathematics: Matrices of the Hypergraph

### The Incidence Matrix $E$
The incidence matrix $E \in \mathbb{R}^{p \times |A|}$ has rows corresponding to local nodes and columns corresponding to underlying assets:

$$E_{l, j} = \begin{cases} 1 & \text{if local node } l \text{ represents asset } j \\ 0 & \text{otherwise} \end{cases}$$

For our 7-node, 3-asset example:

$$E = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
\hline
1 & 0 & 0 \\
0 & 1 & 0 \\
\hline
1 & 0 & 0 \\
0 & 0 & 1
\end{pmatrix}$$

### Edge Degree Matrix $D_E$
Multiplying $E^T E$ counts how many times each asset appears across sub-portfolios:

$$D_E = E^T E = \begin{pmatrix}
3 & 0 & 0 \\
0 & 2 & 0 \\
0 & 0 & 2
\end{pmatrix} = \operatorname{diag}(3, 2, 2)$$

Its inverse is simply:
$$D_E^{-1} = (E^T E)^{-1} = \operatorname{diag}\left(\frac{1}{3}, \frac{1}{2}, \frac{1}{2}\right)$$

### Vertex Degree Matrix $D_V$
Because each local variable in a sub-portfolio represents exactly one asset, each row of $E$ has exactly one 1. Thus:
$$D_V = I_{7 \times 7}$$

### The Hypergraph Laplacian $L$
The normalized hypergraph Laplacian is defined as:

$$L = D_V - E D_E^{-1} E^T = I - E (E^T E)^{-1} E^T$$

Let's compute the block $P_E = E (E^T E)^{-1} E^T$:
For rows corresponding to Asset 1 (nodes 1, 4, 6), $(P_E)_{i, j} = \frac{1}{3}$.
For rows corresponding to Asset 2 (nodes 2, 5), $(P_E)_{i, j} = \frac{1}{2}$.
For rows corresponding to Asset 3 (nodes 3, 7), $(P_E)_{i, j} = \frac{1}{2}$.

Subtracting from identity $I$ yields the paper's exact matrix $L$ from page 11:

$$L = \begin{pmatrix}
\frac{2}{3} & 0 & 0 & -\frac{1}{3} & 0 & -\frac{1}{3} & 0 \\
0 & \frac{1}{2} & 0 & 0 & -\frac{1}{2} & 0 & 0 \\
0 & 0 & \frac{1}{2} & 0 & 0 & 0 & -\frac{1}{2} \\
-\frac{1}{3} & 0 & 0 & \frac{2}{3} & 0 & -\frac{1}{3} & 0 \\
0 & -\frac{1}{2} & 0 & 0 & \frac{1}{2} & 0 & 0 \\
-\frac{1}{3} & 0 & 0 & -\frac{1}{3} & 0 & \frac{2}{3} & 0 \\
0 & 0 & -\frac{1}{2} & 0 & 0 & 0 & \frac{1}{2}
\end{pmatrix}$$

---

## 5. Executable Code

Let's build this hypergraph and inspect all matrices using [`HypergraphPortfolio`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/src/hypergraph_portfolio/hypergraph.py):

```python
import numpy as np
from hypergraph_portfolio.hypergraph import HypergraphPortfolio

assets = ["x1", "x2", "x3"]
subportfolios = [
    ["x1", "x2", "x3"],  # SP1
    ["x1", "x2"],        # SP2
    ["x1", "x3"],        # SP3
]

hp = HypergraphPortfolio(asset_names=assets, subportfolios=subportfolios)

print("Incidence Matrix E (7 x 3):")
print(hp.E.astype(int))

print("\nEdge Degree Matrix D_E = E^T E:")
print(hp.D_E.astype(int))

print("\nHypergraph Laplacian Matrix L (7 x 7):")
print(np.round(hp.L, 4))
```

---

## 6. Numerical Experiment: Verifying Algebraic Properties

Lemma 5 in the paper's Appendix proves that $L$ is an **orthogonal projection matrix**. Let's verify all four fundamental properties:

```python
props = hp.verify_laplacian_properties()
for prop, is_valid in props.items():
    print(f"Property [{prop}]: {is_valid}")

eigenvalues = np.linalg.eigvalsh(hp.L)
print(f"Eigenvalues of L: {np.round(eigenvalues, 4)}")
```

Output:
```
Property [symmetric]: True
Property [idempotent]: True
Property [spectrum_in_0_1]: True
Property [nullspace_contains_range_E]: True
Eigenvalues of L: [0. 0. 0. 1. 1. 1. 1.]
```

Notice:
1. **$L = L^T$**: The Laplacian is symmetric.
2. **$L^2 = L$**: $L$ is an idempotent projection matrix.
3. **Eigenvalues**: Exactly 3 eigenvalues are equal to 0, and 4 eigenvalues are equal to 1.
   - The 3 zero eigenvalues correspond to the 3 underlying assets!
   - The rank of $E$ is 3, so $\dim(\operatorname{Range}(E)) = 3$.
   - The rank of $L$ is $7 - 3 = 4$.

---

## 7. Interpretation: The Projection Geometry

What does multiplying a vector $x$ by $L$ actually do?

$$L x = (I - P_E) x = x - P_E x$$

Where $P_E x = E (E^T E)^{-1} E^T x$.

Let's look at what $P_E x$ computes for each node:
- For Asset 1: nodes 1, 4, 6 hold local allocations $x_1^1, x_1^2, x_1^3$.
- Row 1 of $P_E x$ is:
  $$(P_E x)_1 = \frac{1}{3} (x_1^1 + x_1^2 + x_1^3)$$
  It is the **unweighted arithmetic mean of all local copies of Asset 1**!

Therefore, the $i$-th row of $Lx$ is:
$$(Lx)_i = x_i - \text{mean of all copies of that asset}$$

This reveals the beauty of the hypergraph Laplacian:
Multiplying by $L$ subtracts the consensus average from each local copy.
Therefore:
$$Lx = 0 \iff \text{every local copy equals its group average} \iff \text{perfect consensus!}$$

In Tutorial 03, we explore this consensus constraint in depth.

---

## What You Should Now Understand

1. Why ordinary pairwise graphs cannot cleanly model multi-portfolio shared assets, and why hypergraphs are required.
2. How to construct the incidence matrix $E$, edge degree matrix $D_E$, and normalized Laplacian $L$.
3. The geometric meaning of $L$: it is an orthogonal projector onto the orthogonal complement of the consensus subspace.

---

## Questions Worth Investigating

- If an asset appears in 10 sub-portfolios, what is its edge degree, and what will the diagonal entries of $L$ for that asset be?
- What happens to the eigenvalues of $L$ if we add a 4th sub-portfolio? Does the number of zero eigenvalues change?
