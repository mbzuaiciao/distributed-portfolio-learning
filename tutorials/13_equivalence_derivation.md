# Tutorial 13 — Mathematical Derivation of Objective Equivalence

In Section 4 of *“Distributed Portfolio Optimization & Decentralized Pricing”*, the authors put forward two foundational assertions:
1. That the decentralized and centralized formulations are **informationally equivalent and aggregation-consistent**, representing a **lifted representation** of the centralized problem (Section 4.1).
2. That the decentralized formulation achieves **substantially lower objective values** (e.g., $0.8573$ and $0.7609$ versus $2.8408$), demonstrating the structural risk-reducing superiority of hypergraph-based decentralization (Section 4.2).

In this mathematical note, we audit these claims by analytically substituting the consensus constraint into the distributed objective to derive its exact reduced quadratic form.

---

## 1. The Claim Being Tested

Let $A$ be a universe of $m$ financial assets with expected return vector $\mu \in \mathbb{R}^m$, symmetric positive definite covariance matrix $Q \in \mathbb{R}^{m \times m}$, and scalar management cost $c \in \mathbb{R}$.

The centralized Markowitz mean-variance optimization problem is:

$$\min_{w \in \mathbb{R}^m} \quad J(w) = w^T Q w - \mu^T w + c \quad \text{s.t.} \quad \mathbf{1}^T w = 1, \quad w \ge 0 \tag{1}$$

The paper constructs $K$ overlapping sub-portfolios $\Sigma_1, \dots, \Sigma_K \subseteq A$ with variable dimensions $p_i = |\Sigma_i|$ and stacked decision vector $x = [x_1, \dots, x_K]^T \in \mathbb{R}^p$, where $p = \sum_{i=1}^K p_i$.

The paper's distributed optimization problem is:

$$\min_{x \in \mathbb{R}^p} \quad F(x) = \sum_{i=1}^K a_i \left( x_i^T Q_i x_i - \mu_i^T x_i + c_i \right) \tag{2}$$

subject to:

$$\mathbf{1}^T x = 1 \tag{3a}$$
$$L x = 0 \tag{3b}$$
$$x \ge 0 \tag{3c}$$

Where:
- $Q_i$ is the principal submatrix of $Q$ for assets in $\Sigma_i$.
- $\mu_i$ is the subvector of $\mu$ for assets in $\Sigma_i$.
- $a_i \ge 0$ are weights on the unit simplex ($\sum_{i=1}^K a_i = 1$).
- $L = I - E(E^T E)^{-1}E^T$ is the normalized hypergraph Laplacian.

**Core Research Question**:
Is problem (2)–(3) a true **lifted reparameterization** of problem (1), or is it a **genuinely different optimization problem**?

---

## 2. Consensus Parameterizations

Recall from Lemma 5 that $L = I - P_E$ is the orthogonal projection onto $\operatorname{Range}(E)^\perp$. Therefore:

$$L x = 0 \iff x \in \operatorname{Range}(E)$$

Because the columns of the incidence matrix $E \in \mathbb{R}^{p \times m}$ correspond to the $m$ underlying assets, any feasible consensus vector $x$ can be parameterized in two distinct, mathematically precise ways:

### Parameterization 1: Shared Local Copy Values ($z \in \mathbb{R}^m$)
Every local copy of asset $j$ across all sub-portfolios is assigned the common value $z_j$:

$$x = E z \tag{4}$$

Under this parameterization, the global budget constraint (3a) becomes:

$$\mathbf{1}^T x = \mathbf{1}^T E z = (E^T \mathbf{1})^T z = d^T z = \sum_{j=1}^m d_j z_j = 1 \tag{5}$$

where $d_j = |E_j| = (E^T E)_{j, j}$ is the degree of asset $j$ (the number of sub-portfolios containing asset $j$).

### Parameterization 2: Total Economic Asset Weights ($w \in \mathbb{R}^m$)
In financial reality, an investor's total holding in asset $j$ is the sum of all local copies across desks:

$$w_j = \sum_{i: j \in \Sigma_i} x_i^j = (E^T x)_j \implies w = E^T x \tag{6}$$

Under consensus ($x = Ez$), $w = E^T E z = D_E z$. Inverting this relationship:

$$z = D_E^{-1} w \implies x = E D_E^{-1} w \tag{7}$$

Here, each local copy holds $x_i^j = \frac{w_j}{d_j}$.
The budget constraint $\mathbf{1}^T x = 1$ is equivalent to:

$$\mathbf{1}^T w = \sum_{j=1}^m w_j = 1 \tag{8}$$

Parameterization (7) represents the actual asset weights held by an investor with a normalized \$1.00 portfolio.

---

## 3. Direct Substitution into the Distributed Objective

We now substitute the consensus condition $x = E D_E^{-1} w$ directly into the distributed objective $F(x)$.

Let $E_i \in \mathbb{R}^{p_i \times m}$ denote the row-block of the incidence matrix corresponding to sub-portfolio $i$.
Then the local allocation vector is:

$$x_i = E_i D_E^{-1} w$$

Substituting $x_i$ into the sub-portfolio quadratic function $f_i(x_i)$:

$$f_i(x_i) = \left( E_i D_E^{-1} w \right)^T Q_i \left( E_i D_E^{-1} w \right) - \mu_i^T \left( E_i D_E^{-1} w \right) + c_i$$
$$= w^T \left( D_E^{-1} E_i^T Q_i E_i D_E^{-1} \right) w - \left( D_E^{-1} E_i^T \mu_i \right)^T w + c_i$$

Summing across all $K$ sub-portfolios with weights $a_i$:

$$F(E D_E^{-1} w) = \sum_{i=1}^K a_i f_i(x_i) = w^T Q_{\text{eff}}^{(w)} w - (\mu_{\text{eff}}^{(w)})^T w + c_{\text{eff}} \tag{9}$$

Where:

$$Q_{\text{eff}}^{(w)} = D_E^{-1} \left( \sum_{i=1}^K a_i E_i^T Q_i E_i \right) D_E^{-1} \tag{10}$$

$$\mu_{\text{eff}}^{(w)} = D_E^{-1} \left( \sum_{i=1}^K a_i E_i^T \mu_i \right) \tag{11}$$

$$c_{\text{eff}} = \sum_{i=1}^K a_i c_i \tag{12}$$

This is the exact, closed-form reduced objective governing the paper's distributed problem on the consensus subspace.

---

## 4. Entry-by-Entry Analysis of the Effective Quadratic Form

Let us inspect the individual matrix entries $(Q_{\text{eff}}^{(w)})_{j, k}$ and return entries $(\mu_{\text{eff}}^{(w)})_j$.

Let $S_{j, k} = \{i \in \{1, \dots, K\} : j \in \Sigma_i \text{ and } k \in \Sigma_i\}$ denote the set of sub-portfolios that contain **both** asset $j$ and asset $k$.

By direct matrix expansion:

$$(Q_{\text{eff}}^{(w)})_{j, k} = \left( \frac{\sum_{i \in S_{j, k}} a_i}{d_j d_k} \right) Q_{j, k} \tag{13}$$

$$(\mu_{\text{eff}}^{(w)})_j = \left( \frac{\sum_{i: j \in \Sigma_i} a_i}{d_j} \right) \mu_j \tag{14}$$

When sub-portfolio weights are uniform ($a_i = 1/K$):

### Diagonal Elements (Asset Variances, $j = k$)
Because asset $j$ appears in exactly $d_j$ sub-portfolios, $|S_{j, j}| = d_j$. Thus:

$$(Q_{\text{eff}}^{(w)})_{j, j} = \left( \frac{d_j \cdot \frac{1}{K}}{d_j^2} \right) Q_{j, j} = \mathbf{\frac{1}{K \, d_j} Q_{j, j}} \tag{15}$$

### Off-Diagonal Elements (Asset Covariances, $j \neq k$)
Let $n_{j, k} = |S_{j, k}|$ be the number of sub-portfolios sharing both assets:

$$(Q_{\text{eff}}^{(w)})_{j, k} = \mathbf{\frac{n_{j, k}}{K \, d_j d_k} Q_{j, k}} \tag{16}$$

### Expected Returns
$$(\mu_{\text{eff}}^{(w)})_j = \left( \frac{d_j \cdot \frac{1}{K}}{d_j} \right) \mu_j = \mathbf{\frac{1}{K} \mu_j} \tag{17}$$

---

## 5. Term-by-Term Comparison with Centralized Markowitz

Let us evaluate formulas (15)–(17) for the 3-asset setup from Section 3.2 of the paper:
- $K = 3$ sub-portfolios.
- Degrees: $d = (3, 2, 2)$ for Assets $\{1, 2, 3\}$.
- Objective weights: $a = (1/3, 1/3, 1/3)$.

### Variance Comparison:
- **Asset 1**: $Q_{11} = 4.1294 \implies (Q_{\text{eff}})_{11} = \frac{1}{3 \times 3} Q_{11} = \frac{1}{9}(4.1294) = \mathbf{0.4588}$ (Ratio: **0.1111**).
- **Asset 2**: $Q_{22} = 4.4150 \implies (Q_{\text{eff}})_{22} = \frac{1}{3 \times 2} Q_{22} = \frac{1}{6}(4.4150) = \mathbf{0.7358}$ (Ratio: **0.1667**).
- **Asset 3**: $Q_{33} = 3.3435 \implies (Q_{\text{eff}})_{33} = \frac{1}{3 \times 2} Q_{33} = \frac{1}{6}(3.3435) = \mathbf{0.5573}$ (Ratio: **0.1667**).

### Covariance Comparison:
- **Assets (1, 2)**: Co-occur in $\Sigma_1$ and $\Sigma_2$ ($n_{12} = 2$).
  $$(Q_{\text{eff}})_{12} = \frac{2}{3 \times 3 \times 2} Q_{12} = \frac{1}{9} Q_{12} \quad (\text{Ratio: } \mathbf{0.1111})$$
- **Assets (1, 3)**: Co-occur in $\Sigma_1$ and $\Sigma_3$ ($n_{13} = 2$).
  $$(Q_{\text{eff}})_{13} = \frac{2}{3 \times 3 \times 2} Q_{13} = \frac{1}{9} Q_{13} \quad (\text{Ratio: } \mathbf{0.1111})$$
- **Assets (2, 3)**: Co-occur **only** in $\Sigma_1$ ($n_{23} = 1$).
  $$(Q_{\text{eff}})_{23} = \frac{1}{3 \times 2 \times 2} Q_{23} = \frac{1}{12} Q_{23} \quad (\text{Ratio: } \mathbf{0.0833})$$

### Return Comparison:
$$(\mu_{\text{eff}})_j = \frac{1}{3} \mu_j \quad \text{for all } j \quad (\text{Ratio: } \mathbf{0.3333})$$

### Fixed Management Cost:
$$c_{\text{eff}} = \frac{1.0 + 0.6 + 0.4}{3} = \mathbf{0.6667} \quad (\text{vs } c = \mathbf{1.0000})$$

---

## 6. Construction of the True Lifted Equivalent

If one desires a distributed or lifted formulation that is **provably equivalent** to the centralized Markowitz problem, how must the lifted objective be defined?

For any consensus allocation $x \in \mathbb{R}^p$ satisfying $Lx = 0$, the total asset weight is $w = E^T x$.
To ensure that $F_{\text{lifted}}(x) \equiv J(w)$ for all feasible allocations, we define:

$$\tilde{Q}_{\text{lifted}} = E Q E^T \in \mathbb{R}^{p \times p}$$
$$\tilde{\mu}_{\text{lifted}} = E \mu \in \mathbb{R}^p$$
$$\tilde{c}_{\text{lifted}} = c$$

Then for any allocation satisfying $Lx = 0$:

$$x^T \tilde{Q}_{\text{lifted}} x = x^T (E Q E^T) x = (E^T x)^T Q (E^T x) = w^T Q w$$
$$\tilde{\mu}_{\text{lifted}}^T x = (E \mu)^T x = \mu^T (E^T x) = \mu^T w$$

Thus:

$$F_{\text{lifted}}(x) = w^T Q w - \mu^T w + c \equiv J(w)$$

### Numerical Comparison of All Formulations

| Formulation | Internal Objective | External Market Risk $w^T Q w - \mu^T w$ | Portfolio Variance | Expected Return | Max Weight | HHI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Centralized Markowitz** | **2.8408** | **1.8408** | **2.0436** | **0.2028** | **0.4217** | **0.3471** |
| **True Lifted Equivalent** | **2.8408** | **1.8408** | **2.0436** | **0.2028** | **0.4217** | **0.3471** |
| **Paper-Style Specific (3-Sub)** | **0.8488** | **1.9239** | **2.1490** | **0.2251** | **0.4552** | **0.3597** |
| **Paper-Style Powerset (5-Sub)** | **0.7617** | **1.8688** | **2.0860** | **0.2171** | **0.3619** | **0.3354** |

---

## 7. What We Can and Cannot Conclude

### What the Derivation Proves:
1. **Mathematical Inequality**: Under our reconstruction of the paper's equations, $Q_{\text{eff}} \neq Q$, $\mu_{\text{eff}} \neq \mu$, and $c_{\text{eff}} \neq c$. The Frobenius norm difference $\|Q_{\text{eff}} - Q\|_F / \|Q\|_F$ is over 86%.
2. **Distinct Optimization Landscapes**: The paper-style distributed problem is a **genuinely different quadratic optimization problem**, not a lifted reparameterization of Markowitz.
3. **Apples-to-Oranges Comparison**: The reported drop in objective value ($0.8573$ vs $2.8408$) is primarily an algebraic artifact of:
   - dividing quadratic variance by degrees ($1 / (K d_j)$),
   - omitting cross-subportfolio covariance terms, and
   - reducing effective constant costs from $1.0$ to $0.667$.
4. **Market Performance**: When measured by the true external market risk-return metric $w^T Q w - \mu^T w$, the centralized Markowitz solution achieves the minimal risk ($1.8408$), while the distributed solutions have slightly higher risk ($1.9239$ and $1.8688$).

### What We Cannot and Do Not Conclude:
- We do **not** conclude that the paper's framework is without merit.
- The distributed allocation produces a **more balanced institutional allocation** (as demonstrated in the GICS ETF study, where centralized Markowitz took extreme corner positions on 4 sectors, while the distributed framework invested across all 11 sectors).
- The hypergraph incidence structure and continuous-time primal-dual ODE are mathematically sound and offer valuable tools for decentralized multi-agent coordination.
