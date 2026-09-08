# Tutorial 08 — Decentralized Shadow Pricing

One of the most provocative ideas in Papastaikoudis, Watson, and Lestas (2025) is the transformation of optimization multipliers into a **decentralized pricing mechanism** for structured financial products.

In this tutorial, we dissect the paper's pricing equations, understand their mathematical mechanics, and evaluate their economic interpretation.

---

## 1. Epistemic Status & Attribution

> [!NOTE]
> **What the Paper Explicitly Claims (Section 2.4)**:
> - Dual variables $v^*$ associated with the consensus constraint $Lx = 0$ act as endogenous shadow prices.
> - The price of sub-portfolio component $x_i^j$ associated with underlying asset $z_j$ is given by:
>   $$p_i^{j*} = \left( \frac{v_i^j}{\sum_{\ell} v_{\ell}^j} + 1 \right) z_j$$
> - In vector form (Equation 5):
>   $$p^* = \operatorname{inv}(E^T v^*) \odot v^* + \mathbf{1} \odot z$$
> - This markup reflects the marginal contribution of sub-portfolio $i$ to risk reduction.
> - By applying expectations (Equation 6), this endogenous markup can be overlaid on classical pricing models like CAPM or APT:
>   $$\mathbb{E}[p^*] = \operatorname{inv}(E^T v^*) \odot v^* + \mathbf{1} \odot \mathbb{E}[z]$$

> [!IMPORTANT]
> **Our Explanatory Interpretation**:
> - The term $\frac{v_i^j}{\sum_\ell v_\ell^j}$ is a **normalized dual weight** lying in $[0, 1]$.
> - Consequently, the structured product price always falls in the interval $[z_j, 2 z_j]$.
> - If Desk $i$ bears 100% of the network's consensus tension for asset $j$, its structured product is priced at $2.0 \times z_j$. If it bears 0% tension, its product is priced at $1.0 \times z_j$.

> [!TIP]
> **New Experiments & Critiques Added**:
> - We examine the economic realism of this markup: why would an investor pay up to a 100% markup ($2.0 z_j$) to buy an asset through a structured sub-portfolio if the underlying security can be purchased on an open exchange at price $z_j$?

---

## 2. Intuition: Why Dual Variables Carry Economic Information

Consider two portfolio managers holding Apple stock (AAPL):
- **Manager 1 (High-Volatility Tech Desk)**: Holding AAPL introduces heavy co-movement with Microsoft and Nvidia. Manager 1 feels intense risk pressure from AAPL.
- **Manager 2 (Defensive Income Desk)**: Holding AAPL alongside Utilities and Consumer Staples provides diversification. Manager 2 feels minimal risk pressure.

When the consensus constraint $Lx = 0$ forces both desks to hold identical proportions of AAPL:
- Manager 1 is constrained from buying more (or forced to hold more than their internal risk limits prefer).
- The dual multiplier $v_1^{\text{AAPL}}$ measures how severely Manager 1's local risk objective is being distorted to maintain firm-wide consensus.

The paper argues that this shadow price measures the **marginal value of the network coupling**:
- Desks that absorb more coupling tension provide a valuable coordination service to the network.
- Therefore, the structured investment vehicle offered by that desk should command a **premium** over the raw asset price $z_j$.

---

## 3. Tiny Hand-Verifiable Example

Let's trace the exact numbers from Section 3.2 of the paper:
- Asymptotic dual variables:
  $$v^* = (0.3249, 0.3484, 0.2486, 0.0322, 0, 0, 0)^T$$

Let's look at **Asset 1** ($x_1$):
- It appears in Desk 1 ($v_1^1 = 0.3249$), Desk 2 ($v_2^1 = 0.0322$), and Desk 3 ($v_3^1 = 0.0$).
- Total dual pressure on Asset 1 across all desks:
  $$\sum_\ell v_\ell^1 = 0.3249 + 0.0322 + 0.0 = 0.3571$$
- Relative shadow price loadings:
  - Desk 1: $\frac{0.3249}{0.3571} = 0.9098 \approx 0.91$
  - Desk 2: $\frac{0.0322}{0.3571} = 0.0902 \approx 0.09$
  - Desk 3: $\frac{0.0}{0.3571} = 0.00$

Now apply the paper's pricing formula:
- **Desk 1 Product**: $p_1^{1*} = (0.91 + 1) z_1 = \mathbf{1.91 \, z_1}$
- **Desk 2 Product**: $p_2^{1*} = (0.09 + 1) z_1 = \mathbf{1.09 \, z_1}$
- **Desk 3 Product**: $p_3^{1*} = (0.00 + 1) z_1 = \mathbf{1.00 \, z_1}$

This reproduces the paper's exact pricing equations from page 13:
$$p_1^{1*} = 1.91 z_1, \quad p_2^{1*} = 1.09 z_1, \quad p_3^{1*} = z_1$$

Now check **Asset 2** ($x_2$):
- Desk 1: $v_1^2 = 0.3484$. Desk 2: $v_2^2 = 0$. Sum = $0.3484$.
- Desk 1: $\frac{0.3484}{0.3484} + 1 = 2.0 \implies p_1^{2*} = \mathbf{2.0 \, z_2}$.
- Desk 2: $\frac{0}{0.3484} + 1 = 1.0 \implies p_2^{2*} = \mathbf{1.0 \, z_2}$.

---

## 4. Mathematics: Dissecting the Pricing Formula

Let's dissect the components of Equation (5):

$$p_i^{j*} = \underbrace{z_j}_{\text{Base Primitive Asset Price}} + \underbrace{\left(\frac{v_i^j}{\sum_\ell v_\ell^j}\right)}_{\text{Normalized Dual Share } \in [0, 1]} \times z_j$$

### Properties of the Mechanism
1. **Bounded Markup**: Because $\frac{v_i^j}{\sum_\ell v_\ell^j} \le 1$, the structured product price is bounded:
   $$z_j \le p_i^{j*} \le 2 z_j$$
2. **Zero Tension Baseline**: If an asset experiences zero consensus tension across all desks ($\sum_\ell v_\ell^j = 0$), the markup is zero, and $p_i^{j*} = z_j$.
3. **Additive Markup**: The markup is proportional to the base price $z_j$. A \$1,000 asset receives a \$910 markup, while a \$10 asset receives a \$9.10 markup.

### Integration with External Factor Models (CAPM / APT)
The paper notes that the base price $z_j$ is determined exogenously by the broader capital market.
If $z_j$ follows a standard Capital Asset Pricing Model (CAPM):

$$\mathbb{E}[R_j] = R_f + \beta_j (\mathbb{E}[R_m] - R_f)$$

Then the expected structured product price inherits this market discount factor:

$$\mathbb{E}[p_i^{j*}] = \left( \frac{v_i^j}{\sum_\ell v_\ell^j} + 1 \right) \frac{D_j}{1 + \mathbb{E}[R_j]}$$

Where $D_j$ is the expected terminal dividend of asset $j$.

---

## 5. Executable Code

Let's compute the decentralized prices using [`pricing.py`](file:///Users/xq/Documents/GitHub/distributed-portfolio-learning/src/hypergraph_portfolio/pricing.py):

```python
import numpy as np
from hypergraph_portfolio.hypergraph import HypergraphPortfolio
from hypergraph_portfolio.pricing import compute_decentralized_prices

hp = HypergraphPortfolio(
    asset_names=["Tech", "Industrials", "Treasuries"],
    subportfolios=[["Tech", "Industrials", "Treasuries"], ["Tech", "Industrials"], ["Tech", "Treasuries"]],
)

# Paper dual variables from Section 3.2
v_star = np.array([0.3249, 0.3484, 0.2486, 0.0322, 0.0, 0.0, 0.0])

# Hypothetical market prices ($)
z_base = np.array([150.0, 80.0, 100.0])

pricing = compute_decentralized_prices(v_star=v_star, z_base=z_base, hypergraph=hp)

print("Decentralized Structured Product Prices:")
for asset_name, info in pricing.asset_breakdown.items():
    print(f"\n{asset_name} (Base Price: ${info['base_price']:.2f}):")
    for sp_idx, p_data in info["subportfolio_prices"].items():
        print(f"  Subportfolio {sp_idx + 1}: Multiplier = {p_data['price_multiplier']:.2f}x  ->  Price = ${p_data['adjusted_price']:.2f}")
```

---

## 6. Numerical Experiment & Verification

Output:
```
Decentralized Structured Product Prices:

Tech (Base Price: $150.00):
  Subportfolio 1: Multiplier = 1.91x  ->  Price = $286.47
  Subportfolio 2: Multiplier = 1.09x  ->  Price = $163.53
  Subportfolio 3: Multiplier = 1.00x  ->  Price = $150.00

Industrials (Base Price: $80.00):
  Subportfolio 1: Multiplier = 2.00x  ->  Price = $160.00
  Subportfolio 2: Multiplier = 1.00x  ->  Price = $80.00

Treasuries (Base Price: $100.00):
  Subportfolio 1: Multiplier = 2.00x  ->  Price = $200.00
  Subportfolio 3: Multiplier = 1.00x  ->  Price = $100.00
```

Notice:
- Buying Tech through Desk 1 costs **\$286.47**.
- Buying Tech through Desk 2 costs **\$163.53**.
- Buying Tech through Desk 3 costs **\$150.00** (the raw spot price).

---

## 7. Critical Interpretation: Economic Reality Check

Let us pause and examine this result from an institutional perspective:

1. **The No-Arbitrage Question**:
   In a frictionless market, the Law of One Price states that two identical claims on the same cash flow must trade at the same price. If Desk 1 offers Tech exposure at \$286 while Desk 3 offers it at \$150, any rational investor would buy from Desk 3 and short Desk 1!
2. **When Could a Markup Be Justified?**
   The paper's justification is that Desk 1 is providing a **structured package**:
   - An investor who buys Desk 1 is not just buying Tech; they are buying an optimized multi-asset bundle that includes Tech, Industrials, and Treasuries with guaranteed diversification properties.
   - If there are frictions, management fees, or capital access limits preventing the investor from building that bundle themselves, the desk can charge a fee.
3. **Multiplier vs Price**:
   In microeconomic theory, the Lagrange multiplier $v$ is an **internal transfer price** (shadow cost) within the firm, not necessarily an exchange-traded consumer price. Treating $v$ as a direct consumer price markup is an interesting modeling choice, but one that requires careful economic defense.

In Tutorial 09, we turn our critical lens to the paper's core performance claim: the comparison between centralized and distributed objective values.

---

## What You Should Now Understand

1. How the paper defines the decentralized pricing formula from asymptotic dual variables $v^*$.
2. The role of normalized dual weights in bounding product markups between $1.0 z_j$ and $2.0 z_j$.
3. How endogenous shadow prices can be combined with exogenous asset pricing models like CAPM.

---

## Questions Worth Investigating

- Why did the authors choose an additive formulation $(\frac{v_i^j}{\sum v} + 1) z_j$ rather than an exponential or fee-based formulation $(1 + \text{fee}) z_j$?
- What happens to the product prices if all sub-portfolios have identical risk preferences and identical asset allocations?
