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
