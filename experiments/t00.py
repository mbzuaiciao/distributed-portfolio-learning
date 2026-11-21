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
