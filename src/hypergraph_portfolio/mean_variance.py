"""
Centralized Mean-Variance Portfolio Optimization
================================================
Implements Markowitz-style quadratic portfolio optimization:
    min_x  x^T Q x - mu^T x + c
    s.t.   x^T 1 = budget
           x >= 0

Matches the convention used in Papastaikoudis et al. (2025), where the quadratic risk term
is x^T Q x (without a 1/2 factor), resulting in gradient 2*Q*x - mu.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.optimize as opt


def project_to_simplex(v: np.ndarray, budget: float = 1.0) -> np.ndarray:
    """
    Project vector v onto the simplex {x >= 0, sum(x) == budget} using an exact O(N log N) algorithm.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0:
        return np.array([])
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_indices = np.nonzero(u * np.arange(1, n + 1) > (cssv - budget))[0]
    if len(rho_indices) == 0:
        # Fallback in case of numerical issues
        return np.ones(n) * (budget / n)
    rho = rho_indices[-1]
    theta = (cssv[rho] - budget) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    return w


@dataclass
class MeanVariancePortfolio:
    """
    Container and solver for centralized mean-variance portfolio problems.

    Attributes:
        Q: Covariance matrix (n x n), symmetric positive semidefinite.
        mu: Expected return vector (n,).
        c: Fixed cost scalar (default 0.0).
        budget: Total portfolio weight budget (default 1.0).
        asset_names: Optional list of asset identifier strings.
    """

    Q: np.ndarray
    mu: np.ndarray
    c: float = 0.0
    budget: float = 1.0
    asset_names: Optional[list] = None

    def __post_init__(self):
        self.Q = np.asarray(self.Q, dtype=float)
        self.mu = np.asarray(self.mu, dtype=float)
        self.c = float(self.c)
        self.budget = float(self.budget)
        n = len(self.mu)
        if self.Q.shape != (n, n):
            raise ValueError(f"Shape mismatch: Q {self.Q.shape} vs mu length {n}")
        if self.asset_names is None:
            self.asset_names = [f"Asset_{i+1}" for i in range(n)]

    @property
    def num_assets(self) -> int:
        return len(self.mu)

    def objective(self, x: np.ndarray) -> float:
        """Compute x^T Q x - mu^T x + c."""
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x - self.mu @ x + self.c)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient 2*Q*x - mu."""
        x = np.asarray(x, dtype=float)
        return 2.0 * (self.Q @ x) - self.mu

    def solve(self, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Solve the constrained optimization problem using SciPy SLSQP.

        Returns:
            x_opt: Optimal asset allocation vector.
            f_opt: Optimal objective value.
        """
        n = self.num_assets
        if x0 is None:
            x0 = np.ones(n) * (self.budget / n)

        bounds = [(0.0, None) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - self.budget}]

        res = opt.minimize(
            fun=self.objective,
            x0=x0,
            jac=self.gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not res.success:
            raise RuntimeError(f"Centralized optimization failed: {res.message}")

        return res.x, float(res.fun)

    def solve_cvxpy(self) -> Tuple[np.ndarray, float]:
        """
        Independent verification solver using CVXPY.
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for solve_cvxpy()")

        n = self.num_assets
        x = cp.Variable(n)
        obj = cp.Minimize(cp.quad_form(x, self.Q) - self.mu @ x + self.c)
        constraints = [cp.sum(x) == self.budget, x >= 0]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return x.value, float(prob.value)


def solve_centralized_portfolio(
    Q: np.ndarray,
    mu: np.ndarray,
    c: float = 0.0,
    budget: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Convenience helper to solve a centralized mean-variance portfolio."""
    model = MeanVariancePortfolio(Q=Q, mu=mu, c=c, budget=budget)
    return model.solve()
