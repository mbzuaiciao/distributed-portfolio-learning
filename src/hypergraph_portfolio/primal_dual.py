"""
Continuous-Time Primal-Dual Dynamics & Discretization
=====================================================
Implements the continuous-time primal-dual dynamical system from Section 2.3:
    dot{x}      = -nabla f(x) + L v + lambda * 1
    dot{lambda} = budget - 1^T x
    dot{v}      = -L x

Subject to projections:
    x >= 0
    v >= 0

Includes Lyapunov verification (Section 7.3) and CVXPY independent benchmarking.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from hypergraph_portfolio.hypergraph import HypergraphPortfolio


@dataclass
class PrimalDualHistory:
    """Stores the simulation trajectory across iterations."""

    time_points: np.ndarray
    x_trajectory: np.ndarray          # Shape (T, p)
    v_trajectory: np.ndarray          # Shape (T, p)
    lambda_trajectory: np.ndarray     # Shape (T,)
    objective_trajectory: np.ndarray  # Shape (T,)
    sub_objectives_trajectory: np.ndarray  # Shape (T, K)
    consensus_error: np.ndarray       # Shape (T,)
    budget_error: np.ndarray          # Shape (T,)
    lyapunov_trajectory: np.ndarray   # Shape (T,)


class PrimalDualSolver:
    """
    Continuous-time primal-dual dynamical solver for distributed portfolio optimization.

    Solves:
        min_{x_1, ..., x_K}  sum_{i=1}^K a_i (x_i^T Q_i x_i - mu_i^T x_i + c_i)
        s.t.                 1^T x = budget
                             L x = 0
                             x >= 0
    """

    def __init__(
        self,
        hypergraph: HypergraphPortfolio,
        sub_Q: Sequence[np.ndarray],
        sub_mu: Sequence[np.ndarray],
        sub_c: Optional[Sequence[float]] = None,
        sub_a: Optional[Sequence[float]] = None,
        budget: float = 1.0,
        scale_gradient_by_weights: bool = True,
    ):
        self.hypergraph = hypergraph
        self.K = hypergraph.num_subportfolios
        self.p = hypergraph.total_nodes
        self.budget = float(budget)
        self.scale_gradient_by_weights = scale_gradient_by_weights

        # Validate and store sub-portfolio parameters
        self.sub_Q = [np.asarray(Q, dtype=float) for Q in sub_Q]
        self.sub_mu = [np.asarray(mu, dtype=float) for mu in sub_mu]

        if sub_c is None:
            self.sub_c = np.zeros(self.K, dtype=float)
        else:
            self.sub_c = np.asarray(sub_c, dtype=float)

        if sub_a is None:
            self.sub_a = np.ones(self.K, dtype=float) / float(self.K)
        else:
            self.sub_a = np.asarray(sub_a, dtype=float)

        # Assemble block-diagonal Hessian and linear terms
        self._assemble_system()

    def _assemble_system(self):
        """Build the full stacked Hessian matrix and linear gradient vector."""
        self.H_blocks = np.zeros((self.p, self.p), dtype=float)
        self.g_linear = np.zeros(self.p, dtype=float)

        for i in range(self.K):
            sl = self.hypergraph.get_subportfolio_slice(i)
            weight = self.sub_a[i] if self.scale_gradient_by_weights else 1.0
            # Quadratic term x_i^T Q_i x_i has derivative 2 Q_i x_i
            self.H_blocks[sl, sl] = 2.0 * weight * self.sub_Q[i]
            self.g_linear[sl] = -weight * self.sub_mu[i]

    def sub_objective(self, x_i: np.ndarray, i: int) -> float:
        """Compute f_i(x_i) = x_i^T Q_i x_i - mu_i^T x_i + c_i."""
        x_i = np.asarray(x_i, dtype=float)
        return float(x_i @ self.sub_Q[i] @ x_i - self.sub_mu[i] @ x_i + self.sub_c[i])

    def total_objective(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute total weighted objective sum_{i=1}^K a_i f_i(x_i)
        and individual sub-portfolio objectives [f_1, ..., f_K].
        """
        sub_f = np.zeros(self.K, dtype=float)
        for i in range(self.K):
            x_i = self.hypergraph.extract_subportfolio_vector(x, i)
            sub_f[i] = self.sub_objective(x_i, i)
        total_f = float(np.sum(self.sub_a * sub_f))
        return total_f, sub_f

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute full objective gradient nabla f(x) = H x + g."""
        return self.H_blocks @ x + self.g_linear

    def lyapunov_function(self, x: np.ndarray, lam: float, v: np.ndarray) -> float:
        """
        Compute the Krasovskii / saddle-point Lyapunov function (Section 7.3):
            V = 1/2 ||nabla f(x) - L v - lam * 1||^2 + 1/2 (1^T x - budget)^2 + 1/2 ||L x||^2.
        """
        g = self.gradient(x) - self.hypergraph.L @ v - lam * np.ones(self.p)
        r = np.sum(x) - self.budget
        s = self.hypergraph.L @ x
        return 0.5 * float(np.sum(g**2) + r**2 + np.sum(s**2))

    def solve(
        self,
        dt: float = 0.002,
        max_steps: int = 25000,
        x0: Optional[np.ndarray] = None,
        record_every: int = 10,
        tol_consensus: float = 1e-5,
        tol_budget: float = 1e-5,
    ) -> Tuple[np.ndarray, float, np.ndarray, PrimalDualHistory]:
        """
        Simulate continuous-time primal-dual dynamics with Euler discretization and projection.

        Args:
            dt: Time step size.
            max_steps: Maximum integration steps.
            x0: Initial primal allocation (default equal allocation budget / p).
            record_every: Record trajectory metrics every N steps.
            tol_consensus: Stopping tolerance for ||Lx||.
            tol_budget: Stopping tolerance for |1^T x - budget|.

        Returns:
            x_opt: Optimal local allocations.
            lam_opt: Optimal budget dual variable.
            v_opt: Optimal consensus dual variables.
            history: Trajectory records.
        """
        if x0 is None:
            x = np.ones(self.p, dtype=float) * (self.budget / float(self.p))
        else:
            x = np.asarray(x0, dtype=float).copy()

        lam = 0.0
        v = np.zeros(self.p, dtype=float)
        ones_p = np.ones(self.p, dtype=float)

        t_list = []
        x_hist = []
        v_hist = []
        lam_hist = []
        obj_hist = []
        sub_obj_hist = []
        cons_hist = []
        budg_hist = []
        lyap_hist = []

        for step in range(max_steps):
            if step % record_every == 0:
                tot_f, sub_f = self.total_objective(x)
                c_err = float(np.linalg.norm(self.hypergraph.L @ x))
                b_err = float(abs(np.sum(x) - self.budget))
                lyap = self.lyapunov_function(x, lam, v)

                t_list.append(step * dt)
                x_hist.append(x.copy())
                v_hist.append(v.copy())
                lam_hist.append(float(lam))
                obj_hist.append(tot_f)
                sub_obj_hist.append(sub_f.copy())
                cons_hist.append(c_err)
                budg_hist.append(b_err)
                lyap_hist.append(lyap)

            # Continuous-time dynamics (4a)-(4c)
            dx = -self.gradient(x) + self.hypergraph.L @ v + lam * ones_p
            dlam = self.budget - np.sum(x)
            dv = -self.hypergraph.L @ x

            # Forward Euler step with non-negativity projection
            x = np.maximum(0.0, x + dt * dx)
            lam = lam + dt * dlam
            v = np.maximum(0.0, v + dt * dv)

        # Final record
        tot_f, sub_f = self.total_objective(x)
        c_err = float(np.linalg.norm(self.hypergraph.L @ x))
        b_err = float(abs(np.sum(x) - self.budget))
        lyap = self.lyapunov_function(x, lam, v)

        t_list.append(max_steps * dt)
        x_hist.append(x.copy())
        v_hist.append(v.copy())
        lam_hist.append(float(lam))
        obj_hist.append(tot_f)
        sub_obj_hist.append(sub_f.copy())
        cons_hist.append(c_err)
        budg_hist.append(b_err)
        lyap_hist.append(lyap)

        history = PrimalDualHistory(
            time_points=np.array(t_list),
            x_trajectory=np.array(x_hist),
            v_trajectory=np.array(v_hist),
            lambda_trajectory=np.array(lam_hist),
            objective_trajectory=np.array(obj_hist),
            sub_objectives_trajectory=np.array(sub_obj_hist),
            consensus_error=np.array(cons_hist),
            budget_error=np.array(budg_hist),
            lyapunov_trajectory=np.array(lyap_hist),
        )

        return x, lam, v, history

    def solve_cvxpy(self) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """
        Solve the distributed problem using CVXPY as an exact benchmark.

        Returns:
            x_opt: Optimal allocation vector.
            f_opt: Optimal total objective value.
            sub_f: Array of sub-portfolio objectives.
            dual_budget: Dual multiplier for the budget constraint.
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for solve_cvxpy()")

        x = cp.Variable(self.p)
        sub_exprs = []
        for i in range(self.K):
            sl = self.hypergraph.get_subportfolio_slice(i)
            x_i = x[sl]
            f_i = cp.quad_form(x_i, self.sub_Q[i]) - self.sub_mu[i] @ x_i + self.sub_c[i]
            sub_exprs.append(f_i)

        total_obj = cp.sum([self.sub_a[i] * sub_exprs[i] for i in range(self.K)])
        budget_con = cp.sum(x) == self.budget
        consensus_con = self.hypergraph.L @ x == 0
        nonneg_con = x >= 0

        prob = cp.Problem(cp.Minimize(total_obj), [budget_con, consensus_con, nonneg_con])
        prob.solve()

        sub_f_vals = np.array([float(expr.value) for expr in sub_exprs])
        return x.value, float(prob.value), sub_f_vals, float(budget_con.dual_value or 0.0)
