"""
Consensus Constraints, Projection, and Objective Reduction
===========================================================
Tools for analyzing and enforcing agreement across local copies of shared assets:
    L x = 0  <==>  x in Range(E)  <==>  x_i^j = x_k^j for all i, k sharing asset j.

Also provides analytical tools for Phase 2 equivalence audit:
    - reduce_distributed_objective: Computes closed-form Q_eff, mu_eff, c_eff
    - compare_matrices_term_by_term: Term-by-term algebraic discrepancy audit
    - build_true_lifted_system: Constructs provably equivalent lifted QP
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from hypergraph_portfolio.hypergraph import HypergraphPortfolio


def check_consensus(
    x: np.ndarray,
    hypergraph: HypergraphPortfolio,
    tol: float = 1e-6,
) -> Tuple[bool, float]:
    """
    Check whether vector x satisfies the hypergraph consensus constraint Lx = 0.

    Returns:
        is_consensus: True if ||Lx||_2 <= tol.
        consensus_error: ||Lx||_2 norm.
    """
    x = np.asarray(x, dtype=float)
    Lx = hypergraph.L @ x
    error = float(np.linalg.norm(Lx))
    return (error <= tol), error


def project_onto_consensus(
    x: np.ndarray,
    hypergraph: HypergraphPortfolio,
) -> np.ndarray:
    """
    Project an arbitrary vector x onto the consensus subspace Null(L) = Range(E):
        x_proj = P_E x = E (E^T E)^(-1) E^T x = (I - L) x.
    Every copy of asset j is replaced by the unweighted mean of its local copies.
    """
    x = np.asarray(x, dtype=float)
    return hypergraph.P_E @ x


def lift_centralized_to_distributed(
    z: np.ndarray,
    hypergraph: HypergraphPortfolio,
) -> np.ndarray:
    """
    Lift a centralized asset weight vector z to the distributed space:
        x = E (E^T E)^(-1) z  (i.e. x_i^j = z_j / d_j).
    This ensures Lx = 0 and sum_{copies} x_i^j = z_j.
    """
    return hypergraph.lift_asset_vector(z)


def compute_consensus_discrepancies(
    x: np.ndarray,
    hypergraph: HypergraphPortfolio,
) -> Dict[str, Dict[str, Any]]:
    """
    Inspect local copies for each underlying asset to identify which sub-portfolios disagree.

    Returns:
        A dictionary mapping asset_name -> {
            "values": list of local copy values,
            "subportfolios": list of subportfolio indices,
            "max_discrepancy": max(val) - min(val),
            "std_discrepancy": std(val)
        }
    """
    x = np.asarray(x, dtype=float)
    result = {}

    for asset_idx, asset_name in enumerate(hypergraph.asset_names):
        copies = []
        sp_indices = []
        for row, (sp_idx, local_idx, a_idx) in enumerate(hypergraph.node_meta):
            if a_idx == asset_idx:
                copies.append(float(x[row]))
                sp_indices.append(sp_idx)

        copies_arr = np.array(copies)
        result[asset_name] = {
            "values": copies,
            "subportfolios": sp_indices,
            "max_discrepancy": float(np.ptp(copies_arr)) if len(copies) > 1 else 0.0,
            "std_discrepancy": float(np.std(copies_arr)) if len(copies) > 1 else 0.0,
        }

    return result


def reduce_distributed_objective(
    hypergraph: HypergraphPortfolio,
    sub_Q: Sequence[np.ndarray],
    sub_mu: Sequence[np.ndarray],
    sub_c: Sequence[float],
    sub_a: Optional[Sequence[float]] = None,
    parameterization: str = "weight",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Analytically reduce the distributed objective:
        F(x) = sum_i a_i (x_i^T Q_i x_i - mu_i^T x_i + c_i)
    under the consensus constraint Lx = 0.

    Args:
        hypergraph: HypergraphPortfolio instance.
        sub_Q: List of local covariance matrices Q_i.
        sub_mu: List of local return vectors mu_i.
        sub_c: List of local fixed costs c_i.
        sub_a: Sub-portfolio importance weights a_i (default 1/K).
        parameterization:
            - 'weight': In terms of total asset weights w = E^T x (where sum w_j = 1).
                        x = E (E^T E)^(-1) w.
            - 'local_copy': In terms of common local copy values z (where x = E z).

    Returns:
        Q_eff: Effective covariance matrix, shape (m, m).
        mu_eff: Effective expected return vector, shape (m,).
        c_eff: Effective scalar constant.
    """
    K = hypergraph.num_subportfolios
    m = hypergraph.num_assets

    if sub_a is None:
        sub_a = np.ones(K, dtype=float) / float(K)
    else:
        sub_a = np.asarray(sub_a, dtype=float)

    # Accumulate embedded local matrices in R^{m x m}
    Q_eff_z = np.zeros((m, m), dtype=float)
    mu_eff_z = np.zeros(m, dtype=float)
    c_eff = float(np.sum(sub_a * np.asarray(sub_c, dtype=float)))

    for i in range(K):
        sl = hypergraph.get_subportfolio_slice(i)
        E_i = hypergraph.E[sl, :]  # Shape (p_i, m)
        Q_i = np.asarray(sub_Q[i], dtype=float)
        mu_i = np.asarray(sub_mu[i], dtype=float)
        a_i = sub_a[i]

        Q_eff_z += a_i * (E_i.T @ Q_i @ E_i)
        mu_eff_z += a_i * (E_i.T @ mu_i)

    if parameterization == "local_copy":
        return Q_eff_z, mu_eff_z, c_eff
    elif parameterization == "weight":
        # w = D_E z  ==>  z = D_E^{-1} w
        inv_D = hypergraph.inv_D_E
        Q_eff_w = inv_D @ Q_eff_z @ inv_D
        mu_eff_w = inv_D @ mu_eff_z
        return Q_eff_w, mu_eff_w, c_eff
    else:
        raise ValueError(f"Unknown parameterization '{parameterization}'. Choose 'weight' or 'local_copy'.")


def compare_matrices_term_by_term(
    Q_eff: np.ndarray,
    Q_true: np.ndarray,
    mu_eff: Optional[np.ndarray] = None,
    mu_true: Optional[np.ndarray] = None,
    c_eff: Optional[float] = None,
    c_true: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Term-by-term comparison between effective reduced parameters and true centralized parameters.

    Returns:
        Dict containing matrix differences, Frobenius norm, relative norm, entry ratios,
        and eigenvalue spectrum comparisons.
    """
    Q_eff = np.asarray(Q_eff, dtype=float)
    Q_true = np.asarray(Q_true, dtype=float)

    diff_Q = Q_eff - Q_true
    frob_diff = float(np.linalg.norm(diff_Q, "fro"))
    frob_true = float(np.linalg.norm(Q_true, "fro"))
    rel_frob = frob_diff / (frob_true + 1e-12)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(np.abs(Q_true) > 1e-12, Q_eff / Q_true, np.nan)

    eig_eff = np.sort(np.linalg.eigvalsh(Q_eff))
    eig_true = np.sort(np.linalg.eigvalsh(Q_true))

    result = {
        "diff_Q": diff_Q,
        "frobenius_diff_Q": frob_diff,
        "relative_diff_Q": rel_frob,
        "entry_ratios_Q": ratios,
        "eigenvalues_Q_eff": eig_eff,
        "eigenvalues_Q_true": eig_true,
    }

    if mu_eff is not None and mu_true is not None:
        mu_eff = np.asarray(mu_eff, dtype=float)
        mu_true = np.asarray(mu_true, dtype=float)
        result["diff_mu"] = mu_eff - mu_true
        result["norm_diff_mu"] = float(np.linalg.norm(mu_eff - mu_true))
        with np.errstate(divide="ignore", invalid="ignore"):
            result["entry_ratios_mu"] = np.where(np.abs(mu_true) > 1e-12, mu_eff / mu_true, np.nan)

    if c_eff is not None and c_true is not None:
        result["diff_c"] = float(c_eff - c_true)

    return result


def build_true_lifted_system(
    hypergraph: HypergraphPortfolio,
    Q: np.ndarray,
    mu: np.ndarray,
    c: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Construct the true lifted quadratic programming system in R^p:
        Q_lifted = E Q E^T
        mu_lifted = E mu
    Such that for any allocation x in R^p with total asset holdings w = E^T x:
        x^T Q_lifted x = w^T Q w
        mu_lifted^T x = mu^T w
    Under the consensus constraint Lx = 0 and budget 1^T E^T x = 1,
    the reduced objective satisfies Q_eff == Q and mu_eff == mu,
    proving strict mathematical equivalence to the centralized Markowitz problem.
    """
    Q = np.asarray(Q, dtype=float)
    mu = np.asarray(mu, dtype=float)
    E = hypergraph.E

    Q_lifted = E @ Q @ E.T
    mu_lifted = E @ mu
    return Q_lifted, mu_lifted, float(c)
