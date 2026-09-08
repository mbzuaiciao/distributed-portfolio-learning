"""
Hypergraph Representation for Overlapping Sub-Portfolios
========================================================
Implements the hypergraph formulation from Papastaikoudis et al. (2025):
    - Sub-portfolios Sigma_1, ..., Sigma_K as subsets of asset universe A.
    - Incidence matrix E: shape (p, |A|), where p = sum(p_i).
    - Edge degree matrix D_E = E^T E = diag(d_1, ..., d_|A|).
    - Vertex degree matrix D_V = I_{p x p}.
    - Normalized hypergraph Laplacian:
          L = D_V - E (E^T E)^{-1} E^T = I - E (E^T E)^{-1} E^T.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np


def build_powerset_subportfolios(
    asset_indices: Sequence[int],
    min_size: int = 2,
    include_full: bool = True,
) -> List[List[int]]:
    """
    Generate powerset sub-portfolios of specified sizes.

    Args:
        asset_indices: List of asset indices (e.g. [0, 1, 2]).
        min_size: Minimum subportfolio size (default 2).
        include_full: Whether to include the grand coalition (all assets).

    Returns:
        List of sub-portfolio index lists.
    """
    n = len(asset_indices)
    subportfolios = []
    max_size = n if include_full else n - 1
    for r in range(min_size, max_size + 1):
        for combo in combinations(asset_indices, r):
            subportfolios.append(list(combo))
    return subportfolios


@dataclass
class HypergraphPortfolio:
    """
    Hypergraph model representing overlapping sub-portfolios.

    Attributes:
        asset_names: List of unique asset identifiers in the universe A.
        subportfolios: List of sub-portfolios, each specified as a list of asset names
                       or 0-based integer asset indices.
    """

    asset_names: List[str]
    subportfolios: List[List[Union[str, int]]]

    def __post_init__(self):
        self.num_assets = len(self.asset_names)
        self.asset_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.asset_names)}

        # Standardize subportfolios as lists of integer asset indices
        self.subportfolio_indices: List[List[int]] = []
        for sp in self.subportfolios:
            idx_list = []
            for item in sp:
                if isinstance(item, str):
                    if item not in self.asset_to_idx:
                        raise ValueError(f"Asset '{item}' not found in asset universe.")
                    idx_list.append(self.asset_to_idx[item])
                elif isinstance(item, (int, np.integer)):
                    if not (0 <= item < self.num_assets):
                        raise ValueError(f"Asset index {item} out of range [0, {self.num_assets}).")
                    idx_list.append(int(item))
                else:
                    raise TypeError(f"Invalid asset identifier: {item}")
            self.subportfolio_indices.append(idx_list)

        self.num_subportfolios = len(self.subportfolio_indices)
        self.subportfolio_dims = [len(sp) for sp in self.subportfolio_indices]
        self.total_nodes = sum(self.subportfolio_dims)

        # Build incidence matrix E and Laplacian L
        self._build_matrices()

    def _build_matrices(self):
        """Construct incidence matrix E, degree matrices, and Laplacian L."""
        p = self.total_nodes
        m = self.num_assets

        # E has dimension (p, m)
        self.E = np.zeros((p, m), dtype=float)

        # Track which row belongs to which (subportfolio_idx, local_var_idx, asset_idx)
        self.node_meta: List[Tuple[int, int, int]] = []

        row = 0
        for i, sp in enumerate(self.subportfolio_indices):
            for local_idx, asset_idx in enumerate(sp):
                self.E[row, asset_idx] = 1.0
                self.node_meta.append((i, local_idx, asset_idx))
                row += 1

        # Edge degree matrix D_E = E^T E
        # D_E[j, j] counts how many subportfolios contain asset j
        self.edge_degrees = np.sum(self.E, axis=0)
        if np.any(self.edge_degrees == 0):
            uncovered = [self.asset_names[j] for j in np.where(self.edge_degrees == 0)[0]]
            raise ValueError(f"Assets not covered by any subportfolio: {uncovered}")

        self.D_E = np.diag(self.edge_degrees)
        self.inv_D_E = np.diag(1.0 / self.edge_degrees)

        # Vertex degree is identity matrix I_{p x p} because each node represents 1 asset copy
        self.D_V = np.eye(p)

        # Projection matrix P_E = E (E^T E)^(-1) E^T
        self.P_E = self.E @ self.inv_D_E @ self.E.T

        # Hypergraph Laplacian L = I - P_E
        self.L = self.D_V - self.P_E

    def get_subportfolio_slice(self, subportfolio_idx: int) -> slice:
        """Return the index slice in the global vector x for a given subportfolio."""
        start = sum(self.subportfolio_dims[:subportfolio_idx])
        end = start + self.subportfolio_dims[subportfolio_idx]
        return slice(start, end)

    def extract_subportfolio_vector(self, x: np.ndarray, subportfolio_idx: int) -> np.ndarray:
        """Extract x_i from the stacked vector x."""
        return x[self.get_subportfolio_slice(subportfolio_idx)]

    def stack_subportfolio_vectors(self, subportfolio_vectors: Sequence[np.ndarray]) -> np.ndarray:
        """Stack local sub-portfolio vectors [x_1, ..., x_K] into global vector x."""
        if len(subportfolio_vectors) != self.num_subportfolios:
            raise ValueError(f"Expected {self.num_subportfolios} vectors, got {len(subportfolio_vectors)}")
        return np.concatenate([np.asarray(v, dtype=float) for v in subportfolio_vectors])

    def aggregate_to_assets(self, x: np.ndarray) -> np.ndarray:
        """
        Aggregate local allocations x to global asset weights z:
            z = E^T x.
        """
        return self.E.T @ np.asarray(x, dtype=float)

    def lift_asset_vector(self, z: np.ndarray) -> np.ndarray:
        """
        Lift global asset weights z to consensus local allocation x:
            x = E (E^T E)^(-1) z  (i.e. x_i^j = z_j / d_j).
        This allocation satisfies Lx = 0 and E^T x = z.
        """
        return self.E @ (self.inv_D_E @ np.asarray(z, dtype=float))

    def verify_laplacian_properties(self, tol: float = 1e-10) -> Dict[str, bool]:
        """
        Verify the mathematical properties of the hypergraph Laplacian:
            1. Symmetry: L = L^T
            2. Idempotency: L^2 = L (orthogonal projector)
            3. Spectrum: Eigenvalues in {0, 1}
            4. Nullspace: L E = 0 (consensus subspace)
        """
        is_symmetric = np.allclose(self.L, self.L.T, atol=tol)
        is_idempotent = np.allclose(self.L @ self.L, self.L, atol=tol)
        eigenvalues = np.linalg.eigvalsh(self.L)
        spectrum_valid = np.all(
            (np.abs(eigenvalues) < tol) | (np.abs(eigenvalues - 1.0) < tol)
        )
        nullspace_valid = np.allclose(self.L @ self.E, 0.0, atol=tol)

        return {
            "symmetric": bool(is_symmetric),
            "idempotent": bool(is_idempotent),
            "spectrum_in_0_1": bool(spectrum_valid),
            "nullspace_contains_range_E": bool(nullspace_valid),
        }
