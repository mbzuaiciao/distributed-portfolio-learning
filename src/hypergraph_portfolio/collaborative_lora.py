"""Development-only convex prediction combinations and exact LoRA update merging."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def validate_mixture(probabilities, weights):
    p = np.asarray(probabilities, dtype=float)
    w = np.asarray(weights, dtype=float)
    if p.ndim != 2 or w.shape != (p.shape[1],) or not p.shape[0]:
        raise ValueError("Expected nonempty observations x specialists and matching weights")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1):
        raise ValueError("Weights must be finite, nonnegative and sum to one")
    return p, w


def combine_predictions(probabilities, weights):
    p, w = validate_mixture(probabilities, weights)
    return p @ w


def fit_simplex_weights(development_probabilities, development_labels):
    p = np.asarray(development_probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 1:
        raise ValueError("Expected observations x specialists")
    w = np.full(p.shape[1], 1 / p.shape[1])
    p, _ = validate_mixture(p, w)
    y = np.asarray(development_labels, dtype=float)
    if y.shape != (len(p),) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected aligned binary development labels")
    result = minimize(
        lambda v: np.mean((p @ v - y) ** 2), w,
        jac=lambda v: 2 * p.T @ (p @ v - y) / len(y),
        bounds=[(0, 1)] * len(w), constraints={"type": "eq", "fun": lambda v: v.sum() - 1},
        method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()


def merge_lora_updates(factors, weights):
    """Sum effective B @ A updates, avoiding cross terms from averaging factors.

    Each factor is (B, A, scaling). This is a dense diagnostic merge, not a
    claim that averaging probabilities equals merging nonlinear models.
    """
    w = np.asarray(weights, dtype=float)
    if not factors or w.shape != (len(factors),):
        raise ValueError("Expected matching factors and weights")
    validate_mixture(np.zeros((1, len(w))), w)
    updates = []
    for b, a, scale in factors:
        b, a = np.asarray(b), np.asarray(a)
        if b.ndim != 2 or a.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError("Invalid LoRA factor shapes")
        if not np.isfinite(b).all() or not np.isfinite(a).all() or not np.isfinite(scale):
            raise ValueError("Nonfinite LoRA factor")
        updates.append(scale * (b @ a))
    if any(u.shape != updates[0].shape for u in updates):
        raise ValueError("Incompatible update dimensions")
    return sum(weight * update for weight, update in zip(w, updates))


def subspace_similarity(b_left, b_right):
    """Normalized overlap of numerical column spaces; zero updates give zero."""
    bases = []
    for b in (b_left, b_right):
        u, s, _ = np.linalg.svd(b, full_matrices=False)
        keep = s > (s[0] * 1e-6 if len(s) else 0)
        bases.append(u[:, keep])
    q, r = bases
    return float(np.sum((q.T @ r) ** 2) / min(q.shape[1], r.shape[1])) if min(q.shape[1], r.shape[1]) else 0.0
