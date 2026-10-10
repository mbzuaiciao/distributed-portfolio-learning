"""Diagnostic utilities for post-Experiment 1 follow-up analysis.

Implements:
1. Sanders / Murphy Brier score decomposition (Reliability, Resolution, Uncertainty).
2. Platt temperature scaling calibration for probabilistic models.
3. Log-odds combination rules for specialist probability fusion.
4. CoLoRA task gradient alignment diagnostics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression


def brier_score_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute Murphy/Sanders Brier score decomposition into Reliability, Resolution, and Uncertainty.

    Identity:
        Brier = Reliability - Resolution + Uncertainty
    where:
        Uncertainty = c * (1 - c) with c = base rate of y
        Reliability = sum (n_k / N) * (p_bar_k - y_bar_k)^2 (calibration error, lower is better)
        Resolution  = sum (n_k / N) * (y_bar_k - c)^2       (discrimination power, higher is better)
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)
    N = len(y)

    c = float(np.mean(y))
    uncertainty = c * (1.0 - c)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0

    for k in range(n_bins):
        mask = bin_indices == k
        n_k = np.sum(mask)
        if n_k > 0:
            p_bar_k = float(np.mean(p[mask]))
            y_bar_k = float(np.mean(y[mask]))
            reliability += (n_k / N) * ((p_bar_k - y_bar_k) ** 2)
            resolution += (n_k / N) * ((y_bar_k - c) ** 2)

    brier = float(np.mean((p - y) ** 2))

    return {
        "brier": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "reconstruction_brier": float(reliability - resolution + uncertainty),
    }


def fit_platt_scaling(
    y_prob_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """Fit 1D Platt / temperature scaling using logistic regression on logits."""
    p_clipped = np.clip(y_prob_train, 1e-6, 1.0 - 1e-6)
    logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    clf = LogisticRegression(C=1e9, solver="lbfgs")
    clf.fit(logits, y_train)
    return clf


def apply_platt_scaling(
    clf: LogisticRegression,
    y_prob: np.ndarray,
) -> np.ndarray:
    """Apply fitted Platt scaling to probability predictions."""
    p_clipped = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    return clf.predict_proba(logits)[:, 1]


def fit_log_odds_combiner(
    p_A_train: np.ndarray,
    p_B_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """Fit a bivariate log-odds combination model on specialist probability predictions.

    Model:
        P(y=1 | p_A, p_B) = sigma(w_0 + w_A * logit(p_A) + w_B * logit(p_B))
    """
    pA = np.clip(p_A_train, 1e-6, 1.0 - 1e-6)
    pB = np.clip(p_B_train, 1e-6, 1.0 - 1e-6)
    logitA = np.log(pA / (1.0 - pA))
    logitB = np.log(pB / (1.0 - pB))

    X = np.column_stack([logitA, logitB])
    clf = LogisticRegression(C=1.0, solver="lbfgs")  # mild L2 regularization
    clf.fit(X, y_train)
    return clf


def predict_log_odds_combiner(
    clf: LogisticRegression,
    p_A: np.ndarray,
    p_B: np.ndarray,
) -> np.ndarray:
    """Apply fitted log-odds combiner to specialist probability predictions."""
    pA = np.clip(p_A, 1e-6, 1.0 - 1e-6)
    pB = np.clip(p_B, 1e-6, 1.0 - 1e-6)
    logitA = np.log(pA / (1.0 - pA))
    logitB = np.log(pB / (1.0 - pB))

    X = np.column_stack([logitA, logitB])
    return clf.predict_proba(X)[:, 1]


def compute_task_gradient_cosine(
    grad_task1: np.ndarray,
    grad_task2: np.ndarray,
) -> float:
    """Compute cosine similarity between task gradient vectors on shared weights."""
    g1 = grad_task1.flatten()
    g2 = grad_task2.flatten()
    norm1 = np.linalg.norm(g1)
    norm2 = np.linalg.norm(g2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(g1, g2) / (norm1 * norm2))
