"""Minimal Uncertainty-Estimator Diagnostic Benchmark.

Implements:
1. BayesianLogisticRegression: Newton-Raphson MAP with exact Laplace posterior.
2. NeuralLaplaceEstimator: Matched MLP penultimate-layer Laplace estimator.
3. Diagnostic DGP: Known parameter coefficients with controlled sparse support and extrapolation.
4. Evaluation Metrics: Support sensitivity, interval coverage, selective risk-coverage, and conditioning.
5. Positive and Negative Controls.

Enforces CPU execution with PyTorch limited to two threads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from torch import nn

from hypergraph_portfolio.synthetic_multitask import _sigmoid

# Host safeguard: strictly limit PyTorch CPU threads to 2
torch.set_num_threads(2)


@dataclass
class BayesianLogisticRegression:
    """Bayesian Logistic Regression with Gaussian prior and exact Laplace approximation."""

    weights: np.ndarray  # (d,)
    intercept: float
    covariance: np.ndarray  # (d+1, d+1)
    alpha: float
    converged: bool
    iterations: int
    hessian_condition_number: float

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 1.0,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> BayesianLogisticRegression:
        """Fit MAP parameters via Newton-Raphson Iteratively Reweighted Least Squares."""
        n, d = X.shape
        # Augment with bias column
        X_tilde = np.column_stack([X, np.ones(n)])
        w = np.zeros(d + 1)

        # Prior precision matrix: alpha * I (no regularization on bias)
        prior_prec = alpha * np.eye(d + 1)
        prior_prec[-1, -1] = 1e-4  # diffuse prior on intercept

        converged = False
        iteration = 0
        H = prior_prec.copy()

        for iteration in range(1, max_iter + 1):
            p = _sigmoid(X_tilde @ w)
            p = np.clip(p, 1e-7, 1.0 - 1e-7)

            # Gradient: X^T (p - y) + Lambda w
            grad = X_tilde.T @ (p - y) + prior_prec @ w

            # Hessian: X^T W X + Lambda, where W = diag(p * (1 - p))
            W = p * (1.0 - p)
            H = (X_tilde.T * W) @ X_tilde + prior_prec

            try:
                delta_w = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                H_reg = H + 1e-4 * np.eye(d + 1)
                delta_w = np.linalg.solve(H_reg, grad)

            w_new = w - delta_w
            if np.linalg.norm(delta_w) < tol:
                w = w_new
                converged = True
                break
            w = w_new

        # Compute exact Laplace posterior covariance
        p = _sigmoid(X_tilde @ w)
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        W = p * (1.0 - p)
        H_final = (X_tilde.T * W) @ X_tilde + prior_prec

        evals = np.linalg.eigvalsh(H_final)
        cond_num = float(evals[-1] / max(evals[0], 1e-12))
        cov = np.linalg.inv(H_final)

        return cls(
            weights=w[:d],
            intercept=float(w[-1]),
            covariance=cov,
            alpha=alpha,
            converged=converged,
            iterations=iteration,
            hessian_condition_number=cond_num,
        )

    def predict_latent(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict latent logit mean and epistemic variance."""
        n = X.shape[0]
        X_tilde = np.column_stack([X, np.ones(n)])
        mean = X_tilde @ np.append(self.weights, self.intercept)
        var = np.sum((X_tilde @ self.covariance) * X_tilde, axis=1)
        return mean, np.maximum(var, 1e-12)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probit approximation of the expected predictive probability under posterior."""
        mean, var = self.predict_latent(X)
        kappa = 1.0 / np.sqrt(1.0 + math.pi * var / 8.0)
        return _sigmoid(kappa * mean)

    def credible_interval_95(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return 95% posterior credible interval on latent logit scale."""
        mean, var = self.predict_latent(X)
        half_width = 1.95996 * np.sqrt(var)
        return mean - half_width, mean + half_width


class SmallDiagnosticMLP(nn.Module):
    """Matched 2-layer MLP for diagnostic comparison."""

    def __init__(self, in_features: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc1(x))


@dataclass
class NeuralLaplaceEstimator:
    """Penultimate-layer Laplace estimator on neural representations."""

    model: SmallDiagnosticMLP
    w_readout: np.ndarray  # (hidden_dim,)
    b_readout: float
    covariance: np.ndarray  # (hidden_dim+1, hidden_dim+1)
    alpha: float
    hessian_condition_number: float

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        hidden_dim: int = 32,
        epochs: int = 150,
        lr: float = 0.01,
        alpha: float = 1.0,
        seed: int = 42,
    ) -> NeuralLaplaceEstimator:
        torch.manual_seed(seed)
        model = SmallDiagnosticMLP(in_features=X.shape[1], hidden_dim=hidden_dim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()

        xt = torch.from_numpy(X).float()
        yt = torch.from_numpy(y).float()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(xt)
            loss = criterion(logits, yt)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            phi = model.penultimate(xt).numpy()  # (n, hidden_dim)
            logits = model(xt).numpy()
            probs = _sigmoid(logits)

        # Fit Laplace on penultimate layer
        n, h_dim = phi.shape
        phi_tilde = np.column_stack([phi, np.ones(n)])

        W = probs * (1.0 - probs)
        prior_prec = alpha * np.eye(h_dim + 1)
        prior_prec[-1, -1] = 1e-4

        H = (phi_tilde.T * W) @ phi_tilde + prior_prec
        evals = np.linalg.eigvalsh(H)
        cond_num = float(evals[-1] / max(evals[0], 1e-12))
        cov = np.linalg.inv(H)

        w_read = model.fc2.weight.detach().squeeze(0).numpy()
        b_read = float(model.fc2.bias.detach().item())

        return cls(
            model=model,
            w_readout=w_read,
            b_readout=b_read,
            covariance=cov,
            alpha=alpha,
            hessian_condition_number=cond_num,
        )

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model.penultimate(torch.from_numpy(X).float()).numpy()

    def predict_latent(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        phi = self.extract_features(X)
        n = phi.shape[0]
        phi_tilde = np.column_stack([phi, np.ones(n)])
        w_vec = np.append(self.w_readout, self.b_readout)
        mean = phi_tilde @ w_vec
        var = np.sum((phi_tilde @ self.covariance) * phi_tilde, axis=1)
        return mean, np.maximum(var, 1e-12)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        mean, var = self.predict_latent(X)
        kappa = 1.0 / np.sqrt(1.0 + math.pi * var / 8.0)
        return _sigmoid(kappa * mean)


def generate_diagnostic_dgp(
    seed: int,
    n_train: int = 256,
    n_test_in: int = 512,
    n_test_extrap: int = 512,
    d: int = 4,
) -> Dict[str, Any]:
    """Generate synthetic dataset with known true coefficients and controlled support truncation."""
    rng = np.random.default_rng(seed)

    # Fixed canonical coefficients: norm = 1.5
    beta_star = np.array([1.0, 0.75, -0.5, 0.5])[:d]
    beta_star = beta_star / np.linalg.norm(beta_star) * 1.5

    # Training set: x_0 truncated to [-2.0, 0.0], other features standard normal
    x0_train = rng.uniform(-2.0, 0.0, size=n_train)
    x_other_train = rng.normal(0.0, 1.0, size=(n_train, d - 1))
    X_train = np.column_stack([x0_train, x_other_train])
    z_train = X_train @ beta_star
    p_train = _sigmoid(z_train)
    y_train = (rng.uniform(size=n_train) < p_train).astype(float)

    # In-support test set: x_0 in [-2.0, 0.0]
    x0_test_in = rng.uniform(-2.0, 0.0, size=n_test_in)
    x_other_test_in = rng.normal(0.0, 1.0, size=(n_test_in, d - 1))
    X_test_in = np.column_stack([x0_test_in, x_other_test_in])
    z_test_in = X_test_in @ beta_star
    p_test_in = _sigmoid(z_test_in)
    y_test_in = (rng.uniform(size=n_test_in) < p_test_in).astype(float)

    # Extrapolation test set: x_0 in [0.0, 4.0] (varying distance away from training support)
    x0_test_extrap = rng.uniform(0.0, 4.0, size=n_test_extrap)
    x_other_test_extrap = rng.normal(0.0, 1.0, size=(n_test_extrap, d - 1))
    X_test_extrap = np.column_stack([x0_test_extrap, x_other_test_extrap])
    z_test_extrap = X_test_extrap @ beta_star
    p_test_extrap = _sigmoid(z_test_extrap)
    y_test_extrap = (rng.uniform(size=n_test_extrap) < p_test_extrap).astype(float)

    # Combined test set
    X_test_all = np.vstack([X_test_in, X_test_extrap])
    y_test_all = np.concatenate([y_test_in, y_test_extrap])
    z_test_all = np.concatenate([z_test_in, z_test_extrap])
    is_extrap = np.concatenate([np.zeros(n_test_in, dtype=bool), np.ones(n_test_extrap, dtype=bool)])

    # Mahalanobis distance to training support
    mu_train = np.mean(X_train, axis=0)
    cov_train = np.cov(X_train, rowvar=False) + 1e-3 * np.eye(d)
    cov_inv = np.linalg.inv(cov_train)
    diff = X_test_all - mu_train
    mahalanobis_d = np.sqrt(np.sum((diff @ cov_inv) * diff, axis=1))

    return {
        "beta_star": beta_star,
        "X_train": X_train,
        "y_train": y_train,
        "X_test_in": X_test_in,
        "y_test_in": y_test_in,
        "z_test_in": z_test_in,
        "X_test_extrap": X_test_extrap,
        "y_test_extrap": y_test_extrap,
        "z_test_extrap": z_test_extrap,
        "X_test_all": X_test_all,
        "y_test_all": y_test_all,
        "z_test_all": z_test_all,
        "is_extrap": is_extrap,
        "mahalanobis_d": mahalanobis_d,
    }


def compute_selective_risk_coverage(
    errors: np.ndarray,
    uncertainties: np.ndarray,
    coverages: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute selective risk curve and Area Under Risk-Coverage (AU-RC)."""
    if coverages is None:
        coverages = np.linspace(0.2, 1.0, 20)

    n = len(errors)
    order = np.argsort(uncertainties)  # low uncertainty first
    sorted_errors = errors[order]

    risks = []
    for c in coverages:
        k = max(int(np.ceil(c * n)), 1)
        risk = float(np.mean(sorted_errors[:k]))
        risks.append(risk)

    risks_arr = np.array(risks)
    # Area under risk-coverage normalized over coverage span [0.2, 1.0]
    aurc = float(np.trapezoid(risks_arr, coverages) / (coverages[-1] - coverages[0]))
    return coverages, risks_arr, aurc


def evaluate_estimator_diagnostics(
    uncertainty_scores: np.ndarray,
    pred_probs: np.ndarray,
    y_true: np.ndarray,
    mahalanobis_d: np.ndarray,
    is_extrap: np.ndarray,
    latent_ci: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    z_true: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate multi-faceted uncertainty diagnostics."""
    # 1. Support Sensitivity: Spearman rank correlation with Mahalanobis distance
    rho_s, _ = stats.spearmanr(uncertainty_scores, mahalanobis_d)

    # OOD Discrimination AUROC
    ood_auc = float(roc_auc_score(is_extrap.astype(int), uncertainty_scores))

    # 2. Credible Interval Coverage (if latent CI and ground truth are provided)
    cov_in = float("nan")
    cov_extrap = float("nan")
    if latent_ci is not None and z_true is not None:
        ci_lower, ci_upper = latent_ci
        covered = (z_true >= ci_lower) & (z_true <= ci_upper)
        cov_in = float(np.mean(covered[~is_extrap]))
        cov_extrap = float(np.mean(covered[is_extrap]))

    # 3. Selective Risk-Coverage (Brier error)
    brier_errors = (pred_probs - y_true) ** 2
    _, _, aurc = compute_selective_risk_coverage(brier_errors, uncertainty_scores)

    # 4. Error Identification: Partial correlation controlling for boundary distance
    conf = np.abs(pred_probs - 0.5)
    if np.all(conf == conf[0]) or np.all(brier_errors == brier_errors[0]) or np.all(uncertainty_scores == uncertainty_scores[0]):
        partial_r = 0.0
    else:
        r_xy, _ = stats.pearsonr(uncertainty_scores, brier_errors)
        r_xz, _ = stats.pearsonr(uncertainty_scores, conf)
        r_yz, _ = stats.pearsonr(brier_errors, conf)
        denom = np.sqrt(max(1.0 - r_xz**2, 1e-12)) * np.sqrt(max(1.0 - r_yz**2, 1e-12))
        partial_r = float((r_xy - r_xz * r_yz) / denom)

    return {
        "support_spearman_rho": float(rho_s),
        "ood_auroc": float(ood_auc),
        "interval_coverage_in": float(cov_in),
        "interval_coverage_extrap": float(cov_extrap),
        "aurc_brier": float(aurc),
        "error_partial_corr": float(partial_r),
    }
