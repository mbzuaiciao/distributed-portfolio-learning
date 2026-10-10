"""Bayesian Uncertainty Estimation and Specialist Routing for Experiment 3.

Implements:
1. BayesianPenultimateReadout: Laplace approximation on penultimate representations.
2. LinearMoERouter: Deterministic linear gating network.
3. MLPMoERouter: Deterministic non-linear MLP gating network.
4. DensityRouter: Kernel / Gaussian density in shared feature space.
5. ConfidenceRouter: Predictive confidence gating (|p - 0.5|).
6. InverseVarianceRouter: Dynamic precision weighting with bias intercept.
7. MatchedRouterAblation: Matched router with and without epistemic variance inputs.
8. select_strongest_nonbayes_baseline: Pre-registered deterministic selection rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hypergraph_portfolio.synthetic_models import SmallMLP
from hypergraph_portfolio.synthetic_multitask import _sigmoid


@dataclass
class BayesianPenultimateReadout:
    """Laplace approximation over the top-layer weights of a SmallMLP."""

    weight_map: np.ndarray  # (16,)
    bias_map: float
    covariance: np.ndarray  # (16, 16)
    condition_number: float

    @classmethod
    def fit(
        cls,
        model: SmallMLP,
        x_train: np.ndarray,
        prior_precision: float = 1.0,
    ) -> BayesianPenultimateReadout:
        """Compute Laplace Hessian and posterior covariance on penultimate features."""
        model.eval()
        xt = torch.from_numpy(x_train).float()
        with torch.no_grad():
            # SmallMLP: fc1(x) -> relu -> fc2 -> relu -> phi
            h = F.relu(model.fc1(xt))
            phi = F.relu(model.fc2(h)).numpy()  # (N, 16)
            logits = model.fc3(F.relu(model.fc2(h))).squeeze(-1).numpy()
            probs = _sigmoid(logits)

        # Top-layer MAP weights: model.fc3 is Linear(16, 1)
        w_map = model.fc3.weight.detach().cpu().numpy().squeeze(0)  # (16,)
        b_map = float(model.fc3.bias.detach().cpu().item())

        # Fisher Information Matrix / Hessian of BCE loss: H = sum_n p_n (1 - p_n) phi_n phi_n^T
        n_samples = len(x_train)
        weights_var = probs * (1.0 - probs)  # (N,)
        weighted_phi = phi * np.sqrt(weights_var)[:, None]
        Hessian = weighted_phi.T @ weighted_phi  # (16, 16)

        # Regularized inverse: Sigma = (H + lambda * I)^(-1)
        dim = phi.shape[1]
        regularized_H = Hessian + prior_precision * np.eye(dim)

        eigvals, eigvecs = np.linalg.eigh(regularized_H)
        cond_num = float(eigvals[-1] / max(eigvals[0], 1e-12))

        # Invert with eigenvalue floor to guarantee strict positive definiteness
        min_eigval = max(float(eigvals[0]), 1e-4 * float(eigvals[-1]))
        inv_eigvals = 1.0 / np.clip(eigvals, min_eigval, None)
        cov = (eigvecs * inv_eigvals) @ eigvecs.T

        return cls(
            weight_map=w_map,
            bias_map=b_map,
            covariance=cov,
            condition_number=cond_num,
        )

    def predict(self, model: SmallMLP, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute latent mean, epistemic variance, and Bayesian probit-approximated probability.

        Returns:
            mu: predictive latent logit mean (N,)
            sigma2: predictive epistemic variance (N,)
            p_bayes: probit-approximated predictive probability (N,)
        """
        model.eval()
        xt = torch.from_numpy(x).float()
        with torch.no_grad():
            h = F.relu(model.fc1(xt))
            phi = F.relu(model.fc2(h)).numpy()  # (N, 16)

        # Latent logit mean: phi @ w + b
        mu = phi @ self.weight_map + self.bias_map

        # Epistemic variance: phi @ Sigma @ phi^T (diagonal elements)
        # Vectorized: sum((phi @ cov) * phi, axis=1)
        sigma2 = np.sum((phi @ self.covariance) * phi, axis=1)
        sigma2 = np.maximum(sigma2, 0.0)

        # Probit-approximated Bayesian predictive probability:
        # sigma(mu / sqrt(1 + (pi/8) * sigma2))
        kappa = 1.0 / np.sqrt(1.0 + (math.pi / 8.0) * sigma2)
        p_bayes = _sigmoid(mu * kappa)

        return mu, sigma2, p_bayes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight_map": self.weight_map.tolist(),
            "bias_map": self.bias_map,
            "covariance": self.covariance.tolist(),
            "condition_number": self.condition_number,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BayesianPenultimateReadout:
        return cls(
            weight_map=np.array(d["weight_map"]),
            bias_map=d["bias_map"],
            covariance=np.array(d["covariance"]),
            condition_number=d["condition_number"],
        )


class LinearMoERouter(nn.Module):
    """Deterministic linear mixture-of-experts gating on shared features x_s."""

    def __init__(self, in_features: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, 2)

    def forward(self, x_s: torch.Tensor) -> torch.Tensor:
        """Returns softmax gating weights (N, 2)."""
        return F.softmax(self.fc(x_s), dim=-1)

    @classmethod
    def fit(
        cls,
        x_s: np.ndarray,
        p_A: np.ndarray,
        p_B: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        seed: int = 42,
    ) -> LinearMoERouter:
        torch.manual_seed(seed)
        router = cls(in_features=x_s.shape[1])
        optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()

        xt_s = torch.from_numpy(x_s).float()
        pt_A = torch.from_numpy(p_A).float()
        pt_B = torch.from_numpy(p_B).float()
        yt = torch.from_numpy(y).float()

        router.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            weights = router(xt_s)  # (N, 2)
            p_comb = weights[:, 0] * pt_A + weights[:, 1] * pt_B
            p_comb = torch.clamp(p_comb, 1e-6, 1.0 - 1e-6)
            loss = criterion(p_comb, yt)
            loss.backward()
            optimizer.step()

        router.eval()
        return router

    def predict(self, x_s: np.ndarray, p_A: np.ndarray, p_B: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            weights = self(torch.from_numpy(x_s).float()).numpy()
            return weights[:, 0] * p_A + weights[:, 1] * p_B


class MLPMoERouter(nn.Module):
    """Deterministic non-linear MLP gating network on shared features x_s."""

    def __init__(self, in_features: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x_s: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x_s))
        return F.softmax(self.fc2(h), dim=-1)

    @classmethod
    def fit(
        cls,
        x_s: np.ndarray,
        p_A: np.ndarray,
        p_B: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        seed: int = 42,
    ) -> MLPMoERouter:
        torch.manual_seed(seed)
        router = cls(in_features=x_s.shape[1], hidden_dim=16)
        optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()

        xt_s = torch.from_numpy(x_s).float()
        pt_A = torch.from_numpy(p_A).float()
        pt_B = torch.from_numpy(p_B).float()
        yt = torch.from_numpy(y).float()

        router.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            weights = router(xt_s)
            p_comb = weights[:, 0] * pt_A + weights[:, 1] * pt_B
            p_comb = torch.clamp(p_comb, 1e-6, 1.0 - 1e-6)
            loss = criterion(p_comb, yt)
            loss.backward()
            optimizer.step()

        router.eval()
        return router

    def predict(self, x_s: np.ndarray, p_A: np.ndarray, p_B: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            weights = self(torch.from_numpy(x_s).float()).numpy()
            return weights[:, 0] * p_A + weights[:, 1] * p_B


@dataclass
class DensityRouter:
    """Non-Bayesian router based on empirical Gaussian density in shared feature space."""

    mu_A: np.ndarray
    cov_inv_A: np.ndarray
    mu_B: np.ndarray
    cov_inv_B: np.ndarray

    @classmethod
    def fit(cls, x_s_train_A: np.ndarray, x_s_train_B: np.ndarray) -> DensityRouter:
        mu_A = np.mean(x_s_train_A, axis=0)
        cov_A = np.cov(x_s_train_A, rowvar=False) + 1e-3 * np.eye(x_s_train_A.shape[1])
        cov_inv_A = np.linalg.inv(cov_A)

        mu_B = np.mean(x_s_train_B, axis=0)
        cov_B = np.cov(x_s_train_B, rowvar=False) + 1e-3 * np.eye(x_s_train_B.shape[1])
        cov_inv_B = np.linalg.inv(cov_B)

        return cls(mu_A=mu_A, cov_inv_A=cov_inv_A, mu_B=mu_B, cov_inv_B=cov_inv_B)

    def predict(self, x_s: np.ndarray, p_A: np.ndarray, p_B: np.ndarray) -> np.ndarray:
        diff_A = x_s - self.mu_A
        diff_B = x_s - self.mu_B
        d2_A = np.sum((diff_A @ self.cov_inv_A) * diff_A, axis=1)
        d2_B = np.sum((diff_B @ self.cov_inv_B) * diff_B, axis=1)

        # Log density ratio
        dens_A = np.exp(-0.5 * d2_A)
        dens_B = np.exp(-0.5 * d2_B)
        total = dens_A + dens_B + 1e-12
        w_A = dens_A / total
        w_B = dens_B / total
        return w_A * p_A + w_B * p_B

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mu_A": self.mu_A.tolist(),
            "cov_inv_A": self.cov_inv_A.tolist(),
            "mu_B": self.mu_B.tolist(),
            "cov_inv_B": self.cov_inv_B.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DensityRouter:
        return cls(
            mu_A=np.array(d["mu_A"]),
            cov_inv_A=np.array(d["cov_inv_A"]),
            mu_B=np.array(d["mu_B"]),
            cov_inv_B=np.array(d["cov_inv_B"]),
        )


@dataclass
class ConfidenceRouter:
    """Non-Bayesian heuristic router weighting specialists by distance to decision boundary."""

    @staticmethod
    def predict(p_A: np.ndarray, p_B: np.ndarray) -> np.ndarray:
        conf_A = np.abs(p_A - 0.5) + 1e-6
        conf_B = np.abs(p_B - 0.5) + 1e-6
        w_A = conf_A / (conf_A + conf_B)
        w_B = conf_B / (conf_A + conf_B)
        return w_A * p_A + w_B * p_B


@dataclass
class InverseVarianceRouter:
    """Precision weighting in logit space with learned global intercept to relax unbiased assumptions."""

    intercept: float = 0.0

    @classmethod
    def fit(cls, p_A: np.ndarray, p_B: np.ndarray, s2_A: np.ndarray, s2_B: np.ndarray, y: np.ndarray) -> InverseVarianceRouter:
        pA = np.clip(p_A, 1e-6, 1.0 - 1e-6)
        pB = np.clip(p_B, 1e-6, 1.0 - 1e-6)
        logitA = np.log(pA / (1.0 - pA))
        logitB = np.log(pB / (1.0 - pB))

        precA = 1.0 / np.maximum(s2_A, 1e-6)
        precB = 1.0 / np.maximum(s2_B, 1e-6)
        total_prec = precA + precB
        wA = precA / total_prec
        wB = precB / total_prec
        z_comb = wA * logitA + wB * logitB

        # Fit single intercept b to minimize BCE
        from scipy.optimize import minimize_scalar

        def bce_loss(b: float) -> float:
            p = _sigmoid(z_comb + b)
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        res = minimize_scalar(bce_loss, bracket=[-5.0, 5.0])
        return cls(intercept=float(res.x))

    def predict(self, p_A: np.ndarray, p_B: np.ndarray, s2_A: np.ndarray, s2_B: np.ndarray) -> np.ndarray:
        pA = np.clip(p_A, 1e-6, 1.0 - 1e-6)
        pB = np.clip(p_B, 1e-6, 1.0 - 1e-6)
        logitA = np.log(pA / (1.0 - pA))
        logitB = np.log(pB / (1.0 - pB))

        precA = 1.0 / np.maximum(s2_A, 1e-6)
        precB = 1.0 / np.maximum(s2_B, 1e-6)
        total = precA + precB
        wA = precA / total
        wB = precB / total
        z = self.intercept + wA * logitA + wB * logitB
        return _sigmoid(z)

    def to_dict(self) -> Dict[str, Any]:
        return {"intercept": float(self.intercept)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> InverseVarianceRouter:
        return cls(intercept=float(d["intercept"]))


class MatchedAblationRouter(nn.Module):
    """Matched router for uncertainty feature ablation (trained with or without epistemic variance)."""

    def __init__(self, in_features: int, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h).squeeze(-1))

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.005,
        weight_decay: float = 0.01,
        seed: int = 42,
    ) -> MatchedAblationRouter:
        torch.manual_seed(seed)
        model = cls(in_features=features.shape[1], hidden_dim=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()

        xt = torch.from_numpy(features).float()
        yt = torch.from_numpy(y).float()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = model(xt)
            loss = criterion(preds, yt)
            loss.backward()
            optimizer.step()

        model.eval()
        return model

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self(torch.from_numpy(features).float()).numpy()


def select_strongest_nonbayes_baseline(
    dev_summary: Dict[str, Dict[str, float]],
) -> str:
    """Pre-registered deterministic selection hierarchy for strongest non-Bayesian comparator.

    Eligible Pool:
        M0, M1_cap, M6_cal, M_conf, M_dens, M_moe_lin, M_moe_mlp, M_router_nouq
    Hierarchy:
        1. Lowest mean development Brier.
        2. Highest mean development AUROC (if Brier diff < 1e-5).
        3. Pre-registered priority order.
    """
    eligible = [
        "m_moe_mlp",
        "m_router_nouq",
        "m_moe_lin",
        "m_dens",
        "m_conf",
        "m6_logodds_cal",
        "m1_cap_pooled",
        "m0_target",
    ]
    # Filter only methods present in dev_summary
    candidates = [m for m in eligible if m in dev_summary]
    if not candidates:
        raise ValueError("No eligible non-Bayesian candidates found in dev summary")

    # Sort primarily by brier ascending, secondarily by -auroc
    def sort_key(name: str):
        brier = dev_summary[name]["mean_brier"]
        auroc = dev_summary[name]["mean_auroc"]
        priority_index = eligible.index(name)
        # Round brier to 5 decimals for tie-breaking
        return (round(brier, 5), -round(auroc, 5), priority_index)

    return sorted(candidates, key=sort_key)[0]
