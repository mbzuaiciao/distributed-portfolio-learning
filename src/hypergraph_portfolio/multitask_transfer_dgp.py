"""Multi-Task LoRA Transferability Data-Generating Process and Baseline Models.

Implements:
1. Controlled multi-task factor DGP: Positive Control (strong shared subspace)
   and Negative Control (orthogonal/unrelated task parameters).
2. Five-split data architecture: Source train (x3), Target train (N=64),
   Target calibration (N=128), Target development (N=256), Target test (N=512, quarantined).
3. Matched model suite:
   - Target-Only Logistic
   - Target-Only LoRA
   - Pooled Global LoRA
   - 3-Source Independent Specialists + Calibrated Fusion (M_fuse)
   - Simple Shared-Subspace Multi-Task Linear Model
   - Structural CoLoRA
   - Prevalence Null & Bayes Oracle
4. Balanced Multi-Task Batch Sampler (500 matched gradient steps).
5. Grassmannian subspace recovery diagnostics and cosine weight similarity.

Strictly enforces CPU execution with PyTorch limited to two threads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hypergraph_portfolio.synthetic_multitask import _sigmoid

# Host safeguard: strictly limit PyTorch CPU threads to 2
torch.set_num_threads(2)


def generate_multitask_transfer_dgp(
    seed: int,
    regime: str = "positive_control",
    d: int = 32,
    r: int = 2,
    gamma: float = 0.05,
    n_source: int = 512,
    n_target_train: int = 64,
    n_target_cal: int = 128,
    n_target_dev: int = 256,
    n_target_test: int = 512,
) -> Dict[str, Any]:
    """Generate multi-task transfer dataset with positive or negative shared subspace control."""
    rng = np.random.default_rng(seed)

    if regime == "positive_control":
        # Draw random orthogonal basis U in R^{d x r}
        random_matrix = rng.normal(size=(d, r))
        U_true, _ = np.linalg.qr(random_matrix)

        # 4 tasks: Source 1, Source 2, Source 3, Target
        task_names = ["source_1", "source_2", "source_3", "target"]
        betas = {}
        alphas = {}
        for name in task_names:
            alpha = rng.normal(size=r)
            alpha = alpha / np.linalg.norm(alpha)
            alphas[name] = alpha

            eps = rng.normal(size=d)
            eps = eps / np.linalg.norm(eps)

            beta_raw = U_true @ alpha + gamma * eps
            beta = 1.5 * (beta_raw / np.linalg.norm(beta_raw))
            betas[name] = beta
    elif regime == "negative_control":
        # Completely independent task parameter vectors
        U_true = None
        task_names = ["source_1", "source_2", "source_3", "target"]
        betas = {}
        alphas = {}
        for name in task_names:
            beta_raw = rng.normal(size=d)
            beta = 1.5 * (beta_raw / np.linalg.norm(beta_raw))
            betas[name] = beta
            alphas[name] = None
    else:
        raise ValueError(f"Unknown regime: {regime}")

    # Generate source datasets (N = 512 each)
    source_data = {}
    for s_name in ["source_1", "source_2", "source_3"]:
        X = rng.normal(0.0, 1.0, size=(n_source, d))
        p = _sigmoid(X @ betas[s_name])
        y = (rng.uniform(size=n_source) < p).astype(float)
        source_data[s_name] = {"X": X, "y": y, "p_true": p}

    # Generate target splits
    def make_target_split(n_samples: int) -> Dict[str, np.ndarray]:
        X = rng.normal(0.0, 1.0, size=(n_samples, d))
        p = _sigmoid(X @ betas["target"])
        y = (rng.uniform(size=n_samples) < p).astype(float)
        return {"X": X, "y": y, "p_true": p}

    target_train = make_target_split(n_target_train)
    target_cal = make_target_split(n_target_cal)
    target_dev = make_target_split(n_target_dev)
    target_test = make_target_split(n_target_test)

    return {
        "regime": regime,
        "seed": seed,
        "d": d,
        "r": r,
        "gamma": gamma,
        "U_true": U_true,
        "betas": betas,
        "alphas": alphas,
        "source_data": source_data,
        "target_train": target_train,
        "target_cal": target_cal,
        "target_dev": target_dev,
        "target_test": target_test,
    }


# ---------------- MODEL ARCHITECTURES ----------------


class TargetOnlyLogistic(nn.Module):
    """Target-only full-rank L2-regularized logistic regression (d+1 parameters)."""

    def __init__(self, in_features: int = 32):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X).float()
            return _sigmoid(self(xt).numpy())

    def get_effective_weight(self) -> np.ndarray:
        return self.linear.weight.detach().squeeze(0).numpy()


class TargetOnlyLoRA(nn.Module):
    """Low-rank LoRA adapter on target task alone (r * d + r + 1 parameters)."""

    def __init__(self, in_features: int = 32, rank: int = 2):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, rank) * 0.1)
        self.A = nn.Parameter(torch.randn(rank, 1) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.matmul(self.B, self.A).squeeze(-1)  # (d,)
        return torch.matmul(x, w) + self.bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X).float()
            return _sigmoid(self(xt).numpy())

    def get_effective_weight(self) -> np.ndarray:
        return torch.matmul(self.B, self.A).detach().squeeze(-1).numpy()


class PooledLoRA(nn.Module):
    """Single pooled LoRA adapter trained on union of all source and target samples."""

    def __init__(self, in_features: int = 32, rank: int = 2):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, rank) * 0.1)
        self.A = nn.Parameter(torch.randn(rank, 1) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.matmul(self.B, self.A).squeeze(-1)
        return torch.matmul(x, w) + self.bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X).float()
            return _sigmoid(self(xt).numpy())

    def get_effective_weight(self) -> np.ndarray:
        return torch.matmul(self.B, self.A).detach().squeeze(-1).numpy()


@dataclass
class CalibratedLogOddsFusion3:
    """Combines 3 source specialists using multivariate logistic regression on target calibration split."""

    weights: np.ndarray  # (3,)
    intercept: float

    @classmethod
    def fit(
        cls,
        p_sources: np.ndarray,  # (N, 3) probabilities
        y_cal: np.ndarray,  # (N,) binary labels
        alpha: float = 1.0,
    ) -> CalibratedLogOddsFusion3:
        n = p_sources.shape[0]
        p_clip = np.clip(p_sources, 1e-6, 1.0 - 1e-6)
        logits = np.log(p_clip / (1.0 - p_clip))  # (N, 3)
        X_tilde = np.column_stack([logits, np.ones(n)])

        # Fit via Newton-Raphson L2 regularized logistic regression
        w = np.zeros(4)
        prior_prec = alpha * np.eye(4)
        prior_prec[-1, -1] = 1e-4

        for _ in range(30):
            p = _sigmoid(X_tilde @ w)
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            grad = X_tilde.T @ (p - y_cal) + prior_prec @ w
            W = p * (1.0 - p)
            H = (X_tilde.T * W) @ X_tilde + prior_prec
            delta_w = np.linalg.solve(H, grad)
            w = w - delta_w
            if np.linalg.norm(delta_w) < 1e-6:
                break

        return cls(weights=w[:3], intercept=float(w[-1]))

    def predict_proba(self, p_sources: np.ndarray) -> np.ndarray:
        p_clip = np.clip(p_sources, 1e-6, 1.0 - 1e-6)
        logits = np.log(p_clip / (1.0 - p_clip))
        z = logits @ self.weights + self.intercept
        return _sigmoid(z)


class SharedSubspaceMultiTask(nn.Module):
    """Simple low-rank multi-task baseline: shared basis U in R^{d x r} and task vectors v_k in R^r."""

    def __init__(self, in_features: int = 32, rank: int = 2, n_tasks: int = 4):
        super().__init__()
        self.U = nn.Parameter(torch.randn(in_features, rank) * 0.1)
        self.v = nn.Parameter(torch.randn(n_tasks, rank) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n_tasks))

    def forward_task(self, x: torch.Tensor, task_idx: int) -> torch.Tensor:
        v_k = self.v[task_idx]  # (r,)
        w_k = torch.matmul(self.U, v_k)  # (d,)
        return torch.matmul(x, w_k) + self.bias[task_idx]

    def predict_proba_task(self, X: np.ndarray, task_idx: int) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X).float()
            return _sigmoid(self.forward_task(xt, task_idx).numpy())

    def get_effective_weight(self, task_idx: int) -> np.ndarray:
        v_k = self.v[task_idx]
        return torch.matmul(self.U, v_k).detach().numpy()

    def get_shared_subspace(self) -> np.ndarray:
        return self.U.detach().numpy()


class StructuralCoLoRA(nn.Module):
    """Structural CoLoRA: shared B in R^{d x r}, A in R^{r x 1}, task Lambda_k in R^{r x r}."""

    def __init__(self, in_features: int = 32, rank: int = 2, n_tasks: int = 4):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, rank) * 0.1)
        self.A = nn.Parameter(torch.randn(rank, 1) * 0.1)
        # Task-specific diagonal/coupling matrices Lambda_k
        self.Lambdas = nn.Parameter(torch.stack([torch.eye(rank) for _ in range(n_tasks)]))
        self.bias = nn.Parameter(torch.zeros(n_tasks))

    def forward_task(self, x: torch.Tensor, task_idx: int) -> torch.Tensor:
        lam = self.Lambdas[task_idx]  # (r, r)
        w_k = torch.matmul(self.B, torch.matmul(lam, self.A)).squeeze(-1)  # (d,)
        return torch.matmul(x, w_k) + self.bias[task_idx]

    def predict_proba_task(self, X: np.ndarray, task_idx: int) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X).float()
            return _sigmoid(self.forward_task(xt, task_idx).numpy())

    def get_effective_weight(self, task_idx: int) -> np.ndarray:
        lam = self.Lambdas[task_idx]
        return torch.matmul(self.B, torch.matmul(lam, self.A)).detach().squeeze(-1).numpy()

    def get_shared_subspace(self) -> np.ndarray:
        return self.B.detach().numpy()


# ---------------- SUBSPACE & WEIGHT DIAGNOSTICS ----------------


def compute_subspace_alignment(U_hat: np.ndarray, U_true: Optional[np.ndarray]) -> float:
    """Grassmannian projection alignment S_proj in [0, 1]."""
    if U_true is None:
        return float("nan")
    # Orthonormalize U_hat via QR
    Q_hat, _ = np.linalg.qr(U_hat)
    r = U_true.shape[1]
    # Projection alignment: Tr(U_true^T Q_hat Q_hat^T U_true) / r
    overlap = Q_hat.T @ U_true
    score = float(np.sum(overlap**2) / r)
    return float(np.clip(score, 0.0, 1.0))


def compute_principal_angles(U_hat: np.ndarray, U_true: Optional[np.ndarray]) -> np.ndarray:
    """Return cosines of principal angles between spans of U_hat and U_true."""
    if U_true is None:
        return np.array([float("nan")])
    Q_hat, _ = np.linalg.qr(U_hat)
    overlap = Q_hat.T @ U_true
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    return np.clip(singular_values, 0.0, 1.0)


def compute_weight_cosine_similarity(w_hat: np.ndarray, w_true: np.ndarray) -> float:
    """Cosine similarity between estimated and true parameter vectors."""
    norm_hat = np.linalg.norm(w_hat)
    norm_true = np.linalg.norm(w_true)
    if norm_hat < 1e-12 or norm_true < 1e-12:
        return 0.0
    return float(np.dot(w_hat, w_true) / (norm_hat * norm_true))
