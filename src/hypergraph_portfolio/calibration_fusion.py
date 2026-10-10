"""Calibration and Evidence Fusion modules for Experiment 2.

Implements:
1. PlattCalibrator (temperature/Platt scaling for probability calibration).
2. LogOddsFusion (bivariate log-odds evidence combination).
3. SimplexFusion (SLSQP Brier-minimizing probability mixture).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression

from hypergraph_portfolio.collaborative_lora import combine_predictions, fit_simplex_weights


@dataclass
class PlattCalibrator:
    """1D Platt / temperature scaling model: P(y=1 | p) = sigma(a * logit(p) + b)."""

    weight: float = 1.0
    intercept: float = 0.0

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
        logits = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
        clf = LogisticRegression(C=1e9, solver="lbfgs")
        clf.fit(logits, y)
        return cls(weight=float(clf.coef_[0, 0]), intercept=float(clf.intercept_[0]))

    def predict(self, p: np.ndarray) -> np.ndarray:
        p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
        logits = np.log(p_clip / (1.0 - p_clip))
        z = self.weight * logits + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def to_dict(self) -> Dict[str, float]:
        return {"weight": self.weight, "intercept": self.intercept}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> PlattCalibrator:
        return cls(weight=d["weight"], intercept=d["intercept"])


@dataclass
class LogOddsFusion:
    """Bivariate log-odds combination model: sigma(w_0 + w_A * logit(p_A) + w_B * logit(p_B))."""

    weights: List[float] = None
    intercept: float = 0.0

    @classmethod
    def fit(cls, p_A: np.ndarray, p_B: np.ndarray, y: np.ndarray) -> LogOddsFusion:
        pA = np.clip(p_A, 1e-6, 1.0 - 1e-6)
        pB = np.clip(p_B, 1e-6, 1.0 - 1e-6)
        logitA = np.log(pA / (1.0 - pA))
        logitB = np.log(pB / (1.0 - pB))
        X = np.column_stack([logitA, logitB])
        clf = LogisticRegression(C=1.0, solver="lbfgs")  # mild L2 regularization
        clf.fit(X, y)
        return cls(
            weights=[float(clf.coef_[0, 0]), float(clf.coef_[0, 1])],
            intercept=float(clf.intercept_[0]),
        )

    def predict(self, p_A: np.ndarray, p_B: np.ndarray) -> np.ndarray:
        pA = np.clip(p_A, 1e-6, 1.0 - 1e-6)
        pB = np.clip(p_B, 1e-6, 1.0 - 1e-6)
        logitA = np.log(pA / (1.0 - pA))
        logitB = np.log(pB / (1.0 - pB))
        z = self.intercept + self.weights[0] * logitA + self.weights[1] * logitB
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": self.weights, "intercept": self.intercept}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LogOddsFusion:
        return cls(weights=d["weights"], intercept=d["intercept"])


@dataclass
class SimplexFusion:
    """Simplex linear probability mixture: w_A * p_A + w_B * p_B with w >= 0, sum(w) = 1."""

    weights: List[float] = None

    @classmethod
    def fit(cls, p_A: np.ndarray, p_B: np.ndarray, y: np.ndarray) -> SimplexFusion:
        mat = np.column_stack([p_A, p_B])
        w = fit_simplex_weights(mat, y)
        return cls(weights=w.tolist())

    def predict(self, p_A: np.ndarray, p_B: np.ndarray) -> np.ndarray:
        mat = np.column_stack([p_A, p_B])
        return combine_predictions(mat, np.array(self.weights))

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": self.weights}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SimplexFusion:
        return cls(weights=d["weights"])
