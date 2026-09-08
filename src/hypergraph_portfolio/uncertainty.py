"""
Bayesian Uncertainty Extensions & Empirical Evaluation
======================================================
Provides mechanisms for incorporating predictive epistemic uncertainty
    p(mu | D) ~ ( E[mu], Var(mu) )
into portfolio allocation, alongside a controlled sequential backtest environment
to evaluate when and how epistemic uncertainty adds economic value.

Candidate Mechanisms:
1. Linear Uncertainty Penalty:  tilde{mu}_j = mu_hat_j - lambda_u * sigma_epi_j
2. Return Shrinkage:            tilde{mu}_j = mu_j / (1 + kappa * var_mu_j)
3. Local Risk Augmentation:     tilde{Q}_i  = Q_i + gamma * U_i
4. Importance Re-weighting:     tilde{a}_i  propto 1 / Trace(U_i)^alpha
5. Posterior Downside Penalty:  Direct tail risk adjustment based on P(mu_i < target)
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from hypergraph_portfolio.hypergraph import HypergraphPortfolio
from hypergraph_portfolio.mean_variance import MeanVariancePortfolio


def apply_uncertainty_shrinkage(
    mu_hat: np.ndarray,
    sigma_epi: np.ndarray,
    lambda_u: float = 1.0,
    mu_prior: float | np.ndarray = 0.0,
    mapping: str = "rational",
    floor: Optional[float] = None,
) -> np.ndarray:
    """
    Confidence-weighted posterior-mean shrinkage adjustment:
        tilde_mu_j = rho_j * mu_hat_j + (1 - rho_j) * mu_prior_j

    Where:
        0 <= rho_j <= 1
        d rho_j / d sigma_epi_j < 0

    Mappings:
        - 'rational':    rho_j = 1.0 / (1.0 + lambda_u * sigma_epi_j)
        - 'exponential': rho_j = exp(-lambda_u * sigma_epi_j)

    Properties:
        - sigma_epi = 0 => rho = 1 => tilde_mu = mu_hat (unchanged)
        - lambda_u = 0  => rho = 1 => tilde_mu = mu_hat (deterministic reproduction)
        - Both positive and negative forecasts shrink monotonically toward mu_prior.

    Args:
        mu_hat: Point forecast vector of expected returns.
        sigma_epi: Epistemic uncertainty standard deviation (sigma_epi >= 0).
        lambda_u: Uncertainty penalty parameter (lambda_u >= 0).
        mu_prior: Prior mean vector or scalar (default 0.0).
        mapping: 'rational' or 'exponential'.
        floor: Optional lower bound on adjusted returns.

    Returns:
        tilde_mu: Uncertainty-shrunk return vector.
    """
    mu_hat = np.asarray(mu_hat, dtype=float)
    sigma_epi = np.asarray(sigma_epi, dtype=float)
    lambda_u = float(lambda_u)

    if len(mu_hat) != len(sigma_epi):
        raise ValueError(f"Length mismatch: mu_hat ({len(mu_hat)}) vs sigma_epi ({len(sigma_epi)})")
    if np.any(sigma_epi < 0):
        raise ValueError("Epistemic uncertainty sigma_epi must be non-negative.")
    if lambda_u < 0:
        raise ValueError("Uncertainty penalty parameter lambda_u must be non-negative.")

    if mapping == "rational":
        rho = 1.0 / (1.0 + lambda_u * sigma_epi)
    elif mapping == "exponential":
        rho = np.exp(-lambda_u * sigma_epi)
    else:
        raise ValueError(f"Unknown mapping '{mapping}'. Choose 'rational' or 'exponential'.")

    mu_prior_arr = np.asarray(mu_prior, dtype=float)
    tilde_mu = rho * mu_hat + (1.0 - rho) * mu_prior_arr

    if floor is not None:
        tilde_mu = np.maximum(tilde_mu, floor)
    return tilde_mu


def apply_epistemic_covariance_inflation(
    Q: np.ndarray,
    epistemic_std: np.ndarray,
    lambda_cov: float = 1.0,
) -> np.ndarray:
    """
    Inflate asset covariance matrix by adding diagonal epistemic uncertainty risk (Method 2):
        Q_robust = Q + lambda_cov * diag(sigma_epi^2)

    Properties:
        - Symmetric: Q_robust = Q_robust^T
        - Positive Semidefinite: If Q is PSD and lambda_cov >= 0, Q_robust is PSD
        - lambda_cov = 0 reproduces the input covariance Q exactly
        - Monotonic: Higher epistemic uncertainty weakly increases diagonal variance
        - Preserves off-diagonal market covariances intact: Q_robust[i, j] = Q[i, j] for i != j

    Args:
        Q: Market covariance matrix of shape (m, m).
        epistemic_std: Epistemic uncertainty standard deviations of shape (m,).
        lambda_cov: Penalty intensity multiplier (>= 0).

    Returns:
        Q_robust: Symmetric, positive semidefinite robust covariance matrix.
    """
    Q = np.asarray(Q, dtype=float)
    epistemic_std = np.asarray(epistemic_std, dtype=float)
    lambda_cov = float(lambda_cov)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be a square 2D matrix, got shape {Q.shape}")
    m = Q.shape[0]
    if len(epistemic_std) != m:
        raise ValueError(
            f"Length of epistemic_std ({len(epistemic_std)}) must match Q dimension ({m})"
        )
    if np.any(epistemic_std < 0):
        raise ValueError("Epistemic uncertainty epistemic_std must be non-negative.")
    if lambda_cov < 0:
        raise ValueError(f"lambda_cov must be non-negative, got {lambda_cov}")

    if lambda_cov == 0.0:
        return Q.copy()

    var_epi = epistemic_std**2
    Q_robust = Q + lambda_cov * np.diag(var_epi)
    return 0.5 * (Q_robust + Q_robust.T)


def compute_posterior_downside_probability(
    mu_hat: np.ndarray,
    epistemic_std: np.ndarray,
    r_hurdle: float = 0.0,
) -> np.ndarray:
    """
    Compute posterior downside probability for each asset (Method 3/4):
        p_down_j = P(mu_j < r_hurdle | D)

    Under the Gaussian predictive distribution N(mu_hat_j, sigma_epi_j^2):
        p_down_j = Phi((r_hurdle - mu_hat_j) / sigma_epi_j)

    With numerically stable handling for sigma_epi -> 0:
        - if mu_hat_j > r_hurdle: p_down_j = 0.0
        - if mu_hat_j < r_hurdle: p_down_j = 1.0
        - if mu_hat_j == r_hurdle: p_down_j = 0.5

    Args:
        mu_hat: Point forecast vector of expected returns (shape m).
        epistemic_std: Epistemic uncertainty standard deviation vector (shape m).
        r_hurdle: Downside threshold hurdle return (default 0.0).

    Returns:
        p_down: Vector of posterior downside probabilities in [0, 1].
    """
    mu_hat = np.asarray(mu_hat, dtype=float)
    epistemic_std = np.asarray(epistemic_std, dtype=float)
    r_hurdle = float(r_hurdle)

    if len(mu_hat) != len(epistemic_std):
        raise ValueError(
            f"Length mismatch: mu_hat ({len(mu_hat)}) vs epistemic_std ({len(epistemic_std)})"
        )
    if np.any(epistemic_std < 0):
        raise ValueError("Epistemic uncertainty epistemic_std must be non-negative.")

    p_down = np.zeros_like(mu_hat, dtype=float)
    eps = 1e-12

    for j in range(len(mu_hat)):
        std_j = epistemic_std[j]
        diff_j = r_hurdle - mu_hat[j]
        if std_j < eps:
            if diff_j > 0:
                p_down[j] = 1.0
            elif diff_j < 0:
                p_down[j] = 0.0
            else:
                p_down[j] = 0.5
        else:
            z_j = diff_j / std_j
            p_down[j] = float(norm.cdf(z_j))

    return np.clip(p_down, 0.0, 1.0)


def apply_downside_adjusted_alpha(
    mu_hat: np.ndarray,
    p_down: np.ndarray,
    lambda_down: float = 1.0,
    mapping: str = "linear",
) -> np.ndarray:
    """
    Downside-adjusted expected return (Method 3):
        tilde_mu_j = mu_hat_j * g(p_down_j)

    Where g(p) is a bounded monotonic dampening factor:
        - 'linear':      g(p) = max(0.0, 1.0 - lambda_down * p)
        - 'exponential': g(p) = exp(-lambda_down * p)

    Args:
        mu_hat: Point forecast vector of expected returns.
        p_down: Posterior downside probability vector in [0, 1].
        lambda_down: Penalty multiplier (>= 0).
        mapping: 'linear' or 'exponential'.

    Returns:
        tilde_mu: Downside-adjusted expected returns.
    """
    mu_hat = np.asarray(mu_hat, dtype=float)
    p_down = np.asarray(p_down, dtype=float)
    lambda_down = float(lambda_down)

    if len(mu_hat) != len(p_down):
        raise ValueError(f"Length mismatch: mu_hat ({len(mu_hat)}) vs p_down ({len(p_down)})")
    if np.any((p_down < 0) | (p_down > 1)):
        raise ValueError("p_down probabilities must be within [0, 1].")
    if lambda_down < 0:
        raise ValueError("lambda_down must be non-negative.")

    if mapping == "linear":
        g = np.maximum(0.0, 1.0 - lambda_down * p_down)
    elif mapping == "exponential":
        g = np.exp(-lambda_down * p_down)
    else:
        raise ValueError(f"Unknown mapping '{mapping}'. Choose 'linear' or 'exponential'.")

    return mu_hat * g


def apply_downside_covariance_inflation(
    Q: np.ndarray,
    p_down: np.ndarray,
    epistemic_std: np.ndarray,
    lambda_down: float = 1.0,
    scale_by_variance: bool = True,
) -> np.ndarray:
    """
    Downside-weighted covariance risk inflation (Method 4):
        Q_down = Q + lambda_down * diag(p_down * sigma_epi^2)  [if scale_by_variance]
        or
        Q_down = Q + lambda_down * diag(p_down)                [if not scale_by_variance]

    Args:
        Q: Market covariance matrix of shape (m, m).
        p_down: Posterior downside probabilities of shape (m,).
        epistemic_std: Epistemic standard deviations of shape (m,).
        lambda_down: Penalty intensity multiplier (>= 0).
        scale_by_variance: If True, scales penalty by sigma_epi^2.

    Returns:
        Q_down: Symmetric, positive semi-definite downside-penalized covariance matrix.
    """
    Q = np.asarray(Q, dtype=float)
    p_down = np.asarray(p_down, dtype=float)
    epistemic_std = np.asarray(epistemic_std, dtype=float)
    lambda_down = float(lambda_down)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be a square 2D matrix, got shape {Q.shape}")
    m = Q.shape[0]
    if len(p_down) != m or len(epistemic_std) != m:
        raise ValueError(
            f"Dimensions mismatch: Q ({m}x{m}), p_down ({len(p_down)}), epistemic_std ({len(epistemic_std)})"
        )
    if np.any((p_down < 0) | (p_down > 1)):
        raise ValueError("p_down probabilities must be within [0, 1].")
    if np.any(epistemic_std < 0):
        raise ValueError("epistemic_std must be non-negative.")
    if lambda_down < 0:
        raise ValueError("lambda_down must be non-negative.")

    if lambda_down == 0.0:
        return Q.copy()

    if scale_by_variance:
        penalty = p_down * (epistemic_std**2)
    else:
        penalty = p_down

    Q_down = Q + lambda_down * np.diag(penalty)
    return 0.5 * (Q_down + Q_down.T)


@dataclass
class BayesianSignal:
    """Represents expected returns and epistemic uncertainty."""

    expected_return: np.ndarray  # E[mu], shape (m,)
    uncertainty_variance: np.ndarray  # Var(mu), shape (m,)
    asset_names: Optional[List[str]] = None

    def __post_init__(self):
        self.expected_return = np.asarray(self.expected_return, dtype=float)
        self.uncertainty_variance = np.asarray(self.uncertainty_variance, dtype=float)
        if len(self.expected_return) != len(self.uncertainty_variance):
            raise ValueError("Expected return and uncertainty variance must have same length.")
        if np.any(self.uncertainty_variance < 0):
            raise ValueError("Uncertainty variance cannot be negative.")


class BayesianReturnShrinkage:
    """
    Confidence-weighted return shrinkage:
        tilde{mu}_j = mu_j / (1 + kappa * sigma_{mu, j}^2)
    """

    def __init__(self, kappa: float = 1.0):
        self.kappa = float(kappa)

    def apply(self, mu: np.ndarray, var_mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=float)
        var_mu = np.asarray(var_mu, dtype=float)
        shrinkage_factors = 1.0 / (1.0 + self.kappa * var_mu)
        return mu * shrinkage_factors


class BayesianRiskAugmentation:
    """
    Local Risk Matrix Augmentation:
        tilde{Q}_i = Q_i + gamma * diag(var_mu_i)
    """

    def __init__(self, gamma: float = 1.0):
        self.gamma = float(gamma)

    def apply(self, Q_i: np.ndarray, var_mu_i: np.ndarray) -> np.ndarray:
        Q_i = np.asarray(Q_i, dtype=float)
        var_mu_i = np.asarray(var_mu_i, dtype=float)
        return Q_i + self.gamma * np.diag(var_mu_i)


class SubportfolioImportanceWeighting:
    """
    Subportfolio Importance Re-weighting:
        tilde{a}_i propto a_i / (1 + beta * average_uncertainty_i)
    """

    def __init__(self, beta: float = 1.0):
        self.beta = float(beta)

    def apply(
        self,
        base_a: np.ndarray,
        subportfolio_variances: Sequence[np.ndarray],
    ) -> np.ndarray:
        base_a = np.asarray(base_a, dtype=float)
        avg_vars = np.array([np.mean(v) for v in subportfolio_variances], dtype=float)
        raw_weights = base_a / (1.0 + self.beta * avg_vars)
        return raw_weights / np.sum(raw_weights)


class DownsideRiskAdjustment:
    """
    Posterior Downside Probability Adjustment:
        tilde{mu}_j = mu_j - penalty_scale * Phi((r_target - mu_j) / sigma_j)
    """

    def __init__(self, target_return: float = 0.0, penalty_scale: float = 0.1):
        self.target_return = float(target_return)
        self.penalty_scale = float(penalty_scale)

    def apply(self, mu: np.ndarray, var_mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=float)
        std_mu = np.sqrt(np.maximum(var_mu, 1e-12))
        z_scores = (self.target_return - mu) / std_mu
        prob_underperform = norm.cdf(z_scores)
        return mu - self.penalty_scale * prob_underperform


@dataclass
class DistributedUncertaintyConfig:
    """Configuration bundling candidate Bayesian uncertainty mechanisms."""

    shrinkage_kappa: float = 0.0
    risk_aug_gamma: float = 0.0
    weight_beta: float = 0.0
    downside_penalty_scale: float = 0.0
    target_return: float = 0.0

    def transform_subportfolios(
        self,
        hypergraph: HypergraphPortfolio,
        sub_Q: Sequence[np.ndarray],
        sub_mu: Sequence[np.ndarray],
        sub_var_mu: Sequence[np.ndarray],
        base_a: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        K = hypergraph.num_subportfolios
        if base_a is None:
            base_a = np.ones(K, dtype=float) / float(K)

        transformed_Q = []
        transformed_mu = []

        shrinkage = BayesianReturnShrinkage(self.shrinkage_kappa)
        risk_aug = BayesianRiskAugmentation(self.risk_aug_gamma)
        downside = DownsideRiskAdjustment(self.target_return, self.downside_penalty_scale)

        for i in range(K):
            q_i = sub_Q[i].copy()
            mu_i = sub_mu[i].copy()
            var_i = sub_var_mu[i]

            if self.risk_aug_gamma > 0.0:
                q_i = risk_aug.apply(q_i, var_i)
            if self.shrinkage_kappa > 0.0:
                mu_i = shrinkage.apply(mu_i, var_i)
            if self.downside_penalty_scale > 0.0:
                mu_i = downside.apply(mu_i, var_i)

            transformed_Q.append(q_i)
            transformed_mu.append(mu_i)

        if self.weight_beta > 0.0:
            reweighter = SubportfolioImportanceWeighting(self.weight_beta)
            transformed_a = reweighter.apply(base_a, sub_var_mu)
        else:
            transformed_a = base_a.copy()

        return transformed_Q, transformed_mu, transformed_a


# =====================================================================
# Phase 2: Controlled Empirical Simulation Environment
# =====================================================================

@dataclass
class TimeStepData:
    """Data available at time step t."""
    t: int
    true_mu: np.ndarray       # True expected return E[r_{t+1}]
    mu_hat: np.ndarray        # Point forecast (identical for all methods)
    sigma_epi: np.ndarray     # Epistemic uncertainty
    realized_r: np.ndarray    # Realized returns r_{t+1}
    h_t: Optional[np.ndarray] = None  # Ex-ante latent unreliability state


class SyntheticMarketEnvironment:
    """
    Generates causal time-series data for testing the economic value of epistemic uncertainty.

    Decision flow:
        F_t -> (mu_hat_t, sigma_epi_t) -> w_t -> r_{t+1}

    Causal structure:
        A latent unreliability state h_t is determined at decision time t.
        h_t jointly governs:
        1. The dispersion of forecast errors: epsilon_{t, j} ~ N(0, h_{t, j}^2)
        2. The epistemic uncertainty estimate: sigma_epi_{t, j} = g(h_{t, j})
        Realized returns r_{t+1} are sampled strictly AFTER (mu_hat_t, sigma_epi_t) are generated.

    Regimes:
        - 'well_calibrated' / 'informative': sigma_epi tracks latent state h_t with high precision.
        - 'weakly_calibrated' / 'weakly_informative': sigma_epi has noisy correlation with h_t.
        - 'uninformative' / 'random': sigma_epi is independent random noise.
        - 'miscalibrated' / 'misleading': sigma_epi is negatively coupled with h_t.
        - 'regime_shift' / 'regime_break': h_t and true mu shift abruptly at T/2, causing sharp uncertainty spikes.
    """

    def __init__(
        self,
        num_assets: int = 5,
        horizon: int = 120,
        regime: str = "well_calibrated",
        seed: int = 42,
    ):
        self.num_assets = num_assets
        self.horizon = horizon
        self.regime = regime
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Baseline market covariance Q
        # Generate realistic correlation structure with positive definiteness
        A = self.rng.standard_normal((num_assets, num_assets))
        cov_raw = A @ A.T + num_assets * np.eye(num_assets)
        vols = self.rng.uniform(0.12, 0.28, size=num_assets)  # Annualized 12-28%
        # Convert to monthly
        vols_m = vols / np.sqrt(12.0)
        corr = cov_raw / np.sqrt(np.outer(np.diag(cov_raw), np.diag(cov_raw)))
        self.Q = np.outer(vols_m, vols_m) * corr

        # Baseline expected returns (monthly ~0.5% to 1.5%)
        self.base_mu = self.rng.uniform(0.005, 0.015, size=num_assets)

    def generate_trajectory(self) -> List[TimeStepData]:
        """Generate time-series history for all T steps causally without future target leakage."""
        trajectory = []
        current_mu = self.base_mu.copy()
        m = self.num_assets

        REGIME_MAP = {
            "informative": "informative",
            "well_calibrated": "informative",
            "weakly_informative": "weakly_informative",
            "weakly_calibrated": "weakly_informative",
            "random": "random",
            "uninformative": "random",
            "misleading": "misleading",
            "miscalibrated": "misleading",
            "regime_shift": "regime_shift",
            "regime_break": "regime_shift",
        }
        canonical = REGIME_MAP.get(self.regime, self.regime)
        if canonical not in ["informative", "weakly_informative", "random", "misleading", "regime_shift"]:
            raise ValueError(f"Unknown regime '{self.regime}'")

        for t in range(self.horizon):
            # 1. Slow AR(1) drift in true expected return
            current_mu = 0.95 * current_mu + 0.05 * self.base_mu + self.rng.normal(0, 0.001, size=m)

            # 2. Ex-ante latent unreliability state h_t >= 0 known at decision time
            base_h = np.linspace(0.004, 0.012, m)
            h_t = base_h + self.rng.uniform(0.000, 0.002, size=m)

            # Structural regime shift at midpoint if specified
            if canonical == "regime_shift" and t >= self.horizon // 2 and t <= self.horizon // 2 + 10:
                if t == self.horizon // 2:
                    current_mu[0] -= 0.02
                    current_mu[1] -= 0.015
                    current_mu[2] += 0.015  # Flight to defensive asset
                h_t[0] *= 3.5
                h_t[1] *= 3.0
                h_t[2] *= 1.5

            # 3. Forecast error drawn from distribution N(0, diag(h_t^2))
            forecast_error = self.rng.normal(0, h_t)
            mu_hat = current_mu + forecast_error

            # 4. Epistemic uncertainty generated causally from h_t (ex-ante scale, zero target leakage)
            if canonical == "informative":
                # High calibration: tracks latent state h_t with slight measurement noise
                sigma_epi = h_t + self.rng.normal(0, 0.0008, size=m)
            elif canonical == "weakly_informative":
                # Weakly informative: noisy mixture
                noise = self.rng.exponential(np.mean(base_h), size=m)
                sigma_epi = 0.35 * h_t + 0.65 * noise
            elif canonical == "random":
                # Uninformative: independent random noise
                sigma_epi = self.rng.exponential(np.mean(base_h), size=m)
            elif canonical == "misleading":
                # Miscalibrated: inverted relationship (high uncertainty on reliable assets)
                sigma_epi = (np.max(h_t) + np.min(h_t)) - h_t + self.rng.normal(0, 0.0008, size=m)
            elif canonical == "regime_shift":
                # Calibrated model whose uncertainty naturally spikes during the regime shift
                sigma_epi = h_t + self.rng.normal(0, 0.0008, size=m)

            sigma_epi = np.maximum(sigma_epi, 1e-4)

            # 5. Realized return for period t+1: r ~ N(current_mu, Q)
            # Sampled strictly AFTER mu_hat and sigma_epi have been determined
            realized_r = self.rng.multivariate_normal(current_mu, self.Q)

            trajectory.append(
                TimeStepData(
                    t=t,
                    true_mu=current_mu.copy(),
                    mu_hat=mu_hat.copy(),
                    sigma_epi=sigma_epi.copy(),
                    realized_r=realized_r.copy(),
                    h_t=h_t.copy(),
                )
            )

        return trajectory


@dataclass
class SimulationMetrics:
    """Performance evaluation metrics for a sequential backtest."""
    method_name: str
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    turnover: float
    cost_adjusted_return: float
    mean_utility: float
    cumulative_utility: float
    regret_vs_oracle: float
    worst_5pct_loss: float
    mean_hhi: float
    wealth_path: np.ndarray
    weights_path: np.ndarray
    utility_path: np.ndarray


def run_sequential_portfolio_simulation(
    trajectory: List[TimeStepData],
    Q: np.ndarray,
    lambda_u: float = 1.0,
    gamma: float = 2.0,       # Risk aversion
    transaction_cost: float = 0.0010,  # 10 bps
    kappa_turnover: float = 0.0005,    # Utility turnover penalty
    include_oracle: bool = True,
    mu_prior: float | np.ndarray = 0.0,
    shrinkage_mapping: str = "rational",
) -> Dict[str, SimulationMetrics]:
    """
    Run rolling portfolio allocation comparing:
        - 'Deterministic (mu_hat)': Uses point forecast mu_hat directly.
        - 'Uncertainty-Aware (mu_tilde)': Uses confidence shrinkage tilde_mu = rho(sigma_epi) * mu_hat.
        - 'Oracle (mu_true)': Uses true latent expected return mu_true.

    Strictly no lookahead: at time t, allocations are chosen using only data up to t,
    and returns r_{t+1} are realized afterwards.
    """
    T = len(trajectory)
    m = len(trajectory[0].mu_hat)

    methods = {
        "Deterministic (mu_hat)": {"use_uncertainty": False, "use_oracle": False},
        "Uncertainty-Aware": {"use_uncertainty": True, "use_oracle": False},
    }
    if include_oracle:
        methods["Oracle (mu_true)"] = {"use_uncertainty": False, "use_oracle": True}

    results: Dict[str, SimulationMetrics] = {}

    for method_name, cfg in methods.items():
        wealth = np.zeros(T + 1)
        wealth[0] = 1.0
        weights_history = []
        ret_history = []
        utility_history = []
        prev_w = np.ones(m) / float(m)

        for step in trajectory:
            # 1. Determine input return vector
            if cfg["use_oracle"]:
                input_mu = step.true_mu
            elif cfg["use_uncertainty"]:
                input_mu = apply_uncertainty_shrinkage(
                    step.mu_hat,
                    step.sigma_epi,
                    lambda_u=lambda_u,
                    mu_prior=mu_prior,
                    mapping=shrinkage_mapping,
                )
            else:
                input_mu = step.mu_hat

            # 2. Solve Markowitz allocation
            model = MeanVariancePortfolio(Q=Q, mu=input_mu, budget=1.0)
            try:
                w_opt, _ = model.solve()
            except Exception:
                w_opt = prev_w.copy()

            weights_history.append(w_opt.copy())

            # 3. Calculate turnover and transaction costs
            turnover = float(np.sum(np.abs(w_opt - prev_w)))
            tc_loss = transaction_cost * turnover

            # 4. Observe realized return
            gross_r = float(w_opt @ step.realized_r)
            net_r = gross_r - tc_loss
            ret_history.append(net_r)

            # 5. Calculate realized utility: U_t = net_r - gamma * (w^T Q w) - kappa * turnover
            port_var = float(w_opt @ Q @ w_opt)
            u_t = net_r - gamma * port_var - kappa_turnover * turnover
            utility_history.append(u_t)

            # 6. Update wealth
            wealth[step.t + 1] = wealth[step.t] * (1.0 + net_r)
            prev_w = w_opt.copy()

        # Compute summary metrics
        ret_arr = np.array(ret_history)
        wealth_arr = wealth
        ann_factor = 12.0  # Monthly frequency

        cum_ret = float(wealth_arr[-1] - 1.0)
        ann_ret = float(np.mean(ret_arr) * ann_factor)
        ann_vol = float(np.std(ret_arr) * np.sqrt(ann_factor))
        sharpe = ann_ret / (ann_vol + 1e-12)

        # Sortino (downside deviation)
        downside_ret = ret_arr[ret_arr < 0]
        downside_dev = float(np.std(downside_ret) * np.sqrt(ann_factor)) if len(downside_ret) > 0 else 1e-6
        sortino = ann_ret / (downside_dev + 1e-12)

        # Max drawdown
        peak = np.maximum.accumulate(wealth_arr)
        drawdowns = (wealth_arr - peak) / peak
        max_dd = float(np.min(drawdowns))

        # Turnover
        all_w = np.array(weights_history)
        diff_w = np.abs(np.diff(all_w, axis=0))
        mean_to = float(np.mean(np.sum(diff_w, axis=1))) if len(diff_w) > 0 else 0.0

        # Mean HHI concentration
        hhi = float(np.mean(np.sum(all_w**2, axis=1)))

        # Worst 5% loss (Value at Risk)
        worst_5pct = float(np.percentile(ret_arr, 5))

        results[method_name] = SimulationMetrics(
            method_name=method_name,
            cumulative_return=cum_ret,
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            turnover=mean_to,
            cost_adjusted_return=float(np.mean(ret_arr) * ann_factor),
            mean_utility=float(np.mean(utility_history)),
            cumulative_utility=float(np.sum(utility_history)),
            regret_vs_oracle=0.0,  # Computed below
            worst_5pct_loss=worst_5pct,
            mean_hhi=hhi,
            wealth_path=wealth_arr,
            weights_path=all_w,
            utility_path=np.array(utility_history),
        )

    # Compute regret vs oracle
    if include_oracle and "Oracle (mu_true)" in results:
        oracle_cum_u = results["Oracle (mu_true)"].cumulative_utility
        for name in results:
            results[name].regret_vs_oracle = float(oracle_cum_u - results[name].cumulative_utility)

    return results


# =============================================================================
# Phase 5: Portfolio-Level Exposure Timing & Uncertainty Aggregation
# =============================================================================


def compute_aggregate_epistemic_uncertainty(
    epistemic_stds: np.ndarray,
    weights: Optional[np.ndarray] = None,
    method: str = "mean",
) -> float:
    """
    Compute a single scalar aggregate epistemic uncertainty metric across assets.

    Methods:
        - 'mean': Unweighted mean uncertainty across all assets.
        - 'median': Median uncertainty across all assets.
        - 'weighted': Inner product of base risky weights and asset uncertainties: sum_j w_j * sigma_j.
        - 'max': Maximum uncertainty across all assets.
        - 'q75': 75th percentile uncertainty across assets.

    Args:
        epistemic_stds: 1D array of asset-level epistemic standard deviations (sigma_epi >= 0).
        weights: Optional 1D array of asset weights (required if method='weighted').
        method: Aggregation method ('mean', 'median', 'weighted', 'max', 'q75').

    Returns:
        float: Scalar aggregate uncertainty u_t >= 0.
    """
    stds = np.asarray(epistemic_stds, dtype=float).ravel()
    if len(stds) == 0:
        return 0.0

    if method == "mean":
        return float(np.mean(stds))
    elif method == "median":
        return float(np.median(stds))
    elif method == "weighted":
        if weights is None:
            raise ValueError("weights must be provided when method='weighted'")
        w = np.asarray(weights, dtype=float).ravel()
        if len(w) != len(stds):
            raise ValueError(f"weights length ({len(w)}) must match epistemic_stds ({len(stds)})")
        w_sum = np.sum(np.abs(w))
        if w_sum > 0:
            w_norm = w / w_sum
            return float(np.sum(w_norm * stds))
        return float(np.mean(stds))
    elif method == "max":
        return float(np.max(stds))
    elif method == "q75":
        return float(np.percentile(stds, 75))
    else:
        raise ValueError(f"Unknown uncertainty aggregation method: {method}")


def compute_epistemic_exposure_scale(
    aggregate_uncertainty: float,
    lambda_u: float = 1.0,
    mapping: str = "rational",
    floor: float = 0.0,
) -> float:
    """
    Compute aggregate portfolio risky exposure scale s_epi in [floor, 1.0].

    Mappings:
        - 'rational':    s = 1.0 / (1.0 + lambda_u * u_t)
        - 'exponential': s = exp(-lambda_u * u_t)
        - 'bucket':      stepped scale based on lambda_u thresholds

    Guarantees:
        - lambda_u == 0 => s = 1.0 (full exposure / deterministic reproduction)
        - Monotonically decreasing in aggregate_uncertainty for lambda_u > 0.
        - Clamped strictly in [floor, 1.0].

    Args:
        aggregate_uncertainty: Scalar uncertainty metric u_t >= 0.
        lambda_u: Exposure shrinkage parameter (lambda_u >= 0).
        mapping: 'rational', 'exponential', or 'bucket'.
        floor: Minimum exposure floor in [0.0, 1.0] (default 0.0).

    Returns:
        float: Exposure scale s_epi in [floor, 1.0].
    """
    if lambda_u <= 0.0 or aggregate_uncertainty <= 0.0:
        return 1.0

    u = float(aggregate_uncertainty)
    lam = float(lambda_u)

    if mapping == "rational":
        s = 1.0 / (1.0 + lam * u)
    elif mapping == "exponential":
        s = float(np.exp(-lam * u))
    elif mapping == "bucket":
        # Stepped scale: > 2x / > 1.5x / > 1x baseline threshold
        if lam * u > 2.0:
            s = 0.25
        elif lam * u > 1.0:
            s = 0.50
        elif lam * u > 0.5:
            s = 0.75
        else:
            s = 1.0
    else:
        raise ValueError(f"Unknown exposure mapping: {mapping}")

    return float(np.clip(s, floor, 1.0))


def compute_volatility_targeting_scale(
    rolling_vol: float,
    target_vol: float = 0.20,
    max_leverage: float = 1.0,
    min_exposure: float = 0.0,
) -> float:
    """
    Compute standard realized volatility targeting exposure scale:
        s_vol = min(max_leverage, target_vol / rolling_vol)

    Args:
        rolling_vol: Annualized rolling historical return volatility (rolling_vol >= 0).
        target_vol: Target annualized volatility (target_vol > 0).
        max_leverage: Maximum portfolio exposure (default 1.0, long-only / unleveraged).
        min_exposure: Minimum portfolio exposure (default 0.0).

    Returns:
        float: Exposure scale s_vol in [min_exposure, max_leverage].
    """
    if target_vol <= 0.0:
        raise ValueError(f"target_vol must be positive, got {target_vol}")

    if rolling_vol <= 1e-8:
        # Near-zero historical volatility -> cap at max_leverage
        return float(max_leverage)

    s_vol = target_vol / rolling_vol
    return float(np.clip(s_vol, min_exposure, max_leverage))


def apply_portfolio_exposure_scale(
    risky_weights: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, float]:
    """
    Scale risky portfolio weights and allocate remainder to cash (zero-return asset).

    Maintains identical relative cross-sectional proportions:
        w_risky_j = scale * w_base_j
        w_cash    = 1.0 - scale

    Args:
        risky_weights: 1D array of risky asset weights summing to 1.0 (or normalized).
        scale: Portfolio exposure scale s in [0.0, 1.0].

    Returns:
        Tuple[np.ndarray, float]:
            - w_risky: Scaled risky asset weight vector summing to scale.
            - w_cash: Cash allocation weight equal to (1.0 - scale).
    """
    w_base = np.asarray(risky_weights, dtype=float).ravel()
    s = float(np.clip(scale, 0.0, 1.0))

    # Normalize base weights if they don't sum exactly to 1
    w_sum = np.sum(w_base)
    if w_sum > 0:
        w_normalized = w_base / w_sum
    else:
        w_normalized = np.zeros_like(w_base)

    w_risky = s * w_normalized
    w_cash = 1.0 - s

    return w_risky, w_cash


# =============================================================================
# Phase 6: Relative Epistemic Uncertainty & Matched-Exposure Timing
# =============================================================================


def compute_relative_epistemic_uncertainty(
    history_uncertainty: Sequence[float],
    current_uncertainty: float,
    window: int = 13,
    method: str = "zscore",
    min_history: int = 4,
) -> float:
    """
    Compute causal relative epistemic uncertainty using strictly trailing history up to t-1.

    Guarantees:
        - Strict zero-leakage: history_uncertainty contains only past periods up to t-1.
        - Robust handling of zero-variance or zero-MAD history.
        - Deterministic fallback when history length < min_history.

    Methods:
        - 'zscore': (u_t - mean_{t-1}) / std_{t-1}
        - 'robust_zscore': (u_t - median_{t-1}) / (1.4826 * MAD_{t-1})
        - 'ratio': u_t / median_{t-1}
        - 'percentile_rank': fraction of trailing history where u_t >= u_{t-k}

    Args:
        history_uncertainty: Sequence of scalar aggregate uncertainties observed up to t-1.
        current_uncertainty: Aggregate uncertainty at decision epoch t (u_t >= 0).
        window: Trailing window length (default 13 weeks).
        method: 'zscore', 'robust_zscore', 'ratio', or 'percentile_rank'.
        min_history: Minimum required history length before dynamic normalization (default 4).

    Returns:
        float: Normalized relative uncertainty metric.
    """
    u_t = float(current_uncertainty)
    hist = list(history_uncertainty)[-window:]

    # Fallback if insufficient trailing history
    if len(hist) < min_history:
        if method in ("zscore", "robust_zscore"):
            return 0.0
        elif method == "ratio":
            return 1.0
        elif method == "percentile_rank":
            return 0.5

    arr = np.asarray(hist, dtype=float)

    if method == "zscore":
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        if std_val < 1e-8:
            return 0.0
        return float((u_t - mean_val) / std_val)

    elif method == "robust_zscore":
        med_val = float(np.median(arr))
        mad_val = float(np.median(np.abs(arr - med_val)))
        denom = 1.4826 * mad_val
        if denom < 1e-8:
            # Fall back to sample std if MAD is zero (e.g., discrete values)
            std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            if std_val < 1e-8:
                return 0.0
            return float((u_t - med_val) / std_val)
        return float((u_t - med_val) / denom)

    elif method == "ratio":
        med_val = float(np.median(arr))
        if med_val < 1e-8:
            return 1.0
        return float(u_t / med_val)

    elif method == "percentile_rank":
        return float(np.mean(u_t >= arr))

    else:
        raise ValueError(f"Unknown relative uncertainty normalization method: {method}")


def compute_relative_epistemic_exposure_scale(
    relative_uncertainty: float,
    s_base: float = 0.70,
    lambda_rel: float = 0.20,
    tau: float = 0.0,
    s_min: float = 0.0,
    s_max: float = 1.0,
    smooth: bool = False,
    k_smooth: float = 2.0,
) -> float:
    """
    Compute dynamic exposure scale based on relative (abnormally elevated) uncertainty.

    Piecewise linear rule:
        s_t = clip(s_base - lambda_rel * max(0, z_t - tau), s_min, s_max)

    Smooth logistic rule:
        s_t = s_min + (s_max - s_min) / (1 + exp(k * (z_t - tau)))

    Guarantees:
        - Weakly decreasing in relative_uncertainty.
        - Clamped strictly in [s_min, s_max].
        - No leverage (s_max <= 1.0 unless explicitly configured).

    Args:
        relative_uncertainty: Relative uncertainty metric (e.g. z-score).
        s_base: Baseline exposure under normal uncertainty (z <= tau).
        lambda_rel: Sensitivity penalty per unit of excess relative uncertainty.
        tau: Threshold above which de-risking begins (default 0.0).
        s_min: Minimum exposure floor in [0.0, 1.0] (default 0.0).
        s_max: Maximum exposure ceiling in [s_min, 1.0] (default 1.0).
        smooth: If True, uses smooth logistic mapping instead of piecewise linear.
        k_smooth: Logistic steepness parameter.

    Returns:
        float: Exposure scale s_t in [s_min, s_max].
    """
    z = float(relative_uncertainty)

    if smooth:
        scale = s_min + (s_max - s_min) / (1.0 + float(np.exp(k_smooth * (z - tau))))
    else:
        excess = max(0.0, z - tau)
        scale = s_base - lambda_rel * excess

    return float(np.clip(scale, s_min, s_max))


def generate_placebo_exposure_paths(
    actual_exposure_path: Sequence[float],
    n_placebos: int = 1000,
    method: str = "circular_shift",
    seed: int = 42,
) -> np.ndarray:
    """
    Generate placebo exposure trajectories that preserve the exact empirical distribution
    (mean, variance, quantiles) of actual exposures while altering temporal sequencing.

    Methods:
        - 'circular_shift': Shifts exposure trajectory by random lag k in [1, T-1].
        - 'permutation': Randomly shuffles the sequence of exposure values.

    Args:
        actual_exposure_path: 1D array of realized exposure values s_t.
        n_placebos: Number of placebo paths to generate (default 1,000).
        method: 'circular_shift' or 'permutation'.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Matrix of shape (n_placebos, T) of permuted exposure paths.
    """
    arr = np.asarray(actual_exposure_path, dtype=float)
    T = len(arr)
    rng = np.random.default_rng(seed)

    placebos = np.zeros((n_placebos, T), dtype=float)

    if method == "circular_shift":
        shifts = rng.integers(1, T, size=n_placebos)
        for i, s in enumerate(shifts):
            placebos[i] = np.roll(arr, s)
    elif method == "permutation":
        for i in range(n_placebos):
            placebos[i] = rng.permutation(arr)
    else:
        raise ValueError(f"Unknown placebo generation method: {method}")

    return placebos


def generate_block_placebo_paths(
    actual_exposure_path: Sequence[float],
    n_placebos: int = 1000,
    block_size: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate block-permuted placebo exposure paths that preserve local temporal
    autocorrelation while breaking causal alignment with future market events.

    Args:
        actual_exposure_path: 1D array of realized exposure values s_t.
        n_placebos: Number of placebo paths to generate.
        block_size: Size of contiguous weekly exposure blocks (default 4).
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Matrix of shape (n_placebos, T) of block-permuted exposures.
    """
    arr = np.asarray(actual_exposure_path, dtype=float)
    T = len(arr)
    rng = np.random.default_rng(seed)

    n_blocks = int(math.ceil(T / block_size))
    blocks = [arr[i * block_size : min((i + 1) * block_size, T)] for i in range(n_blocks)]

    placebos = np.zeros((n_placebos, T), dtype=float)
    for i in range(n_placebos):
        perm_idx = rng.permutation(n_blocks)
        perm_arr = np.concatenate([blocks[idx] for idx in perm_idx])[:T]
        placebos[i] = perm_arr

    return placebos


def decompose_timing_economic_benefit(
    base_returns: Sequence[float],
    timing_exposures: Sequence[float],
    constant_exposure: float,
    gamma: float = 1.0,
    cost_bps: float = 0.0010,
) -> Dict[str, float]:
    """
    Decompose the economic benefit of dynamic timing relative to constant matched exposure:
        Timing Benefit = Losses Avoided - Upside Sacrificed - Excess Transaction Costs - Extra Risk Penalty

    Args:
        base_returns: Sequence of base risky portfolio returns r_{base, t}.
        timing_exposures: Sequence of dynamic exposures s_t.
        constant_exposure: Scalar constant exposure c matching mean(s_t).
        gamma: Risk aversion coefficient (default 1.0).
        cost_bps: Proportional transaction cost in decimals (default 0.0010).

    Returns:
        Dict[str, float]: Component breakdown of timing value.
    """
    r_arr = np.asarray(base_returns, dtype=float)
    s_arr = np.asarray(timing_exposures, dtype=float)
    c = float(constant_exposure)

    # Losses avoided: periods where r < 0 and s_t < c
    down_mask = r_arr < 0.0
    avoided_losses = float(np.sum((c - s_arr[down_mask]) * (-r_arr[down_mask])))

    # Upside sacrificed: periods where r > 0 and s_t < c
    up_mask = r_arr > 0.0
    sacrificed_upside = float(np.sum((c - s_arr[up_mask]) * r_arr[up_mask]))

    # Captured bonus: periods where r > 0 and s_t > c
    bonus_upside = float(np.sum((s_arr[up_mask] - c) * r_arr[up_mask]))

    # Return delta
    delta_return = float(np.sum((s_arr - c) * r_arr))

    # Turnover delta
    to_timing = np.sum(np.abs(np.diff(s_arr))) if len(s_arr) > 1 else 0.0
    # Constant exposure has zero cash rebalancing turnover
    to_excess = float(to_timing * 2.0)  # risky + cash sleeves
    excess_tc = float(cost_bps * to_excess)

    # Net benefit
    net_economic_benefit = delta_return - excess_tc

    return {
        "avoided_losses": avoided_losses,
        "sacrificed_upside": sacrificed_upside,
        "bonus_upside": bonus_upside,
        "net_return_difference": delta_return,
        "excess_turnover": to_excess,
        "excess_transaction_costs": excess_tc,
        "net_economic_benefit": net_economic_benefit,
    }


# =============================================================================
# Phase 8: Cost-Aware Epistemic Timing & Execution Rules
# =============================================================================


def compute_hysteresis_exposure_path(
    signal: Sequence[float],
    tau_enter: float,
    tau_exit: float,
    s_normal: float = 0.85,
    s_reduced: float = 0.45,
    initial_state: str = "normal",
) -> np.ndarray:
    """
    Compute causal exposure path using a two-threshold hysteresis state machine.

    Prevents rapid flip-flopping around a single threshold:
        - If NORMAL: transition to REDUCED only when signal > tau_enter.
        - If REDUCED: return to NORMAL only when signal < tau_exit.
        - Inside dead-band [tau_exit, tau_enter]: state remains unchanged.

    Args:
        signal: Sequence of relative uncertainty metrics z_t.
        tau_enter: Entry threshold into de-risked state (tau_enter > tau_exit).
        tau_exit: Exit threshold returning to normal state.
        s_normal: Exposure when in normal state (default 0.85).
        s_reduced: Exposure when in reduced state (default 0.45).
        initial_state: Starting state, 'normal' or 'reduced' (default 'normal').

    Returns:
        np.ndarray: Causal exposure path of length len(signal).
    """
    if tau_exit >= tau_enter:
        raise ValueError(f"tau_exit ({tau_exit}) must be strictly less than tau_enter ({tau_enter})")
    if not (0.0 <= s_normal <= 1.0) or not (0.0 <= s_reduced <= 1.0):
        raise ValueError("Exposures s_normal and s_reduced must be within [0.0, 1.0]")

    state = initial_state
    exposures = []

    for z in signal:
        z_val = float(z)
        if state == "normal":
            if z_val > tau_enter:
                state = "reduced"
        else:  # state == "reduced"
            if z_val < tau_exit:
                state = "normal"

        exposures.append(s_reduced if state == "reduced" else s_normal)

    return np.asarray(exposures, dtype=float)


def compute_persistent_exposure_path(
    signal: Sequence[float],
    tau: float,
    k: int = 2,
    n: int = 2,
    s_normal: float = 0.85,
    s_reduced: float = 0.45,
    re_risk_k: int = 1,
    re_risk_n: int = 1,
    initial_state: str = "normal",
) -> np.ndarray:
    """
    Compute causal exposure path requiring uncertainty persistence before de-risking.

    Eliminates transient single-period spikes:
        - To de-risk: at least k of the last n observations must exceed tau.
        - To re-risk: at least re_risk_k of the last re_risk_n observations must fall <= tau.

    Args:
        signal: Sequence of relative uncertainty metrics z_t.
        tau: Uncertainty threshold.
        k: Minimum required exceedance count to de-risk (k <= n).
        n: Trailing evaluation window length to de-risk.
        s_normal: Normal risk exposure (default 0.85).
        s_reduced: Reduced risk exposure (default 0.45).
        re_risk_k: Required count to re-risk.
        re_risk_n: Window to re-risk.
        initial_state: Initial state ('normal' or 'reduced').

    Returns:
        np.ndarray: Causal exposure path.
    """
    if k < 1 or k > n:
        raise ValueError(f"k ({k}) must satisfy 1 <= k <= n ({n})")
    if re_risk_k < 1 or re_risk_k > re_risk_n:
        raise ValueError(f"re_risk_k ({re_risk_k}) must satisfy 1 <= re_risk_k <= re_risk_n ({re_risk_n})")

    sig_arr = list(signal)
    state = initial_state
    exposures = []

    for t in range(len(sig_arr)):
        if state == "normal":
            win = sig_arr[max(0, t - n + 1) : t + 1]
            exceed_count = sum(1 for x in win if float(x) > tau)
            if len(win) == n and exceed_count >= k:
                state = "reduced"
        else:  # state == "reduced"
            win_re = sig_arr[max(0, t - re_risk_n + 1) : t + 1]
            calm_count = sum(1 for x in win_re if float(x) <= tau)
            if len(win_re) == re_risk_n and calm_count >= re_risk_k:
                state = "normal"

        exposures.append(s_reduced if state == "reduced" else s_normal)

    return np.asarray(exposures, dtype=float)


def compute_band_exposure_path(
    desired_exposure_path: Sequence[float],
    delta: float = 0.10,
    eta: float = 1.0,
    initial_exposure: Optional[float] = None,
) -> np.ndarray:
    """
    Apply a no-rebalancing inertia band around current exposure.

    If |s*_t - s_{t-1}| < delta:
        keep s_t = s_{t-1}
    Else:
        s_t = s_{t-1} + eta * (s*_t - s_{t-1})

    Args:
        desired_exposure_path: Sequence of unconstrained desired exposures s*_t.
        delta: Inactivity band half-width (default 0.10).
        eta: Partial adjustment speed in (0, 1.0] (default 1.0).
        initial_exposure: Exposure prior to t=0 (defaults to desired_exposure_path[0]).

    Returns:
        np.ndarray: Band-filtered exposure path.
    """
    if delta < 0:
        raise ValueError("delta must be non-negative")
    if not (0.0 < eta <= 1.0):
        raise ValueError("eta must be in (0, 1.0]")

    des_arr = list(desired_exposure_path)
    if not des_arr:
        return np.array([], dtype=float)

    prev_s = float(initial_exposure) if initial_exposure is not None else float(des_arr[0])
    exposures = []

    for s_target in des_arr:
        target = float(s_target)
        if abs(target - prev_s) < delta:
            curr_s = prev_s
        else:
            curr_s = prev_s + eta * (target - prev_s)

        curr_s = float(np.clip(curr_s, 0.0, 1.0))
        exposures.append(curr_s)
        prev_s = curr_s

    return np.asarray(exposures, dtype=float)


def compute_lower_frequency_exposure_path(
    desired_exposure_path: Sequence[float],
    frequency: int = 2,
    severe_signal: Optional[Sequence[float]] = None,
    tau_severe: Optional[float] = None,
    s_emergency: float = 0.30,
) -> np.ndarray:
    """
    Rebalance exposure only on allowed periodic dates (e.g. biweekly=2, monthly=4),
    with an optional emergency override for severe uncertainty spikes.

    Args:
        desired_exposure_path: Sequence of desired exposures.
        frequency: Rebalance interval in weeks (default 2).
        severe_signal: Optional relative uncertainty signal sequence.
        tau_severe: Optional severe spike threshold that overrides periodicity.
        s_emergency: Target exposure during emergency override.

    Returns:
        np.ndarray: Lower-frequency exposure path.
    """
    if frequency < 1:
        raise ValueError("frequency must be at least 1")

    des_arr = list(desired_exposure_path)
    if not des_arr:
        return np.array([], dtype=float)

    curr_s = float(des_arr[0])
    exposures = []

    for t, des in enumerate(des_arr):
        if severe_signal is not None and tau_severe is not None and float(severe_signal[t]) > tau_severe:
            curr_s = float(s_emergency)
        elif t % frequency == 0:
            curr_s = float(des)

        curr_s = float(np.clip(curr_s, 0.0, 1.0))
        exposures.append(curr_s)

    return np.asarray(exposures, dtype=float)


def compute_min_hold_exposure_path(
    desired_exposure_path: Sequence[float],
    min_hold_periods: int = 2,
    threshold: float = 0.60,
    s_reduced: float = 0.45,
    s_normal: float = 0.85,
) -> np.ndarray:
    """
    Enforce a minimum holding duration upon entering the reduced-risk state.

    Args:
        desired_exposure_path: Sequence of unconstrained exposures.
        min_hold_periods: Minimum duration (in weeks) to remain defensive.
        threshold: Threshold below which state is deemed 'reduced'.
        s_reduced: Fixed reduced exposure value.
        s_normal: Fixed normal exposure value.

    Returns:
        np.ndarray: Lock-in exposure path.
    """
    des_arr = list(desired_exposure_path)
    hold_counter = 0
    state = "normal"
    exposures = []

    for s_target in des_arr:
        is_defensive = float(s_target) < threshold

        if state == "normal":
            if is_defensive:
                state = "reduced"
                hold_counter = min_hold_periods - 1
        else:  # state == "reduced"
            if hold_counter > 0:
                hold_counter -= 1
            else:
                if not is_defensive:
                    state = "normal"

        exposures.append(s_reduced if state == "reduced" else s_normal)

    return np.asarray(exposures, dtype=float)


def compute_turnover_attribution(
    exposure_path: Sequence[float],
    base_returns: Sequence[float],
    cost_bps: float = 0.0010,
    m9_exposure_path: Optional[Sequence[float]] = None,
    gamma: float = 1.0,
) -> Dict[str, float]:
    """
    Compute fine-grained turnover attribution, trading frequency, and cost savings vs raw timing.

    Args:
        exposure_path: Exposure path of policy under evaluation.
        base_returns: Returns of the base portfolio.
        cost_bps: Transaction cost per unit of turnover (default 0.0010).
        m9_exposure_path: Optional raw timing path (M9) for relative savings.
        gamma: Risk aversion coefficient.

    Returns:
        Dict[str, float]: Detailed turnover and friction statistics.
    """
    s_arr = np.asarray(exposure_path, dtype=float)
    r_arr = np.asarray(base_returns, dtype=float)
    T = len(s_arr)

    diffs = np.abs(np.diff(s_arr)) if T > 1 else np.array([0.0])
    annual_factor = 52.0 / T if T > 0 else 1.0

    total_turnover = float(np.sum(diffs) * 2.0)
    annual_turnover = total_turnover * annual_factor
    n_changes = int(np.sum(diffs > 1e-4))
    mean_change = float(np.mean(diffs[diffs > 1e-4])) if n_changes > 0 else 0.0
    tc_drag = float(cost_bps * total_turnover)

    defensive = s_arr < 0.60
    n_transitions = int(np.sum(defensive[1:] != defensive[:-1])) if T > 1 else 0

    metrics = {
        "total_turnover": total_turnover,
        "annual_turnover": annual_turnover,
        "n_exposure_changes": float(n_changes),
        "mean_change_size": mean_change,
        "n_state_transitions": float(n_transitions),
        "total_tc_drag": tc_drag,
    }

    if m9_exposure_path is not None:
        m9_arr = np.asarray(m9_exposure_path, dtype=float)
        m9_diffs = np.abs(np.diff(m9_arr)) if len(m9_arr) > 1 else np.array([0.0])
        m9_tc = float(cost_bps * np.sum(m9_diffs) * 2.0)
        cost_saved = m9_tc - tc_drag
        metrics["cost_saved_vs_m9"] = cost_saved

        mean_exp = float(np.mean(s_arr))
        gross_policy = float(np.sum((s_arr - mean_exp) * r_arr))
        m9_mean = float(np.mean(m9_arr))
        gross_m9 = float(np.sum((m9_arr - m9_mean) * r_arr))
        benefit_retained = (gross_policy / gross_m9) if abs(gross_m9) > 1e-8 else 1.0
        metrics["gross_benefit_retained"] = float(benefit_retained)

    return metrics


def analyze_uncertainty_alarms(
    signal: Sequence[float],
    realized_returns: Sequence[float],
    exposure_path: Sequence[float],
    tail_threshold: float = -0.02,
) -> Dict[str, float]:
    """
    Evaluate the quality of uncertainty de-risking alarms:
        - Measures how frequently an alarm is followed by negative returns vs tail crashes.
        - Computes false positive rate.

    Args:
        signal: Relative uncertainty signal series.
        realized_returns: Base portfolio realized returns.
        exposure_path: Policy exposure path.
        tail_threshold: Return threshold defining tail market loss (default -0.02).

    Returns:
        Dict[str, float]: Alarm quality metrics.
    """
    s_arr = np.asarray(exposure_path, dtype=float)
    r_arr = np.asarray(realized_returns, dtype=float)
    med_s = float(np.median(s_arr))

    alarms = s_arr < med_s
    n_alarms = int(np.sum(alarms))
    total_periods = len(s_arr)

    if n_alarms == 0:
        return {
            "n_alarms": 0.0,
            "alarm_frequency": 0.0,
            "pct_negative_followed": 0.0,
            "pct_tail_loss_followed": 0.0,
            "mean_return_during_alarm": 0.0,
            "false_positive_rate": 0.0,
        }

    alarm_rets = r_arr[alarms]
    pct_neg = float(np.mean(alarm_rets < 0.0)) * 100.0
    pct_tail = float(np.mean(alarm_rets < tail_threshold)) * 100.0
    mean_ret = float(np.mean(alarm_rets))
    fp_rate = float(np.mean(alarm_rets >= 0.0)) * 100.0

    return {
        "n_alarms": float(n_alarms),
        "alarm_frequency": float(n_alarms / total_periods) * 100.0,
        "pct_negative_followed": pct_neg,
        "pct_tail_loss_followed": pct_tail,
        "mean_return_during_alarm": mean_ret,
        "false_positive_rate": fp_rate,
    }


def compute_canonical_turnover(
    w_risky_cur: Sequence[float],
    w_risky_prev: Optional[Sequence[float]] = None,
    w_cash_cur: float = 0.0,
    w_cash_prev: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute explicit, non-overlapping portfolio turnover components:
        1. Risky turnover: sum_j |w_risky,j,t - w_risky,j,t-1|
        2. Cash turnover: |w_cash,t - w_cash,t-1|
        3. Total two-way turnover: risky + cash
        4. Half (one-way) turnover: 0.5 * (risky + cash)

    Args:
        w_risky_cur: Current risky asset weights.
        w_risky_prev: Previous risky asset weights (None for first period).
        w_cash_cur: Current cash weight.
        w_cash_prev: Previous cash weight (None for first period).

    Returns:
        Dict[str, float]: Detailed turnover metrics.
    """
    w_r_cur = np.asarray(w_risky_cur, dtype=float)
    if w_risky_prev is None:
        to_risky = 0.0
    else:
        w_r_prev = np.asarray(w_risky_prev, dtype=float)
        to_risky = float(np.sum(np.abs(w_r_cur - w_r_prev)))

    if w_cash_prev is None:
        to_cash = 0.0
    else:
        to_cash = float(abs(w_cash_cur - w_cash_prev))

    to_total = to_risky + to_cash
    to_half = 0.5 * to_total

    return {
        "risky": to_risky,
        "cash": to_cash,
        "total": to_total,
        "half": to_half,
    }


def compute_actual_traded_turnover(
    current_risky_composition: Sequence[float],
    previous_risky_composition: Optional[Sequence[float]] = None,
    current_exposure: float = 1.0,
    previous_exposure: Optional[float] = None,
    assets: Optional[Sequence[str]] = None,
    previous_assets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compute actual market-traded portfolio turnover under exposure scaling.

    For every asset j:
        q_j,t = normalized composition of risky sleeve (sum_j q_j,t = 1)
        s_t   = aggregate risky exposure
        w_j,t = s_t * q_j,t (actual risky portfolio weight)
        w_cash,t = 1.0 - s_t (cash allocation)

    Actual risky securities traded between t-1 and t:
        Delta w_j,t = w_j,t - w_j,t-1 = s_t * q_j,t - s_{t-1} * q_{j,t-1}

    Canonical one-way risky turnover (costed turnover):
        TO_traded_risky = sum_j |Delta w_j,t|

    Diagnostic composition turnover:
        TO_comp = sum_j |q_j,t - q_{j,t-1}|

    Decomposition bounds:
        TO_traded_risky <= s_t * TO_comp + |s_t - s_{t-1}|
        interaction_netting = (s_t * TO_comp + |s_t - s_{t-1}|) - TO_traded_risky

    Args:
        current_risky_composition: Normalized weights q_t (or unnormalized base weights).
        previous_risky_composition: Previous normalized weights q_{t-1} (None for t=0).
        current_exposure: Current exposure scale s_t in [0.0, 1.0].
        previous_exposure: Previous exposure scale s_{t-1} (None for t=0).
        assets: Optional list of asset tickers for current period.
        previous_assets: Optional list of asset tickers for previous period.

    Returns:
        Dict[str, Any]: Detailed actual vs composition turnover metrics.
    """
    q_cur = np.asarray(current_risky_composition, dtype=float).ravel()
    s_cur = float(np.clip(current_exposure, 0.0, 1.0))

    # Normalize composition if sum > 0
    q_cur_sum = np.sum(q_cur)
    if q_cur_sum > 0:
        q_cur_norm = q_cur / q_cur_sum
    else:
        q_cur_norm = np.zeros_like(q_cur)

    w_actual_cur = s_cur * q_cur_norm
    w_cash_cur = 1.0 - s_cur

    if previous_risky_composition is None or previous_exposure is None:
        to_comp = 0.0
        to_actual_risky = 0.0
        to_cash = 0.0
        w_actual_prev = None
        s_prev = None
        q_prev_norm = None
    else:
        q_prev = np.asarray(previous_risky_composition, dtype=float).ravel()
        s_prev = float(np.clip(previous_exposure, 0.0, 1.0))
        q_prev_sum = np.sum(q_prev)
        if q_prev_sum > 0:
            q_prev_norm = q_prev / q_prev_sum
        else:
            q_prev_norm = np.zeros_like(q_prev)

        w_actual_prev = s_prev * q_prev_norm
        w_cash_prev = 1.0 - s_prev

        if assets is not None and previous_assets is not None:
            all_assets = sorted(set(assets) | set(previous_assets))
            c_q_map = {assets[i]: q_cur_norm[i] for i in range(len(assets))}
            p_q_map = {previous_assets[i]: q_prev_norm[i] for i in range(len(previous_assets))}
            c_w_map = {assets[i]: w_actual_cur[i] for i in range(len(assets))}
            p_w_map = {previous_assets[i]: w_actual_prev[i] for i in range(len(previous_assets))}

            to_comp = float(sum(abs(c_q_map.get(a, 0.0) - p_q_map.get(a, 0.0)) for a in all_assets))
            to_actual_risky = float(sum(abs(c_w_map.get(a, 0.0) - p_w_map.get(a, 0.0)) for a in all_assets))
        else:
            # Assume identical length
            to_comp = float(np.sum(np.abs(q_cur_norm - q_prev_norm)))
            to_actual_risky = float(np.sum(np.abs(w_actual_cur - w_actual_prev)))

        to_cash = float(abs(w_cash_cur - w_cash_prev))

    total_l1 = to_actual_risky + to_cash
    costed = to_actual_risky  # Canonical brokerage rule: fees applied to actual risky trades
    half = 0.5 * total_l1

    comp_bound = float(s_cur * to_comp) if s_cur is not None else 0.0
    exp_bound = to_cash
    interaction = float((comp_bound + exp_bound) - to_actual_risky)

    return {
        "previous_actual_risky_weights": w_actual_prev.tolist() if w_actual_prev is not None else None,
        "current_actual_risky_weights": w_actual_cur.tolist(),
        "previous_risky_composition": q_prev_norm.tolist() if q_prev_norm is not None else None,
        "current_risky_composition": q_cur_norm.tolist(),
        "previous_exposure": s_prev,
        "current_exposure": s_cur,
        "risky_composition_turnover": to_comp,
        "actual_risky_traded_turnover": to_actual_risky,
        "cash_weight_turnover": to_cash,
        "total_l1_turnover": total_l1,
        "costed_turnover": costed,
        "half_turnover": half,
        "composition_driven_turnover_bound": comp_bound,
        "exposure_driven_turnover_bound": exp_bound,
        "interaction_netting": interaction,
    }


def compute_canonical_portfolio_metrics(
    net_returns: Sequence[float],
    net_utilities: Optional[Sequence[float]] = None,
    risk_penalties: Optional[Sequence[float]] = None,
    turnovers: Optional[Sequence[float]] = None,
    cost_bps: float = 0.0010,
    ann_factor: float = 52.0,
    gamma: float = 1.0,
) -> Dict[str, float]:
    """
    Compute single canonical portfolio performance metrics from period-level net returns.

    Standard mathematical definitions:
        annualized_return = mean(r_net) * ann_factor
        annualized_volatility = std(r_net, ddof=1) * sqrt(ann_factor)
        sharpe_ratio = annualized_return / annualized_volatility
        downside_volatility = std([r for r in r_net if r < 0], ddof=1) * sqrt(ann_factor)
        sortino_ratio = annualized_return / downside_volatility
        max_drawdown = min((W - cummax(W)) / cummax(W)) where W_t = prod_{i<=t} (1 + r_net,i)
        mean_net_utility = mean(u_net)

    Args:
        net_returns: 1D array of net periodic returns r_{net, t}.
        net_utilities: Optional 1D array of net utilities u_{net, t}.
        risk_penalties: Optional 1D array of risk penalties 0.5 * gamma * var_t.
        turnovers: Optional 1D array of period costed turnovers.
        cost_bps: Proportional transaction cost per unit of turnover.
        ann_factor: Annualization factor (52.0 for weekly).
        gamma: Risk aversion parameter.

    Returns:
        Dict[str, float]: Standardized canonical portfolio performance metrics.
    """
    r_arr = np.asarray(net_returns, dtype=float).ravel()
    T = len(r_arr)
    if T == 0:
        return {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "downside_volatility": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "mean_net_utility": 0.0,
            "annual_costed_turnover": 0.0,
            "annual_tc_drag_bps": 0.0,
        }

    ann_ret = float(np.mean(r_arr) * ann_factor)
    ann_vol = float(np.std(r_arr, ddof=1) * np.sqrt(ann_factor)) if T > 1 else 0.0
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0

    downside = [r for r in r_arr if r < 0.0]
    if len(downside) > 1:
        down_vol = float(np.std(downside, ddof=1) * np.sqrt(ann_factor))
    else:
        down_vol = ann_vol
    sortino = float(ann_ret / down_vol) if down_vol > 1e-12 else 0.0

    # Compounded wealth trajectory for exact drawdown
    wealth = np.cumprod(1.0 + r_arr)
    peaks = np.maximum.accumulate(wealth)
    drawdowns = (wealth - peaks) / peaks
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    if net_utilities is not None:
        mean_u = float(np.mean(net_utilities))
    elif risk_penalties is not None:
        p_arr = np.asarray(risk_penalties, dtype=float).ravel()
        mean_u = float(np.mean(r_arr - p_arr))
    else:
        # Approximate variance penalty from sample variance if not provided
        sample_var = float(np.var(r_arr, ddof=1)) if T > 1 else 0.0
        mean_u = float(np.mean(r_arr) - 0.5 * gamma * sample_var)

    if turnovers is not None:
        to_arr = np.asarray(turnovers, dtype=float).ravel()
        ann_to = float(np.sum(to_arr) * (ann_factor / T))
        tc_drag_bps = float(ann_to * cost_bps * 10000.0)
    else:
        ann_to = 0.0
        tc_drag_bps = 0.0

    return {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "downside_volatility": down_vol,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "mean_net_utility": mean_u,
        "annual_costed_turnover": ann_to,
        "annual_tc_drag_bps": tc_drag_bps,
    }


def decompose_timing_economic_attribution_rebuilt(
    base_returns: Sequence[float],
    timing_exposures: Sequence[float],
    constant_exposure: float,
    cost_bps: float = 0.0010,
    timing_turnovers: Optional[Sequence[float]] = None,
    constant_turnovers: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Rebuild timing economic attribution using a mutually exclusive, non-overlapping 4-quadrant decomposition:
        Gross Return Difference = Losses Avoided - Upside Sacrificed + Extra Upside Captured - Extra Downside Suffered
        Net Economic Benefit = Gross Return Difference - Incremental Transaction Costs

    Algebraic closure guarantee:
        sum(component contributions) == realized cumulative net P&L difference (to machine precision).

    Args:
        base_returns: Base risky portfolio returns r_{base, t}.
        timing_exposures: Dynamic exposure path s_t.
        constant_exposure: Matched constant exposure c.
        cost_bps: Proportional transaction cost per unit of risky turnover.
        timing_turnovers: Optional period risky turnover for timing policy.
        constant_turnovers: Optional period risky turnover for constant benchmark.

    Returns:
        Dict[str, float]: Component breakdown and closure diagnostic.
    """
    r_arr = np.asarray(base_returns, dtype=float)
    s_arr = np.asarray(timing_exposures, dtype=float)
    c = float(constant_exposure)
    T = len(r_arr)

    diff_s = s_arr - c

    # 1. Losses avoided: policy exposure < constant exposure AND base return < 0
    mask_avoid = (diff_s < -1e-12) & (r_arr < 0.0)
    l_avoid = float(np.sum((c - s_arr[mask_avoid]) * (-r_arr[mask_avoid])))

    # 2. Upside sacrificed: policy exposure < constant exposure AND base return >= 0
    mask_sac = (diff_s < -1e-12) & (r_arr >= 0.0)
    u_sac = float(np.sum((c - s_arr[mask_sac]) * r_arr[mask_sac]))

    # 3. Extra upside captured: policy exposure > constant exposure AND base return >= 0
    mask_cap = (diff_s > 1e-12) & (r_arr >= 0.0)
    u_cap = float(np.sum((s_arr[mask_cap] - c) * r_arr[mask_cap]))

    # 4. Extra downside suffered: policy exposure > constant exposure AND base return < 0
    mask_suf = (diff_s > 1e-12) & (r_arr < 0.0)
    d_suf = float(np.sum((s_arr[mask_suf] - c) * (-r_arr[mask_suf])))

    # Cumulative gross difference
    gross_diff = float(np.sum(diff_s * r_arr))
    quadrant_sum = l_avoid - u_sac + u_cap - d_suf
    closure_err = abs(gross_diff - quadrant_sum)

    # Incremental transaction costs
    if timing_turnovers is not None and constant_turnovers is not None:
        to_timing = np.asarray(timing_turnovers, dtype=float)
        to_const = np.asarray(constant_turnovers, dtype=float)
        inc_tc = float(cost_bps * np.sum(to_timing - to_const))
    else:
        # Fallback to macro exposure turnover differences if asset-level turnovers not passed
        to_timing_macro = np.abs(np.diff(s_arr)) if T > 1 else np.array([0.0])
        inc_tc = float(cost_bps * np.sum(to_timing_macro))

    net_benefit = gross_diff - inc_tc

    return {
        "losses_avoided": l_avoid,
        "upside_sacrificed": u_sac,
        "extra_upside_captured": u_cap,
        "extra_downside_suffered": d_suf,
        "gross_return_difference": gross_diff,
        "incremental_transaction_costs": inc_tc,
        "net_economic_benefit": net_benefit,
        "closure_error": closure_err,
    }


def build_portfolio_ledger(
    epochs: List[Dict[str, Any]],
    exposure_path: Sequence[float],
    policy_name: str = "Strategy",
    gamma: float = 1.0,
    cost_bps: float = 0.0010,
    r_cash_annual: float = 0.0,
    ann_factor: float = 52.0,
) -> List[Dict[str, Any]]:
    """
    Generate an auditable, period-level source-of-truth portfolio ledger.
    Applies the canonical brokerage convention:
        - Transaction cost TC_t = cost_bps * turnover_risky_t
        - Net return r_net,t = r_gross,t - TC_t
        - Net utility u_t = r_net,t - 0.5 * gamma * variance_t (single subtraction of TC)

    Args:
        epochs: List of epoch dictionaries with base weights, returns, and covariance.
        exposure_path: Sequence of portfolio exposure scales s_t.
        policy_name: Label of the policy.
        gamma: Risk aversion parameter.
        cost_bps: Cost rate per unit of traded risky volume.
        r_cash_annual: Risk-free rate.
        ann_factor: Annualization factor.

    Returns:
        List[Dict[str, Any]]: Complete period-by-period ledger.
    """
    T = len(epochs)
    r_cash_weekly = r_cash_annual / ann_factor
    ledger = []

    prev_w_r = None
    prev_w_c = None
    prev_q = None
    prev_assets = []
    wealth = 1.0

    for t in range(T):
        ep = epochs[t]
        s = float(exposure_path[t])
        w_risky, w_cash = apply_portfolio_exposure_scale(ep["w_base"], scale=s)
        base_w = np.asarray(ep["w_base"], dtype=float)
        base_sum = np.sum(base_w)
        cur_q = base_w / base_sum if base_sum > 0 else np.zeros_like(base_w)

        # Asset alignment
        cur_assets = ep["assets"]
        if prev_w_r is None:
            to_r = 0.0
            to_c = 0.0
            to_comp = 0.0
        else:
            all_a = sorted(set(cur_assets) | set(prev_assets))
            c_map = {cur_assets[i]: w_risky[i] for i in range(len(cur_assets))}
            p_map = {prev_assets[i]: prev_w_r[i] for i in range(len(prev_assets))}
            c_q_map = {cur_assets[i]: cur_q[i] for i in range(len(cur_assets))}
            p_q_map = {prev_assets[i]: prev_q[i] for i in range(len(prev_assets))}
            to_r = float(sum(abs(c_map.get(a, 0.0) - p_map.get(a, 0.0)) for a in all_a))
            to_c = float(abs(w_cash - prev_w_c))
            to_comp = float(sum(abs(c_q_map.get(a, 0.0) - p_q_map.get(a, 0.0)) for a in all_a))

        to_tot = to_r + to_c
        to_half = 0.5 * to_tot

        # Returns and PnL
        risky_pnl = float(w_risky @ ep["real_r"])
        cash_pnl = float(w_cash * r_cash_weekly)
        gross_r = risky_pnl + cash_pnl

        # Canonical transaction cost applied to risky asset trades
        tc = float(cost_bps * to_r)
        net_r = gross_r - tc

        # Risk penalty & Net utility (zero double-subtraction)
        port_var = float(w_risky @ ep["Q_market"] @ w_risky)
        risk_penalty = float(0.5 * gamma * port_var)
        u_net = net_r - risk_penalty

        wealth *= (1.0 + net_r)

        record = {
            "period_idx": t,
            "date": ep.get("week") or ep.get("date") or f"P_{t:03d}",
            "policy": policy_name,
            "exposure": s,
            "risky_exposure": round(s, 6),
            "risky_composition_weights": [round(float(x), 6) for x in cur_q],
            "actual_risky_weights": [round(float(x), 6) for x in w_risky],
            "risky_weights": [round(float(x), 6) for x in w_risky],
            "cash_weight": round(float(w_cash), 6),
            "cash_weight_change": to_c,
            "gross_return": gross_r,
            "risky_pnl": risky_pnl,
            "cash_pnl": cash_pnl,
            "risky_composition_turnover": to_comp,
            "actual_risky_traded_turnover": to_r,
            "turnover_risky": to_r,
            "turnover_cash": to_c,
            "turnover_total": to_tot,
            "turnover_half": to_half,
            "costed_turnover": to_r,
            "transaction_cost": tc,
            "net_return": net_r,
            "risk_penalty": risk_penalty,
            "net_utility": u_net,
            "cumulative_wealth": wealth,
        }
        ledger.append(record)

        prev_w_r = w_risky
        prev_w_c = w_cash
        prev_q = cur_q
        prev_assets = cur_assets

    return ledger


# =============================================================================
# Phase 8.4: Base-Portfolio Churn & Stability Diagnostic Helpers
# =============================================================================


def compute_base_portfolio_churn(
    q_prev: Sequence[float],
    q_cur: Sequence[float],
    assets_prev: Sequence[str],
    assets_cur: Sequence[str],
) -> Dict[str, float]:
    """
    Compute base portfolio L1 turnover, half-turnover, and holding overlap.

    Mathematical identity for simplex portfolios:
        TO_base = 2 * (1 - overlap)
    """
    all_assets = sorted(set(assets_prev) | set(assets_cur))
    p_map = {assets_prev[i]: float(q_prev[i]) for i in range(len(assets_prev))}
    c_map = {assets_cur[i]: float(q_cur[i]) for i in range(len(assets_cur))}

    to_base = float(sum(abs(c_map.get(a, 0.0) - p_map.get(a, 0.0)) for a in all_assets))
    to_base_half = 0.5 * to_base
    overlap = float(sum(min(c_map.get(a, 0.0), p_map.get(a, 0.0)) for a in all_assets))
    expected_to = 2.0 * (1.0 - overlap)
    identity_error = float(abs(to_base - expected_to))

    return {
        "to_base": to_base,
        "to_base_half": to_base_half,
        "overlap": overlap,
        "identity_error": identity_error,
    }


def classify_asset_transitions(
    q_prev: Sequence[float],
    q_cur: Sequence[float],
    assets_prev: Sequence[str],
    assets_cur: Sequence[str],
    tol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Classify asset positions into retained, entered, exited, increased, and decreased.
    """
    all_assets = sorted(set(assets_prev) | set(assets_cur))
    p_map = {assets_prev[i]: float(q_prev[i]) for i in range(len(assets_prev))}
    c_map = {assets_cur[i]: float(q_cur[i]) for i in range(len(assets_cur))}

    retained = []
    entered_from_zero = []
    exited_to_zero = []
    increased = []
    decreased = []

    for a in all_assets:
        w_prev = p_map.get(a, 0.0)
        w_cur = c_map.get(a, 0.0)

        if w_prev > tol and w_cur > tol:
            retained.append(a)
        elif w_prev <= tol and w_cur > tol:
            entered_from_zero.append(a)
        elif w_prev > tol and w_cur <= tol:
            exited_to_zero.append(a)

        if w_cur > w_prev + tol:
            increased.append(a)
        elif w_cur < w_prev - tol:
            decreased.append(a)

    return {
        "retained": retained,
        "entered_from_zero": entered_from_zero,
        "exited_to_zero": exited_to_zero,
        "increased": increased,
        "decreased": decreased,
        "n_retained": len(retained),
        "n_entered": len(entered_from_zero),
        "n_exited": len(exited_to_zero),
        "n_increased": len(increased),
        "n_decreased": len(decreased),
    }


def compute_weight_boundary_metrics(
    weights: Sequence[float],
    tol: float = 1e-4,
) -> Dict[str, float]:
    """
    Evaluate boundary-hit frequency and concentration on the simplex.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    if n == 0:
        return {
            "fraction_zero": 0.0,
            "has_zero": 0.0,
            "max_weight": 0.0,
            "gt_25": 0.0,
            "gt_50": 0.0,
            "gt_75": 0.0,
            "hhi": 0.0,
        }

    zeros = np.sum(w <= tol)
    max_w = float(np.max(w))
    hhi = float(np.sum(w**2))

    return {
        "fraction_zero": float(zeros / n),
        "has_zero": 1.0 if zeros > 0 else 0.0,
        "max_weight": max_w,
        "gt_25": 1.0 if max_w > 0.25 else 0.0,
        "gt_50": 1.0 if max_w > 0.50 else 0.0,
        "gt_75": 1.0 if max_w > 0.75 else 0.0,
        "hhi": hhi,
    }


def compute_portfolio_objective_regret(
    ep_prev: Dict[str, Any],
    ep_cur: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute objective switch cost and regret of holding legacy portfolio into current period.

    Returns:
        f_prev: Objective value of legacy portfolio under current conditions.
        f_cur: Objective value of current base portfolio under current conditions.
        f_opt_joint: Optimal objective value over the joint universe.
        switch_cost: f_prev - f_cur (objective difference vs current base).
        regret_joint: f_prev - f_opt_joint (guaranteed >= 0).
        to_base: Base turnover between legacy and current base.
        regret_per_turnover: regret_joint / (to_base + epsilon).
    """
    assets_union = sorted(set(ep_prev["assets"]) | set(ep_cur["assets"]))
    m = len(assets_union)
    u_map = {a: i for i, a in enumerate(assets_union)}

    q_prev = np.zeros(m)
    for a, w in zip(ep_prev["assets"], ep_prev["w_base"]):
        q_prev[u_map[a]] = float(w)

    q_cur = np.zeros(m)
    for a, w in zip(ep_cur["assets"], ep_cur["w_base"]):
        q_cur[u_map[a]] = float(w)

    mu_cur = np.zeros(m)
    for a, mu in zip(ep_cur["assets"], ep_cur["mu_hat"]):
        mu_cur[u_map[a]] = float(mu)

    # Standard equicorrelated covariance matrix on union
    Q_cur = 0.04 * np.eye(m) + 0.01 * np.ones((m, m))

    f_prev = float(q_prev @ Q_cur @ q_prev - mu_cur @ q_prev)
    f_cur = float(q_cur @ Q_cur @ q_cur - mu_cur @ q_cur)

    # Solve optimal on joint universe
    from hypergraph_portfolio.mean_variance import MeanVariancePortfolio

    mv = MeanVariancePortfolio(Q=Q_cur, mu=mu_cur, budget=1.0)
    w_opt, f_opt = mv.solve()

    switch_cost = float(f_prev - f_cur)
    regret_joint = float(max(0.0, f_prev - f_opt))
    to_base = float(np.sum(np.abs(q_cur - q_prev)))
    regret_per_to = float(regret_joint / (to_base + 1e-6))

    return {
        "f_prev": f_prev,
        "f_cur": f_cur,
        "f_opt_joint": f_opt,
        "switch_cost": switch_cost,
        "regret_joint": regret_joint,
        "to_base": to_base,
        "regret_per_turnover": regret_per_to,
    }

