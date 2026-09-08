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

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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

