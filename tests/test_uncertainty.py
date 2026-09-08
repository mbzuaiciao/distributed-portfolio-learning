"""Unit tests for Bayesian uncertainty mechanisms."""

import numpy as np

from hypergraph_portfolio.uncertainty import (
    BayesianReturnShrinkage,
    BayesianRiskAugmentation,
    DownsideRiskAdjustment,
    SubportfolioImportanceWeighting,
)


def test_bayesian_return_shrinkage():
    mu = np.array([0.10, 0.20])
    var_mu = np.array([0.01, 0.10])  # Asset 2 has 10x higher epistemic uncertainty
    shrinkage = BayesianReturnShrinkage(kappa=10.0)
    mu_shrunk = shrinkage.apply(mu, var_mu)

    # Asset 1: 0.10 / (1 + 0.10) = 0.0909 (9% drop)
    # Asset 2: 0.20 / (1 + 1.00) = 0.1000 (50% drop)
    assert mu_shrunk[0] > 0.09
    assert np.isclose(mu_shrunk[1], 0.10)
    assert mu_shrunk[1] / mu[1] < mu_shrunk[0] / mu[0]


def test_bayesian_risk_augmentation():
    Q = np.array([[2.0, 0.5], [0.5, 2.0]])
    var_mu = np.array([0.0, 1.0])
    risk_aug = BayesianRiskAugmentation(gamma=2.0)
    Q_aug = risk_aug.apply(Q, var_mu)

    assert Q_aug[0, 0] == 2.0
    assert Q_aug[1, 1] == 4.0  # 2.0 + 2.0 * 1.0 = 4.0
    assert Q_aug[0, 1] == 0.5  # Cross-covariances unchanged


def test_subportfolio_importance_weighting():
    base_a = np.array([0.5, 0.5])
    # Subportfolio 1 has low uncertainty, SP2 has high uncertainty
    sub_vars = [np.array([0.01, 0.01]), np.array([0.20, 0.30])]
    reweighter = SubportfolioImportanceWeighting(beta=2.0)
    new_a = reweighter.apply(base_a, sub_vars)

    assert new_a[0] > new_a[1]
    assert np.isclose(np.sum(new_a), 1.0)


def test_downside_penalty():
    mu = np.array([0.05, 0.05])
    var_mu = np.array([0.0001, 0.04])  # std = 0.01 vs 0.20
    downside = DownsideRiskAdjustment(target_return=0.04, penalty_scale=0.02)
    adj_mu = downside.apply(mu, var_mu)

    # Asset 1 is 1 std above target -> low prob underperform
    # Asset 2 has large std -> significant prob underperform
    assert adj_mu[0] > adj_mu[1]
