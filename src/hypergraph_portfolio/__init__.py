"""
Hypergraph Portfolio Lab
========================
A pedagogical library for distributed portfolio optimization and decentralized pricing,
implementing and auditing the methods from Papastaikoudis, Watson, and Lestas (2025).
"""

from hypergraph_portfolio.bayesian_lora_adapter import (
    BayesianDatasetAdapter,
    BayesianPredictionRecord,
    validate_prediction_records,
)
from hypergraph_portfolio.consensus import (
    build_true_lifted_system,
    check_consensus,
    compare_matrices_term_by_term,
    lift_centralized_to_distributed,
    project_onto_consensus,
    reduce_distributed_objective,
)
from hypergraph_portfolio.hypergraph import (
    HypergraphPortfolio,
    build_powerset_subportfolios,
)
from hypergraph_portfolio.mean_variance import (
    MeanVariancePortfolio,
    solve_centralized_portfolio,
)
from hypergraph_portfolio.pricing import (
    compute_decentralized_prices,
    compute_expected_decentralized_prices,
)
from hypergraph_portfolio.primal_dual import (
    PrimalDualHistory,
    PrimalDualSolver,
)
from hypergraph_portfolio.uncertainty import (
    BayesianReturnShrinkage,
    BayesianRiskAugmentation,
    DistributedUncertaintyConfig,
    SubportfolioImportanceWeighting,
    SyntheticMarketEnvironment,
    apply_downside_adjusted_alpha,
    apply_downside_covariance_inflation,
    apply_epistemic_covariance_inflation,
    apply_portfolio_exposure_scale,
    apply_uncertainty_shrinkage,
    compute_aggregate_epistemic_uncertainty,
    compute_epistemic_exposure_scale,
    compute_posterior_downside_probability,
    compute_relative_epistemic_exposure_scale,
    compute_relative_epistemic_uncertainty,
    compute_volatility_targeting_scale,
    generate_placebo_exposure_paths,
    run_sequential_portfolio_simulation,
)

__all__ = [
    "MeanVariancePortfolio",
    "solve_centralized_portfolio",
    "HypergraphPortfolio",
    "build_powerset_subportfolios",
    "check_consensus",
    "project_onto_consensus",
    "lift_centralized_to_distributed",
    "reduce_distributed_objective",
    "compare_matrices_term_by_term",
    "build_true_lifted_system",
    "PrimalDualSolver",
    "PrimalDualHistory",
    "compute_decentralized_prices",
    "compute_expected_decentralized_prices",
    "BayesianReturnShrinkage",
    "BayesianRiskAugmentation",
    "SubportfolioImportanceWeighting",
    "DistributedUncertaintyConfig",
    "apply_uncertainty_shrinkage",
    "apply_epistemic_covariance_inflation",
    "compute_posterior_downside_probability",
    "apply_downside_adjusted_alpha",
    "apply_downside_covariance_inflation",
    "compute_aggregate_epistemic_uncertainty",
    "compute_epistemic_exposure_scale",
    "compute_volatility_targeting_scale",
    "apply_portfolio_exposure_scale",
    "compute_relative_epistemic_uncertainty",
    "compute_relative_epistemic_exposure_scale",
    "generate_placebo_exposure_paths",
    "SyntheticMarketEnvironment",
    "run_sequential_portfolio_simulation",
    "BayesianPredictionRecord",
    "BayesianDatasetAdapter",
    "validate_prediction_records",
]



