export interface ResearchFinding {
  id: string;
  number: number;
  question: string;
  summary: string;
  detailedFindings: string;
  status: 'confirmed' | 'negative' | 'audited' | 'mixed';
  statusLabel: string;
  methodology: 'synthetic' | 'empirical' | 'mathematical' | 'protocol';
  methodologyLabel: string;
  caveats: string[];
  canonicalSource: {
    title: string;
    path: string;
    isExternal?: boolean;
  };
  relatedTutorials: number[];
}

export interface ResearchTheme {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  findings: ResearchFinding[];
}

export const RESEARCH_FINDINGS: ResearchFinding[] = [
  {
    id: 'distributed-equivalence-audit',
    number: 1,
    question: 'Are the centralized Markowitz problem and the paper’s distributed hypergraph problem informationally equivalent?',
    summary: 'No. The reported objective reduction (0.8573 vs 2.8408) is a mathematical artifact of omitting cross-subportfolio covariances in local budget splits.',
    detailedFindings: 'By substituting the consensus condition into the paper’s objective, diagonal variance is downscaled by 1/(K*d_j) and off-diagonal cross-subportfolio covariances vanish. Evaluated under the true ground-truth market quadratic form, the distributed allocations yield higher true variance (2.1490 vs 2.0436). Exact Markowitz equivalence is restored by the true lifted formulation Q_lifted = E Q E^T.',
    status: 'audited',
    statusLabel: 'Audited & Corrected',
    methodology: 'mathematical',
    methodologyLabel: 'Analytical Proof & CVXPY Audit',
    caveats: [
      'The paper’s continuous-time primal-dual ODE is mathematically sound and Lyapunov-stable, but optimizes a relaxed proxy objective.',
      'True Markowitz equivalence requires the lifted formulation, which coordinates across all overlapping hyperedges.'
    ],
    canonicalSource: {
      title: 'Tutorial 09: Critical Audit of the Central Claim',
      path: '/learn/09_equivalence_audit',
    },
    relatedTutorials: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13],
  },
  {
    id: 'bayesian-lora-leakage-audit',
    number: 2,
    question: 'Does Bayesian-LoRA uncertainty guide downside risk classification, or did outcome leakage enter training?',
    summary: 'Phase 16.1 audit revealed outcome leakage in earlier training text; clean regularized deterministic LoRA strictly dominates Bayesian LoRA on clean data.',
    detailedFindings: 'Selective risk curves failed to reduce downside classification error (20.3% -> 20.9%). An audit revealed that training text in Phase 11/16 leaked "move_1d", causing artificial 0.9999 validation AUROC and test-time collapse to 0.50. When trained on strictly clean text, regularized deterministic LoRA strictly outperformed Bayesian LoRA: AUROC 0.5420 vs 0.5001, Brier 0.1597 vs 0.1932, and ECE 0.0215 vs 0.1757.',
    status: 'audited',
    statusLabel: 'Audited & Falsified (Phase 16.1)',
    methodology: 'empirical',
    methodologyLabel: 'Empirical Holdout & Data Provenance Audit',
    caveats: [
      'Entropy-error correlation in prior phases was a mechanical artifact of 80/20 class imbalance, not epistemic insight.',
      'The simple null base-rate predictor (p = 0.20) achieves Brier 0.1600, beating raw Bayesian LoRA (0.1932).'
    ],
    canonicalSource: {
      title: 'Phase 16.1 Audit Report',
      path: '/experiments/artifacts/phase16_1_audit/phase16_1_report',
    },
    relatedTutorials: [11, 15, 16, 17, 21],
  },
  {
    id: 'collaborative-lora-closure',
    number: 3,
    question: 'Can heterogeneous LoRA specialists trained on sector tasks collaborate to improve a held-out target task?',
    summary: 'No collaborative gain was established under the frozen protocol. Learned weights degenerated to banking-only selection, and the pilot was closed.',
    detailedFindings: 'Under a pre-registered protocol requiring mean Brier improvement >= 0.002 across all 3 seeds and lower 95% bootstrap CI > 0, CoLoRA and learned mixtures failed all gates. Mean Brier: target-only 0.1937, pooled 0.1964, equal mixture 0.2086, learned mixture 0.1948, CoLoRA 0.1964, prevalence null 0.1841. The branch was formally closed as a negative result.',
    status: 'negative',
    statusLabel: 'Closed — Negative Result',
    methodology: 'protocol',
    methodologyLabel: 'Pre-registered Retrospective Pilot',
    caveats: [
      'Results hold for financial announcement metadata with ~128 events per specialist domain.',
      'Prevalence baseline remained superior to all collaborative models on log loss and Brier score.',
      'The laboratory honored pre-registered stopping rules rather than pursuing data-mining.'
    ],
    canonicalSource: {
      title: 'Collaborative LoRA Branch Closure Summary',
      path: '/experiments/docs/collaborative_lora_branch_closure',
    },
    relatedTutorials: [11, 12, 17, 18, 21],
  },
  {
    id: 'temporal-regularization-frontier',
    number: 4,
    question: 'What determines the optimal overlapping-cohort holding horizon in quantitative portfolios?',
    summary: 'Phase 20 confirmed that overlapping cohort stacking acts as a formal temporal regularizer, establishing a 5-parameter monotonic economic frontier.',
    detailedFindings: 'Evaluated across 288,000 cell-seed simulations under exact accounting closure (max error 3.14e-17): H1 Cost monotonicity confirmed (100%), H2 Persistence monotonicity confirmed (90%), H3 Short-lived-alpha reversal confirmed (100%), H4 Noise regularization confirmed (100%), and H5 Risk-aversion effect confirmed (100%). Stacking amortizes turnover by ~1/H and diversifies idiosyncratic timing noise.',
    status: 'confirmed',
    statusLabel: 'Confirmed Frontier (Phase 20)',
    methodology: 'synthetic',
    methodologyLabel: 'Controlled Synthetic DGP (30 Seeds, 288k Cells)',
    caveats: [
      'Simulations utilize controlled synthetic data-generating processes where all active cohorts share realized returns.',
      'Under exact null alpha (beta = 0), turnover is identically zero and all horizons tie mathematically.',
      'When predictive signals decay very rapidly (tau < 1 wk) without friction, holding stale cohorts dilutes alpha (favoring H=1).'
    ],
    canonicalSource: {
      title: 'Phase 20 Scientific Report: Optimal Holding Horizon',
      path: '/experiments/artifacts/phase20_optimal_cohort_horizon/scientific_report',
    },
    relatedTutorials: [41, 12],
  },
  {
    id: 'epistemic-exposure-timing',
    number: 5,
    question: 'Does relative epistemic timing survive 15 adversarial falsification dimensions?',
    summary: 'Relative epistemic timing survives walk-forward validation and block placebos, but requires hysteresis filters to overcome transaction costs.',
    detailedFindings: 'Raw continuous exposure scaling suffers severe turnover churn under real-world transaction friction (10-25 bps). Introducing a two-threshold hysteresis state machine (tau_enter = 1.2, tau_exit = 0.2) slashes execution turnover by over 70% while preserving downside drawdown avoidance across historical market shocks.',
    status: 'confirmed',
    statusLabel: 'Confirmed Under Hysteresis',
    methodology: 'empirical',
    methodologyLabel: 'Nested Walk-Forward & Block Placebos',
    caveats: [
      'Raw point-in-time epistemic dispersion is noisy and prone to whipsaws without persistence filtering.',
      'Walk-forward model selection between M13 and M17 exhibited sensitivity to burn-in window length (Tutorial 39).'
    ],
    canonicalSource: {
      title: 'Tutorial 40: Frozen Confirmation & Scientific Verdict',
      path: '/learn/40_frozen_confirmation_and_verdict',
    },
    relatedTutorials: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
  },
  {
    id: 'cohort-stacking-h4-revalidation',
    number: 6,
    question: 'Does H=4 cohort stacking survive clean deterministic Phase 16.1 recipe reproduction?',
    summary: 'Structural mechanics survive (turnover -74%, volatility -66%), but persistent alpha and net economic gain failed frozen confirmation gates.',
    detailedFindings: 'In Phase 18 clean cohort revalidation, H=4 delivered net weekly utility 27.88 bp vs H=1 14.07 bp (delta +13.80 bp), but the paired 95% bootstrap CI spanned [-54.68, +84.33] bp with permutation p = 0.105. Turnover reduction (-73.97%) and volatility dampening (-66.21%) were verified as structural mathematical properties.',
    status: 'mixed',
    statusLabel: 'Outcome B — Architecture Only',
    methodology: 'empirical',
    methodologyLabel: 'Historical Reconstructed Backtest',
    caveats: [
      'Endpoint reconstruction only; did not pass statistical significance gates for prospective alpha.',
      'Led directly to Phase 19 stress-testing and the Phase 20 theoretical resolution.'
    ],
    canonicalSource: {
      title: 'Phase 18 Scientific Report: Clean Cohort Revalidation',
      path: '/experiments/artifacts/phase18_clean_cohort_revalidation/scientific_report',
    },
    relatedTutorials: [37, 38, 39, 40],
  },
];

export const RESEARCH_THEMES: ResearchTheme[] = [
  {
    id: 'distributed-optimization',
    title: 'Distributed Hypergraph Portfolio Optimization',
    subtitle: 'Decentralized primal-dual consensus, pricing, and informational equivalence',
    description: 'Investigating whether decentralized asset desks can achieve optimal global diversification without sharing proprietary trade weights or private alpha forecasts.',
    findings: [RESEARCH_FINDINGS[0]],
  },
  {
    id: 'uncertainty-audits',
    title: 'Epistemic Uncertainty & LLM Integrity Audits',
    subtitle: 'Bayesian LoRA, training leakage discovery, and specialist collaboration',
    description: 'Rigorously auditing whether LLM predictive uncertainty adds economic value, exposing subtle leakage in historical training text, and testing multi-task LoRA collaboration.',
    findings: [RESEARCH_FINDINGS[1], RESEARCH_FINDINGS[2]],
  },
  {
    id: 'cohort-stacking',
    title: 'Temporal Regularization & Cohort Stacking',
    subtitle: 'Overcoming portfolio churn and mapping the optimal holding horizon',
    description: 'Resolving portfolio turnover drag through staggered overlapping cohorts, establishing the mathematical frontier of temporal regularization.',
    findings: [RESEARCH_FINDINGS[3], RESEARCH_FINDINGS[5]],
  },
  {
    id: 'falsification-timing',
    title: 'Falsification & Execution Mechanics',
    subtitle: 'Adversarial placebo stress tests, hysteresis filters, and cost hurdles',
    description: 'Applying strict Popperian falsification protocols to market timing, separating true predictive signal from execution friction and hindsight bias.',
    findings: [RESEARCH_FINDINGS[4]],
  },
];
