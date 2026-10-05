# Results Inventory & Provenance Taxonomy

This inventory classifies every empirical result and artifact generated across the research trajectory (Phases 11 through 17). It establishes an authoritative scientific boundary between **compromised historical artifacts** (preserved strictly for forensic documentation) and **valid clean evidence** supporting the paper.

---

## Provenance Classification Scheme

1. **COMPROMISED BY LEAKAGE**: Artifacts and models trained on input strings containing continuous outcome realizations (`move_1d`). Strictly invalid for evaluating predictive accuracy, calibration, or model superiority. Quarantined for forensic inspection only.
2. **SUPERSEDED**: Historical experimental comparisons or baselines whose conclusions were rendered obsolete by subsequent forensic discoveries or controlled retraining.
3. **DIAGNOSTIC ONLY**: Methodological probes, provenance audits, and mechanistic decompositions designed to isolate pipeline behavior rather than serve as headline positive evidence.
4. **VALID CLEAN SECONDARY**: Historical clean evaluations or portfolio simulations conducted on clean test text, but potentially evaluated against compromised upstream model predictions, or stylized portfolio sanity checks.
5. **VALID CLEAN PRIMARY**: Retrained models, audits, and evaluations conducted with strictly clean inputs, preregistered protocols, and controlled statistical inference.

---

## Master Inventory of Phases 11–17

| Phase | Description / Artifact Set | Provenance Status | Rationale & Paper Role |
| :---: | :--- | :---: | :--- |
| **Phase 11** | Replicated Bayesian-LoRA training checkpoints (`bayesian_lora_seed_*.pt`), validation AUROC ($>0.999$), and 2024–2025 holdout predictions. | **COMPROMISED BY LEAKAGE** | Training text concatenated `. 1-day move was ` + `move_1d`. Formally quarantined in `legacy_checkpoint_status.json`. Used in Section 3/4 solely to document the mechanism of target leakage. |
| **Phase 12** | Prospective 2026 blind event ingestion, predictions, and evaluation. | **DIAGNOSTIC ONLY / COMPROMISED MODEL** | Input text for 2026 was clean, but the evaluated model adapter was the leaked Phase-11 checkpoint. Serves as forensic evidence of distribution collapse on clean text. |
| **Phase 13** | Clean post-training regime partitions (2022, 2023, 2024–2025; $N=4,100$ clean events) and overlap audit. | **VALID CLEAN SECONDARY** | Partitioning and clean universe definitions are mathematically valid and form the basis for the Phase 17 clean test universe ($N_{\text{unique}}=4,100$). |
| **Phase 13C** | Frozen portfolio evaluation ($M_0, M_{13}, M_{14}, M_{15}, M_{16}$) on clean regimes. | **DIAGNOSTIC ONLY** | Evaluated portfolio policies using leaked Phase-11 model signals. Demonstrates that uncertainty mappings failed to provide stable gains across regimes even under high training confidence. |
| **Phase 14** | Mechanistic decomposition of uncertainty-aware risk inflation ($M_{16}$). | **DIAGNOSTIC ONLY** | Decomposed why risk inflation helped in 2023 (bear-market de-risking) but hurt in 2022 and 2024–2025. Illustrates market-regime sensitivity. |
| **Phase 15** | Incremental predictive information content of Bayesian epistemic uncertainty ($\sigma_{\text{epi}}^2$) vs. deterministic confidence. | **SUPERSEDED / DIAGNOSTIC** | Demonstrated that predictive entropy matches MC variance, but the probabilities evaluated derived from leaked Phase-11 models. |
| **Phase 15.1** | Deterministic confidence provenance audit. | **VALID CLEAN SECONDARY** | Demonstrated that evaluating the Bayesian posterior mean parameters $\mu$ in a single forward pass reproduces MC32 confidence ranking with $\rho > 0.999$, establishing inference redundancy. |
| **Phase 16** | Deterministic LoRA control training and comparison against Phase 11 Bayesian LoRA. | **COMPROMISED BY LEAKAGE** | The original deterministic control script mirrored Phase 11's training text, inadvertently inheriting the `. 1-day move was ` string. Formally quarantined. |
| **Phase 16.1** | Reliability metric audit, within-class decomposition, and base-rate null control. | **VALID CLEAN PRIMARY** | Discovered the leakage pipeline bug; proved mathematically that $\text{Spearman}(\text{entropy}, \text{error})$ within-class is mechanically $\pm 1.0$; demonstrated base-rate null beats leaked models. |
| **Phase 16.1** | Regularized deterministic LoRA grid search and clean OOS evaluation. | **VALID CLEAN PRIMARY** | First clean retraining of deterministic LoRA with $L_2$ regularization, proving clean deterministic models match or beat leaked Bayesian models. |
| **Phase 17** | Preregistration (`phase17_preregistration.json`) and quarantine commit (`bd84126`). | **VALID CLEAN PRIMARY** | Predeclared all architectures, hyperparameters, data hashes ($N_{\text{train}}=11,985$, $N_{\text{val}}=2,116$, $N_{\text{test}}=4,100$), and equivalence margins prior to clean OOS evaluation. |
| **Phase 17** | Clean Bayesian-LoRA and Clean Deterministic LoRA retraining (`clean_*_state.pt`). | **VALID CLEAN CONTEXTUAL** | Retrained both models from scratch on strictly clean metadata (`f"{company} ({ticker}) quarterly earnings announcement on {date_str}."`). Established scoring equivalence with deterministic LoRA and base-rate null. |
| **Phase 18** | Clean Cohort Revalidation on strictly clean metadata inputs (`artifacts/phase18_clean_cohort_revalidation/`). | **VALID CLEAN SECONDARY** | Proved post-earnings alpha did not persist across weeks 2–4 ($p=0.105$), but $H=4$ mechanics survived ($-74\%$ turnover, $-66\%$ vol). Classified as Outcome B (architecture only). |
| **Phase 19** | Architecture-General Signal Stress Test across empirical, random, constant, and synthetic signals (`artifacts/phase19_architecture_signal_stress_test/`). | **VALID CLEAN SECONDARY** | Confirmed turnover and vol savings are architecture-general; revealed reversal showing stacking is most advantageous under weak/null signals. Classified as Outcome D. |
| **Phase 20** | Controlled Synthetic Optimal Cohort Holding Horizon Experiment ($N=360{,}000$ evaluations, 30 seeds; `artifacts/phase20_optimal_cohort_horizon/`). | **VALID CLEAN PRIMARY (HEADLINE EVIDENCE)** | Established the formal temporal regularization frontier $H^* = f(\tau_{\text{signal}}, c, \sigma_{\text{noise}}, \gamma, \beta)$. Confirmed 5/5 comparative static hypotheses, machine-precision accounting closure ($3.14\times 10^{-17}$), and exact-null tie invariance. Classified as Outcome A. |

---

## Detailed Data Inventory for Clean Primary Science

All primary empirical claims, tables, and figures in the paper are derived exclusively from the following verified artifacts:

```
artifacts/phase20_optimal_cohort_horizon/
├── preregistration.json                  # Frozen simulation design specification
├── experiment_matrix.csv                 # 360,000 full factorial evaluation results
├── horizon_optimum_table.csv             # Optimal horizon selections across 600 parameter cells
├── uncertainty_and_ties.csv              # Paired t-test statistics and tie classifications
├── breakeven_boundaries.csv              # Closed-form and interpolated break-even cost surfaces
├── empirical_sanity_mapping.json         # Mapping of empirical LoRA parameters into synthetic space
├── null_regime_invariant_audit.json      # Machine-precision invariant audit across 72,000 beta=0 evaluations
├── null_regime_invariant_audit.md        # Comprehensive null-regime invariant audit report
├── outcome_classification.json           # Formal Outcome A decision record
├── scientific_report.md                  # Comprehensive scientific report
├── analytical_derivation.md              # Mathematical derivation of utility and comparative statics
├── horizon_phase_diagram.png             # Primary 2D horizon phase diagram
└── horizon_phase_diagram.pdf             # Vector phase diagram for manuscript
```

And contextual historical audit artifacts:
```
artifacts/phase17_clean_bayesian/
├── clean_text_manifest.csv              # 18,201 clean event strings (11,985 train, 2,116 val, 4,100 test)
├── phase17_preregistration.json         # Frozen preregistration specification
├── legacy_checkpoint_status.json        # Formal quarantine manifest of legacy checkpoints
├── clean_regularized_deterministic_lora_seed_42_state.pt   # Clean Deterministic Checkpoint (Seed 42)
├── clean_regularized_deterministic_lora_seed_123_state.pt  # Clean Deterministic Checkpoint (Seed 123)
├── clean_regularized_deterministic_lora_seed_777_state.pt  # Clean Deterministic Checkpoint (Seed 777)
├── clean_bayesian_lora_seed_42_state.pt  # Clean Bayesian Checkpoint (Seed 42)
├── clean_bayesian_lora_seed_123_state.pt # Clean Bayesian Checkpoint (Seed 123)
├── clean_bayesian_lora_seed_777_state.pt # Clean Bayesian Checkpoint (Seed 777)
├── clean_oos_predictions.csv            # 12,300 OOS event predictions (4,100 unique x 3 seeds)
├── forecast_metrics.csv                 # Primary pooled forecast metrics
├── regime_metrics.csv                   # Regime-by-regime breakdowns (2022, 2023, 2024-2025)
├── seed_metrics.csv                     # Seed-by-seed breakdowns (42, 123, 777)
├── selective_risk.csv                   # Selective abstention risk curves (100% to 50%)
├── calibration_metrics.csv              # 10-bin calibration data
├── bootstrap_intervals.csv              # B=1,000 calendar-week clustered bootstrap CIs
├── equivalence_analysis.csv             # TOST equivalence analysis against predeclared margins
├── compute_cost.csv                     # Training and inference compute/latency profile
├── portfolio_sanity_check.csv           # M0 unadjusted Markowitz portfolio simulation (Secondary)
└── phase17_summary.json                 # Structured machine-readable results summary
```
