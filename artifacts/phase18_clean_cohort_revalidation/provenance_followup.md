# Subsequent H=4 lineage detail

The preregistration and numerical rules remain unchanged. This table expands the pre-run provenance audit with exact downstream file names; no outcome inspection motivated it.

| Analysis | Prediction artifact | Fitted checkpoint source | Leakage conclusion |
|---|---|---|---|
| Phase 11.3 downstream H=4 inference and Phase 13 local profiling | `artifacts/phase11_replication/replicated_model_historical_predictions.csv` | `artifacts/phase11_replication/frozen_models/bayesian_lora_seed_{42,123,777}_state.pt` | Proven outcome-bearing training strings in Phase 11. |
| Phase 12 prospective evaluation and 12.1/12.2 audits | `artifacts/phase11_replication/frozen_2026_predictions.csv` | Same Phase 11 frozen models | Leaked training lineage; several claimed prospective events were later shown retrospective. Excluded. |
| Phase 13B development mapping selection | `artifacts/phase13_uncertainty_mapping/development_predictions.csv` | Same Phase 11 frozen models, genuine inference for development events | Leaked training lineage even where inference text is clean. |
| Phase 13C H=4 regime evaluation | `artifacts/phase13_uncertainty_mapping/evaluation_predictions.csv` | Same Phase 11 models; 2024–2025 rows reused from `replicated_model_historical_predictions.csv` where available | Leaked training lineage; 2020–2021 also overlap model training and are not clean OOS. |
| Phase 14 M16 mechanism | `artifacts/phase13_uncertainty_mapping/evaluation_predictions.csv` | Inherits Phase 13C/Phase 11 lineage | Leaked training lineage. |
| Phase 15 incremental uncertainty controls | Same Phase 13C `evaluation_predictions.csv` | Inherits Phase 11 lineage | Leaked training lineage. This analysis also evaluates timing/control endpoints; it is not a new clean H=4 signal. |

These are distinct from the original Phase 9 CSVs. The main Phase 18 predictor uses none of them.
