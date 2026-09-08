"""
Bayesian-LoRA Prediction Adapter and Record Schema
=================================================
Provides a standardized, validated ingestion pipeline for empirical Bayesian-LoRA
predictions, ensuring temporal order, strict separation of forecast and realization,
and robust data validation.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass(frozen=True)
class BayesianPredictionRecord:
    """
    Standardized predictive record for one asset at one decision timestamp.

    Attributes:
        timestamp: Observation or decision timestamp (ISO string or datetime).
        asset: Asset identifier (ticker symbol, asset ID, or desk key).
        mu_hat: Point prediction or posterior predictive mean.
        epistemic_std: Epistemic uncertainty standard deviation (>= 0).
        aleatoric_std: Optional aleatoric uncertainty standard deviation (>= 0).
        realized_return: Ex-post realized next-period return (observed after decision).
        metadata: Optional dictionary holding auxiliary variables (e.g., raw labels).
    """

    timestamp: str
    asset: str
    mu_hat: float
    epistemic_std: float
    aleatoric_std: Optional[float] = None
    realized_return: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            raise ValueError("Record timestamp cannot be empty.")
        if not self.asset:
            raise ValueError("Record asset identifier cannot be empty.")
        if math.isnan(self.mu_hat) or math.isinf(self.mu_hat):
            raise ValueError(f"Invalid mu_hat {self.mu_hat} for asset {self.asset} at {self.timestamp}.")
        if self.epistemic_std < 0 or math.isnan(self.epistemic_std) or math.isinf(self.epistemic_std):
            raise ValueError(
                f"Epistemic uncertainty must be non-negative and finite, got {self.epistemic_std} "
                f"for asset {self.asset} at {self.timestamp}."
            )
        if self.aleatoric_std is not None:
            if self.aleatoric_std < 0 or math.isnan(self.aleatoric_std) or math.isinf(self.aleatoric_std):
                raise ValueError(
                    f"Aleatoric uncertainty must be non-negative and finite, got {self.aleatoric_std}."
                )
        if math.isnan(self.realized_return) or math.isinf(self.realized_return):
            raise ValueError(
                f"Realized return must be finite, got {self.realized_return} for asset {self.asset}."
            )


def validate_prediction_records(
    records: Sequence[BayesianPredictionRecord],
    allow_ties: bool = True,
) -> None:
    """
    Validate an entire sequence of Bayesian prediction records.

    Validation rules:
    1. Records cannot be empty.
    2. Timestamps must be sorted in non-decreasing order (temporal causality).
    3. No duplicate (timestamp, asset) pairs.
    4. Epistemic uncertainty is non-negative and finite.
    5. No obvious lookahead leakage between epistemic uncertainty and realized return.
    """
    if not records:
        raise ValueError("Record sequence is empty.")

    seen_keys: set[tuple[str, str]] = set()
    prev_dt: Optional[datetime] = None

    for idx, rec in enumerate(records):
        # 1. Unique (timestamp, asset) pair
        key = (rec.timestamp, rec.asset)
        if key in seen_keys:
            raise ValueError(f"Duplicate (timestamp, asset) pair found at index {idx}: {key}")
        seen_keys.add(key)

        # 2. Sorted timestamps
        try:
            # Parse standard timestamp formats
            cur_dt = datetime.fromisoformat(rec.timestamp.replace(" ", "T"))
        except Exception:
            # Fallback to string comparison if not ISO
            cur_dt = None

        if cur_dt is not None and prev_dt is not None:
            if allow_ties:
                if cur_dt < prev_dt:
                    raise ValueError(
                        f"Timestamps are out of chronological order at index {idx}: "
                        f"{rec.timestamp} preceded by {records[idx-1].timestamp}"
                    )
            else:
                if cur_dt <= prev_dt:
                    raise ValueError(
                        f"Timestamps are not strictly increasing at index {idx}: {rec.timestamp}"
                    )
        if cur_dt is not None:
            prev_dt = cur_dt


class BayesianDatasetAdapter:
    """
    Adapter for loading and transforming exported Bayesian-LoRA predictions
    into validated BayesianPredictionRecord sequences.
    """

    @staticmethod
    def load_from_csv(
        predictions_path: Union[str, Path],
        reconstructed_meta_path: Optional[Union[str, Path]] = None,
        asset_col: Optional[str] = None,
        mu_hat_col: str = "posterior_mean",
        epistemic_var_col: str = "epistemic_variance",
        return_col: str = "move_1d",
        timestamp_col: str = "date",
        convert_downside_prob_to_alpha: bool = True,
    ) -> List[BayesianPredictionRecord]:
        """
        Load exported predictions and produce a validated list of BayesianPredictionRecord.

        Args:
            predictions_path: Path to the predictions CSV file.
            reconstructed_meta_path: Optional path to reconstructed metadata CSV with symbol/company mappings.
            asset_col: Column name for asset identifier (if present in predictions).
            mu_hat_col: Column name for point forecast or posterior probability.
            epistemic_var_col: Column name for epistemic variance.
            return_col: Column name for realized 1-day percentage return.
            timestamp_col: Column name for timestamp.
            convert_downside_prob_to_alpha: If True, maps downside event probability p into
                directional return signal: mu_hat = -(p - p_mean).

        Returns:
            Validated list of BayesianPredictionRecord objects sorted by timestamp.
        """
        pred_path = Path(predictions_path)
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")

        # Load metadata dictionary if available (date -> list of rows)
        meta_by_date: Dict[str, List[Dict[str, str]]] = {}
        if reconstructed_meta_path is not None:
            meta_path = Path(reconstructed_meta_path)
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        d = r.get("date", "").strip()
                        if d:
                            meta_by_date.setdefault(d, []).append(r)

        # Read predictions
        raw_records = []
        date_occurrences: Dict[str, int] = {}

        with open(pred_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Precompute mean probability if converting downside prob to alpha
        if convert_downside_prob_to_alpha:
            probs = [float(r[mu_hat_col]) for r in rows if mu_hat_col in r and r[mu_hat_col] != ""]
            base_p = float(sum(probs) / len(probs)) if probs else 0.5
        else:
            base_p = 0.0

        for row in rows:
            ts = row[timestamp_col].strip()
            date_occurrences[ts] = date_occurrences.get(ts, 0) + 1
            idx_for_date = date_occurrences[ts] - 1

            # Resolve asset identifier
            asset_id = None
            if asset_col and asset_col in row and row[asset_col]:
                asset_id = row[asset_col].strip()
            elif ts in meta_by_date and idx_for_date < len(meta_by_date[ts]):
                asset_id = meta_by_date[ts][idx_for_date].get("symbol", "").strip()
                if not asset_id:
                    asset_id = meta_by_date[ts][idx_for_date].get("episode_id", "").strip()

            if not asset_id:
                asset_id = f"ASSET_{idx_for_date:03d}"

            # Parse numerical fields
            raw_p = float(row[mu_hat_col])
            var_epi = float(row[epistemic_var_col])
            std_epi = math.sqrt(max(var_epi, 0.0))
            realized_r = float(row[return_col])

            # Convert to expected return signal if requested
            if convert_downside_prob_to_alpha:
                # Downside probability p: higher p means negative expected return
                mu_hat = -(raw_p - base_p)
            else:
                mu_hat = raw_p

            meta_dict = {
                "raw_probability": raw_p,
                "label": int(row.get("label", 0)) if "label" in row else None,
                "correct_at_0.5": (
                    row.get("correct_at_0.5", "").lower() == "true"
                    if "correct_at_0.5" in row
                    else None
                ),
            }
            if "posterior_mean_deterministic" in row:
                meta_dict["det_prediction"] = float(row["posterior_mean_deterministic"])

            record = BayesianPredictionRecord(
                timestamp=ts,
                asset=asset_id,
                mu_hat=mu_hat,
                epistemic_std=std_epi,
                aleatoric_std=None,
                realized_return=realized_r,
                metadata=meta_dict,
            )
            raw_records.append(record)

        # Sort chronologically
        raw_records.sort(key=lambda r: r.timestamp)

        # Validate
        validate_prediction_records(raw_records, allow_ties=True)
        return raw_records
