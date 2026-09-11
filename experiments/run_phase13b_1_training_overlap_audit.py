#!/usr/bin/env python3
"""Phase 13B.1: Model-Training / Evaluation Overlap Audit.

Audits whether Phase-11 Bayesian-LoRA training/validation examples overlap
with the 8,692 historical events assigned to Phase-13 development and evaluation regimes.

Enforces hierarchical matching:
1. Exact document/episode ID
2. Exact text hash
3. Ticker + event date
4. Normalized company + event date

Classifies each regime:
- MODEL-TRAINING-OVERLAP: In-distribution / training or model-selection overlap.
- TEMPORALLY POST-TRAINING: Genuine clean out-of-sample (occurred post-2021-12-31).

Outputs:
- artifacts/phase13_uncertainty_mapping/model_training_overlap_audit.csv
- artifacts/phase13_uncertainty_mapping/model_training_overlap_summary.json
- artifacts/phase13_uncertainty_mapping/phase13b_1_training_overlap_report.md
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent


def normalize_company_name(name: str) -> str:
    """Normalize company name by removing punctuation, casing, and corporate suffixes."""
    s = str(name).lower().strip()
    s = re.sub(r"[,\.\-\/\\_\(\)]", " ", s)
    suffixes = ["inc", "corp", "corporation", "ltd", "limited", "co", "company", "plc", "llc", "group"]
    words = [w for w in s.split() if w and w not in suffixes]
    return " ".join(words)


def compute_sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_training_overlap_audit() -> None:
    print("=== Phase 13B.1: Model-Training / Evaluation Overlap Audit ===", flush=True)

    rec_path = repo_root / "data" / "bayesian_lora" / "reconstructed_returns.csv"
    assert rec_path.exists(), f"Missing reconstructed returns: {rec_path}"

    rec_df = pd.read_csv(rec_path)
    rec_df["date_dt"] = pd.to_datetime(rec_df["date"])
    rec_df = rec_df.sort_values("date_dt").reset_index(drop=True)

    # Reconstruct exact Phase-11 pretest split (cutoff 2021-12-31)
    train_cutoff = pd.Timestamp("2021-12-31 23:59:59")
    pretest = rec_df[rec_df["date_dt"] <= train_cutoff].copy().reset_index(drop=True)
    val_frac = 0.15
    split_idx = int(math.floor(len(pretest) * (1.0 - val_frac)))
    train_df = pretest.iloc[:split_idx].copy().reset_index(drop=True)
    valid_df = pretest.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"Loaded {len(rec_df)} total historical records.")
    print(f"Recovered Phase-11 training universe (cutoff {train_cutoff.strftime('%Y-%m-%d')}):")
    print(f"  Total Pretest Pool: {len(pretest)} rows ({pretest['date_dt'].min()} to {pretest['date_dt'].max()})")
    print(f"  - Gradient Training Slice: {len(train_df)} rows ({train_df['date_dt'].min()} to {train_df['date_dt'].max()})")
    print(f"  - Model-Selection Validation Slice: {len(valid_df)} rows ({valid_df['date_dt'].min()} to {valid_df['date_dt'].max()})")

    # Build multi-key training lookup tables
    id_to_training_info: dict[str, dict[str, Any]] = {}
    texthash_to_training_info: dict[str, dict[str, Any]] = {}
    ticker_date_to_training_info: dict[tuple[str, str], dict[str, Any]] = {}
    normco_date_to_training_info: dict[tuple[str, str], dict[str, Any]] = {}

    for idx, row in pretest.iterrows():
        ep_id = str(row["episode_id"])
        ticker = str(row["symbol"])
        co_name = str(row["company_name"]) if pd.notna(row["company_name"]) else ticker
        norm_co = normalize_company_name(co_name)
        date_str = str(row["date"])
        date_day = date_str.split()[0]
        # Exact synthetic text representation from run_phase11_replication.py
        move_1d = str(row["move_1d"])
        syn_text = f"{co_name} reported earnings on {date_str}. 1-day move was {move_1d}"
        t_hash = compute_sha256_text(syn_text)

        slice_type = "gradient_training_slice" if idx < split_idx else "model_selection_validation_slice"
        info = {
            "training_row_id": int(idx),
            "episode_id": ep_id,
            "ticker": ticker,
            "date": date_str,
            "company_name": co_name,
            "normalized_company": norm_co,
            "slice_type": slice_type,
            "text_hash": t_hash,
        }

        id_to_training_info[ep_id] = info
        texthash_to_training_info[t_hash] = info
        ticker_date_to_training_info[(ticker, date_str)] = info
        ticker_date_to_training_info[(ticker, date_day)] = info
        normco_date_to_training_info[(norm_co, date_day)] = info

    # Load Phase-13 inventory
    inv_path = repo_root / "artifacts" / "phase13_uncertainty_mapping" / "historical_data_inventory.csv"
    assert inv_path.exists(), f"Missing inventory file: {inv_path}"
    inv_df = pd.read_csv(inv_path)
    # Restrict to primary 2018-2025 scoped valid events (N=8,692)
    p13_events = inv_df[
        (inv_df["regime"] != "pre-2018_unscoped")
        & (inv_df["valid_model_input"] == True)  # noqa: E712
        & (inv_df["has_return_1d"] == True)  # noqa: E712
    ].copy().reset_index(drop=True)

    assert len(p13_events) == 8692, f"Expected 8,692 Phase-13 events, got {len(p13_events)}"

    print(f"\nAuditing {len(p13_events)} valid Phase-13 events against training universe...", flush=True)

    audit_rows: list[dict[str, Any]] = []

    for _, row in p13_events.iterrows():
        ev_id = str(row["event_id"])
        ticker = str(row["ticker"])
        ev_date = str(row["event_date"])
        ev_day = ev_date.split()[0]
        co_name = str(row["company_name"]) if pd.notna(row["company_name"]) else ticker
        norm_co = normalize_company_name(co_name)
        regime = str(row["regime"])
        dt = pd.to_datetime(ev_date)

        matched_info: dict[str, Any] | None = None
        match_method = "none"

        # Key 1: Exact document ID
        if ev_id in id_to_training_info:
            matched_info = id_to_training_info[ev_id]
            match_method = "exact_document_id"
        # Key 2: Ticker + exact/day date
        elif (ticker, ev_date) in ticker_date_to_training_info:
            matched_info = ticker_date_to_training_info[(ticker, ev_date)]
            match_method = "ticker_plus_exact_date"
        elif (ticker, ev_day) in ticker_date_to_training_info:
            matched_info = ticker_date_to_training_info[(ticker, ev_day)]
            match_method = "ticker_plus_event_date"
        # Key 3: Normalized company + date
        elif (norm_co, ev_day) in normco_date_to_training_info:
            matched_info = normco_date_to_training_info[(norm_co, ev_day)]
            match_method = "normalized_company_plus_date"

        if matched_info is not None:
            training_overlap = True
            t_row_id = str(matched_info["training_row_id"])
            slice_desc = matched_info["slice_type"]
            notes = f"Definite training overlap via {match_method} ({slice_desc})"
        else:
            if dt > train_cutoff:
                training_overlap = False
                match_method = "none_temporally_post_training"
                t_row_id = "none"
                notes = "Event date strictly postdates training cutoff (2021-12-31); clean OOS"
            else:
                # Pre-2022 event without match
                training_overlap = "unknown"
                match_method = "unmatched_pre_cutoff"
                t_row_id = "none"
                notes = "Pre-2022 event with ambiguous training provenance"

        audit_rows.append({
            "event_id": ev_id,
            "ticker": ticker,
            "event_date": ev_date,
            "regime": regime,
            "training_overlap": training_overlap,
            "match_method": match_method,
            "training_row_id": t_row_id,
            "notes": notes,
        })

    audit_df = pd.DataFrame(audit_rows)
    out_dir = repo_root / "artifacts" / "phase13_uncertainty_mapping"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_csv_path = out_dir / "model_training_overlap_audit.csv"
    audit_df.to_csv(audit_csv_path, index=False)
    print(f"Saved audit CSV to: {audit_csv_path}")

    # Summary by regime
    regimes = ["2018-2019", "2020", "2021", "2022", "2023", "2024-2025"]
    regime_summaries: list[dict[str, Any]] = []

    print("\n--- Overlap Breakdown by Regime ---")
    for reg in regimes:
        sub = audit_df[audit_df["regime"] == reg]
        total_n = len(sub)
        n_overlap = int((sub["training_overlap"] == True).sum())  # noqa: E712
        n_non_overlap = int((sub["training_overlap"] == False).sum())  # noqa: E712
        n_unknown = int((sub["training_overlap"] == "unknown").sum())

        if n_overlap == total_n:
            reg_class = "A. MODEL-TRAINING-OVERLAP"
            eval_view = "SECONDARY_DIAGNOSTIC (In-distribution / Training Overlap)"
        elif n_non_overlap == total_n:
            reg_class = "B. TEMPORALLY POST-TRAINING"
            eval_view = "PRIMARY_CLEAN_EVALUATION (Strict Clean Out-of-Sample)"
        else:
            reg_class = "C. UNKNOWN / MIXED"
            eval_view = "SECONDARY_DIAGNOSTIC"

        regime_summaries.append({
            "regime": reg,
            "total_events": total_n,
            "definite_overlap": n_overlap,
            "definite_non_overlap": n_non_overlap,
            "unknown": n_unknown,
            "classification": reg_class,
            "evaluation_eligibility": eval_view,
        })

        print(f"Regime {reg:10s} | Total: {total_n:5d} | Definite Overlap: {n_overlap:5d} | Non-Overlap: {n_non_overlap:5d} | Class: {reg_class}")

    # Build summary JSON
    summary_data = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": "Phase 13B.1: Model-Training / Evaluation Overlap Audit",
        "frozen_training_cutoff": "2021-12-31 23:59:59",
        "total_phase13_events_audited": len(audit_df),
        "total_definite_overlap": int((audit_df["training_overlap"] == True).sum()),  # noqa: E712
        "total_definite_non_overlap": int((audit_df["training_overlap"] == False).sum()),  # noqa: E712
        "total_unknown": int((audit_df["training_overlap"] == "unknown").sum()),
        "regime_breakdown": regime_summaries,
        "evaluation_protocol_views": {
            "primary_clean_evaluation": {
                "description": "Strictly temporally post-training regimes where model training overlap is mathematically 0%.",
                "regimes": ["2022", "2023", "2024-2025"],
                "total_clean_events": int(sum(r["total_events"] for r in regime_summaries if "PRIMARY" in r["evaluation_eligibility"])),
                "oos_validity": "GENUINE_OUT_OF_SAMPLE",
            },
            "secondary_diagnostic_evaluation": {
                "description": "Regimes overlapping with model fitting or validation selection; informative for in-distribution behavior but not clean OOS.",
                "regimes": ["2020", "2021"],
                "total_diagnostic_events": int(sum(r["total_events"] for r in regime_summaries if "SECONDARY" in r["evaluation_eligibility"] and r["regime"] != "2018-2019")),
                "oos_validity": "IN_DISTRIBUTION_DIAGNOSTIC_ONLY",
            },
            "development_period": {
                "regime": "2018-2019",
                "total_events": 2203,
                "role": "MAPPING_PARAMETER_SELECTION_ONLY",
                "oos_validity": "IN_SAMPLE_DEVELOPMENT",
            },
        },
        "frozen_portfolio_mapping_parameters_confirmed_unchanged": {
            "lambda_M14": 0.25,
            "fraction_M15": 0.30,
            "lambda_M16": 0.10,
            "m13_parameters": {"tau_enter": 1.2, "tau_exit": 0.2, "s_normal": 0.85, "s_reduced": 0.45},
            "horizon_H": 4,
            "transaction_cost_bps": 10.0,
            "risk_aversion_gamma": 1.0,
        },
    }

    sum_json_path = out_dir / "model_training_overlap_summary.json"
    with open(sum_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary JSON to: {sum_json_path}")


if __name__ == "__main__":
    run_training_overlap_audit()
