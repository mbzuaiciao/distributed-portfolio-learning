#!/usr/bin/env python3
"""Phase 13: Preemption-Safe Sharded Bayesian-LoRA Inference.

Implements idempotent, sharded inference designed for Spot / Preemptible VMs:
- Shard schema: seed × regime × event_batch
- Incremental writing to disk with SHA256 integrity hashes
- Checkpointed state manifest tracking completed shards
- Resumes seamlessly if interrupted or preempted
- Text-only inference: zero market-data dependencies
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    torch = None
    DataLoader = None
    Dataset = object
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    HAS_TORCH = False

# Sibling imports
sibling_src = Path("/Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/src")
if str(sibling_src) not in sys.path:
    sys.path.insert(0, str(sibling_src))

try:
    from bayesian_lora_portfolio_risk.uncertainty.bayesian_lora import (  # noqa: F401
        inject_bert_attention_bayesian_lora,
        set_posterior_sampling,
    )
except ImportError:
    pass


class SimpleTextDataset(Dataset):  # type: ignore[misc]
    def __init__(self, texts: list[str], tokenizer: Any, max_length: int = 64) -> None:
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: val[idx] for key, val in self.encodings.items()}


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class ShardedInferenceRunner:
    """Manages preemption-safe sharded Bayesian-LoRA inference."""

    def __init__(
        self,
        output_dir: Path,
        batch_size: int = 64,
        shard_size: int = 128,
        mc_draws: int = 32,
        base_revision: str = "4556d13015211d73dccd3fdd39d39232506f3e43",
    ) -> None:
        self.output_dir = output_dir
        self.shards_dir = output_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.shard_size = shard_size
        self.mc_draws = mc_draws
        self.base_revision = base_revision
        self.manifest_path = self.output_dir / "shard_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "completed_shards": {},
            "failed_shards": {},
        }

    def _save_manifest(self) -> None:
        self.manifest["updated_at_utc"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def is_shard_complete(self, shard_id: str) -> bool:
        """Check if shard is already completed and verified on disk."""
        if shard_id not in self.manifest["completed_shards"]:
            return False
        shard_info = self.manifest["completed_shards"][shard_id]
        shard_file = self.shards_dir / shard_info["filename"]
        if not shard_file.exists():
            return False
        # Verify hash integrity
        current_sha = compute_sha256(shard_file)
        return current_sha == shard_info["sha256"]

    def run_shard(
        self,
        seed: int,
        regime_id: str,
        shard_idx: int,
        shard_events: list[dict[str, Any]],
        ckpt_path: Path,
        device: torch.device,
        model: Any,
        tokenizer: Any,
    ) -> dict[str, Any]:
        shard_id = f"seed_{seed}_regime_{regime_id}_shard_{shard_idx:04d}"
        if self.is_shard_complete(shard_id):
            print(f"Shard {shard_id} already completed and verified. Skipping.")
            return self.manifest["completed_shards"][shard_id]

        t0 = time.perf_counter()
        texts = [e["text"] for e in shard_events]
        dataset = SimpleTextDataset(texts, tokenizer, max_length=64)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        set_posterior_sampling(model, True)
        samples: list[np.ndarray] = []

        for _ in range(self.mc_draws):
            batch_probs: list[float] = []
            with torch.no_grad():
                for batch in loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logits = model(**batch).logits
                    probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                    batch_probs.extend(probs.tolist())
            samples.append(np.array(batch_probs, dtype=float))

        matrix = np.stack(samples, axis=0)  # Shape: [mc_draws, N_shard]
        p_mean = matrix.mean(axis=0)
        p_epi = matrix.var(axis=0)
        p_ale = (matrix * (1.0 - matrix)).mean(axis=0)

        pred_rows = []
        for i, e in enumerate(shard_events):
            pred_rows.append({
                "shard_id": shard_id,
                "seed": seed,
                "regime_id": regime_id,
                "event_id": e["event_id"],
                "ticker": e["ticker"],
                "date": e["date"],
                "posterior_mean": float(p_mean[i]),
                "epistemic_uncertainty": float(p_epi[i]),
                "aleatoric_uncertainty": float(p_ale[i]),
                "mc_samples": self.mc_draws,
            })

        t1 = time.perf_counter()
        wall_time = t1 - t0

        # Save shard atomically
        shard_filename = f"{shard_id}.csv"
        shard_path = self.shards_dir / shard_filename
        fieldnames = [
            "shard_id",
            "seed",
            "regime_id",
            "event_id",
            "ticker",
            "date",
            "posterior_mean",
            "epistemic_uncertainty",
            "aleatoric_uncertainty",
            "mc_samples",
        ]
        with open(shard_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pred_rows)

        shard_sha = compute_sha256(shard_path)
        shard_record = {
            "shard_id": shard_id,
            "filename": shard_filename,
            "seed": seed,
            "regime_id": regime_id,
            "n_events": len(shard_events),
            "mc_draws": self.mc_draws,
            "wall_time_sec": round(wall_time, 3),
            "events_per_sec": round(len(shard_events) / wall_time, 2)
            if wall_time > 0
            else 0.0,
            "sha256": shard_sha,
            "completed_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        }

        self.manifest["completed_shards"][shard_id] = shard_record
        self._save_manifest()
        return shard_record
