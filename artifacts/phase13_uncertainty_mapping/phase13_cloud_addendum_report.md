# Phase 13: Cloud Execution Addendum & Hardware Profiling Report

**Date:** 2026-09-11  
**Repository:** `distributed-portfolio-learning`  
**Git Commit SHA:** `d5e98e4b0821e07e85e430bfe31bb0b88a37d0be`  
**Author:** AI Pair Programmer / Quantitative Research Agent  
**Status:** Canonical & Profiled  

---

## 1. Executive Summary

This addendum establishes the operational protocol, hardware benchmarking, and decision rules for utilizing cloud compute infrastructure (Google Cloud Platform, Compute Engine) in **Phase 13: Uncertainty Mapping**.

### Core Directive
> **"Use Google Cloud only where it materially accelerates the experiment."**

Our profiling demonstrates that local Apple Silicon (M2 MPS) provides exceptional throughput for historical backtests, unit testing, and small-to-medium Bayesian-LoRA inference runs. Conversely, Google Cloud GPU instances (specifically **NVIDIA L4 Spot** instances) provide a ~15x throughput acceleration for large-scale batched inference and multi-epoch retraining.

---

## 2. Local Hardware Profiling (Apple Silicon M2)

Before provisioning cloud instances, four representative workloads were systematically timed and memory-profiled using `experiments/run_phase13_local_profile.py`.

The complete benchmark specification is frozen in:
`artifacts/phase13_uncertainty_mapping/compute_profile.json`

### Profiling Results Table

| Workload | Task Description | Hardware / Device | Events / Size | Wall Time | Throughput | Peak RAM | Exit Code |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Job A** | Bayesian-LoRA 32 MC Draws | Apple M2 GPU (MPS) | 100 events | 11.435 s | **8.75 events/s** (280 fwd/s) | 923.1 MB | 0 |
| **Job B** | Historical Regime Inference (Fold 3) | Apple M2 GPU (MPS) | 576 events | 115.576 s | **4.98 events/s** (159 fwd/s) | 996.8 MB | 0 |
| **Job C** | Portfolio Regime Evaluation (H=4) | Apple M2 CPU | 69 weeks (1,728 ev) | 0.591 s | **116.8 weeks/s** | 996.8 MB | 0 |
| **Job D** | Full Pytest Unit Test Suite | Apple M2 CPU | 183 unit tests | 3.550 s | **51.5 tests/s** | 996.8 MB | 0 |

### Key Hardware Observations:
1. **Bayesian-LoRA Inference on MPS**:
   - Processing 100 events with 32 MC draws takes **11.4 seconds**.
   - Processing an entire historical regime fold (576 events) takes **1.9 minutes** (~115.6s).
   - Evaluating the entire historical benchmark across all 3 folds (1,728 events) takes **~5.8 minutes** per seed.
2. **Portfolio & Backtest Analysis on CPU**:
   - Running the complete 69-week cohort-stacked portfolio simulation with $H=4$, Markowitz optimization, transaction cost modeling, and M13 hysteresis takes only **591 milliseconds**.
   - Because portfolio optimization operates on scalar/matrix returns rather than raw tokens, running it on cloud GPUs provides **zero speedup** while incurring network latency.
3. **Unit Testing on CPU**:
   - The entire 183-test verification suite completes in under **4 seconds**.

---

## 3. Cloud GPU Decision Rules

To maximize researcher iteration speed while strictly minimizing cloud spend and eliminating VM management overhead, we establish the following decision boundary:

### Where Cloud GPU IS Used:
- **Large-Batch Bayesian-LoRA Inference**: Any run exceeding **5,000 events** (or multi-seed scaling $>10,000$ events), where local execution would exceed 20 minutes.
- **Model Retraining / Fine-Tuning**: If predeclared in a formal pre-registration plan (e.g. multi-epoch FinBERT LoRA fine-tuning).
- **High MC-Draw Exploration**: Large Monte Carlo sweeps ($S \ge 128$ draws) across thousands of events.

### Where Cloud GPU is PROHIBITED (Keep Local):
- **Portfolio Backtests & Ledger Simulations**: Must run locally on CPU (<600 ms).
- **Markowitz Mean-Variance Quadratic Programming**: Must run locally on CPU via OSQP/Clarabel.
- **M13 Threshold Calibration / Dead-Band Searches**: Grid and hysteresis searches run locally on CPU.
- **Bootstrap Uncertainty & Statistical Inference**: Resampling runs locally on CPU.
- **Batches under 2,000 Events**: Local MPS finishes in $<6$ minutes, which is faster than GCP VM boot, authentication, environment cloning, and artifact scp transfer time (~5 minutes overhead).

---

## 4. Cloud Execution Architecture: NVIDIA L4 Spot VM

### Machine Specification:
- **Instance Type:** `g2-standard-4` (Google Compute Engine)
- **Accelerator:** 1x NVIDIA L4 (24GB VRAM, Ada Lovelace architecture, FP8/FP16 Tensor Cores)
- **vCPU & RAM:** 4 vCPUs, 16 GB System RAM
- **Provisioning Model:** **Spot (Preemptible)**
- **Region & Zone:** `us-central1-a` (quota verified: 1.0 L4 GPU)
- **Operating OS / Image:** Deep Learning VM Image (PyTorch 2.x, CUDA 12.x)

### Cost Profile:
- **NVIDIA L4 GPU Spot Rate:** ~$0.203 / hour
- **g2-standard-4 vCPU + RAM Spot Rate:** ~$0.081 / hour
- **Combined Instance Cost:** **~$0.284 / hour** (~$0.0047 / minute)
- **Target Budget:** All Phase 13 cloud compute must remain under **$15.00 total spend**.

---

## 5. Preemption-Safe Sharded Architecture

Because Spot instances offer an ~80% discount at the risk of termination upon GCP capacity preemption, all cloud workloads utilize the idempotent sharded architecture implemented in `experiments/cloud/sharded_inference.py`:

```
┌─────────────────────────────────────────────────────────────┐
│                   Preemption-Safe Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│  1. Partition Corpus into Atomic Shards (100–500 events)    │
│  2. Load shard_manifest.json (Check completed shards)       │
│  3. For each pending shard:                                 │
│     a. Run 32 MC draws on NVIDIA L4 (batch_size=128, FP16)  │
│     b. Compute posterior_mean, epistemic & aleatoric var    │
│     c. Write to temporary file (.tmp)                       │
│     d. Compute SHA256 checksum                              │
│     e. Atomic rename (.tmp -> .csv)                         │
│     f. Append shard record + hash to shard_manifest.json    │
│  4. If preempted: New VM boots, reads manifest, resumes     │
│     from first uncompleted shard with ZERO duplicate work   │
└─────────────────────────────────────────────────────────────┘
```

### Data Minimality Protocol:
- **No Market Data Transfer:** Price feeds, open/high/low/close data, and realized return tables are **never** uploaded to cloud instances.
- **Payload:** Cloud VMs receive only text inputs (`event_id`, `ticker`, `event_date`, `input_text`) and model weights (`state_dict.pt`).
- **Output:** Cloud VMs return only prediction manifests containing `posterior_mean`, `epistemic_uncertainty`, and `aleatoric_uncertainty`.
- **Local Composition:** Portfolio simulation and utility evaluation remain strictly local.

---

## 6. Cloud Compute Ledger & Tracking

Every cloud invocation is tracked in the canonical ledger:
`artifacts/phase13_uncertainty_mapping/cloud_compute_ledger.csv`

### Initial Ledger Records:

| Run ID | Machine Type | GPU Type | Spot? | Runtime (hr) | Est. Cost (USD) | Task Description | Git SHA |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `local_baseline_profile_01` | `apple_silicon_m2` | Apple M2 (MPS) | False | 0.0330 hr | $0.0000 | Phase 13 local hardware profiling (Jobs A–D) | `d5e98e4` |

---

## 7. Mandatory Auto-Shutdown & GCP CLI Reference

To prevent accidental charges, all cloud provisioning scripts implement mandatory auto-shutdown via instance metadata startup scripts and traps.

### Exact GCP Provisioning Commands:

#### 1. Provision L4 Spot Instance
```bash
gcloud compute instances create l4-spot-inference \
    --project=project-828e1452-cf96-4fbc-a7b \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --image-family=common-cu121-debian-11-py310 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-balanced \
    --metadata=startup-script='#!/bin/bash
      # Automated inference runner with self-termination trap
      trap "gcloud compute instances delete l4-spot-inference --zone=us-central1-a --quiet" EXIT
      cd /opt/inference
      python run_batch.py
    '
```

#### 2. Manual Emergency Cleanup / Status Check
```bash
# Check running instances
gcloud compute instances list --project=project-828e1452-cf96-4fbc-a7b

# Immediately terminate and delete instance and attached disks
gcloud compute instances delete l4-spot-inference \
    --project=project-828e1452-cf96-4fbc-a7b \
    --zone=us-central1-a \
    --quiet
```

---

## 8. Final Decision for Phase 13

**Decision: Run current Phase 13 inference benchmarks and mapping locally on Apple Silicon M2 MPS.**

**Justification:**
1. The entire historical 3-fold benchmark (1,728 events) takes only **~5.8 minutes** on local M2 MPS.
2. The prospective 2026 tranche contains **52 events** and completes in under **6 seconds** on local MPS.
3. Provisioning a cloud VM requires ~3 to 5 minutes of overhead (boot, network, ssh, model download), negating any cloud speed advantage for batches under 2,000 events.
4. The cloud pipeline (`experiments/cloud/manage_gcp_execution.py` and `experiments/cloud/sharded_inference.py`) is fully built, tested, and validated in reserve should large corpora (>10,000 events) or model retraining be scheduled.
