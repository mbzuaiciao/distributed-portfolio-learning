#!/usr/bin/env bash
# Reviewable commands for the dedicated Phase18 VM only.
set -euo pipefail
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip
mkdir -p "$HOME/phase18-run"
tar -xzf "$HOME/phase18_cloud_minimal.tar.gz" -C "$HOME/phase18-run"
MODEL="$HOME/phase18-run/hf/hub/models--ProsusAI--finbert/snapshots/4556d13015211d73dccd3fdd39d39232506f3e43/pytorch_model.bin"
curl -fL --retry 3 https://huggingface.co/ProsusAI/finbert/resolve/4556d13015211d73dccd3fdd39d39232506f3e43/pytorch_model.bin -o "$MODEL"
printf '%s  %s\n' e15a7b5738df7f17553399b6d94c6e2ff69c89245d066e8e5d183f5803a554e3 "$MODEL" | sha256sum -c -
python3 -m venv "$HOME/phase18-run/clean-env"
"$HOME/phase18-run/clean-env/bin/pip" install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
"$HOME/phase18-run/clean-env/bin/pip" install transformers==5.16.1 pandas scipy scikit-learn matplotlib pytest

cd "$HOME/phase18-run/distributed-portfolio-learning"
# Use a dedicated CPU environment; the image did not provide /opt/conda.

PY="$HOME/phase18-run/clean-env/bin/python"
"$PY" -c 'import torch, transformers; from transformers import AutoModelForSequenceClassification; print(torch.__version__, transformers.__version__)'
export HF_HOME="$HOME/phase18-run/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="src:.:$HOME/phase18-run/sibling-scripts"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
OUT=artifacts/phase18_clean_cohort_revalidation
"$PY" experiments/run_phase18_clean_cohort_revalidation.py --stage train --device cpu --cloud-cpu --seed 42 > "$OUT/seed_42_cloud.log" 2>&1 &
PID42=$!
"$PY" experiments/run_phase18_clean_cohort_revalidation.py --stage train --device cpu --cloud-cpu --seed 123 > "$OUT/seed_123_cloud.log" 2>&1 &
PID123=$!
"$PY" experiments/run_phase18_clean_cohort_revalidation.py --stage train --device cpu --cloud-cpu --seed 777 > "$OUT/seed_777_cloud.log" 2>&1 &
PID777=$!
EXIT42=0; wait "$PID42" || EXIT42=$?
EXIT123=0; wait "$PID123" || EXIT123=$?
EXIT777=0; wait "$PID777" || EXIT777=$?
"$PY" - "$EXIT42" "$EXIT123" "$EXIT777" <<'PY'
import json,sys
from pathlib import Path
out=Path('artifacts/phase18_clean_cohort_revalidation')
statuses=dict(zip([42,123,777],map(int,sys.argv[1:])))
(out/'cloud_worker_status.json').write_text(json.dumps(statuses,indent=2)+'\n')
if any(statuses.values()): raise SystemExit('A frozen seed failed; no outcomes evaluated')
manifest=[]
for seed in statuses:
 manifest.extend(json.loads((out/f'training_manifest_seed_{seed}.json').read_text()))
(out/'training_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
# Only model reproduction runs remotely. Retrieve and verify hashes locally,
# then run the checkpoint tests before evaluating portfolio outcomes.
tar -czf "$HOME/phase18_results.tar.gz" -C "$OUT" .
