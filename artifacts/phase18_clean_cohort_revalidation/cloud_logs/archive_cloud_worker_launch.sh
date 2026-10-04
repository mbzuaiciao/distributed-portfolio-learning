#!/usr/bin/env bash
# Reviewable commands for the dedicated Phase18 VM only.
set -euo pipefail
mkdir -p "$HOME/phase18-run"
tar -xzf "$HOME/phase18_cloud.tar.gz" -C "$HOME/phase18-run"
cd "$HOME/phase18-run/distributed-portfolio-learning"
# The VM image supplies PyTorch. Use an isolated dependency environment.
/opt/conda/bin/python -m venv --system-site-packages "$HOME/phase18-run/env"
PY="$HOME/phase18-run/env/bin/python"
"$PY" -m pip install 'transformers==5.16.1' pandas scipy scikit-learn matplotlib pytest
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
