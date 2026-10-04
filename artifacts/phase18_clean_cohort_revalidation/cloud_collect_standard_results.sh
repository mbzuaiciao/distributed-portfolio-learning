#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/phase18-run/distributed-portfolio-learning"
OUT="$ROOT/artifacts/phase18_clean_cohort_revalidation"
test -f "$OUT/training_manifest.json"
mkdir -p "$HOME/phase18-delivery/results" "$HOME/phase18-delivery/logs"
cp -a "$OUT/." "$HOME/phase18-delivery/results/"
for f in phase18_standard_master.log phase18_standard_environment_setup.log phase18_standard_launch.sh phase18_collect_results.sh; do
  test ! -f "$HOME/$f" || cp "$HOME/$f" "$HOME/phase18-delivery/logs/"
done
"$HOME/phase18-run/clean-env/bin/pip" freeze > "$HOME/phase18-delivery/logs/environment_freeze.txt"
"$HOME/phase18-run/clean-env/bin/python" - "$HOME/phase18-delivery" <<'PY'
import hashlib,json,sys,datetime
from pathlib import Path
root=Path(sys.argv[1]); files=[]
for p in sorted(root.rglob('*')):
 if p.is_file(): files.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(root/'result_file_manifest.json').write_text(json.dumps({'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files},indent=2)+'\n')
PY
tar -czf "$HOME/phase18_delivery.tar.gz" -C "$HOME/phase18-delivery" .
sha256sum "$HOME/phase18_delivery.tar.gz"
stat -c '%s' "$HOME/phase18_delivery.tar.gz"
date -u +%Y-%m-%dT%H:%M:%SZ
