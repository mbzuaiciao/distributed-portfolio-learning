# Execution readiness — outcomes not evaluated

- Frozen preregistration commit: `ef1b859f`.
- Implementation commit: `d5ad2ee1`; cloud-compatible worker commits: `34d4c5f7`, `d2f9f221`.
- New scientific/provenance tests: 10 passed, 1 pending completed-checkpoint test.
- New plus Phase9/9.1 accounting tests: 22 passed, 1 skipped.
- Full repository suite: 276 passed, 4 skipped in 13.89 seconds.
- Ruff: passed for the new runner and test file. No unrelated changes made.
- Local MPS first-ten-batch preflight projected 1466.14 seconds (24.4 min). It aborted before predictions or portfolio results; see `runtime_preflight.json` and `training.log`.
- Azure CLI/profile not configured. Original GCP project could not provision isolated compute: zone stockout, GPU quota rejection, then global CPU quota 32/32. Other projects' existing instances were not changed.
- Dedicated VM created in otherwise empty accessible project `institutional-risk-fleet-826`: `dpl-phase18-clean-cpu-20261004`, `us-central1-a`, n2-standard-32, Spot, auto-delete after two hours. No training payload transferred. The dedicated VM has been deleted and deletion confirmed to avoid idle charges; see `cloud_execution_status.json`. It can be recreated after authorization.
- The prepared `/private/tmp/phase18_cloud.tar.gz` archive is 492,267,613 bytes. Contents and SHA256 are recorded in `cloud_transfer_manifest.json`. It includes the experiment prerequisites, repository Git history, cached public FinBERT weights, and one sibling helper script. No credential file was intentionally included.
- Automatic approval review rejected SCP because this payload and separate cloud project were not explicitly authorized. Explicit human authorization is pending. No workaround transfer is permitted.
- `cloud_worker_launch.sh` contains reviewable setup and three fixed-seed worker commands. Remote reproduction is a backend change, not a model search; all checkpoint selection remains fixed at epoch two. Remote worker outputs will be hash-checked, and checkpoint tests run locally before portfolio outcomes.
- README has not been updated: signal, portfolio, decomposition and final scientific classification artifacts require the run to complete. No A/B/C/D outcome has been assigned to an unexecuted experiment.
