# Phase 18 cloud execution record (in progress)

The user explicitly authorized the minimized frozen-experiment payload and destination `institutional-risk-fleet-826`. The original automatic transfer rejection was resolved by this authorization. The subsequent interruption was an automatic-review usage limit; credits were replenished and work resumed.

Preregistration commit `ef1b859f7f56cacc8be9e70e60e032da8cecd170` and SHA256 `82c41e671da2af23b4aa94bb5d42703fef4fe4822a403eae16c776399b178340` remain unchanged. Scientific execution source is commit `d8d9255e44cb60c7354949534517e217f3d3ebb4`.

The authorized archive has 410,235,376 bytes and SHA256 `cff97c288e4c24636ba50d2dd7d2da9e7933b695cfa42f7e1c384b3dfad39698`. Exact member paths, individual sizes and hashes are in `authorized_cloud_transfer_manifest.json`; local and remote checksums match in `cloud_preexecution_verification.json`. The minimized archive contains only experiment prerequisites and minimal Git provenance. No credentials or unrelated local files were transferred.

The dedicated VM `dpl-phase18-clean-cpu-20261004`, `us-central1-a`, is n2-standard-32 (32 vCPU, 128 GiB), Spot, with a 100 GB balanced persistent boot disk and two-hour maximum lifetime with automatic deletion. Full metadata is recorded in `cloud_vm_record.json`. Other project VMs were not modified. Azure CLI/profile was unavailable; earlier GCP provisioning attempts encountered stockout/quota constraints.

Initial launch at 2026-10-04 22:01:15 UTC failed in all seeds before training at the preinstalled incompatible torchaudio shared library. Failed logs were preserved. A fresh isolated environment uses torch 2.9.1 CPU and transformers 5.16.1; no scientific code, threshold, model-selection rule or matrix was changed. The repaired launcher SHA256 is `580c531c81b8b3bae45a0b50c5bb01cfdcf246147025a37f79f4c805117c9456`. It restarted the frozen seeds 42/123/777 at 22:09:43 UTC. First ten batches projected approximately 17 minutes in parallel (the runner estimate of 2,988.6 seconds assumes three sequential seeds).

Before cloud execution: 10 new tests passed and one checkpoint test awaited completed weights; combined accounting checks 22 passed/one skipped; full repository 276 passed/four skipped in 13.89 seconds; Ruff passed. Local MPS preflight projected 24.4 minutes and aborted before predictions/outcomes. Final checkpoint validation, evaluation, cost and cleanup records will be appended after retrieval.

## Spot interruption and standard replacement

Cloud Logging confirms Spot preemption at 2026-10-04 22:19:32 UTC. The automatic DELETE action removed the experiment boot disk. No seed had completed and no portfolio outcomes were evaluated. Last retained worker observation: all seeds epoch1 batch300/375 at22:18:38. Full worker logs on that disk were lost; available lifecycle/boot logs are in `cloud_interrupted_instance_audit.json`, with retained observations in `spot_worker_observations.md`.

The unchanged experiment was restarted on `dpl-phase18-clean-standard-20261004` in the same authorized project/zone, n2-standard-32 STANDARD, 100 GB balanced boot disk, two-hour auto-delete. Metadata is in `cloud_standard_vm_record.json`. The large local upload was superseded to reduce idle time: `standard_cloud_transfer_manifest.json` inventories a 4,791,053-byte archive with SHA256 `42d29ab1f8904086ec05abed16675d3547387039525e6b10e526f4109bc423a9`. All private research files and minimal provenance are identical to the authorized payload; the omitted public FinBERT base weight is fetched directly from its exact frozen revision. Its 437,992,753 bytes have the identical frozen SHA256 `e15a7b5738df7f17553399b6d94c6e2ff69c89245d066e8e5d183f5803a554e3`, verified before workers start. Transfer and supplementary-script hashes match, recorded before execution in `cloud_standard_preexecution_verification.json`.

Standard launch started22:29:31 UTC, all three workers were running by22:30:03. The frozen public-model hash gate passed. Official catalog standard compute is $1.553888/hour; Spot is $0.932096/hour. Costs include all attempts. No experiment source, criterion, endpoint or matrix was altered. Scientific recipe still runs fixed two epochs for42/123/777.

## Finalized outcome and retention

All fixed seeds42/123/777 completed on the standard VM. The 6,815,395-byte result archive and all25 individual result/log hashes matched; checkpoint tensor/provenance/event invariants and accounting checks passed24 tests. Full suite:278 passed,3 skipped in14.56 seconds. Ruff passed. Frozen local evaluation completed successfully in101.87 seconds.

Frozen **Outcome B — architecture only**: residual weeks2–4 IC0.006027, permutation p0.105; primary utility delta+0.00138041145/week with95% CI[-0.005468,+0.008433]. Mechanical gate passes, persistence/economic gates fail. No threshold or outcome category was redefined. README was updated only after outcomes finalized.

Parallel training/inference longest seed:18.31 min. Total conservative usage-cost estimate:$1.2933 USD across all attempts. Invoice/export/credits/taxes unavailable; see `compute_cost_report.md`. All temporary Phase18 instances/disks are absent and all cloud-resident payloads removed. Local models, predictions, reports, manifests and available logs retained. Original Spot worker logs lost to preemption are explicitly documented. Other-project VMs were untouched.
