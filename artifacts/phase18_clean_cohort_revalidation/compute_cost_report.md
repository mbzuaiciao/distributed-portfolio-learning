# Phase 18 compute cost and runtime

Estimated total before credits/taxes: **$1.2933 USD**. This is a conservative list-price usage estimate, not an invoiced final charge; no billing export is available. All three temporary VM lifetimes are included, including the idle authorization attempt and the preempted attempt.

| VM | Provisioning | Minutes | Compute | Disk | IPv4 | Total USD |
|---|---|---:|---:|---:|---:|---:|
| dpl-phase18-clean-cpu-20261004 / 1452975580859299532 | SPOT | 4.815 | 0.07480 | 0.00108 | 0.00020 | 0.07608 |
| dpl-phase18-clean-cpu-20261004 / 1646823689904157775 | SPOT | 27.166 | 0.42202 | 0.00609 | 0.00113 | 0.42924 |
| dpl-phase18-clean-standard-20261004 / 1481816380741218143 | STANDARD | 30.041 | 0.77801 | 0.00673 | 0.00250 | 0.78724 |

Result archive: 6,815,395 bytes; North America payload egress upper estimate $0.00076, potentially free under the account allowance. Ingress is free. Small SSH/log protocol traffic is not metered here. IPv4/egress prices are from [Google VPC pricing](https://cloud.google.com/vpc/network-pricing); N2/disk SKU rates and effective times are retained in the official catalog JSON records. Disk conversion uses the catalog's 744-hour month.

The completed parallel run's longest seed training plus inference took **18.31 minutes**. Sum over three workers: 53.69 CPU-worker minutes (8 threads each). Local frozen evaluation took 1.70 minutes. Lifecycle windows include setup, interruption, and retrieval overhead, and are not interchangeable with training runtime. Full timing details are in `training_manifest.json` and `compute_cost_record.json`.

All temporary experiment VMs/boot disks and cloud-resident payloads were deleted or confirmed absent; see `cloud_cleanup_verification.json`. No other-project resources were changed.
