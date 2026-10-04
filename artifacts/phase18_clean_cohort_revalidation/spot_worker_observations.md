# Interrupted Spot attempt: retained observations

The first launcher started 2026-10-04 22:01:15 UTC; all seeds failed before training when transformers imported the image-provided incompatible torchaudio library. An isolated torch 2.9.1 CPU / transformers 5.16.1 environment passed imports and restarted the exact frozen seeds at 22:09:43 UTC.

Observed at 22:10:08: all three seed processes running. FinBERT classifier mismatch warnings were expected because the public three-class checkpoint is reinitialized for the frozen two-class target. At 22:10:43: seed42 first-ten-batch time 12.882687954 seconds, projected sequential three-seed runtime 2988.60478965 seconds. At 22:12:28: all three seeds epoch1 batch100/375, memory 7.9 GiB used of125 GiB. At 22:15:38: all three epoch1 batch200/375. At22:18:38: all three epoch1 batch300/375. No completed checkpoints, predictions, or portfolio outcomes were observed or retrieved.

Cloud Logging confirms `compute.instances.preempted`, message `Instance was preempted.`, at22:19:32.620580 UTC (followed by22:19:42.202391). The configured termination action DELETE removed the boot disk and its worker/dependency/master logs. Thus those full logs could not be retrieved. This file records retained terminal observations; it is not a substitute for complete worker logs. Available lifecycle and boot logs are preserved verbatim in `cloud_interrupted_instance_audit.json`. The interruption is recorded for costs and reproducibility; no outcomes were used to change the experiment.

The run is restarted from scratch for all three fixed seeds on a bounded standard VM. No model-search choice, threshold, primary endpoint or frozen matrix changes.
