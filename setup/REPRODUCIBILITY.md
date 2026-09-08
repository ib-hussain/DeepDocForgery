# Reproducibility checklist

- Archive the resolved YAML, manifest, manifest SHA-256, code version, seed,
  Python, PyTorch, CUDA, driver, and GPU details.
- Keep source groups disjoint and preserve TestingSet/FCD/SCD as test-only.
- Select thresholds and checkpoints on validation data only.
- Report at least three seeds for scientific comparisons.
- Report macro and micro localization, failure-tail and area-bucket metrics,
  image false positives/AUROC, instances, and every benchmark separately.
- Record exact-DCT usage and all JPEG/noise augmentation settings.
- Do not report classification metrics for DocTamper-only samples.
- Run spatial-only, frequency-only, degradation-only, and combined ablations
  under the same split and budget.
- Keep all generated run evidence beneath `output/`; keep `data/` input-only.
- Preserve the command `.log`, event `.jsonl`, `latest.json`, model
  `status.json`, epoch `metrics.jsonl`, and matching `output/state/` files for
  every reported run.
- Keep resume fingerprints and record journals with the exact input data they
  describe; never edit a journal manually or transplant it across datasets.
- Inspect `supervision_draws`, `main/image_initial`, and `main/image_final` in
  every epoch log before claiming that an architectural branch was exercised.

Automatic CPU sizing respects process affinity and is recorded in every run.
For strict comparisons, either retain `auto` on equivalent hardware or pin
explicit `cpu_threads` and `num_workers` values in the archived YAML.

`training.deterministic: true` enables deterministic PyTorch algorithms where
available, but exact equality across hardware and releases is not guaranteed.
