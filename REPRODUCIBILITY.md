# Reproducibility checklist

- Save the resolved YAML and exact git commit with every run.
- Record Python, PyTorch, CUDA, driver, GPU, and operating-system versions.
- Retain the generated JSONL manifest and dataset checksums.
- Verify that source groups do not cross splits.
- Select checkpoints and thresholds on validation only.
- Run at least three seeds for reported comparisons.
- Report image-, pixel-, and instance-level metrics separately.
- Report confidence intervals or variation across seeds.
- Disclose resize, crop, JPEG, noise, and exact-DCT policies.
- Run external, cross-generator, and post-processing robustness tests.

`training.deterministic: true` enables deterministic-algorithm warnings and
fixed DataLoader seeding. Exact equality across different hardware, PyTorch
releases, or CPU/CUDA backends is not guaranteed; archive the environment.
