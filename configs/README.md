# Configurations

`cpu/` contains sample-only configurations. `cuda/` contains complete-dataset
configurations sized for an approximately 20 GiB NVIDIA GPU. Dataset paths,
model widths, batches, augmentation, sampling, metrics, and loss weights are
kept in YAML so runs are reproducible.

Both profiles read generated manifests from `output/manifests/`; raw datasets
remain input-only beneath `data/`.

`cpu_threads: auto` uses all logical CPUs granted to the process.
`num_workers: auto` uses a safe physical-core-aware worker count. The `logging`
section controls how often batch-level resource telemetry is persisted.
