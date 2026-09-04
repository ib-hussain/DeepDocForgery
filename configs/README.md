# Configurations

`cpu/` contains sample-only configurations. `cuda/` contains complete-dataset
configurations sized for an approximately 20 GiB NVIDIA GPU. Dataset paths,
model widths, batches, augmentation, sampling, metrics, and loss weights are
kept in YAML so runs are reproducible.

Both profiles read generated manifests from `output/manifests/`; raw datasets
remain input-only beneath `data/`.
