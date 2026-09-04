# Data inputs

This directory is input-only. No dataset bytes are distributed with the code.

- CPU inputs: `sample-doctamper/` and `sample-midv/`.
- CUDA inputs: `dataset-doctamper/` and `dataset-midv/`.
- MIDV roots contain `images/`, `masks/`, and optional `annotations/` trees.

Generated manifests go to `output/manifests/`. Restart-safe DocTamper LMDB
exports go to `output/processed/`. Never place generated files in `data/`.

Do not mix official DocTamper test subsets into training. The preparation
command enforces this.
