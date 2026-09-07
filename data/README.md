# Data inputs

This directory is input-only. No dataset bytes are distributed with the code.

- CPU inputs: `sample-doctamper/` and `sample-midv/`.
- CUDA inputs: `dataset-doctamper/` and `dataset-midv/`.
- MIDV roots contain `images/`, `masks/`, and optional `annotations/` trees.
- MIDV photographs may be `2268x4032` while masks are `1152x2048`. This is a
  valid scale-aligned pair. Other size differences are accepted only when the
  aspect ratio agrees within the strict preparation tolerance.

Generated manifests go to `output/manifests/`. Restart-safe DocTamper LMDB
exports go to `output/processed/`. Never place generated files in `data/`.

Do not mix official DocTamper test subsets into training. The preparation
command enforces this.
