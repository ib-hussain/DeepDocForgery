# Data inputs

This directory is input-only. No dataset bytes are distributed with the code.

- CPU inputs: `sample-doctamper/` and `sample-midv/`.
- CUDA inputs: `dataset-doctamper/` and `dataset-midv/`.
- MIDV roots contain `images/`, `masks/`, and optional `annotations/` trees.
- MIDV photographs may be upright `2268x4032`, landscape `4032x2268`, or use
  EXIF rotation to map the stored grid to the displayed orientation. Masks may
  be `1152x2048` (or the matching landscape scale). These are valid aligned
  pairs. Other size differences require the same upright aspect ratio.

Generated manifests go to `output/manifests/`. Restart-safe DocTamper LMDB
exports go to `output/processed/`. Never place generated files in `data/`.

Do not mix official DocTamper test subsets into training. The preparation
command enforces this.
