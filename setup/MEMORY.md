# MEMORY.md — DeepDocForgery

## Durable identity and purpose

- Owner: Ibrahim Hussain.
- Project/repository family: `deepFakeDetection` / `DeepDocForgery`.
- Licence: GNU.
- Purpose: explainable document/image tampering detection with image-level classification and spatial localisation.

## Durable architecture

1. A1 exact JPEG DCT with pixel-DCT fallback, YCbCr evidence, and consistency supervision where applicable.
2. A2 degradation estimator using RGB/DCT/JPEG metadata and predicting regional multi-scale degradation, JPEG quality/double compression, and noise evidence.
3. A3 spatial ViT + ADN branch.
4. B1 per-region evidence gate.
5. C1 attention fusion.
6. D1 denoising decoder.
7. D2 image-level classification.
8. D3 localisation.

## Durable data/output decisions

- CPU profile uses samples for development/smoke.
- CUDA profile uses full DocTamper + MIDV under the declared boundaries.
- Raw inputs remain outside generated output.
- All generated manifests, prepared artifacts, stage state, logs, HPO, models, evaluation, and inference live beneath `output/`.
- Releases are code-only and exclude datasets/output/generated artifacts.
- MIDV must accept both `2268 × 4032` and `4032 × 2268` orientations as valid rotations when image and mask can be aligned safely.
- Splits are source/group-disjoint where derivatives share an original.
- Preparation, doctor, HPO, training, evaluation, and inference are resumable/fingerprinted.

## Evidence rules remembered from Ibrahim

- Do not claim success from positive-only FCD data.
- Do not accept leaky splits.
- Do not treat zero losses as automatically good.
- Do not leave inert branches and call the architecture implemented.
- Do not call generic segmentation true forgery localisation without valid masks/evaluation.
- Report macro/micro/per-class/source/small-forgery evidence appropriately.
- Distinguish exact JPEG DCT from approximation.
- Record augmentation, resolution, pretrained weights, authentic negatives, and ablations.

## Historical environments

- Development used Arch WSL/Python 3.12.x with a CPU PyTorch environment.
- Remote/full runs used Linux path `/home/fastlab/deepFakeDetection/DeepDocForgery` and `.venv-cuda`.
- The current environment must be detected; historical PyTorch versions across conversations were not fully consistent.

## Historical preparation snapshots

### CPU

- Command family: `python -m deepdocforgery prepare --profile cpu`, then doctor/smoke.
- Recorded sample count: 282.
- Recorded split counts: train 164, validation 55, test 63.
- Manifest path under `output/manifests/cpu.jsonl`.

### CUDA / v0.5.2-era successful preparation

- Total samples: 178,999.
- Train: 108,382.
- Validation: 19,105.
- Test: 51,512.
- DocTamper records reused: 170,000.
- MIDV processed: 8,999.
- MIDV EXIF-transposed: 499.
- `doctamper_testing_fcd_scd`: test only.
- `doctamper_training`: train/validation only.
- MIDV: source-group-disjoint train/validation/test.
- Historical state path: `output/state/prepare/cuda/state.json`.
- Historical summary path: `output/manifests/cuda.summary.json`.

These are artifact-specific snapshots, not hard-coded expected counts.

## Historical failures and fixes

### MIDV orientation

- `10.jpg`/`10.png` style case raised `Image/mask size mismatch`.
- `11.jpg` image `(4032, 2268)` and mask `(2268, 4032)` raised an aspect-ratio mismatch.
- Later versions added EXIF/portrait/landscape tolerance and shared scale-aligned image/mask handling.

### Doctor progress

- A combined prepare/doctor/train chain failed in `doctor.py` because `next(progress)` was used on a `tqdm_asyncio` object.
- v0.5.3 corrected progress handling through explicit update logic.

### Pytest environment

- An unrelated ROS pytest entry point caused `PluginValidationError` for unknown hook `pytest_launch_collect_makemodule`.
- v0.5.3 isolated/fixed the test environment rather than making ROS part of the project.

## Version/test timeline snapshots

- v0.4.1: 59 passed, 2 skipped.
- v0.4.2-era refactor was associated with 61 passed, 2 skipped in prior reporting.
- v0.5.3: 87 passed, 3 skipped in the recorded release validation.

Re-run tests for the current tree. Do not reuse these pass totals as current proof.

## Release history snapshot

- v0.5.1 code-only SHA-256 recorded: `1885413778168be94e42a06ed88a3c9bf02b51a2fac70abca7fc92d7bd0c9126`.
- v0.5.2 addressed DocTamper cross-split proxy leakage and preparation reuse/journaling.
- v0.5.3 addressed ROS pytest plugin isolation and doctor progress failure.

## Start-of-session verification

Recheck current version/commit, profile configs, output contract, manifest counts/fingerprint, split boundaries, orientation implementation, stage states, latest/best checkpoints, HPO study, test result, and archive exclusions before modifying the project.
