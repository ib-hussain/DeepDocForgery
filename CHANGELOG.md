# Changelog

## 0.5.2

- Fixed full DocTamper preparation failing after successful conversion when the
  coarse masked perceptual proxy collides across TrainingSet and official test
  benchmarks. Official TestingSet/FCD/SCD membership now takes precedence;
  cross-benchmark proxy collisions are emitted as diagnostics instead of fatal
  leakage errors.
- Kept strict source-group disjointness for MIDV and for DocTamper TrainingSet
  train/validation splitting.
- Kept v0.5.1 preparation journal contracts compatible so the already prepared
  170,000 DocTamper records and completed MIDV records can be reused.
- Added bounded pip retries to CPU/CUDA setup for transient package-index
  failures without weakening the active-environment reset guard.
- Fixed CLI dispatch so `<command> --help` shows the selected command's
  arguments instead of the top-level help screen.

## 0.5.1

- Added contract-checked, atomic recovery state for setup, preparation,
  training, doctor, HPO, evaluation/testing, and inference.
- Added append-only preparation and inference journals with torn-tail repair.
- Made training auto-resume the latest epoch and restore model, optimiser,
  scheduler, AMP scaler, RNG, and deterministic loader-generator state.
- Made HPO reuse successful trials and resume each interrupted child trainer.
- Added resumable evaluation metric accumulators and completed-report reuse.
- Made completed inference runs return from their validated report without
  reconstructing the model, and made completed external training checkpoints
  self-contained in the selected output directory.
- Extended `status` to report stage progress and exact recovery commands.
- Isolated pytest from ambient ROS/user-site plugin discovery during setup.
- Added EXIF-aware MIDV image/mask orientation handling for both `4032x2268`
  and `2268x4032` geometries; exact DCT safely falls back after rotation.
- Added regression tests for state contracts, journal recovery, metric-state
  restoration, landscape MIDV pairs, and EXIF-rotated MIDV JPEGs.
- Made doctor verify every manifest image, mask, and explicit ADN target with
  bounded parallel filesystem checks before training.

## 0.5.0

- Added one shared observability layer for every executable command.
- Added `tqdm` progress for preparation, doctor checks, smoke, training,
  validation/testing, HPO, and inference.
- Added timestamped human `.log` and structured `.jsonl` events plus
  per-command `latest.json` status reports.
- Added live process/system RAM telemetry and CUDA allocated, reserved, peak,
  free, used, and total VRAM telemetry.
- Added affinity-aware automatic CPU thread selection and physical-core-aware
  preparation/DataLoader workers with oversubscription protection.
- Added model-run `status.json`, failure/interruption recording, concise epoch
  summaries, duration/throughput metrics, class-imbalance warnings, and atomic
  status writes.
- Isolated each HPO invocation beneath its own run directory and prevented
  fresh training from silently appending to an existing experiment.
- Kept HPO stdout machine-readable by saving each child trainer's JSON result
  inside its trial while leaving live progress on stderr.
- Added first-class `test` and `status` CLI commands.
- Added telemetry and lifecycle regression tests.
- Accepted MIDV high-resolution-image/lower-resolution-mask pairs only when
  their aspect ratios are scale-aligned; alignment happens before shared
  augmentation and the count is reported during preparation.
- Added GPU utilisation, temperature, power and competing-process telemetry;
  the CUDA doctor now blocks a busy or low-free-VRAM training launch.
- Setup now logs full transcripts, cleans unmarked legacy virtual environments,
  rejects unsafe active-environment clearing, checks dependency consistency,
  and validates the test suite.
- Evaluation throughput now records the number of batches actually processed,
  and doctor verifies required splits, both datasets, and localization labels.

## 0.4.2

- Made protocol tests completely independent of user-managed dataset folders.
- Added explicit coverage for native `2268x4032` MIDV image/mask geometry.
- Moved generated manifests and DocTamper exports from `data/` into `output/`.
- Added persistent preparation, doctor, smoke, and default evaluation reports.
- Removed all dataset bytes from the distributable release.

## 0.4.1

- Made the exact-DCT exception test assert the alignment invariant rather than
  incidental exception wording.
- Made the supplied MIDV trio test coexist with larger local CPU samples.
- Removed a test-only tensor-to-scalar autograd warning.

## 0.4.0

- Consolidated all Python code under `deepdocforgery/` and one CLI.
- Added combined DocTamper + MIDV-DM preparation with source-group safety.
- Added separate CPU sample and approximately 20 GiB CUDA full profiles.
- Corrected DocTamper official split and positive-only classification handling.
- Added exact-DCT/fallback mixed batches and JPEG/noise augmentation targets.
- Added a pretrained timm backbone option and stride-2 decoder detail path.
- Added tamper-aware crops, positive-pixel weighting, masked perceptual
  DocTamper grouping, and pretrained-backbone input normalisation.
- Added branch supervision counts, initial/final classifier losses, image FPR,
  scheduler-resume safety, and best-epoch HPO selection.
- Made ADN supervision explicit; missing labels no longer alias tamper masks.
- Added per-image, failure-tail, area-bucket, baseline, instance, and
  per-benchmark evaluation.
- Added diagnostics, HPO, setup scripts, folder documentation, and regression
  tests based on the supplied MIDV image/mask/JSON contract.

## 0.3.1 — DocTamper benchmark-split and supervision correction

- Preserved TestingSet, FCD, and SCD as separate test-only benchmarks.
- Restricted optimization to TrainingSet with a deterministic validation holdout.
- Added per-record `classification_supervised` masking for positive-only datasets.
- Excluded unsupervised image labels from classification loss, agreement, and image metrics.
- Added resumable/reusable preparation for the complete four-subset release.
- Added command-line manifest overrides and a CPU-only DocTamper sanity configuration.
- Added regression and miniature-LMDB end-to-end validation for the corrected policies.

## 0.3.0 — complete end-to-end research pipeline

- Replaced the temporary fusion probe with a clean-room synergy denoising decoder.
- Added image classification, pixel localization, boundaries, confidence, and instances.
- Added bidirectional trainable classification/localization conditioning.
- Added the complete joint objective and fixed-threshold research metrics.
- Added JSONL datasets, letterboxing, source-group leakage validation, and manifest merging.
- Added a 24-image generated demo dataset and reproducible generator.
- Added generic folder and authorized DocTamper LMDB preparation.
- Added training, atomic checkpoints, resume, evaluation, and inference commands.
- Added research/demo configurations and real-data/reproducibility guides.

## 0.2.0 — spatial features and attention fusion

- Added hierarchical windowed ViT features.
- Added ConvNeXt-style Artifact Decouple Network (ADN) features and auxiliary mask.
- Added ADN supervision objectives and aligned internal/external patch shuffling.
- Added learned ViT/ADN channel attention.
- Added trainable regional attention over spatial, DCT, and degradation evidence.
- Extended configs, smoke training, documentation, attribution, and tests.

## 0.1.0 — DCT and degradation estimator

- Added exact JPEG coefficients with differentiable pixel-DCT fallback.
- Added multi-scale dense DCT features and consistency loss.
- Added regional JPEG/noise degradation estimation, tests, and CPU/CUDA smoke paths.
