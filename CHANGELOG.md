# Changelog

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
