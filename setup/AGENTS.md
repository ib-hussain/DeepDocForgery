# AGENTS.md — DeepDocForgery

## Ownership and scope

DeepDocForgery is Ibrahim Hussain's GNU-licensed document/image tampering-detection project. This file applies only to that repository. Do not import context from other account users or unrelated physics, teaching, travel, German/Vienna, IoT/electronics, or Montessori work.

The target is a production-grade, explainable, resumable pipeline for image-level forgery classification and spatial localisation. Preserve the existing repository and integrate changes into its architecture rather than building a parallel prototype.

## Canonical architecture

The implemented design is layered:

1. **Layer 1 — Evidence extractors**
   - **A1 DCT/frequency branch:** prefer exact JPEG coefficients when available; provide a pixel-domain DCT fallback; operate on YCbCr information rather than luminance alone; retain JPEG consistency supervision where configured.
   - **A2 degradation estimator:** consume RGB, DCT-derived features, and JPEG metadata; emit regional multi-scale degradation maps; predict JPEG quality/compression evidence, double compression, noise type, and noise strength.
   - **A3 spatial branch:** combine a Vision Transformer with ADN-style spatial/noise evidence.
2. **Layer 2 — B1 regional gate:** learn per-region/per-branch evidence weights.
3. **Layer 3 — C1 attention fusion:** fuse complementary evidence end to end.
4. **Layer 4 — task heads:** D1 denoising decoder, D2 image-level classification, and D3 localisation.

Do not collapse these responsibilities or substitute a different architecture without an explicit decision.

## Data and orientation invariants

- Raw datasets remain outside generated output and outside code-only archives.
- Support both `4032 × 2268` and `2268 × 4032` source orientations as equivalent rotations.
- Correct image/mask orientation together using recorded metadata. Never independently reshape a mask merely to satisfy dimensions.
- Preserve aspect ratio through the configured letterbox/resize path and transform image and mask identically.
- Validate image/mask dimensions, orientation, interpolation method, mask class values, and post-transform alignment.
- Dataset splits must be source/group-disjoint where related captures or document identities exist.
- Preserve established dataset boundaries, including the appropriate Doctamper training/validation versus testing separation.

## Output layout

All generated state belongs beneath `output/`, including:

- manifests;
- processed/prepared artifacts;
- preparation and doctor state;
- HPO studies and trial artifacts;
- checkpoints and final models;
- logs and telemetry;
- evaluation and inference outputs.

Do not place generated artifacts beside raw inputs or source code. Tests must use temporary fixtures.

## Resumability requirements

Preparation, doctor/diagnostics, HPO, training, evaluation, and inference must all be restart-safe.

- Persist a fingerprint of source metadata, relevant configuration, code/schema version, and stage dependencies.
- Use atomic state/checkpoint writes.
- Record completed work at a granularity that avoids expensive repetition.
- Refuse incompatible resume state with a clear explanation and an explicit reset/force route.
- Preserve best and latest checkpoints separately.
- Resume persistent HPO studies rather than recreating them.
- Ensure orientation handling and split logic are part of the fingerprint.

## Environment profiles

- Maintain both CPU and CUDA profiles.
- Detect the installed Python/PyTorch/CUDA environment rather than hard-coding an old snapshot.
- CPU development historically used Python 3.12.7 and a CPU PyTorch build; CUDA training used a dedicated `.venv-cuda` on the remote Linux machine.
- A skipped CUDA-only test on CPU is acceptable only when accurately reported.
- Keep commands compatible with the active Bash/WSL environment and provide one-line invocations.

## Training and validation

- Validate a batch end to end before a long run: tensors, masks, targets, losses, gradients, device, and finite values.
- Log stage progress, dataset/sample counts, split counts, CPU/RAM, GPU/VRAM, and checkpoint/resume decisions.
- Monitor each head and branch so a valid aggregate loss cannot hide a dead component.
- Evaluate classification and localisation separately, and preserve stable sample IDs for error analysis.
- Test exact-JPEG and fallback-DCT paths.
- Include rotated portrait/landscape pairs and image/mask transpose cases in regression tests.
- Run the focused suite, CPU smoke pipeline, resume smoke, and affordable full suite before packaging.

## Release rules

- Deliver code-only archives by default.
- Exclude datasets, `output/`, models, manifests, logs, HPO databases, inference results, caches, virtual environments, and secrets.
- Include source, configs, tests, documentation, licence, and lightweight synthetic fixtures only.
- Derive the archive from the verified current tree and inspect its contents.
- Use clear versioned names and a SHA-256 companion when the release line already uses checksums.

## Documentation

- Explain computer-vision concepts step by step from Ibrahim's baseline: general AI/NLP knowledge, basic CNN/ViT familiarity, limited specialised compression/DCT background.
- Keep architecture, data contracts, output structure, recovery, commands, and known limitations current.
- Historical counts or test results belong in dated status notes and must not be presented as guarantees for the current tree.

## Scientific evidence standard

This project must not claim success from weak proxies or incomplete supervision. Ibrahim explicitly requires agents to distinguish:

- a branch being present from it being connected;
- connected from supervised;
- supervised from receiving meaningful gradients;
- receiving gradients from contributing useful evidence;
- a generic segmentation output from true forgery localisation;
- positive-only manipulation data from a valid authentic-versus-forged classifier;
- a tiny/zero loss from a correct target and non-trivial task;
- random split performance from source/group-safe generalisation;
- approximate pixel DCT from exact JPEG coefficients;
- a CPU/sample smoke run from a CUDA/full-data experiment;
- static/tests passing from target-GPU training completing.

Every major branch and head must have an explicit supervision story, an executed diagnostic, and an ablation/evaluation plan.

## Dataset-role contract

### DocTamper

- Preserve official/declared subset boundaries.
- Historical preparation policy used DocTamper training records only for train/validation.
- Historical `doctamper_testing_fcd_scd` records were test-only.
- Treat positive-only or manipulation-focused subsets as localisation evidence unless authentic negatives and a valid classification protocol exist.
- Do not manufacture authentic labels by assuming a missing mask means authentic without verifying dataset semantics.

### MIDV

- Provides authentic and forged/derived document evidence for classification and localisation/synergy tasks.
- Group derivatives with their authentic source/document identity before splitting.
- Use source-group-disjoint train/validation/test membership.
- Handle EXIF orientation and portrait/landscape resolution consistently for image and mask.
- Preserve document/template/identity grouping in the manifest.

### Synthetic degradation targets

- Synthetic corruption parameters may supervise the degradation estimator when generated through a controlled pipeline.
- Mark synthetic versus observed/proxy targets.
- Do not claim real-world noise/compression estimation solely from synthetic training without evaluation.
- ADN supervision must be explicit or clearly labelled as proxy; an inert auxiliary loss is not acceptable.

## Canonical sample/manifest contract

Each manifest entry should be able to carry:

- stable sample ID;
- dataset/subset/source path;
- source document/group/derivative ID;
- image path and optional mask path;
- authentic/forged/class label where valid;
- localisation availability;
- image width/height and orientation decision;
- mask width/height and transformed alignment status;
- JPEG/native coefficient availability and metadata;
- degradation target availability/provenance;
- split and split reason;
- profile/sample-selection status;
- source and preparation fingerprints.

Reject or quarantine a sample with a precise reason rather than silently rewriting its geometry.

## Orientation and alignment algorithm

For the MIDV phone-photo cases that may appear as `2268 × 4032` or `4032 × 2268`:

1. Decode image and mask dimensions without assuming Pillow/OpenCV order equivalence.
2. Resolve EXIF orientation for the image.
3. Determine whether the mask is stored in pre- or post-orientation geometry using allowed rotations and aspect ratio.
4. Apply the same canonical orientation to both.
5. Use nearest-neighbour interpolation for categorical masks.
6. Letterbox/resize both with the same scale and offsets.
7. Validate exact final spatial match and legal mask values.
8. Record the chosen transform in the manifest/state.

Never fix an image/mask mismatch by independently transposing or stretching the mask without proving the mapping.

## Exact JPEG/DCT branch

- Prefer actual quantised JPEG DCT coefficients and quantisation metadata when the source is a compatible JPEG and the library supports it.
- Validate coefficient block layout, component sampling, quantisation tables, channel/component order, and image boundary padding.
- Process YCbCr evidence, not only luminance, according to the current design.
- The pixel-domain DCT path is a documented fallback, not equivalent ground truth.
- Mixed batches may contain exact-coefficient and fallback samples; carry an availability/source mask so loss/consistency terms are applied correctly.
- Test recompression, non-JPEG inputs, chroma subsampling, odd dimensions, and coefficient-library absence.

## Degradation estimator contract

Inputs may include RGB features, DCT/frequency features, and JPEG metadata. Outputs include regional/multi-scale degradation evidence and configured tasks such as:

- JPEG quality/compression level;
- double-compression evidence;
- noise type;
- noise strength;
- regional degradation maps/features for the gate/fusion path.

Every target must have a provenance flag and applicable loss mask. Missing targets cannot be silently converted to zero. Losses must be reported by task/source to prevent one easy synthetic task hiding another dead task.

## Spatial ViT + ADN contract

- Preserve the confirmed ViT plus ADN design.
- State the pretrained weight identity and input resolution.
- Verify positional embedding/interpolation behaviour when resolution changes.
- Keep colour/spatial augmentation aligned with masks and compatible with forensic evidence; avoid transformations that destroy the signal without an explicit experiment.
- Verify ADN/noise features are connected to fusion and receive their intended supervision/gradients.
- Do not call an arbitrary convolutional noise residual “ADN” without matching the chosen reference/implementation rationale.

## Regional gate and attention fusion

- Gate weights are per-region/per-branch evidence weights, not a decorative scalar.
- Validate normalisation/range and spatial resolution alignment.
- Track gate distributions by source/class so one branch is not permanently ignored.
- Attention fusion must preserve the intended evidence identity long enough for explanations/ablations.
- Test missing/unavailable exact-DCT metadata and branch masks.
- A fallback branch must not look like strong positive evidence simply because a missing-feature vector uses zeros.

## Task heads and losses

### Denoising decoder

Define its target and relationship to forensic features. If it is an auxiliary reconstruction/denoising objective, verify that it does not erase manipulation evidence or dominate the loss.

### Image-level classification

Train/evaluate only on samples with valid class semantics and authentic negatives. Report per-class metrics and source slices.

### Localisation

Use valid masks and appropriate losses/ignore behaviour. Report empty-mask/authentic handling and small-forgery performance separately.

### Multi-task loss

- Log each raw and weighted term.
- Mask unavailable targets.
- Verify gradient reach per branch/head.
- Preserve configurable weights.
- Add a controlled step test demonstrating parameters update.
- Investigate constant zero/tiny terms rather than celebrating the aggregate.

## CPU and CUDA profiles

### CPU profile

- Uses a bounded sample manifest for full pipeline development.
- May use a compact backbone/settings while preserving interfaces.
- Must cover preparation, doctor, model smoke, checkpoint, evaluation/inference, and resume semantics.
- Must not require the full datasets in tests.

### CUDA profile

- Uses full DocTamper + MIDV according to split policy.
- Historical sizing targeted roughly 20 GB usable VRAM / approximately 5,000 CUDA cores.
- Historical planning used batch size 4, mixed precision, and effective batch 16 through accumulation; treat this as a starting snapshot, not a universal current setting.
- Detect actual free VRAM and competing processes.

Keep separate setup/config files such as `configs/cpu/sample.yaml` and `configs/cuda/full.yaml` where the current repository follows that layout.

## Stage-state model

Each expensive stage should report at least:

- `not_started`;
- `running`;
- `completed`;
- `failed` or `attention_required`;
- fingerprint compatibility;
- latest resumable cursor/artifact;
- recovery command.

Preparation state historically lived under `output/state/prepare/<profile>/state.json`; preserve current canonical paths from code/config. Doctor must store its own state and not mark the whole pipeline healthy when a diagnostic crashes.

## Status command expectations

The status surface should summarise:

- profile/config;
- manifest/summary paths and counts;
- stage state/fingerprint;
- preparation progress/reuse;
- doctor checks/errors and recovery command;
- HPO study/trials;
- training latest/best checkpoint/epoch;
- evaluation/inference outputs;
- current attention-required blockers.

Status must inspect real artifacts; it cannot infer completion only from an old log line.

## Progress and resource telemetry

- Use tqdm or a compatible progress abstraction correctly for the execution model.
- Do not call `next()` on a tqdm variant that is not an iterator; update progress explicitly for callback/concurrent flows.
- Report sample/batch/trial/epoch progress, elapsed/ETA where meaningful, CPU/RAM, and CUDA utilisation/VRAM.
- Telemetry failures should not corrupt the training state.

## Pytest/plugin isolation

The environment may contain unrelated ROS pytest plugins. The project should remain testable without ROS. Protect the suite from external plugin auto-loading where appropriate or remove/avoid the offending plugin in the isolated environment. Do not add ROS as a project dependency to satisfy an accidental plugin hook.

## Required regression tests

- `2268 × 4032` and `4032 × 2268` image/mask pairs;
- EXIF-transposed source and mask alignment;
- invalid aspect ratio rejection;
- group-disjoint derivatives;
- official DocTamper subset boundaries;
- positive-only classification guard;
- exact-DCT/fallback mixed batches;
- small-forgery localisation reporting;
- branch/head gradient connectivity;
- target-availability loss masking;
- preparation/doctor/HPO/training/evaluation/inference resume;
- stale fingerprint rejection;
- code-only archive exclusions;
- external pytest plugin isolation.

## Project definition of done

“All tests pass” is necessary but not sufficient. The current requested profile must prepare/doctor/smoke or train as authorised, resume correctly, produce valid classification/localisation outputs, retain scientific evidence boundaries, and package only reproducible code/config/tests/docs.
