---
name: deepdocforgery-engineering
description: Implement, diagnose, resume, validate, and release Ibrahim Hussain's multi-branch document forgery detection pipeline.
---

# DeepDocForgery Technical Workflow

## 1. Inspect and preserve

1. Read project instructions, architecture/configs, entry points, dataset adapters, output manager, state/checkpoint code, and tests.
2. Inspect Git status/diff and the latest user logs or failure file.
3. Detect CPU/CUDA profile, Python, PyTorch, available memory, dataset roots, and output root.
4. Preserve existing work and identify the smallest failing stage.

## 2. Validate source data without mutating it

1. Sample each dataset and record source ID, group/document ID, image size/orientation, mask size/orientation, label, and split eligibility.
2. Detect `4032 × 2268` versus `2268 × 4032` orientation pairs.
3. Resolve EXIF/metadata rotation once and apply the same transform to image and mask.
4. Verify mask values and alignment after letterboxing/resizing.
5. Audit source/group-disjoint split rules and cross-split duplicates.

## 3. Build or resume preparation

1. Create the stage fingerprint from source inventory, relevant config, orientation policy, split policy, and schema version.
2. Load state only if the fingerprint and expected artifacts match.
3. Process in restartable units and atomically commit state.
4. Write the manifest beneath `output/manifests/` with stable IDs and traceable paths.
5. Validate manifest counts, split counts, duplicate groups, paths, shapes, and mask presence before training.

## 4. Run the doctor stage

Check every branch dependency and contract:

- JPEG coefficient availability and pixel-DCT fallback.
- YCbCr channel order and block geometry.
- RGB/DCT/JPEG metadata shapes.
- Degradation targets and multi-scale map alignment.
- ViT/ADN feature sizes.
- Gate/fusion dimensional compatibility.
- Decoder/classifier/localiser output geometry.
- Loss inputs, class mapping, finite values, and device consistency.

Persist diagnostic state and allow restart without repeating passed checks unless dependencies changed.

## 5. Validate modules independently

1. Unit-test exact coefficient and fallback DCT paths.
2. Test JPEG consistency and degradation targets with synthetic controlled inputs.
3. Test portrait/landscape and transposed-mask regression fixtures.
4. Test spatial branch, gate, fusion, and all heads for forward shape and gradient flow.
5. Test config errors, missing metadata, corrupt files, and incompatible resume state.

## 6. Tune and train

1. Run a single-batch forward/backward smoke on the selected device.
2. Resume the persistent compatible HPO study or create one with an explicit new identity.
3. Save latest/best checkpoints with model, optimiser, scheduler, mixed-precision scaler, epoch/step, metrics, configuration, and fingerprint.
4. Log per-branch/head losses and metrics, progress, CPU/RAM, and GPU/VRAM.
5. Stop cleanly on NaN/Inf, OOM, bad mask geometry, or checkpoint corruption and retain a recoverable state.
6. Prove resume with a short interrupted run.

## 7. Evaluate and infer

1. Load training-fitted transforms and the intended checkpoint.
2. Preserve stable sample IDs through classification and localisation outputs.
3. Report image-level and localisation metrics separately, including dataset/source slices.
4. Inspect qualitative overlays for alignment, not just numerical scores.
5. Save results beneath the configured `output/` subtree with the run fingerprint.

## 8. Final validation and release

1. Run syntax/static checks.
2. Run focused regression tests, CPU smoke, resume smoke, and full affordable suite.
3. Review `git diff`, `git status`, configs, docs, and archive exclusions.
4. Build a versioned code-only ZIP from the verified tree.
5. Inspect ZIP contents and generate SHA-256 when required.
6. Report actual pass/skip counts, hardware paths not exercised, and the exact delivered version.

## Expanded stage workflows

### Preparation state schema

Persist enough information to resume and audit:

```text
schema_version
profile
stage_status
source_inventory_fingerprint
configuration_fingerprint
split_policy_version
orientation_policy_version
completed_source/subset/cursor
accepted/rejected/reused counts
output manifest and summary identities
last_error and recovery command
```

Write after bounded work units. Validate the last committed unit and referenced output before resuming.

### DocTamper preparation

1. Inventory official training and testing/FCD/SCD subsets.
2. Derive stable sample/source IDs.
3. Apply the declared boundary: training data to train/validation; official testing subsets to test only.
4. Validate masks and localisation availability.
5. Mark classification eligibility separately from localisation eligibility.
6. Group derivatives/related records where needed.
7. Reuse already prepared compatible records by fingerprint rather than reprocessing them.
8. Record accepted/rejected counts by subset and reason.

### MIDV preparation

1. Inventory authentic sources and every derivative/forgery.
2. Group by original source/document identity before splitting.
3. Resolve EXIF and dimension orientation for image and mask.
4. Validate `2268 × 4032`/`4032 × 2268` equivalence only through a valid 90-degree mapping.
5. Apply shared letterbox/resize transform.
6. Assign source-group-disjoint split.
7. Record authentic/forged/localisation tasks and transform metadata.
8. Assert no source group appears in multiple splits.

### Doctor checks

Organise doctor output by category:

- environment/device/dependencies;
- raw roots and permissions;
- manifest/schema/fingerprint;
- split leakage;
- image/mask decodability and alignment;
- exact JPEG coefficient path;
- degradation target availability;
- model config/module construction;
- one-batch forward/loss/backward;
- checkpoint write/read;
- output/state writeability;
- resource budget.

Each check records pass/warn/fail, evidence, and recovery. A doctor implementation error is `attention_required`, not a clean bill of health.

### Branch-connectivity probe

1. Create/load a batch that activates all available tasks.
2. Register or inspect gradients for DCT, degradation, spatial/ADN, gate, fusion, decoder, classification, and localisation parameters.
3. Run forward and individual loss terms.
4. Backpropagate the configured total.
5. Assert intended trainable branches receive finite non-zero gradients.
6. Run one optimiser step and compare parameter changes.
7. Report legitimately frozen/unavailable branches separately.

### HPO workflow

1. Verify prepared manifest/fingerprint and doctor pass.
2. Resolve persistent study DB/name and search-space version.
3. Resume compatible incomplete trials.
4. Use validation metrics appropriate to the selected tasks and available labels.
5. Store trial config, seed, resource use, checkpoints, and failure/prune reason.
6. Do not score positive-only classification data as a normal binary objective.
7. Select a configuration through declared multi-task priorities.
8. Preserve study state under `output/`.

### Training workflow

1. Inspect target GPU and free VRAM.
2. Validate config, manifest, and selected HPO params.
3. Run one-batch probe.
4. Start or resume latest compatible checkpoint.
5. Log raw/weighted losses and metrics per head/source.
6. Save latest atomically and best by explicit metric policy.
7. Track resource telemetry.
8. Stop safely on NaN/Inf/OOM/corruption, recording recovery state.
9. Validate after each configured interval.
10. Prove resume on a bounded run before committing a multi-day job.

### Evaluation workflow

1. Freeze checkpoint and preprocessing identities.
2. Evaluate source/group-safe test partitions.
3. Classification: confusion matrix and per-class/macro metrics on eligible authentic/forged samples.
4. Localisation: pixel/region metrics, authentic empty-mask behaviour, and small-forgery slices.
5. Report by DocTamper/MIDV/subset/orientation/size/source.
6. Run ablations for evidence branches/gate/fusion where required.
7. Save predictions, stable IDs, metrics, config, and fingerprint under `output/`.

### Inference workflow

1. Decode and canonicalise orientation.
2. Determine exact-DCT availability and construct fallback mask.
3. Apply training-parity transforms.
4. Run all connected branches, gate/fusion, classifier, and localiser.
5. Restore localisation to original image coordinates using recorded inverse transforms.
6. Emit image-level result, confidence/uncertainty, localisation mask/overlay, and evidence metadata.
7. Validate output dimensions and finite values.

### Historical failure reproduction guards

Keep regression fixtures for:

- `Image/mask size mismatch` on MIDV `10.jpg`/`10.png` style cases;
- `Image/mask aspect-ratio mismatch` where image `(4032, 2268)` and mask `(2268, 4032)` require a valid rotation;
- doctor crash from `next(progress)` on a `tqdm_asyncio` object;
- ROS plugin `PluginValidationError` caused by an unknown pytest hook.

### Code-only archive audit

The archive must exclude:

- `data/` and external dataset roots;
- `output/` preparation/doctor/HPO/train/eval/inference state;
- checkpoints/models;
- logs;
- environments/caches;
- editor/system files;
- credentials.

It includes current source, configs, tests, READMEs/docs, licence/attribution, and only lightweight intentional fixtures.

After building, list contents, extract, run import/tests from the extracted tree, and generate SHA-256.
