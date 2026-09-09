# DeepDocForgery 0.5.2

DeepDocForgery is a research pipeline for joint document-level forgery
classification and pixel-level tamper localization. It trains through one
manifest while preserving the different roles of DocTamper and MIDV-DM.

## What changed in 0.5.2

- Fixed full-data preparation aborts caused by collisions in DocTamper's
  non-authoritative masked perceptual proxy. TrainingSet train/validation
  grouping remains strict, while immutable TestingSet/FCD/SCD boundaries now
  take precedence and cross-benchmark proxy collisions are logged for audit.
- Existing v0.5.1 preparation journals remain reusable; rerun the same prepare
  command without `--fresh` to continue from the completed records.
- CPU/CUDA setup retries transient pip installation failures while preserving
  the existing environment-resume and active-environment safety checks.
- Subcommand help now dispatches correctly (`deepdocforgery prepare --help`,
  `train --help`, and the other command-specific help screens).

## What changed in 0.5.1

- Setup, preparation, training, HPO, evaluation/testing, doctor, and inference
  now persist recovery state. `python -m deepdocforgery status` reports exact
  resume commands and completed record, batch, stage, trial, or epoch counts.
- Training resumes from `last.pt` by default, restoring model, optimiser,
  scheduler, AMP scaler, random generators, and deterministic loader state.
- Preparation and inference use append-only journals with atomic checkpoints;
  a torn final JSONL write is repaired safely after interruption.
- Resume fingerprints reject changed inputs or behaviour-affecting settings
  instead of mixing incompatible work.
- MIDV camera JPEGs are EXIF-transposed to their upright geometry. Both
  `4032x2268` and `2268x4032` storage/display orientations are supported, as
  are strictly scale-aligned masks. Geometry-changing samples use pixel DCT.
- Setup isolates pytest from ambient ROS/user-site plugins and reuses partially
  installed environments and cached package downloads after failure.
- Doctor checks every image, mask, and explicit ADN target referenced by the
  manifest before an expensive run, using bounded parallel filesystem checks.

## 0.5 observability baseline

- Every long operation has a `tqdm` progress bar and timestamped status events.
- Prepare, doctor, smoke, train, validation/test, HPO, and inference persist a
  readable log plus structured JSONL events beneath `output/logs/<command>/`.
- CPU thread and worker counts use `auto` by default. Torch uses every logical
  CPU available through process affinity; data workers are bounded by physical
  cores and use one Torch thread each to prevent oversubscription.
- Live telemetry shows process/system RAM on every device and allocated,
  reserved, peak, free, and total VRAM on CUDA. When `nvidia-smi` is present it
  also records GPU utilisation, temperature, power, and competing processes.
- Training maintains `output/model/<run>/status.json`; `deepdocforgery status`
  summarises the latest command and training states.
- Final results remain clean JSON on stdout; human progress goes to stderr, so
  commands remain scriptable.
- Run-start events record Python, PyTorch, CUDA-runtime, and platform versions
  alongside the resolved configuration/checkpoint paths for reproducibility.

The 0.4 scientific and data-protocol corrections remain in place:

- All Python code lives in `deepdocforgery/` behind one CLI.
- `prepare --profile cpu|cuda` indexes both datasets together.
- CPU uses the small samples; CUDA uses the complete datasets.
- DocTamper TrainingSet is train/validation only. TestingSet, FCD, and SCD are
  immutable test benchmarks.
- DocTamper never supervises image classification because its public records
  are positive-only. MIDV supplies authentic and forged classification labels.
- MIDV records are grouped by the JSON `base_image` (or a category-independent
  fallback), so derivatives of one source document cannot cross splits.
- DocTamper uses a masked perceptual source proxy for TrainingSet train/validation
  grouping; absent authoritative source IDs, this reduces but cannot mathematically
  eliminate source-document leakage. The official TestingSet/FCD/SCD boundary is
  authoritative, so proxy collisions across it are audited rather than treated as proof
  of leakage.
- Exact JPEG coefficients are used only when geometry is unchanged; resized or
  augmented samples use the differentiable pixel-DCT fallback.
- JPEG recompression and noise augmentations provide degradation targets.
- A stride-2 detail path reduces the old stride-8 localization bottleneck.
- Tamper-aware paired crops and positive-pixel weighting target small edits.
- CUDA uses a pretrained timm Swin backbone; CPU uses a compact native model.
- Only the timm stream receives its published ImageNet normalisation; forensic
  branches retain raw `[0,1]` pixels.
- Evaluation reports pixel micro F1, per-image macro F1, catastrophic misses,
  tamper-area buckets, all-background baseline, image metrics, instances, and
  each benchmark separately.

## Repository map

| Path | Purpose |
|---|---|
| `deepdocforgery/` | Model, data, training, evaluation, inference, HPO, and CLI |
| `configs/cpu/` | Sample-only CPU smoke and training profiles |
| `configs/cuda/` | Full-data 20 GiB CUDA and HPO profiles |
| `data/` | User-managed input datasets only; never generated output |
| `setup/` | Separate CPU and CUDA environment installers |
| `tests/` | Unit, protocol, and end-to-end regression tests |
| `output/` | Checkpoints, metrics, inference artifacts, and HPO results |
| `third_party/` | Attribution and third-party notices |

## Logs and live status

Each command prints concise progress such as loss, learning rate, RAM, process
RSS, and (on CUDA) VRAM. Detailed metric dictionaries are written to files
instead of flooding the terminal. Epoch/test summaries keep macro F1,
precision, recall, catastrophic-miss rate, and image AUROC visible.

```text
output/logs/train/<run-id>.log       human-readable events
output/logs/train/<run-id>.jsonl     structured events and resource samples
output/logs/train/latest.json        latest train command status
output/model/<name>/metrics.jsonl    complete metrics for every epoch
output/model/<name>/status.json      running/succeeded/failed model status
output/state/                         restart cursors and stage contracts
```

Each `latest.json` changes to `running` as soon as its command starts and is
atomically refreshed at progress events, so `status` can inspect active work as
well as completed and failed work.

| Command | Current status | Primary result |
|---|---|---|
| `prepare` | `output/state/prepare/<profile>/state.json` | `output/manifests/*.jsonl` |
| `doctor` | `output/state/doctor/<profile>.json` | `output/logs/doctor-*.json` |
| `hpo` | `output/hpo/study-*/state.json` | `output/hpo/study-*/summary.json` |
| `train` | `output/model/<name>/status.json` | checkpoints and `metrics.jsonl` |
| `test` | `<report>.state.json` | checkpoint-adjacent test metrics |
| `evaluate` | `output/logs/evaluate/latest.json` | requested evaluation report |
| `infer` | `output/inference/state.json` | `output/inference/predictions.json` |

Inspect all latest states at any time:

```bash
python -m deepdocforgery status
```

The status command exits with code `2` when work is missing, still running,
failed, interrupted, or completed with warnings, making it usable in scripts.

Use `--no-progress` on prepare, doctor, smoke, train, test/evaluate, HPO, or
inference when redirecting output in CI. Persistent logs are still written.

## CPU quick start

```bash
bash setup/cpu.sh
source .venv-cpu/bin/activate
python -m deepdocforgery prepare --profile cpu
python -m deepdocforgery doctor --profile cpu
python -m deepdocforgery smoke --device cpu
python -m deepdocforgery train --config configs/cpu/sample.yaml --device cpu --output output/model/cpu-sample
```

The CPU profile deliberately uses the sample datasets, but it no longer leaves
cores idle: `cpu_threads: auto` and `num_workers: auto` are resolved from the
CPUs actually available to the process. Torch, OpenMP, MKL, OpenBLAS, NumExpr,
Accelerate, and BLIS receive the same resolved limit; each data worker is then
restricted to one compute thread to avoid multiplying that limit.

CPU preparation reads your local `data/sample-doctamper` and
`data/sample-midv` trees. Dataset bytes are deliberately not distributed in
the code archive. Tests create isolated temporary fixtures and never inspect,
modify, or make assumptions about these directories.

MIDV's `2268x4032` upright portrait images are letterboxed without distortion: to
`216x384` inside the CPU `384x384` canvas and to `288x512` inside the CUDA
`512x512` canvas. MIDV masks may be stored at `1152x2048`; preparation accepts
that exact scale-aligned geometry, records it in the manifest, and the loader
uses nearest-neighbour alignment before shared crop/flip augmentation. A true
aspect-ratio mismatch remains a hard error. JPEGs stored as `4032x2268` with
EXIF rotation are transposed to the same upright geometry; already-upright and
landscape image/mask pairs are also accepted. Padding is excluded from losses
and metrics.

## ROS/pytest isolation

The repository pytest configuration blocks ROS 2 launch/ament pytest plugins that
can be auto-discovered when `/opt/ros/<distro>/...` is present on `PYTHONPATH`.
This means a normal `python -m pytest -q` remains project-local even in a shell
that previously sourced ROS. The setup scripts additionally set
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` while validating a fresh environment.

## CUDA full-data start

Expected paths:

```text
data/dataset-doctamper/
  DocTamperV1-TrainingSet/data.mdb
  DocTamperV1-TestingSet/data.mdb
  DocTamperV1-FCD/data.mdb
  DocTamperV1-SCD/data.mdb

data/dataset-midv/
  images/<forgery_type>/<document_type>/...
  masks/<forgery_type>/<document_type>/...
  annotations/<forgery_type>/<document_type>/...
```

Then run:

```bash
bash setup/cuda.sh
source .venv-cuda/bin/activate
python -m deepdocforgery prepare --profile cuda
python -m deepdocforgery doctor --profile cuda
python -m deepdocforgery train --config configs/cuda/full.yaml --device cuda --output output/model/cuda-full
```

`configs/cuda/full.yaml` targets roughly 20 GiB VRAM: physical batch 4,
gradient accumulation 4 (effective batch 16), AMP, checkpointed Swin-T, and
512x512 inputs. Its pretrained backbone uses one quarter of the main learning
rate. If the first real batch runs out of memory, lower physical
batch size to 3 or 2 and increase accumulation to preserve the effective batch.

The CUDA doctor checks the complete manifest and requires at least 16 GiB free
before this profile starts. It exits with status `2` when a referenced dataset
file is missing, another compute job (for example `hashcat`) is using the GPU,
or utilisation is already high. Resolve the reported issue and rerun `doctor`;
do not make the training process compete for VRAM.

## HPO

HPO is CUDA-only and intentionally small:

```bash
python -m deepdocforgery hpo --config configs/cuda/full.yaml --search configs/cuda/hpo.yaml --device cuda --output output/hpo
```

The deterministic study directory is derived from the search contract. Reruns
reuse successful trials and continue interrupted/failed trials; `--fresh`
creates a separate study when a genuinely new run is intended.

Run HPO only after `doctor --profile cuda` and a one-epoch bounded training
run succeed on the real datasets.

Training automatically resumes `output/model/<name>/last.pt`. Pass an explicit
`--resume CHECKPOINT` to recover another compatible epoch. `--fresh` disables
resume and refuses to overwrite an existing experiment directory.

## Test, evaluate, and infer

```bash
python -m deepdocforgery test --config configs/cuda/full.yaml --checkpoint output/model/cuda-full/best.pt --split test --device cuda --output output/model/cuda-full/test-metrics.json
python -m deepdocforgery infer --input path/to/image-or-folder --checkpoint output/model/cuda-full/best.pt --output output/inference --device cuda
```

`test` is an alias for `evaluate --split test`; `evaluate` remains available
for explicit train/validation diagnostics. Both show progress and RAM/VRAM,
checkpoint their metric accumulators, and reuse a completed compatible report.
Inference checkpoints every ten images by default. Rerun the identical command
to resume either stage; use `--fresh` only to discard matching state.

`infer --exact-jpeg --native-size` is an expert option. It preserves a JPEG’s
coefficient grid but can consume much more memory for large MIDV photographs;
ordinary inference uses the trained letterbox geometry.

## Scientific interpretation

This code makes the intended branches trainable; it does not claim benchmark
superiority before a full run. Always report all four DocTamper benchmarks and
the custom group-safe MIDV split separately. Do not compare the combined micro
F1 alone with a paper’s per-image score. Inspect:

- `pixel_macro/f1_forged`;
- `failure/catastrophic_miss_rate`;
- `area/lt_2pct/f1`;
- precision and recall separately;
- authentic-document false positives and image AUROC;
- `image/false_positive_rate` and supervised authentic/forged counts;
- per-benchmark metrics and exact-DCT usage.

Every epoch records `supervision_draws`; initial and final image-classifier
losses are also separate. This makes inactive or routed-around branches
visible instead of allowing aggregate F1 to conceal them.

See `MIGRATION.md` before replacing an older checkout containing large data.

ADN supervision is explicit. `proxy` means a weak text-stroke proxy derived
from the image; `ground_truth` requires a separate `adn_text_mask`. A missing
ADN target produces zero ADN loss—it is never silently replaced by the tamper
mask.

## Useful ablations

Set `model.fusion.enabled_branches` to any non-empty subset of `spatial`,
`frequency`, and `degradation`. Keep the manifest, split, seed, training budget,
and metric definitions fixed when comparing ablations.

Dataset licences and access terms remain the user’s responsibility. Dataset
bytes are neither downloaded by setup scripts nor included in release archives.
