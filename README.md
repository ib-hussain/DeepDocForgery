# DeepDocForgery 0.4.2

DeepDocForgery is a research pipeline for joint document-level forgery
classification and pixel-level tamper localization. It trains through one
manifest while preserving the different roles of DocTamper and MIDV-DM.

## What changed in 0.4

- All Python code lives in `deepdocforgery/` behind one CLI.
- `prepare --profile cpu|cuda` indexes both datasets together.
- CPU uses the small samples; CUDA uses the complete datasets.
- DocTamper TrainingSet is train/validation only. TestingSet, FCD, and SCD are
  immutable test benchmarks.
- DocTamper never supervises image classification because its public records
  are positive-only. MIDV supplies authentic and forged classification labels.
- MIDV records are grouped by the JSON `base_image` (or a category-independent
  fallback), so derivatives of one source document cannot cross splits.
- DocTamper uses a masked perceptual source proxy for group-safe splitting;
  absent authoritative source IDs, this reduces but cannot mathematically
  eliminate source-document leakage.
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

## CPU quick start

```bash
bash setup/cpu.sh
source .venv-cpu/bin/activate

python -m deepdocforgery prepare --profile cpu
python -m deepdocforgery doctor --profile cpu
python -m deepdocforgery smoke --device cpu

python -m deepdocforgery train \
  --config configs/cpu/sample.yaml \
  --device cpu \
  --output output/model/cpu-sample
```

CPU preparation reads your local `data/sample-doctamper` and
`data/sample-midv` trees. Dataset bytes are deliberately not distributed in
the code archive. Tests create isolated temporary fixtures and never inspect,
modify, or make assumptions about these directories.

MIDV's `2268x4032` portrait images are letterboxed to `288x512` inside the
configured `512x512` canvas. Images and masks always share the exact transform,
and padding is excluded from losses and metrics.

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
python -m deepdocforgery train \
  --config configs/cuda/full.yaml \
  --device cuda \
  --output output/model/cuda-full
```

`configs/cuda/full.yaml` targets roughly 20 GiB VRAM: physical batch 4,
gradient accumulation 4 (effective batch 16), AMP, checkpointed Swin-T, and
512x512 inputs. Its pretrained backbone uses one quarter of the main learning
rate. If the first real batch runs out of memory, lower physical
batch size to 3 or 2 and increase accumulation to preserve the effective batch.

## HPO

HPO is CUDA-only and intentionally small:

```bash
python -m deepdocforgery hpo \
  --config configs/cuda/full.yaml \
  --search configs/cuda/hpo.yaml \
  --device cuda \
  --output output/hpo
```

Run HPO only after `doctor --profile cuda` and a one-epoch bounded training
run succeed on the real datasets.

## Evaluate and infer

```bash
python -m deepdocforgery evaluate \
  --config configs/cuda/full.yaml \
  --checkpoint output/model/cuda-full/best.pt \
  --split test \
  --device cuda \
  --output output/logs/test-metrics.json

python -m deepdocforgery infer \
  --input path/to/image-or-folder \
  --checkpoint output/model/cuda-full/best.pt \
  --output output/inference \
  --device cuda
```

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
