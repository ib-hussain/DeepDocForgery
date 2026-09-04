# CUDA profiles

- `full.yaml`: full DocTamper + MIDV-DM training at 512x512 with AMP,
  checkpointed pretrained Swin-T, a physical batch of 4, and an effective
  batch of 16.
- `hpo.yaml`: bounded random-search space for the same hardware.

Run `python -m deepdocforgery doctor --profile cuda` before training.
The generated manifest is `output/manifests/cuda.jsonl`.
