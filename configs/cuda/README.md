# CUDA profiles

- `full.yaml`: full DocTamper + MIDV-DM training at 512x512 with AMP,
  checkpointed pretrained Swin-T, a physical batch of 4, and an effective
  batch of 16.
- `hpo.yaml`: bounded random-search space for the same hardware.

Run `python -m deepdocforgery doctor --profile cuda` before training.
The generated manifest is `output/manifests/cuda.jsonl`.

The full profile declares preflight limits for total/free VRAM and idle GPU
utilisation. The doctor enforces them before a long run begins.

Progress reports host RAM plus allocated, reserved, peak, free, and total VRAM,
as well as GPU utilisation and temperature when `nvidia-smi` is available.
Data-loader workers are auto-sized from the host CPU; model compute remains on
CUDA with AMP and gradient checkpointing.
