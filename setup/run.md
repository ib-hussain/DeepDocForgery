I've prepared the complete epoch 33 recovery and training restart package, including the source-code fixes.

Download DeepDocForgery Epoch 33 Resume Fix

Checkpoint conversion, AUROC repair, pre-validation checkpoint saving, verification tests, and training launch scripts.

## Step 1 — Place the ZIP in your repository

Copy the downloaded ZIP to:

```
/home/fastlab/deepFakeDetection/DeepDocForgery/
```

Then open the Ubuntu terminal on the GPU PC.

## Step 2 — Prepare the environment

Copy and run:

```
cd ~/deepFakeDetection/DeepDocForgery

source .venv-cuda/bin/activate

mountpoint -q output/processed/cuda || \
sudo mount /dev/nvme0n1p5 output/processed/cuda

unzip -o DeepDocForgery_epoch33_resume_fix.zip -d .
```

## Step 3 — Apply fixes, verify, and start training

For a long training run, I recommend using `tmux` so disconnecting your terminal does not end the process.

If `tmux` is installed, enter:

```
tmux new -s deepdoc-epoch34
```

Then execute:

```
cd ~/deepFakeDetection/DeepDocForgery

bash DeepDocForgery_epoch33_resume_fix/prepare.sh && \
bash DeepDocForgery_epoch33_resume_fix/launch.sh
```

This is the main command. It will automatically:

1. Verify the SHA-256 hashes of the original and rescued checkpoints.
2. Check all rescued model and optimizer tensors for non-finite values.
3. Convert the emergency checkpoint into the native format as `epoch33_RESUMABLE.pt`.
4. Set the completed epoch index to `32`, so training begins at epoch 34/50.
5. Patch `binary_auroc()` to prevent the NaN infinite loop.
6. Add a recovery checkpoint immediately before validation on every future epoch.
7. Run syntax checks, AUROC unit tests, and the native GPU checkpoint preflight.
8. Start training only if every preceding check succeeds.

The original emergency checkpoint, `last.pt`, and `best.pt` are backed up. The conversion resets RNG streams and inherits the previous AMP scaler, so continuation will not be bit-for-bit identical.

## Step 4 — Confirm epoch 34 starts

Look for these messages:

```
READY: epoch 33 native checkpoint passed preflight.
Resumed after epoch 33/50 ...
Epoch 34/50 started
```

The exact log wording may vary, but the epoch must be 34/50, not 33/50.

To monitor from another terminal:

```
cd ~/deepFakeDetection/DeepDocForgery

tail -f output/recovery/epoch34-launch.log
```

The new pre-validation checkpoints will look like:

```
output/model/cuda-full/epoch-0034-prevalidation.pt
output/model/cuda-full/epoch-0035-prevalidation.pt
...
```

If you use `tmux`, press `Ctrl+B`, then `D` to detach without stopping training.

Important: If the preparation command reports `STOP`, an exception, or a failed preflight, the training command will not run. Do not bypass that failure.

Our target is to preserve the completed 33 epochs and continue through epoch 50 without repeating the previous AUROC blockage.
