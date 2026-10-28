# DeepDocForgery epoch 33 recovery audit

**Do not rename `epoch33_EMERGENCY.pt` to `last.pt`.** Keep native epoch-32 checkpoints and configs intact.

Run from the repository root after activating `.venv-cuda`:

```bash
mkdir -p output/recovery/epoch33-backup
cp -n output/model/cuda-full/epoch33_EMERGENCY.pt output/recovery/epoch33-backup/
cp -n output/model/cuda-full/last.pt output/recovery/epoch33-backup/last-epoch32.pt
python inspect_checkpoints.py | tee output/recovery/epoch33-checkpoint-audit.txt
python -m unittest -v test_robust_auroc.py
```

`inspect_checkpoints.py` is read-only and shows the actual native and emergency keys, hashes, numerical finiteness and model-key compatibility. It does not claim resumability. Before native conversion, inspect the exact resume/save code in `deepdocforgery/train.py`, including expected key names and epoch/global_step conventions.

`robust_auroc.py` is a standalone *candidate*, not an automatic code patch. It computes rank-based AUROC with tie support and filters non-finite scores without looping forever. The caller should record the number of invalid predictions; if no positive/negative class survives, AUROC is undefined and should not be used for checkpoint selection. A successful AUROC calculation does not establish that the model is numerically healthy.

Future robust checkpointing architecture:
1. Save a full recovery checkpoint **before validation metrics**; do not make it depend on AUROC.
2. `torch.save` to a temporary file on the **same filesystem**, `flush`/`fsync` file, `os.replace` atomically, optionally fsync its directory, and update `status.json` only **after** success.
3. Maintain rolling `last.pt`, `last.previous.pt`, immutable `epoch-XXXX.pt`, and independent `best.pt`.
4. Include model, optimizer, scheduler, AMP scaler, RNG(s), epoch, global_step, configuration fingerprint, and sampler metadata.
5. Automatically test loading a saved checkpoint in a new process with expected epoch and finite model weights; support a controlled failure test for disk errors.
6. Fix FP16/BF16 fallback and non-finite score generation; do not consider dropped AUROC values a model-quality fix.

`atomic_checkpoint.py` provides `save_checkpoint_atomic(checkpoint, destination)` for integrating with the existing native save routine. It writes the new object to a same-directory temporary file, fsyncs, retains the previous file by hard link, replaces atomically and fsyncs the directory. It preserves the *bytes* of the previous checkpoint if the process crashes during the new write; correct **contents and checkpoint scheduling** still depend on the training loop. Hardlinks require the output directory's filesystem to support them.
