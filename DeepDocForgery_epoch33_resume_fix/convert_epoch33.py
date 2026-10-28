#!/usr/bin/env python3
"""Convert the GDB-rescued epoch 33 state to DeepDocForgery native format 2.
Strictly validates both source checkpoints and never alters them.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

NATIVE = Path('output/model/cuda-full/last.pt')
EMERGENCY = Path('output/model/cuda-full/epoch33_EMERGENCY.pt')
DEST = Path('output/model/cuda-full/epoch33_RESUMABLE.pt')
MANIFEST = Path('output/recovery/epoch33-conversion-manifest.json')
EXPECTED_CONTRACT = 'cc140f9fda5031deb120f7d7cffd1c06965b8fef96a5bf472596e842b8b506b1'
EXPECTED_NATIVE_SHA = '27f841c60df62fc7cfb5b842f077f078d91d63c293cd9dabb35b22e6d3bce1fc'
EXPECTED_EMERGENCY_SHA = '62419ffb745121e6df5667b99101c26f35287513b8cb10ff1fd7253951816ace'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for piece in iter(lambda: file.read(8 * 1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def check_finite(mapping, name):
    failures = []
    checked = 0
    for key, tensor in mapping.items():
        if torch.is_tensor(tensor):
            checked += 1
            if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all().item():
                failures.append(str(key))
    if failures:
        raise RuntimeError(f'{name} contains non-finite tensors: {failures[:8]}')
    print(f'PASS: {name} {checked} tensors finite', flush=True)


def main():
    if DEST.exists():
        raise FileExistsError(f'REFUSING to overwrite {DEST}. Move it to a backup if conversion must be rerun.')
    for path, digest in ((NATIVE, EXPECTED_NATIVE_SHA), (EMERGENCY, EXPECTED_EMERGENCY_SHA)):
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha256(path)
        if got != digest:
            raise RuntimeError(f'STOP: {path} SHA256 differs from the Oct 28 audit. Expected {digest}, got {got}')
        print(f'PASS: checksum {path}', flush=True)

    old = torch.load(NATIVE, map_location='cpu', weights_only=False)
    new = torch.load(EMERGENCY, map_location='cpu', weights_only=False)
    if old.get('format_version') != 2 or old.get('epoch') != 31 or old.get('global_step') != 78848:
        raise RuntimeError('Unexpected native checkpoint version/epoch/global_step')
    if new.get('epoch') != 33 or new.get('rescue') is not True:
        raise RuntimeError('Rescue checkpoint is not the expected completed epoch 33')
    for saved in (old, new):
        if saved.get('contract') != EXPECTED_CONTRACT:
            raise RuntimeError('Configuration/checkpoint contract mismatch')
    a = old['model']
    b = new['model_state_dict']
    if set(a) != set(b):
        raise RuntimeError('Model parameter/buffer names do not match')
    if any(not torch.is_tensor(a[k]) or not torch.is_tensor(b[k]) or a[k].shape != b[k].shape for k in a):
        raise RuntimeError('Model tensors/types/shapes do not match')
    check_finite(b, 'rescue model')
    changed = sum(not torch.equal(a[k], b[k]) for k in a)
    if changed < 1:
        raise RuntimeError('Rescued model is identical to the older model; refusing recovery')
    print(f'PASS: {changed}/{len(a)} model tensors changed since epoch 32')
    old_opt = old['optimizer']
    new_opt = new['optimizer_state_dict']
    if not new_opt.get('state') or len(old_opt['param_groups']) != len(new_opt['param_groups']):
        raise RuntimeError('Rescued optimizer empty or parameter groups changed')
    for p, q in zip(old_opt['param_groups'], new_opt['param_groups']):
        if p['params'] != q['params']:
            raise RuntimeError('Rescue optimizer parameter group IDs/order differ')
    for key, state in new_opt['state'].items():
        if not isinstance(state, dict):
            raise RuntimeError(f'Invalid optimizer entry for {key}')
        check_finite_quiet = [name for name, value in state.items() if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item()]
        if check_finite_quiet:
            raise RuntimeError(f'Non-finite optimizer states for parameter {key}: {check_finite_quiet}')
    print(f'PASS: rescued optimizer has {len(new_opt["state"])} finite parameter states', flush=True)
    scheduler = new['scheduler_state_dict']
    if int(scheduler.get('last_epoch', -1)) != 33:
        raise RuntimeError(f'Scheduler did not finish epoch 33: {scheduler}')
    if scheduler.get('total_epochs') != old['scheduler'].get('total_epochs'):
        raise RuntimeError('Scheduler schedule differs from old checkpoint')
    if not math.isfinite(float(new['best_metric'])):
        raise RuntimeError('Rescue best metric is not finite')
    if 'scaler' not in old or 'config' not in old:
        raise RuntimeError('Native checkpoint missing AMP scaler or config')
    cfg = old['config']
    training = cfg['training']
    accumulate = int(training.get('gradient_accumulation_steps', training.get('accumulation_steps', 4)))
    if accumulate != 4:
        raise RuntimeError(f'Expected gradient accumulation 4, got {accumulate}')
    # 9853 physical minibatches were completed; the final partial accumulation
    # is flushed by train.py's finish_accumulation(). The training loop counts
    # optimizer step attempts including AMP-skipped steps.
    attempts = (9853 + accumulate - 1) // accumulate
    global_step = int(old['global_step']) + attempts
    if global_step != 81312:
        raise RuntimeError(f'Unexpected reconstructed global step: {global_step}')

    # The live process's RNG and AMP scaler were not captured by the emergency
    # script. Generate fresh, reproducible RNG streams; carry over the older
    # scaler state. This is a valid training continuation but not bitwise identical.
    base_seed = int(training.get('seed', 7))
    seed = base_seed + 1000033
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda_rng = torch.cuda.get_rng_state_all()
    else:
        cuda_rng = []
    rng = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': cuda_rng,
    }
    generator = torch.Generator().manual_seed(seed + 1)

    checkpoint = dict(old)
    checkpoint.update({
        'model': b,
        'optimizer': new_opt,
        'scheduler': scheduler,
        'epoch': 32,  # native zero-based last completed epoch; next = epoch 34
        'global_step': global_step,
        'best_metric': float(new['best_metric']),
        'validation': {'rescue': 'epoch33 trained and validated; full validation metric calculation did not complete'},
        'rng_state': rng,
        'data_generator_state': generator.get_state(),
        # 'scaler' preserved from epoch32 -- current scaler not captured in rescue.
    })
    DEST.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f'.{DEST.name}.', suffix='.tmp', dir=DEST.parent, delete=False) as file:
            temp = Path(file.name)
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, DEST)
        temp = None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)

    # Read the actual saved file back and assert metadata is complete.
    del checkpoint
    verified = torch.load(DEST, map_location='cpu', weights_only=False)
    for key in old:
        if key not in verified:
            raise RuntimeError(f'Native key missing after save: {key}')
    if verified['epoch'] != 32 or verified['global_step'] != 81312:
        raise RuntimeError('Written native epoch/global_step incorrect')
    if set(verified['model']) != set(a):
        raise RuntimeError('Written model keys differ')
    print(f'PASS: native checkpoint reloaded; epoch={verified["epoch"]} (zero-based), global_step={verified["global_step"]}')
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        'source_epoch32_sha256': sha256(NATIVE),
        'source_rescue_sha256': sha256(EMERGENCY),
        'converted_sha256': sha256(DEST),
        'converted_checkpoint': str(DEST),
        'completed_epoch': 33,
        'next_epoch': 34,
        'global_step_estimated': global_step,
        'rng_state': 'reseeded, not reproduced exactly',
        'amp_scaler': 'inherited from epoch32',
        'validation': 'epoch33 full metrics incomplete',
    }, indent=2) + '\n')
    print(f'SUCCESS: {DEST} ({DEST.stat().st_size:,} bytes)', flush=True)
    print('WARNING: resumed data order, RNG, and AMP scale are not bitwise continuous with the original process.',flush=True)


if __name__ == '__main__':
    main()
