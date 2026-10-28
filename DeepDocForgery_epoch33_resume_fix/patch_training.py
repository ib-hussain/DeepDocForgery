#!/usr/bin/env python3
"""Patch AUROC and add full pre-validation checkpoints; fails closed on source drift."""
import ast
import re
import shutil
from pathlib import Path

ROOT = Path.cwd()
METRICS = ROOT/'deepdocforgery/metrics.py'
TRAIN = ROOT/'deepdocforgery/train.py'
BACKUP = ROOT/'output/recovery/epoch33-source-before-resume-fix'
MARKER = '# EPOCH33_RECOVERY_PREVALIDATION_CHECKPOINT_V1'

AUROC = '''def binary_auroc(scores: list[float], labels: list[int]) -> float | None:
    """Tie-correct AUROC. Discard non-finite scores so NaNs cannot stall the loop."""
    import math
    import warnings

    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    pairs = []
    invalid = 0
    for score, label in zip(scores, labels):
        if int(label) not in (0, 1):
            raise ValueError("AUROC labels must be 0 or 1")
        value = float(score)
        if not math.isfinite(value):
            invalid += 1
            continue
        pairs.append((value, int(label)))
    if invalid:
        warnings.warn(
            f"AUROC omitted {invalid} non-finite scores; inspect model numerics",
            RuntimeWarning,
            stacklevel=2,
        )
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    pairs.sort(key=lambda entry: entry[0])
    positive_rank_total = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        positive_rank_total += average_rank * sum(label for _, label in pairs[i:j])
        i = j
    area = (positive_rank_total - positives * (positives + 1) / 2.0) / (positives * negatives)
    return max(0.0, min(1.0, area))
'''

RECOVERY = '''        # EPOCH33_RECOVERY_PREVALIDATION_CHECKPOINT_V1
        # Persist all train state before expensive validation and AUROC metrics.
        # An AUROC failure cannot erase the epoch's completed optimizer work.
        _prevalidation_checkpoint = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            validation={"recovery_status": "training_complete_validation_pending"},
            contract=contract,
            data_generator_state=(
                None if getattr(train_loader, "generator", None) is None
                else train_loader.generator.get_state()
            ),
        )
        atomic_torch_save(
            _prevalidation_checkpoint,
            output_dir / f"epoch-{epoch + 1:04d}-prevalidation.pt",
        )
        del _prevalidation_checkpoint
        run.info(
            f"Pre-validation recovery checkpoint saved for epoch {epoch + 1}",
            event="prevalidation_checkpoint_saved",
            epoch=epoch + 1,
        )
'''


def main():
    m = METRICS.read_text()
    t = TRAIN.read_text()
    if MARKER in t:
        raise RuntimeError('Training already patched. Refusing to apply twice.')
    module = ast.parse(m)
    funcs = [node for node in module.body if isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef)) and node.name == 'binary_auroc']
    if len(funcs) != 1:
        raise RuntimeError('Cannot locate exactly one top-level binary_auroc()')
    func = funcs[0]
    old_fn = '\n'.join(m.splitlines()[func.lineno - 1:func.end_lineno])
    for expected in ('while index < len(order)', 'scores[order[index]] == threshold'):
        if expected not in old_fn:
            raise RuntimeError(f'AUROC source differs from observed buggy version: {expected!r}')
    new_m = '\n'.join(m.splitlines()[:func.lineno-1] + AUROC.rstrip().splitlines() + m.splitlines()[func.end_lineno:]) + '\n'
    ast.parse(new_m)
    pattern = re.compile(r'(?m)^        scheduler\.step\(epoch \+ 1\)\s*\n(?=        validation = evaluate_model\()')
    if len(pattern.findall(t)) != 1:
        raise RuntimeError('Expected exactly one scheduler.step(epoch + 1) immediately followed by validation = evaluate_model; source drift.')
    new_t = pattern.sub('        scheduler.step(epoch + 1)\n\n' + RECOVERY + '\n', t)
    ast.parse(new_t)
    BACKUP.mkdir(parents=True, exist_ok=True)
    for source in (METRICS, TRAIN):
        backup = BACKUP / source.name
        if backup.exists():
            raise RuntimeError(f'Backup already exists: {backup}; refusing repeated patch')
        shutil.copy2(source, backup)
    METRICS.write_text(new_m)
    TRAIN.write_text(new_t)
    print(f'PATCHED: {METRICS}')
    print(f'PATCHED: {TRAIN}')
    print(f'BACKUPS: {BACKUP}')


if __name__ == '__main__':
    main()
