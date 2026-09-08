# Migrating an existing checkout

Install this release into a fresh directory. Do not duplicate the 100+ GB
datasets merely to update the Python code.

Keep the existing roots, or link them into the fresh checkout:

```bash
ln -s /absolute/path/to/dataset-doctamper data/dataset-doctamper
ln -s /absolute/path/to/dataset-midv data/dataset-midv
```

Alternatively pass the roots explicitly:

```bash
python -m deepdocforgery prepare --profile cuda --doctamper-root /absolute/path/to/dataset-doctamper --midv-root /absolute/path/to/dataset-midv
```

Regenerate manifests with v0.5.1. Older manifests under `data/manifests/` may
contain invalid random train/validation splits and must not be reused. New
manifests are written to `output/manifests/`, while restart-safe DocTamper
exports are written to `output/processed/`. Raw LMDBs and MIDV images remain
input-only and are read in place.

The old top-level `src/` and `scripts/` packages are obsolete. All commands now
use `python -m deepdocforgery ...`; do not mix modules from both layouts.

Version 0.5 adds core `tqdm` and `psutil` dependencies. Re-run the relevant
setup script or `python -m pip install -e ".[dev,data]"`. Existing checkpoints
and v0.4.2 manifests remain format-readable, but official experiments must use
the regenerated `output/manifests/` files described above. New commands create
timestamped logs and latest-status files under `output/logs/`.

Run `deactivate` before the first v0.5 setup if an older `.venv-cpu` or
`.venv-cuda` is active. The setup script clears an unmarked legacy environment
to remove leaked packages, but deliberately refuses to clear the environment
currently backing the shell.

MIDV manifests must also be regenerated. Version 0.5 records scale-aligned mask
geometry instead of rejecting valid `2268x4032` image / `1152x2048` mask pairs.
Version 0.5.1 additionally normalises EXIF-oriented `4032x2268` JPEG storage to
the upright `2268x4032` display/mask grid. The source images and masks are never
modified.

Version 0.5.1 writes restart state beneath `output/state/`, alongside HPO,
evaluation, inference, and model output directories. Preserve `output/` when
upgrading if you want to resume. Existing v0.5 DocTamper exports are reused and
not rewritten; because v0.5 did not create record journals, the first v0.5.1
preparation pass reconstructs and checkpoints their manifest records once.

Training now resumes `<output>/last.pt` automatically. Existing older
checkpoints remain loadable when their embedded configuration and current
manifest match, but their first resumed epoch cannot be bit-for-bit identical
because older checkpoints did not store every random-generator state.
