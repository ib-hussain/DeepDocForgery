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

Regenerate manifests with v0.4.2. Older manifests under `data/manifests/` may
contain invalid random train/validation splits and must not be reused. New
manifests are written to `output/manifests/`, while restart-safe DocTamper
exports are written to `output/processed/`. Raw LMDBs and MIDV images remain
input-only and are read in place.

The old top-level `src/` and `scripts/` packages are obsolete. All commands now
use `python -m deepdocforgery ...`; do not mix modules from both layouts.
