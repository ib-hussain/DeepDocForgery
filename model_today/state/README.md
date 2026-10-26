# Recovery state

Atomic stage state, progress cursors, append-only record journals, and exact
resume commands are written here. Preserve this directory with the associated
manifests, processed exports, checkpoints, reports, and inference artefacts.

Do not edit state manually. Rerun the command reported by
`python -m deepdocforgery status`; use a stage's `--fresh` flag only when its
saved work is intentionally being discarded.
