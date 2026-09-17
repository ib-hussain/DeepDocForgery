# Manifests

`prepare` writes `cpu.jsonl` or `cuda.jsonl` here with a matching
`*.summary.json`. Manifests contain paths and supervision metadata, not image
bytes. Regenerate them when dataset roots or split rules change.
