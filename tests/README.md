# Tests

Tests cover tensor contracts, losses, exact/fallback DCT, source-group split
safety, official DocTamper protocol, MIDV `2268x4032` semantics, metrics,
checkpoint scheduling, profile configuration, and short end-to-end workflows.

All dataset-contract fixtures are created under pytest temporary directories.
The suite never scans or modifies the user's `data/` directory.

Telemetry tests verify automatic CPU sizing, RAM reporting, successful and
failed lifecycle logs, and the first-class `test`/`status` CLI commands.

MIDV coverage includes lower-resolution scale-aligned masks and rejection of
true aspect-ratio mismatches.

Run `python -m pytest -q` from the repository root.
