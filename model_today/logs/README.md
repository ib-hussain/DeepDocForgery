# Logs

Every pipeline command creates `COMMAND/RUN_ID.log` for people,
`COMMAND/RUN_ID.jsonl` for tools, and `COMMAND/latest.json` for current status.
The JSONL stream contains lifecycle events, progress checkpoints, RAM usage and,
for CUDA operations, VRAM allocation/reservation/peak/device totals.
Final status documents also record wall-clock duration; model metric logs add
epoch and evaluation throughput.
`latest.json` is atomically refreshed while a command runs, so an interruption
cannot leave a partially written status file.

Standalone doctor reports also live here. Evaluation/test reports default next
to their checkpoint so each experiment remains self-contained.
Run `python -m deepdocforgery status` to collect the latest statuses.

The shell installers additionally retain their complete pip/test transcripts
in `setup/` so environment failures remain auditable.
