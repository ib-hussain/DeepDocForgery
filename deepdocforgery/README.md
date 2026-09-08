# Python package

All executable project code is consolidated here. Run it through
`python -m deepdocforgery <command>` or the installed `deepdocforgery` command.

The package contains dataset preparation, the DCT/degradation/spatial branches,
fusion, the detail decoder, objectives, metrics, training, evaluation,
inference, diagnostics, and HPO. No code under legacy `src/` or `scripts/` is
required by the release.

`telemetry.py` supplies the shared logging, progress, RAM/VRAM monitoring, CPU
thread detection, and latest-run status contract used by every command.
`state.py` supplies atomic fingerprints, stale-process locks, record journals,
and recovery metadata used by every long-running stage.
