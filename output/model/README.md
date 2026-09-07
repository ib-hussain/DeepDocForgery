# Models

Training runs store checkpoints, the resolved configuration, and epoch metrics
inside a named subdirectory here. Each run also maintains `status.json` with
its current epoch, best metric, checkpoints, resource snapshot, and terminal
state (`running`, `succeeded`, `failed`, or `interrupted`).
