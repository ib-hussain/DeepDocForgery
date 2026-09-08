# Models

Training runs store checkpoints, the resolved configuration, and epoch metrics
inside a named subdirectory here. Each run also maintains `status.json` with
its current epoch, best metric, checkpoints, resource snapshot, and terminal
state (`running`, `succeeded`, `failed`, or `interrupted`).

`last.pt` is atomically replaced after every completed epoch and contains the
full optimiser/scheduler/scaler and RNG recovery state. Training resumes it by
default; `best.pt` remains the validation-selected model for reporting.
Evaluation reports and their `*.state.json` accumulators may also be kept in
the same run directory.
