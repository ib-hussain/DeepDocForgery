# Inference

Inference writes probability maps, binary masks, overlays, instances, and the
JSON prediction report here by default.
Completed images are appended to `predictions.records.jsonl`; `state.json`
stores its input/checkpoint contract and resume cursor. Rerun the same command
after interruption, or use `--fresh` to regenerate deliberately.
