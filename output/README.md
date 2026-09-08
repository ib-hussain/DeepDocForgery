# Outputs

All generated artefacts belong here and are ignored by version control:

- `manifests/`: combined JSONL manifests and preparation summaries;
- `processed/`: restart-safe DocTamper LMDB exports;
- `model/`: checkpoints, resolved configurations, and epoch metrics;
- `logs/`: timestamped text/JSONL run logs, latest statuses, and reports;
- `inference/`: probability maps, masks, overlays, and predictions;
- `hpo/`: hyperparameter-search trials and summary;
- `embeddings/`: optional exported representations;
- `state/`: atomic restart contracts, cursors, and exact resume commands.

Dataset inputs never belong here and release archives contain no generated
artefacts from these folders.
