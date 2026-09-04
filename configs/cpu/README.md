# CPU profiles

- `smoke.yaml`: tiny synthetic forward/backward check.
- `sample.yaml`: short combined run on `data/sample-doctamper` and
  `data/sample-midv` after `prepare --profile cpu`.

These profiles validate plumbing. They are not benchmark configurations.
The generated manifest is `output/manifests/cpu.jsonl`.
