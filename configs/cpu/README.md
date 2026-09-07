# CPU profiles

- `smoke.yaml`: tiny synthetic forward/backward check.
- `sample.yaml`: short combined run on `data/sample-doctamper` and
  `data/sample-midv` after `prepare --profile cpu`.

These profiles validate plumbing. They are not benchmark configurations.
The generated manifest is `output/manifests/cpu.jsonl`.

The CPU profile auto-detects process CPU affinity, uses all available logical
threads for model compute, and parallelises data loading across physical cores.
Progress reports normal RAM and process RSS.
