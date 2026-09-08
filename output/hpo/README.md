# Hyperparameter optimisation

Each CUDA HPO contract gets a stable study directory. Within it, every trial
keeps its resolved configuration, model run, resource peak, metrics, and
result. Rerunning the same command skips successful trials and resumes an
interrupted child trainer from its latest epoch. `--fresh` creates a separate
study directory.
The summary identifies the best successful trial and retains failed-trial exit
codes rather than hiding them.

Each trial also retains the child trainer's machine-readable stdout as
`train-result.json`. Training progress remains visible on stderr, while HPO
itself emits exactly one final JSON document on stdout.
