# Hyperparameter optimisation

Each CUDA HPO invocation gets a timestamped run-ID directory. Within it, every
trial keeps its resolved configuration, model run, resource peak, and metrics.
The summary identifies the best successful trial and retains failed-trial exit
codes rather than hiding them.

Each trial also retains the child trainer's machine-readable stdout as
`train-result.json`. Training progress remains visible on stderr, while HPO
itself emits exactly one final JSON document on stdout.
