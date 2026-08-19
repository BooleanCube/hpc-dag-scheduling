# tasks

DAG math builder, node operations, and task definitions for the
[HPC DAG Scheduler Research Baseline](../README.md).

The builder is lazily evaluated: graph construction records intent, and all logical validation
happens in Python before the DAG is serialised for the C++ MPI engine. Build-time faults raise
`ShapeMismatchError`, `DimensionalityError`, `CyclicDependencyError`, or
`UninitializedNodeError` — all subclasses of `DagBuildError`. See the root README's
Error-Handling Contract for the full division of responsibility.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy .
```
