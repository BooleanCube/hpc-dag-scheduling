# hpcctl

Typer-based CLI for managing AWS ParallelCluster lifecycles for the
[HPC DAG Scheduler Research Baseline](../README.md).

Every command that mutates remote AWS state must support `--dry-run`, printing the intended
payload instead of executing it. Credentials and cluster addresses come from environment
variables or the standard AWS credential chain — never from committed config.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy .

uv run hpcctl --help
```
