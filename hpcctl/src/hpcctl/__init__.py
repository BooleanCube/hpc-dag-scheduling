"""Typer CLI for managing AWS ParallelCluster lifecycles.

All commands that would mutate remote AWS state accept a ``--dry-run`` flag which
prints the intended payload instead of executing it.
"""

from hpcctl.cli import app, main

__all__ = ["app", "main"]
