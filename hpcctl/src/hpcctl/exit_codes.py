"""Process exit statuses for the hpcctl CLI."""

from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit statuses. Stable contract for CI and shell scripts.

    ``2`` is reserved rather than assigned: Typer already exits 2 on argument-parsing
    errors, and claiming it would give one code two meanings.
    """

    OK = 0
    INTERNAL = 1
    USAGE = 2
    CONFIG = 3
    DAG_INVALID = 4
    TOOL_MISSING = 5
    COMMAND_FAILED = 6
    ABORTED = 7
    CLUSTER_STATE = 8
