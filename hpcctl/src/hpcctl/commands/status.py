"""The ``status`` command: report cluster and Slurm queue state."""

import json
from typing import Annotated, Any

import typer
from rich.table import Table

from hpcctl import console
from hpcctl.commands.options import DryRunOption, StrictOption, resolve_dry_run
from hpcctl.config import REQUIRED_FOR_CLUSTER, REQUIRED_FOR_REMOTE, Settings, load_settings
from hpcctl.errors import ClusterStateError, HpcctlError
from hpcctl.external import require_tools, run, ssh_argv

SQUEUE_FORMAT = "%.18i %.24j %.10T %.6D %.10M"
"""``squeue`` format: job ID, name, state, node count, elapsed."""

FAILED_STATES = frozenset(
    {
        "CREATE_FAILED",
        "DELETE_FAILED",
        "UPDATE_FAILED",
        "DELETE_COMPLETE",
    }
)
"""Cluster states that mean the cluster cannot be used."""


def status(
    dry_run: DryRunOption = True,
    queue: Annotated[
        bool, typer.Option("--queue/--no-queue", help="Include the Slurm queue.")
    ] = True,
    watch: Annotated[bool, typer.Option("--watch", help="Re-render on an interval.")] = False,
    strict: StrictOption = False,
) -> None:
    """Report cluster and Slurm queue status.

    Degrades rather than fails: if the cluster query succeeds but SSH does not, the cluster table
    is printed with a warning and the exit status stays 0. A cluster that is still creating has no
    reachable head node yet, and that is normal rather than an error. Exit 8 is reserved for a
    cluster that is absent or in a failed state.
    """
    dry_run = resolve_dry_run(dry_run)
    if watch and dry_run:
        # typer.BadParameter rather than HpcctlError: this is an illegal combination of flags,
        # so it belongs to Typer's exit code 2. Raising the base HpcctlError would exit 1, which
        # the enum defines as "unexpected exception; always a bug" -- telling CI that hpcctl
        # crashed when it in fact refused on purpose.
        raise typer.BadParameter(
            "--watch is not available in dry-run; output is static and it would loop forever. "
            "Pass --execute to watch a real cluster.",
            param_hint="--watch",
        )

    required = REQUIRED_FOR_CLUSTER | (REQUIRED_FOR_REMOTE if queue else frozenset())
    settings = load_settings(live=not dry_run, strict=strict, required=required)

    describe = _describe_argv(settings)
    squeue = ssh_argv(
        key_path=settings.ssh_key_path,
        user=settings.ssh_user,
        host=settings.head_node_host,
        remote_command=f"squeue --format='{SQUEUE_FORMAT}'",
    )

    if dry_run:
        console.render_artifact(
            "status queries (bash)",
            "\n".join(
                console.format_command(argv)
                for argv in ((describe, squeue) if queue else (describe,))
            ),
            "bash",
        )
        console.out().print(_cluster_table(settings, None))
        if queue:
            console.out().print(_queue_table([]))
        console.render_notice("dry-run shows placeholder rows so the layout is reviewable")
        console.render_placeholder_warning(settings)
        return

    require_tools("pcluster")
    completed = run(describe, dry_run=False)
    payload = _parse_describe(completed.stdout if completed else "")
    state = str(payload.get("clusterStatus", "UNKNOWN"))
    if state in FAILED_STATES:
        raise ClusterStateError(
            f"cluster {settings.cluster_name!r} is in state {state}",
            hint="Inspect it with 'pcluster describe-cluster' or recreate it with 'hpcctl boot'.",
        )
    console.out().print(_cluster_table(settings, payload))

    if not queue:
        return
    try:
        require_tools("ssh")
        queue_result = run(squeue, dry_run=False)
    except HpcctlError as exc:
        # A creating cluster has no reachable head node yet; that is not a failure.
        console.render_warning(f"queue unavailable: {exc.message}")
        return
    console.out().print(_queue_table(_parse_squeue(queue_result.stdout if queue_result else "")))


def _describe_argv(settings: Settings) -> list[str]:
    """Build the ``pcluster describe-cluster`` command.

    Args:
        settings: Resolved settings supplying the cluster name and region.

    Returns:
        The ``pcluster`` argument vector.
    """
    return [
        "pcluster",
        "describe-cluster",
        "--cluster-name",
        settings.cluster_name,
        "--region",
        settings.region,
    ]


def _parse_describe(stdout: str) -> dict[str, Any]:
    """Parse ``pcluster describe-cluster`` JSON output.

    Args:
        stdout: Raw command output.

    Returns:
        The parsed payload, empty when output was not a JSON object.

    Raises:
        ClusterStateError: If the output is not parseable as JSON at all.
    """
    if not stdout.strip():
        raise ClusterStateError(
            "pcluster describe-cluster returned no output",
            hint="Does the cluster exist? Create it with 'hpcctl boot --execute'.",
        )
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ClusterStateError(f"could not parse pcluster output as JSON: {exc.msg}") from exc
    return payload if isinstance(payload, dict) else {}


def _cluster_table(settings: Settings, payload: dict[str, Any] | None) -> Table:
    """Build the cluster summary table.

    Args:
        settings: Resolved settings, used for the dry-run placeholder row.
        payload: Parsed ``describe-cluster`` output, or ``None`` in dry-run.

    Returns:
        A renderable rich table with exactly one row.
    """
    table = console.new_table("cluster", "name", "status", "region", "head node", "compute fleet")
    if payload is None:
        table.add_row(settings.cluster_name, "<dry-run>", settings.region, "<dry-run>", "<dry-run>")
        return table
    head = payload.get("headNode")
    head_ip = ""
    if isinstance(head, dict):
        head_ip = str(head.get("publicIpAddress") or head.get("privateIpAddress") or "")
    table.add_row(
        str(payload.get("clusterName", settings.cluster_name)),
        str(payload.get("clusterStatus", "UNKNOWN")),
        str(payload.get("region", settings.region)),
        head_ip,
        str(payload.get("computeFleetStatus", "")),
    )
    return table


def _parse_squeue(stdout: str) -> list[tuple[str, ...]]:
    """Parse ``squeue`` tabular output into rows.

    Args:
        stdout: Raw command output including its header line.

    Returns:
        One tuple per job, with the header discarded.
    """
    lines = [line for line in stdout.splitlines() if line.strip()]
    return [tuple(line.split()) for line in lines[1:]]


def _queue_table(rows: list[tuple[str, ...]]) -> Table:
    """Build the Slurm queue table.

    Args:
        rows: Parsed ``squeue`` rows; empty renders a placeholder line.

    Returns:
        A renderable rich table.
    """
    table = console.new_table("slurm queue", "job id", "name", "state", "nodes", "elapsed")
    if not rows:
        table.add_row("<none>", "<dry-run>", "<dry-run>", "-", "-")
        return table
    for row in rows:
        padded = (*row, *("",) * 5)[:5]
        table.add_row(*padded)
    return table
