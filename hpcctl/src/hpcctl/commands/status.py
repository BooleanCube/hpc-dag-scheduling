"""The ``status`` command: report cluster and Slurm queue state."""

import json
from typing import Annotated, Any

import typer
from rich.table import Table

from hpcctl import console
from hpcctl.commands.options import DryRunOption, StrictOption, resolve_dry_run
from hpcctl.config import (
    REQUIRED_FOR_CLUSTER,
    REQUIRED_FOR_REMOTE,
    Settings,
    is_placeholder,
    load_settings,
)
from hpcctl.discovery import describe_argv, head_node_address
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
    dry_run: DryRunOption = False,
    queue: Annotated[
        bool, typer.Option("--queue/--no-queue", help="Include the Slurm queue.")
    ] = True,
    strict: StrictOption = False,
) -> None:
    """Report cluster and Slurm queue status.

    Degrades rather than fails: if the cluster query succeeds but SSH does not, the cluster table
    is printed with a warning and the exit status stays 0. A cluster that is still creating has no
    reachable head node yet, and that is normal rather than an error. Exit 8 is reserved for a
    cluster that is absent or in a failed state.

    A ``--watch`` mode existed as a flag but was never wired to a render loop, so it was
    removed rather than shipped as a silent no-op; use ``watch -n5 'hpcctl status'`` instead.
    """
    dry_run = resolve_dry_run(dry_run)
    required = REQUIRED_FOR_CLUSTER | (REQUIRED_FOR_REMOTE if queue else frozenset())
    settings = load_settings(live=not dry_run, strict=strict, required=required)

    describe = describe_argv(settings)
    squeue = _squeue_argv(settings, settings.head_node_host)

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
    # The describe payload already names the head node, so an unset HPCCTL_HEAD_NODE_HOST is
    # filled in from it here rather than by a second describe-cluster round trip.
    if is_placeholder(settings.head_node_host):
        discovered = head_node_address(payload)
        if not discovered:
            console.render_warning("queue unavailable: the cluster reports no head node address")
            return
        squeue = _squeue_argv(settings, discovered)
    try:
        require_tools("ssh")
        queue_result = run(squeue, dry_run=False)
    except HpcctlError as exc:
        # A creating cluster has no reachable head node yet; that is not a failure.
        console.render_warning(f"queue unavailable: {exc.message}")
        return
    console.out().print(_queue_table(_parse_squeue(queue_result.stdout if queue_result else "")))


def _squeue_argv(settings: Settings, host: str) -> list[str]:
    """Build the SSH command that reads the Slurm queue.

    Args:
        settings: Resolved settings supplying the SSH identity and Slurm location.
        host: Head-node address, from the environment or from discovery.

    Returns:
        The ``ssh`` argument vector.
    """
    return ssh_argv(
        key_path=settings.ssh_key_path,
        user=settings.ssh_user,
        host=host,
        remote_command=f"{settings.slurm_bin}/squeue --format='{SQUEUE_FORMAT}'",
    )


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
            hint="Does the cluster exist? Create it with 'hpcctl boot'.",
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
