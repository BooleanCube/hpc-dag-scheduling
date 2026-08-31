"""Resolve the cluster's SSH endpoint at execute time.

The head node gets a fresh public IP on every cluster re-boot, which makes a hand-maintained
``HPCCTL_HEAD_NODE_HOST`` worse than none at all: a stale value points ``rsync``/``ssh`` at
someone else's address space and fails in confusing ways. So the variable is an *override*,
not a requirement — when it is unset, every live remote command asks the cluster itself via
``pcluster describe-cluster`` and uses the address it reports. Dry-run never queries anything
(P1) and simply carries the placeholder through the printed commands.
"""

import dataclasses
import json

from hpcctl import console
from hpcctl.config import Settings, is_placeholder
from hpcctl.errors import ClusterStateError, HpcctlError
from hpcctl.external import require_tools, run


def describe_argv(settings: Settings) -> list[str]:
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


def head_node_address(payload: dict[str, object]) -> str:
    """Extract the head node's address from a ``describe-cluster`` payload.

    Args:
        payload: Parsed ``pcluster describe-cluster`` output.

    Returns:
        The public IP when present, else the private IP, else an empty string.
    """
    head = payload.get("headNode")
    if not isinstance(head, dict):
        return ""
    return str(head.get("publicIpAddress") or head.get("privateIpAddress") or "")


def resolve_head_node(settings: Settings, *, dry_run: bool) -> Settings:
    """Fill in the head-node host, asking the cluster when the environment did not say.

    An explicitly set ``HPCCTL_HEAD_NODE_HOST`` always wins, so an operator can still pin the
    address (e.g. a private IP over a VPN). Discovery only runs live; in dry-run the
    placeholder flows through so the command stays fully offline.

    Args:
        settings: Resolved settings, possibly carrying a host placeholder.
        dry_run: Whether the caller is in dry-run.

    Returns:
        Settings whose ``head_node_host`` is usable, or unchanged in dry-run.

    Raises:
        ClusterStateError: If the cluster cannot be described, its output is unparseable, or
            it reports no reachable head node (still creating, or already deleted).
        ToolMissingError: If ``pcluster`` is not on PATH.
    """
    if not is_placeholder(settings.head_node_host):
        return settings
    if dry_run:
        console.render_notice(
            "HPCCTL_HEAD_NODE_HOST is unset; outside --dry-run the address is discovered "
            "from the cluster automatically"
        )
        # Drop the host from `missing` so the incomplete-configuration panel does not tell
        # the user to set a variable that discovery has just promised to handle.
        return dataclasses.replace(
            settings,
            missing=tuple(m for m in settings.missing if m != "HPCCTL_HEAD_NODE_HOST"),
        )

    require_tools("pcluster")
    try:
        completed = run(describe_argv(settings), dry_run=False)
    except HpcctlError as exc:
        raise ClusterStateError(
            f"could not discover the head node of cluster {settings.cluster_name!r}: {exc.message}",
            hint="Does the cluster exist? Create it with 'hpcctl boot'.",
        ) from exc
    try:
        payload = json.loads(completed.stdout if completed else "")
    except json.JSONDecodeError as exc:
        raise ClusterStateError(
            f"could not parse pcluster output while discovering the head node: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        payload = {}

    host = head_node_address(payload)
    if not host:
        state = payload.get("clusterStatus", "UNKNOWN")
        raise ClusterStateError(
            f"cluster {settings.cluster_name!r} reports no head node address (status: {state})",
            hint="A creating cluster has no address yet; poll 'hpcctl status'.",
        )
    console.render_notice(f"discovered head node {host} from cluster {settings.cluster_name!r}")
    return dataclasses.replace(settings, head_node_host=host)
