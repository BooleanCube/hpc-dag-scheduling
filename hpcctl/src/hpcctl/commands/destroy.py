"""The ``destroy`` command: delete the cluster and its compute resources."""

import sys
from typing import Annotated

import typer
from rich.panel import Panel

from hpcctl import console
from hpcctl.commands.options import DryRunOption, StrictOption, resolve_dry_run
from hpcctl.config import REQUIRED_FOR_CLUSTER, Settings, load_settings
from hpcctl.errors import AbortedError
from hpcctl.external import require_tools, run


def destroy(
    dry_run: DryRunOption = True,
    yes: Annotated[
        bool, typer.Option("--yes", help="Skip the confirmation prompt, for automation.")
    ] = False,
    strict: StrictOption = False,
) -> None:
    """Delete the cluster and all associated compute resources.

    Confirmation requires typing the cluster name, the way GitHub gates repository deletion: a
    ``y/N`` prompt is one keystroke away from destroying a running experiment, and the cluster
    name is the one thing a user who means it can always produce.

    Dry-run never prompts. Prompting there would train users to type the confirmation
    reflexively, which is precisely the habit this UX exists to prevent.
    """
    dry_run = resolve_dry_run(dry_run)
    settings = load_settings(live=not dry_run, strict=strict, required=REQUIRED_FOR_CLUSTER)
    argv = _delete_argv(settings)

    if dry_run:
        console.render_artifact("deletion command (bash)", console.format_command(argv), "bash")
        console.render_notice("dry-run does not prompt; re-run with --execute to delete")
        console.render_placeholder_warning(settings)
        return

    _confirm(settings, yes=yes)
    require_tools("pcluster")
    completed = run(argv, dry_run=False)
    if completed is not None and completed.stdout:
        console.out().print(completed.stdout.strip())
    console.render_notice(
        f"deletion of {settings.cluster_name!r} requested; it completes asynchronously"
    )


def _delete_argv(settings: Settings) -> list[str]:
    """Build the ``pcluster delete-cluster`` command.

    Args:
        settings: Resolved settings supplying the cluster name and region.

    Returns:
        The ``pcluster`` argument vector.
    """
    return [
        "pcluster",
        "delete-cluster",
        "--cluster-name",
        settings.cluster_name,
        "--region",
        settings.region,
    ]


def _confirm(settings: Settings, *, yes: bool) -> None:
    """Require the operator to type the cluster name before deleting it.

    Args:
        settings: Resolved settings naming the cluster at risk.
        yes: Skip the prompt entirely, for automation.

    Raises:
        AbortedError: If the typed name does not match, or stdin is not a TTY and ``--yes`` was
            not passed. Never blocks on a pipe.
    """
    if yes:
        return
    if not sys.stdin.isatty():
        raise AbortedError(
            "refusing to delete without confirmation: stdin is not a TTY",
            hint="Pass --yes to confirm non-interactively.",
        )
    console.err().print(
        Panel(
            f"About to DELETE cluster '{settings.cluster_name}' in {settings.region}.\n"
            "This terminates the head node and all compute nodes. Running jobs will be lost.",
            title="destructive action",
            border_style="red",
        )
    )
    typed = typer.prompt("Type the cluster name to confirm", default="", show_default=False)
    if typed.strip() != settings.cluster_name:
        raise AbortedError(
            "confirmation did not match the cluster name; nothing was deleted",
            hint=f"Expected exactly: {settings.cluster_name}",
        )
