"""Shared Typer options and per-command helpers.

The ``--dry-run/--execute`` pair lives here rather than being retyped in five modules so the
wording and default cannot drift apart between commands.
"""

from pathlib import Path
from typing import Annotated

import typer

from hpcctl import console
from hpcctl.config import Settings, dry_run_forced

DryRunOption = Annotated[
    bool,
    typer.Option(
        "--dry-run/--execute",
        help="Print intended actions instead of performing them. Default: dry-run.",
    ),
]
"""Every AWS-touching command carries this identical pair."""

StrictOption = Annotated[
    bool,
    typer.Option(
        "--strict",
        help="Treat missing required environment variables as fatal even in dry-run.",
    ),
]

RawOption = Annotated[
    bool,
    typer.Option(
        "--raw",
        help="Print artifacts unformatted, for piping. Bypasses all rich rendering.",
    ),
]

EmitDirOption = Annotated[
    Path | None,
    typer.Option(help="Write artifacts here instead of HPCCTL_RUN_DIR."),
]


def resolve_dry_run(dry_run: bool) -> bool:
    """Apply the ``HPCCTL_DRY_RUN`` global kill switch.

    A kill switch that a flag cannot override is worth more than flag-precedence purity while
    the AWS account does not exist, so this defeats ``--execute`` rather than deferring to it.

    Args:
        dry_run: The value parsed from ``--dry-run/--execute``.

    Returns:
        ``True`` when the command must not touch AWS.
    """
    if not dry_run and dry_run_forced():
        console.render_warning(
            "HPCCTL_DRY_RUN is set; refusing to execute. Unset it to allow --execute."
        )
        return True
    return dry_run


def artifact_dir(settings: Settings, emit_dir: Path | None) -> Path:
    """Resolve and create the directory generated artifacts are written to.

    Args:
        settings: Resolved settings supplying the default run directory.
        emit_dir: Explicit override from ``--emit-dir``.

    Returns:
        An existing directory, created with parents when absent.
    """
    target = emit_dir if emit_dir is not None else settings.run_dir
    target.mkdir(parents=True, exist_ok=True)
    return target
