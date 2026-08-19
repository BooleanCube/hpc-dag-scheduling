"""The ``deploy`` command: sync compiled engine binaries to the shared filesystem."""

import shlex
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from hpcctl import console
from hpcctl.commands.options import DryRunOption, StrictOption, resolve_dry_run
from hpcctl.config import REQUIRED_FOR_REMOTE, Settings, load_settings
from hpcctl.errors import InvalidConfigError
from hpcctl.external import require_tools, run

BuildDirOption = Annotated[
    Path | None,
    typer.Option(help="Local directory of compiled engine binaries. Overrides the environment."),
]

MANIFEST_LIMIT = 25
"""Rows shown in the dry-run transfer manifest before it is summarised."""


def deploy(
    dry_run: DryRunOption = True,
    build_dir: BuildDirOption = None,
    strict: StrictOption = False,
) -> None:
    """Sync compiled engine binaries to the cluster's shared filesystem.

    The target lives under the shared filesystem rather than the head node's home directory:
    compute nodes must see the same binary the head node has, so this and the cluster's
    ``SharedStorage`` mount point must agree. Deploying to ``~ubuntu`` would produce jobs that
    run on the head node and fail everywhere else with "No such file or directory".
    """
    dry_run = resolve_dry_run(dry_run)
    settings = load_settings(live=not dry_run, strict=strict, required=REQUIRED_FOR_REMOTE)
    source = build_dir if build_dir is not None else settings.engine_build_dir

    # A local precondition, so it is checked in dry-run too: catching "you have not built the
    # engine yet" costs nothing and needs no AWS account.
    files = _verify_build_dir(source)

    argv = _rsync_argv(settings, source)

    if dry_run:
        console.out().print(_manifest(source, files))
        console.render_artifact("rsync invocation (bash)", console.format_command(argv), "bash")
        console.render_notice(
            f"would sync {len(files)} file(s) to "
            f"{settings.ssh_user}@{settings.head_node_host}:{settings.remote_engine_dir}/"
        )
        console.render_placeholder_warning(settings)
        return

    require_tools("rsync", "ssh")
    run(argv, dry_run=False, capture=False)
    console.render_notice(f"synced {len(files)} file(s) to {settings.remote_engine_dir}")


def _rsync_argv(settings: Settings, source: Path) -> list[str]:
    """Build the ``rsync`` command that syncs binaries to the head node.

    ``StrictHostKeyChecking=accept-new`` still pins the host key after first contact, so it
    protects against a later MITM without prompting on first connect. Never ``no``.

    The ``-e`` transport is built with ``shlex.join`` rather than an f-string. rsync splits that
    value into words itself, honouring quotes, so an unquoted key path containing a space would
    be torn into two arguments and ssh would be handed the wrong identity file.

    Args:
        settings: Resolved settings supplying the SSH identity and remote target.
        source: Local build directory.

    Returns:
        The ``rsync`` argument vector.
    """
    ssh_transport = shlex.join(
        ["ssh", "-i", settings.ssh_key_path, "-o", "StrictHostKeyChecking=accept-new"]
    )
    return [
        "rsync",
        "-avz",
        "--delete",
        "-e",
        ssh_transport,
        f"{source.as_posix().rstrip('/')}/",
        f"{settings.ssh_user}@{settings.head_node_host}:{settings.remote_engine_dir}/",
    ]


def _verify_build_dir(source: Path) -> list[Path]:
    """Check that the local build directory exists and holds files.

    Args:
        source: Local build directory.

    Returns:
        Every regular file beneath ``source``, sorted.

    Raises:
        InvalidConfigError: If the directory is absent, is not a directory, or is empty.
    """
    if not source.exists():
        raise InvalidConfigError(
            f"engine build directory does not exist: {source}",
            hint="Build the C++ engine first, or set HPCCTL_ENGINE_BUILD_DIR.",
        )
    if not source.is_dir():
        raise InvalidConfigError(f"engine build path is not a directory: {source}")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise InvalidConfigError(
            f"engine build directory is empty: {source}",
            hint="Build the C++ engine first; there is nothing to deploy.",
        )
    return files


def _manifest(source: Path, files: list[Path]) -> Table:
    """Build the dry-run table of files that would transfer.

    The manifest is assembled from the local filesystem rather than by invoking rsync, so it
    needs neither SSH nor rsync installed and cannot fail offline.

    Args:
        source: Local build directory, used to relativise names.
        files: Regular files that would be synced.

    Returns:
        A renderable rich table.
    """
    table = console.new_table(f"transfer manifest ({source})", "file", "bytes", "modified")
    for path in files[:MANIFEST_LIMIT]:
        stat = path.stat()
        table.add_row(
            path.relative_to(source).as_posix(),
            str(stat.st_size),
            f"{stat.st_mtime:.0f}",
        )
    if len(files) > MANIFEST_LIMIT:
        table.add_row(f"... and {len(files) - MANIFEST_LIMIT} more", "", "")
    return table
