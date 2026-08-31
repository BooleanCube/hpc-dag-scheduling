"""The ``deploy`` command: sync compiled engine binaries to the shared filesystem."""

import shlex
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from hpcctl import console
from hpcctl.commands.options import DryRunOption, StrictOption, resolve_dry_run
from hpcctl.config import (
    REQUIRED_FOR_REMOTE,
    Settings,
    default_engine_source_dir,
    load_settings,
)
from hpcctl.discovery import resolve_head_node
from hpcctl.errors import InvalidConfigError
from hpcctl.external import require_tools, run

BuildDirOption = Annotated[
    Path | None,
    typer.Option(
        help="Local engine build tree; only its bin/ subdirectory is synced. "
        "Overrides the environment."
    ),
]

MANIFEST_LIMIT = 25
"""Rows shown in the dry-run transfer manifest before it is summarised."""


def deploy(
    dry_run: DryRunOption = False,
    build_dir: BuildDirOption = None,
    no_build: Annotated[
        bool,
        typer.Option("--no-build", help="Skip the local CMake build and ship bin/ as it is."),
    ] = False,
    strict: StrictOption = False,
) -> None:
    """Rebuild the engine locally, then sync its binaries to the cluster's shared filesystem.

    The build runs first so a deploy can never ship a binary that is older than the engine
    sources. ``--no-build`` skips it, for shipping a prebuilt out-of-tree binary. In
    ``--dry-run`` nothing is built (dry-run executes no external command at all); the cmake
    invocations are printed instead.

    The target lives under the shared filesystem rather than the head node's home directory:
    compute nodes must see the same binary the head node has, so this and the cluster's
    ``SharedStorage`` mount point must agree. Deploying to ``~ubuntu`` would produce jobs that
    run on the head node and fail everywhere else with "No such file or directory".

    Only ``bin/`` inside the build tree ships. CMake writes executables there
    (``RUNTIME_OUTPUT_DIRECTORY`` in ``engine/CMakeLists.txt``); the rest of the tree is cache,
    object files, and compiler probes that would waste transfer and clutter ``/shared``.
    """
    dry_run = resolve_dry_run(dry_run)
    settings = load_settings(live=not dry_run, strict=strict, required=REQUIRED_FOR_REMOTE)
    source = build_dir if build_dir is not None else settings.engine_build_dir
    bin_dir = source / "bin"
    build_argvs = None if no_build else _cmake_argvs(source)

    if not dry_run and build_argvs is not None:
        require_tools("cmake")
        for build_argv in build_argvs:
            run(build_argv, dry_run=False, capture=False)

    # A local precondition, so it is checked in dry-run too: catching "you have no engine
    # binaries" costs nothing and needs no AWS account. Checked before head-node discovery
    # so the free failure comes before the AWS query.
    files = _verify_build_dir(source)
    settings = resolve_head_node(settings, dry_run=dry_run)

    argv = _rsync_argv(settings, bin_dir)
    target = f"{settings.remote_engine_dir}/bin"

    if dry_run:
        if build_argvs is not None:
            console.render_artifact(
                "engine build (bash)",
                "\n".join(console.format_command(a) for a in build_argvs),
                "bash",
            )
        console.out().print(_manifest(bin_dir, files))
        console.render_artifact("rsync invocation (bash)", console.format_command(argv), "bash")
        console.render_notice(
            f"would sync {len(files)} file(s) to "
            f"{settings.ssh_user}@{settings.head_node_host}:{target}/"
        )
        console.render_placeholder_warning(settings)
        return

    require_tools("rsync", "ssh")
    run(argv, dry_run=False, capture=False)
    console.render_notice(f"synced {len(files)} file(s) to {target}")


def _cmake_argvs(build_tree: Path) -> list[list[str]]:
    """Build the cmake configure and compile commands for the engine.

    The source tree is always the repository's ``/engine`` (humans write it, hpcctl only
    builds it); the build tree is wherever the binaries are configured to live, so an
    out-of-tree ``--build-dir`` still gets a fresh binary.

    Args:
        build_tree: The CMake build directory to configure and compile into.

    Returns:
        The configure and compile argument vectors, in execution order.

    Raises:
        InvalidConfigError: If the engine source tree cannot be found.
    """
    source_tree = default_engine_source_dir()
    if not (source_tree / "CMakeLists.txt").is_file():
        raise InvalidConfigError(
            f"engine source tree not found at {source_tree}",
            hint="Run from within the repository, or pass --no-build to ship prebuilt binaries.",
        )
    return [
        [
            "cmake",
            "-S",
            source_tree.as_posix(),
            "-B",
            build_tree.as_posix(),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        ["cmake", "--build", build_tree.as_posix(), "--parallel"],
    ]


def _rsync_argv(settings: Settings, bin_dir: Path) -> list[str]:
    """Build the ``rsync`` command that syncs binaries to the head node.

    ``StrictHostKeyChecking=accept-new`` still pins the host key after first contact, so it
    protects against a later MITM without prompting on first connect. Never ``no``.

    The ``-e`` transport is built with ``shlex.join`` rather than an f-string. rsync splits that
    value into words itself, honouring quotes, so an unquoted key path containing a space would
    be torn into two arguments and ssh would be handed the wrong identity file.

    The destination is ``<remote_engine_dir>/bin/``, mirroring the local ``bin/`` source, so
    the default ``HPCCTL_ENGINE_BINARY`` of ``<remote_engine_dir>/bin/engine`` keeps holding.
    Scoping ``--delete`` to ``bin/`` also keeps it from reaping anything an operator has
    parked elsewhere under the remote engine directory.

    ``--mkpath`` because a freshly booted cluster has an empty ``/shared``: rsync creates only
    the final path component, and the destination sits two levels below the mount (observed
    live: ``mkdir "/shared/engine/bin" failed: No such file or directory``). Needs rsync >=
    3.2.3 on both ends; ParallelCluster's Ubuntu 22.04 ships 3.2.7.

    Args:
        settings: Resolved settings supplying the SSH identity and remote target.
        bin_dir: Local directory of compiled binaries (the build tree's ``bin/``).

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
        "--mkpath",
        "-e",
        ssh_transport,
        f"{bin_dir.as_posix().rstrip('/')}/",
        f"{settings.ssh_user}@{settings.head_node_host}:{settings.remote_engine_dir}/bin/",
    ]


def _verify_build_dir(source: Path) -> list[Path]:
    """Check that the local build tree holds compiled binaries under ``bin/``.

    Args:
        source: Local engine build tree.

    Returns:
        Every regular file beneath ``source/bin``, sorted.

    Raises:
        InvalidConfigError: If the tree is absent, is not a directory, or has no ``bin/``
            files to ship.
    """
    if not source.exists():
        raise InvalidConfigError(
            f"engine build directory does not exist: {source}",
            hint=(
                "deploy builds the engine automatically outside --dry-run. Build once "
                "manually to preview: cmake -S engine -B engine/build && "
                "cmake --build engine/build. Or point --build-dir / HPCCTL_ENGINE_BUILD_DIR "
                "at an existing build tree."
            ),
        )
    if not source.is_dir():
        raise InvalidConfigError(f"engine build path is not a directory: {source}")
    bin_dir = source / "bin"
    files = sorted(path for path in bin_dir.rglob("*") if path.is_file())
    if not files:
        raise InvalidConfigError(
            f"engine build directory holds no binaries: {bin_dir}",
            hint=(
                "cmake --build writes the engine executable to bin/ inside the build tree; "
                "an absent or empty bin/ means the build did not complete or --no-build "
                "skipped it."
            ),
        )
    return files


def _manifest(source: Path, files: list[Path]) -> Table:
    """Build the dry-run table of files that would transfer.

    The manifest is assembled from the local filesystem rather than by invoking rsync, so it
    needs neither SSH nor rsync installed and cannot fail offline.

    Args:
        source: Local binaries directory, used to relativise names.
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
