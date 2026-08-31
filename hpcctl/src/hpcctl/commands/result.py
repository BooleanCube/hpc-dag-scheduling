"""The ``result`` command: download one job's outputs for local inspection.

Batch jobs run with their working directory set to a per-job result directory on the shared
filesystem (see :mod:`hpcctl.generators.sbatch`), so "the job's results" is a well-defined set
of files: Slurm's stdout and stderr logs, plus everything the engine wrote to its cwd. This
command copies all of it to ``<out-dir>/<job id>/`` locally.
"""

import re
from pathlib import Path
from typing import Annotated, Final

import typer

from hpcctl import console
from hpcctl.commands.options import DryRunOption, StrictOption, resolve_dry_run
from hpcctl.config import REQUIRED_FOR_REMOTE, Settings, load_settings
from hpcctl.discovery import resolve_head_node
from hpcctl.errors import ClusterStateError, InvalidConfigError
from hpcctl.external import require_tools, run, scp_fetch_argv
from hpcctl.generators.sbatch import remote_results_dir

JOB_ID_FORM: Final[re.Pattern[str]] = re.compile(r"^\d{1,32}$")
"""Slurm job IDs are numeric; anything else must not reach a remote shell path."""

OutDirOption = Annotated[
    Path,
    typer.Option(help="Local directory results are downloaded into (one subdirectory per job)."),
]


def result(
    job_id: Annotated[str, typer.Argument(help="Numeric Slurm job ID, as printed by submit.")],
    dry_run: DryRunOption = False,
    out_dir: OutDirOption = Path("results"),
    strict: StrictOption = False,
) -> None:
    """Download a job's stdout, stderr, and engine-written result files.

    Fetches three things into ``<out-dir>/<job id>/``: the Slurm stdout log, the stderr log,
    and the contents of the job's result directory on the shared filesystem. A missing piece
    is a warning rather than an error -- a healthy job may have an empty stderr -- but finding
    nothing at all fails, because that means the job never ran or has not finished.
    """
    if not JOB_ID_FORM.match(job_id):
        raise InvalidConfigError(
            f"job id must be numeric, got {job_id!r}",
            hint="Use the ID printed by 'hpcctl submit' or shown by 'hpcctl status'.",
        )
    dry_run = resolve_dry_run(dry_run)
    settings = load_settings(live=not dry_run, strict=strict, required=REQUIRED_FOR_REMOTE)
    settings = resolve_head_node(settings, dry_run=dry_run)

    target = out_dir / job_id
    fetches = _fetches(settings, job_id, target)

    if dry_run:
        console.render_artifact(
            "download commands (bash)",
            "\n".join(console.format_command(argv) for _, argv in fetches),
            "bash",
        )
        console.render_notice(f"files would land in {target}/")
        console.render_placeholder_warning(settings)
        return

    require_tools("scp")
    target.mkdir(parents=True, exist_ok=True)
    fetched_anything = False
    for label, argv in fetches:
        completed = run(argv, dry_run=False, check=False)
        if completed is not None and completed.returncode == 0:
            fetched_anything = True
        else:
            console.render_warning(f"no {label} found for job {job_id}")
    if not fetched_anything:
        _remove_if_empty(target)
        raise ClusterStateError(
            f"nothing to download for job {job_id}",
            hint="Is the job still queued or running? Check 'hpcctl status'.",
        )
    console.out().print(_downloaded_table(target))


def _fetches(settings: Settings, job_id: str, target: Path) -> list[tuple[str, list[str]]]:
    """Build the three download commands for one job.

    The log fetches glob on ``*-<job id>`` because the log filenames embed the job *name*,
    which this command deliberately does not require the caller to remember.

    Args:
        settings: Resolved settings supplying SSH identity and remote layout.
        job_id: Validated numeric Slurm job ID.
        target: Local destination directory.

    Returns:
        ``(label, argv)`` pairs, in download order.
    """

    def fetch(remote: str, *, recursive: bool = False) -> list[str]:
        return scp_fetch_argv(
            key_path=settings.ssh_key_path,
            user=settings.ssh_user,
            host=settings.head_node_host,
            remote=remote,
            local=str(target),
            recursive=recursive,
        )

    return [
        ("stdout log", fetch(f"{settings.remote_dag_dir}/*-{job_id}.out")),
        ("stderr log", fetch(f"{settings.remote_dag_dir}/*-{job_id}.err")),
        (
            "engine result files",
            fetch(f"{remote_results_dir(settings, job_id)}/*", recursive=True),
        ),
    ]


def _downloaded_table(target: Path) -> object:
    """Build the summary table of everything now sitting in the local result directory.

    Args:
        target: Local directory the downloads landed in.

    Returns:
        A renderable rich table.
    """
    table = console.new_table(f"downloaded to {target}/", "file", "bytes")
    for path in sorted(p for p in target.rglob("*") if p.is_file()):
        table.add_row(path.relative_to(target).as_posix(), str(path.stat().st_size))
    return table


def _remove_if_empty(target: Path) -> None:
    """Remove the destination directory if nothing was downloaded into it.

    Args:
        target: Local directory created for the downloads.
    """
    try:
        target.rmdir()
    except OSError:
        pass
