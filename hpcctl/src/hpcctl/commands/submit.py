"""The ``submit`` command: validate a serialized DAG and hand it to Slurm."""

import dataclasses
import re
from pathlib import Path
from typing import Annotated

import typer

from hpcctl import console
from hpcctl.commands.options import DryRunOption, RawOption, StrictOption, resolve_dry_run
from hpcctl.config import REQUIRED_FOR_REMOTE, Settings, load_settings
from hpcctl.errors import ExternalCommandError, InvalidConfigError
from hpcctl.external import require_tools, run, scp_argv, ssh_argv
from hpcctl.generators.sbatch import remote_dag_path, remote_sbatch_path, render_sbatch
from hpcctl.validation import check_version_compatibility, load_schema, validate_dag_file

JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")
"""Slurm's confirmation line, from which the job ID is extracted."""

JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
"""Characters a job name may contain.

Deliberately the same pattern the contract pins on ``metadata.dag_id``, since that is where a
default job name comes from. The restriction is load-bearing in three places at once: the name
becomes a filename under the run directory, a path inside a remote command string that ``ssh``
hands to a shell, and literal text in the ``#SBATCH`` directive block.
"""

DagOption = Annotated[
    Path,
    typer.Option(exists=True, dir_okay=False, readable=True, help="Serialized DAG JSON file."),
]


def submit(
    dag: DagOption,
    dry_run: DryRunOption = True,
    validate_only: Annotated[
        bool, typer.Option("--validate-only", help="Validate the DAG and stop. Fully local.")
    ] = False,
    job_name: Annotated[str | None, typer.Option(help="Slurm job name.")] = None,
    nodes: Annotated[int | None, typer.Option(help="Override HPCCTL_NODES.")] = None,
    ntasks: Annotated[int | None, typer.Option(help="Override HPCCTL_NTASKS.")] = None,
    time_limit: Annotated[str | None, typer.Option(help="Override HPCCTL_TIME_LIMIT.")] = None,
    strict: StrictOption = False,
    raw: RawOption = False,
) -> None:
    """Validate a serialized DAG and submit it to Slurm as a batch job.

    Validation always runs first, in every mode: an invalid DAG must never reach the cluster, and
    ``--validate-only`` is the fastest useful thing this CLI can do without an AWS account.
    """
    dry_run = resolve_dry_run(dry_run)
    settings = load_settings(live=not dry_run, strict=strict, required=REQUIRED_FOR_REMOTE)

    document = validate_dag_file(dag, schema_path=settings.schema_path)
    warning = check_version_compatibility(document, load_schema(settings.schema_path))
    if warning:
        console.render_warning(warning)

    resolved_name = _checked_job_name(job_name or _default_job_name(document, dag))
    if not raw:
        console.render_notice(f"{dag} is a valid DAG ({len(document['nodes'])} node(s))")

    if validate_only:
        return

    effective = _with_overrides(settings, nodes=nodes, ntasks=ntasks, time_limit=time_limit)
    dag_remote = remote_dag_path(effective, dag.name)
    script = render_sbatch(effective, dag_remote_path=dag_remote, job_name=resolved_name)

    effective.run_dir.mkdir(parents=True, exist_ok=True)
    script_path = effective.run_dir / f"{resolved_name}.sbatch.generated"
    script_path.write_text(script, encoding="utf-8")

    script_remote = remote_sbatch_path(effective, resolved_name)
    copy_dag = scp_argv(
        key_path=effective.ssh_key_path,
        local=str(dag),
        user=effective.ssh_user,
        host=effective.head_node_host,
        remote=dag_remote,
    )
    copy_script = scp_argv(
        key_path=effective.ssh_key_path,
        local=str(script_path),
        user=effective.ssh_user,
        host=effective.head_node_host,
        remote=script_remote,
    )
    submit_cmd = ssh_argv(
        key_path=effective.ssh_key_path,
        user=effective.ssh_user,
        host=effective.head_node_host,
        remote_command=f"sbatch {script_remote}",
    )

    if dry_run:
        if raw:
            console.write_raw("sbatch", script)
            return
        console.render_artifact("batch script (bash)", script, "bash")
        console.render_notice(f"artifact written to {script_path}")
        console.render_artifact(
            "staging and submission (bash)",
            "\n".join(console.format_command(argv) for argv in (copy_dag, copy_script, submit_cmd)),
            "bash",
        )
        console.render_placeholder_warning(effective)
        return

    require_tools("scp", "ssh")
    run(copy_dag, dry_run=False)
    run(copy_script, dry_run=False)
    completed = run(submit_cmd, dry_run=False)
    if completed is None:  # pragma: no cover - only reachable in dry-run
        return
    match = JOB_ID_PATTERN.search(completed.stdout or "")
    if match is None:
        raise ExternalCommandError(
            "sbatch did not report a job ID",
            returncode=completed.returncode,
            stderr=(completed.stdout or "") + (completed.stderr or ""),
            hint="Check the batch script and the queue name on the head node.",
        )
    console.out().print(match.group(1))
    console.render_notice(f"submitted job {match.group(1)} as {resolved_name!r}")


def _with_overrides(
    settings: Settings,
    *,
    nodes: int | None,
    ntasks: int | None,
    time_limit: str | None,
) -> Settings:
    """Apply CLI overrides on top of environment-derived job geometry.

    Job geometry is what a user tunes per experiment, so the flags win over the environment.

    Args:
        settings: Environment-derived settings.
        nodes: Node-count override, or ``None`` to keep the configured value.
        ntasks: Task-count override, or ``None``.
        time_limit: Wall-clock override, or ``None``.

    Returns:
        A new settings value; the original is frozen and untouched.
    """
    return dataclasses.replace(
        settings,
        nodes=nodes if nodes is not None else settings.nodes,
        ntasks=ntasks if ntasks is not None else settings.ntasks,
        time_limit=time_limit if time_limit is not None else settings.time_limit,
    )


def _checked_job_name(name: str) -> str:
    """Reject a job name that cannot be used safely as a path, argument, or script text.

    The name reaches three places that each treat it differently, so one unvalidated value has
    three separate failure modes: ``../x`` escapes the run directory and writes the generated
    script somewhere else entirely; a name carrying shell metacharacters lands inside the
    ``sbatch <path>`` string that ``ssh`` executes through the remote shell; and an embedded
    newline injects extra lines into the ``#SBATCH`` block, silently changing the job's
    resources. Validating once, here, closes all three.

    Args:
        name: Job name from ``--job-name`` or derived from the DAG.

    Returns:
        The name, unchanged, when it is safe.

    Raises:
        InvalidConfigError: If the name is empty or contains anything outside
            ``[A-Za-z0-9_.-]``, or does not start with an alphanumeric.
    """
    if not JOB_NAME_PATTERN.match(name):
        raise InvalidConfigError(
            f"unusable job name {name!r}",
            hint=(
                "A job name must start with a letter or digit and contain only letters, "
                "digits, '_', '.', or '-'. Pass a different --job-name."
            ),
        )
    return name


def _default_job_name(document: dict[str, object], dag: Path) -> str:
    """Derive a job name from the DAG's own identifier.

    Args:
        document: The validated DAG document.
        dag: Path to the DAG file, used as a fallback.

    Returns:
        ``metadata.dag_id`` when present, otherwise the file stem.
    """
    metadata = document.get("metadata")
    if isinstance(metadata, dict):
        dag_id = metadata.get("dag_id")
        if isinstance(dag_id, str) and dag_id:
            return dag_id
    return dag.stem
