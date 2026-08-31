"""The ``submit`` command: validate a serialized DAG and hand it to Slurm."""

import dataclasses
import re
import sys
from pathlib import Path
from typing import Annotated

import typer

from hpcctl import console
from hpcctl.commands.options import DryRunOption, RawOption, StrictOption, resolve_dry_run
from hpcctl.config import REQUIRED_FOR_REMOTE, Settings, load_settings
from hpcctl.discovery import resolve_head_node
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

DagArgument = Annotated[
    Path,
    typer.Argument(
        exists=True,
        dir_okay=False,
        readable=True,
        help="A Python case script that builds a DAG (compiled automatically), "
        "or an already-serialized DAG JSON file.",
        show_default=False,
    ),
]


def submit(
    dag: DagArgument,
    dry_run: DryRunOption = False,
    validate_only: Annotated[
        bool, typer.Option("--validate-only", help="Validate the DAG and stop. Fully local.")
    ] = False,
    job_name: Annotated[str | None, typer.Option(help="Slurm job name.")] = None,
    nodes: Annotated[
        int | None,
        typer.Option(
            "--nodes",
            "-n",
            help="Compute instances to run on, within the queue's node bounds (default 1). "
            "MPI ranks default to this plus one: the scheduler rank shares an instance with "
            "one worker so no machine sits idle.",
        ),
    ] = None,
    ntasks: Annotated[
        int | None, typer.Option(help="Override the derived rank count (nodes + 1).")
    ] = None,
    time_limit: Annotated[str | None, typer.Option(help="Override HPCCTL_TIME_LIMIT.")] = None,
    strict: StrictOption = False,
    raw: RawOption = False,
) -> None:
    """Compile and validate a DAG, then submit it to Slurm as a batch job.

    A ``.py`` argument is a case script: it is run first (see :func:`_compile_case`) and the
    document it writes is what proceeds. Anything else is treated as an already-serialized
    DAG JSON file.

    Validation always runs next, in every mode: an invalid DAG must never reach the cluster, and
    ``--validate-only`` is the fastest useful thing this CLI can do without an AWS account.

    Geometry: ``-n N`` requests N compute instances and N+1 MPI ranks. Slurm's cyclic placement
    gives every instance one worker and wraps the highest rank (the scheduler, by engine
    convention) onto the first instance, so the scheduler shares a machine with exactly one
    worker rather than wasting a whole instance.
    """
    dry_run = resolve_dry_run(dry_run)
    # --validate-only is a purely local operation, so it must not demand live-only
    # configuration (or discovery) even though the command now defaults to --execute.
    settings = load_settings(
        live=not dry_run and not validate_only, strict=strict, required=REQUIRED_FOR_REMOTE
    )

    if dag.suffix == ".py":
        dag = _compile_case(dag, settings, raw=raw)

    document = validate_dag_file(dag, schema_path=settings.schema_path)
    warning = check_version_compatibility(document, load_schema(settings.schema_path))
    if warning:
        console.render_warning(warning)

    resolved_name = _checked_job_name(job_name or _default_job_name(document, dag))
    if not raw:
        console.render_notice(f"{dag} is a valid DAG ({len(document['nodes'])} node(s))")

    if validate_only:
        return

    settings = resolve_head_node(settings, dry_run=dry_run)
    effective = _with_overrides(settings, nodes=nodes, ntasks=ntasks, time_limit=time_limit)
    _checked_nodes(effective)
    dag_remote = remote_dag_path(effective, dag.name)
    script = render_sbatch(effective, dag_remote_path=dag_remote, job_name=resolved_name)

    effective.run_dir.mkdir(parents=True, exist_ok=True)
    script_path = effective.run_dir / f"{resolved_name}.sbatch.generated"
    script_path.write_text(script, encoding="utf-8")

    script_remote = remote_sbatch_path(effective, resolved_name)
    # scp cannot create the destination directory, and a freshly booted cluster's shared
    # volume is empty -- without this the very first submit fails with "No such file or
    # directory" (observed on first live run).
    make_dir = ssh_argv(
        key_path=effective.ssh_key_path,
        user=effective.ssh_user,
        host=effective.head_node_host,
        remote_command=f"mkdir -p {effective.remote_dag_dir}",
    )
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
        remote_command=f"{effective.slurm_bin}/sbatch {script_remote}",
    )

    if dry_run:
        if raw:
            console.write_raw("sbatch", script)
            return
        console.render_artifact("batch script (bash)", script, "bash")
        console.render_notice(f"artifact written to {script_path}")
        console.render_artifact(
            "staging and submission (bash)",
            "\n".join(
                console.format_command(argv)
                for argv in (make_dir, copy_dag, copy_script, submit_cmd)
            ),
            "bash",
        )
        console.render_placeholder_warning(effective)
        return

    require_tools("scp", "ssh")
    run(make_dir, dry_run=False)
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


def _compile_case(script: Path, settings: Settings, *, raw: bool) -> Path:
    """Run a Python case script and return the DAG document it wrote.

    The contract (implemented by ``vmath.emit``): the script is executed with the destination
    path as its first argument and must serialize exactly one graph there. Running the script
    in a subprocess rather than importing it keeps hpcctl on the right side of its own design
    rule -- the CLI validates serialized documents and never imports the builder.

    Compilation happens in every mode, dry-run included: it is local and free, and neither
    validation nor the sbatch preview is possible without the document. It is the one
    deliberate exception to "dry-run executes nothing" -- the interpreter running this very
    process, never a remote or AWS-touching tool.

    Args:
        script: The case script.
        settings: Resolved settings supplying the run directory.
        raw: Suppress the compilation notice, keeping ``--raw`` output byte-exact.

    Returns:
        Path of the compiled document, ``<run_dir>/<script stem>.json``.

    Raises:
        ExternalCommandError: If the script exits non-zero (a ``DagBuildError`` traceback
            lands on its stderr and is surfaced).
        InvalidConfigError: If the script exits zero without writing the document.
    """
    settings.run_dir.mkdir(parents=True, exist_ok=True)
    target = settings.run_dir / f"{script.stem}.json"
    run([sys.executable, str(script), str(target)], dry_run=False, capture=True)
    if not target.is_file():
        raise InvalidConfigError(
            f"case script wrote no DAG document: {script}",
            hint=(
                "A case script must end with vmath.emit(graph), which writes to the path "
                "hpcctl passes as the script's first argument."
            ),
        )
    if not raw:
        console.render_notice(f"compiled {script} -> {target}")
    return target


def _with_overrides(
    settings: Settings,
    *,
    nodes: int | None,
    ntasks: int | None,
    time_limit: str | None,
) -> Settings:
    """Apply CLI overrides on top of environment-derived job geometry.

    Job geometry is what a user tunes per experiment, so the flags win over the environment.
    The rank count is a *function of the instance count* — one scheduler plus one worker per
    instance, ``N + 1`` ranks on ``N`` nodes — so overriding ``-n`` re-derives it. An explicit
    ``--ntasks`` wins over the derivation for irregular experiments.

    Args:
        settings: Environment-derived settings.
        nodes: Instance-count override, or ``None`` to keep the configured value.
        ntasks: Rank-count override, or ``None`` to derive ``nodes + 1``.
        time_limit: Wall-clock override, or ``None``.

    Returns:
        A new settings value; the original is frozen and untouched.
    """
    effective_nodes = nodes if nodes is not None else settings.nodes
    return dataclasses.replace(
        settings,
        nodes=effective_nodes,
        ntasks=ntasks if ntasks is not None else effective_nodes + 1,
        time_limit=time_limit if time_limit is not None else settings.time_limit,
    )


def _checked_nodes(settings: Settings) -> None:
    """Reject an instance count the cluster's queue cannot satisfy.

    Slurm would accept a request beyond ``MaxCount`` and leave the job pending forever waiting
    for instances that will never scale out, so the mismatch is cheapest to catch here. The
    floor is at least 1 regardless of ``HPCCTL_MIN_NODES``: a MinCount of 0 is a scale-to-zero
    idle policy, not a valid job size.

    Args:
        settings: Effective settings, after CLI overrides.

    Raises:
        InvalidConfigError: If the count falls outside the queue's node bounds.
    """
    floor = max(settings.min_nodes, 1)
    if not floor <= settings.nodes <= settings.max_nodes:
        raise InvalidConfigError(
            f"requested {settings.nodes} node(s), but the queue allows "
            f"{floor} to {settings.max_nodes}",
            hint=(
                "Pass -n within the queue's bounds, or change HPCCTL_MIN_NODES / "
                "HPCCTL_MAX_NODES and re-boot the cluster."
            ),
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
