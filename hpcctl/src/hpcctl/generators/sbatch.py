"""Render the Slurm batch script for one DAG execution.

**Every ``#SBATCH`` directive must precede the first non-comment, non-blank line.** Slurm stops
scanning for directives at the first real command, so a directive placed after
``set -euo pipefail`` is *silently ignored* -- the job runs with defaults instead of failing,
which is far worse than an error. The directive block below is therefore built as one contiguous
list emitted before any command, and the test suite asserts the property positionally rather than
trusting the convention.

``--mpi=pmix`` matches ParallelCluster's Slurm build.
"""

from hpcctl.config import Settings

SBATCH_PREFIX: str = "#SBATCH "
"""Directive marker Slurm scans for."""


def sbatch_directives(settings: Settings, *, job_name: str) -> list[str]:
    """Build the ordered ``#SBATCH`` directive lines.

    Args:
        settings: Resolved settings supplying job geometry and output locations.
        job_name: Slurm job name, also used to key the output and error files.

    Returns:
        Directive lines, each already prefixed with ``#SBATCH ``.
    """
    log_stem = f"{settings.remote_dag_dir}/{job_name}"
    options = [
        f"--job-name={job_name}",
        f"--partition={settings.queue_name}",
        f"--nodes={settings.nodes}",
        f"--ntasks={settings.ntasks}",
        f"--time={settings.time_limit}",
        f"--output={log_stem}-%j.out",
        f"--error={log_stem}-%j.err",
    ]
    return [f"{SBATCH_PREFIX}{option}" for option in options]


def render_sbatch(settings: Settings, *, dag_remote_path: str, job_name: str) -> str:
    """Render the Slurm batch script for one DAG execution.

    Args:
        settings: Resolved settings supplying job geometry and remote paths.
        dag_remote_path: Absolute path of the serialized DAG on the shared filesystem.
        job_name: Slurm job name.

    Returns:
        The complete batch script, with every ``#SBATCH`` directive ahead of the first command.
    """
    lines = [
        "#!/bin/bash",
        *sbatch_directives(settings, job_name=job_name),
        "",
        "set -euo pipefail",
        "",
        # The job runs with its working directory set to a per-job result directory on the
        # shared filesystem, so anything the engine writes to its cwd is collected later by
        # `hpcctl result <job id>` without the engine needing to know about paths.
        f'RESULT_DIR="{settings.remote_dag_dir}/results/${{SLURM_JOB_ID}}"',
        'mkdir -p "${RESULT_DIR}"',
        'cd "${RESULT_DIR}"',
        "",
        'echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NUM_NODES} node(s), ${SLURM_NTASKS} task(s)"',
        # srun by absolute path: sbatch propagates the *submitting* environment into the job,
        # and hpcctl submits over a non-interactive ssh whose PATH lacks Slurm's bin directory.
        # Observed live as "srun: command not found" on an otherwise healthy job.
        #
        # --distribution=cyclic pins the placement the geometry relies on: with N+1 ranks on
        # N nodes, round-robin gives every node one worker and wraps the final rank -- the
        # scheduler, by engine convention the highest rank -- back onto the first node, so the
        # scheduler shares an instance with exactly one worker and no machine idles under it.
        f"{settings.slurm_bin}/srun --mpi=pmix --distribution=cyclic "
        f"{settings.engine_binary} --dag {dag_remote_path}",
        "",
    ]
    return "\n".join(lines)


def remote_dag_path(settings: Settings, dag_filename: str) -> str:
    """Return where a local DAG file will be staged on the shared filesystem.

    Args:
        settings: Resolved settings supplying the remote DAG directory.
        dag_filename: Base name of the local DAG file.

    Returns:
        The absolute remote path.
    """
    return f"{settings.remote_dag_dir}/{dag_filename}"


def remote_results_dir(settings: Settings, job_id: str) -> str:
    """Return the shared-filesystem directory one job's engine outputs land in.

    Must mirror the ``RESULT_DIR`` the batch script computes from ``SLURM_JOB_ID``: this is
    the read side of that convention, used by ``hpcctl result``.

    Args:
        settings: Resolved settings supplying the remote DAG directory.
        job_id: Numeric Slurm job ID.

    Returns:
        The absolute remote directory path.
    """
    return f"{settings.remote_dag_dir}/results/{job_id}"


def remote_sbatch_path(settings: Settings, job_name: str) -> str:
    """Return where the generated batch script will be staged on the shared filesystem.

    Args:
        settings: Resolved settings supplying the remote DAG directory.
        job_name: Slurm job name.

    Returns:
        The absolute remote path of the ``.sbatch.generated`` file.
    """
    return f"{settings.remote_dag_dir}/{job_name}.sbatch.generated"
