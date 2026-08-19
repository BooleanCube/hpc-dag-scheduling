"""Tests for the generated Slurm batch script.

The directive-placement test is the important one. Slurm stops scanning for ``#SBATCH`` lines at
the first real command, so a directive after ``set -euo pipefail`` is *silently ignored* and the
job runs with defaults. That failure produces no error anywhere, so it is asserted positionally
rather than left to convention.
"""

import dataclasses
import subprocess
from pathlib import Path

import pytest

from hpcctl.config import Settings, load_settings
from hpcctl.generators.sbatch import (
    SBATCH_PREFIX,
    remote_dag_path,
    remote_sbatch_path,
    render_sbatch,
    sbatch_directives,
)

JOB = "bench-matmul-001"


@pytest.fixture
def script(settings: Settings) -> str:
    """Render a batch script from default settings.

    Args:
        settings: Dry-run settings from an empty environment.

    Returns:
        The rendered script text.
    """
    return render_sbatch(settings, dag_remote_path=f"/shared/dags/{JOB}.json", job_name=JOB)


def first_command_index(lines: list[str]) -> int:
    """Return the index of the first non-comment, non-blank line.

    Args:
        lines: The script split into lines.

    Returns:
        Index of the first real command.
    """
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return index
    raise AssertionError("script contains no commands")


class TestSyntax:
    def test_passes_bash_syntax_check(self, script: str, tmp_path: Path) -> None:
        path = tmp_path / "job.sbatch"
        path.write_text(script, encoding="utf-8")
        result = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_starts_with_a_shebang(self, script: str) -> None:
        assert script.startswith("#!/bin/bash\n")

    def test_sets_strict_mode(self, script: str) -> None:
        assert "set -euo pipefail" in script

    def test_ends_with_a_newline(self, script: str) -> None:
        assert script.endswith("\n")


class TestDirectivePlacement:
    """Every #SBATCH line must precede the first real command."""

    def test_all_directives_come_before_the_first_command(self, script: str) -> None:
        lines = script.splitlines()
        boundary = first_command_index(lines)
        for index, line in enumerate(lines):
            if line.startswith(SBATCH_PREFIX):
                assert index < boundary, f"directive at line {index} is after the first command"

    def test_the_first_command_is_set_euo_pipefail(self, script: str) -> None:
        lines = script.splitlines()
        assert lines[first_command_index(lines)].strip() == "set -euo pipefail"

    def test_directives_are_contiguous(self, script: str) -> None:
        """A blank line between directives would end Slurm's scan early."""
        lines = script.splitlines()
        indices = [i for i, line in enumerate(lines) if line.startswith(SBATCH_PREFIX)]
        assert indices == list(range(indices[0], indices[-1] + 1))

    def test_directives_start_immediately_after_the_shebang(self, script: str) -> None:
        lines = script.splitlines()
        assert lines[1].startswith(SBATCH_PREFIX)

    def test_no_directive_appears_after_srun(self, script: str) -> None:
        lines = script.splitlines()
        srun_index = next(i for i, line in enumerate(lines) if line.startswith("srun"))
        assert not any(line.startswith(SBATCH_PREFIX) for line in lines[srun_index:])


class TestDirectiveContent:
    def test_every_required_directive_is_present(self, script: str) -> None:
        for option in (
            "--job-name=",
            "--partition=",
            "--nodes=",
            "--ntasks=",
            "--time=",
            "--output=",
            "--error=",
        ):
            assert f"{SBATCH_PREFIX}{option}" in script

    def test_job_name(self, script: str) -> None:
        assert f"{SBATCH_PREFIX}--job-name={JOB}" in script

    def test_partition_comes_from_the_queue_name(self, settings: Settings) -> None:
        directives = sbatch_directives(settings, job_name=JOB)
        assert f"{SBATCH_PREFIX}--partition=compute" in directives

    def test_defaults_from_the_environment(self, script: str) -> None:
        assert f"{SBATCH_PREFIX}--nodes=2" in script
        assert f"{SBATCH_PREFIX}--ntasks=4" in script
        assert f"{SBATCH_PREFIX}--time=00:30:00" in script

    def test_log_paths_land_in_the_remote_dag_dir(self, script: str) -> None:
        assert f"{SBATCH_PREFIX}--output=/shared/dags/{JOB}-%j.out" in script
        assert f"{SBATCH_PREFIX}--error=/shared/dags/{JOB}-%j.err" in script

    def test_log_paths_include_the_job_id_token(self, script: str) -> None:
        """%j keeps concurrent runs of one DAG from overwriting each other's logs."""
        assert "-%j.out" in script
        assert "-%j.err" in script


class TestOverrides:
    def test_nodes_ntasks_and_time_reflect_overrides(self, settings: Settings) -> None:
        overridden = dataclasses.replace(settings, nodes=8, ntasks=64, time_limit="02:00:00")
        script = render_sbatch(overridden, dag_remote_path="/shared/dags/x.json", job_name="x")
        assert f"{SBATCH_PREFIX}--nodes=8" in script
        assert f"{SBATCH_PREFIX}--ntasks=64" in script
        assert f"{SBATCH_PREFIX}--time=02:00:00" in script

    def test_queue_name_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_QUEUE_NAME", "gpu")
        script = render_sbatch(
            load_settings(live=False), dag_remote_path="/shared/dags/x.json", job_name="x"
        )
        assert f"{SBATCH_PREFIX}--partition=gpu" in script


class TestExecution:
    def test_uses_srun_with_pmix(self, script: str) -> None:
        assert "srun --mpi=pmix " in script

    def test_invokes_the_resolved_engine_binary(self, script: str) -> None:
        assert "/shared/engine/bin/engine" in script

    def test_passes_the_remote_dag_path(self, script: str) -> None:
        assert f"--dag /shared/dags/{JOB}.json" in script

    def test_engine_binary_honours_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_ENGINE_BINARY", "/opt/bin/engine")
        script = render_sbatch(load_settings(live=False), dag_remote_path="/d/x.json", job_name="x")
        assert "srun --mpi=pmix /opt/bin/engine --dag /d/x.json" in script

    def test_shared_dir_propagates_to_every_remote_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HPCCTL_SHARED_DIR", "/mnt/fsx")
        s = load_settings(live=False)
        script = render_sbatch(s, dag_remote_path=remote_dag_path(s, "x.json"), job_name="x")
        assert "/mnt/fsx/engine/bin/engine" in script
        assert "/mnt/fsx/dags/x.json" in script
        assert f"{SBATCH_PREFIX}--output=/mnt/fsx/dags/x-%j.out" in script

    def test_reports_geometry_at_runtime(self, script: str) -> None:
        assert "${SLURM_JOB_ID}" in script
        assert "${SLURM_NTASKS}" in script


class TestRemotePaths:
    def test_dag_path_keeps_the_local_filename(self, settings: Settings) -> None:
        assert remote_dag_path(settings, "abc.json") == "/shared/dags/abc.json"

    def test_sbatch_path_uses_the_ignored_extension(self, settings: Settings) -> None:
        """*.sbatch.generated is already covered by the root .gitignore."""
        assert remote_sbatch_path(settings, JOB) == f"/shared/dags/{JOB}.sbatch.generated"
