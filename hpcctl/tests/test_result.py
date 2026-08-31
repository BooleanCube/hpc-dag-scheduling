"""Tests for the ``result`` command and the per-job result-directory convention.

The convention has a write side and a read side that must agree byte-for-byte: the batch
script computes ``RESULT_DIR`` from ``SLURM_JOB_ID`` and runs the engine with that as its
working directory, and ``hpcctl result`` later reads ``remote_results_dir()`` for the same
job. A drift between the two would make results silently undownloadable.
"""

from pathlib import Path

from conftest import blank_env
from typer.testing import CliRunner

from hpcctl.cli import app
from hpcctl.config import Settings, load_settings
from hpcctl.exit_codes import ExitCode
from hpcctl.generators.sbatch import remote_results_dir, render_sbatch


class TestResultDirConvention:
    def script(self, settings: Settings) -> str:
        return render_sbatch(settings, dag_remote_path="/shared/dags/x.json", job_name="x")

    def test_batch_script_computes_the_result_dir_from_the_job_id(self, settings: Settings) -> None:
        assert 'RESULT_DIR="/shared/dags/results/${SLURM_JOB_ID}"' in self.script(settings)

    def test_batch_script_creates_and_enters_it_before_srun(self, settings: Settings) -> None:
        """The engine writes results to its cwd; the cwd must exist and be per-job."""
        script = self.script(settings)
        mkdir = script.index('mkdir -p "${RESULT_DIR}"')
        cd = script.index('cd "${RESULT_DIR}"')
        srun = script.index("/srun ")
        assert mkdir < cd < srun

    def test_read_side_matches_the_write_side(self, settings: Settings) -> None:
        """remote_results_dir() must resolve to the same path the script computes."""
        assert remote_results_dir(settings, "42") == "/shared/dags/results/42"
        assert (
            'RESULT_DIR="/shared/dags/results/${SLURM_JOB_ID}"'.replace("${SLURM_JOB_ID}", "42")
            == f'RESULT_DIR="{remote_results_dir(settings, "42")}"'
        )

    def test_convention_follows_the_remote_dag_dir(self) -> None:
        settings = load_settings(live=False)
        assert remote_results_dir(settings, "7").startswith(settings.remote_dag_dir)


class TestCli:
    def test_dry_run_needs_nothing(self, runner: CliRunner) -> None:
        """P1: dry-run completes offline with a completely empty environment."""
        result = runner.invoke(app, ["result", "--dry-run", "42"], env=blank_env())
        assert result.exit_code == 0, result.stdout + result.stderr

    def test_dry_run_prints_three_download_commands(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["result", "--dry-run", "42"], env=blank_env())
        assert result.stdout.count("scp ") == 3

    def test_fetches_logs_by_job_id_glob(self, runner: CliRunner) -> None:
        """The log filenames embed the job *name*, which the caller should not have to know."""
        result = runner.invoke(app, ["result", "--dry-run", "42"], env=blank_env())
        assert "*-42.out" in result.stdout
        assert "*-42.err" in result.stdout

    def test_fetches_the_engine_result_directory(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["result", "--dry-run", "42"], env=blank_env())
        assert "/shared/dags/results/42/*" in result.stdout

    def test_result_dir_fetch_is_recursive(self, runner: CliRunner) -> None:
        """Whitespace-squashed match: the rendered command wraps at terminal width."""
        result = runner.invoke(app, ["result", "--dry-run", "42"], env=blank_env())
        squashed = "".join(result.stdout.split())
        assert "-r" in squashed
        assert "results/42/*" in squashed

    def test_non_numeric_job_id_exits_config(self, runner: CliRunner) -> None:
        """The ID lands inside remote shell paths, so anything non-numeric is refused."""
        result = runner.invoke(app, ["result", "--dry-run", "42; rm -rf /"], env=blank_env())
        assert result.exit_code == ExitCode.CONFIG

    def test_dry_run_downloads_nothing(self, runner: CliRunner, tmp_path: Path) -> None:
        runner.invoke(app, ["result", "--dry-run", "42"], env=blank_env())
        assert not (tmp_path / "results").exists()

    def test_out_dir_is_configurable(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app, ["result", "--dry-run", "42", "--out-dir", "elsewhere"], env=blank_env()
        )
        assert "elsewhere/42" in result.stdout + result.stderr

    def test_kill_switch_defeats_execute(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app, ["result", "42", "--execute"], env=blank_env(HPCCTL_DRY_RUN="1")
        )
        assert result.exit_code == 0
        assert "HPCCTL_DRY_RUN" in result.stderr

    def test_uses_accept_new_host_key_policy(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["result", "--dry-run", "42"], env=blank_env())
        assert "StrictHostKeyChecking=accept-new" in result.stdout
        assert "StrictHostKeyChecking=no" not in result.stdout
