"""Tests for the ``--execute`` paths, driven by stand-in executables.

The live branches are where money gets spent, so "untestable without an AWS account" is not good
enough: ``pcluster``, ``aws``, ``ssh``, ``scp``, and ``rsync`` are replaced with shell scripts
that record their arguments. That exercises the real command construction, ordering, and output
parsing while still requiring no network, no credentials, and no real tooling.
"""

import json
import os
import shlex
import stat
from pathlib import Path

import pytest
from conftest import blank_env
from typer.testing import CliRunner

from hpcctl.cli import app
from hpcctl.exit_codes import ExitCode

DESCRIBE_PAYLOAD = {
    "clusterName": "hpc-dag-baseline",
    "clusterStatus": "CREATE_COMPLETE",
    "region": "us-east-1",
    "computeFleetStatus": "RUNNING",
    "headNode": {"publicIpAddress": "203.0.113.10"},
}

LIVE_ENV = {
    "AWS_REGION": "us-east-1",
    "HPCCTL_KEY_NAME": "kp",
    "HPCCTL_HEAD_SUBNET_ID": "subnet-aaaa",
    "HPCCTL_BOOTSTRAP_BUCKET": "bucket",
    "HPCCTL_HEAD_NODE_HOST": "203.0.113.10",
}


class Recorder:
    """A directory of stand-in executables that log how they were called."""

    def __init__(self, directory: Path, log: Path) -> None:
        """Set up the stand-in directory.

        Args:
            directory: Directory placed at the front of PATH.
            log: File each stand-in appends its argv to.
        """
        self.directory = directory
        self.log = log

    def install(self, name: str, *, stdout: str = "", exit_code: int = 0) -> None:
        """Create one stand-in executable.

        Args:
            name: Executable name.
            stdout: Text the stand-in prints.
            exit_code: Status the stand-in exits with.
        """
        path = self.directory / name
        payload = shlex.quote(stdout)
        path.write_text(
            "#!/bin/bash\n"
            f'printf "%s" "{name}" >> {shlex.quote(str(self.log))}\n'
            f'printf " %s" "$@" >> {shlex.quote(str(self.log))}\n'
            f'printf "\\n" >> {shlex.quote(str(self.log))}\n'
            f"printf '%s' {payload}\n"
            f"exit {exit_code}\n",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    def calls(self) -> list[str]:
        """Return every recorded invocation.

        Returns:
            One line per call, starting with the executable name.
        """
        if not self.log.exists():
            return []
        return [line for line in self.log.read_text(encoding="utf-8").splitlines() if line]

    def called(self, name: str) -> bool:
        """Report whether a stand-in was invoked.

        Args:
            name: Executable name.

        Returns:
            ``True`` if at least one invocation was recorded.
        """
        return any(line.startswith(name) for line in self.calls())


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Put recording stand-ins for every external tool at the front of PATH.

    Args:
        tmp_path: Scratch directory.
        monkeypatch: pytest's environment patcher.

    Returns:
        The recorder, with sensible default behaviour for each tool.
    """
    directory = tmp_path / "bin"
    directory.mkdir()
    recorder = Recorder(directory, tmp_path / "calls.log")
    recorder.install("aws")
    recorder.install("pcluster", stdout=json.dumps(DESCRIBE_PAYLOAD))
    recorder.install("rsync")
    recorder.install("scp")
    recorder.install("ssh", stdout="Submitted batch job 12345\n")
    monkeypatch.setenv("PATH", f"{directory}:{os.environ['PATH']}")
    return recorder


class TestBootLive:
    def test_exits_zero(self, runner: CliRunner, tools: Recorder) -> None:
        result = runner.invoke(app, ["boot", "--execute"], env=blank_env(**LIVE_ENV))
        assert result.exit_code == 0, result.stderr

    def test_uploads_before_creating(self, runner: CliRunner, tools: Recorder) -> None:
        """Each step is a precondition for the next: the config references the uploaded key."""
        runner.invoke(app, ["boot", "--execute"], env=blank_env(**LIVE_ENV))
        calls = tools.calls()
        upload = next(i for i, line in enumerate(calls) if line.startswith("aws"))
        create = next(i for i, line in enumerate(calls) if line.startswith("pcluster"))
        assert upload < create

    def test_uploads_to_the_content_addressed_key(self, runner: CliRunner, tools: Recorder) -> None:
        from hpcctl.generators.bootstrap import bootstrap_digest

        runner.invoke(app, ["boot", "--execute"], env=blank_env(**LIVE_ENV))
        upload = next(line for line in tools.calls() if line.startswith("aws"))
        assert f"install_engine_deps-{bootstrap_digest()[:8]}.sh" in upload

    def test_create_cluster_receives_the_written_config(
        self, runner: CliRunner, tools: Recorder, tmp_path: Path
    ) -> None:
        runner.invoke(app, ["boot", "--execute"], env=blank_env(**LIVE_ENV))
        create = next(line for line in tools.calls() if line.startswith("pcluster"))
        assert "--cluster-configuration" in create
        assert (tmp_path / ".hpcctl-run" / "hpc-dag-baseline-config.yaml").is_file()

    def test_failure_from_pcluster_exits_command_failed(
        self, runner: CliRunner, tools: Recorder
    ) -> None:
        tools.install("pcluster", stdout="boom", exit_code=1)
        result = runner.invoke(app, ["boot", "--execute"], env=blank_env(**LIVE_ENV))
        assert result.exit_code == ExitCode.COMMAND_FAILED

    def test_points_at_status_afterwards(self, runner: CliRunner, tools: Recorder) -> None:
        result = runner.invoke(app, ["boot", "--execute"], env=blank_env(**LIVE_ENV))
        assert "hpcctl status" in result.stderr


class TestStatusLive:
    def test_exits_zero_and_reports_the_cluster(self, runner: CliRunner, tools: Recorder) -> None:
        result = runner.invoke(app, ["status", "--execute"], env=blank_env(**LIVE_ENV))
        assert result.exit_code == 0, result.stderr
        assert "CREATE_COMPLETE" in result.stdout

    def test_shows_the_head_node_address(self, runner: CliRunner, tools: Recorder) -> None:
        result = runner.invoke(app, ["status", "--execute"], env=blank_env(**LIVE_ENV))
        assert "203.0.113.10" in result.stdout

    def test_queries_the_queue_over_ssh(self, runner: CliRunner, tools: Recorder) -> None:
        runner.invoke(app, ["status", "--execute"], env=blank_env(**LIVE_ENV))
        assert tools.called("ssh")

    def test_no_queue_skips_ssh(self, runner: CliRunner, tools: Recorder) -> None:
        runner.invoke(app, ["status", "--execute", "--no-queue"], env=blank_env(**LIVE_ENV))
        assert not tools.called("ssh")

    def test_failed_cluster_state_exits_eight(self, runner: CliRunner, tools: Recorder) -> None:
        tools.install(
            "pcluster", stdout=json.dumps({**DESCRIBE_PAYLOAD, "clusterStatus": "CREATE_FAILED"})
        )
        result = runner.invoke(app, ["status", "--execute"], env=blank_env(**LIVE_ENV))
        assert result.exit_code == ExitCode.CLUSTER_STATE

    def test_empty_describe_output_exits_eight(self, runner: CliRunner, tools: Recorder) -> None:
        tools.install("pcluster", stdout="")
        result = runner.invoke(app, ["status", "--execute"], env=blank_env(**LIVE_ENV))
        assert result.exit_code == ExitCode.CLUSTER_STATE

    def test_unparseable_describe_output_exits_eight(
        self, runner: CliRunner, tools: Recorder
    ) -> None:
        tools.install("pcluster", stdout="not json at all")
        result = runner.invoke(app, ["status", "--execute"], env=blank_env(**LIVE_ENV))
        assert result.exit_code == ExitCode.CLUSTER_STATE

    def test_unreachable_head_node_degrades_rather_than_failing(
        self, runner: CliRunner, tools: Recorder
    ) -> None:
        """A cluster that is still creating has no reachable head node; that is normal."""
        tools.install("ssh", stdout="", exit_code=255)
        result = runner.invoke(app, ["status", "--execute"], env=blank_env(**LIVE_ENV))
        assert result.exit_code == 0
        assert "queue unavailable" in result.stderr
        assert "CREATE_COMPLETE" in result.stdout

    def test_queue_rows_are_tabulated(self, runner: CliRunner, tools: Recorder) -> None:
        tools.install(
            "ssh",
            stdout="JOBID NAME STATE NODES TIME\n42 bench-matmul RUNNING 2 00:01:15\n",
        )
        result = runner.invoke(app, ["status", "--execute"], env=blank_env(**LIVE_ENV))
        assert "bench-matmul" in result.stdout
        assert "RUNNING" in result.stdout


class TestDestroyLive:
    def test_yes_deletes_without_prompting(self, runner: CliRunner, tools: Recorder) -> None:
        result = runner.invoke(app, ["destroy", "--execute", "--yes"], env=blank_env(**LIVE_ENV))
        assert result.exit_code == 0, result.stderr
        assert tools.called("pcluster")

    def test_issues_delete_cluster(self, runner: CliRunner, tools: Recorder) -> None:
        runner.invoke(app, ["destroy", "--execute", "--yes"], env=blank_env(**LIVE_ENV))
        call = next(line for line in tools.calls() if line.startswith("pcluster"))
        assert "delete-cluster" in call
        assert "hpc-dag-baseline" in call

    def test_aborting_deletes_nothing(self, runner: CliRunner, tools: Recorder) -> None:
        """The confirmation gate must run before pcluster is ever invoked."""
        result = runner.invoke(app, ["destroy", "--execute"], env=blank_env(**LIVE_ENV))
        assert result.exit_code == ExitCode.ABORTED
        assert not tools.called("pcluster")


class TestDeployLive:
    def test_syncs_with_rsync(self, runner: CliRunner, tools: Recorder, build_dir: Path) -> None:
        result = runner.invoke(
            app,
            ["deploy", "--execute", "--build-dir", str(build_dir)],
            env=blank_env(**LIVE_ENV),
        )
        assert result.exit_code == 0, result.stderr
        assert tools.called("rsync")

    def test_targets_the_shared_filesystem(
        self, runner: CliRunner, tools: Recorder, build_dir: Path
    ) -> None:
        runner.invoke(
            app,
            ["deploy", "--execute", "--build-dir", str(build_dir)],
            env=blank_env(**LIVE_ENV),
        )
        call = next(line for line in tools.calls() if line.startswith("rsync"))
        assert "ubuntu@203.0.113.10:/shared/engine/" in call

    def test_unbuilt_engine_fails_before_touching_the_network(
        self, runner: CliRunner, tools: Recorder, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            ["deploy", "--execute", "--build-dir", str(tmp_path / "absent")],
            env=blank_env(**LIVE_ENV),
        )
        assert result.exit_code == ExitCode.CONFIG
        assert not tools.called("rsync")


class TestSubmitLive:
    def test_stages_then_submits(self, runner: CliRunner, tools: Recorder, valid_dag: Path) -> None:
        result = runner.invoke(
            app, ["submit", "--execute", "--dag", str(valid_dag)], env=blank_env(**LIVE_ENV)
        )
        assert result.exit_code == 0, result.stderr
        calls = tools.calls()
        assert sum(line.startswith("scp") for line in calls) == 2
        assert any(line.startswith("ssh") for line in calls)

    def test_copies_before_submitting(
        self, runner: CliRunner, tools: Recorder, valid_dag: Path
    ) -> None:
        runner.invoke(
            app, ["submit", "--execute", "--dag", str(valid_dag)], env=blank_env(**LIVE_ENV)
        )
        calls = tools.calls()
        last_scp = max(i for i, line in enumerate(calls) if line.startswith("scp"))
        first_ssh = min(i for i, line in enumerate(calls) if line.startswith("ssh"))
        assert last_scp < first_ssh

    def test_reports_the_parsed_job_id(
        self, runner: CliRunner, tools: Recorder, valid_dag: Path
    ) -> None:
        result = runner.invoke(
            app, ["submit", "--execute", "--dag", str(valid_dag)], env=blank_env(**LIVE_ENV)
        )
        assert "12345" in result.stdout

    def test_unparseable_sbatch_output_exits_command_failed(
        self, runner: CliRunner, tools: Recorder, valid_dag: Path
    ) -> None:
        tools.install("ssh", stdout="something unexpected\n")
        result = runner.invoke(
            app, ["submit", "--execute", "--dag", str(valid_dag)], env=blank_env(**LIVE_ENV)
        )
        assert result.exit_code == ExitCode.COMMAND_FAILED

    def test_invalid_dag_never_reaches_the_cluster(
        self, runner: CliRunner, tools: Recorder, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{oops", encoding="utf-8")
        result = runner.invoke(
            app, ["submit", "--execute", "--dag", str(bad)], env=blank_env(**LIVE_ENV)
        )
        assert result.exit_code == ExitCode.DAG_INVALID
        assert not tools.called("scp")
        assert not tools.called("ssh")

    def test_sbatch_targets_the_remote_script(
        self, runner: CliRunner, tools: Recorder, valid_dag: Path
    ) -> None:
        runner.invoke(
            app, ["submit", "--execute", "--dag", str(valid_dag)], env=blank_env(**LIVE_ENV)
        )
        ssh_call = next(line for line in tools.calls() if line.startswith("ssh"))
        assert "sbatch /shared/dags/bench-matmul-001.sbatch.generated" in ssh_call
