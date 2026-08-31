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
    recorder.install("cmake")
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

    def test_failure_surfaces_the_tools_own_diagnostic(
        self, runner: CliRunner, tools: Recorder
    ) -> None:
        """Surface the tool's own diagnostic.

        pcluster explains failures on stdout; hiding that sends the operator off to re-run
        the command by hand just to see why it failed (observed on first live boot).
        """
        tools.install(
            "pcluster", stdout='{"message": "Invalid cluster configuration."}', exit_code=1
        )
        result = runner.invoke(app, ["boot", "--execute"], env=blank_env(**LIVE_ENV))
        assert "Invalid cluster configuration." in result.stderr

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
    def test_builds_the_engine_before_syncing(
        self, runner: CliRunner, tools: Recorder, build_dir: Path
    ) -> None:
        """A deploy must never ship a binary older than the engine sources."""
        result = runner.invoke(
            app,
            ["deploy", "--execute", "--build-dir", str(build_dir)],
            env=blank_env(**LIVE_ENV),
        )
        assert result.exit_code == 0, result.stderr
        calls = tools.calls()
        configure = next(i for i, line in enumerate(calls) if line.startswith("cmake -S"))
        compile_ = next(i for i, line in enumerate(calls) if line.startswith("cmake --build"))
        sync = next(i for i, line in enumerate(calls) if line.startswith("rsync"))
        assert configure < compile_ < sync

    def test_build_targets_the_requested_build_dir(
        self, runner: CliRunner, tools: Recorder, build_dir: Path
    ) -> None:
        runner.invoke(
            app,
            ["deploy", "--execute", "--build-dir", str(build_dir)],
            env=blank_env(**LIVE_ENV),
        )
        compile_ = next(line for line in tools.calls() if line.startswith("cmake --build"))
        assert str(build_dir) in compile_

    def test_no_build_skips_cmake(
        self, runner: CliRunner, tools: Recorder, build_dir: Path
    ) -> None:
        result = runner.invoke(
            app,
            ["deploy", "--execute", "--no-build", "--build-dir", str(build_dir)],
            env=blank_env(**LIVE_ENV),
        )
        assert result.exit_code == 0, result.stderr
        assert not tools.called("cmake")
        assert tools.called("rsync")

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
        assert "ubuntu@203.0.113.10:/shared/engine/bin/" in call

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
            app, ["submit", "--execute", str(valid_dag)], env=blank_env(**LIVE_ENV)
        )
        assert result.exit_code == 0, result.stderr
        calls = tools.calls()
        assert sum(line.startswith("scp") for line in calls) == 2
        assert any(line.startswith("ssh") for line in calls)

    def test_creates_the_remote_dag_dir_before_copying(
        self, runner: CliRunner, tools: Recorder, valid_dag: Path
    ) -> None:
        """Create the remote DAG directory before copying into it.

        Without the mkdir the very first submit against a new cluster fails with
        "No such file or directory" (observed on first live run).
        """
        runner.invoke(app, ["submit", "--execute", str(valid_dag)], env=blank_env(**LIVE_ENV))
        calls = tools.calls()
        mkdir = next(i for i, line in enumerate(calls) if "mkdir -p" in line)
        first_scp = min(i for i, line in enumerate(calls) if line.startswith("scp"))
        assert "ssh" in calls[mkdir]
        assert mkdir < first_scp

    def test_copies_before_submitting(
        self, runner: CliRunner, tools: Recorder, valid_dag: Path
    ) -> None:
        runner.invoke(app, ["submit", "--execute", str(valid_dag)], env=blank_env(**LIVE_ENV))
        calls = tools.calls()
        last_scp = max(i for i, line in enumerate(calls) if line.startswith("scp"))
        # "/sbatch " (the executable), not "sbatch": the staged script's own filename ends in
        # ".sbatch.generated", so a bare substring also matches the scp lines.
        sbatch = next(i for i, line in enumerate(calls) if "/sbatch " in line)
        assert last_scp < sbatch

    def test_reports_the_parsed_job_id(
        self, runner: CliRunner, tools: Recorder, valid_dag: Path
    ) -> None:
        result = runner.invoke(
            app, ["submit", "--execute", str(valid_dag)], env=blank_env(**LIVE_ENV)
        )
        assert "12345" in result.stdout

    def test_unparseable_sbatch_output_exits_command_failed(
        self, runner: CliRunner, tools: Recorder, valid_dag: Path
    ) -> None:
        tools.install("ssh", stdout="something unexpected\n")
        result = runner.invoke(
            app, ["submit", "--execute", str(valid_dag)], env=blank_env(**LIVE_ENV)
        )
        assert result.exit_code == ExitCode.COMMAND_FAILED

    def test_invalid_dag_never_reaches_the_cluster(
        self, runner: CliRunner, tools: Recorder, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{oops", encoding="utf-8")
        result = runner.invoke(app, ["submit", "--execute", str(bad)], env=blank_env(**LIVE_ENV))
        assert result.exit_code == ExitCode.DAG_INVALID
        assert not tools.called("scp")
        assert not tools.called("ssh")

    def test_sbatch_targets_the_remote_script(
        self, runner: CliRunner, tools: Recorder, valid_dag: Path
    ) -> None:
        runner.invoke(app, ["submit", "--execute", str(valid_dag)], env=blank_env(**LIVE_ENV))
        ssh_call = next(line for line in tools.calls() if "/sbatch " in line)
        assert "/opt/slurm/bin/sbatch /shared/dags/bench-matmul-001.sbatch.generated" in ssh_call


class TestHeadNodeDiscovery:
    """Remote commands must work without HPCCTL_HEAD_NODE_HOST set.

    The address changes on every cluster re-boot, so requiring it in the environment made
    every second session fail with a stale IP. When unset, live commands ask
    ``pcluster describe-cluster`` and use the address it reports.
    """

    ENV_WITHOUT_HOST = {k: v for k, v in LIVE_ENV.items() if k != "HPCCTL_HEAD_NODE_HOST"}

    def test_deploy_discovers_the_head_node(
        self, runner: CliRunner, tools: Recorder, build_dir: Path
    ) -> None:
        result = runner.invoke(
            app,
            ["deploy", "--execute", "--build-dir", str(build_dir)],
            env=blank_env(**self.ENV_WITHOUT_HOST),
        )
        assert result.exit_code == 0, result.stderr
        rsync = next(line for line in tools.calls() if line.startswith("rsync"))
        assert "ubuntu@203.0.113.10:" in rsync

    def test_submit_discovers_the_head_node(
        self, runner: CliRunner, tools: Recorder, valid_dag: Path
    ) -> None:
        result = runner.invoke(
            app,
            ["submit", "--execute", str(valid_dag)],
            env=blank_env(**self.ENV_WITHOUT_HOST),
        )
        assert result.exit_code == 0, result.stderr
        scp = next(line for line in tools.calls() if line.startswith("scp"))
        assert "ubuntu@203.0.113.10:" in scp

    def test_status_reads_the_queue_without_a_configured_host(
        self, runner: CliRunner, tools: Recorder
    ) -> None:
        result = runner.invoke(app, ["status", "--execute"], env=blank_env(**self.ENV_WITHOUT_HOST))
        assert result.exit_code == 0, result.stderr
        assert any("squeue" in line for line in tools.calls())

    def test_an_explicit_host_is_never_second_guessed(
        self, runner: CliRunner, tools: Recorder, build_dir: Path
    ) -> None:
        """An operator-pinned address (e.g. a VPN-only private IP) must win over discovery."""
        result = runner.invoke(
            app,
            ["deploy", "--execute", "--build-dir", str(build_dir)],
            env=blank_env(**{**LIVE_ENV, "HPCCTL_HEAD_NODE_HOST": "10.0.0.99"}),
        )
        assert result.exit_code == 0, result.stderr
        assert not tools.called("pcluster")
        rsync = next(line for line in tools.calls() if line.startswith("rsync"))
        assert "ubuntu@10.0.0.99:" in rsync

    def test_a_headless_cluster_exits_cluster_state(
        self, runner: CliRunner, tools: Recorder, build_dir: Path
    ) -> None:
        """A creating or deleted cluster has no address; that is exit 8, not a stack trace."""
        tools.install(
            "pcluster",
            stdout=json.dumps({"clusterName": "x", "clusterStatus": "CREATE_IN_PROGRESS"}),
        )
        result = runner.invoke(
            app,
            ["deploy", "--execute", "--build-dir", str(build_dir)],
            env=blank_env(**self.ENV_WITHOUT_HOST),
        )
        assert result.exit_code == ExitCode.CLUSTER_STATE


class TestResultLive:
    def test_downloads_all_three_pieces(self, runner: CliRunner, tools: Recorder) -> None:
        result = runner.invoke(app, ["result", "42", "--execute"], env=blank_env(**LIVE_ENV))
        assert result.exit_code == 0, result.stderr
        scps = [line for line in tools.calls() if line.startswith("scp")]
        assert len(scps) == 3
        assert any("*-42.out" in line for line in scps)
        assert any("*-42.err" in line for line in scps)
        assert any("results/42/*" in line for line in scps)

    def test_lands_in_a_per_job_directory(
        self, runner: CliRunner, tools: Recorder, tmp_path: Path
    ) -> None:
        result = runner.invoke(app, ["result", "42", "--execute"], env=blank_env(**LIVE_ENV))
        assert result.exit_code == 0, result.stderr
        assert (tmp_path / "results" / "42").is_dir()

    def test_nothing_found_exits_cluster_state(
        self, runner: CliRunner, tools: Recorder, tmp_path: Path
    ) -> None:
        """All three fetches failing means the job never ran; that is an error, not silence."""
        tools.install("scp", exit_code=1)
        result = runner.invoke(app, ["result", "42", "--execute"], env=blank_env(**LIVE_ENV))
        assert result.exit_code == ExitCode.CLUSTER_STATE
        assert not (tmp_path / "results" / "42").exists()

    def test_a_missing_piece_is_a_warning_not_a_failure(
        self, runner: CliRunner, tools: Recorder
    ) -> None:
        """A healthy job may have an empty stderr and no result files; stdout alone is fine."""
        result = runner.invoke(app, ["result", "42", "--execute"], env=blank_env(**LIVE_ENV))
        assert result.exit_code == 0
