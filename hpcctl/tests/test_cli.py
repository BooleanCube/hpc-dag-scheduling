"""End-to-end CLI tests.

:class:`TestP1DryRunNeedsNothing` is the most important class in the suite. It asserts the
governing constraint of the whole design: every dry-run completes with no AWS credentials, no
``pcluster``, no ``aws``, no network, and no environment variables set. If that ever breaks, the
CLI becomes untestable until the AWS account exists, which is the outcome the architecture is
built to avoid.

No test here may require network, credentials, ``pcluster``, or ``aws``.
"""

import json
from pathlib import Path

import pytest
from conftest import blank_env, valid_dag_document
from typer.testing import CliRunner

from hpcctl.cli import app
from hpcctl.exit_codes import ExitCode

AWS_COMMANDS = ["boot", "deploy", "submit", "status", "destroy"]


class TestHelpAndVersion:
    def test_version_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["version"], env=blank_env())
        assert result.exit_code == 0
        assert result.stdout.strip()

    def test_bare_invocation_shows_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, [], env=blank_env())
        assert "hpcctl" in result.stdout

    @pytest.mark.parametrize("command", AWS_COMMANDS)
    def test_every_command_documents_dry_run(self, command: str, runner: CliRunner) -> None:
        result = runner.invoke(app, [command, "--help"], env=blank_env())
        assert result.exit_code == 0
        assert "--execute" in result.stdout

    def test_unknown_command_exits_usage(self, runner: CliRunner) -> None:
        """Exit 2 is reserved for Typer, which is why ExitCode never assigns it."""
        result = runner.invoke(app, ["nope"], env=blank_env())
        assert result.exit_code == ExitCode.USAGE


class TestP1DryRunNeedsNothing:
    """Dry-run must work offline with a completely empty environment."""

    def test_boot(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["boot"], env=blank_env())
        assert result.exit_code == 0, result.stdout + result.stderr

    def test_status(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["status"], env=blank_env())
        assert result.exit_code == 0, result.stdout + result.stderr

    def test_destroy(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["destroy"], env=blank_env())
        assert result.exit_code == 0, result.stdout + result.stderr

    def test_deploy(self, runner: CliRunner, build_dir: Path) -> None:
        result = runner.invoke(app, ["deploy", "--build-dir", str(build_dir)], env=blank_env())
        assert result.exit_code == 0, result.stdout + result.stderr

    def test_submit(self, runner: CliRunner, valid_dag: Path) -> None:
        result = runner.invoke(app, ["submit", "--dag", str(valid_dag)], env=blank_env())
        assert result.exit_code == 0, result.stdout + result.stderr

    def test_no_aws_tooling_is_installed(self) -> None:
        """Guards the premise: if pcluster appears on this box, these tests prove less."""
        import shutil

        assert shutil.which("pcluster") is None
        assert shutil.which("aws") is None

    def test_boot_warns_about_placeholders(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["boot"], env=blank_env())
        assert "HPCCTL_KEY_NAME" in result.stderr

    def test_boot_emits_all_three_artifacts(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["boot"], env=blank_env())
        assert "1/3" in result.stdout
        assert "2/3" in result.stdout
        assert "3/3" in result.stdout

    def test_boot_writes_artifacts_to_the_run_dir(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(app, ["boot"], env=blank_env())
        assert result.exit_code == 0
        written = sorted(p.name for p in (tmp_path / ".hpcctl-run").iterdir())
        assert "hpc-dag-baseline-config.yaml" in written
        assert any(name.startswith("install_engine_deps-") for name in written)

    def test_emit_dir_redirects_artifacts(self, runner: CliRunner, tmp_path: Path) -> None:
        target = tmp_path / "elsewhere"
        result = runner.invoke(app, ["boot", "--emit-dir", str(target)], env=blank_env())
        assert result.exit_code == 0
        assert (target / "hpc-dag-baseline-config.yaml").is_file()


class TestStrictFlag:
    @pytest.mark.parametrize("command", ["boot", "status", "destroy"])
    def test_strict_turns_placeholders_into_failure(self, command: str, runner: CliRunner) -> None:
        result = runner.invoke(app, [command, "--strict"], env=blank_env())
        assert result.exit_code == ExitCode.CONFIG

    def test_strict_passes_with_a_complete_environment(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app,
            ["boot", "--strict"],
            env=blank_env(
                AWS_REGION="us-east-1",
                HPCCTL_KEY_NAME="kp",
                HPCCTL_HEAD_SUBNET_ID="subnet-aaaa",
                HPCCTL_BOOTSTRAP_BUCKET="bucket",
            ),
        )
        assert result.exit_code == 0, result.stderr


class TestExecuteWithoutTools:
    """--execute must fail on the missing tool (5), never on a stray exception (1)."""

    def test_boot_execute_exits_tool_missing(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app,
            ["boot", "--execute"],
            env=blank_env(
                AWS_REGION="us-east-1",
                HPCCTL_KEY_NAME="kp",
                HPCCTL_HEAD_SUBNET_ID="subnet-aaaa",
                HPCCTL_BOOTSTRAP_BUCKET="bucket",
            ),
        )
        assert result.exit_code == ExitCode.TOOL_MISSING

    def test_message_names_the_missing_tools(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app,
            ["boot", "--execute"],
            env=blank_env(
                AWS_REGION="us-east-1",
                HPCCTL_KEY_NAME="kp",
                HPCCTL_HEAD_SUBNET_ID="subnet-aaaa",
                HPCCTL_BOOTSTRAP_BUCKET="bucket",
            ),
        )
        assert "pcluster" in result.stderr

    def test_status_execute_exits_tool_missing(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app,
            ["status", "--execute"],
            env=blank_env(AWS_REGION="us-east-1", HPCCTL_HEAD_NODE_HOST="1.2.3.4"),
        )
        assert result.exit_code == ExitCode.TOOL_MISSING

    def test_destroy_execute_exits_tool_missing_after_confirmation(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app,
            ["destroy", "--execute", "--yes"],
            env=blank_env(AWS_REGION="us-east-1"),
        )
        assert result.exit_code == ExitCode.TOOL_MISSING

    def test_boot_execute_without_config_exits_config_first(self, runner: CliRunner) -> None:
        """Config resolution precedes tool discovery: fail before spending anything."""
        result = runner.invoke(app, ["boot", "--execute"], env=blank_env())
        assert result.exit_code == ExitCode.CONFIG


class TestDryRunKillSwitch:
    def test_hpcctl_dry_run_defeats_execute(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["boot", "--execute"], env=blank_env(HPCCTL_DRY_RUN="1"))
        assert result.exit_code == 0
        assert "HPCCTL_DRY_RUN" in result.stderr

    def test_kill_switch_still_produces_artifacts(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["boot", "--execute"], env=blank_env(HPCCTL_DRY_RUN="1"))
        assert "1/3" in result.stdout

    @pytest.mark.parametrize("command", ["boot", "status", "destroy"])
    def test_applies_to_every_command(self, command: str, runner: CliRunner) -> None:
        result = runner.invoke(app, [command, "--execute"], env=blank_env(HPCCTL_DRY_RUN="yes"))
        assert result.exit_code == 0

    def test_empty_value_does_not_engage(self, runner: CliRunner) -> None:
        """An empty HPCCTL_DRY_RUN must not silently block a deliberate --execute."""
        result = runner.invoke(app, ["boot", "--execute"], env=blank_env(HPCCTL_DRY_RUN=""))
        assert result.exit_code == ExitCode.CONFIG


class TestDestroyConfirmation:
    def test_dry_run_never_prompts(self, runner: CliRunner) -> None:
        """Prompting in dry-run would train the reflex this UX exists to prevent."""
        result = runner.invoke(app, ["destroy"], env=blank_env())
        assert result.exit_code == 0
        assert "Type the cluster name" not in result.stdout

    def test_wrong_name_aborts(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app,
            ["destroy", "--execute"],
            input="wrong-name\n",
            env=blank_env(AWS_REGION="us-east-1"),
        )
        assert result.exit_code == ExitCode.ABORTED

    def test_piping_even_the_correct_name_aborts(self, runner: CliRunner) -> None:
        """--yes is the supported automation path; a pipe is never accepted as consent.

        CliRunner's stdin is never a TTY, so this also documents why the matching-name path is
        covered by unit tests against ``_confirm`` rather than through the CLI.
        """
        result = runner.invoke(
            app,
            ["destroy", "--execute"],
            input="hpc-dag-baseline\n",
            env=blank_env(AWS_REGION="us-east-1"),
        )
        assert result.exit_code == ExitCode.ABORTED
        assert "not a TTY" in result.stderr

    def test_no_tty_without_yes_aborts_without_hanging(self, runner: CliRunner) -> None:
        """A pipe must abort rather than block forever waiting for a confirmation."""
        result = runner.invoke(
            app, ["destroy", "--execute"], input="", env=blank_env(AWS_REGION="us-east-1")
        )
        assert result.exit_code == ExitCode.ABORTED

    def test_yes_skips_the_prompt(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app, ["destroy", "--execute", "--yes"], env=blank_env(AWS_REGION="us-east-1")
        )
        assert result.exit_code == ExitCode.TOOL_MISSING

    def test_dry_run_prints_the_delete_command(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["destroy"], env=blank_env())
        assert "delete-cluster" in result.stdout


class TestSubmit:
    def test_validate_only_on_a_good_dag_exits_zero(
        self, runner: CliRunner, valid_dag: Path
    ) -> None:
        result = runner.invoke(
            app, ["submit", "--dag", str(valid_dag), "--validate-only"], env=blank_env()
        )
        assert result.exit_code == 0

    def test_validate_only_on_a_bad_dag_exits_four(self, runner: CliRunner, tmp_path: Path) -> None:
        document = valid_dag_document()
        document["nodes"][2]["op"] = "transpose"
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(document), encoding="utf-8")
        result = runner.invoke(
            app, ["submit", "--dag", str(bad), "--validate-only"], env=blank_env()
        )
        assert result.exit_code == ExitCode.DAG_INVALID

    def test_malformed_json_exits_four(self, runner: CliRunner, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{oops", encoding="utf-8")
        result = runner.invoke(app, ["submit", "--dag", str(bad)], env=blank_env())
        assert result.exit_code == ExitCode.DAG_INVALID

    def test_violations_are_tabulated_on_stderr(self, runner: CliRunner, tmp_path: Path) -> None:
        document = valid_dag_document()
        document["nodes"][2]["op"] = "transpose"
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(document), encoding="utf-8")
        result = runner.invoke(app, ["submit", "--dag", str(bad)], env=blank_env())
        assert "schema violations" in result.stderr

    def test_validation_runs_before_anything_else(self, runner: CliRunner, tmp_path: Path) -> None:
        """An invalid DAG must never reach the cluster, even with --execute."""
        bad = tmp_path / "bad.json"
        bad.write_text("{oops", encoding="utf-8")
        result = runner.invoke(
            app,
            ["submit", "--dag", str(bad), "--execute"],
            env=blank_env(HPCCTL_HEAD_NODE_HOST="1.2.3.4"),
        )
        assert result.exit_code == ExitCode.DAG_INVALID

    def test_missing_dag_file_exits_usage(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["submit", "--dag", str(tmp_path / "absent.json")], env=blank_env()
        )
        assert result.exit_code == ExitCode.USAGE

    def test_dry_run_prints_the_batch_script(self, runner: CliRunner, valid_dag: Path) -> None:
        result = runner.invoke(app, ["submit", "--dag", str(valid_dag)], env=blank_env())
        assert "#SBATCH --job-name=bench-matmul-001" in result.stdout

    def test_job_name_defaults_to_the_dag_id(self, runner: CliRunner, valid_dag: Path) -> None:
        result = runner.invoke(app, ["submit", "--dag", str(valid_dag)], env=blank_env())
        assert "bench-matmul-001" in result.stdout

    def test_job_name_override(self, runner: CliRunner, valid_dag: Path) -> None:
        result = runner.invoke(
            app,
            ["submit", "--dag", str(valid_dag), "--job-name", "custom-job"],
            env=blank_env(),
        )
        assert "--job-name=custom-job" in result.stdout

    def test_geometry_overrides_reach_the_script(self, runner: CliRunner, valid_dag: Path) -> None:
        result = runner.invoke(
            app,
            [
                "submit",
                "--dag",
                str(valid_dag),
                "--nodes",
                "8",
                "--ntasks",
                "64",
                "--time-limit",
                "01:00:00",
            ],
            env=blank_env(),
        )
        assert "--nodes=8" in result.stdout
        assert "--ntasks=64" in result.stdout
        assert "--time=01:00:00" in result.stdout

    def test_writes_the_generated_script(
        self, runner: CliRunner, valid_dag: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(app, ["submit", "--dag", str(valid_dag)], env=blank_env())
        assert result.exit_code == 0
        written = tmp_path / ".hpcctl-run" / "bench-matmul-001.sbatch.generated"
        assert written.is_file()

    def test_validate_only_writes_nothing(
        self, runner: CliRunner, valid_dag: Path, tmp_path: Path
    ) -> None:
        """The fully-local path touches nothing but reads."""
        runner.invoke(app, ["submit", "--dag", str(valid_dag), "--validate-only"], env=blank_env())
        assert not (tmp_path / ".hpcctl-run").exists()

    def test_version_mismatch_warns_without_failing(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        document = valid_dag_document()
        document["metadata"]["schema_version"] = "2.0.0"
        path = tmp_path / "future.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        result = runner.invoke(
            app, ["submit", "--dag", str(path), "--validate-only"], env=blank_env()
        )
        assert result.exit_code == 0
        assert "major versions differ" in result.stderr


class TestDeploy:
    def test_missing_build_dir_exits_config(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["deploy", "--build-dir", str(tmp_path / "absent")], env=blank_env()
        )
        assert result.exit_code == ExitCode.CONFIG

    def test_empty_build_dir_exits_config(self, runner: CliRunner, tmp_path: Path) -> None:
        """Catching 'you have not built the engine yet' needs no AWS account."""
        empty = tmp_path / "empty"
        empty.mkdir()
        result = runner.invoke(app, ["deploy", "--build-dir", str(empty)], env=blank_env())
        assert result.exit_code == ExitCode.CONFIG

    def test_dry_run_prints_the_rsync_command(self, runner: CliRunner, build_dir: Path) -> None:
        result = runner.invoke(app, ["deploy", "--build-dir", str(build_dir)], env=blank_env())
        assert "rsync" in result.stdout

    def test_uses_accept_new_host_key_policy(self, runner: CliRunner, build_dir: Path) -> None:
        """Never StrictHostKeyChecking=no, and never UserKnownHostsFile=/dev/null."""
        result = runner.invoke(app, ["deploy", "--build-dir", str(build_dir)], env=blank_env())
        assert "StrictHostKeyChecking=accept-new" in result.stdout
        assert "StrictHostKeyChecking=no" not in result.stdout
        assert "/dev/null" not in result.stdout

    def test_targets_the_shared_filesystem_not_the_home_dir(
        self, runner: CliRunner, build_dir: Path
    ) -> None:
        """Deploying to ~ubuntu would work on the head node and fail on every compute node."""
        result = runner.invoke(app, ["deploy", "--build-dir", str(build_dir)], env=blank_env())
        assert "/shared/engine/" in result.stdout

    def test_shows_a_transfer_manifest(self, runner: CliRunner, build_dir: Path) -> None:
        result = runner.invoke(app, ["deploy", "--build-dir", str(build_dir)], env=blank_env())
        assert "transfer manifest" in result.stdout

    def test_never_prints_key_contents(
        self, runner: CliRunner, build_dir: Path, tmp_path: Path
    ) -> None:
        """Only the key's path is ever read or displayed."""
        key = tmp_path / "id_rsa"
        key.write_text("-----BEGIN PRIVATE KEY-----\nSECRETMATERIAL\n", encoding="utf-8")
        result = runner.invoke(
            app,
            ["deploy", "--build-dir", str(build_dir)],
            env=blank_env(HPCCTL_SSH_KEY_PATH=str(key)),
        )
        assert "SECRETMATERIAL" not in result.stdout + result.stderr
        assert "id_rsa" in result.stdout


class TestStatus:
    def test_dry_run_prints_both_queries(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["status"], env=blank_env())
        assert "describe-cluster" in result.stdout
        assert "squeue" in result.stdout

    def test_no_queue_omits_the_ssh_query(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["status", "--no-queue"], env=blank_env())
        assert "squeue" not in result.stdout

    def test_dry_run_shows_placeholder_tables(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["status"], env=blank_env())
        assert "cluster" in result.stdout
        assert "slurm queue" in result.stdout

    def test_watch_is_refused_in_dry_run(self, runner: CliRunner) -> None:
        """It would loop over static text forever.

        Exit 2, not 1: an illegal flag combination is a usage error, and the enum reserves 1
        for unexpected exceptions that are always a bug.
        """
        result = runner.invoke(app, ["status", "--watch"], env=blank_env())
        assert result.exit_code == ExitCode.USAGE
        assert "--watch" in result.stderr


class TestNoColor:
    def test_no_color_suppresses_ansi(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--no-color", "status"], env=blank_env())
        assert "\x1b[" not in result.stdout

    def test_no_color_env_var_is_honoured(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["status"], env=blank_env(NO_COLOR="1"))
        assert "\x1b[" not in result.stdout

    def test_verbose_is_accepted(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["-v", "status"], env=blank_env())
        assert result.exit_code == 0


class TestExitCodeContract:
    def test_codes_are_the_documented_values(self) -> None:
        assert ExitCode.OK.value == 0
        assert ExitCode.INTERNAL.value == 1
        assert ExitCode.USAGE.value == 2
        assert ExitCode.CONFIG.value == 3
        assert ExitCode.DAG_INVALID.value == 4
        assert ExitCode.TOOL_MISSING.value == 5
        assert ExitCode.COMMAND_FAILED.value == 6
        assert ExitCode.ABORTED.value == 7
        assert ExitCode.CLUSTER_STATE.value == 8

    def test_no_command_calls_sys_exit(self) -> None:
        """Commands raise; one handler in cli.py owns the single exit path."""
        commands = Path(__file__).resolve().parents[1] / "src" / "hpcctl" / "commands"
        for module in commands.glob("*.py"):
            assert "sys.exit" not in module.read_text(encoding="utf-8")
