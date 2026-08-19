"""Coverage for defensive branches that the main flows do not reach.

Most of these guard against inputs the schema or the CLI would normally reject first. They are
still worth exercising: an untested defensive branch is a branch nobody has ever run.
"""

from pathlib import Path

import pytest
import typer
from conftest import blank_env
from typer.testing import CliRunner

from hpcctl import cli, console
from hpcctl.commands.submit import _default_job_name
from hpcctl.config import Settings, default_schema_path, load_settings
from hpcctl.errors import HpcctlError, InvalidConfigError
from hpcctl.exit_codes import ExitCode
from hpcctl.external import require_tool
from hpcctl.validation import schema_version


class TestErrorHandler:
    def test_broken_pipe_exits_zero(self) -> None:
        """`hpcctl boot --raw | head` must not look like a failure."""

        def command() -> None:
            raise BrokenPipeError

        with pytest.raises(typer.Exit) as excinfo:
            cli.handled(command)()
        assert excinfo.value.exit_code == 0

    def test_hpcctl_error_uses_its_own_code(self) -> None:
        def command() -> None:
            raise HpcctlError("boom")

        with pytest.raises(typer.Exit) as excinfo:
            cli.handled(command)()
        assert excinfo.value.exit_code == ExitCode.INTERNAL

    def test_subclass_code_is_honoured(self) -> None:
        def command() -> None:
            raise InvalidConfigError("bad value")

        with pytest.raises(typer.Exit) as excinfo:
            cli.handled(command)()
        assert excinfo.value.exit_code == ExitCode.CONFIG

    def test_successful_command_returns_normally(self) -> None:
        calls: list[int] = []

        def command() -> None:
            calls.append(1)

        cli.handled(command)()
        assert calls == [1]

    def test_signature_is_preserved_for_typer(self) -> None:
        """functools.wraps is what lets Typer still see the command's options."""

        def command(flag: bool = False) -> None:
            """Do a thing."""

        wrapped = cli.handled(command)
        assert wrapped.__doc__ == "Do a thing."
        assert wrapped.__name__ == "command"

    def test_unexpected_exceptions_propagate(self) -> None:
        """A bug should surface as a traceback, not be flattened into a tidy exit code."""

        def command() -> None:
            raise ZeroDivisionError("a genuine bug")

        with pytest.raises(ZeroDivisionError):
            cli.handled(command)()


class TestConsoleNoOps:
    def test_placeholder_warning_is_silent_when_complete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name, value in {
            "AWS_REGION": "us-east-1",
            "HPCCTL_KEY_NAME": "kp",
            "HPCCTL_HEAD_SUBNET_ID": "subnet-aaaa",
            "HPCCTL_BOOTSTRAP_BUCKET": "bucket",
            "HPCCTL_HEAD_NODE_HOST": "1.2.3.4",
        }.items():
            monkeypatch.setenv(name, value)
        settings = load_settings(live=False)
        console.configure(no_color=True)
        with console.err().capture() as captured:
            console.render_placeholder_warning(settings)
        assert captured.get() == ""

    def test_warning_is_emitted_when_incomplete(self, settings: Settings) -> None:
        console.configure(no_color=True)
        with console.err().capture() as captured:
            console.render_placeholder_warning(settings)
        assert "HPCCTL_KEY_NAME" in captured.get()

    def test_error_hint_is_optional(self) -> None:
        console.configure(no_color=True)
        with console.err().capture() as captured:
            console.render_error("something failed")
        assert "hint:" not in captured.get()

    def test_error_hint_is_shown_when_given(self) -> None:
        console.configure(no_color=True)
        with console.err().capture() as captured:
            console.render_error("something failed", hint="try this")
        assert "try this" in captured.get()


class TestJobNameFallback:
    def test_dag_id_is_preferred(self, tmp_path: Path) -> None:
        document: dict[str, object] = {"metadata": {"dag_id": "from-metadata"}}
        assert _default_job_name(document, tmp_path / "from-file.json") == "from-metadata"

    def test_falls_back_to_the_file_stem(self, tmp_path: Path) -> None:
        """Defensive: the schema requires dag_id, so this only fires on a hand-edited file."""
        assert _default_job_name({}, tmp_path / "from-file.json") == "from-file"

    def test_non_string_dag_id_falls_back(self, tmp_path: Path) -> None:
        document: dict[str, object] = {"metadata": {"dag_id": 42}}
        assert _default_job_name(document, tmp_path / "from-file.json") == "from-file"

    def test_empty_dag_id_falls_back(self, tmp_path: Path) -> None:
        document: dict[str, object] = {"metadata": {"dag_id": ""}}
        assert _default_job_name(document, tmp_path / "from-file.json") == "from-file"


class TestDeployPreconditions:
    def test_build_path_that_is_a_file_is_rejected(self, runner: CliRunner, tmp_path: Path) -> None:
        not_a_dir = tmp_path / "engine"
        not_a_dir.write_text("oops", encoding="utf-8")
        result = runner.invoke(cli.app, ["deploy", "--build-dir", str(not_a_dir)], env=blank_env())
        assert result.exit_code == ExitCode.CONFIG
        assert "not a directory" in result.stderr

    def test_large_manifest_is_truncated(self, runner: CliRunner, tmp_path: Path) -> None:
        build = tmp_path / "big"
        build.mkdir()
        for index in range(40):
            (build / f"lib{index:03d}.so").write_text("x", encoding="utf-8")
        result = runner.invoke(cli.app, ["deploy", "--build-dir", str(build)], env=blank_env())
        assert result.exit_code == 0
        assert "and 15 more" in result.stdout


class TestSchemaHelpers:
    def test_default_schema_path_finds_the_contract(self) -> None:
        assert default_schema_path().is_file()

    def test_schema_version_returns_none_without_examples(self) -> None:
        assert schema_version({}) is None

    def test_schema_version_ignores_non_string_examples(self) -> None:
        schema = {
            "$defs": {
                "metadata": {"properties": {"schema_version": {"examples": [1, 2]}}},
            }
        }
        assert schema_version(schema) is None


class TestToolDiscoverySuccess:
    def test_require_tool_returns_an_absolute_path(self) -> None:
        found = require_tool("bash")
        assert Path(found).is_absolute()
        assert Path(found).exists()


class TestMain:
    def test_main_delegates_to_the_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[bool] = []
        monkeypatch.setattr(cli, "app", lambda *a, **k: called.append(True))
        cli.main()
        assert called == [True]
