"""Tests for tool discovery and the subprocess wrapper.

Two invariants matter more than the rest: commands are always an argv list (never a shell
string, because cluster names and paths come from the environment), and anything displayed goes
through ``shlex.join`` so a printed command is one the user can actually paste.
"""

import ast
import os
import stat
import subprocess
from pathlib import Path

import pytest

from hpcctl.config import Settings
from hpcctl.console import format_command
from hpcctl.errors import ExternalCommandError, ToolMissingError
from hpcctl.external import (
    find_tool,
    require_tool,
    require_tools,
    run,
    scp_argv,
    ssh_argv,
)


@pytest.fixture
def fake_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a directory on PATH for stand-in executables.

    Args:
        tmp_path: Scratch directory.
        monkeypatch: pytest's environment patcher.

    Returns:
        The directory, already prepended to PATH.
    """
    target = tmp_path / "fakebin"
    target.mkdir()
    monkeypatch.setenv("PATH", f"{target}:{os.environ['PATH']}")
    return target


def install(directory: Path, name: str, body: str) -> Path:
    """Write an executable stand-in for a tool.

    Args:
        directory: Directory on PATH.
        name: Executable name.
        body: Shell body, without the shebang.

    Returns:
        The created path.
    """
    path = directory / name
    path.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


class TestDiscovery:
    def test_find_tool_returns_none_when_absent(self, fake_bin: Path) -> None:
        assert find_tool("definitely-not-a-tool") is None

    def test_find_tool_locates_an_installed_tool(self, fake_bin: Path) -> None:
        install(fake_bin, "faketool", "exit 0")
        assert find_tool("faketool") is not None

    def test_require_tool_raises_when_absent(self) -> None:
        with pytest.raises(ToolMissingError, match="not on PATH"):
            require_tool("definitely-not-a-tool")

    def test_require_tool_exit_code_is_five(self) -> None:
        with pytest.raises(ToolMissingError) as excinfo:
            require_tool("pcluster")
        assert int(excinfo.value.code) == 5

    def test_hint_is_actionable(self) -> None:
        with pytest.raises(ToolMissingError) as excinfo:
            require_tool("pcluster")
        assert excinfo.value.hint is not None
        assert "aws-parallelcluster" in excinfo.value.hint

    def test_require_tools_reports_every_missing_tool(self) -> None:
        with pytest.raises(ToolMissingError) as excinfo:
            require_tools("pcluster", "aws")
        assert "pcluster" in excinfo.value.message
        assert "aws" in excinfo.value.message

    def test_require_tools_returns_paths_when_present(self, fake_bin: Path) -> None:
        install(fake_bin, "toola", "exit 0")
        install(fake_bin, "toolb", "exit 0")
        found = require_tools("toola", "toolb")
        assert set(found) == {"toola", "toolb"}


class TestRun:
    def test_dry_run_returns_none_without_executing(self, tmp_path: Path) -> None:
        marker = tmp_path / "touched"
        result = run(["touch", str(marker)], dry_run=True)
        assert result is None
        assert not marker.exists()

    def test_executes_when_not_dry_run(self, tmp_path: Path) -> None:
        marker = tmp_path / "touched"
        result = run(["touch", str(marker)], dry_run=False)
        assert result is not None
        assert marker.exists()

    def test_captures_stdout(self) -> None:
        result = run(["echo", "hello"], dry_run=False)
        assert result is not None
        assert result.stdout.strip() == "hello"

    def test_non_zero_raises_by_default(self, fake_bin: Path) -> None:
        install(fake_bin, "failing", "exit 3")
        with pytest.raises(ExternalCommandError) as excinfo:
            run(["failing"], dry_run=False)
        assert excinfo.value.returncode == 3

    def test_failure_exit_code_is_six(self, fake_bin: Path) -> None:
        install(fake_bin, "failing", "exit 1")
        with pytest.raises(ExternalCommandError) as excinfo:
            run(["failing"], dry_run=False)
        assert int(excinfo.value.code) == 6

    def test_check_false_tolerates_failure(self, fake_bin: Path) -> None:
        install(fake_bin, "failing", "exit 7")
        result = run(["failing"], dry_run=False, check=False)
        assert result is not None
        assert result.returncode == 7

    def test_stderr_is_carried_on_the_error(self, fake_bin: Path) -> None:
        install(fake_bin, "noisy", "echo 'went wrong' >&2; exit 1")
        with pytest.raises(ExternalCommandError) as excinfo:
            run(["noisy"], dry_run=False)
        assert "went wrong" in excinfo.value.stderr

    def test_missing_executable_becomes_tool_missing(self) -> None:
        """Exit 5, not a raw FileNotFoundError traceback."""
        with pytest.raises(ToolMissingError):
            run(["definitely-not-a-tool"], dry_run=False)

    def test_never_uses_a_shell(self) -> None:
        """A shell string would turn any env-derived path into an injection vector.

        Checked against the AST rather than the text: the module docstring names ``shell=True``
        precisely to explain why it is never used, and a substring search cannot tell the two
        apart.
        """
        package = Path(__file__).resolve().parents[1] / "src" / "hpcctl"
        for module in package.rglob("*.py"):
            tree = ast.parse(module.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    names = {kw.arg for kw in node.keywords}
                    assert "shell" not in names, f"{module.name} passes shell= to a call"

    def test_argv_elements_are_not_word_split(self, tmp_path: Path) -> None:
        spaced = tmp_path / "a directory with spaces"
        spaced.mkdir()
        target = spaced / "file"
        result = run(["touch", str(target)], dry_run=False)
        assert result is not None
        assert target.exists()


class TestQuoting:
    def test_paths_with_spaces_are_quoted(self) -> None:
        """A hand-rolled ' '.join looks right and breaks on the first spaced path."""
        rendered = format_command(["cp", "/tmp/a b/c", "/dest"])
        assert "'/tmp/a b/c'" in rendered

    def test_rendered_command_round_trips_through_the_shell(self, tmp_path: Path) -> None:
        target = tmp_path / "spaced dir"
        target.mkdir()
        rendered = format_command(["ls", "-d", str(target)])
        completed = subprocess.run(
            ["bash", "-c", rendered], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0
        assert str(target) in completed.stdout

    def test_simple_commands_are_left_unquoted(self) -> None:
        assert format_command(["ls", "-la"]) == "ls -la"


class TestSshPolicy:
    def test_ssh_uses_accept_new(self, settings: Settings) -> None:
        argv = ssh_argv(key_path="/k", user="ubuntu", host="h")
        assert "StrictHostKeyChecking=accept-new" in argv

    def test_ssh_never_disables_host_key_checking(self) -> None:
        argv = ssh_argv(key_path="/k", user="ubuntu", host="h")
        joined = " ".join(argv)
        assert "StrictHostKeyChecking=no" not in joined
        assert "UserKnownHostsFile" not in joined

    def test_ssh_target_format(self) -> None:
        argv = ssh_argv(key_path="/k", user="ubuntu", host="1.2.3.4")
        assert "ubuntu@1.2.3.4" in argv

    def test_ssh_remote_command_is_a_single_argv_element(self) -> None:
        argv = ssh_argv(key_path="/k", user="u", host="h", remote_command="squeue --long")
        assert argv[-1] == "squeue --long"

    def test_ssh_without_a_remote_command(self) -> None:
        argv = ssh_argv(key_path="/k", user="u", host="h")
        assert argv[-1] == "u@h"

    def test_scp_target_format(self) -> None:
        argv = scp_argv(key_path="/k", local="/l", user="u", host="h", remote="/r")
        assert argv[-1] == "u@h:/r"

    def test_scp_uses_accept_new(self) -> None:
        assert "StrictHostKeyChecking=accept-new" in scp_argv(
            key_path="/k", local="/l", user="u", host="h", remote="/r"
        )

    def test_key_path_is_passed_by_path_not_contents(self, tmp_path: Path) -> None:
        key = tmp_path / "id_rsa"
        key.write_text("SECRET", encoding="utf-8")
        argv = ssh_argv(key_path=str(key), user="u", host="h")
        assert str(key) in argv
        assert "SECRET" not in " ".join(argv)
