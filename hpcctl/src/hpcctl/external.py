"""External tool discovery and subprocess execution.

Two rules, both non-negotiable. **Always an argv list, never** ``shell=True`` -- cluster names
and paths come from the environment, and a shell string turns any of them into an injection
vector. **Rendering for display goes through** ``shlex.join`` (via :mod:`hpcctl.console`), so
what dry-run prints is a command the user can actually paste, correctly quoted.

Tools are discovered at call time rather than imported as dependencies: the dry-run path must
work on a machine with neither ``pcluster`` nor ``aws`` installed.
"""

import shutil
import subprocess
from collections.abc import Sequence

from hpcctl import console
from hpcctl.errors import ExternalCommandError, ToolMissingError

INSTALL_HINTS: dict[str, str] = {
    "pcluster": "pip install aws-parallelcluster",
    "aws": "See https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html",
    "rsync": "apt-get install -y rsync",
    "ssh": "apt-get install -y openssh-client",
    "scp": "apt-get install -y openssh-client",
}
"""Actionable install hint per tool, surfaced when a live command needs a missing one."""


def find_tool(name: str) -> str | None:
    """Locate an external tool without raising.

    Args:
        name: Executable name to look up on PATH.

    Returns:
        The absolute path, or ``None`` when the tool is not installed.
    """
    return shutil.which(name)


def require_tool(name: str) -> str:
    """Return the absolute path to an external tool.

    Args:
        name: Executable name to look up on PATH.

    Returns:
        The absolute path to the tool.

    Raises:
        ToolMissingError: If the tool is not on PATH. Exits 5, with an install hint.
    """
    found = find_tool(name)
    if found is None:
        raise ToolMissingError(
            f"required tool {name!r} is not on PATH",
            hint=INSTALL_HINTS.get(name, f"Install {name} and re-run."),
        )
    return found


def require_tools(*names: str) -> dict[str, str]:
    """Resolve several tools at once, reporting all missing ones together.

    Args:
        *names: Executable names to look up.

    Returns:
        A mapping of name to absolute path.

    Raises:
        ToolMissingError: If any tool is missing. Names every missing tool in one message.
    """
    found: dict[str, str] = {}
    absent: list[str] = []
    for name in names:
        path = find_tool(name)
        if path is None:
            absent.append(name)
        else:
            found[name] = path
    if absent:
        hints = "; ".join(f"{name}: {INSTALL_HINTS.get(name, 'install it')}" for name in absent)
        raise ToolMissingError(
            f"required tool(s) not on PATH: {', '.join(absent)}",
            hint=hints,
        )
    return found


def run(
    argv: Sequence[str],
    *,
    dry_run: bool,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str] | None:
    """Execute an external command, or print it when in dry-run.

    Args:
        argv: Command as an argument vector. Never a shell string.
        dry_run: When true, render the command and return None without executing.
        check: Raise ExternalCommandError (exit 6) on a non-zero return code.
        capture: Capture stdout/stderr rather than streaming to the terminal.

    Returns:
        The completed process, or None in dry-run.

    Raises:
        ExternalCommandError: If the command fails and ``check`` is set.
        ToolMissingError: If the executable disappeared between discovery and execution.
    """
    if dry_run:
        console.render_command(argv)
        return None

    try:
        completed = subprocess.run(  # noqa: S603 - argv list, never shell=True
            list(argv),
            check=False,
            capture_output=capture,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ToolMissingError(
            f"command not found: {argv[0]!r}",
            hint=INSTALL_HINTS.get(str(argv[0]), "Install it and re-run."),
        ) from exc

    if check and completed.returncode != 0:
        raise ExternalCommandError(
            f"command failed with exit status {completed.returncode}: "
            f"{console.format_command(argv)}",
            returncode=completed.returncode,
            stderr=(completed.stderr or "").strip(),
        )
    return completed


def ssh_argv(
    *, key_path: str, user: str, host: str, remote_command: str | None = None
) -> list[str]:
    """Build an ``ssh`` argument vector with the project's host-key policy.

    ``StrictHostKeyChecking=accept-new`` rather than ``no``: it still pins the key after first
    contact, so it protects against a later MITM while not prompting on first connect. Never
    ``no``, and never ``UserKnownHostsFile=/dev/null``.

    Args:
        key_path: Path to the private key. Only the path is ever used or printed.
        user: Remote login user.
        host: Remote hostname or IP.
        remote_command: Optional command to run remotely.

    Returns:
        The ``ssh`` argument vector.
    """
    argv = [
        "ssh",
        "-i",
        key_path,
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"{user}@{host}",
    ]
    if remote_command is not None:
        argv.append(remote_command)
    return argv


def scp_argv(*, key_path: str, local: str, user: str, host: str, remote: str) -> list[str]:
    """Build an ``scp`` argument vector with the project's host-key policy.

    Args:
        key_path: Path to the private key.
        local: Local source path.
        user: Remote login user.
        host: Remote hostname or IP.
        remote: Remote destination path.

    Returns:
        The ``scp`` argument vector.
    """
    return [
        "scp",
        "-i",
        key_path,
        "-o",
        "StrictHostKeyChecking=accept-new",
        local,
        f"{user}@{host}:{remote}",
    ]
