"""Console output, and the one measured constraint that shapes it.

Rich truncates long lines, and a ``Panel`` around syntax makes it worse. Rendering a bash script
containing a 224-character ``apt-get install`` line at terminal width 80:

===================================================  ====================
Rendering                                            Full line preserved
===================================================  ====================
``Syntax(text, "bash")``                             no, truncated silently
``Syntax(text, "bash", word_wrap=True)``             yes
``Syntax(text, "bash")`` with ``crop=False``         no
``Panel(Syntax(text, "bash", word_wrap=True))``      no, the border steals width
plain ``str`` with ``soft_wrap=True``                yes
===================================================  ====================

The ``Panel`` row is the trap: panels are the obvious way to present a labelled artifact, and the
failure is invisible -- the output looks clean and is missing packages. So artifact bodies use a
:class:`rich.rule.Rule` label plus word-wrapped syntax, never a panel; panels are reserved for
short content where truncation cannot occur. For byte-exact output, callers use ``--raw``, which
bypasses this module entirely via :func:`write_raw`.
"""

import shlex
import sys
from collections.abc import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from hpcctl.config import Settings, color_disabled

RAW_DELIMITER: str = "# ===== hpcctl:artifact:{name} ====="
"""Separator emitted between artifacts in ``--raw`` mode.

A ``#`` comment line, so it is inert in bash, YAML, and shell alike; splitting on it recovers
each artifact byte-for-byte.
"""

_stdout = Console(soft_wrap=False)
_stderr = Console(stderr=True)


def configure(*, no_color: bool = False) -> None:
    """Configure the module-level consoles for this invocation.

    Args:
        no_color: Disable colour and styling. The conventional ``NO_COLOR`` environment
            variable has the same effect.
    """
    global _stdout, _stderr
    disabled = no_color or color_disabled()
    _stdout = Console(soft_wrap=False, no_color=disabled, highlight=not disabled)
    _stderr = Console(stderr=True, no_color=disabled, highlight=not disabled)


def out() -> Console:
    """Return the stdout console.

    Returns:
        The console artifacts and tables are written to.
    """
    return _stdout


def err() -> Console:
    """Return the stderr console.

    Keeping warnings and errors off stdout is what lets ``hpcctl boot --raw`` be piped.

    Returns:
        The console warnings and errors are written to.
    """
    return _stderr


def render_artifact(title: str, body: str, lexer: str) -> None:
    """Print a generated artifact with a labelled rule and syntax highlighting.

    Uses a Rule rather than a Panel, and ``word_wrap``, because Panel borders reduce the
    available width and silently truncate long lines.

    Args:
        title: Human-readable artifact label, e.g. ``"cluster config (YAML)"``.
        body: Exact artifact text.
        lexer: Pygments lexer name, one of ``"yaml"``, ``"bash"``, or ``"json"``.
    """
    _stdout.print(Rule(title, align="left"))
    _stdout.print(Syntax(body, lexer, word_wrap=True, theme="ansi_dark"))


def write_raw(name: str, body: str) -> None:
    """Write an artifact to stdout byte-for-byte, bypassing rich entirely.

    This is the supported path for piping into ``bash -n`` or ``yaml.safe_load``.

    Args:
        name: Short artifact identifier used in the delimiter comment.
        body: Exact artifact text. A trailing newline is added when absent.
    """
    sys.stdout.write(RAW_DELIMITER.format(name=name) + "\n")
    sys.stdout.write(body if body.endswith("\n") else body + "\n")


def render_command(argv: Sequence[str]) -> None:
    """Print an external command as a copy-pasteable, shell-quoted line.

    Args:
        argv: Command as an argument vector.
    """
    _stdout.print(Syntax(shlex.join(argv), "bash", word_wrap=True, theme="ansi_dark"))


def format_command(argv: Sequence[str]) -> str:
    """Render a command as a correctly quoted shell line.

    ``shlex.join`` rather than ``" ".join``: hand-rolled joining produces output that looks
    right and breaks on the first path containing a space.

    Args:
        argv: Command as an argument vector.

    Returns:
        A pasteable shell command line.
    """
    return shlex.join(argv)


def render_placeholder_warning(settings: Settings) -> None:
    """Warn that required configuration was replaced with placeholders.

    Short, bounded content, so a Panel is safe here.

    Args:
        settings: Resolved settings whose ``missing`` tuple is non-empty.
    """
    if not settings.has_placeholders:
        return
    listed = "\n".join(f"  {name}" for name in settings.missing)
    _stderr.print(
        Panel(
            f"These variables are unset and were replaced with placeholders:\n{listed}\n\n"
            "Dry-run output is still structurally valid, but no AWS API would accept it.\n"
            "Set them (see hpcctl/.env.example) before using --execute.",
            title="incomplete configuration",
            border_style="yellow",
        )
    )


def render_notice(message: str) -> None:
    """Print an informational notice to stderr.

    Args:
        message: Text to display.
    """
    _stderr.print(f"[cyan]note[/cyan] {message}")


def render_warning(message: str) -> None:
    """Print a warning to stderr.

    Args:
        message: Text to display.
    """
    _stderr.print(f"[yellow]warning[/yellow] {message}")


def render_error(message: str, *, hint: str | None = None) -> None:
    """Print an error, and an optional hint, to stderr.

    Args:
        message: Text to display.
        hint: Optional actionable follow-up.
    """
    _stderr.print(f"[red]error[/red] {message}")
    if hint:
        _stderr.print(f"[dim]hint:[/dim] {hint}")


def new_table(title: str, *columns: str) -> Table:
    """Build a rich table with the project's standard styling.

    Args:
        title: Table caption.
        *columns: Column headers, in order.

    Returns:
        An empty table ready for ``add_row`` calls.
    """
    table = Table(title=title, title_justify="left", header_style="bold")
    for column in columns:
        table.add_column(column, overflow="fold")
    return table
