"""Root Typer application and the single error-handling path.

Individual commands never call ``sys.exit``; they raise
:class:`~hpcctl.errors.HpcctlError`. One decorator, applied at registration, turns every such
error into a message on stderr plus the exception's own exit code. One exit path is testable,
five are not.

The handler is applied to the command functions rather than wrapped around ``main()`` on purpose:
``typer.testing.CliRunner`` invokes the app directly, so a handler living in ``main()`` would be
invisible to the entire test suite.
"""

import functools
from collections.abc import Callable
from typing import Annotated, Any, TypeVar

import typer

from hpcctl import console
from hpcctl.commands.boot import boot
from hpcctl.commands.deploy import deploy
from hpcctl.commands.destroy import destroy
from hpcctl.commands.status import status
from hpcctl.commands.submit import submit
from hpcctl.errors import DagValidationError, HpcctlError

F = TypeVar("F", bound=Callable[..., None])

app = typer.Typer(
    name="hpcctl",
    help="Manage AWS ParallelCluster lifecycles for the HPC DAG scheduling baseline.",
    no_args_is_help=True,
)


def handled(command: F) -> F:
    """Wrap a command so expected failures become clean exits.

    Args:
        command: The command function to wrap.

    Returns:
        The wrapped function, with ``functools.wraps`` preserving the signature Typer
        introspects to build its options.
    """

    @functools.wraps(command)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            command(*args, **kwargs)
        except BrokenPipeError:
            # A downstream reader closed the pipe, as `hpcctl boot --raw | head` does. The user
            # got what they asked for, so this is a clean exit rather than a failure.
            #
            # Deliberately no os.dup2 onto stdout's descriptor here. That is the common recipe
            # for silencing the interpreter's shutdown flush, but it mutates a global file
            # descriptor -- under pytest's capture it clobbers the capture file and takes the
            # whole session down with EBADF. Not worth it to suppress one shutdown message.
            raise typer.Exit(0) from None
        except HpcctlError as err:
            console.render_error(err.message, hint=err.hint)
            if isinstance(err, DagValidationError) and err.problems:
                table = console.new_table("schema violations", "path", "problem")
                for pointer, message in err.problems:
                    table.add_row(pointer or "/", message)
                console.err().print(table)
            raise typer.Exit(int(err.code)) from err

    return wrapper  # type: ignore[return-value]


@app.callback()
def root(
    no_color: Annotated[
        bool, typer.Option("--no-color", help="Disable colour and styling in all output.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Print additional diagnostic detail.")
    ] = False,
) -> None:
    """Manage AWS ParallelCluster lifecycles for the HPC DAG scheduling baseline.

    Every AWS-touching command defaults to a dry-run and needs an explicit ``--execute`` to do
    anything real. Setting ``HPCCTL_DRY_RUN`` refuses execution globally, even with ``--execute``.
    """
    console.configure(no_color=no_color)
    if verbose:
        console.render_notice("verbose mode enabled")


@app.command()
def version() -> None:
    """Print the installed hpcctl version."""
    from importlib.metadata import version as _version

    typer.echo(_version("hpcctl"))


app.command()(handled(boot))
app.command()(handled(deploy))
app.command()(handled(submit))
app.command()(handled(status))
app.command()(handled(destroy))


def main() -> None:
    """Run the hpcctl command-line application."""
    app()
