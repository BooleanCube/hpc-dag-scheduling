"""Exception hierarchy for the hpcctl CLI.

Every error carries the :class:`~hpcctl.exit_codes.ExitCode` it should exit with, so the
single handler in :mod:`hpcctl.cli` needs no mapping table. Commands raise; they never call
``sys.exit``. One exit path is testable, five are not.
"""

from collections.abc import Sequence

from hpcctl.exit_codes import ExitCode


class HpcctlError(Exception):
    """Base class for every expected hpcctl failure.

    Attributes:
        code: Exit status the process should terminate with.
        hint: Optional actionable follow-up printed after the message.
    """

    code: ExitCode = ExitCode.INTERNAL

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        """Record the failure message and an optional remediation hint.

        Args:
            message: Human-readable description of what went wrong.
            hint: Optional next step the user can take.
        """
        super().__init__(message)
        self.message = message
        self.hint = hint


class MissingConfigError(HpcctlError):
    """A required environment variable was unset when a real value was needed."""

    code = ExitCode.CONFIG

    def __init__(self, variables: Sequence[str], *, hint: str | None = None) -> None:
        """Report every missing variable at once.

        A live run that discovers a missing variable halfway through has already created
        billable resources, so all failures are reported in one pass.

        Args:
            variables: Names of the unset required variables, reported together.
            hint: Optional remediation hint; a default pointing at ``.env.example`` is used
                when omitted.
        """
        self.variables = tuple(variables)
        listed = ", ".join(self.variables)
        super().__init__(
            f"missing required environment variable(s): {listed}",
            hint=hint or "See hpcctl/.env.example for the full variable contract.",
        )


class InvalidConfigError(HpcctlError):
    """An environment variable was set but could not be parsed.

    Distinct from :class:`MissingConfigError`: the value is present and wrong, which is a typo
    rather than an absent AWS account, so it is fatal in dry-run too.
    """

    code = ExitCode.CONFIG


class DagValidationError(HpcctlError):
    """A DAG file was malformed or did not conform to the serialization contract."""

    code = ExitCode.DAG_INVALID

    def __init__(
        self, message: str, *, problems: Sequence[tuple[str, str]] = (), hint: str | None = None
    ) -> None:
        """Report a validation failure with every schema error it found.

        Args:
            message: Summary of the failure.
            problems: ``(json_pointer, message)`` pairs, one per schema violation.
            hint: Optional remediation hint.
        """
        super().__init__(message, hint=hint)
        self.problems = tuple(problems)


class ToolMissingError(HpcctlError):
    """An external tool required for live execution is not on PATH."""

    code = ExitCode.TOOL_MISSING


class ExternalCommandError(HpcctlError):
    """An external command returned a non-zero exit status."""

    code = ExitCode.COMMAND_FAILED

    def __init__(
        self, message: str, *, returncode: int, stderr: str = "", hint: str | None = None
    ) -> None:
        """Report a failed subprocess.

        Args:
            message: Summary naming the command that failed.
            returncode: The command's exit status.
            stderr: Captured standard error, if any.
            hint: Optional remediation hint.
        """
        super().__init__(message, hint=hint)
        self.returncode = returncode
        self.stderr = stderr


class AbortedError(HpcctlError):
    """The user declined a confirmation, or no TTY was available to ask for one."""

    code = ExitCode.ABORTED


class ClusterStateError(HpcctlError):
    """The cluster is absent or in an unexpected state."""

    code = ExitCode.CLUSTER_STATE
