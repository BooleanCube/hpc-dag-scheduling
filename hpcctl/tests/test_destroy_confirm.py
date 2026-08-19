"""Unit tests for the destroy confirmation gate.

``CliRunner`` never presents a TTY, so the branch where the operator successfully types the
cluster name is unreachable through the CLI. It is also the branch that deletes a running
experiment, so it is tested directly here rather than left uncovered.
"""

import sys

import pytest
import typer

from hpcctl.commands import destroy as destroy_module
from hpcctl.config import Settings, load_settings
from hpcctl.errors import AbortedError


@pytest.fixture
def tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend stdin is an interactive terminal.

    Args:
        monkeypatch: pytest's attribute patcher.
    """
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)


def answer(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Stub the confirmation prompt with a canned response.

    Args:
        monkeypatch: pytest's attribute patcher.
        text: What the operator "types".
    """
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: text)


class TestTypedNameGate:
    def test_exact_match_is_accepted(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch, tty: None
    ) -> None:
        answer(monkeypatch, "hpc-dag-baseline")
        destroy_module._confirm(settings, yes=False)

    def test_mismatch_aborts(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch, tty: None
    ) -> None:
        answer(monkeypatch, "wrong")
        with pytest.raises(AbortedError, match="did not match"):
            destroy_module._confirm(settings, yes=False)

    def test_empty_answer_aborts(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch, tty: None
    ) -> None:
        answer(monkeypatch, "")
        with pytest.raises(AbortedError):
            destroy_module._confirm(settings, yes=False)

    def test_surrounding_whitespace_is_tolerated(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch, tty: None
    ) -> None:
        answer(monkeypatch, "  hpc-dag-baseline\n")
        destroy_module._confirm(settings, yes=False)

    def test_yes_confirmation_is_not_accepted(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch, tty: None
    ) -> None:
        """The whole point of typing the name is that 'y' must not be enough."""
        answer(monkeypatch, "y")
        with pytest.raises(AbortedError):
            destroy_module._confirm(settings, yes=False)

    def test_a_different_cluster_name_aborts(
        self, monkeypatch: pytest.MonkeyPatch, tty: None
    ) -> None:
        monkeypatch.setenv("HPCCTL_CLUSTER_NAME", "production-cluster")
        answer(monkeypatch, "hpc-dag-baseline")
        with pytest.raises(AbortedError):
            destroy_module._confirm(load_settings(live=False), yes=False)

    def test_hint_states_the_expected_text(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch, tty: None
    ) -> None:
        answer(monkeypatch, "wrong")
        with pytest.raises(AbortedError) as excinfo:
            destroy_module._confirm(settings, yes=False)
        assert excinfo.value.hint is not None
        assert "hpc-dag-baseline" in excinfo.value.hint

    def test_abort_exit_code(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch, tty: None
    ) -> None:
        answer(monkeypatch, "wrong")
        with pytest.raises(AbortedError) as excinfo:
            destroy_module._confirm(settings, yes=False)
        assert int(excinfo.value.code) == 7


class TestBypasses:
    def test_yes_skips_the_prompt_entirely(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*args: object, **kwargs: object) -> str:
            raise AssertionError("prompt must not be reached when --yes is passed")

        monkeypatch.setattr(typer, "prompt", explode)
        destroy_module._confirm(settings, yes=True)

    def test_non_tty_aborts_without_prompting(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*args: object, **kwargs: object) -> str:
            raise AssertionError("must not block on a pipe")

        monkeypatch.setattr(typer, "prompt", explode)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        with pytest.raises(AbortedError, match="not a TTY"):
            destroy_module._confirm(settings, yes=False)
