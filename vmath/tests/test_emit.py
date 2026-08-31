"""Tests for :func:`vmath.emit`, the case-script output contract."""

import json
from pathlib import Path

import pytest

from vmath import Graph, InitNode, emit


def _graph() -> Graph:
    """Build a minimal two-node graph.

    Returns:
        A closed graph with one matmul output.
    """
    a = InitNode((4, 3), seed=1)
    b = InitNode((3, 2), seed=2)
    return Graph([a @ b], dag_id="emit-test-001")


class TestEmit:
    def test_writes_to_the_argv_path_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The hpcctl submit calling convention: destination is the first argument."""
        target = tmp_path / "out.json"
        monkeypatch.setattr("sys.argv", ["case.py", str(target)])
        emit(_graph())
        document = json.loads(target.read_text(encoding="utf-8"))
        assert document["metadata"]["dag_id"] == "emit-test-001"

    def test_prints_to_stdout_when_run_standalone(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("sys.argv", ["case.py"])
        emit(_graph())
        document = json.loads(capsys.readouterr().out)
        assert document["metadata"]["dag_id"] == "emit-test-001"

    def test_forwards_serialize_options(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        target = tmp_path / "out.json"
        monkeypatch.setattr("sys.argv", ["case.py", str(target)])
        emit(_graph(), include_timestamp=False)
        document = json.loads(target.read_text(encoding="utf-8"))
        assert "created_at" not in document["metadata"]
