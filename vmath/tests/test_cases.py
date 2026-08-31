"""Conformance tests for every case script in the repository's ``/cases`` suite.

Each case is executed exactly the way ``hpcctl submit`` executes it — a subprocess with the
output path as ``argv[1]`` — and the document it writes is validated against the shared
schema. A case that stops conforming here would otherwise fail on the cluster, after money
has been spent.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema
import pytest

REPO = Path(__file__).resolve().parents[2]
CASES = sorted((REPO / "cases").glob("*.py"))
SCHEMA = json.loads((REPO / "shared" / "dag_schema.json").read_text(encoding="utf-8"))


def _compile(script: Path, tmp_path: Path) -> dict[str, Any]:
    """Run a case script under the hpcctl calling convention and load its document.

    Args:
        script: The case script to execute.
        tmp_path: Scratch directory for the output document.

    Returns:
        The parsed DAG document.
    """
    target = tmp_path / f"{script.stem}.json"
    completed = subprocess.run(
        [sys.executable, str(script), str(target)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, f"{script.name} failed:\n{completed.stderr}"
    return json.loads(target.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_the_suite_is_at_least_twenty_cases() -> None:
    """The suite's whole point is breadth; a shrinking suite is a regression."""
    assert len(CASES) >= 20


@pytest.mark.parametrize("script", CASES, ids=lambda path: path.stem)
class TestEveryCase:
    def test_validates_against_the_shared_schema(self, script: Path, tmp_path: Path) -> None:
        document = _compile(script, tmp_path)
        jsonschema.validate(document, SCHEMA)

    def test_dag_id_matches_the_filename(self, script: Path, tmp_path: Path) -> None:
        """`hpcctl result` artifacts are correlated by dag_id; drift makes traces ambiguous."""
        document = _compile(script, tmp_path)
        assert document["metadata"]["dag_id"] == script.stem.replace("_", "-")

    def test_carries_a_description(self, script: Path, tmp_path: Path) -> None:
        document = _compile(script, tmp_path)
        assert document["metadata"].get("description")
