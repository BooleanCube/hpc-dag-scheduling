"""Tests for DAG validation against the /shared contract.

``hpcctl`` validates the contract, not the builder: these tests must never import ``tasks``.
Fixtures are hand-written documents, so a bug in the Python builder cannot mask a bug here.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import valid_dag_document

from hpcctl.errors import DagValidationError
from hpcctl.validation import (
    SchemaLoadError,
    check_version_compatibility,
    load_schema,
    major_of,
    schema_version,
    validate_dag_file,
)


def write(tmp_path: Path, document: Any, name: str = "dag.json") -> Path:
    """Write a document to disk as JSON.

    Args:
        tmp_path: Scratch directory.
        document: Any JSON-serializable value.
        name: File name to use.

    Returns:
        The written path.
    """
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class TestDoesNotDependOnTasks:
    def test_tasks_is_not_imported_by_hpcctl(self) -> None:
        """Importing tasks would drag NumPy into a tool that manages EC2 instances."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import hpcctl.validation, sys; "
                "print('tasks' in sys.modules or 'numpy' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"


class TestSchemaLoading:
    def test_loads_the_repository_contract(self) -> None:
        schema = load_schema()
        assert schema["title"].startswith("HPC Mathematical DAG")

    def test_explicit_path(self, schema_path: Path) -> None:
        assert load_schema(schema_path)["$defs"]["shape"]["minItems"] == 0

    def test_missing_schema_is_a_contract_error(self, tmp_path: Path) -> None:
        with pytest.raises(SchemaLoadError, match="not found"):
            load_schema(tmp_path / "nope.json")

    def test_malformed_schema_is_a_contract_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SchemaLoadError, match="not valid JSON"):
            load_schema(path)

    def test_invalid_draft_schema_is_a_contract_error(self, tmp_path: Path) -> None:
        """check_schema runs at load so a broken contract is not blamed on the DAG."""
        path = write(tmp_path, {"type": "not-a-type"}, "schema.json")
        with pytest.raises(SchemaLoadError, match="not a valid draft 2020-12"):
            load_schema(path)

    def test_schema_load_error_exits_config(self, tmp_path: Path) -> None:
        with pytest.raises(SchemaLoadError) as excinfo:
            load_schema(tmp_path / "nope.json")
        assert int(excinfo.value.code) == 3


class TestValidDocuments:
    def test_conforming_dag_passes(self, valid_dag: Path) -> None:
        document = validate_dag_file(valid_dag)
        assert document["metadata"]["dag_id"] == "bench-matmul-001"

    def test_returns_the_parsed_document(self, valid_dag: Path) -> None:
        assert len(validate_dag_file(valid_dag)["nodes"]) == 5

    def test_rank_zero_output_shape_is_accepted(self, tmp_path: Path) -> None:
        """Contract 1.1.0 admits the empty shape array for a vector-vector dot."""
        document = valid_dag_document()
        document["nodes"] = [
            {
                "id": "u",
                "op": "init",
                "output_shape": [3],
                "dtype": "float64",
                "seed": 1,
                "shape": [3],
                "distribution": "uniform",
            },
            {
                "id": "v",
                "op": "init",
                "output_shape": [3],
                "dtype": "float64",
                "seed": 2,
                "shape": [3],
                "distribution": "uniform",
            },
            {
                "id": "d",
                "op": "dot_product",
                "output_shape": [],
                "dtype": "float64",
                "inputs": ["u", "v"],
            },
        ]
        document["outputs"] = ["d"]
        assert validate_dag_file(write(tmp_path, document))["outputs"] == ["d"]


class TestMalformedJson:
    def test_raises_dag_validation_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"metadata": ', encoding="utf-8")
        with pytest.raises(DagValidationError, match="not valid JSON"):
            validate_dag_file(path)

    def test_message_carries_line_and_column(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{\n  "a": ,\n}', encoding="utf-8")
        with pytest.raises(DagValidationError, match=r"line \d+, column \d+"):
            validate_dag_file(path)

    def test_exit_code_is_four(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("nope", encoding="utf-8")
        with pytest.raises(DagValidationError) as excinfo:
            validate_dag_file(path)
        assert int(excinfo.value.code) == 4

    def test_unreadable_file(self, tmp_path: Path) -> None:
        with pytest.raises(DagValidationError, match="cannot read"):
            validate_dag_file(tmp_path / "absent.json")


class TestSchemaViolations:
    def test_bad_op_is_rejected(self, tmp_path: Path) -> None:
        document = valid_dag_document()
        document["nodes"][2]["op"] = "transpose"
        with pytest.raises(DagValidationError) as excinfo:
            validate_dag_file(write(tmp_path, document))
        assert excinfo.value.problems

    def test_error_carries_a_json_pointer_path(self, tmp_path: Path) -> None:
        document = valid_dag_document()
        document["nodes"][2]["op"] = "transpose"
        with pytest.raises(DagValidationError) as excinfo:
            validate_dag_file(write(tmp_path, document))
        paths = [pointer for pointer, _ in excinfo.value.problems]
        assert any("nodes/2" in pointer for pointer in paths)

    def test_init_carrying_inputs_is_rejected(self, tmp_path: Path) -> None:
        document = valid_dag_document()
        document["nodes"][0]["inputs"] = []
        with pytest.raises(DagValidationError):
            validate_dag_file(write(tmp_path, document))

    def test_missing_seed_is_rejected(self, tmp_path: Path) -> None:
        document = valid_dag_document()
        del document["nodes"][0]["seed"]
        with pytest.raises(DagValidationError):
            validate_dag_file(write(tmp_path, document))

    def test_scale_without_factor_is_rejected(self, tmp_path: Path) -> None:
        document = valid_dag_document()
        del document["nodes"][3]["factor"]
        with pytest.raises(DagValidationError):
            validate_dag_file(write(tmp_path, document))

    def test_rank_zero_init_shape_is_accepted_as_of_contract_120(self, tmp_path: Path) -> None:
        """Contract 1.2.0 lifted the init rank floor; this asserted the opposite under 1.1.0.

        The floor existed because nothing could consume a rank-0 source. The elementwise
        ``multiply`` and ``mod`` ops added in 1.2.0 can, and the composite expansions need a
        rank-0 ones constant, so the restriction was reversed rather than worked around.
        """
        document = valid_dag_document()
        document["nodes"][0]["shape"] = []
        document["nodes"][0]["output_shape"] = []
        # The dot_product consuming it now sees a rank-0 operand, which is its own error, so
        # validate the init node in isolation.
        document["nodes"] = [document["nodes"][0]]
        document["outputs"] = [document["nodes"][0]["id"]]
        assert validate_dag_file(write(tmp_path, document))["nodes"][0]["shape"] == []

    def test_multiple_problems_are_all_reported(self, tmp_path: Path) -> None:
        """One error per run would mean N runs to fix N problems."""
        document = valid_dag_document()
        document["nodes"][0]["op"] = "transpose"
        document["metadata"]["ordering"] = "insertion"
        del document["outputs"]
        with pytest.raises(DagValidationError) as excinfo:
            validate_dag_file(write(tmp_path, document))
        assert len(excinfo.value.problems) >= 3

    def test_problem_order_is_stable(self, tmp_path: Path) -> None:
        document = valid_dag_document()
        document["nodes"][0]["op"] = "transpose"
        document["metadata"]["ordering"] = "insertion"
        path = write(tmp_path, document)
        first = _problems(path)
        second = _problems(path)
        assert first == second

    def test_top_level_non_object_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DagValidationError):
            validate_dag_file(write(tmp_path, [1, 2, 3]))

    def test_empty_nodes_array_is_rejected(self, tmp_path: Path) -> None:
        document = valid_dag_document()
        document["nodes"] = []
        with pytest.raises(DagValidationError):
            validate_dag_file(write(tmp_path, document))

    def test_stray_metadata_field_is_rejected(self, tmp_path: Path) -> None:
        document = valid_dag_document()
        document["metadata"]["cluster_ip"] = "10.0.0.1"
        with pytest.raises(DagValidationError):
            validate_dag_file(write(tmp_path, document))


def _problems(path: Path) -> list[tuple[str, str]]:
    """Collect the reported problems for a known-invalid DAG.

    Args:
        path: Path to an invalid DAG file.

    Returns:
        The problem list from the raised error.
    """
    try:
        validate_dag_file(path)
    except DagValidationError as exc:
        return list(exc.problems)
    raise AssertionError("expected validation to fail")


class TestVersionCompatibility:
    def test_contract_documents_its_own_version(self) -> None:
        assert schema_version(load_schema()) is not None

    def test_major_of(self) -> None:
        assert major_of("1.1.0") == "1"
        assert major_of("2.0.0") == "2"

    def test_matching_major_produces_no_warning(self, valid_dag: Path) -> None:
        schema = load_schema()
        document = validate_dag_file(valid_dag)
        assert check_version_compatibility(document, schema) is None

    def test_differing_major_produces_a_warning(self, valid_dag: Path) -> None:
        schema = load_schema()
        document = validate_dag_file(valid_dag)
        document["metadata"]["schema_version"] = "2.0.0"
        warning = check_version_compatibility(document, schema)
        assert warning is not None
        assert "2.0.0" in warning

    def test_a_newer_minor_is_compatible(self, valid_dag: Path) -> None:
        """Minor and patch differences are backward compatible by the contract's own rule."""
        schema = load_schema()
        document = validate_dag_file(valid_dag)
        document["metadata"]["schema_version"] = "1.9.3"
        assert check_version_compatibility(document, schema) is None

    def test_absent_metadata_is_tolerated(self) -> None:
        assert check_version_compatibility({}, load_schema()) is None
