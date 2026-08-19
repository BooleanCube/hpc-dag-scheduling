"""DAG file validation against the ``/shared`` serialization contract.

``hpcctl`` deliberately does **not** depend on the ``tasks`` package. It validates a serialized
DAG against ``/shared/dag_schema.json`` with ``jsonschema``, exactly as the C++ engine will.
Importing ``tasks`` would drag NumPy and the whole builder into a tool whose job is to manage EC2
instances, and would couple the two projects that the ``/shared`` contract exists to decouple.
If ``tasks`` happens to be importable, it is still not used: one validation path means one set of
error messages.
"""

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from hpcctl.config import default_schema_path
from hpcctl.errors import DagValidationError, HpcctlError
from hpcctl.exit_codes import ExitCode


class SchemaLoadError(HpcctlError):
    """The serialization contract itself is missing or malformed.

    A corrupted contract is a contract bug, not a DAG bug, so it gets its own error rather than
    being reported as a validation failure.
    """

    code = ExitCode.CONFIG


def load_schema(path: Path | None = None) -> dict[str, Any]:
    """Load the DAG JSON Schema from ``/shared/dag_schema.json``.

    ``check_schema`` is called once here so a corrupted contract is reported as a contract bug
    rather than surfacing later as a confusing DAG error.

    Args:
        path: Schema location. Defaults to the repository contract when omitted.

    Returns:
        The parsed schema document.

    Raises:
        SchemaLoadError: If the schema is absent, is not JSON, or is not a valid draft 2020-12
            schema.
    """
    if path is None:
        path = default_schema_path()
    if not path.is_file():
        raise SchemaLoadError(
            f"DAG schema not found at {path}",
            hint="Set HPCCTL_SCHEMA_PATH to the location of shared/dag_schema.json.",
        )
    try:
        with path.open(encoding="utf-8") as handle:
            schema: dict[str, Any] = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SchemaLoadError(f"DAG schema at {path} is not valid JSON: {exc}") from exc
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise SchemaLoadError(
            f"DAG schema at {path} is not a valid draft 2020-12 schema: {exc.message}"
        ) from exc
    return schema


def schema_version(schema: dict[str, Any]) -> str | None:
    """Extract the contract's own documented version, when it advertises one.

    Args:
        schema: A parsed schema document.

    Returns:
        The first documented example version, or ``None`` when absent.
    """
    examples = (
        schema.get("$defs", {})
        .get("metadata", {})
        .get("properties", {})
        .get("schema_version", {})
        .get("examples", [])
    )
    if isinstance(examples, list) and examples:
        first = examples[0]
        if isinstance(first, str):
            return first
    return None


def major_of(version: str) -> str:
    """Return the major component of a semantic version string.

    Args:
        version: A version such as ``"1.1.0"``.

    Returns:
        The text before the first dot.
    """
    return version.split(".", 1)[0]


def validate_dag_file(path: Path, *, schema_path: Path | None = None) -> dict[str, Any]:
    """Parse and validate a serialized DAG document.

    Errors are collected with ``sorted(iter_errors(doc), key=str)`` so every problem is reported
    at once and in a stable order, rather than one per invocation.

    Args:
        path: Path to the serialized DAG JSON file.
        schema_path: Optional override for the contract location.

    Returns:
        The parsed document, when valid.

    Raises:
        DagValidationError: If the file is not valid JSON, or does not conform to the schema.
            Carries every schema error, not just the first.
        SchemaLoadError: If the contract itself could not be loaded.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DagValidationError(f"cannot read DAG file {path}: {exc}") from exc

    try:
        document: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DagValidationError(
            f"{path} is not valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}",
            hint="Regenerate it with tasks.Graph.to_json().",
        ) from exc

    schema = load_schema(schema_path)
    validator = Draft202012Validator(schema)
    problems = [
        ("/" + "/".join(str(part) for part in error.absolute_path), error.message)
        for error in sorted(validator.iter_errors(document), key=str)
    ]
    if problems:
        raise DagValidationError(
            f"{path} does not conform to the DAG contract ({len(problems)} problem(s))",
            problems=problems,
            hint="The Python builder should have caught this before serializing.",
        )
    if not isinstance(document, dict):
        raise DagValidationError(f"{path} must contain a JSON object at the top level")
    return document


def check_version_compatibility(document: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Compare a document's declared contract version against the loaded contract.

    Schema 1.1.0's compatibility rule is that major mismatches are incompatible, and catching
    that locally is much cheaper than catching it after a job launches. This returns a warning
    rather than raising, because the schema is the authority: if the document validates
    structurally, it is loadable.

    Args:
        document: A validated DAG document.
        schema: The contract it was validated against.

    Returns:
        A warning message when the major versions differ, otherwise ``None``.
    """
    declared = document.get("metadata", {}).get("schema_version")
    contract = schema_version(schema)
    if not isinstance(declared, str) or contract is None:
        return None
    if major_of(declared) != major_of(contract):
        return (
            f"DAG declares schema_version {declared} but the contract documents {contract}; "
            "major versions differ and the engine will reject the document"
        )
    return None
