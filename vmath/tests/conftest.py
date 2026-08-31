"""Shared fixtures for the DAG builder test suite.

The most valuable thing in here is :func:`assert_conforms`, which validates a serialized
document against the real ``/shared/dag_schema.json``. Every test that builds a graph should
run its output through it, so the producer can never drift from the contract silently.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "shared" / "dag_schema.json"


@pytest.fixture(scope="session")
def schema() -> dict[str, Any]:
    """Load the serialization contract from /shared.

    Returns:
        The parsed JSON Schema document.
    """
    with SCHEMA_PATH.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


@pytest.fixture(scope="session")
def validator(schema: dict[str, Any]) -> Draft202012Validator:
    """Build a draft 2020-12 validator with RFC 3339 date-time checking enabled.

    Args:
        schema: The parsed contract.

    Returns:
        A validator that also enforces the ``format: date-time`` annotation.
    """
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


@pytest.fixture
def assert_conforms(
    validator: Draft202012Validator,
) -> Callable[[dict[str, Any]], None]:
    """Return a helper that asserts a document validates against the contract.

    Args:
        validator: The session-scoped schema validator.

    Returns:
        A callable raising ``AssertionError`` with every violation listed.
    """

    def _assert(document: dict[str, Any]) -> None:
        errors = sorted(validator.iter_errors(document), key=str)
        if errors:
            detail = "\n".join(
                f"  at {list(error.absolute_path)}: {error.message}" for error in errors
            )
            raise AssertionError(f"document does not conform to the contract:\n{detail}")

    return _assert
