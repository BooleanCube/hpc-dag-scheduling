"""Shared fixtures for the hpcctl test suite.

Two things here are load-bearing. :func:`clean_env` strips every variable the CLI reads, so a
developer's own exported ``AWS_REGION`` cannot make a test pass that would fail in CI -- and it is
what lets the suite assert the P1 guarantee that dry-run works with *nothing* set. The autouse
chdir keeps generated artifacts inside ``tmp_path`` instead of polluting the repository.

No test in this suite may require network, credentials, ``pcluster``, or ``aws``.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from hpcctl.config import Settings, load_settings

HPCCTL_VARS: tuple[str, ...] = (
    "HPCCTL_CLUSTER_NAME",
    "HPCCTL_OS",
    "HPCCTL_KEY_NAME",
    "HPCCTL_HEAD_SUBNET_ID",
    "HPCCTL_COMPUTE_SUBNET_ID",
    "HPCCTL_HEAD_INSTANCE_TYPE",
    "HPCCTL_COMPUTE_INSTANCE_TYPE",
    "HPCCTL_QUEUE_NAME",
    "HPCCTL_MIN_NODES",
    "HPCCTL_MAX_NODES",
    "HPCCTL_SHARED_DIR",
    "HPCCTL_SHARED_VOLUME_GB",
    "HPCCTL_BOOTSTRAP_BUCKET",
    "HPCCTL_BOOTSTRAP_PREFIX",
    "HPCCTL_HEAD_NODE_HOST",
    "HPCCTL_SSH_USER",
    "HPCCTL_SSH_KEY_PATH",
    "HPCCTL_ENGINE_BUILD_DIR",
    "HPCCTL_REMOTE_ENGINE_DIR",
    "HPCCTL_REMOTE_DAG_DIR",
    "HPCCTL_ENGINE_BINARY",
    "HPCCTL_NTASKS",
    "HPCCTL_NODES",
    "HPCCTL_TIME_LIMIT",
    "HPCCTL_SCHEMA_PATH",
    "HPCCTL_RUN_DIR",
    "HPCCTL_DRY_RUN",
)
"""Every ``HPCCTL_``-prefixed variable the CLI reads."""

FOREIGN_VARS: tuple[str, ...] = ("AWS_REGION", "AWS_DEFAULT_REGION", "NO_COLOR")
"""Non-namespaced variables the CLI also consults."""

ALL_VARS: tuple[str, ...] = (*HPCCTL_VARS, *FOREIGN_VARS)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every variable the CLI reads, for direct calls into the library.

    Args:
        monkeypatch: pytest's environment patcher.
    """
    for name in ALL_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test in a scratch directory so artifacts never touch the repository.

    Args:
        tmp_path: Per-test scratch directory.
        monkeypatch: pytest's chdir helper.
    """
    monkeypatch.chdir(tmp_path)


def blank_env(**overrides: str) -> dict[str, str | None]:
    """Build a CliRunner ``env`` mapping that clears everything, then applies overrides.

    ``CliRunner`` merges ``env`` into the real environment rather than replacing it, and a
    ``None`` value removes a variable. Passing every name explicitly is what makes a CLI test
    independent of the developer's shell.

    Args:
        **overrides: Variables to set for this invocation.

    Returns:
        A mapping suitable for ``CliRunner.invoke(env=...)``.
    """
    env: dict[str, str | None] = dict.fromkeys(ALL_VARS)
    env.update(overrides)
    return env


@pytest.fixture
def runner() -> CliRunner:
    """Return a Typer CLI runner.

    Returns:
        A runner whose ``Result`` keeps stdout and stderr separate.
    """
    return CliRunner()


@pytest.fixture
def settings() -> Settings:
    """Return dry-run settings resolved from an empty environment.

    Returns:
        Settings whose required values are all placeholders.
    """
    return load_settings(live=False)


@pytest.fixture
def schema_path() -> Path:
    """Return the path to the repository's DAG contract.

    Returns:
        The location of ``shared/dag_schema.json``.
    """
    return Path(__file__).resolve().parents[2] / "shared" / "dag_schema.json"


def valid_dag_document() -> dict[str, Any]:
    """Build a DAG document that conforms to contract 1.1.0.

    Written by hand rather than generated with the ``tasks`` package: ``hpcctl`` validates the
    contract, not the builder, and must not import ``tasks``.

    Returns:
        A valid five-node DAG document.
    """
    return {
        "metadata": {
            "schema_version": "1.1.0",
            "dag_id": "bench-matmul-001",
            "ordering": "topological",
            "created_at": "2026-08-16T12:00:00Z",
            "generator": "tasks-builder 0.1.0",
        },
        "nodes": [
            {
                "id": "lhs",
                "op": "init",
                "output_shape": [64, 32],
                "dtype": "float64",
                "seed": 42,
                "shape": [64, 32],
                "distribution": "normal",
            },
            {
                "id": "init_1",
                "op": "init",
                "output_shape": [32, 16],
                "dtype": "float64",
                "seed": 43,
                "shape": [32, 16],
                "distribution": "uniform",
            },
            {
                "id": "dot_product_2",
                "op": "dot_product",
                "output_shape": [64, 16],
                "dtype": "float64",
                "inputs": ["lhs", "init_1"],
                "hints": {"est_flops": 65536.0},
            },
            {
                "id": "scale_3",
                "op": "scale",
                "output_shape": [64, 16],
                "dtype": "float64",
                "inputs": ["dot_product_2"],
                "factor": 0.5,
            },
            {
                "id": "add_4",
                "op": "add",
                "output_shape": [64, 16],
                "dtype": "float64",
                "inputs": ["scale_3", "scale_3"],
            },
        ],
        "outputs": ["add_4"],
    }


@pytest.fixture
def valid_dag(tmp_path: Path) -> Path:
    """Write a conforming DAG document to disk.

    Args:
        tmp_path: Per-test scratch directory.

    Returns:
        Path to the written DAG file.
    """
    path = tmp_path / "bench-matmul-001.json"
    path.write_text(json.dumps(valid_dag_document(), indent=2), encoding="utf-8")
    return path


@pytest.fixture
def build_dir(tmp_path: Path) -> Path:
    """Create a non-empty stand-in for a compiled engine build directory.

    Args:
        tmp_path: Per-test scratch directory.

    Returns:
        Path to a directory containing one fake binary.
    """
    target = tmp_path / "build"
    (target / "bin").mkdir(parents=True)
    (target / "bin" / "engine").write_bytes(b"\x7fELF fake binary")
    return target


@pytest.fixture
def wide_console(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the terminal width so rendering assertions are reproducible.

    Args:
        monkeypatch: pytest's environment patcher.

    Yields:
        ``None``; the width applies for the duration of the test.
    """
    monkeypatch.setenv("COLUMNS", "80")
    yield
