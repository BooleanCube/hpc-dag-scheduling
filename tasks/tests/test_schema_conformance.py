"""Validate builder output against the real ``/shared/dag_schema.json``.

This is the highest-value test in the suite: it is the only place the producer is checked
against the actual contract the C++ engine consumes, rather than against this library's own
idea of that contract.

:class:`TestValidatorBites` is a negative control. Without it, a validator that silently
accepted everything would make every other test in this file pass for the wrong reason.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from tasks import (
    Graph,
    InitNode,
    ModNode,
    MultiplyNode,
    Node,
    ScaleNode,
    cos,
    cosh,
    cross,
    exp,
    matpow,
    powmod,
    sin,
    sinh,
)
from tasks.graph import SCHEMA_VERSION
from tasks.math import pow as tpow

Conforms = Callable[[dict[str, Any]], None]


def _matmul_chain() -> Graph:
    """Build the worked example: two sources, a contraction, a scale, and a shared add."""
    a = InitNode((64, 32), seed=42, distribution="normal", name="lhs")
    b = InitNode((32, 16), seed=43)
    c = (a @ b) * 0.5
    return Graph([c + c], dag_id="bench-matmul-001", description="worked example")


def _cross_float32() -> Graph:
    """Build a float32 cross product over ``zeros`` and ``ones`` sources."""
    u = InitNode((3,), seed=1, dtype="float32", distribution="zeros", name="u")
    v = InitNode((3,), seed=2, dtype="float32", distribution="ones", name="v")
    return Graph([cross(u, v)], dag_id="cross-1")


def _rank_zero_dot() -> Graph:
    """Build a vector-vector contraction, whose result is a rank-0 scalar."""
    u = InitNode((7,), seed=1)
    v = InitNode((7,), seed=2)
    return Graph([u @ v], dag_id="rank-zero-dot")


def _rank_zero_composition() -> Graph:
    """Build a rank-0 result and compose it onward through scale and add."""
    u = InitNode((3,), seed=1)
    v = InitNode((3,), seed=2)
    scalar = u @ v
    return Graph([scalar * 2.0 + scalar], dag_id="rank-zero-composition")


def _all_five_ops() -> Graph:
    """Build one graph exercising every op in the enum."""
    m = InitNode((3, 3), seed=1)
    u = InitNode((3,), seed=2)
    v = InitNode((3,), seed=3)
    contracted = m @ u
    crossed = cross(u, v)
    return Graph([(contracted + crossed) * 2.0], dag_id="all-ops")


def _mixed_dtypes() -> Graph:
    """Build a graph whose operands promote from float32 to float64."""
    a = InitNode((4, 4), seed=1, dtype="float32")
    b = InitNode((4, 4), seed=2, dtype="float64")
    return Graph([a + b], dag_id="mixed-dtypes")


def _every_distribution() -> Graph:
    """Build a graph containing one source per PRNG distribution."""
    sources = [
        InitNode((2,), seed=index, distribution=dist)  # type: ignore[arg-type]
        for index, dist in enumerate(("uniform", "normal", "zeros", "ones"))
    ]
    total: Node = sources[0]
    for source in sources[1:]:
        total = total + source
    return Graph([total], dag_id="distributions")


def _multiple_outputs() -> Graph:
    """Build a graph that materializes two independent tensors."""
    a = InitNode((2,), seed=1)
    b = InitNode((3,), seed=2)
    return Graph([a * 2.0, b * 3.0], dag_id="two-outputs")


def _negative_and_division() -> Graph:
    """Build a graph using the negation and division sugar."""
    a = InitNode((2, 2), seed=1)
    return Graph([-a + (a / 4)], dag_id="sugar")


def _boundary_values() -> Graph:
    """Build a graph at the edges of the schema's numeric ranges."""
    from tasks.dtypes import UINT64_MAX

    a = InitNode((1,) * 8, seed=UINT64_MAX, name="max_seed")
    b = InitNode((1,) * 8, seed=0, name="zero_seed")
    return Graph([a + b], dag_id="0")


def _deep_chain() -> Graph:
    """Build a long linear chain of scale nodes."""
    tail: Node = InitNode((2,), seed=1)
    for _ in range(50):
        tail = tail * 1.5
    return Graph([tail], dag_id="deep-chain")


def _elementwise_multiply() -> Graph:
    """Build a graph using the v1.2.0 multiply primitive."""
    a = InitNode((4, 4), seed=1, name="a")
    b = InitNode((4, 4), seed=2, name="b")
    return Graph([MultiplyNode(a, b, label="hadamard")], dag_id="multiply-1")


def _mod_chain() -> Graph:
    """Build a graph using the v1.2.0 mod primitive, including a non-integer modulus."""
    a = InitNode((3,), seed=1)
    return Graph([ModNode(a % 7, 2.5)], dag_id="mod-1")


def _subtraction() -> Graph:
    """Build the two-node subtraction expansion."""
    a = InitNode((3,), seed=1)
    b = InitNode((3,), seed=2)
    return Graph([a - b], dag_id="subtract-1")


def _rank_zero_init() -> Graph:
    """Build a graph with a genuinely rank-0 init source, legal only from v1.2.0."""
    scalar = InitNode((), seed=7, name="s")
    return Graph([scalar * scalar], dag_id="rank0-init")


def _series_sin() -> Graph:
    """Build a 29-node sin expansion."""
    return Graph([sin(InitNode((6,), seed=1), terms=10)], dag_id="sin-10")


def _series_cos_rank0() -> Graph:
    """Build cos over a rank-0 contraction, exercising the rank-0 ones constant."""
    u = InitNode((3,), seed=1)
    v = InitNode((3,), seed=2)
    return Graph([cos(u @ v, terms=8)], dag_id="cos-rank0")


def _series_exp() -> Graph:
    """Build an exp expansion."""
    return Graph([exp(InitNode((2, 2), seed=1), terms=12)], dag_id="exp-12")


def _series_hyperbolic() -> Graph:
    """Build sinh and cosh over one source, as two outputs."""
    x = InitNode((4,), seed=1)
    return Graph([sinh(x, terms=6), cosh(x, terms=6)], dag_id="hyperbolic")


def _pow_large() -> Graph:
    """Build the ten-node binary exponentiation for n = 1024."""
    return Graph([tpow(InitNode((3,), seed=1), 1024)], dag_id="pow-1024")


def _pow_zero() -> Graph:
    """Build the surprising pow(x, 0) case, whose DAG never mentions x."""
    return Graph([tpow(InitNode((3,), seed=1), 0)], dag_id="pow-0")


def _powmod_large() -> Graph:
    """Build the 21-node powmod chain."""
    return Graph([powmod(InitNode((4,), seed=1), 1024, 1000003)], dag_id="powmod-1024")


def _matpow_large() -> Graph:
    """Build the six-node matpow chain."""
    return Graph([matpow(InitNode((4, 4), seed=1), 64)], dag_id="matpow-64")


def _every_op() -> Graph:
    """Build one graph touching all seven ops in the enum."""
    m = InitNode((3, 3), seed=1)
    u = InitNode((3,), seed=2)
    v = InitNode((3,), seed=3)
    combined = ((m @ u) + u.cross(v)) * u
    return Graph([(combined * 2.0) % 11], dag_id="every-op")


def _composite_mixture() -> Graph:
    """Build a graph combining several composites, exercising the no-CSE policy."""
    x = InitNode((5,), seed=1)
    return Graph([sin(x, terms=6) + cos(x, terms=6) + tpow(x, 9)], dag_id="mixture")


BUILDERS: list[Callable[[], Graph]] = [
    _matmul_chain,
    _cross_float32,
    _rank_zero_dot,
    _rank_zero_composition,
    _all_five_ops,
    _mixed_dtypes,
    _every_distribution,
    _multiple_outputs,
    _negative_and_division,
    _boundary_values,
    _deep_chain,
    _elementwise_multiply,
    _mod_chain,
    _subtraction,
    _rank_zero_init,
    _series_sin,
    _series_cos_rank0,
    _series_exp,
    _series_hyperbolic,
    _pow_large,
    _pow_zero,
    _powmod_large,
    _matpow_large,
    _every_op,
    _composite_mixture,
]

OPTIONS: list[dict[str, bool]] = [
    {},
    {"include_hints": False},
    {"include_timestamp": False},
    {"renumber": False},
    {"renumber": False, "include_hints": False, "include_timestamp": False},
]


class TestSchemaItself:
    def test_is_a_valid_draft_2020_12_schema(self, schema: dict[str, Any]) -> None:
        Draft202012Validator.check_schema(schema)

    def test_permits_rank_zero_shapes(self, schema: dict[str, Any]) -> None:
        """Guards the 1.1.0 amendment this builder depends on for vector-vector dot."""
        assert schema["$defs"]["shape"]["minItems"] == 0

    def test_init_no_longer_pins_a_rank_floor(self, schema: dict[str, Any]) -> None:
        """v1.2.0 lifted the rank-1 floor on init: multiply and mod consume rank-0 operands.

        This replaces two v1.1.0 tests that asserted the opposite. Their premise -- that nothing
        could consume a rank-0 source -- is what the elementwise ops invalidated.
        """
        assert "minItems" not in schema["$defs"]["node"]["properties"]["shape"]
        init_branch = schema["$defs"]["node"]["allOf"][0]["then"]
        assert "output_shape" not in init_branch["properties"]

    def test_op_enum_carries_the_v12_primitives(self, schema: dict[str, Any]) -> None:
        assert schema["$defs"]["node"]["properties"]["op"]["enum"] == [
            "init",
            "add",
            "multiply",
            "scale",
            "mod",
            "dot_product",
            "cross_product",
        ]

    def test_modulus_is_constrained_positive(self, schema: dict[str, Any]) -> None:
        assert schema["$defs"]["node"]["properties"]["modulus"]["exclusiveMinimum"] == 0

    def test_emitted_version_matches_the_contract_major(self) -> None:
        assert SCHEMA_VERSION.split(".")[0] == "1"

    def test_builder_targets_1_2_0(self) -> None:
        assert SCHEMA_VERSION == "1.2.0"

    def test_metadata_declares_the_targeted_version(self) -> None:
        assert _matmul_chain().serialize()["metadata"]["schema_version"] == "1.2.0"

    def test_vector_vector_dot_serializes_an_empty_output_shape(self) -> None:
        document = _rank_zero_dot().serialize()
        assert document["nodes"][-1]["op"] == "dot_product"
        assert document["nodes"][-1]["output_shape"] == []

    def test_rank_zero_composes_through_scale_and_add(self) -> None:
        document = _rank_zero_composition().serialize()
        shapes = {node["op"]: node["output_shape"] for node in document["nodes"]}
        assert shapes["dot_product"] == []
        assert shapes["scale"] == []
        assert shapes["add"] == []


class TestValidatorBites:
    """Negative control: prove the validator rejects bad documents."""

    def test_rejects_an_empty_document(self, assert_conforms: Conforms) -> None:
        with pytest.raises(AssertionError):
            assert_conforms({})

    def test_rejects_init_carrying_inputs(self, assert_conforms: Conforms) -> None:
        document = _matmul_chain().serialize()
        document["nodes"][0]["inputs"] = []
        with pytest.raises(AssertionError):
            assert_conforms(document)

    def test_rejects_a_stray_field(self, assert_conforms: Conforms) -> None:
        document = _matmul_chain().serialize()
        document["nodes"][0]["device"] = "gpu0"
        with pytest.raises(AssertionError):
            assert_conforms(document)

    def test_rejects_an_unknown_op(self, assert_conforms: Conforms) -> None:
        document = _matmul_chain().serialize()
        document["nodes"][2]["op"] = "transpose"
        with pytest.raises(AssertionError):
            assert_conforms(document)

    def test_rejects_a_malformed_timestamp(self, assert_conforms: Conforms) -> None:
        """Confirms the RFC 3339 format checker is actually wired up."""
        document = _matmul_chain().serialize()
        document["metadata"]["created_at"] = "yesterday"
        with pytest.raises(AssertionError):
            assert_conforms(document)

    def test_rejects_a_stray_metadata_field(self, assert_conforms: Conforms) -> None:
        document = _matmul_chain().serialize()
        document["metadata"]["cluster_ip"] = "10.0.0.1"
        with pytest.raises(AssertionError):
            assert_conforms(document)


class TestBuilderOutputConforms:
    @pytest.mark.parametrize("builder", BUILDERS, ids=lambda fn: fn.__name__)
    def test_default_serialization(
        self, builder: Callable[[], Graph], assert_conforms: Conforms
    ) -> None:
        assert_conforms(builder().serialize())

    @pytest.mark.parametrize("builder", BUILDERS, ids=lambda fn: fn.__name__)
    @pytest.mark.parametrize("options", OPTIONS, ids=lambda opt: repr(sorted(opt)))
    def test_every_serialization_option_combination(
        self,
        builder: Callable[[], Graph],
        options: dict[str, bool],
        assert_conforms: Conforms,
    ) -> None:
        assert_conforms(builder().serialize(**options))

    @pytest.mark.parametrize("builder", BUILDERS, ids=lambda fn: fn.__name__)
    def test_round_trips_through_disk(
        self, builder: Callable[[], Graph], tmp_path: Path, assert_conforms: Conforms
    ) -> None:
        path = tmp_path / "dag.json"
        builder().to_json(path)
        assert_conforms(json.loads(path.read_text(encoding="utf-8")))


class TestContractInvariants:
    @pytest.mark.parametrize("builder", BUILDERS, ids=lambda fn: fn.__name__)
    def test_every_node_declares_shape_and_dtype(self, builder: Callable[[], Graph]) -> None:
        for node in builder().serialize()["nodes"]:
            assert "output_shape" in node
            assert node["dtype"] in {"float32", "float64"}

    @pytest.mark.parametrize("builder", BUILDERS, ids=lambda fn: fn.__name__)
    def test_node_ids_are_unique(self, builder: Callable[[], Graph]) -> None:
        """JSON Schema cannot express this, so the engine relies on the producer."""
        ids = [node["id"] for node in builder().serialize()["nodes"]]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("builder", BUILDERS, ids=lambda fn: fn.__name__)
    def test_inputs_reference_earlier_nodes_only(self, builder: Callable[[], Graph]) -> None:
        """The declared `topological` ordering lets the engine resolve in one forward pass."""
        seen: set[str] = set()
        for node in builder().serialize()["nodes"]:
            for reference in node.get("inputs", []):
                assert reference in seen
            seen.add(node["id"])

    @pytest.mark.parametrize("builder", BUILDERS, ids=lambda fn: fn.__name__)
    def test_outputs_name_declared_nodes(self, builder: Callable[[], Graph]) -> None:
        document = builder().serialize()
        ids = {node["id"] for node in document["nodes"]}
        assert set(document["outputs"]) <= ids

    @pytest.mark.parametrize("builder", BUILDERS, ids=lambda fn: fn.__name__)
    def test_init_nodes_agree_with_their_own_output_shape(
        self, builder: Callable[[], Graph]
    ) -> None:
        for node in builder().serialize()["nodes"]:
            if node["op"] == "init":
                assert node["shape"] == node["output_shape"]

    @pytest.mark.parametrize("builder", BUILDERS, ids=lambda fn: fn.__name__)
    def test_flop_hints_are_non_negative(self, builder: Callable[[], Graph]) -> None:
        for node in builder().serialize()["nodes"]:
            assert node["hints"]["est_flops"] >= 0


class TestArchitectFixtures:
    """The two hand-written fixtures from the schema review, as a cross-check."""

    def test_reference_matmul_fixture(self, assert_conforms: Conforms) -> None:
        assert_conforms(
            {
                "metadata": {
                    "schema_version": "1.0.0",
                    "dag_id": "bench-matmul-001",
                    "ordering": "topological",
                    "created_at": "2026-08-16T12:00:00Z",
                    "generator": "tasks-builder 0.1.0",
                },
                "nodes": [
                    {
                        "id": "a",
                        "op": "init",
                        "output_shape": [64, 32],
                        "dtype": "float64",
                        "seed": 42,
                        "shape": [64, 32],
                        "distribution": "normal",
                    },
                    {
                        "id": "b",
                        "op": "init",
                        "output_shape": [32, 16],
                        "dtype": "float64",
                        "seed": 43,
                        "shape": [32, 16],
                        "distribution": "uniform",
                    },
                    {
                        "id": "c",
                        "op": "dot_product",
                        "output_shape": [64, 16],
                        "dtype": "float64",
                        "inputs": ["a", "b"],
                        "hints": {"est_flops": 65536.0, "priority": 3},
                    },
                ],
                "outputs": ["c"],
            }
        )

    def test_our_output_matches_the_fixture_shape_for_the_same_maths(self) -> None:
        """Our builder derives the same output_shape and est_flops the fixture hard-codes."""
        a = InitNode((64, 32), seed=42, distribution="normal", name="a")
        b = InitNode((32, 16), seed=43, name="b")
        document = Graph([a @ b], dag_id="bench-matmul-001").serialize()
        contraction = document["nodes"][2]
        assert contraction["output_shape"] == [64, 16]
        assert contraction["hints"]["est_flops"] == 65536.0


class TestNonFiniteRejection:
    def test_nan_factor_never_reaches_the_wire(self) -> None:
        node = InitNode((2,), seed=1)
        scale = ScaleNode(node, 1.0)
        scale._factor = float("inf")
        with pytest.raises(ValueError, match="Out of range float"):
            json.dumps(Graph([scale], dag_id="d").serialize(), allow_nan=False)
