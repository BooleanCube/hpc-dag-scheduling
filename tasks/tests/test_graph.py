"""Tests for graph closure, reachability, topological ordering, and serialization."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tasks import Graph, InitNode, Node, ScaleNode
from tasks.graph import SCHEMA_VERSION

Conforms = Callable[[dict[str, Any]], None]


@pytest.fixture
def diamond() -> tuple[Node, Node]:
    """Return ``(scale_node, add_node)`` where the add consumes the scale twice."""
    a = InitNode((64, 32), seed=42, distribution="normal", name="lhs")
    b = InitNode((32, 16), seed=43)
    c = (a @ b) * 0.5
    return c, c + c


class TestConstruction:
    def test_empty_outputs_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one node"):
            Graph([], dag_id="empty")

    def test_duplicate_output_rejected(self) -> None:
        node = InitNode((2,), seed=1)
        with pytest.raises(ValueError, match="duplicate output node"):
            Graph([node, node], dag_id="dup")

    def test_non_node_output_rejected(self) -> None:
        with pytest.raises(TypeError, match="must contain Node instances"):
            Graph(["not-a-node"], dag_id="bad")  # type: ignore[list-item]

    @pytest.mark.parametrize(
        "dag_id",
        ["_leading-underscore", "has space", "", "x" * 129, "has/slash"],
        ids=["underscore", "space", "empty", "too-long", "slash"],
    )
    def test_illegal_dag_id_rejected(self, dag_id: str) -> None:
        with pytest.raises(ValueError, match="dag_id must match"):
            Graph([InitNode((2,), seed=1)], dag_id=dag_id)

    @pytest.mark.parametrize("dag_id", ["a", "0", "bench-matmul-001", "x" * 128])
    def test_legal_dag_id_accepted(self, dag_id: str) -> None:
        assert Graph([InitNode((2,), seed=1)], dag_id=dag_id).dag_id == dag_id

    def test_description_is_optional(self) -> None:
        graph = Graph([InitNode((2,), seed=1)], dag_id="d")
        assert graph.description is None

    def test_outputs_are_exposed_as_a_tuple(self) -> None:
        node = InitNode((2,), seed=1)
        assert Graph([node], dag_id="d").outputs == (node,)


class TestReachability:
    def test_unreachable_nodes_are_dropped(self) -> None:
        """Dead-code elimination lets a script explore alternatives without polluting output."""
        kept = InitNode((2,), seed=1)
        dead = InitNode((99,), seed=2)
        _ = dead * 3.0
        graph = Graph([kept], dag_id="dce")
        assert graph.nodes() == (kept,)

    def test_shared_node_is_collected_once(self, diamond: tuple[Node, Node]) -> None:
        scale, add = diamond
        graph = Graph([add], dag_id="diamond")
        assert graph.nodes().count(scale) == 1
        assert len(graph.nodes()) == 5

    def test_nodes_are_returned_in_construction_order(self) -> None:
        a = InitNode((2,), seed=1)
        b = InitNode((2,), seed=2)
        total = a + b
        assert Graph([total], dag_id="order").nodes() == (a, b, total)

    def test_multiple_outputs_are_all_reachable(self) -> None:
        a = InitNode((2,), seed=1)
        b = InitNode((2,), seed=2)
        graph = Graph([a, b], dag_id="two-outputs")
        assert set(graph.nodes()) == {a, b}


class TestTopologicalOrder:
    def test_every_operand_precedes_its_consumer(self, diamond: tuple[Node, Node]) -> None:
        _, add = diamond
        order = Graph([add], dag_id="topo").topological_order()
        position = {id(node): index for index, node in enumerate(order)}
        for node in order:
            for operand in node.inputs:
                assert position[id(operand)] < position[id(node)]

    def test_order_is_deterministic(self, diamond: tuple[Node, Node]) -> None:
        _, add = diamond
        graph = Graph([add], dag_id="topo")
        assert graph.topological_order() == graph.topological_order()

    def test_covers_every_reachable_node(self, diamond: tuple[Node, Node]) -> None:
        _, add = diamond
        graph = Graph([add], dag_id="topo")
        assert set(graph.topological_order()) == set(graph.nodes())


class TestIdAssignment:
    def test_worked_example_ids(self, diamond: tuple[Node, Node]) -> None:
        _, add = diamond
        document = Graph([add], dag_id="bench-matmul-001").serialize(include_timestamp=False)
        assert [node["id"] for node in document["nodes"]] == [
            "lhs",
            "init_1",
            "dot_product_2",
            "scale_3",
            "add_4",
        ]
        assert document["outputs"] == ["add_4"]

    def test_shared_operand_appears_twice_in_inputs(self, diamond: tuple[Node, Node]) -> None:
        _, add = diamond
        document = Graph([add], dag_id="d").serialize(include_timestamp=False)
        assert document["nodes"][-1]["inputs"] == ["scale_3", "scale_3"]

    def test_renumber_false_keeps_provisional_ids(self) -> None:
        a = InitNode((2,), seed=1)
        document = Graph([a], dag_id="d").serialize(renumber=False, include_timestamp=False)
        assert document["nodes"][0]["id"] == a._provisional_id

    def test_duplicate_user_names_rejected(self) -> None:
        a = InitNode((2,), seed=1, name="same")
        b = InitNode((2,), seed=2, name="same")
        with pytest.raises(ValueError, match="duplicate node ID 'same'"):
            Graph([a + b], dag_id="d").serialize()

    def test_user_name_colliding_with_a_generated_id_rejected(self) -> None:
        """A name that shadows the ID another node would generate is still a collision."""
        a = InitNode((2,), seed=1, name="init_1")
        b = InitNode((2,), seed=2)
        with pytest.raises(ValueError, match="duplicate node ID 'init_1'"):
            Graph([a + b], dag_id="d").serialize()

    def test_validate_surfaces_duplicate_names(self) -> None:
        a = InitNode((2,), seed=1, name="same")
        b = InitNode((2,), seed=2, name="same")
        with pytest.raises(ValueError, match="duplicate node ID"):
            Graph([a + b], dag_id="d").validate()

    def test_validate_passes_on_a_sound_graph(self, diamond: tuple[Node, Node]) -> None:
        _, add = diamond
        Graph([add], dag_id="d").validate()


class TestSerialization:
    def test_metadata_block(self) -> None:
        graph = Graph([InitNode((2,), seed=1)], dag_id="bench-1", description="a description")
        meta = graph.serialize()["metadata"]
        assert meta["schema_version"] == SCHEMA_VERSION
        assert meta["dag_id"] == "bench-1"
        assert meta["ordering"] == "topological"
        assert meta["description"] == "a description"
        assert meta["generator"].startswith("tasks-builder ")
        assert "created_at" in meta

    def test_description_omitted_when_absent(self) -> None:
        meta = Graph([InitNode((2,), seed=1)], dag_id="d").serialize()["metadata"]
        assert "description" not in meta

    def test_timestamp_omitted_on_request(self) -> None:
        meta = Graph([InitNode((2,), seed=1)], dag_id="d").serialize(include_timestamp=False)[
            "metadata"
        ]
        assert "created_at" not in meta

    def test_init_nodes_omit_the_inputs_key_entirely(self) -> None:
        """The schema sets `inputs: false` on init, so even an empty array is rejected."""
        document = Graph([InitNode((2,), seed=1)], dag_id="d").serialize()
        assert "inputs" not in document["nodes"][0]

    def test_non_init_nodes_carry_inputs(self) -> None:
        document = Graph([InitNode((2,), seed=1) * 2.0], dag_id="d").serialize()
        assert document["nodes"][1]["inputs"] == ["init_0"]

    def test_hints_included_by_default(self) -> None:
        document = Graph([InitNode((4, 4), seed=1)], dag_id="d").serialize()
        assert document["nodes"][0]["hints"] == {"est_flops": 16.0}

    def test_hints_omitted_on_request(self) -> None:
        document = Graph([InitNode((4, 4), seed=1)], dag_id="d").serialize(include_hints=False)
        assert "hints" not in document["nodes"][0]

    def test_no_key_is_ever_null(self, diamond: tuple[Node, Node]) -> None:
        """`additionalProperties: false` everywhere means absent beats null."""
        _, add = diamond
        document = Graph([add], dag_id="d").serialize()
        for node in document["nodes"]:
            assert None not in node.values()
        assert None not in document["metadata"].values()

    def test_scale_carries_its_factor(self) -> None:
        document = Graph([InitNode((2,), seed=1) * 0.25], dag_id="d").serialize()
        assert document["nodes"][1]["factor"] == 0.25

    def test_rank_zero_output_shape_is_an_empty_array(self) -> None:
        left, right = InitNode((5,), seed=1), InitNode((5,), seed=2)
        document = Graph([left @ right], dag_id="d").serialize()
        assert document["nodes"][-1]["output_shape"] == []

    def test_top_level_keys(self) -> None:
        document = Graph([InitNode((2,), seed=1)], dag_id="d").serialize()
        assert set(document) == {"metadata", "nodes", "outputs"}


class TestDeterminism:
    def test_two_serializations_are_byte_identical(self, diamond: tuple[Node, Node]) -> None:
        _, add = diamond
        graph = Graph([add], dag_id="stable")
        first = json.dumps(graph.serialize(include_timestamp=False), sort_keys=False)
        second = json.dumps(graph.serialize(include_timestamp=False), sort_keys=False)
        assert first == second

    def test_ids_derive_from_topological_position_not_construction_history(self) -> None:
        """Building throwaway nodes first must not shift the emitted IDs."""

        def build() -> dict[str, Any]:
            a = InitNode((2,), seed=1)
            b = InitNode((2,), seed=2)
            return Graph([a + b], dag_id="d").serialize(include_timestamp=False)

        _ = InitNode((9,), seed=99)  # advances the process-wide counter
        first = build()
        for _ in range(5):
            _ = InitNode((9,), seed=99)
        second = build()
        assert first == second


class TestToJson:
    def test_writes_conforming_utf8_json(
        self, tmp_path: Path, diamond: tuple[Node, Node], assert_conforms: Conforms
    ) -> None:
        _, add = diamond
        path = tmp_path / "dag.json"
        Graph([add], dag_id="bench-matmul-001").to_json(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        assert_conforms(document)

    def test_forwards_serialize_options(self, tmp_path: Path, diamond: tuple[Node, Node]) -> None:
        _, add = diamond
        path = tmp_path / "dag.json"
        Graph([add], dag_id="d").to_json(path, include_timestamp=False)
        assert "created_at" not in json.loads(path.read_text(encoding="utf-8"))["metadata"]

    def test_output_ends_with_a_newline(self, tmp_path: Path) -> None:
        path = tmp_path / "dag.json"
        Graph([InitNode((2,), seed=1)], dag_id="d").to_json(path)
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_rejects_non_finite_floats(self, tmp_path: Path) -> None:
        """Defence in depth: ScaleNode already blocks this at construction."""
        node = InitNode((2,), seed=1)
        scale = ScaleNode(node, 1.0)
        scale._factor = float("nan")
        with pytest.raises(ValueError, match="Out of range float"):
            Graph([scale], dag_id="d").to_json(tmp_path / "dag.json")

    def test_indent_is_configurable(self, tmp_path: Path) -> None:
        path = tmp_path / "dag.json"
        Graph([InitNode((2,), seed=1)], dag_id="d").to_json(path, indent=4)
        assert '\n    "metadata"' in path.read_text(encoding="utf-8")
