"""Adversarial tests: deliberate attempts to make the builder emit an unsound DAG.

Where the other suites confirm the documented behaviour, this one attacks it. Each test asks
one question: can a user get a mathematically impossible document past Python and into the C++
engine? Per CLAUDE.md the answer must always be no, and the failure must arrive as one of the
four :class:`~tasks.exceptions.DagBuildError` subclasses, at the earliest moment it is knowable.

Two regression classes at the end pin bugs this pass uncovered:
:class:`TestRewireStalenessRegression` (a rewired operand left consumers declaring a shape their
inputs no longer produced) and :class:`TestFlopEstimateOverflowRegression` (an ``OverflowError``
escaping ``serialize`` from a scheduling hint).
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tasks import (
    AddNode,
    CrossProductNode,
    CyclicDependencyError,
    DagBuildError,
    DimensionalityError,
    DotProductNode,
    Graph,
    InitNode,
    Node,
    ScaleNode,
    ShapeMismatchError,
    UninitializedNodeError,
    cross,
)
from tasks.dtypes import MAX_RANK, UINT64_MAX
from tasks.graph import SCHEMA_VERSION

Conforms = Callable[[dict[str, Any]], None]

# Chain lengths chosen to exceed CPython's default recursion limit of 1000, so a recursive
# reachability walk, topological sort, or cycle search fails loudly instead of passing by luck.
DEEP = 1500
LONG_CYCLE = 1200


def _scalar_node() -> Node:
    """Build a rank-0 node, the only kind schema 1.1.0 can produce.

    Returns:
        A ``dot_product`` of two length-4 vectors, whose output shape is ``()``.
    """
    return InitNode((4,), seed=1) @ InitNode((4,), seed=2)


def _chain(length: int) -> Node:
    """Build a linear chain of scale nodes above one init node.

    Args:
        length: Number of scale nodes to stack.

    Returns:
        The final node in the chain.
    """
    node: Node = InitNode((2, 2), seed=1)
    for _ in range(length):
        node = node * 1.5
    return node


def _walk(document: Any) -> list[Any]:
    """Flatten every value in a nested JSON document.

    Args:
        document: A parsed JSON value.

    Returns:
        Every scalar leaf reachable inside it.
    """
    if isinstance(document, dict):
        return [leaf for value in document.values() for leaf in _walk(value)]
    if isinstance(document, list):
        return [leaf for value in document for leaf in _walk(value)]
    return [document]


def _assert_topological(document: dict[str, Any]) -> None:
    """Assert every node's operands appear earlier in the ``nodes`` array than the node itself.

    Args:
        document: A serialized DAG document.
    """
    seen: set[str] = set()
    for node in document["nodes"]:
        for operand in node.get("inputs", []):
            assert operand in seen, f"{node['id']} references {operand} before it is defined"
        seen.add(node["id"])


class TestUserMandatedCases:
    """The two scenarios the user called out by name."""

    def test_adding_2x2_to_3x3_raises_at_the_expression(self) -> None:
        a = InitNode((2, 2), seed=1)
        b = InitNode((3, 3), seed=2)
        with pytest.raises(ShapeMismatchError) as excinfo:
            a + b
        message = str(excinfo.value)
        assert "(2, 2)" in message
        assert "(3, 3)" in message

    def test_the_2x2_plus_3x3_failure_needs_no_graph_at_all(self) -> None:
        """Nothing is constructed, closed, or serialized -- the operator itself refuses."""
        a = InitNode((2, 2), seed=1)
        b = InitNode((3, 3), seed=2)
        with pytest.raises(ShapeMismatchError):
            AddNode(a, b)
        # And the operands survive intact, so the failure had no side effect on the graph.
        assert a.output_shape == (2, 2)
        assert b.output_shape == (3, 3)

    def test_manually_wired_loop_raises_cyclic_dependency_at_serialize(self) -> None:
        first = InitNode((3,), seed=1) * 2.0
        second = first * 3.0
        first.rewire(0, second)
        with pytest.raises(CyclicDependencyError) as excinfo:
            Graph([second], dag_id="manual-loop").serialize()
        message = str(excinfo.value)
        assert first.display_id in message
        assert second.display_id in message

    def test_the_cycle_is_invisible_until_the_graph_is_closed(self) -> None:
        """Acyclicity is the one property that is not local to a single expression."""
        first = InitNode((3,), seed=1) * 2.0
        second = first * 3.0
        first.rewire(0, second)  # the offending edge -- no exception here
        assert second.output_shape == (3,)


class TestDotProductBoundaries:
    """``dot_product`` must split cleanly between shape faults and rank faults."""

    def test_matrix_inner_dimensions_disagree(self) -> None:
        with pytest.raises(ShapeMismatchError, match="3 != 5"):
            InitNode((4, 3), seed=1) @ InitNode((5, 2), seed=2)

    def test_vector_vector_length_mismatch_is_a_shape_error_not_a_rank_error(self) -> None:
        """(3,) @ (4,) is two well-ranked operands that simply do not line up."""
        with pytest.raises(ShapeMismatchError) as excinfo:
            InitNode((3,), seed=1) @ InitNode((4,), seed=2)
        assert not isinstance(excinfo.value, DimensionalityError)

    def test_matching_vectors_contract_to_rank_zero(self) -> None:
        node = InitNode((6,), seed=1) @ InitNode((6,), seed=2)
        assert node.output_shape == ()
        assert node.est_flops() == 12.0

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ((3, 4), (4, 5), (3, 5)),
            ((3,), (3, 5), (5,)),
            ((3, 4), (4,), (3,)),
            ((4,), (4,), ()),
        ],
    )
    def test_every_supported_rank_combination(
        self, left: tuple[int, ...], right: tuple[int, ...], expected: tuple[int, ...]
    ) -> None:
        node = InitNode(left, seed=1) @ InitNode(right, seed=2)
        assert node.output_shape == expected

    def test_rank_zero_left_operand_is_rejected(self) -> None:
        with pytest.raises(DimensionalityError, match="rank-1 or rank-2"):
            _scalar_node() @ InitNode((3,), seed=3)

    def test_rank_zero_right_operand_is_rejected(self) -> None:
        with pytest.raises(DimensionalityError, match="rank-1 or rank-2"):
            InitNode((3,), seed=3) @ _scalar_node()

    def test_rank_three_operand_is_rejected(self) -> None:
        with pytest.raises(DimensionalityError, match="rank 3"):
            InitNode((2, 2, 2), seed=1) @ InitNode((2, 2), seed=2)

    def test_rank_check_precedes_the_extent_check(self) -> None:
        """A rank-3 operand whose extents also disagree still reports the rank."""
        with pytest.raises(DimensionalityError):
            InitNode((2, 2, 9), seed=1) @ InitNode((7, 7), seed=2)


class TestCrossProductBoundaries:
    """``cross_product`` checks rank, then equality, then length -- in that order."""

    def test_two_dimensional_operands_are_a_rank_error(self) -> None:
        with pytest.raises(DimensionalityError, match="rank-1 vectors"):
            InitNode((3, 3), seed=1).cross(InitNode((3,), seed=2))

    def test_both_operands_two_dimensional(self) -> None:
        with pytest.raises(DimensionalityError, match="rank-1 vectors"):
            InitNode((3, 3), seed=1).cross(InitNode((3, 3), seed=2))

    def test_length_two_vectors_are_a_dimensionality_error(self) -> None:
        """Rank-1 and equal, so the failure is that 3-space is the only defined case."""
        with pytest.raises(DimensionalityError, match="length-3"):
            InitNode((2,), seed=1).cross(InitNode((2,), seed=2))

    def test_length_four_vectors_are_a_dimensionality_error(self) -> None:
        with pytest.raises(DimensionalityError, match="length-3"):
            InitNode((4,), seed=1).cross(InitNode((4,), seed=2))

    def test_unequal_rank_one_lengths_are_a_shape_error(self) -> None:
        with pytest.raises(ShapeMismatchError) as excinfo:
            InitNode((3,), seed=1).cross(InitNode((4,), seed=2))
        assert not isinstance(excinfo.value, DimensionalityError)

    def test_rank_zero_operand_is_a_rank_error(self) -> None:
        with pytest.raises(DimensionalityError, match="rank-1 vectors"):
            _scalar_node().cross(InitNode((3,), seed=3))

    def test_free_function_enforces_the_same_rules(self) -> None:
        with pytest.raises(DimensionalityError):
            cross(InitNode((2,), seed=1), InitNode((2,), seed=2))


class TestRankZeroComposition:
    """Rank 0 is a legal derived value, but only some ops accept it."""

    def test_scale_preserves_rank_zero(self) -> None:
        assert (_scalar_node() * 3.0).output_shape == ()

    def test_add_accepts_two_rank_zero_operands(self) -> None:
        assert (_scalar_node() + _scalar_node()).output_shape == ()

    def test_add_rejects_rank_zero_against_a_vector(self) -> None:
        with pytest.raises(ShapeMismatchError):
            _scalar_node() + InitNode((3,), seed=9)

    def test_rank_zero_composes_through_scale_and_add(self) -> None:
        scalar = _scalar_node()
        assert (scalar * 2.0 + scalar).output_shape == ()


class TestInitNodeAttacks:
    """Every way to under-specify or corrupt a source node."""

    def test_no_arguments_at_all(self) -> None:
        with pytest.raises(UninitializedNodeError, match="shape is required"):
            InitNode()

    def test_shape_without_seed(self) -> None:
        with pytest.raises(UninitializedNodeError, match="seed is required"):
            InitNode((4, 4))

    def test_seed_without_shape(self) -> None:
        with pytest.raises(UninitializedNodeError, match="shape is required"):
            InitNode(seed=7)

    def test_rank_zero_shape_is_accepted_as_of_v120(self) -> None:
        """v1.2.0 lifted the init rank floor once multiply and mod could consume rank 0."""
        assert InitNode((), seed=1).output_shape == ()

    def test_rank_zero_still_requires_a_seed(self) -> None:
        with pytest.raises(UninitializedNodeError, match="seed is required"):
            InitNode(())

    def test_rank_above_the_maximum(self) -> None:
        with pytest.raises(UninitializedNodeError, match="rank 0 to 8"):
            InitNode((1,) * (MAX_RANK + 1), seed=1)

    def test_maximum_rank_is_accepted(self) -> None:
        assert InitNode((1,) * MAX_RANK, seed=1).output_shape == (1,) * MAX_RANK

    @pytest.mark.parametrize("shape", [(0,), (4, 0), (0, 0), (2, 3, 0)])
    def test_zero_extents(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(UninitializedNodeError, match="positive integers"):
            InitNode(shape, seed=1)

    @pytest.mark.parametrize("shape", [(-1,), (4, -2), (-3, -3)])
    def test_negative_extents(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(UninitializedNodeError, match="positive integers"):
            InitNode(shape, seed=1)

    def test_float_extent(self) -> None:
        with pytest.raises(UninitializedNodeError, match="positive integers"):
            InitNode((4.0, 4), seed=1)  # type: ignore[arg-type]

    def test_boolean_extent_is_rejected_despite_bool_subclassing_int(self) -> None:
        with pytest.raises(UninitializedNodeError, match="positive integers"):
            InitNode((True, 2), seed=1)

    def test_string_extent(self) -> None:
        with pytest.raises(UninitializedNodeError, match="positive integers"):
            InitNode(("4", 4), seed=1)  # type: ignore[arg-type]

    def test_seed_true_is_rejected_despite_bool_subclassing_int(self) -> None:
        """``seed=True`` reaches the builder from untyped call sites such as parsed config."""
        with pytest.raises(UninitializedNodeError, match=r"\[0, 2\*\*64\)"):
            InitNode((2, 2), seed=True)

    def test_seed_false_is_rejected_too(self) -> None:
        with pytest.raises(UninitializedNodeError, match=r"\[0, 2\*\*64\)"):
            InitNode((2, 2), seed=False)

    def test_float_seed(self) -> None:
        with pytest.raises(UninitializedNodeError, match=r"\[0, 2\*\*64\)"):
            InitNode((2, 2), seed=1.5)  # type: ignore[arg-type]

    def test_whole_float_seed_is_still_not_an_integer(self) -> None:
        with pytest.raises(UninitializedNodeError, match=r"\[0, 2\*\*64\)"):
            InitNode((2, 2), seed=3.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("seed", [-1, -(10**30), UINT64_MAX + 1])
    def test_seed_outside_the_unsigned_64_bit_range(self, seed: int) -> None:
        with pytest.raises(UninitializedNodeError, match=r"\[0, 2\*\*64\)"):
            InitNode((2, 2), seed=seed)

    @pytest.mark.parametrize("seed", [0, 1, UINT64_MAX])
    def test_seed_boundaries_are_accepted(self, seed: int) -> None:
        assert InitNode((2, 2), seed=seed).seed == seed

    def test_shape_is_checked_before_seed(self) -> None:
        """Validation order is fixed so messages are deterministic."""
        with pytest.raises(UninitializedNodeError, match="shape is required"):
            InitNode(None, seed=-1)

    def test_bad_dtype_is_api_misuse_not_a_build_error(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            InitNode((2, 2), seed=1, dtype="float16")  # type: ignore[arg-type]
        assert not isinstance(excinfo.value, DagBuildError)

    def test_bad_distribution_is_api_misuse_not_a_build_error(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            InitNode((2, 2), seed=1, distribution="poisson")  # type: ignore[arg-type]
        assert not isinstance(excinfo.value, DagBuildError)

    @pytest.mark.parametrize("distribution", ["uniform", "normal", "zeros", "ones"])
    def test_zeros_and_ones_still_require_a_seed(self, distribution: str) -> None:
        with pytest.raises(UninitializedNodeError, match="seed is required"):
            InitNode((2, 2), distribution=distribution)  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["2bad", "-bad", "has space", "a" * 65, "", "bad!"])
    def test_illegal_names_are_value_errors(self, name: str) -> None:
        with pytest.raises(ValueError, match="name must match"):
            InitNode((2,), seed=1, name=name)

    def test_rewire_on_a_source_node_is_always_out_of_range(self) -> None:
        with pytest.raises(IndexError):
            InitNode((2,), seed=1).rewire(0, InitNode((2,), seed=2))


class TestExceptionFamilyPurity:
    """The four build errors are the contract with the engine; keep them semantically clean."""

    @pytest.mark.parametrize(
        "exc",
        [ShapeMismatchError, DimensionalityError, CyclicDependencyError, UninitializedNodeError],
    )
    def test_every_build_error_derives_from_the_common_base(self, exc: type[DagBuildError]) -> None:
        assert issubclass(exc, DagBuildError)

    @pytest.mark.parametrize(
        "exc",
        [ShapeMismatchError, DimensionalityError, CyclicDependencyError, UninitializedNodeError],
    )
    def test_build_errors_are_not_value_or_type_errors(self, exc: type[DagBuildError]) -> None:
        """Catching ValueError must never swallow a mathematical fault."""
        assert not issubclass(exc, ValueError | TypeError)

    def test_build_errors_survive_a_pickle_round_trip(self) -> None:
        """They cross the hpcctl process boundary, so they must stay trivially picklable."""
        import pickle

        restored = pickle.loads(pickle.dumps(ShapeMismatchError("3x4 vs 5x6")))
        assert isinstance(restored, ShapeMismatchError)
        assert str(restored) == "3x4 vs 5x6"


class TestGraphAttacks:
    """Whole-graph structure: cycles, reuse, reachability, and depth."""

    def test_self_referencing_node(self) -> None:
        node = InitNode((2,), seed=1) * 2.0
        node.rewire(0, node)
        with pytest.raises(CyclicDependencyError) as excinfo:
            Graph([node], dag_id="self-loop").serialize()
        assert str(excinfo.value).count(node.display_id) == 2

    def test_two_node_cycle(self) -> None:
        first = InitNode((2,), seed=1) * 2.0
        second = first * 3.0
        first.rewire(0, second)
        with pytest.raises(CyclicDependencyError):
            Graph([second], dag_id="two-cycle").serialize()

    def test_three_node_cycle_names_a_closed_path(self) -> None:
        head = InitNode((2,), seed=1) * 2.0
        mid = head * 3.0
        tail = mid * 4.0
        head.rewire(0, tail)
        with pytest.raises(CyclicDependencyError) as excinfo:
            Graph([tail], dag_id="three-cycle").serialize()
        path = str(excinfo.value).removeprefix("Cyclic dependency detected: ").split(" -> ")
        assert path[0] == path[-1]

    def test_long_cycle_does_not_exhaust_the_stack(self) -> None:
        """A recursive reachability walk or cycle search would raise RecursionError here."""
        head = InitNode((2,), seed=1) * 1.0
        node: Node = head
        for _ in range(LONG_CYCLE):
            node = node * 1.0
        head.rewire(0, node)
        with pytest.raises(CyclicDependencyError):
            Graph([node], dag_id="long-cycle").serialize()

    def test_cycle_detection_precedes_id_assignment(self) -> None:
        """A cyclic graph reports the cycle, not some downstream confusion."""
        node = InitNode((2,), seed=1) * 2.0
        node.rewire(0, node)
        with pytest.raises(CyclicDependencyError):
            Graph([node], dag_id="cycle-first").validate()

    def test_diamond_reuse_is_legal_and_emits_the_shared_node_once(
        self, assert_conforms: Conforms
    ) -> None:
        base = InitNode((2,), seed=1) + InitNode((2,), seed=2)
        document = Graph([base + base], dag_id="diamond").serialize(include_timestamp=False)
        assert_conforms(document)
        add_nodes = [node for node in document["nodes"] if node["op"] == "add"]
        shared = add_nodes[0]["id"]
        assert add_nodes[1]["inputs"] == [shared, shared]
        assert len(document["nodes"]) == 4

    def test_a_node_may_belong_to_two_graphs(self, assert_conforms: Conforms) -> None:
        shared = InitNode((2,), seed=1) + InitNode((2,), seed=2)
        first = Graph([shared], dag_id="graph-one").serialize(include_timestamp=False)
        second = Graph([shared * 2.0], dag_id="graph-two").serialize(include_timestamp=False)
        assert_conforms(first)
        assert_conforms(second)
        assert len(first["nodes"]) == 3
        assert len(second["nodes"]) == 4

    def test_unreachable_nodes_are_silently_dropped(self, assert_conforms: Conforms) -> None:
        live = InitNode((2,), seed=1)
        dead = InitNode((9, 9), seed=2) * 3.0
        document = Graph([live], dag_id="dead-code").serialize(include_timestamp=False)
        assert_conforms(document)
        assert [node["id"] for node in document["nodes"]] == ["init_0"]
        assert dead.output_shape == (9, 9)

    def test_a_cycle_among_unreachable_nodes_is_ignored(self) -> None:
        """Dead code is dropped before cycle detection, so it cannot fail a live graph."""
        orphan = InitNode((2,), seed=1) * 2.0
        orphan.rewire(0, orphan)
        document = Graph([InitNode((2,), seed=2)], dag_id="live").serialize(include_timestamp=False)
        assert len(document["nodes"]) == 1

    def test_empty_output_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one node"):
            Graph([], dag_id="empty")

    def test_duplicate_output_node_is_rejected(self) -> None:
        node = InitNode((2,), seed=1)
        with pytest.raises(ValueError, match="duplicate output node"):
            Graph([node, node], dag_id="dup-output")

    def test_non_node_output_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="must contain Node instances"):
            Graph(["not-a-node"], dag_id="bad")  # type: ignore[list-item]

    @pytest.mark.parametrize("dag_id", ["-leading", "_leading", "has space", "a" * 129, ""])
    def test_illegal_dag_ids(self, dag_id: str) -> None:
        with pytest.raises(ValueError, match="dag_id must match"):
            Graph([InitNode((2,), seed=1)], dag_id=dag_id)

    def test_user_name_colliding_with_a_generated_id_is_caught(self) -> None:
        """A hand-picked name that lands on a renumbered slot must not silently merge nodes."""
        left = InitNode((2,), seed=1, name="init_1")
        right = InitNode((2,), seed=2)
        with pytest.raises(ValueError, match="duplicate node ID"):
            Graph([left + right], dag_id="collision").serialize()

    def test_deep_chain_serializes_without_recursion_error(self, assert_conforms: Conforms) -> None:
        document = Graph([_chain(DEEP)], dag_id="deep-chain").serialize(include_timestamp=False)
        assert_conforms(document)
        assert len(document["nodes"]) == DEEP + 1
        _assert_topological(document)

    def test_deep_chain_stays_within_the_interpreter_recursion_limit(self) -> None:
        """Pinned explicitly: the walk must be iterative, not merely deep enough to survive."""
        assert DEEP > sys.getrecursionlimit()

    def test_deep_diamond_of_shared_operands(self, assert_conforms: Conforms) -> None:
        """Every level consumes the level below twice, so in-degree bookkeeping must balance."""
        node: Node = InitNode((2,), seed=1)
        for _ in range(200):
            node = node + node
        document = Graph([node], dag_id="deep-diamond").serialize(include_timestamp=False)
        assert_conforms(document)
        assert len(document["nodes"]) == 201
        _assert_topological(document)


class TestOperatorMisuse:
    """The operator table is deliberately narrow; every omission must fail loudly."""

    def test_node_times_node_is_elementwise_never_a_contraction(self) -> None:
        """Flipped in v1.2.0. The original objection survives: `*` demands equal shapes."""
        left = InitNode((2, 2), seed=1)
        right = InitNode((2, 2), seed=2)
        product = left * right
        assert product.op == "multiply"
        assert product.output_shape == (2, 2)
        with pytest.raises(ShapeMismatchError):
            InitNode((2, 3), seed=3) * InitNode((3, 2), seed=4)

    def test_subtraction_expands_to_add_over_negative_scale(self) -> None:
        left = InitNode((2, 2), seed=1)
        right = InitNode((2, 2), seed=2)
        difference = left - right
        assert difference.op == "add"
        negated = difference.inputs[1]
        assert negated.op == "scale"
        assert isinstance(negated, ScaleNode)
        assert negated.factor == -1.0

    def test_addition_with_a_scalar_is_a_type_error(self) -> None:
        with pytest.raises(TypeError):
            InitNode((2,), seed=1) + 5  # type: ignore[operator]

    def test_matmul_with_a_scalar_is_a_type_error(self) -> None:
        with pytest.raises(TypeError):
            InitNode((2,), seed=1) @ 5  # type: ignore[operator]

    def test_scalar_matmul_node_is_a_type_error(self) -> None:
        with pytest.raises(TypeError):
            5 @ InitNode((2,), seed=1)  # type: ignore[operator]

    def test_multiplying_by_a_bool_is_rejected(self) -> None:
        """``a * True`` would otherwise build a silent scale-by-1 node.

        No ``type: ignore`` here on purpose: ``bool`` is a subtype of ``int``, so mypy sees
        nothing wrong with this call. The runtime guard is the only thing standing in the way.
        """
        with pytest.raises(TypeError):
            InitNode((2,), seed=1) * True

    def test_negation_lowers_to_exactly_one_scale_node(self) -> None:
        node = -InitNode((2, 2), seed=1)
        assert isinstance(node, ScaleNode)
        assert node.factor == -1.0
        assert len(node.inputs) == 1

    def test_division_lowers_to_exactly_one_scale_node(self) -> None:
        node = InitNode((2, 2), seed=1) / 4
        assert isinstance(node, ScaleNode)
        assert node.factor == 0.25
        assert len(node.inputs) == 1

    def test_division_by_zero(self) -> None:
        with pytest.raises(ZeroDivisionError):
            InitNode((2,), seed=1) / 0

    def test_division_by_float_zero(self) -> None:
        with pytest.raises(ZeroDivisionError):
            InitNode((2,), seed=1) / 0.0

    @pytest.mark.parametrize("factor", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_scale_factors_are_rejected_at_construction(self, factor: float) -> None:
        """``to_json(allow_nan=False)`` is defence in depth; the real guard fires far earlier."""
        with pytest.raises(ValueError, match="must be finite"):
            InitNode((2,), seed=1) * factor

    def test_division_by_nan_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            InitNode((2,), seed=1) / float("nan")

    def test_non_finite_factor_is_api_misuse_not_a_build_error(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            ScaleNode(InitNode((2,), seed=1), float("inf"))
        assert not isinstance(excinfo.value, DagBuildError)

    def test_string_factor_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="int or float"):
            ScaleNode(InitNode((2,), seed=1), "2")  # type: ignore[arg-type]

    def test_left_and_right_multiplication_agree(self) -> None:
        node = InitNode((2, 2), seed=1)
        assert (2 * node).factor == (node * 2).factor  # type: ignore[attr-defined]

    def test_division_by_infinity_yields_a_finite_zero_factor(self) -> None:
        node = InitNode((2,), seed=1) / float("inf")
        assert node.factor == 0.0  # type: ignore[attr-defined]
        assert math.isfinite(node.factor)  # type: ignore[attr-defined]


class TestDtypePromotion:
    """Mixed dtypes promote silently; §3 forbids a fifth exception for them."""

    def test_float32_plus_float64_promotes(self) -> None:
        left = InitNode((2,), seed=1, dtype="float32")
        right = InitNode((2,), seed=2, dtype="float64")
        assert (left + right).dtype == "float64"

    def test_promotion_is_symmetric(self) -> None:
        left = InitNode((2,), seed=1, dtype="float64")
        right = InitNode((2,), seed=2, dtype="float32")
        assert (left + right).dtype == "float64"

    def test_float32_pair_stays_narrow(self) -> None:
        left = InitNode((2,), seed=1, dtype="float32")
        right = InitNode((2,), seed=2, dtype="float32")
        assert (left + right).dtype == "float32"

    def test_dot_product_promotes(self) -> None:
        left = InitNode((2, 3), seed=1, dtype="float32")
        right = InitNode((3, 4), seed=2, dtype="float64")
        assert (left @ right).dtype == "float64"

    def test_cross_product_promotes(self) -> None:
        left = InitNode((3,), seed=1, dtype="float32")
        right = InitNode((3,), seed=2, dtype="float64")
        assert left.cross(right).dtype == "float64"

    def test_scale_never_widens_its_operand(self) -> None:
        """A float32 tensor scaled by a Python float stays float32."""
        assert (InitNode((2,), seed=1, dtype="float32") * 0.5).dtype == "float32"

    def test_mixed_dtype_graph_still_conforms(self, assert_conforms: Conforms) -> None:
        left = InitNode((2,), seed=1, dtype="float32")
        right = InitNode((2,), seed=2, dtype="float64")
        assert_conforms(Graph([left + right], dag_id="mixed").serialize(include_timestamp=False))


class TestSerializationConformance:
    """Every document that escapes the builder must satisfy the real contract."""

    def test_schema_version_is_pinned(self, assert_conforms: Conforms) -> None:
        document = Graph([InitNode((2,), seed=1)], dag_id="v").serialize(include_timestamp=False)
        assert_conforms(document)
        assert document["metadata"]["schema_version"] == "1.2.0"
        assert SCHEMA_VERSION == "1.2.0"

    def test_ordering_is_declared_and_actually_holds(self, assert_conforms: Conforms) -> None:
        a = InitNode((3, 4), seed=1)
        b = InitNode((4, 5), seed=2)
        document = Graph([(a @ b) * 0.5 + (a @ b)], dag_id="ordered").serialize(
            include_timestamp=False
        )
        assert_conforms(document)
        assert document["metadata"]["ordering"] == "topological"
        _assert_topological(document)

    def test_rank_zero_result_serializes_as_an_empty_array(self, assert_conforms: Conforms) -> None:
        document = Graph([_scalar_node()], dag_id="rank-zero").serialize(include_timestamp=False)
        assert_conforms(document)
        dot = next(node for node in document["nodes"] if node["op"] == "dot_product")
        assert dot["output_shape"] == []

    def test_no_key_anywhere_carries_a_null(self, assert_conforms: Conforms) -> None:
        """``additionalProperties: false`` means an absent field must be omitted, not nulled."""
        document = Graph([_chain(3)], dag_id="no-nulls").serialize(include_timestamp=False)
        assert_conforms(document)
        assert None not in _walk(document)

    def test_init_nodes_omit_the_inputs_key_entirely(self, assert_conforms: Conforms) -> None:
        """The schema sets ``inputs: false`` on init, so even an empty array is rejected."""
        document = Graph([InitNode((2,), seed=1)], dag_id="src").serialize(include_timestamp=False)
        assert_conforms(document)
        assert "inputs" not in document["nodes"][0]

    def test_non_init_nodes_omit_the_init_only_fields(self, assert_conforms: Conforms) -> None:
        document = Graph([_chain(2)], dag_id="derived").serialize(include_timestamp=False)
        assert_conforms(document)
        for node in document["nodes"]:
            if node["op"] == "init":
                continue
            assert not {"seed", "shape", "distribution"} & node.keys()

    def test_only_scale_carries_a_factor(self, assert_conforms: Conforms) -> None:
        a = InitNode((3,), seed=1)
        b = InitNode((3,), seed=2)
        document = Graph([a.cross(b) + (a * 2.0)], dag_id="factors").serialize(
            include_timestamp=False
        )
        assert_conforms(document)
        for node in document["nodes"]:
            assert ("factor" in node) == (node["op"] == "scale")

    def test_hints_are_omitted_when_disabled(self, assert_conforms: Conforms) -> None:
        document = Graph([_chain(2)], dag_id="nohints").serialize(
            include_hints=False, include_timestamp=False
        )
        assert_conforms(document)
        assert all("hints" not in node for node in document["nodes"])

    def test_timestamp_is_omitted_when_disabled(self, assert_conforms: Conforms) -> None:
        document = Graph([_chain(1)], dag_id="nots").serialize(include_timestamp=False)
        assert_conforms(document)
        assert "created_at" not in document["metadata"]

    def test_timestamp_conforms_to_the_date_time_format(self, assert_conforms: Conforms) -> None:
        document = Graph([_chain(1)], dag_id="ts").serialize(include_timestamp=True)
        assert_conforms(document)
        assert "created_at" in document["metadata"]

    def test_named_nodes_use_the_name_as_the_id(self, assert_conforms: Conforms) -> None:
        document = Graph([InitNode((2,), seed=1, name="lhs")], dag_id="named").serialize(
            include_timestamp=False
        )
        assert_conforms(document)
        assert document["nodes"][0]["id"] == "lhs"
        assert document["outputs"] == ["lhs"]

    def test_provisional_ids_also_conform(self, assert_conforms: Conforms) -> None:
        document = Graph([_chain(3)], dag_id="raw").serialize(
            renumber=False, include_timestamp=False
        )
        assert_conforms(document)
        _assert_topological(document)

    def test_serialization_is_byte_stable_across_runs(self) -> None:
        graph = Graph([_chain(5)], dag_id="stable")
        first = json.dumps(graph.serialize(include_timestamp=False), sort_keys=True)
        second = json.dumps(graph.serialize(include_timestamp=False), sort_keys=True)
        assert first == second

    def test_outputs_are_unique_and_all_present_in_nodes(self, assert_conforms: Conforms) -> None:
        a = InitNode((2,), seed=1)
        b = InitNode((2,), seed=2)
        document = Graph([a + b, a * 2.0], dag_id="multi").serialize(include_timestamp=False)
        assert_conforms(document)
        ids = {node["id"] for node in document["nodes"]}
        assert len(set(document["outputs"])) == len(document["outputs"])
        assert set(document["outputs"]) <= ids

    def test_to_json_round_trips_through_disk(
        self, tmp_path: Path, assert_conforms: Conforms
    ) -> None:
        target = tmp_path / "dag.json"
        Graph([_chain(4)], dag_id="ondisk").to_json(target, include_timestamp=False)
        assert_conforms(json.loads(target.read_text(encoding="utf-8")))

    def test_every_op_appears_in_one_conforming_document(self, assert_conforms: Conforms) -> None:
        u = InitNode((3,), seed=1)
        v = InitNode((3,), seed=2)
        m = InitNode((3, 3), seed=3)
        document = Graph([u.cross(v) + (m @ u) * 2.0, u @ v], dag_id="all-ops").serialize(
            include_timestamp=False
        )
        assert_conforms(document)
        assert {node["op"] for node in document["nodes"]} == {
            "init",
            "add",
            "scale",
            "dot_product",
            "cross_product",
        }


class TestRewireStalenessRegression:
    """Regression: a rewired operand used to leave its consumers declaring a stale shape.

    ``rewire`` re-infers only the node it edits, and nodes hold no back-references to their
    consumers. Before the fix, ``c.rewire(...)`` could change ``c`` from ``(3, 5)`` to
    ``(3, 7)`` while ``-c`` went on declaring ``output_shape`` ``[3, 5]``. That document
    validated against the schema and was mathematically impossible -- precisely the failure the
    Lazy Evaluation contract promises the engine will never see. ``Graph`` now refreshes the
    topological order before emitting anything.
    """

    def test_downstream_shape_is_refreshed_before_emission(self, assert_conforms: Conforms) -> None:
        left = InitNode((3, 4), seed=1)
        contraction = left @ InitNode((4, 5), seed=2)
        consumer = -contraction
        contraction.rewire(1, InitNode((4, 7), seed=3))
        document = Graph([consumer], dag_id="refresh").serialize(include_timestamp=False)
        assert_conforms(document)
        shapes = {node["op"]: node["output_shape"] for node in document["nodes"]}
        assert shapes["dot_product"] == [3, 7]
        assert shapes["scale"] == [3, 7]

    def test_downstream_dtype_is_refreshed_before_emission(self, assert_conforms: Conforms) -> None:
        total = InitNode((4,), seed=1, dtype="float32") + InitNode((4,), seed=2, dtype="float32")
        consumer = -total
        total.rewire(1, InitNode((4,), seed=3, dtype="float64"))
        document = Graph([consumer], dag_id="refresh-dtype").serialize(include_timestamp=False)
        assert_conforms(document)
        assert all(node["dtype"] == "float64" for node in document["nodes"] if node["op"] != "init")

    def test_a_consumer_left_misaligned_raises_shape_mismatch(self) -> None:
        contraction = InitNode((3, 4), seed=1) @ InitNode((4, 5), seed=2)
        total = contraction + InitNode((3, 5), seed=3)
        contraction.rewire(1, InitNode((4, 7), seed=4))
        with pytest.raises(ShapeMismatchError):
            Graph([total], dag_id="misaligned").serialize()

    def test_a_consumer_left_at_the_wrong_rank_raises_dimensionality_error(self) -> None:
        contraction = InitNode((3, 4), seed=1) @ InitNode((4,), seed=2)
        product = contraction.cross(InitNode((3,), seed=3))
        contraction.rewire(1, InitNode((4, 5), seed=4))
        with pytest.raises(DimensionalityError):
            Graph([product], dag_id="rank-drift").serialize()

    def test_validate_catches_the_drift_without_producing_output(self) -> None:
        contraction = InitNode((3, 4), seed=1) @ InitNode((4, 5), seed=2)
        total = contraction + InitNode((3, 5), seed=3)
        contraction.rewire(1, InitNode((4, 7), seed=4))
        with pytest.raises(ShapeMismatchError):
            Graph([total], dag_id="validate-drift").validate()

    def test_drift_propagates_through_a_multi_level_chain(self, assert_conforms: Conforms) -> None:
        contraction = InitNode((3, 4), seed=1) @ InitNode((4, 5), seed=2)
        node: Node = contraction
        for _ in range(5):
            node = node * 2.0
        contraction.rewire(1, InitNode((4, 9), seed=3))
        document = Graph([node], dag_id="deep-refresh").serialize(include_timestamp=False)
        assert_conforms(document)
        assert all(
            entry["output_shape"] == [3, 9] for entry in document["nodes"] if entry["op"] == "scale"
        )

    def test_an_untouched_graph_is_unchanged_by_the_refresh(self) -> None:
        graph = Graph([_chain(4)], dag_id="idempotent")
        before = graph.serialize(include_timestamp=False)
        after = graph.serialize(include_timestamp=False)
        assert before == after

    def test_refresh_runs_after_cycle_detection(self) -> None:
        """A cyclic graph must still report the cycle rather than looping in inference."""
        node = InitNode((2,), seed=1) * 2.0
        node.rewire(0, node)
        with pytest.raises(CyclicDependencyError):
            Graph([node], dag_id="cycle-before-refresh").serialize()


class TestFlopEstimateOverflowRegression:
    """Regression: an element count above the double range used to escape as OverflowError.

    Extents are unbounded above in both the schema and ``InitNode``, so ``float(math.prod(...))``
    could raise ``OverflowError`` out of ``Graph.serialize`` -- an exception in neither the
    build-error family nor the engine's runtime-physics list. Hints are non-authoritative by
    contract, so the estimate now saturates at the largest finite float instead.
    """

    def test_astronomical_extents_do_not_raise(self) -> None:
        assert InitNode((10**200, 10**200), seed=1).est_flops() == sys.float_info.max

    def test_astronomical_contraction_does_not_raise(self) -> None:
        left = InitNode((10**200, 10**200), seed=1)
        right = InitNode((10**200, 10**200), seed=2)
        assert math.isfinite((left @ right).est_flops())

    def test_saturated_hints_still_serialize_and_conform(self, assert_conforms: Conforms) -> None:
        node = InitNode((10**200, 10**200), seed=1) * 2.0
        document = Graph([node], dag_id="huge").serialize(include_timestamp=False)
        assert_conforms(document)
        assert all(math.isfinite(entry["hints"]["est_flops"]) for entry in document["nodes"])

    def test_saturated_hints_survive_allow_nan_false(self, tmp_path: Path) -> None:
        target = tmp_path / "huge.json"
        node = InitNode((10**200, 10**200), seed=1) * 2.0
        Graph([node], dag_id="huge-disk").to_json(target, include_timestamp=False)
        document = json.loads(target.read_text(encoding="utf-8"))
        assert math.isfinite(document["nodes"][0]["hints"]["est_flops"])

    def test_ordinary_estimates_are_unaffected(self) -> None:
        assert InitNode((4, 4), seed=1).est_flops() == 16.0
        assert (InitNode((4, 4), seed=1) * 2.0).est_flops() == 16.0
        assert (InitNode((2, 3), seed=1) @ InitNode((3, 4), seed=2)).est_flops() == 48.0
        assert InitNode((3,), seed=1).cross(InitNode((3,), seed=2)).est_flops() == 9.0


class TestConstructorsMatchOperators:
    """The explicit node classes and the operators must be interchangeable."""

    def test_add_node_matches_the_plus_operator(self) -> None:
        a = InitNode((2,), seed=1)
        b = InitNode((2,), seed=2)
        assert AddNode(a, b).output_shape == (a + b).output_shape

    def test_dot_product_node_matches_the_matmul_operator(self) -> None:
        a = InitNode((2, 3), seed=1)
        b = InitNode((3, 4), seed=2)
        assert DotProductNode(a, b).output_shape == (a @ b).output_shape

    def test_cross_product_node_matches_the_method_and_the_free_function(self) -> None:
        a = InitNode((3,), seed=1)
        b = InitNode((3,), seed=2)
        assert (
            CrossProductNode(a, b).output_shape
            == a.cross(b).output_shape
            == cross(a, b).output_shape
        )

    def test_operand_order_is_preserved_for_non_commutative_ops(
        self, assert_conforms: Conforms
    ) -> None:
        a = InitNode((2, 3), seed=1, name="lhs")
        b = InitNode((3, 4), seed=2, name="rhs")
        document = Graph([a @ b], dag_id="order").serialize(include_timestamp=False)
        assert_conforms(document)
        dot = next(node for node in document["nodes"] if node["op"] == "dot_product")
        assert dot["inputs"] == ["lhs", "rhs"]
