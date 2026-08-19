"""Tests for the operator table as revised in v1.2.0.

Three rows changed and two are new, and each one reverses or extends a documented earlier decision,
so they are pinned here rather than folded into the older operator tests: ``a * b`` flipped from
``TypeError`` to ``MultiplyNode``, ``a - b`` flipped from ``TypeError`` to a two-node expansion, and
``a ** n`` / ``a % m`` are new.
"""

from typing import Any

import pytest

from tasks import AddNode, Graph, InitNode, ModNode, MultiplyNode, Node, ScaleNode
from tasks.exceptions import ShapeMismatchError
from tasks.math import multiplies
from tasks.math import pow as tpow


@pytest.fixture
def m22() -> Node:
    """A 2x2 float64 source node."""
    return InitNode((2, 2), seed=1)


def node_count(output: Node, *, exclude: Node | None = None) -> int:
    """Count reachable nodes, optionally discounting a known input.

    Args:
        output: Node to close a graph over.
        exclude: Node not to count, when it is still reachable.

    Returns:
        The number of emitted nodes.
    """
    nodes = Graph([output], dag_id="count").nodes()
    total = len(nodes)
    if exclude is not None and exclude in nodes:
        total -= 1
    return total


class TestStarFlipped:
    def test_node_times_node_is_a_multiply(self, m22: Node) -> None:
        other = InitNode((2, 2), seed=2)
        result = m22 * other
        assert isinstance(result, MultiplyNode)
        assert result.op == "multiply"

    def test_it_is_exactly_one_node(self, m22: Node) -> None:
        other = InitNode((2, 2), seed=2)
        assert node_count(m22 * other) == 3

    def test_node_times_scalar_is_still_a_scale(self, m22: Node) -> None:
        result = m22 * 2.5
        assert isinstance(result, ScaleNode)
        assert result.factor == 2.5

    def test_int_scalar_is_still_a_scale(self, m22: Node) -> None:
        assert isinstance(m22 * 3, ScaleNode)

    def test_reflected_scalar_is_still_a_scale(self, m22: Node) -> None:
        assert isinstance(2 * m22, ScaleNode)

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_still_raises(self, m22: Node, value: bool) -> None:
        """Bool is an int subclass, so `a * True` must not become a scale by 1.0."""
        with pytest.raises(TypeError):
            m22 * value

    @pytest.mark.parametrize("value", ["text", None, [1]])
    def test_foreign_types_still_raise(self, m22: Node, value: Any) -> None:
        with pytest.raises(TypeError):
            m22 * value

    def test_star_still_demands_exact_shapes(self) -> None:
        """The original objection is preserved: `*` is never a contraction."""
        left, right = InitNode((4, 3), seed=1), InitNode((3, 2), seed=2)
        with pytest.raises(ShapeMismatchError):
            left * right

    def test_star_and_matmul_build_different_ops(self) -> None:
        left, right = InitNode((3, 3), seed=1), InitNode((3, 3), seed=2)
        assert (left * right).op == "multiply"
        assert (left @ right).op == "dot_product"

    def test_star_is_commutative_in_shape_but_records_order(self, m22: Node) -> None:
        other = InitNode((2, 2), seed=2)
        assert (m22 * other).inputs == (m22, other)
        assert (other * m22).inputs == (other, m22)


class TestSubtractionFlipped:
    def test_is_an_add_over_a_negating_scale(self, m22: Node) -> None:
        other = InitNode((2, 2), seed=2)
        result = m22 - other
        assert isinstance(result, AddNode)
        negated = result.inputs[1]
        assert isinstance(negated, ScaleNode)
        assert negated.factor == -1.0
        assert negated.inputs == (other,)

    def test_is_exactly_two_nodes(self, m22: Node) -> None:
        other = InitNode((2, 2), seed=2)
        assert node_count(m22 - other) - 2 == 2

    def test_left_operand_is_untouched(self, m22: Node) -> None:
        other = InitNode((2, 2), seed=2)
        assert (m22 - other).inputs[0] is m22

    def test_shapes_must_still_match(self, m22: Node) -> None:
        with pytest.raises(ShapeMismatchError):
            m22 - InitNode((3, 3), seed=2)

    def test_scalar_subtraction_is_a_type_error(self, m22: Node) -> None:
        with pytest.raises(TypeError):
            m22 - 5  # type: ignore[operator]

    def test_costs_one_more_node_than_addition(self, m22: Node) -> None:
        other = InitNode((2, 2), seed=2)
        assert node_count(m22 - other) == node_count(m22 + other) + 1

    def test_matches_the_manual_equivalent(self, m22: Node) -> None:
        other = InitNode((2, 2), seed=2)
        expanded = m22 - other
        manual = m22 + -other
        assert expanded.op == manual.op
        assert expanded.inputs[1].op == manual.inputs[1].op


class TestModOperator:
    def test_scalar_modulus_builds_a_mod_node(self, m22: Node) -> None:
        result = m22 % 7
        assert isinstance(result, ModNode)
        assert result.modulus == 7.0

    def test_float_modulus(self, m22: Node) -> None:
        result = m22 % 2.5
        assert isinstance(result, ModNode)
        assert result.modulus == 2.5

    def test_is_exactly_one_node(self, m22: Node) -> None:
        assert node_count(m22 % 7, exclude=m22) == 1

    def test_node_modulus_is_a_type_error(self, m22: Node) -> None:
        """The modulus is a scalar field, not an operand."""
        with pytest.raises(TypeError):
            m22 % InitNode((2, 2), seed=2)  # type: ignore[operator]

    def test_bool_modulus_is_a_type_error(self, m22: Node) -> None:
        with pytest.raises(TypeError):
            m22 % True

    def test_non_positive_modulus_is_a_value_error(self, m22: Node) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            m22 % 0

    def test_preserves_shape_and_dtype(self) -> None:
        source = InitNode((3,), seed=1, dtype="float32")
        result = source % 5
        assert result.output_shape == (3,)
        assert result.dtype == "float32"


class TestPowOperator:
    def test_matches_the_composite_node_for_node(self, m22: Node) -> None:
        via_operator = Graph([m22**3], dag_id="op").serialize(include_timestamp=False)
        other = InitNode((2, 2), seed=1)
        via_function = Graph([tpow(other, 3)], dag_id="op").serialize(include_timestamp=False)
        assert [n["op"] for n in via_operator["nodes"]] == [n["op"] for n in via_function["nodes"]]
        assert len(via_operator["nodes"]) == len(via_function["nodes"])

    @pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 16, 100, 1024])
    def test_node_count_follows_the_formula(self, m22: Node, n: int) -> None:
        assert node_count(m22**n, exclude=m22) == multiplies(n)

    def test_exponent_one_returns_the_operand(self, m22: Node) -> None:
        assert m22**1 is m22

    def test_exponent_zero_is_a_ones_init(self, m22: Node) -> None:
        result = m22**0
        assert result.op == "init"
        assert node_count(result) == 1

    def test_negative_exponent_is_a_value_error(self, m22: Node) -> None:
        with pytest.raises(ValueError, match="int >= 0"):
            m22**-1

    def test_float_exponent_is_a_type_error(self, m22: Node) -> None:
        with pytest.raises(TypeError):
            m22**2.5  # type: ignore[operator]

    def test_bool_exponent_is_a_type_error(self, m22: Node) -> None:
        with pytest.raises(TypeError):
            m22**True

    def test_all_emitted_nodes_are_multiplies(self, m22: Node) -> None:
        document = Graph([m22**10], dag_id="p").serialize(include_timestamp=False)
        ops = [n["op"] for n in document["nodes"]]
        assert ops.count("multiply") == multiplies(10)
        assert set(ops) == {"init", "multiply"}


class TestDivisionStillRejected:
    def test_node_divided_by_node_is_a_type_error(self, m22: Node) -> None:
        """No division primitive exists; division-by-zero is unvalidatable at build time."""
        with pytest.raises(TypeError):
            m22 / InitNode((2, 2), seed=2)  # type: ignore[operator]

    def test_scalar_division_still_works(self, m22: Node) -> None:
        assert isinstance(m22 / 4, ScaleNode)

    def test_division_by_zero_still_raises(self, m22: Node) -> None:
        with pytest.raises(ZeroDivisionError):
            m22 / 0


class TestFullOperatorTable:
    """One assertion per row of the design's operator table, with its node count."""

    @pytest.mark.parametrize(
        ("build", "expected_op", "expected_nodes"),
        [
            (lambda a, b: a + b, "add", 1),
            (lambda a, b: a - b, "add", 2),
            (lambda a, b: a * b, "multiply", 1),
            (lambda a, b: a * 2.0, "scale", 1),
            (lambda a, b: 2.0 * a, "scale", 1),
            (lambda a, b: a @ b, "dot_product", 1),
            (lambda a, b: -a, "scale", 1),
            (lambda a, b: a / 2.0, "scale", 1),
            (lambda a, b: a % 5, "mod", 1),
            (lambda a, b: a**3, "multiply", 2),
        ],
        ids=[
            "add",
            "subtract",
            "multiply",
            "scale-right",
            "scale-left",
            "matmul",
            "negate",
            "divide-scalar",
            "mod",
            "pow",
        ],
    )
    def test_row(
        self,
        build: Any,
        expected_op: str,
        expected_nodes: int,
    ) -> None:
        left = InitNode((2, 2), seed=1)
        right = InitNode((2, 2), seed=2)
        result = build(left, right)
        assert result.op == expected_op
        nodes = Graph([result], dag_id="row").nodes()
        emitted = len(nodes) - sum(1 for source in (left, right) if source in nodes)
        assert emitted == expected_nodes
