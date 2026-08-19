"""Tests for the operator overloading contract and eager validation.

The headline guarantee of this library is that ``a + b`` with mismatched shapes raises at the
line that wrote it, not at ``serialize()``. :class:`TestEagerValidation` is where that is
pinned down; the rest of the file walks the operator table row by row.
"""

from typing import Any

import pytest

from tasks import (
    AddNode,
    CrossProductNode,
    DotProductNode,
    Graph,
    InitNode,
    MultiplyNode,
    Node,
    ScaleNode,
    cross,
)
from tasks.exceptions import (
    DimensionalityError,
    ShapeMismatchError,
    UninitializedNodeError,
)


@pytest.fixture
def m44() -> Node:
    """A 4x4 float64 matrix source node."""
    return InitNode((4, 4), seed=1)


@pytest.fixture
def v3() -> Node:
    """A length-3 float64 vector source node."""
    return InitNode((3,), seed=2)


class TestAdd:
    def test_builds_an_add_node(self, m44: Node) -> None:
        other = InitNode((4, 4), seed=9)
        result = m44 + other
        assert isinstance(result, AddNode)
        assert result.op == "add"
        assert result.inputs == (m44, other)
        assert result.output_shape == (4, 4)

    def test_non_node_operand_raises_type_error(self, m44: Node) -> None:
        with pytest.raises(TypeError):
            m44 + 5  # type: ignore[operator]

    def test_reflected_add_from_non_node_raises(self, m44: Node) -> None:
        with pytest.raises(TypeError):
            5 + m44  # type: ignore[operator]


class TestScale:
    def test_multiply_on_the_right(self, m44: Node) -> None:
        result = m44 * 2.5
        assert isinstance(result, ScaleNode)
        assert result.factor == 2.5
        assert result.output_shape == (4, 4)

    def test_multiply_on_the_left(self, m44: Node) -> None:
        assert isinstance(2 * m44, ScaleNode)

    def test_left_and_right_multiply_agree(self, m44: Node) -> None:
        left, right = 2 * m44, m44 * 2
        assert left.op == right.op == "scale"
        assert isinstance(left, ScaleNode)
        assert isinstance(right, ScaleNode)
        assert left.factor == right.factor == 2.0
        assert left.inputs == right.inputs == (m44,)
        assert left.output_shape == right.output_shape

    def test_integer_factor_is_stored_as_float(self, m44: Node) -> None:
        result = m44 * 3
        assert isinstance(result, ScaleNode)
        assert isinstance(result.factor, float)

    def test_negation_is_a_single_scale_node(self, m44: Node) -> None:
        result = -m44
        assert isinstance(result, ScaleNode)
        assert result.factor == -1.0
        assert result.inputs == (m44,)

    def test_true_division(self, m44: Node) -> None:
        result = m44 / 4
        assert isinstance(result, ScaleNode)
        assert result.factor == 0.25

    def test_division_by_zero(self, m44: Node) -> None:
        with pytest.raises(ZeroDivisionError):
            m44 / 0

    def test_division_by_zero_float(self, m44: Node) -> None:
        with pytest.raises(ZeroDivisionError):
            m44 / 0.0

    def test_scale_preserves_float32(self) -> None:
        source = InitNode((2,), seed=1, dtype="float32")
        result = source * 2.0
        assert result.dtype == "float32"

    @pytest.mark.parametrize("factor", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_factor_rejected(self, m44: Node, factor: float) -> None:
        """JSON has no NaN or Infinity literal, so a non-finite constant is unserializable."""
        with pytest.raises(ValueError, match="factor must be finite"):
            ScaleNode(m44, factor)

    def test_non_numeric_factor_rejected(self, m44: Node) -> None:
        with pytest.raises(TypeError, match="factor must be an int or float"):
            ScaleNode(m44, "2.0")  # type: ignore[arg-type]

    def test_bool_factor_rejected(self, m44: Node) -> None:
        with pytest.raises(TypeError, match="factor must be an int or float"):
            ScaleNode(m44, True)

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_rejected_through_the_operators_too(self, m44: Node, value: bool) -> None:
        """`bool` subclasses `int`, so `a * True` must not become a silent scale-by-1.0."""
        with pytest.raises(TypeError):
            m44 * value
        with pytest.raises(TypeError):
            value * m44
        with pytest.raises(TypeError):
            m44 / value


class TestMultiplyBetweenNodes:
    def test_node_times_node_is_elementwise(self, m44: Node) -> None:
        """Flipped in v1.2.0: `*` is elementwise, as in NumPy. `@` keeps the contraction."""
        other = InitNode((4, 4), seed=9)
        result = m44 * other
        assert isinstance(result, MultiplyNode)
        assert result.op == "multiply"
        assert result.inputs == (m44, other)

    def test_star_is_still_never_a_contraction(self) -> None:
        """The original objection is preserved: `*` requires exact shape equality."""
        left, right = InitNode((4, 3), seed=1), InitNode((3, 2), seed=2)
        with pytest.raises(ShapeMismatchError):
            left * right


class TestSubtraction:
    def test_node_minus_node_expands_to_two_nodes(self, m44: Node) -> None:
        """New in v1.2.0: `a - b` lowers to AddNode(a, ScaleNode(b, -1.0))."""
        other = InitNode((4, 4), seed=9)
        result = m44 - other
        assert isinstance(result, AddNode)
        negated = result.inputs[1]
        assert isinstance(negated, ScaleNode)
        assert negated.factor == -1.0
        assert negated.inputs == (other,)
        assert result.inputs[0] is m44

    def test_subtraction_costs_one_more_node_than_addition(self, m44: Node) -> None:
        """So a topology comparison between `+` and `-` is not apples to apples."""
        other = InitNode((4, 4), seed=9)
        summed = Graph([m44 + other], dag_id="s").nodes()
        differenced = Graph([m44 - other], dag_id="d").nodes()
        assert len(differenced) == len(summed) + 1

    def test_documented_equivalent_builds_the_same_shape(self, m44: Node) -> None:
        other = InitNode((4, 4), seed=9)
        result = m44 + -other
        assert isinstance(result, AddNode)
        assert isinstance(result.inputs[1], ScaleNode)


class TestMatmul:
    def test_builds_a_dot_product_node(self) -> None:
        left, right = InitNode((4, 3), seed=1), InitNode((3, 2), seed=2)
        result = left @ right
        assert isinstance(result, DotProductNode)
        assert result.output_shape == (4, 2)
        assert result.inputs == (left, right)

    def test_operand_order_is_preserved(self) -> None:
        left, right = InitNode((4, 3), seed=1), InitNode((3, 2), seed=2)
        assert (left @ right).inputs == (left, right)

    def test_vector_vector_yields_rank_zero(self) -> None:
        left, right = InitNode((5,), seed=1), InitNode((5,), seed=2)
        assert (left @ right).output_shape == ()

    def test_non_node_operand_raises(self, m44: Node) -> None:
        with pytest.raises(TypeError):
            m44 @ 5  # type: ignore[operator]


class TestCross:
    def test_method_form(self, v3: Node) -> None:
        other = InitNode((3,), seed=9)
        result = v3.cross(other)
        assert isinstance(result, CrossProductNode)
        assert result.output_shape == (3,)

    def test_function_form(self, v3: Node) -> None:
        other = InitNode((3,), seed=9)
        assert isinstance(cross(v3, other), CrossProductNode)

    def test_both_forms_agree(self, v3: Node) -> None:
        other = InitNode((3,), seed=9)
        assert v3.cross(other).output_shape == cross(v3, other).output_shape

    def test_accepts_a_name(self, v3: Node) -> None:
        other = InitNode((3,), seed=9)
        assert v3.cross(other, name="torque").label == "torque"


class TestEagerValidation:
    """Logical errors fire at the offending expression, never at serialization."""

    def test_add_shape_mismatch_is_instant(self) -> None:
        left, right = InitNode((4, 4), seed=1), InitNode((3, 3), seed=2)
        with pytest.raises(ShapeMismatchError):
            left + right

    def test_dot_inner_dimension_mismatch_is_instant(self) -> None:
        left, right = InitNode((4, 3), seed=1), InitNode((5, 2), seed=2)
        with pytest.raises(ShapeMismatchError, match="3 != 5"):
            left @ right

    def test_cross_on_a_matrix_is_instant(self) -> None:
        matrix, vector = InitNode((4, 4), seed=1), InitNode((3,), seed=2)
        with pytest.raises(DimensionalityError, match="rank-1 vectors"):
            matrix.cross(vector)

    def test_cross_wrong_length_is_instant(self) -> None:
        left, right = InitNode((4,), seed=1), InitNode((4,), seed=2)
        with pytest.raises(DimensionalityError, match="length-3"):
            left.cross(right)

    def test_dot_rank_3_is_instant(self) -> None:
        left, right = InitNode((2, 2, 2), seed=1), InitNode((2, 2), seed=2)
        with pytest.raises(DimensionalityError, match="rank-1 or rank-2"):
            left @ right

    def test_error_names_the_operands(self) -> None:
        left = InitNode((4, 4), seed=1, name="lhs")
        right = InitNode((3, 3), seed=2, name="rhs")
        with pytest.raises(ShapeMismatchError) as excinfo:
            left + right
        assert "nodes 'lhs' and 'rhs'" in str(excinfo.value)

    def test_failure_leaves_no_node_behind(self) -> None:
        """A rejected expression must not contribute anything to a later graph."""
        left = InitNode((4, 4), seed=1)
        right = InitNode((3, 3), seed=2)
        with pytest.raises(ShapeMismatchError):
            left + right
        graph = Graph([left], dag_id="only-lhs")
        assert graph.nodes() == (left,)


class TestDesignWorkedSession:
    """The exact failure cases listed in the design's worked example."""

    def test_missing_seed(self) -> None:
        with pytest.raises(UninitializedNodeError, match="seed is required"):
            InitNode((4, 4))

    def test_rank_zero_init_is_now_accepted(self) -> None:
        """v1.2.0 lifted the floor; a missing seed is still the failure here."""
        with pytest.raises(UninitializedNodeError, match="seed is required"):
            InitNode(())
        assert InitNode((), seed=1).output_shape == ()

    def test_add_shape_mismatch(self) -> None:
        with pytest.raises(ShapeMismatchError):
            InitNode((4, 4), seed=1) + InitNode((3, 3), seed=2)

    def test_cross_on_rank_2(self) -> None:
        with pytest.raises(DimensionalityError, match="rank 2"):
            InitNode((4, 4), seed=1).cross(InitNode((3,), seed=2))

    def test_dot_inner_mismatch(self) -> None:
        with pytest.raises(ShapeMismatchError, match="3 != 5"):
            InitNode((4, 3), seed=1) @ InitNode((5, 2), seed=2)

    def test_vector_vector_length_mismatch(self) -> None:
        with pytest.raises(ShapeMismatchError, match="3 != 4"):
            InitNode((3,), seed=1) @ InitNode((4,), seed=2)

    def test_rank_zero_operand_cannot_feed_cross(self) -> None:
        u, v = InitNode((3,), seed=1), InitNode((3,), seed=2)
        scalar = u @ v
        with pytest.raises(DimensionalityError, match="rank-1 vectors"):
            scalar.cross(u)

    def test_rank_zero_operand_cannot_feed_dot(self) -> None:
        u, v = InitNode((3,), seed=1), InitNode((3,), seed=2)
        scalar = u @ v
        with pytest.raises(DimensionalityError, match="rank-1 or rank-2"):
            scalar @ u


class TestRankZeroComposition:
    """Rank-0 results compose through scale and add, but feed no product op."""

    def test_scale_preserves_rank_zero(self) -> None:
        u, v = InitNode((3,), seed=1), InitNode((3,), seed=2)
        assert ((u @ v) * 2.0).output_shape == ()

    def test_add_accepts_two_rank_zero_operands(self) -> None:
        u, v = InitNode((3,), seed=1), InitNode((3,), seed=2)
        scalar = u @ v
        assert (scalar * 2.0 + scalar).output_shape == ()

    def test_rank_zero_cannot_be_added_to_a_vector(self) -> None:
        u, v = InitNode((3,), seed=1), InitNode((3,), seed=2)
        with pytest.raises(ShapeMismatchError):
            (u @ v) + u


class TestDtypePromotion:
    def test_mixed_dtypes_promote_silently(self) -> None:
        left = InitNode((2, 2), seed=1, dtype="float32")
        right = InitNode((2, 2), seed=2, dtype="float64")
        assert (left + right).dtype == "float64"

    def test_promotion_is_order_independent(self) -> None:
        left = InitNode((2, 2), seed=1, dtype="float32")
        right = InitNode((2, 2), seed=2, dtype="float64")
        assert (left + right).dtype == (right + left).dtype

    def test_matching_float32_stays_float32(self) -> None:
        left = InitNode((2, 2), seed=1, dtype="float32")
        right = InitNode((2, 2), seed=2, dtype="float32")
        assert (left + right).dtype == "float32"

    def test_promotion_applies_to_dot(self) -> None:
        left = InitNode((2, 2), seed=1, dtype="float32")
        right = InitNode((2, 2), seed=2, dtype="float64")
        assert (left @ right).dtype == "float64"

    def test_promotion_applies_to_cross(self) -> None:
        left = InitNode((3,), seed=1, dtype="float32")
        right = InitNode((3,), seed=2, dtype="float64")
        assert left.cross(right).dtype == "float64"


class TestChaining:
    def test_worked_example_expression(self) -> None:
        a = InitNode((64, 32), seed=42, distribution="normal", name="lhs")
        b = InitNode((32, 16), seed=43)
        c = (a @ b) * 0.5
        d = c + c
        assert isinstance(d, AddNode)
        assert d.output_shape == (64, 16)
        assert d.inputs == (c, c)

    def test_numpy_reading_order(self) -> None:
        """`(a @ b) * 0.5` contracts then scales, exactly as it would in NumPy."""
        a, b = InitNode((2, 3), seed=1), InitNode((3, 4), seed=2)
        result = (a @ b) * 0.5
        assert isinstance(result, ScaleNode)
        assert isinstance(result.inputs[0], DotProductNode)


class TestNodeIdentitySemantics:
    def test_structurally_identical_nodes_are_distinct(self) -> None:
        left, right = InitNode((2, 2), seed=1), InitNode((2, 2), seed=1)
        assert left != right
        assert len({left, right}) == 2

    def test_a_node_equals_itself(self, m44: Node) -> None:
        assert m44 == m44
        assert len({m44, m44}) == 1

    def test_repr_names_the_node(self) -> None:
        assert "lhs" in repr(InitNode((2,), seed=1, name="lhs"))


class TestRewire:
    def test_replaces_the_operand_and_reinfers(self) -> None:
        a, b = InitNode((4, 4), seed=1), InitNode((4, 4), seed=2)
        node = a + b
        replacement = InitNode((4, 4), seed=3)
        node.rewire(1, replacement)
        assert node.inputs == (a, replacement)

    def test_shape_is_recomputed(self) -> None:
        a, b = InitNode((4, 3), seed=1), InitNode((3, 2), seed=2)
        node = a @ b
        assert node.output_shape == (4, 2)
        node.rewire(1, InitNode((3, 7), seed=3))
        assert node.output_shape == (4, 7)

    def test_dtype_is_recomputed(self) -> None:
        a = InitNode((2, 2), seed=1, dtype="float32")
        b = InitNode((2, 2), seed=2, dtype="float32")
        node = a + b
        before = node.dtype
        node.rewire(1, InitNode((2, 2), seed=3, dtype="float64"))
        after = node.dtype
        assert before == "float32"
        assert after == "float64"

    def test_incompatible_replacement_raises(self) -> None:
        a, b = InitNode((4, 4), seed=1), InitNode((4, 4), seed=2)
        node = a + b
        with pytest.raises(ShapeMismatchError):
            node.rewire(1, InitNode((5, 5), seed=3))

    def test_failed_rewire_leaves_the_node_untouched(self) -> None:
        a, b = InitNode((4, 4), seed=1), InitNode((4, 4), seed=2)
        node = a + b
        with pytest.raises(ShapeMismatchError):
            node.rewire(1, InitNode((5, 5), seed=3))
        assert node.inputs == (a, b)
        assert node.output_shape == (4, 4)

    def test_out_of_range_index(self) -> None:
        node = InitNode((4, 4), seed=1) + InitNode((4, 4), seed=2)
        with pytest.raises(IndexError):
            node.rewire(5, InitNode((4, 4), seed=3))

    def test_negative_index_counts_from_the_end(self) -> None:
        a, b = InitNode((4, 4), seed=1), InitNode((4, 4), seed=2)
        node = a + b
        replacement = InitNode((4, 4), seed=3)
        node.rewire(-1, replacement)
        assert node.inputs == (a, replacement)

    def test_non_node_replacement(self) -> None:
        node = InitNode((4, 4), seed=1) + InitNode((4, 4), seed=2)
        with pytest.raises(TypeError, match="must be a Node"):
            node.rewire(0, "nope")  # type: ignore[arg-type]

    def test_rewire_into_rank_error(self) -> None:
        a, b = InitNode((3,), seed=1), InitNode((3,), seed=2)
        node = a.cross(b)
        with pytest.raises(DimensionalityError):
            node.rewire(0, InitNode((3, 3), seed=3))


class TestNotImplementedProtocol:
    @pytest.mark.parametrize("other", ["text", None, [1, 2]])
    def test_add_returns_not_implemented_for_foreign_types(self, m44: Node, other: Any) -> None:
        assert m44.__add__(other) is NotImplemented

    @pytest.mark.parametrize("other", ["text", None])
    def test_mul_returns_not_implemented_for_foreign_types(self, m44: Node, other: Any) -> None:
        assert m44.__mul__(other) is NotImplemented

    def test_mul_no_longer_returns_not_implemented_for_nodes(self, m44: Node) -> None:
        """Flipped in v1.2.0: node * node builds a MultiplyNode rather than deferring."""
        assert isinstance(m44.__mul__(InitNode((4, 4), seed=9)), MultiplyNode)

    def test_truediv_returns_not_implemented_for_nodes(self, m44: Node) -> None:
        other = InitNode((4, 4), seed=9)
        assert m44.__truediv__(other) is NotImplemented  # type: ignore[operator]

    def test_matmul_returns_not_implemented_for_foreign_types(self, m44: Node) -> None:
        assert m44.__matmul__("text") is NotImplemented  # type: ignore[operator]
