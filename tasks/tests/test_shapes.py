"""Tests for the pure shape-inference rules.

This is the densest file in the suite by design: the rules live here as free functions over
plain tuples, so the mathematically interesting behaviour can be tested without constructing a
single node.
"""

import pytest

from tasks.dtypes import Shape, promote
from tasks.exceptions import DimensionalityError, ShapeMismatchError
from tasks.shapes import (
    describe,
    flops_cross,
    flops_dot,
    flops_elementwise,
    flops_init,
    infer_add,
    infer_cross,
    infer_dot,
    infer_scale,
)

WHERE = "nodes 'a' and 'b'"


class TestDescribe:
    def test_single_node(self) -> None:
        assert describe("init_0") == "node 'init_0'"

    def test_two_nodes(self) -> None:
        assert describe("init_0", "init_1") == "nodes 'init_0' and 'init_1'"

    def test_three_nodes(self) -> None:
        assert describe("a", "b", "c") == "nodes 'a', 'b' and 'c'"

    def test_no_nodes(self) -> None:
        assert describe() == "no nodes"


class TestInferAdd:
    @pytest.mark.parametrize("shape", [(4,), (4, 4), (2, 3, 5), (3,) * 8, ()])
    def test_identical_shapes_pass_through(self, shape: Shape) -> None:
        assert infer_add(shape, shape, where=WHERE) == shape

    def test_extent_mismatch_raises(self) -> None:
        with pytest.raises(ShapeMismatchError) as excinfo:
            infer_add((2, 2), (3, 3), where=WHERE)
        assert str(excinfo.value) == (
            "add: operand shapes must match exactly, got (2, 2) and (3, 3) (nodes 'a' and 'b')"
        )

    def test_rank_mismatch_raises(self) -> None:
        with pytest.raises(ShapeMismatchError):
            infer_add((4,), (4, 1), where=WHERE)

    def test_no_broadcasting(self) -> None:
        """A trailing 1 does not broadcast; the engine implements no such rule."""
        with pytest.raises(ShapeMismatchError):
            infer_add((4, 3), (1, 3), where=WHERE)


class TestInferScale:
    @pytest.mark.parametrize("shape", [(), (5,), (4, 4), (2,) * 8])
    def test_preserves_shape(self, shape: Shape) -> None:
        assert infer_scale(shape, where=WHERE) == shape

    def test_rejects_rank_above_maximum(self) -> None:
        with pytest.raises(DimensionalityError, match="rank must not exceed 8"):
            infer_scale((1,) * 9, where=WHERE)


class TestInferDot:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ((4, 3), (3, 2), (4, 2)),
            ((3,), (3, 2), (2,)),
            ((4, 3), (3,), (4,)),
            ((3,), (3,), ()),
        ],
        ids=["matrix@matrix", "vector@matrix", "matrix@vector", "vector@vector"],
    )
    def test_supported_rank_combinations(self, a: Shape, b: Shape, expected: Shape) -> None:
        assert infer_dot(a, b, where=WHERE) == expected

    def test_vector_vector_yields_rank_zero(self) -> None:
        """Schema 1.1.0 represents the collapsed scalar as an empty shape array."""
        assert infer_dot((7,), (7,), where=WHERE) == ()

    def test_inner_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ShapeMismatchError) as excinfo:
            infer_dot((4, 3), (5, 2), where=WHERE)
        assert str(excinfo.value) == (
            "dot_product: inner dimensions must agree, got (4, 3) @ (5, 2), 3 != 5 "
            "(nodes 'a' and 'b')"
        )

    def test_vector_vector_length_mismatch_raises_shape_error(self) -> None:
        """Two rank-1 operands of different lengths is a shape problem, not a rank problem."""
        with pytest.raises(ShapeMismatchError):
            infer_dot((3,), (4,), where=WHERE)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ((2, 2, 2), (2, 2)),
            ((2, 2), (2, 2, 2)),
            ((), (2, 2)),
            ((2, 2), ()),
        ],
        ids=["lhs-rank-3", "rhs-rank-3", "lhs-rank-0", "rhs-rank-0"],
    )
    def test_unsupported_rank_raises(self, a: Shape, b: Shape) -> None:
        with pytest.raises(DimensionalityError, match="must be rank-1 or rank-2"):
            infer_dot(a, b, where=WHERE)

    def test_rank_error_message_names_both_ranks(self) -> None:
        with pytest.raises(DimensionalityError) as excinfo:
            infer_dot((2, 2, 2), (2, 2), where="nodes 't' and 'm'")
        assert str(excinfo.value) == (
            "dot_product: operands must be rank-1 or rank-2, got rank 3 and rank 2 "
            "(nodes 't' and 'm')"
        )

    def test_rank_check_precedes_extent_check(self) -> None:
        """A rank-3 operand reports its rank even when the extents also disagree."""
        with pytest.raises(DimensionalityError):
            infer_dot((9, 9, 9), (5, 5), where=WHERE)


class TestInferCross:
    def test_valid_three_vectors(self) -> None:
        assert infer_cross((3,), (3,), where=WHERE) == (3,)

    def test_branch_1_rank_checked_first(self) -> None:
        """A matrix operand hears 'this needs a vector', not 'this needs length 3'."""
        with pytest.raises(DimensionalityError) as excinfo:
            infer_cross((3, 3), (3,), where="nodes 'm' and 'v'")
        assert str(excinfo.value) == (
            "cross_product: operands must be rank-1 vectors, got rank 2 and rank 1 "
            "(nodes 'm' and 'v')"
        )

    def test_branch_2_length_mismatch(self) -> None:
        with pytest.raises(ShapeMismatchError) as excinfo:
            infer_cross((3,), (4,), where="nodes 'u' and 'v'")
        assert str(excinfo.value) == (
            "cross_product: operand shapes must match exactly, got (3,) and (4,) "
            "(nodes 'u' and 'v')"
        )

    def test_branch_3_wrong_length(self) -> None:
        with pytest.raises(DimensionalityError) as excinfo:
            infer_cross((4,), (4,), where="nodes 'u' and 'v'")
        assert str(excinfo.value) == (
            "cross_product: only defined for length-3 vectors, got length 4 (nodes 'u' and 'v')"
        )

    def test_rank_beats_length_when_both_wrong(self) -> None:
        """Ordering guarantee: a rank-2 operand of the wrong size is still a rank error."""
        with pytest.raises(DimensionalityError, match="must be rank-1 vectors"):
            infer_cross((4, 4), (4,), where=WHERE)

    def test_rank_zero_operand_is_a_rank_error(self) -> None:
        with pytest.raises(DimensionalityError, match="must be rank-1 vectors"):
            infer_cross((), (3,), where=WHERE)


class TestFlopEstimates:
    def test_init_counts_one_draw_per_element(self) -> None:
        assert flops_init((64, 32)) == 2048.0

    def test_init_of_rank_zero_is_one(self) -> None:
        assert flops_init(()) == 1.0

    def test_elementwise_counts_one_op_per_element(self) -> None:
        assert flops_elementwise((64, 16)) == 1024.0

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ((64, 32), (32, 16), 2 * 64 * 16 * 32.0),
            ((32,), (32, 16), 2 * 16 * 32.0),
            ((64, 32), (32,), 2 * 64 * 32.0),
            ((32,), (32,), 2 * 32.0),
        ],
        ids=["matrix@matrix", "vector@matrix", "matrix@vector", "vector@vector"],
    )
    def test_dot_is_two_flops_per_element_per_step(
        self, a: Shape, b: Shape, expected: float
    ) -> None:
        assert flops_dot(a, b) == expected

    def test_cross_is_fixed(self) -> None:
        assert flops_cross() == 9.0

    def test_estimates_are_never_negative(self) -> None:
        """The schema constrains hints.est_flops to be non-negative."""
        assert flops_init((1,)) >= 0
        assert flops_dot((1,), (1,)) >= 0


class TestPromote:
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("float32", "float32", "float32"),
            ("float32", "float64", "float64"),
            ("float64", "float32", "float64"),
            ("float64", "float64", "float64"),
        ],
    )
    def test_numpy_widening_rule(self, left: str, right: str, expected: str) -> None:
        assert promote(left, right) == expected  # type: ignore[arg-type]

    def test_promotion_is_commutative(self) -> None:
        assert promote("float32", "float64") == promote("float64", "float32")
