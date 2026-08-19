"""Tests for ``InitNode`` construction and its seven-step validation order.

The validation order is fixed and load-bearing: it is what makes the error a user sees
predictable when an init node is wrong in more than one way at once. Several tests below
deliberately supply two faults and assert which one is reported.

Tests that assert an exact message pass ``name=`` so the reported ID does not depend on the
process-wide construction counter, which other tests advance.
"""

from typing import Any

import pytest

from tasks import InitNode
from tasks.dtypes import UINT64_MAX
from tasks.exceptions import DagBuildError, UninitializedNodeError


class TestValidationOrder:
    def test_step_1_missing_shape(self) -> None:
        with pytest.raises(UninitializedNodeError) as excinfo:
            InitNode(name="src")
        assert str(excinfo.value) == ("InitNode 'src': shape is required for init nodes, got None")

    def test_step_1_precedes_everything(self) -> None:
        """A missing shape is reported even when the seed is also missing."""
        with pytest.raises(UninitializedNodeError, match="shape is required"):
            InitNode(None, seed=None)

    def test_rank_zero_is_accepted(self) -> None:
        """Accept rank 0, reversing the v1.1.0 floor.

        Multiply and mod consume rank-0 operands, and the composite expansions need a rank-0
        ones constant for terms like ``cos(u @ v)``'s ``x**0``.
        """
        node = InitNode((), seed=1, name="src")
        assert node.output_shape == ()

    def test_rank_zero_payload_agrees_with_output_shape(self) -> None:
        """InitNode is now the only thing keeping shape and output_shape consistent."""
        doc = InitNode((), seed=1).to_dict("n", [], include_hints=False)
        assert doc["shape"] == []
        assert doc["output_shape"] == []

    def test_step_3_rank_above_maximum(self) -> None:
        with pytest.raises(UninitializedNodeError, match="rank 0 to 8"):
            InitNode((1,) * 9, seed=1)

    def test_rank_above_maximum_message(self) -> None:
        with pytest.raises(UninitializedNodeError) as excinfo:
            InitNode((1,) * 9, seed=1, name="src")
        assert "shape must have rank 0 to 8" in str(excinfo.value)

    def test_rank_8_is_accepted(self) -> None:
        node = InitNode((1,) * 8, seed=1)
        assert node.output_shape == (1,) * 8

    def test_rank_check_precedes_seed_check(self) -> None:
        """Uses an over-rank shape, since rank 0 is no longer itself a failure."""
        with pytest.raises(UninitializedNodeError, match="rank 0 to 8"):
            InitNode((1,) * 9, seed=None)

    @pytest.mark.parametrize(
        "shape",
        [(4, 0), (0,), (-1,), (4, -3)],
        ids=["zero-extent", "zero-only", "negative", "negative-trailing"],
    )
    def test_step_4_non_positive_extents(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(UninitializedNodeError, match="positive integers"):
            InitNode(shape, seed=1)

    def test_step_4_message_shows_whole_shape(self) -> None:
        with pytest.raises(UninitializedNodeError) as excinfo:
            InitNode((4, 0), seed=1, name="src")
        assert str(excinfo.value) == (
            "InitNode 'src': shape extents must be positive integers, got (4, 0)"
        )

    def test_step_4_rejects_float_extent(self) -> None:
        with pytest.raises(UninitializedNodeError, match="positive integers"):
            InitNode((4.5,), seed=1)  # type: ignore[arg-type]

    def test_step_4_precedes_seed_check(self) -> None:
        with pytest.raises(UninitializedNodeError, match="positive integers"):
            InitNode((0,), seed=None)

    def test_step_5_missing_seed(self) -> None:
        with pytest.raises(UninitializedNodeError) as excinfo:
            InitNode((4, 4), name="src")
        assert str(excinfo.value) == ("InitNode 'src': seed is required for init nodes, got None")

    @pytest.mark.parametrize(
        "seed",
        [-1, UINT64_MAX + 1, 1.5],
        ids=["negative", "above-uint64", "float"],
    )
    def test_step_6_invalid_seed(self, seed: Any) -> None:
        with pytest.raises(UninitializedNodeError, match=r"integer in \[0, 2\*\*64\)"):
            InitNode((4, 4), seed=seed)

    def test_step_6_message(self) -> None:
        with pytest.raises(UninitializedNodeError) as excinfo:
            InitNode((4, 4), seed=-1, name="src")
        assert str(excinfo.value) == (
            "InitNode 'src': seed must be an integer in [0, 2**64), got -1"
        )

    def test_step_6_precedes_enum_check(self) -> None:
        with pytest.raises(UninitializedNodeError, match="seed must be an integer"):
            InitNode((4, 4), seed=-1, distribution="poisson")  # type: ignore[arg-type]

    def test_step_7_bad_dtype_is_api_misuse_not_a_dag_error(self) -> None:
        """A bad enum is a ValueError: the DagBuildError family is reserved for maths."""
        with pytest.raises(ValueError, match="dtype must be one of") as excinfo:
            InitNode((4,), seed=1, dtype="int8")  # type: ignore[arg-type]
        assert not isinstance(excinfo.value, DagBuildError)

    def test_step_7_bad_distribution(self) -> None:
        with pytest.raises(ValueError, match="distribution must be one of") as excinfo:
            InitNode((4,), seed=1, distribution="poisson")  # type: ignore[arg-type]
        assert not isinstance(excinfo.value, DagBuildError)


class TestBoolTrap:
    """``bool`` subclasses ``int``, so it slips past a naive isinstance check."""

    def test_seed_true_is_rejected(self) -> None:
        with pytest.raises(UninitializedNodeError, match="seed must be an integer"):
            InitNode((4, 4), seed=True)

    def test_seed_false_is_rejected(self) -> None:
        """``False`` is 0, an otherwise legal seed, so this is the sharper case."""
        with pytest.raises(UninitializedNodeError, match="seed must be an integer"):
            InitNode((4, 4), seed=False)

    def test_bool_extent_is_rejected(self) -> None:
        with pytest.raises(UninitializedNodeError, match="positive integers"):
            InitNode((True, 2), seed=1)

    def test_bool_extent_rejected_even_though_true_is_one(self) -> None:
        with pytest.raises(UninitializedNodeError, match="positive integers"):
            InitNode((True,), seed=1)


class TestBoundaryValues:
    def test_seed_zero_is_accepted(self) -> None:
        assert InitNode((2,), seed=0).seed == 0

    def test_seed_uint64_max_is_accepted(self) -> None:
        assert InitNode((2,), seed=UINT64_MAX).seed == UINT64_MAX

    def test_extent_one_is_accepted(self) -> None:
        assert InitNode((1,), seed=1).output_shape == (1,)


class TestAccepted:
    @pytest.mark.parametrize("distribution", ["uniform", "normal", "zeros", "ones"])
    def test_all_four_distributions(self, distribution: str) -> None:
        node = InitNode((3,), seed=7, distribution=distribution)  # type: ignore[arg-type]
        assert node.distribution == distribution

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_both_dtypes(self, dtype: str) -> None:
        assert InitNode((3,), seed=7, dtype=dtype).dtype == dtype  # type: ignore[arg-type]

    def test_default_dtype_and_distribution(self) -> None:
        node = InitNode((3,), seed=7)
        assert node.dtype == "float64"
        assert node.distribution == "uniform"

    def test_zeros_and_ones_still_require_a_seed(self) -> None:
        """The engine needs no conditional logic, so the seed is mandatory regardless."""
        with pytest.raises(UninitializedNodeError, match="seed is required"):
            InitNode((3,), distribution="zeros")

    def test_shape_accepts_any_sequence(self) -> None:
        assert InitNode([2, 3], seed=1).output_shape == (2, 3)

    def test_source_node_has_no_inputs(self) -> None:
        assert InitNode((3,), seed=1).inputs == ()

    def test_op_discriminator(self) -> None:
        assert InitNode((3,), seed=1).op == "init"

    def test_payload_shape_is_a_list_not_a_tuple(self) -> None:
        """Keeps to_dict output directly comparable against parsed JSON."""
        doc = InitNode((2, 3), seed=1).to_dict("n", [], include_hints=False)
        assert doc["shape"] == [2, 3]


class TestNameValidation:
    @pytest.mark.parametrize(
        "name",
        ["1leading_digit", "has space", "has/slash", "", "x" * 65],
        ids=["leading-digit", "space", "slash", "empty", "too-long"],
    )
    def test_illegal_names_rejected(self, name: str) -> None:
        with pytest.raises(ValueError, match="name must match"):
            InitNode((2,), seed=1, name=name)

    @pytest.mark.parametrize("name", ["lhs", "_private", "a.b-c_1", "x" * 64])
    def test_legal_names_accepted(self, name: str) -> None:
        assert InitNode((2,), seed=1, name=name).label == name

    def test_unnamed_node_has_no_label(self) -> None:
        assert InitNode((2,), seed=1).label is None


class TestInference:
    def test_infer_returns_the_declared_shape_and_dtype(self) -> None:
        """A source node has nothing to infer, so the abstract hook is a lookup."""
        node = InitNode((2, 3), seed=1, dtype="float32")
        assert node._infer(()) == ((2, 3), "float32")


class TestRewireOnSourceNode:
    def test_rewire_always_raises_index_error(self) -> None:
        """An init node has no operands, so every index is out of range."""
        node = InitNode((2,), seed=1)
        with pytest.raises(IndexError):
            node.rewire(0, InitNode((2,), seed=2))
