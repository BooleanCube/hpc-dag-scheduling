"""Tests for the two v1.2.0 primitives, ``multiply`` and ``mod``.

``multiply`` shares its inference rule with ``add`` by delegation, so the interesting assertions
are that the two really do stay in lockstep and that ``multiply`` is never mistaken for a
contraction. ``mod``'s interesting surface is its scalar validation and the fact that it carries a
``modulus`` payload where ``scale`` carries a ``factor``.
"""

import pytest

from tasks import (
    AddNode,
    Graph,
    InitNode,
    ModNode,
    MultiplyNode,
    Node,
    ScaleNode,
)
from tasks.dtypes import Shape
from tasks.exceptions import DimensionalityError, ShapeMismatchError
from tasks.shapes import flops_elementwise, flops_mod, infer_add, infer_mod, infer_multiply

WHERE = "nodes 'a' and 'b'"


class TestMultiplyShapeRule:
    @pytest.mark.parametrize("shape", [(), (4,), (4, 4), (2, 3, 5), (1,) * 8])
    def test_identical_shapes_pass_through(self, shape: Shape) -> None:
        assert infer_multiply(shape, shape, where=WHERE) == shape

    def test_rank_zero_operands(self) -> None:
        """Rank 0 is exactly why the init floor could be lifted."""
        assert infer_multiply((), (), where=WHERE) == ()

    def test_extent_mismatch_raises(self) -> None:
        with pytest.raises(ShapeMismatchError) as excinfo:
            infer_multiply((2, 2), (3, 3), where=WHERE)
        assert str(excinfo.value) == (
            "multiply: operand shapes must match exactly, got (2, 2) and (3, 3) (nodes 'a' and 'b')"
        )

    def test_rank_mismatch_raises(self) -> None:
        with pytest.raises(ShapeMismatchError):
            infer_multiply((4,), (4, 1), where=WHERE)

    def test_no_broadcasting(self) -> None:
        with pytest.raises(ShapeMismatchError):
            infer_multiply((4, 3), (1, 3), where=WHERE)

    def test_contraction_shapes_are_rejected(self) -> None:
        """`(4,3) * (3,2)` is a shape error, not a silent matrix product."""
        with pytest.raises(ShapeMismatchError):
            infer_multiply((4, 3), (3, 2), where=WHERE)


class TestMultiplyMatchesAddExactly:
    """The two delegate to one helper; a divergence would be a silent correctness bug."""

    @pytest.mark.parametrize(
        ("a", "b"),
        [((), ()), ((3,), (3,)), ((2, 2), (2, 2)), ((1,) * 8, (1,) * 8)],
    )
    def test_accepted_shapes_agree(self, a: Shape, b: Shape) -> None:
        assert infer_multiply(a, b, where=WHERE) == infer_add(a, b, where=WHERE)

    @pytest.mark.parametrize(
        ("a", "b"), [((2,), (3,)), ((2, 2), (2,)), ((), (1,)), ((4, 3), (3, 4))]
    )
    def test_rejected_shapes_agree(self, a: Shape, b: Shape) -> None:
        with pytest.raises(ShapeMismatchError):
            infer_add(a, b, where=WHERE)
        with pytest.raises(ShapeMismatchError):
            infer_multiply(a, b, where=WHERE)

    def test_only_the_op_name_differs_in_the_message(self) -> None:
        with pytest.raises(ShapeMismatchError) as add_err:
            infer_add((2,), (3,), where=WHERE)
        with pytest.raises(ShapeMismatchError) as mul_err:
            infer_multiply((2,), (3,), where=WHERE)
        assert str(add_err.value).replace("add:", "OP:") == str(mul_err.value).replace(
            "multiply:", "OP:"
        )

    def test_topology_is_identical_apart_from_op(self) -> None:
        left, right = InitNode((2, 2), seed=1), InitNode((2, 2), seed=2)
        summed = AddNode(left, right)
        product = MultiplyNode(left, right)
        assert summed.inputs == product.inputs
        assert summed.output_shape == product.output_shape
        assert summed.dtype == product.dtype
        assert summed.est_flops() == product.est_flops()
        assert summed.op != product.op

    def test_serialized_forms_differ_only_in_op(self) -> None:
        left, right = InitNode((2, 2), seed=1), InitNode((2, 2), seed=2)
        add_doc = AddNode(left, right).to_dict("n", ["a", "b"], include_hints=True)
        mul_doc = MultiplyNode(left, right).to_dict("n", ["a", "b"], include_hints=True)
        assert add_doc.pop("op") == "add"
        assert mul_doc.pop("op") == "multiply"
        assert add_doc == mul_doc


class TestMultiplyNode:
    def test_op_discriminator(self) -> None:
        left, right = InitNode((2,), seed=1), InitNode((2,), seed=2)
        assert MultiplyNode(left, right).op == "multiply"

    def test_preserves_shape(self) -> None:
        left, right = InitNode((3, 4), seed=1), InitNode((3, 4), seed=2)
        assert MultiplyNode(left, right).output_shape == (3, 4)

    def test_dtype_promotes(self) -> None:
        left = InitNode((2,), seed=1, dtype="float32")
        right = InitNode((2,), seed=2, dtype="float64")
        assert MultiplyNode(left, right).dtype == "float64"

    def test_matching_float32_stays_float32(self) -> None:
        left = InitNode((2,), seed=1, dtype="float32")
        right = InitNode((2,), seed=2, dtype="float32")
        assert MultiplyNode(left, right).dtype == "float32"

    def test_payload_is_empty(self) -> None:
        left, right = InitNode((2,), seed=1), InitNode((2,), seed=2)
        doc = MultiplyNode(left, right).to_dict("n", ["a", "b"], include_hints=False)
        assert "factor" not in doc
        assert "modulus" not in doc

    def test_flops_is_one_per_element(self) -> None:
        left, right = InitNode((4, 4), seed=1), InitNode((4, 4), seed=2)
        assert MultiplyNode(left, right).est_flops() == flops_elementwise((4, 4))

    def test_operand_order_is_preserved(self) -> None:
        left, right = InitNode((2,), seed=1), InitNode((2,), seed=2)
        assert MultiplyNode(left, right).inputs == (left, right)

    def test_rewire_reinfers(self) -> None:
        left, right = InitNode((2,), seed=1), InitNode((2,), seed=2)
        node = MultiplyNode(left, right)
        with pytest.raises(ShapeMismatchError):
            node.rewire(1, InitNode((3,), seed=3))
        assert node.inputs == (left, right)


class TestModShapeRule:
    @pytest.mark.parametrize("shape", [(), (5,), (4, 4), (1,) * 8])
    def test_preserves_shape(self, shape: Shape) -> None:
        assert infer_mod(shape, where=WHERE) == shape

    def test_rejects_rank_above_maximum(self) -> None:
        with pytest.raises(DimensionalityError, match="rank must not exceed 8"):
            infer_mod((1,) * 9, where=WHERE)


class TestModNode:
    def test_op_discriminator(self) -> None:
        assert ModNode(InitNode((2,), seed=1), 7.0).op == "mod"

    @pytest.mark.parametrize("shape", [(), (3,), (2, 2)])
    def test_preserves_shape(self, shape: Shape) -> None:
        assert ModNode(InitNode(shape, seed=1), 7.0).output_shape == shape

    def test_preserves_dtype(self) -> None:
        source = InitNode((2,), seed=1, dtype="float32")
        assert ModNode(source, 7.0).dtype == "float32"

    def test_modulus_is_stored_as_a_float(self) -> None:
        node = ModNode(InitNode((2,), seed=1), 7)
        assert isinstance(node.modulus, float)
        assert node.modulus == 7.0

    def test_payload_carries_modulus_not_factor(self) -> None:
        doc = ModNode(InitNode((2,), seed=1), 7.0).to_dict("n", ["a"], include_hints=False)
        assert doc["modulus"] == 7.0
        assert "factor" not in doc

    def test_flops_is_two_per_element(self) -> None:
        node = ModNode(InitNode((4, 4), seed=1), 7.0)
        assert node.est_flops() == flops_mod((4, 4))
        assert node.est_flops() == 32.0

    def test_single_input(self) -> None:
        source = InitNode((2,), seed=1)
        assert ModNode(source, 7.0).inputs == (source,)


class TestModulusValidation:
    def test_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            ModNode(InitNode((2,), seed=1), 0.0)

    @pytest.mark.parametrize("modulus", [-1.0, -7, -0.5])
    def test_negative_is_rejected(self, modulus: float) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            ModNode(InitNode((2,), seed=1), modulus)

    @pytest.mark.parametrize("modulus", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_is_rejected(self, modulus: float) -> None:
        """JSON has no NaN or Infinity literal, so a non-finite constant is unserializable."""
        with pytest.raises(ValueError, match="must be finite"):
            ModNode(InitNode((2,), seed=1), modulus)

    def test_bool_is_rejected(self) -> None:
        """True would otherwise silently mean a modulus of 1."""
        with pytest.raises(TypeError, match="must be an int or float"):
            ModNode(InitNode((2,), seed=1), True)

    def test_non_numeric_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be an int or float"):
            ModNode(InitNode((2,), seed=1), "7")  # type: ignore[arg-type]

    def test_non_integer_modulus_is_allowed(self) -> None:
        """A bare mod on non-integer data is a legitimate use; only powmod checks integrality."""
        assert ModNode(InitNode((2,), seed=1), 0.5).modulus == 0.5

    def test_very_small_positive_modulus_is_allowed(self) -> None:
        assert ModNode(InitNode((2,), seed=1), 1e-300).modulus == 1e-300


class TestSerializedGraphs:
    def test_multiply_graph_conforms(self, assert_conforms: object) -> None:
        left, right = InitNode((2, 2), seed=1), InitNode((2, 2), seed=2)
        document = Graph([left * right], dag_id="mul").serialize()
        assert callable(assert_conforms)
        assert_conforms(document)
        assert document["nodes"][-1]["op"] == "multiply"

    def test_mod_graph_conforms(self, assert_conforms: object) -> None:
        document = Graph([InitNode((2, 2), seed=1) % 7], dag_id="mod").serialize()
        assert callable(assert_conforms)
        assert_conforms(document)
        assert document["nodes"][-1]["modulus"] == 7.0

    def test_rank_zero_multiply_chain_conforms(self, assert_conforms: object) -> None:
        u, v = InitNode((3,), seed=1), InitNode((3,), seed=2)
        scalar = u @ v
        document = Graph([scalar * scalar], dag_id="rank0mul").serialize()
        assert callable(assert_conforms)
        assert_conforms(document)
        assert document["nodes"][-1]["output_shape"] == []

    def test_scale_and_mod_are_distinguishable_on_the_wire(self) -> None:
        source = InitNode((2,), seed=1)
        document = Graph([ScaleNode(source, 3.0) + ModNode(source, 3.0)], dag_id="both").serialize()
        by_op = {node["op"]: node for node in document["nodes"]}
        assert "factor" in by_op["scale"]
        assert "modulus" not in by_op["scale"]
        assert "modulus" in by_op["mod"]
        assert "factor" not in by_op["mod"]


class TestNodeCompatibility:
    def test_multiply_accepts_a_label(self) -> None:
        left, right = InitNode((2,), seed=1), InitNode((2,), seed=2)
        assert MultiplyNode(left, right, label="grp/step").label == "grp/step"

    def test_mod_accepts_a_label(self) -> None:
        assert ModNode(InitNode((2,), seed=1), 7.0, label="grp/mod").label == "grp/mod"

    def test_label_is_not_the_node_id(self) -> None:
        """Labels may repeat and may contain '/', which the node-ID pattern forbids."""
        left, right = InitNode((2,), seed=1), InitNode((2,), seed=2)
        node: Node = MultiplyNode(left, right, label="grp/step")
        assert node.name is None
        document = Graph([node], dag_id="lbl").serialize(include_timestamp=False)
        emitted = document["nodes"][-1]
        assert emitted["id"] == "multiply_2"
        assert emitted["label"] == "grp/step"


class TestLabelValidation:
    """``label`` is capped by the schema's ``maxLength``, unlike the node-ID pattern."""

    def test_label_at_the_limit_is_accepted(self) -> None:
        label = "x" * 128
        assert (
            MultiplyNode(InitNode((2,), seed=1), InitNode((2,), seed=2), label=label).label == label
        )

    def test_label_past_the_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="1 to 128 characters"):
            MultiplyNode(InitNode((2,), seed=1), InitNode((2,), seed=2), label="x" * 129)

    def test_empty_label_is_rejected(self) -> None:
        """An empty label would serialize a meaningless key rather than omitting it."""
        with pytest.raises(ValueError, match="1 to 128 characters"):
            ModNode(InitNode((2,), seed=1), 7.0, label="")

    def test_label_may_contain_characters_the_node_id_pattern_forbids(self) -> None:
        """Which is the whole reason label is separate from name."""
        node = ModNode(InitNode((2,), seed=1), 7.0, label="sin/coeff3")
        assert node.label == "sin/coeff3"
        with pytest.raises(ValueError, match="name must match"):
            ModNode(InitNode((2,), seed=1), 7.0, name="sin/coeff3")
