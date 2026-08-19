"""The ``dot_product`` and ``cross_product`` nodes.

Operand order is significant for both and is preserved in the serialized ``inputs`` array.
"""

from __future__ import annotations

from typing import ClassVar

from tasks.dtypes import DType, JsonDict, OpName, Shape, promote
from tasks.node import Node
from tasks.shapes import describe, flops_cross, flops_dot, infer_cross, infer_dot


class DotProductNode(Node):
    """Contraction of two rank-1 or rank-2 operands.

    A vector-vector contraction collapses both operands to a rank-0 scalar, serialized as an
    empty ``output_shape`` array under schema 1.1.0.
    """

    OP: ClassVar[OpName] = "dot_product"

    def __init__(
        self,
        left: Node,
        right: Node,
        *,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        """Contract two tensors along the inner dimension.

        Args:
            left: Left operand; its trailing extent is contracted.
            right: Right operand; its leading extent is contracted.
            name: Optional explicit node ID.
            label: Optional free-form annotation emitted as the schema's ``label``.

        Raises:
            DimensionalityError: If either operand is not rank 1 or rank 2.
            ShapeMismatchError: If the contracted extents disagree.
            ValueError: If ``name`` is not a legal node ID.
        """
        operands = (left, right)
        output_shape, dtype = self._infer(operands)
        super().__init__(operands, output_shape, dtype, name=name, label=label)

    def _infer(self, inputs: tuple[Node, ...]) -> tuple[Shape, DType]:
        """Resolve the contracted shape and the promoted dtype.

        Args:
            inputs: The two operand nodes.

        Returns:
            The contracted shape and the promoted dtype.

        Raises:
            DimensionalityError: If either operand has an unsupported rank.
            ShapeMismatchError: If the contracted extents disagree.
        """
        left, right = inputs
        where = describe(left.display_id, right.display_id)
        shape = infer_dot(left.output_shape, right.output_shape, where=where)
        return shape, promote(left.dtype, right.dtype)

    def _payload(self) -> JsonDict:
        """Return no op-specific fields.

        Returns:
            An empty mapping; ``dot_product`` carries only the common node fields.
        """
        return {}

    def est_flops(self) -> float:
        """Return two FLOPs per output element per contracted step.

        Returns:
            ``2 * prod(output_shape) * contraction_dim``.
        """
        left, right = self._inputs
        return flops_dot(left.output_shape, right.output_shape)


class CrossProductNode(Node):
    """The 3-space cross product of two length-3 vectors."""

    OP: ClassVar[OpName] = "cross_product"

    def __init__(
        self,
        left: Node,
        right: Node,
        *,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        """Take the cross product of two length-3 vectors.

        Args:
            left: Left operand, a length-3 rank-1 vector.
            right: Right operand, a length-3 rank-1 vector.
            name: Optional explicit node ID.
            label: Optional free-form annotation emitted as the schema's ``label``.

        Raises:
            DimensionalityError: If either operand is not rank 1, or is not length 3.
            ShapeMismatchError: If both operands are rank 1 but of different lengths.
            ValueError: If ``name`` is not a legal node ID.
        """
        operands = (left, right)
        output_shape, dtype = self._infer(operands)
        super().__init__(operands, output_shape, dtype, name=name, label=label)

    def _infer(self, inputs: tuple[Node, ...]) -> tuple[Shape, DType]:
        """Resolve the ``(3,)`` output shape and the promoted dtype.

        Args:
            inputs: The two operand nodes.

        Returns:
            ``(3,)`` and the promoted dtype.

        Raises:
            DimensionalityError: If either operand is not a length-3 rank-1 vector.
            ShapeMismatchError: If both operands are rank 1 but of different lengths.
        """
        left, right = inputs
        where = describe(left.display_id, right.display_id)
        shape = infer_cross(left.output_shape, right.output_shape, where=where)
        return shape, promote(left.dtype, right.dtype)

    def _payload(self) -> JsonDict:
        """Return no op-specific fields.

        Returns:
            An empty mapping; ``cross_product`` carries only the common node fields.
        """
        return {}

    def est_flops(self) -> float:
        """Return the fixed cost of a 3-space cross product.

        Returns:
            ``9.0`` -- six multiplies and three subtractions.
        """
        return flops_cross()
