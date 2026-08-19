"""The ``add``, ``multiply``, ``scale``, and ``mod`` nodes."""

from __future__ import annotations

import math
from typing import ClassVar

from tasks.dtypes import DType, JsonDict, OpName, Shape, promote
from tasks.node import Node
from tasks.shapes import (
    describe,
    flops_elementwise,
    flops_mod,
    infer_add,
    infer_mod,
    infer_multiply,
    infer_scale,
)


class AddNode(Node):
    """Elementwise sum of exactly two operands of identical shape."""

    OP: ClassVar[OpName] = "add"

    def __init__(
        self,
        left: Node,
        right: Node,
        *,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        """Sum two tensors elementwise.

        Args:
            left: Left operand.
            right: Right operand; its shape must equal ``left``'s exactly.
            name: Optional explicit node ID.
            label: Optional free-form annotation emitted as the schema's ``label``.

        Raises:
            ShapeMismatchError: If the operand shapes differ. Broadcasting is unsupported.
            ValueError: If ``name`` is not a legal node ID.
        """
        operands = (left, right)
        output_shape, dtype = self._infer(operands)
        super().__init__(operands, output_shape, dtype, name=name, label=label)

    def _infer(self, inputs: tuple[Node, ...]) -> tuple[Shape, DType]:
        """Resolve the summed shape and the promoted dtype.

        Args:
            inputs: The two operand nodes.

        Returns:
            The common shape and the promoted dtype.

        Raises:
            ShapeMismatchError: If the operand shapes differ.
        """
        left, right = inputs
        where = describe(left.display_id, right.display_id)
        shape = infer_add(left.output_shape, right.output_shape, where=where)
        return shape, promote(left.dtype, right.dtype)

    def _payload(self) -> JsonDict:
        """Return no op-specific fields.

        Returns:
            An empty mapping; ``add`` carries only the common node fields.
        """
        return {}

    def est_flops(self) -> float:
        """Return one addition per output element.

        Returns:
            The output element count.
        """
        return flops_elementwise(self._output_shape)


class MultiplyNode(Node):
    """Elementwise (Hadamard) product of two tensors of identical shape.

    Structurally identical to :class:`AddNode`; only ``OP`` and the inference rule's error text
    differ. This is deliberately *not* a contraction -- ``dot_product`` is the contraction, and
    ``*`` never means one.
    """

    OP: ClassVar[OpName] = "multiply"

    def __init__(
        self,
        left: Node,
        right: Node,
        *,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        """Multiply two tensors elementwise.

        Args:
            left: Left operand.
            right: Right operand; its shape must equal ``left``'s exactly.
            name: Optional explicit node ID.
            label: Optional free-form annotation emitted as the schema's ``label``.

        Raises:
            ShapeMismatchError: If the operand shapes differ. Broadcasting is unsupported.
            ValueError: If ``name`` is not a legal node ID.
        """
        operands = (left, right)
        output_shape, dtype = self._infer(operands)
        super().__init__(operands, output_shape, dtype, name=name, label=label)

    def _infer(self, inputs: tuple[Node, ...]) -> tuple[Shape, DType]:
        """Resolve the product shape and the promoted dtype.

        Args:
            inputs: The two operand nodes.

        Returns:
            The common shape and the promoted dtype.

        Raises:
            ShapeMismatchError: If the operand shapes differ.
        """
        left, right = inputs
        where = describe(left.display_id, right.display_id)
        shape = infer_multiply(left.output_shape, right.output_shape, where=where)
        return shape, promote(left.dtype, right.dtype)

    def _payload(self) -> JsonDict:
        """Return no op-specific fields.

        Returns:
            An empty mapping; ``multiply`` carries only the common node fields.
        """
        return {}

    def est_flops(self) -> float:
        """Return one multiplication per output element.

        Returns:
            The output element count.
        """
        return flops_elementwise(self._output_shape)


class ModNode(Node):
    """A tensor reduced elementwise to its non-negative remainder modulo a scalar."""

    OP: ClassVar[OpName] = "mod"

    def __init__(
        self,
        operand: Node,
        modulus: float,
        *,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        """Reduce a tensor elementwise to the non-negative remainder modulo a scalar.

        The result lies in ``[0, modulus)`` -- floored semantics as in Python's ``%`` and
        ``numpy.mod``, not C's ``std::fmod``, which carries the sign of the dividend. The
        schema's ``modulus`` description spells this out for the C++ authors.

        These are floating-point dtypes, so the remainder is exact only while operands stay
        inside the exactly-representable integer range (2**53 for ``float64``, 2**24 for
        ``float32``). That bound is checked by :func:`tasks.math.powmod`, not here: a
        non-integer operand is a legitimate use of a bare ``mod``.

        Args:
            operand: Tensor to reduce.
            modulus: Strictly positive scalar modulus.
            name: Optional explicit node ID.
            label: Optional free-form annotation emitted as the schema's ``label``.

        Raises:
            TypeError: If ``modulus`` is not an ``int`` or ``float``, or is a ``bool``.
            ValueError: If ``modulus`` is not strictly positive, or is NaN or infinite.
        """
        if isinstance(modulus, bool) or not isinstance(modulus, int | float):
            raise TypeError(f"modulus must be an int or float, got {type(modulus).__name__}")
        if not math.isfinite(modulus):
            raise ValueError(f"modulus must be finite, got {modulus!r}")
        # The schema sets exclusiveMinimum: 0, so a non-positive modulus would produce a
        # document the engine rejects at parse time. Catching it here gives a line number.
        if modulus <= 0:
            raise ValueError(f"modulus must be strictly positive, got {modulus!r}")
        self._modulus = float(modulus)
        operands = (operand,)
        output_shape, dtype = self._infer(operands)
        super().__init__(operands, output_shape, dtype, name=name, label=label)

    @property
    def modulus(self) -> float:
        """Strictly positive scalar the operand is reduced by."""
        return self._modulus

    def _infer(self, inputs: tuple[Node, ...]) -> tuple[Shape, DType]:
        """Resolve the reduced shape and dtype, both inherited from the operand.

        Args:
            inputs: The single operand node.

        Returns:
            The operand's shape and dtype, unchanged.

        Raises:
            DimensionalityError: If the operand rank exceeds the maximum.
        """
        (operand,) = inputs
        where = describe(operand.display_id)
        shape = infer_mod(operand.output_shape, where=where)
        return shape, operand.dtype

    def _payload(self) -> JsonDict:
        """Return the mod-specific schema field.

        Returns:
            The required ``modulus`` field. Never ``factor``: the schema forbids it here.
        """
        return {"modulus": self._modulus}

    def est_flops(self) -> float:
        """Return two operations per output element.

        Returns:
            ``2 * prod(output_shape)``, covering the division and the floored correction.
        """
        return flops_mod(self._output_shape)


class ScaleNode(Node):
    """A tensor multiplied by a scalar constant, preserving shape and dtype."""

    OP: ClassVar[OpName] = "scale"

    def __init__(
        self,
        operand: Node,
        factor: float,
        *,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        """Scale a tensor by a scalar constant.

        Shape and dtype are inherited from the operand, so a ``float32`` tensor scaled by a
        Python float stays ``float32``.

        Args:
            operand: The tensor node to scale.
            factor: Scalar multiplier; stored as a float.
            name: Optional explicit node ID.
            label: Optional free-form annotation emitted as the schema's ``label``.

        Raises:
            TypeError: If ``factor`` is not an ``int`` or ``float``.
            ValueError: If ``factor`` is NaN or infinite and therefore not serializable.
        """
        if isinstance(factor, bool) or not isinstance(factor, int | float):
            raise TypeError(f"factor must be an int or float, got {type(factor).__name__}")
        # JSON has no NaN or Infinity literal, so a non-finite constant produces a document
        # the engine cannot parse. NaN arising at runtime is the engine's problem; a NaN
        # constant is ours.
        if not math.isfinite(factor):
            raise ValueError(f"factor must be finite, got {factor!r}")
        self._factor = float(factor)
        operands = (operand,)
        output_shape, dtype = self._infer(operands)
        super().__init__(operands, output_shape, dtype, name=name, label=label)

    @property
    def factor(self) -> float:
        """Scalar multiplier applied to the operand."""
        return self._factor

    def _infer(self, inputs: tuple[Node, ...]) -> tuple[Shape, DType]:
        """Resolve the scaled shape and dtype, both inherited from the operand.

        Args:
            inputs: The single operand node.

        Returns:
            The operand's shape and dtype, unchanged.

        Raises:
            DimensionalityError: If the operand rank exceeds the maximum.
        """
        (operand,) = inputs
        where = describe(operand.display_id)
        shape = infer_scale(operand.output_shape, where=where)
        return shape, operand.dtype

    def _payload(self) -> JsonDict:
        """Return the scale-specific schema field.

        Returns:
            The required ``factor`` field.
        """
        return {"factor": self._factor}

    def est_flops(self) -> float:
        """Return one multiply per output element.

        Returns:
            The output element count.
        """
        return flops_elementwise(self._output_shape)
