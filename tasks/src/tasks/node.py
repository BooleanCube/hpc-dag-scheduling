"""The :class:`Node` abstract base class and its operator overloads.

Nodes are values, not graph members: a node never holds a reference to a
:class:`~tasks.graph.Graph`. That is what makes bare operator overloading work, because
``a + b`` has no graph to register with. A :class:`~tasks.graph.Graph` is built *from* output
nodes and discovers the rest by walking backwards through :attr:`Node.inputs`.

Shape and dtype are resolved eagerly in ``__init__``, so an ill-formed expression raises at the
line that wrote it rather than at serialization time.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Final

from tasks.dtypes import DType, JsonDict, OpName, Shape

NODE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
"""Schema pattern every node ID -- user-supplied or generated -- must satisfy."""

MAX_LABEL_LENGTH: Final[int] = 128
"""Schema ``maxLength`` for the optional ``label`` field."""

_seq_counter = 0
"""Process-wide construction counter backing provisional IDs and the sort tie-break.

A plain integer rather than an ``itertools.count`` so that :func:`peek_provisional_id` can
report the next ID without consuming it -- which is what lets a subclass name itself in a
validation error raised before ``Node.__init__`` ever runs.
"""


def _next_seq() -> int:
    """Consume and return the next construction sequence number.

    Returns:
        A process-unique, monotonically increasing integer.
    """
    global _seq_counter
    value = _seq_counter
    _seq_counter += 1
    return value


def peek_provisional_id(op: OpName, name: str | None = None) -> str:
    """Return the ID the next node of this op would take, without consuming it.

    Used to name a node in a validation error raised before the node exists.

    Args:
        op: Schema ``op`` discriminator of the node about to be built.
        name: User-supplied name, which takes precedence when present.

    Returns:
        ``name`` if given, otherwise the provisional ID the next node would receive.
    """
    return name if name is not None else f"{op}_{_seq_counter}"


class Node(ABC):
    """Abstract base for every node in a mathematical DAG.

    A node is an immutable-by-convention description of one operation. Its output shape and
    dtype are resolved eagerly at construction, so an ill-formed expression fails at the line
    that wrote it rather than at serialization time.

    Nodes are compared and hashed by *identity*. The reachability walk and topological sort
    both key dictionaries on node objects, and two structurally identical nodes are genuinely
    two nodes; do not give this class a value-based ``__eq__``.

    Attributes:
        OP: Schema ``op`` discriminator for the concrete subclass.
    """

    OP: ClassVar[OpName]

    def __init__(
        self,
        inputs: Sequence[Node],
        output_shape: Shape,
        dtype: DType,
        *,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        """Record one operation's operands and its resolved output type.

        ``name`` and ``label`` are different things and must not be confused. ``name`` becomes
        the serialized ``id``: it has to match the schema's node-ID pattern and has to be unique
        across the graph. ``label`` is free-form annotation, capped only in length, never
        load-bearing, and deliberately non-unique -- which is what lets a composite tag all
        thirty of its nodes ``"sin/..."`` without any of them colliding.

        Args:
            inputs: Operand nodes, in an order that is significant for non-commutative ops.
            output_shape: Shape this node produces, already inferred by the subclass.
            dtype: Element type this node produces, already promoted by the subclass.
            name: Optional explicit node ID, used verbatim as the serialized ``id``.
            label: Optional human-readable annotation, emitted as the schema's ``label``.

        Raises:
            ValueError: If ``name`` does not match the schema's node-ID pattern, or ``label`` is
                empty or longer than the schema's limit.
        """
        if name is not None and not NODE_ID_PATTERN.match(name):
            raise ValueError(f"name must match {NODE_ID_PATTERN.pattern!r}, got {name!r}")
        if label is not None and not 1 <= len(label) <= MAX_LABEL_LENGTH:
            raise ValueError(f"label must be 1 to {MAX_LABEL_LENGTH} characters, got {len(label)}")
        self._inputs: tuple[Node, ...] = tuple(inputs)
        self._output_shape: Shape = output_shape
        self._dtype: DType = dtype
        self._name = name
        self._label = label
        self._seq: int = _next_seq()
        self._provisional_id: str = f"{type(self).OP}_{self._seq}"

    @property
    def inputs(self) -> tuple[Node, ...]:
        """Immutable view of this node's operands."""
        return self._inputs

    @property
    def output_shape(self) -> Shape:
        """Shape of the tensor this node produces; empty for a rank-0 scalar."""
        return self._output_shape

    @property
    def dtype(self) -> DType:
        """Element type of the tensor this node produces."""
        return self._dtype

    @property
    def name(self) -> str | None:
        """Explicit node ID this node was given, or ``None`` if it was left unnamed.

        This is what :meth:`~tasks.graph.Graph.serialize` uses verbatim as the ``id``.
        """
        return self._name

    @property
    def label(self) -> str | None:
        """Free-form annotation emitted as the schema's ``label`` field.

        Falls back to :attr:`name` when no explicit label was given, so a named node is still
        labelled for traces and visualisations.
        """
        return self._label if self._label is not None else self._name

    @property
    def op(self) -> OpName:
        """Schema ``op`` discriminator for this node."""
        return type(self).OP

    @property
    def display_id(self) -> str:
        """Best-effort identifier for error messages.

        Returns:
            The user-supplied name when there is one, otherwise the provisional
            construction-time ID. The canonical serialized ID is assigned by
            :meth:`~tasks.graph.Graph.serialize` and is not known here.
        """
        return self._name if self._name is not None else self._provisional_id

    def __repr__(self) -> str:
        """Return a debug representation naming the op, ID, shape, and dtype."""
        return (
            f"<{type(self).__name__} {self.display_id} "
            f"shape={self._output_shape} dtype={self._dtype}>"
        )

    @abstractmethod
    def _infer(self, inputs: tuple[Node, ...]) -> tuple[Shape, DType]:
        """Resolve the output shape and dtype for a candidate operand tuple.

        Called once at construction and again by :meth:`rewire`, so the two paths cannot
        drift apart.

        Args:
            inputs: Candidate operand nodes.

        Returns:
            The inferred ``(output_shape, dtype)`` pair.

        Raises:
            ShapeMismatchError: If the operands do not align for this operation.
            DimensionalityError: If an operand has the wrong rank for this operation.
        """

    @abstractmethod
    def _payload(self) -> JsonDict:
        """Return the op-specific schema fields for this node.

        Returns:
            ``{"seed", "shape", "distribution"}`` for init nodes, ``{"factor"}`` for scale
            nodes, and an empty mapping for every other op.
        """

    @abstractmethod
    def est_flops(self) -> float:
        """Return the estimated floating-point operation count for this node.

        Returns:
            A non-negative cost proxy for the scheduler, not an exact count.
        """

    def rewire(self, index: int, new_input: Node) -> None:
        """Replace one operand in place, re-running this node's shape inference.

        The advanced escape hatch for programmatically generated graphs. Shape and dtype are
        re-derived and may raise, but cycles cannot be detected here because reachability is a
        whole-graph property -- :meth:`~tasks.graph.Graph.serialize` is what catches those.

        Args:
            index: Position in :attr:`inputs` to replace. Negative indices count from the end.
            new_input: Replacement operand node.

        Raises:
            TypeError: If ``new_input`` is not a :class:`Node`.
            IndexError: If ``index`` is out of range.
            ShapeMismatchError: If the new operand does not align with the remaining operands.
            DimensionalityError: If the new operand has the wrong rank for this operation.
        """
        if not isinstance(new_input, Node):
            raise TypeError(f"new_input must be a Node, got {type(new_input).__name__}")
        count = len(self._inputs)
        if not -count <= index < count:
            raise IndexError(
                f"{self.display_id}: operand index {index} out of range for {count} input(s)"
            )
        candidate = list(self._inputs)
        candidate[index] = new_input
        output_shape, dtype = self._infer(tuple(candidate))
        self._inputs = tuple(candidate)
        self._output_shape = output_shape
        self._dtype = dtype

    def _refresh(self) -> None:
        """Re-derive this node's output shape and dtype from its current operands.

        :meth:`rewire` updates the node it edits, but nodes hold no reference to their
        consumers, so a downstream node keeps whatever it was built with.
        :class:`~tasks.graph.Graph` calls this across the topological order before emitting
        anything, so a rewired graph cannot serialize a node whose declared ``output_shape``
        disagrees with the operands it names.

        Raises:
            ShapeMismatchError: If the current operands no longer align.
            DimensionalityError: If a current operand has the wrong rank.
        """
        self._output_shape, self._dtype = self._infer(self._inputs)

    def to_dict(self, node_id: str, input_ids: Sequence[str], *, include_hints: bool) -> JsonDict:
        """Render this node as a schema-conformant JSON object.

        Optional fields are omitted rather than emitted as ``null``: the schema sets
        ``additionalProperties: false`` at every level and forbids ``inputs`` on init nodes
        outright, so even an empty array would be rejected there.

        Args:
            node_id: Canonical ID assigned by the graph.
            input_ids: Resolved IDs of this node's operands, in operand order.
            include_hints: Whether to emit the ``hints.est_flops`` block.

        Returns:
            A mapping conforming to the schema's ``node`` definition.
        """
        doc: JsonDict = {
            "id": node_id,
            "op": self.op,
            "output_shape": list(self._output_shape),
            "dtype": self._dtype,
        }
        if input_ids:
            doc["inputs"] = list(input_ids)
        doc.update(self._payload())
        resolved_label = self.label
        if resolved_label is not None:
            doc["label"] = resolved_label
        if include_hints:
            doc["hints"] = {"est_flops": self.est_flops()}
        return doc

    def cross(self, other: Node, *, name: str | None = None, label: str | None = None) -> Node:
        """Return the 3-space cross product of this node with another.

        The mathematical notation is a multiplication sign, which is not a Python operator,
        and every available symbol would be cryptic; a named method reads better.

        Args:
            other: The right-hand length-3 vector node.
            name: Optional explicit node ID for the resulting node.
            label: Optional free-form annotation emitted as the schema's ``label``.

        Returns:
            A ``cross_product`` node of shape ``(3,)``.

        Raises:
            DimensionalityError: If either operand is not a length-3 rank-1 vector.
            ShapeMismatchError: If both operands are rank-1 but of different lengths.
        """
        from tasks.ops.products import CrossProductNode  # breaks import cycle

        return CrossProductNode(self, other, name=name, label=label)

    def __add__(self, other: Node) -> Node:
        """Return an ``add`` node summing this node with another, elementwise.

        Args:
            other: The right-hand operand; shapes must match exactly.

        Returns:
            An :class:`~tasks.ops.arithmetic.AddNode`, or ``NotImplemented`` if ``other`` is
            not a :class:`Node`.

        Raises:
            ShapeMismatchError: If the operand shapes differ.
        """
        from tasks.ops.arithmetic import AddNode  # breaks import cycle

        if not isinstance(other, Node):
            return NotImplemented
        return AddNode(self, other)

    def __sub__(self, other: Node) -> Node:
        """Return the two-node expansion of a difference.

        New in v1.2.0. Lowers to ``AddNode(self, ScaleNode(other, -1.0))``: subtraction is not a
        primitive, because a negative ``scale`` factor plus ``add`` covers it with no new C++
        path. Under the revised P3 an operator may expand, provided it expands to *its own*
        standard definition, and this is exactly the definition of subtraction.

        Note the asymmetry this creates when reading DAG sizes: ``a - b`` costs two nodes where
        ``a + b`` costs one, so comparing topologies across the two is not apples to apples.

        Args:
            other: The right-hand operand; shapes must match exactly.

        Returns:
            An :class:`~tasks.ops.arithmetic.AddNode` over a negating
            :class:`~tasks.ops.arithmetic.ScaleNode`, or ``NotImplemented`` if ``other`` is not
            a :class:`Node`.

        Raises:
            ShapeMismatchError: If the operand shapes differ.
        """
        from tasks.ops.arithmetic import AddNode, ScaleNode  # breaks import cycle

        if not isinstance(other, Node):
            return NotImplemented
        return AddNode(self, ScaleNode(other, -1.0))

    def __mul__(self, other: Node | float) -> Node:
        """Return an elementwise product, or a scalar scaling.

        Dispatches on the operand type: a :class:`Node` builds a
        :class:`~tasks.ops.arithmetic.MultiplyNode`, an ``int`` or ``float`` builds a
        :class:`~tasks.ops.arithmetic.ScaleNode`.

        **Node times node flipped to multiply in v1.2.0.** It previously raised, on the grounds
        that no elementwise product existed and reinterpreting ``*`` as a contraction would
        train users to write dimension-sensitive code with the dimension-insensitive operator.
        That objection stands and is preserved -- ``*`` still never means a contraction -- but
        v1.2.0 added the ``multiply`` primitive, so ``*`` can now mean elementwise exactly as it
        does in NumPy. ``@`` keeps the contraction to itself.

        ``bool`` is rejected. It subclasses ``int``, so ``a * True`` would otherwise silently
        build a scale-by-1.0 node, and it matches nothing a user could mean.

        Args:
            other: Another node, or a real scalar multiplier.

        Returns:
            A ``multiply`` or ``scale`` node, or ``NotImplemented`` for anything else.

        Raises:
            ShapeMismatchError: If ``other`` is a node whose shape differs from this one's.
        """
        from tasks.ops.arithmetic import MultiplyNode, ScaleNode  # breaks import cycle

        if isinstance(other, Node):
            return MultiplyNode(self, other)
        if isinstance(other, bool) or not isinstance(other, int | float):
            return NotImplemented
        return ScaleNode(self, float(other))

    def __rmul__(self, other: float) -> Node:
        """Return a ``scale`` node for a scalar written on the left, as in ``2 * a``.

        Only reached for non-node left operands; ``node * node`` is handled by
        :meth:`__mul__`.

        Args:
            other: Scalar multiplier.

        Returns:
            A :class:`~tasks.ops.arithmetic.ScaleNode`, or ``NotImplemented`` if ``other`` is
            not a real scalar.
        """
        return self.__mul__(other)

    def __truediv__(self, other: float) -> Node:
        """Return a ``scale`` node dividing this node by a scalar constant.

        Scalar divisors only. ``a / b`` between two nodes is ``NotImplemented``: there is no
        division primitive, deliberately, because division brings division-by-zero -- a runtime
        concern the engine owns, and one we cannot validate at build time without evaluating the
        tensor, which the lazy-evaluation contract forbids.

        Args:
            other: Scalar divisor.

        Returns:
            A :class:`~tasks.ops.arithmetic.ScaleNode` scaling by ``1 / other``, or
            ``NotImplemented`` if ``other`` is not a real scalar.

        Raises:
            ZeroDivisionError: If ``other`` is zero.
        """
        from tasks.ops.arithmetic import ScaleNode  # breaks import cycle

        if isinstance(other, Node | bool) or not isinstance(other, int | float):
            return NotImplemented
        return ScaleNode(self, 1.0 / other)

    def __mod__(self, other: float) -> Node:
        """Return a ``mod`` node reducing this node modulo a positive scalar.

        New in v1.2.0. The modulus is a scalar *field* on the node, not an operand, so
        ``a % b`` between two nodes is ``NotImplemented``.

        Args:
            other: Strictly positive scalar modulus.

        Returns:
            A :class:`~tasks.ops.arithmetic.ModNode`, or ``NotImplemented`` if ``other`` is not
            a real scalar.

        Raises:
            ValueError: If the modulus is not strictly positive, or is NaN or infinite.
        """
        from tasks.ops.arithmetic import ModNode  # breaks import cycle

        if isinstance(other, Node | bool) or not isinstance(other, int | float):
            return NotImplemented
        return ModNode(self, float(other))

    def __pow__(self, other: int) -> Node:
        """Return the binary-exponentiation expansion of this node raised to a power.

        New in v1.2.0. Delegates to :func:`tasks.math.pow`, so ``a ** n`` and
        ``tasks.math.pow(a, n)`` build identical subgraphs. The expansion costs
        ``floor(log2 n) + popcount(n) - 1`` multiply nodes, so ``a ** 1024`` is ten nodes rather
        than 1023.

        Args:
            other: Non-negative integer exponent.

        Returns:
            The node holding the result, or ``NotImplemented`` if ``other`` is not an ``int``.

        Raises:
            ValueError: If ``other`` is negative.
        """
        from tasks.math import pow as _pow  # breaks import cycle

        if isinstance(other, bool) or not isinstance(other, int):
            return NotImplemented
        return _pow(self, other)

    def __neg__(self) -> Node:
        """Return a ``scale`` node negating this node.

        Returns:
            A :class:`~tasks.ops.arithmetic.ScaleNode` with factor ``-1.0``.
        """
        from tasks.ops.arithmetic import ScaleNode  # breaks import cycle

        return ScaleNode(self, -1.0)

    def __matmul__(self, other: Node) -> Node:
        """Return a ``dot_product`` node contracting this node with another.

        PEP 465 added ``@`` to Python specifically as the matrix-multiplication operator and
        NumPy follows it, so ``(a @ b) * 0.5`` reads here exactly as it does in NumPy.

        Args:
            other: The right-hand operand.

        Returns:
            A :class:`~tasks.ops.products.DotProductNode`, or ``NotImplemented`` if ``other``
            is not a :class:`Node`.

        Raises:
            DimensionalityError: If either operand is not rank 1 or rank 2.
            ShapeMismatchError: If the contracted extents disagree.
        """
        from tasks.ops.products import DotProductNode  # breaks import cycle

        if not isinstance(other, Node):
            return NotImplemented
        return DotProductNode(self, other)
