"""Pure shape-inference rules for the supported operations.

Every function here takes plain shape tuples, raises on an invalid combination, and returns
the output shape. Nothing in this module imports ``Node``, which keeps the mathematically
interesting logic unit-testable without any graph-construction ceremony.

The ``where`` keyword on each rule is a pre-rendered operand description built by
:func:`describe`; it exists purely so error messages can name the offending nodes.

Error messages follow one convention: ``"{op}: {problem}, got {actual} ({where})"``.
"""

import math
import sys

from tasks.dtypes import MAX_RANK, Shape
from tasks.exceptions import DimensionalityError, ShapeMismatchError


def _as_cost(count: int) -> float:
    """Convert an exact operation count to a float cost proxy, saturating on overflow.

    Extents are unbounded above, so a product can exceed the double range and make a plain
    ``float()`` raise ``OverflowError`` out of ``Graph.serialize``. The schema calls hints
    non-authoritative and requires that ignoring them cannot change the computed result, so
    saturating keeps the document emittable instead of failing the whole serialization over
    a scheduler estimate.

    Args:
        count: Exact integer operation count.

    Returns:
        ``float(count)``, or the largest finite float when the count exceeds it.
    """
    try:
        return float(count)
    except OverflowError:
        return sys.float_info.max


def describe(*nodes: str) -> str:
    """Render a parenthetical operand reference for an error message.

    Args:
        *nodes: Provisional or user-assigned node IDs, in operand order.

    Returns:
        For example ``"nodes 'init_0' and 'init_1'"``. A single ID yields ``"node 'x'"``,
        and three or more are comma-separated with a trailing ``"and"``.
    """
    quoted = [f"'{n}'" for n in nodes]
    if not quoted:
        return "no nodes"
    if len(quoted) == 1:
        return f"node {quoted[0]}"
    joined = ", ".join(quoted[:-1])
    return f"nodes {joined} and {quoted[-1]}"


def _infer_elementwise_binary(a: Shape, b: Shape, *, op: str, where: str) -> Shape:
    """Infer the output shape of a binary elementwise op requiring identical shapes.

    ``add`` and ``multiply`` share this rule exactly, differing only in the op name they put in
    the error message. They delegate here rather than each carrying a copy of the branch: a
    divergence between the two would be a silent correctness bug.

    Args:
        a: Shape of the left operand.
        b: Shape of the right operand.
        op: Schema op name, used only in the error message.
        where: Operand description for the error message, from :func:`describe`.

    Returns:
        The common shape.

    Raises:
        ShapeMismatchError: If the two shapes are not identical.
    """
    if a != b:
        raise ShapeMismatchError(
            f"{op}: operand shapes must match exactly, got {a} and {b} ({where})"
        )
    return a


def infer_add(a: Shape, b: Shape, *, where: str) -> Shape:
    """Infer the output shape of an elementwise sum.

    Shapes must be exactly equal, including rank. Broadcasting is deliberately unsupported:
    it would make the engine's buffer allocation and MPI decomposition depend on a rule the
    C++ side does not implement.

    Args:
        a: Shape of the left operand.
        b: Shape of the right operand.
        where: Operand description for the error message, from :func:`describe`.

    Returns:
        The common shape.

    Raises:
        ShapeMismatchError: If the two shapes are not identical.
    """
    return _infer_elementwise_binary(a, b, op="add", where=where)


def infer_multiply(a: Shape, b: Shape, *, where: str) -> Shape:
    """Infer the output shape of an elementwise (Hadamard) product.

    Identical rule to :func:`infer_add`: exact shape equality, no broadcasting. This is *not* a
    contraction -- ``dot_product`` is the contraction, and the two never share an operator.

    Args:
        a: Shape of the left operand.
        b: Shape of the right operand.
        where: Operand description for the error message, from :func:`describe`.

    Returns:
        The common shape.

    Raises:
        ShapeMismatchError: If the two shapes are not identical.
    """
    return _infer_elementwise_binary(a, b, op="multiply", where=where)


def infer_scale(a: Shape, *, where: str) -> Shape:
    """Infer the output shape of a scalar multiply.

    Scaling preserves shape and dtype unconditionally. This rule cannot fail for any shape
    that entered the graph through a validated node; the rank guard is defence in depth and
    gives the rank invariant a single home.

    Args:
        a: Shape of the operand.
        where: Operand description for the error message, from :func:`describe`.

    Returns:
        The operand shape, unchanged.

    Raises:
        DimensionalityError: If the operand rank exceeds ``MAX_RANK``.
    """
    if len(a) > MAX_RANK:
        raise DimensionalityError(
            f"scale: operand rank must not exceed {MAX_RANK}, got rank {len(a)} ({where})"
        )
    return a


def infer_mod(a: Shape, *, where: str) -> Shape:
    """Infer the output shape of an elementwise remainder by a positive scalar.

    Shape and dtype are preserved unconditionally, exactly like :func:`infer_scale`. This rule
    cannot fail on shapes; the modulus itself is validated in ``ModNode.__init__``.

    Args:
        a: Shape of the operand.
        where: Operand description for the error message, from :func:`describe`.

    Returns:
        The operand shape, unchanged.

    Raises:
        DimensionalityError: If the operand rank exceeds ``MAX_RANK``.
    """
    if len(a) > MAX_RANK:
        raise DimensionalityError(
            f"mod: operand rank must not exceed {MAX_RANK}, got rank {len(a)} ({where})"
        )
    return a


def infer_dot(a: Shape, b: Shape, *, where: str) -> Shape:
    """Infer the output shape of a contraction.

    Ranks 1 and 2 are supported in all four combinations; the contracted extent is always
    ``a[-1]`` against ``b[0]``, and the result is ``a[:-1] + b[1:]``. Vector-vector therefore
    collapses to rank 0, which schema 1.1.0 represents as an empty shape array.

    Args:
        a: Shape of the left operand.
        b: Shape of the right operand.
        where: Operand description for the error message, from :func:`describe`.

    Returns:
        The contracted shape, empty for a vector-vector product.

    Raises:
        DimensionalityError: If either operand is not rank 1 or rank 2.
        ShapeMismatchError: If the contracted extents disagree.
    """
    rank_a, rank_b = len(a), len(b)
    if rank_a not in (1, 2) or rank_b not in (1, 2):
        raise DimensionalityError(
            "dot_product: operands must be rank-1 or rank-2, "
            f"got rank {rank_a} and rank {rank_b} ({where})"
        )
    if a[-1] != b[0]:
        raise ShapeMismatchError(
            f"dot_product: inner dimensions must agree, got {a} @ {b}, {a[-1]} != {b[0]} ({where})"
        )
    return a[:-1] + b[1:]


def infer_cross(a: Shape, b: Shape, *, where: str) -> Shape:
    """Infer the output shape of a 3-space cross product.

    Checks run rank-first: a user who passed a matrix wants to hear "this needs a vector",
    not "this needs length 3".

    Args:
        a: Shape of the left operand.
        b: Shape of the right operand.
        where: Operand description for the error message, from :func:`describe`.

    Returns:
        Always ``(3,)``.

    Raises:
        DimensionalityError: If either operand is not rank 1, or the vectors are not length 3.
        ShapeMismatchError: If both operands are rank 1 but of different lengths.
    """
    if len(a) != 1 or len(b) != 1:
        raise DimensionalityError(
            "cross_product: operands must be rank-1 vectors, "
            f"got rank {len(a)} and rank {len(b)} ({where})"
        )
    if a != b:
        raise ShapeMismatchError(
            f"cross_product: operand shapes must match exactly, got {a} and {b} ({where})"
        )
    if a != (3,):
        raise DimensionalityError(
            f"cross_product: only defined for length-3 vectors, got length {a[0]} ({where})"
        )
    return (3,)


def flops_init(shape: Shape) -> float:
    """Estimate the cost of filling a source tensor: one PRNG draw per element.

    Args:
        shape: Shape of the tensor being initialized.

    Returns:
        The element count as a float, saturated at the largest finite float.
    """
    return _as_cost(math.prod(shape))


def flops_elementwise(shape: Shape) -> float:
    """Estimate the cost of an elementwise op: one operation per output element.

    Covers ``add``, ``multiply``, and ``scale``.

    Args:
        shape: Output shape of the node.

    Returns:
        The element count as a float, saturated at the largest finite float.
    """
    return _as_cost(math.prod(shape))


def flops_mod(shape: Shape) -> float:
    """Estimate the cost of an elementwise remainder.

    Args:
        shape: Output shape of the node.

    Returns:
        ``2 * prod(shape)`` -- a division plus the conditional add that floored semantics need
        on top of a truncated remainder, saturated at the largest finite float.
    """
    return _as_cost(2 * math.prod(shape))


def flops_dot(a: Shape, b: Shape) -> float:
    """Estimate the cost of a contraction as two FLOPs per output element per contracted step.

    Args:
        a: Shape of the left operand.
        b: Shape of the right operand.

    Returns:
        ``2 * prod(output_shape) * contraction_dim``, which is correct for all four supported
        rank combinations including the rank-0 vector-vector result, saturated at the largest
        finite float.
    """
    output_shape = a[:-1] + b[1:]
    return _as_cost(2 * math.prod(output_shape) * a[-1])


def flops_cross() -> float:
    """Estimate the cost of a 3-space cross product.

    Returns:
        ``9.0`` -- six multiplies and three subtractions.
    """
    return 9.0
