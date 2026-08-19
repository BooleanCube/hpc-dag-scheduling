"""DAG math builder and node operations for the HPC DAG scheduling baseline.

The builder is lazily evaluated in the Polars sense: graph construction records *intent* and
performs no arithmetic. Validation, however, is eager -- every logical error is raised at the
moment the offending expression is written, so ``a + b`` with mismatched shapes produces a
traceback pointing at that line rather than a surprise at serialization time.

The one exception is :class:`CyclicDependencyError`, which is a whole-graph property and
cannot be known until the graph is closed with a :class:`Graph`.

Example:
    >>> a = InitNode((64, 32), seed=42, distribution="normal", name="lhs")
    >>> b = InitNode((32, 16), seed=43)
    >>> graph = Graph([(a @ b) * 0.5], dag_id="bench-matmul-001")
    >>> graph.serialize(include_timestamp=False)["outputs"]
    ['scale_3']
"""

from tasks.dtypes import Distribution, DType, OpName, Shape
from tasks.exceptions import (
    CyclicDependencyError,
    DagBuildError,
    DimensionalityError,
    ShapeMismatchError,
    UninitializedNodeError,
)
from tasks.graph import SCHEMA_VERSION, Graph
from tasks.math import cos, cosh, exp, matpow, powmod, sin, sinh
from tasks.node import Node
from tasks.ops import (
    AddNode,
    CrossProductNode,
    DotProductNode,
    InitNode,
    ModNode,
    MultiplyNode,
    ScaleNode,
)


def cross(left: Node, right: Node, *, name: str | None = None) -> Node:
    """Return the cross product of two length-3 vector nodes.

    A free-function alias for :meth:`Node.cross`, for call sites that read better in prefix
    form.

    Args:
        left: Left operand, a length-3 rank-1 vector node.
        right: Right operand, a length-3 rank-1 vector node.
        name: Optional explicit node ID for the resulting node.

    Returns:
        A ``cross_product`` node of shape ``(3,)``.

    Raises:
        DimensionalityError: If either operand is not a length-3 rank-1 vector.
        ShapeMismatchError: If both operands are rank-1 but of different lengths.
    """
    return CrossProductNode(left, right, name=name)


__all__ = [
    "SCHEMA_VERSION",
    "AddNode",
    "CrossProductNode",
    "CyclicDependencyError",
    "DType",
    "DagBuildError",
    "DimensionalityError",
    "Distribution",
    "DotProductNode",
    "Graph",
    "InitNode",
    "ModNode",
    "MultiplyNode",
    "Node",
    "OpName",
    "ScaleNode",
    "Shape",
    "ShapeMismatchError",
    "UninitializedNodeError",
    "cos",
    "cosh",
    "cross",
    "exp",
    "matpow",
    "powmod",
    "sin",
    "sinh",
]
