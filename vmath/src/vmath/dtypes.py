"""Scalar type aliases and promotion rules for DAG tensors.

``Literal`` aliases are used rather than an ``Enum``: the values serialize directly to the
JSON strings the schema expects, ``mypy --strict`` checks them at call sites with no
``.value`` plumbing, and the C++ engine reads the same strings.
"""

from typing import Any, Literal

DType = Literal["float32", "float64"]
Distribution = Literal["uniform", "normal", "zeros", "ones"]
OpName = Literal["init", "add", "multiply", "scale", "mod", "dot_product", "cross_product"]
Shape = tuple[int, ...]
JsonDict = dict[str, Any]

DTYPES: frozenset[str] = frozenset({"float32", "float64"})
DISTRIBUTIONS: frozenset[str] = frozenset({"uniform", "normal", "zeros", "ones"})
UINT64_MAX: int = 18_446_744_073_709_551_615
MAX_RANK: int = 8


def promote(left: DType, right: DType) -> DType:
    """Return the result dtype for a binary op over two operand dtypes.

    Follows NumPy's widening rule: mixing ``float32`` and ``float64`` yields ``float64``.
    Promotion is silent by design -- a mixed-dtype graph is legal maths, so it is not a
    ``DagBuildError``.

    Args:
        left: dtype of the first operand.
        right: dtype of the second operand.

    Returns:
        ``"float64"`` if either operand is ``float64``, otherwise ``"float32"``.
    """
    if left == "float64" or right == "float64":
        return "float64"
    return "float32"
