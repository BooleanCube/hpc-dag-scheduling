"""Tier-2 composites: ordinary Python functions that build subgraphs of primitives.

Nothing here reaches the wire. A composite emits only the seven primitive ops, so adding one costs
no C++ work, and everything the primitives guarantee still holds: shape inference runs, exceptions
fire eagerly, and :meth:`tasks.graph.Graph.serialize` sorts and validates the result like any other
graph.

Composites are free functions rather than :class:`~tasks.node.Node` methods for one reason that
matters: **a method implies atomicity.** ``x.cross(y)`` is one node, and every other ``Node`` method
is O(1) in nodes. Putting a thirty-node expansion behind the same syntax would make the two
indistinguishable at the call site, which is intolerable when DAG topology is the object of study.
``__pow__`` and ``__mod__`` are the two sanctioned operator exceptions.

Two policies worth knowing before reading further:

**No common-subexpression elimination across calls.** ``sin(x) + cos(x)`` emits two separate
``x**2`` nodes. Within a call, powers are aggressively reused; across calls, nothing is shared. A
CSE pass
belongs on ``Graph``, where it can see the whole DAG, and redundant subtrees are realistic scheduler
input anyway.

**Unit coefficients are not optimized away.** ``sin(x, terms=1)`` still emits a ``scale`` by exactly
1.0. Skipping it would make node count depend on the *values* of coefficients rather than on the
parameters, and a topology that changes on numeric coincidence is a poor experimental subject. The
node-count formulas below are exact because of this.

**Accuracy: the series composites have no range reduction.** Maclaurin series are accurate only for
small ``|x|``; argument reduction would need ``floor`` and division primitives that deliberately do
not exist. Truncation error is bounded by the first omitted term, ``|x|**P / P!``. At
``terms=8`` and ``|x| = 1`` that is about 3e-15 for ``sin``, but at ``|x| = 10`` it is 2.8e+02 --
the result is not
merely inaccurate, it is meaningless. These DAGs are workload generators for a scheduling study
first and a numerics library second.
"""

import math as _math
from collections.abc import Callable, Sequence
from typing import Literal

from tasks.dtypes import DType
from tasks.exceptions import DimensionalityError, ShapeMismatchError
from tasks.node import Node
from tasks.ops.arithmetic import AddNode, ModNode, MultiplyNode, ScaleNode
from tasks.ops.init_op import InitNode
from tasks.ops.products import DotProductNode

__all__ = [
    "cos",
    "cosh",
    "exp",
    "matpow",
    "multiplies",
    "pow",
    "powmod",
    "sin",
    "sinh",
]

Parity = Literal["odd", "even", "all"]
"""Which power schedule a series uses: ``2k+1``, ``2k``, or ``k``."""

EXACT_INTEGER_BITS: dict[str, int] = {"float64": 53, "float32": 24}
"""Mantissa width: the exponent of the largest exactly-representable integer, per dtype."""

Combine = Callable[[Node, Node, str], Node]
"""Binary step used by the exponentiation walk, taking two operands and a label."""


def safe_modulus_limit(dtype: DType) -> int:
    """Return the largest modulus for which :func:`powmod` stays exact.

    Because the modulus is applied after every multiply, intermediates are bounded by
    ``(m-1)**2``, so exactness needs ``(m-1)**2 <= 2**mantissa_bits``.

    Args:
        dtype: Element type the expansion will run in.

    Returns:
        94906266 for ``float64`` and 4097 for ``float32``.
    """
    return _math.isqrt(2 ** EXACT_INTEGER_BITS[dtype]) + 1


def multiplies(n: int) -> int:
    """Return the number of binary-exponentiation multiplies needed for exponent ``n``.

    Args:
        n: Non-negative exponent.

    Returns:
        ``floor(log2 n) + popcount(n) - 1`` for ``n >= 1``, and 0 for ``n == 0``. This is why
        ``pow(x, 1024)`` costs ten nodes rather than 1023.
    """
    if n <= 0:
        return 0
    return (n.bit_length() - 1) + bin(n).count("1") - 1


def _validate_terms(terms: int) -> None:
    """Check that a series term count is a usable positive integer.

    Args:
        terms: Requested number of series terms.

    Raises:
        ValueError: If ``terms`` is a ``bool``, is not an ``int``, or is less than 1.
    """
    # bool first: it is an int subclass, so sin(x, terms=True) would otherwise mean terms=1.
    if isinstance(terms, bool):
        raise ValueError(f"terms must be an int >= 1, got {terms!r}")
    if not isinstance(terms, int):
        raise ValueError(f"terms must be an int >= 1, got {type(terms).__name__}")
    if terms < 1:
        raise ValueError(f"terms must be an int >= 1, got {terms}")


def _validate_exponent(n: int) -> None:
    """Check that an exponent is a non-negative integer.

    Args:
        n: Requested exponent.

    Raises:
        ValueError: If ``n`` is a ``bool``, is not an ``int``, or is negative. Negative exponents
            would need a division primitive (or a matrix inverse), neither of which exists.
    """
    if isinstance(n, bool):
        raise ValueError(f"exponent must be an int >= 0, got {n!r}")
    if not isinstance(n, int):
        raise ValueError(f"exponent must be an int >= 0, got {type(n).__name__}")
    if n < 0:
        raise ValueError(
            f"exponent must be an int >= 0, got {n}; negative exponents need a division "
            "primitive, which is deliberately out of scope"
        )


def _coefficient(power: int, *, sign: float, prefix: str, index: int) -> float:
    """Compute one series coefficient, rejecting terms too large to represent.

    ``1.0 / float(math.factorial(p))`` raises ``OverflowError`` rather than underflowing to zero
    once ``p >= 171``, because the exact integer factorial leaves the float64 range during
    conversion. The cap is computed here rather than hardcoded, so it stays correct if the power
    schedule ever changes.

    Args:
        power: Power whose factorial divides the term.
        sign: ``+1.0`` or ``-1.0``.
        prefix: Composite label prefix, for the error message.
        index: Term index, for the error message.

    Returns:
        ``sign / power!``.

    Raises:
        ValueError: If the coefficient is not representable as a float.
    """
    try:
        return sign / float(_math.factorial(power))
    except OverflowError as exc:
        raise ValueError(
            f"{prefix}: term {index} needs 1/{power}!, which is not representable as a float; "
            "reduce terms"
        ) from exc


def _ones_like(x: Node, *, label: str) -> Node:
    """Build a constant-ones tensor shaped like ``x``.

    ``init`` with ``distribution="ones"`` is the contract's constant-tensor mechanism; there is no
    dedicated ``constant`` op. The seed is required even though ``ones`` ignores it, so the engine
    needs no conditional logic.

    Args:
        x: Node whose shape and dtype the constant matches.
        label: Label for the emitted node.

    Returns:
        An ``init`` node of ones.
    """
    return InitNode(x.output_shape, seed=0, dtype=x.dtype, distribution="ones", label=label)


def _sum_tree(nodes: Sequence[Node], *, prefix: str) -> Node:
    """Sum nodes with a balanced pairwise tree, left to right.

    A balanced tree rather than a left-to-right chain: identical node count, but logarithmic depth,
    which exposes the parallelism a scheduler exists to exploit. Chaining would make every series a
    serial dependency. The pairing order is specified exactly because it fixes both the node count
    and the floating-point summation order, which is what makes the numeric tests reproducible.

    Args:
        nodes: Terms to sum; must be non-empty.
        prefix: Label prefix for emitted ``add`` nodes.

    Returns:
        The node holding the total. A single input is returned unchanged, emitting no nodes.
    """
    current = list(nodes)
    level = 0
    while len(current) > 1:
        nxt: list[Node] = []
        for index in range(0, len(current) - 1, 2):
            nxt.append(
                AddNode(
                    current[index],
                    current[index + 1],
                    label=f"{prefix}/sum{level}_{index // 2}",
                )
            )
        if len(current) % 2:
            nxt.append(current[-1])
        current = nxt
        level += 1
    return current[0]


def _maclaurin(x: Node, terms: int, *, parity: Parity, alternate: bool, prefix: str) -> Node:
    """Expand a Maclaurin series over elementwise primitives.

    The power cache is the whole trick. For odd/even parity it emits ``x2 = x * x`` once and steps
    powers by two, so ``x**3 = x**1 * x**2`` and ``x**5 = x**3 * x**2``: **one multiply per term**
    rather than one per unit of exponent. For ``exp`` it steps by one, which is optimal there
    because every intermediate power is itself a needed term.

    Args:
        x: Operand node, any rank from 0 to 8.
        terms: Number of series terms, already validated.
        parity: ``"odd"`` for powers ``2k+1``, ``"even"`` for ``2k``, ``"all"`` for ``k``.
        alternate: Whether to apply the ``(-1)**k`` sign factor.
        prefix: Label prefix for every emitted node.

    Returns:
        The node holding the summed series.

    Raises:
        ValueError: If a coefficient is not representable as a float.
    """
    if parity == "odd":
        powers = [2 * k + 1 for k in range(terms)]
    elif parity == "even":
        powers = [2 * k for k in range(terms)]
    else:
        powers = list(range(terms))

    step = 1 if parity == "all" else 2
    cache: dict[int, Node] = {1: x}
    if 0 in powers:
        cache[0] = _ones_like(x, label=f"{prefix}/ones")
    if step == 2 and max(powers) >= 2:
        cache[2] = MultiplyNode(x, x, label=f"{prefix}/pow2")
    for power in sorted(set(powers)):
        if power in cache:
            continue
        cache[power] = MultiplyNode(cache[power - step], cache[step], label=f"{prefix}/pow{power}")

    scaled = [
        ScaleNode(
            cache[power],
            _coefficient(
                power,
                sign=(-1.0) ** index if alternate else 1.0,
                prefix=prefix,
                index=index,
            ),
            label=f"{prefix}/coeff{index}",
        )
        for index, power in enumerate(powers)
    ]
    return _sum_tree(scaled, prefix=prefix)


def sin(x: Node, *, terms: int = 8, label_prefix: str = "sin") -> Node:
    """Expand ``sin(x)`` as a Maclaurin series of elementwise primitive ops.

    Computes the sum over ``k`` in ``[0, terms)`` of ``(-1)**k * x**(2k+1) / (2k+1)!``, reusing
    ``x**2`` so the odd powers cost one multiply each rather than one per unit of exponent. Emits
    ``3 * terms - 1`` nodes, or 1 node when ``terms == 1``.

    There is **no range reduction**: accuracy degrades as ``|x|`` grows, and past roughly
    ``|x| = 10`` the result is meaningless. See the module docstring.

    Args:
        x: Operand node. Any rank, including rank 0; the expansion is elementwise.
        terms: Number of series terms. Must be an int >= 1. Capped at 85 by float range.
        label_prefix: Prefix for the ``label`` of every emitted node, for traceability.

    Returns:
        The node holding the summed series.

    Raises:
        ValueError: If ``terms`` is not an int >= 1, or is so large that a coefficient is not
            representable as a float.
    """
    _validate_terms(terms)
    return _maclaurin(x, terms, parity="odd", alternate=True, prefix=label_prefix)


def sinh(x: Node, *, terms: int = 8, label_prefix: str = "sinh") -> Node:
    """Expand ``sinh(x)`` as a Maclaurin series of elementwise primitive ops.

    Identical to :func:`sin` but without the alternating sign: the sum over ``k`` of
    ``x**(2k+1) / (2k+1)!``. Emits ``3 * terms - 1`` nodes, or 1 when ``terms == 1``.

    Args:
        x: Operand node. Any rank, including rank 0.
        terms: Number of series terms. Must be an int >= 1. Capped at 85 by float range.
        label_prefix: Prefix for the ``label`` of every emitted node.

    Returns:
        The node holding the summed series.

    Raises:
        ValueError: If ``terms`` is invalid or a coefficient is unrepresentable.
    """
    _validate_terms(terms)
    return _maclaurin(x, terms, parity="odd", alternate=False, prefix=label_prefix)


def cos(x: Node, *, terms: int = 8, label_prefix: str = "cos") -> Node:
    """Expand ``cos(x)`` as a Maclaurin series of elementwise primitive ops.

    Computes the sum over ``k`` of ``(-1)**k * x**(2k) / (2k)!``. The ``k = 0`` term is ``x**0``,
    emitted as an ``init``/``ones`` node shaped like ``x``. Emits ``3 * terms - 1`` nodes, or 2
    when ``terms == 1``.

    **At ``terms == 1`` the result does not depend on ``x`` at all.** The only term is ``x**0``,
    which is the constant 1, so the expansion is a ``ones`` node and a ``scale`` -- and if ``x``
    has no other consumer, :class:`~tasks.graph.Graph`'s reachability walk drops it, yielding a
    DAG that never mentions ``x``. This is the same surprise :func:`pow` has at ``n == 0``, for
    the same reason, and it applies equally to :func:`cosh` and :func:`exp`.

    There is **no range reduction**; see the module docstring.

    Args:
        x: Operand node. Any rank, including rank 0.
        terms: Number of series terms. Must be an int >= 1. Capped at 86 by float range.
        label_prefix: Prefix for the ``label`` of every emitted node.

    Returns:
        The node holding the summed series.

    Raises:
        ValueError: If ``terms`` is invalid or a coefficient is unrepresentable.
    """
    _validate_terms(terms)
    return _maclaurin(x, terms, parity="even", alternate=True, prefix=label_prefix)


def cosh(x: Node, *, terms: int = 8, label_prefix: str = "cosh") -> Node:
    """Expand ``cosh(x)`` as a Maclaurin series of elementwise primitive ops.

    Identical to :func:`cos` but without the alternating sign. Emits ``3 * terms - 1`` nodes, or 2
    when ``terms == 1``.

    **At ``terms == 1`` the result does not depend on ``x`` at all** -- the sole term is the
    constant ``x**0``, so ``x`` can be dropped as unreachable. See :func:`cos`.

    Args:
        x: Operand node. Any rank, including rank 0.
        terms: Number of series terms. Must be an int >= 1. Capped at 86 by float range.
        label_prefix: Prefix for the ``label`` of every emitted node.

    Returns:
        The node holding the summed series.

    Raises:
        ValueError: If ``terms`` is invalid or a coefficient is unrepresentable.
    """
    _validate_terms(terms)
    return _maclaurin(x, terms, parity="even", alternate=False, prefix=label_prefix)


def exp(x: Node, *, terms: int = 8, label_prefix: str = "exp") -> Node:
    """Expand ``exp(x)`` as a Maclaurin series of elementwise primitive ops.

    Computes the sum over ``k`` of ``x**k / k!``, stepping powers by one because every
    intermediate power is itself a needed term. Emits ``3 * terms - 2`` nodes, or 2 when
    ``terms == 1``.

    **At ``terms == 1`` the result does not depend on ``x`` at all** -- the sole term is the
    constant ``x**0``, so ``x`` can be dropped as unreachable. See :func:`cos`.

    ``exp`` converges markedly more slowly than :func:`sin` or :func:`cos` at equal ``terms``,
    because its first omitted power is ``terms`` rather than roughly ``2 * terms``. At
    ``terms=8`` and ``|x| = 1`` the truncation error is about 2.5e-05, not 3e-15. Ask for more
    terms.

    Args:
        x: Operand node. Any rank, including rank 0.
        terms: Number of series terms. Must be an int >= 1. Capped at 171 by float range.
        label_prefix: Prefix for the ``label`` of every emitted node.

    Returns:
        The node holding the summed series.

    Raises:
        ValueError: If ``terms`` is invalid or a coefficient is unrepresentable.
    """
    _validate_terms(terms)
    return _maclaurin(x, terms, parity="all", alternate=False, prefix=label_prefix)


def _binexp(base: Node, n: int, *, combine: Combine, prefix: str) -> Node:
    """Walk the bits of ``n`` from the most significant down, skipping the leading 1.

    Left-to-right rather than right-to-left, chosen for simplicity. Right-to-left would let the
    squaring chain and the accumulation partially overlap, giving slightly more parallelism at
    identical node count -- worth adding as a second strategy if the research wants to vary that
    axis.

    Args:
        base: Node the accumulator starts from.
        n: Exponent, at least 1.
        combine: Binary step, called with ``(left, right, label)``.
        prefix: Label prefix for emitted nodes.

    Returns:
        The accumulator node. For ``n == 1`` this is ``base`` itself, emitting nothing.
    """
    acc = base
    for index, bit in enumerate(bin(n)[3:]):
        acc = combine(acc, acc, f"{prefix}/sq{index}")
        if bit == "1":
            acc = combine(acc, base, f"{prefix}/mul{index}")
    return acc


def pow(x: Node, n: int, *, label_prefix: str = "pow") -> Node:  # noqa: A001
    """Raise a tensor to a non-negative integer power by binary exponentiation.

    Emits exactly ``multiplies(n)`` ``multiply`` nodes -- ``pow(x, 1024)`` is **10** nodes, not
    1023 -- in a pure serial chain, so depth equals node count. That makes this the narrow-and-deep
    counterpart to the wide-and-shallow series expansions, which is a contrast a scheduling
    baseline should contain.

    Two edge cases behave notably:

    * ``n == 1`` returns ``x`` **itself**, emitting no nodes. ``multiplies(1) == 0`` already
      predicts this, and an identity ``scale`` would contradict the formula.
    * ``n == 0`` returns a constant-ones tensor that **does not depend on ``x`` at all**. That is
      mathematically right (``x**0 == 1``, matching NumPy's ``0**0 == 1``) but genuinely
      surprising: if ``x`` has no other consumer, ``Graph``'s reachability walk drops it, so a
      function of ``x`` yields a DAG that never mentions ``x``. It is supported rather than
      rejected so that a parameter sweep over ``n = 0..10`` does not crash on its first iteration.

    Shadowing note: this is deliberately **not** re-exported as ``tasks.pow``, since
    ``from tasks import pow`` would shadow the builtin in the caller's namespace. Reach it as
    ``tasks.math.pow(x, n)`` or as ``x ** n``.

    Args:
        x: Base node, any shape.
        n: Non-negative integer exponent.
        label_prefix: Prefix for the ``label`` of every emitted node.

    Returns:
        The node holding ``x ** n``.

    Raises:
        ValueError: If ``n`` is negative or not an ``int``.
    """
    _validate_exponent(n)
    if n == 0:
        return _ones_like(x, label=f"{label_prefix}/ones")
    return _binexp(
        x,
        n,
        combine=lambda left, right, label: MultiplyNode(left, right, label=label),
        prefix=label_prefix,
    )


def powmod(
    x: Node,
    n: int,
    m: float,
    *,
    allow_inexact: bool = False,
    label_prefix: str = "powmod",
) -> Node:
    """Raise a tensor to a power modulo a scalar, reducing after every multiply.

    Emits ``2 * multiplies(n) + 1`` nodes: one ``mod`` on the base, then a ``multiply`` and a
    ``mod`` per exponentiation step. Reducing after *every* multiply is what keeps intermediates
    bounded, which is what makes the exactness bound below checkable.

    **This is float modular arithmetic, not integer modpow.** ``mod`` is exact only while operands
    remain exactly-representable integers. Intermediates are bounded by ``(m-1)**2``, so the
    expansion agrees with Python's ``pow(a, n, m)`` exactly while
    ``(m - 1)**2 <= 2**53`` (``float64``) or ``2**24`` (``float32``) -- that is, ``m <= 94906266``
    or ``m <= 4097``. Outside the bound results silently drift; the honest fix is an integer dtype
    in the contract, which is a much larger change than adding an op.

    Args:
        x: Base node, any shape.
        n: Non-negative integer exponent.
        m: Positive integral modulus.
        allow_inexact: Skip the exactness bound, for callers who genuinely want float remainder
            arithmetic outside it.
        label_prefix: Prefix for the ``label`` of every emitted node.

    Returns:
        The node holding ``x ** n mod m``.

    Raises:
        TypeError: If ``m`` is not an ``int`` or ``float``, or is a ``bool``.
        ValueError: If ``n`` is negative, or ``m`` is non-positive, non-integral, non-finite, or
            exceeds the exactness bound while ``allow_inexact`` is false.
    """
    _validate_exponent(n)
    modulus = _validate_modulus(m, dtype=x.dtype, allow_inexact=allow_inexact)

    if n == 0:
        ones = _ones_like(x, label=f"{label_prefix}/ones")
        return ModNode(ones, modulus, label=f"{label_prefix}/ones_mod")

    def combine(left: Node, right: Node, label: str) -> Node:
        """Multiply then immediately reduce, so intermediates stay bounded.

        Args:
            left: Left operand.
            right: Right operand.
            label: Label for the multiply; the reduction appends a suffix.

        Returns:
            The reduced product node.
        """
        product = MultiplyNode(left, right, label=label)
        return ModNode(product, modulus, label=f"{label}_mod")

    base = ModNode(x, modulus, label=f"{label_prefix}/base")
    return _binexp(base, n, combine=combine, prefix=label_prefix)


def _validate_modulus(m: float, *, dtype: DType, allow_inexact: bool) -> float:
    """Check a modulus for positivity, integrality, and float exactness.

    Args:
        m: Requested modulus.
        dtype: Element type the expansion will run in.
        allow_inexact: Skip the exactness bound.

    Returns:
        The modulus as a float.

    Raises:
        TypeError: If ``m`` is not an ``int`` or ``float``, or is a ``bool``.
        ValueError: If ``m`` is non-finite, non-positive, non-integral, or exceeds the bound.
    """
    if isinstance(m, bool) or not isinstance(m, int | float):
        raise TypeError(f"modulus must be an int or float, got {type(m).__name__}")
    if not _math.isfinite(m):
        raise ValueError(f"modulus must be finite, got {m!r}")
    if m <= 0:
        raise ValueError(f"modulus must be strictly positive, got {m!r}")
    if float(m) != int(m):
        raise ValueError(
            f"modulus must be integral for modular arithmetic to be meaningful, got {m!r}"
        )
    if not allow_inexact:
        limit = safe_modulus_limit(dtype)
        if int(m) > limit:
            raise ValueError(
                f"modulus {int(m)} exceeds the largest value exact in {dtype} ({limit}): "
                f"intermediates reach (m-1)**2, which would leave the exactly-representable "
                f"integer range. Pass allow_inexact=True to proceed anyway."
            )
    return float(m)


def matpow(a: Node, n: int, *, label_prefix: str = "matpow") -> Node:
    """Raise a square matrix to a positive integer power by binary exponentiation.

    Emits exactly ``multiplies(n)`` ``dot_product`` nodes -- ``matpow(A, 64)`` is **6** -- in a
    pure serial chain.

    ``n == 0`` is **rejected**, unlike :func:`pow`. The identity matrix is not expressible: the
    ``init`` distribution enum is ``uniform | normal | zeros | ones``, with no ``eye``, and
    ``ones((n, n))`` is emphatically not the multiplicative identity. Adding ``"eye"`` would fix
    one degenerate case at the cost of another C++ code path, so it is recorded as a candidate for
    a future version rather than smuggled in.

    Args:
        a: Square rank-2 operand.
        n: Positive integer exponent.
        label_prefix: Prefix for the ``label`` of every emitted node.

    Returns:
        The node holding ``a ** n``.

    Raises:
        DimensionalityError: If ``a`` is not rank 2.
        ShapeMismatchError: If ``a`` is rank 2 but not square.
        ValueError: If ``n`` is negative, or is zero (no identity matrix is expressible).
    """
    # Shape first, and rank before squareness: a caller who passed a vector wants to hear "this
    # needs a matrix", not "this needs to be square". Same ordering rationale as cross_product.
    shape = a.output_shape
    if len(shape) != 2:
        raise DimensionalityError(
            f"matpow: operand must be a rank-2 matrix, got rank {len(shape)} "
            f"(node '{a.display_id}')"
        )
    if shape[0] != shape[1]:
        raise ShapeMismatchError(
            f"matpow: operand must be square, got {shape} (node '{a.display_id}')"
        )

    _validate_exponent(n)
    if n == 0:
        raise ValueError(
            "matpow: exponent 0 would need an identity matrix, which the init distribution enum "
            "cannot express (no 'eye'); ones((n, n)) is not the multiplicative identity"
        )
    return _binexp(
        a,
        n,
        combine=lambda left, right, label: DotProductNode(left, right, label=label),
        prefix=label_prefix,
    )
