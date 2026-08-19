"""The ``init`` source node."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from tasks.dtypes import (
    DISTRIBUTIONS,
    DTYPES,
    MAX_RANK,
    UINT64_MAX,
    Distribution,
    DType,
    JsonDict,
    OpName,
    Shape,
)
from tasks.exceptions import UninitializedNodeError
from tasks.node import Node, peek_provisional_id
from tasks.shapes import flops_init


class InitNode(Node):
    """A source node holding a PRNG-generated tensor.

    ``shape`` and ``seed`` both default to ``None`` on purpose: it makes ``InitNode()`` and
    ``InitNode((4, 4))`` raise :class:`~tasks.exceptions.UninitializedNodeError` rather than
    ``TypeError``, which is precisely the behaviour the exception protocol specifies for a
    missing seed or shape definition.
    """

    OP: ClassVar[OpName] = "init"

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        *,
        seed: int | None = None,
        dtype: DType = "float64",
        distribution: Distribution = "uniform",
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        """Declare a randomly initialized source tensor.

        Validation runs in a fixed order -- shape presence, rank, extents, seed presence,
        seed range, then enums -- so error messages are deterministic and testable.

        Args:
            shape: Tensor extents; every entry must be a positive integer. Rank 0 to 8. Rank 0
                (``()``) is legal as of v1.2.0: the elementwise ``multiply`` and ``mod`` ops
                consume rank-0 operands, and the composite expansions need a rank-0 constant
                for terms like ``cos(u @ v)``'s ``x**0``.
            seed: PRNG seed in ``[0, 2**64)``. Required even for the ``zeros`` and ``ones``
                distributions, which ignore it, so the engine needs no conditional logic.
            dtype: Element type of the buffer.
            distribution: PRNG distribution used to fill the tensor.
            name: Optional explicit node ID.
            label: Optional free-form annotation emitted as the schema's ``label``.

        Raises:
            UninitializedNodeError: If ``shape`` or ``seed`` is missing or invalid.
            ValueError: If ``dtype``, ``distribution``, or ``name`` is not a legal value.
        """
        who = f"InitNode '{peek_provisional_id(InitNode.OP, name)}'"

        if shape is None:
            raise UninitializedNodeError(f"{who}: shape is required for init nodes, got None")
        extents = tuple(shape)
        if len(extents) > MAX_RANK:
            raise UninitializedNodeError(
                f"{who}: shape must have rank 0 to {MAX_RANK}, got {extents}"
            )
        for extent in extents:
            # bool is a subclass of int, and mypy cannot catch seed=True from an untyped
            # call site such as parsed config, so reject it explicitly.
            if isinstance(extent, bool) or not isinstance(extent, int) or extent < 1:
                raise UninitializedNodeError(
                    f"{who}: shape extents must be positive integers, got {extents}"
                )
        if seed is None:
            raise UninitializedNodeError(f"{who}: seed is required for init nodes, got None")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= UINT64_MAX:
            raise UninitializedNodeError(
                f"{who}: seed must be an integer in [0, 2**64), got {seed!r}"
            )
        if dtype not in DTYPES:
            raise ValueError(f"{who}: dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
        if distribution not in DISTRIBUTIONS:
            raise ValueError(
                f"{who}: distribution must be one of {sorted(DISTRIBUTIONS)}, got {distribution!r}"
            )

        self._shape: Shape = extents
        self._seed = seed
        self._distribution: Distribution = distribution
        super().__init__((), extents, dtype, name=name, label=label)

    @property
    def seed(self) -> int:
        """PRNG seed used to fill this tensor."""
        return self._seed

    @property
    def distribution(self) -> Distribution:
        """PRNG distribution used to fill this tensor."""
        return self._distribution

    def _infer(self, inputs: tuple[Node, ...]) -> tuple[Shape, DType]:
        """Return this node's declared shape and dtype.

        An init node has no operands, so inference is a lookup and ``rewire`` can never
        reach it.

        Args:
            inputs: Ignored; always empty for a source node.

        Returns:
            The declared ``(shape, dtype)`` pair.
        """
        return self._shape, self._dtype

    def _payload(self) -> JsonDict:
        """Return the init-specific schema fields.

        Returns:
            The ``seed``, ``shape``, and ``distribution`` fields the schema requires on
            ``init``. ``shape`` is a list, not a tuple, so ``to_dict`` output compares
            directly against parsed JSON in tests.
        """
        return {
            "seed": self._seed,
            "shape": list(self._shape),
            "distribution": self._distribution,
        }

    def est_flops(self) -> float:
        """Return one PRNG draw per element.

        Returns:
            The tensor's element count.
        """
        return flops_init(self._shape)
