# `/tasks` DAG Builder — Interface Design

Implementation-ready design for the mathematical DAG builder. Target: Python 3.11, `mypy --strict`,
`ruff` (rules `E,F,I,N,UP,B,D`, line length 100, Google docstring convention).

The producing contract is [`/shared/dag_schema.json`](../shared/dag_schema.json) (JSON Schema draft
2020-12, `schema_version` `1.2.0`). Every statement below about field names, enums, and constraints
is taken from that file; where the two ever disagree, **the schema wins** and this document is the
bug.

**Two tiers of operation.** §2–§12 describe **primitives** — the seven ops in the schema's `op` enum,
each of which is exactly one serialized node and one C++ implementation the humans must write. §13–§16
describe **composites** — Python functions that expand at build time into subgraphs of primitives.
The engine never sees a composite. Growing the primitive set is expensive (C++ work in a directory
that is read-only to us); growing the composite set is nearly free. That asymmetry is why `sin` is
twenty-nine nodes of `multiply`/`scale`/`add` rather than an eighth `op`.

---

## 1. Guiding principles

**P1 — Eager validation, lazy evaluation.** Graph construction records *intent*; no arithmetic is
performed. But every *logical* error is raised at the moment the offending expression is written,
not at `serialize()`. A user who writes `a + b` with mismatched shapes gets a traceback pointing at
that line. The single exception is `CyclicDependencyError`, which is a whole-graph property and
cannot be known until the graph is closed.

**P2 — Two error families, cleanly separated.**

| Family | Base | Meaning |
| --- | --- | --- |
| Mathematical / graph-theoretic | `DagBuildError` | The DAG the user described is not well-formed maths. |
| API misuse | `ValueError` / `TypeError` | The user called the library wrong (bad enum, non-finite scalar, duplicate name). |

Never raise a `DagBuildError` for a plain API mistake, and never raise a bare `ValueError` for a
shape problem. The four `DagBuildError` subclasses are the contract with the C++ engine; keeping
them semantically pure is what lets the engine assume mathematical soundness.

**P3 — An operator lowers to its own standard definition, never to a different operation.**

> **Revised in v1.2.0.** P3 originally read "one operator, one node", which is why `a - b` raised
> `TypeError`. That rule does not survive contact with composites: `x ** 1024` is unambiguously
> exponentiation and unambiguously ten nodes, so a strict one-to-one rule would have to reject the
> most natural spelling of the feature we were asked to build. Keeping the old rule while adding
> `__pow__` would have been simply inconsistent.
>
> The replacement draws the line where the real hazard is. What matters is not how many nodes an
> operator produces but whether it produces *the operation the reader expects*. `a - b` expanding to
> `AddNode(a, ScaleNode(b, -1.0))` surprises nobody, because that is the definition of subtraction.
> `a * b` meaning a contraction *would* surprise, because `*` is elementwise everywhere else in
> Python numerics — and that objection is preserved verbatim below. So: operators may expand, every
> expansion has a documented node-count formula and a test asserting it (§14), and no operator is
> ever a synonym for a different operation.

**P4 — Nodes are values; the Graph is a view.** Nodes never hold a reference to a graph. A `Graph`
is constructed *from* its output nodes and discovers the rest by walking backwards through
`.inputs`. This is what makes bare operator overloading work: `a + b` has no graph to register with.

---

## 2. Module layout

```
tasks/src/tasks/
├── __init__.py        # public API surface (re-exports only)
├── exceptions.py      # ALREADY IMPLEMENTED — do not change
├── dtypes.py          # type aliases, dtype promotion
├── shapes.py          # pure shape-inference functions (op maths, no Node imports)
├── node.py            # Node ABC + all operator overloads
├── graph.py           # Graph, topological sort, serialization
├── math.py            # TIER 2: composite expansions (§13-§16)
└── ops/
    ├── __init__.py    # re-exports the seven concrete node classes
    ├── init_op.py     # InitNode
    ├── arithmetic.py  # AddNode, MultiplyNode, ScaleNode, ModNode
    └── products.py    # DotProductNode, CrossProductNode
```

`math.py` shadows the stdlib `math` module *name* but not the module: Python 3's absolute imports mean
`import math` inside any `tasks/*.py` still resolves to the standard library, and only
`from tasks import math` reaches ours. Ruff's builtin-module-shadowing rule (`A005`) is not in the
selected rule set, so this does not trip linting. Inside `math.py` itself, `pow` shadows the builtin
within that module only; the module needs `math.factorial` and `**`, not `builtins.pow`, so nothing
breaks — but do not add a call to bare `pow()` in that file later.

`shapes.py` holds the inference rules as free functions over plain tuples, with no dependency on
`Node`. That keeps the mathematically interesting logic unit-testable without constructing graphs,
and it is where the bulk of the `ShapeMismatchError` / `DimensionalityError` test suite should aim.

### Circular imports

`node.py` defines `Node.__add__`, which must construct an `AddNode` from `ops/arithmetic.py`, which
imports `Node` from `node.py`. Resolve this with **function-scoped imports inside the dunder
methods** — not with `TYPE_CHECKING` guards, since the constructor call is needed at runtime:

```python
def __add__(self, other: Node) -> Node:
    from tasks.ops.arithmetic import AddNode  # noqa: PLC0415  (breaks import cycle)

    if not isinstance(other, Node):
        return NotImplemented
    return AddNode(self, other)
```

`PLC0415` is not in the selected ruff rule set, so the `noqa` is optional; keep the trailing comment
either way so the next reader knows the import placement is deliberate.

---

## 3. `dtypes.py`

```python
"""Scalar type aliases and promotion rules for DAG tensors."""

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

    Args:
        left: dtype of the first operand.
        right: dtype of the second operand.

    Returns:
        ``"float64"`` if either operand is ``float64``, otherwise ``"float32"``.
    """
```

`Literal` aliases rather than an `Enum`: the values serialize directly to the JSON strings the schema
expects, `mypy --strict` checks them at call sites with no `.value` plumbing, and the C++ side reads
the same four/two strings.

> **Resolved (team-lead, 2026-08-16).** Mixed-dtype operands are silently promoted rather than
> rejected: NumPy-style widening, no fifth exception. The consideration against it was that
> promotion doubles a buffer's memory footprint, which is exactly the quantity this scheduling study
> measures; the mitigation is the explicit `dtype=` argument on `InitNode`, which keeps mixed graphs
> rare in practice. Do not add a `DTypeMismatchError`.

---

## 4. `shapes.py` — inference rules

Each function takes operand shapes, raises on invalid combinations, and returns the output shape.
`op_name` and the two node labels are passed in purely to build good error messages.

```python
def infer_add(a: Shape, b: Shape, *, where: str) -> Shape
def infer_multiply(a: Shape, b: Shape, *, where: str) -> Shape
def infer_scale(a: Shape, *, where: str) -> Shape
def infer_mod(a: Shape, *, where: str) -> Shape
def infer_dot(a: Shape, b: Shape, *, where: str) -> Shape
def infer_cross(a: Shape, b: Shape, *, where: str) -> Shape
```

`where` is the pre-rendered operand description (see §8), e.g. `"nodes 'init_0' and 'init_1'"`.

### `add` — elementwise sum

Shapes must be **exactly equal**, including rank. No broadcasting: broadcasting would make the
engine's buffer allocation and MPI decomposition depend on a rule the C++ side does not implement.

- `a != b` → `ShapeMismatchError`
- otherwise → `a`

### `multiply` — elementwise (Hadamard) product

Identical rule to `add`: shapes must be **exactly equal**, including rank, and the result is that
shape. No broadcasting, for the same reason. `infer_multiply` and `infer_add` differ only in the `op`
name they put in the error message, so implement both by delegating to one private helper rather than
copying the branch — a divergence between the two would be a silent correctness bug.

This is *not* a contraction. `dot_product` is the contraction; see §5 for why the two must never
share an operator.

### `scale` — scalar multiply

Shape and dtype are preserved unconditionally. `infer_scale` cannot fail; it exists for symmetry and
so the rank ≤ `MAX_RANK` invariant has one home.

### `mod` — elementwise remainder by a positive scalar

Shape and dtype are preserved unconditionally, exactly like `scale`. `infer_mod` cannot fail on
shapes; the modulus itself is validated in `ModNode.__init__` (§7).

**Semantics are floored, not truncated.** The result lies in `[0, modulus)`, matching Python's `%`
and `numpy.mod` — *not* C's `std::fmod`, which returns a value carrying the sign of the dividend.
Since the modulus is required to be positive, the engine's correction is one branch, and the schema's
`modulus` description spells it out for the C++ authors. Two reasons this was worth specifying rather
than defaulting to `fmod`: floored remainder is what modular arithmetic means, so `powmod` (§14) is
only meaningful under it; and it keeps `x % m` faithful to the NumPy operator it mirrors, which
truncated semantics would silently violate for negative inputs.

These are floating-point dtypes, so `mod` is exact only while operands stay within the
exactly-representable integer range — 2\*\*53 for `float64`, 2\*\*24 for `float32`. `powmod` checks
that bound; bare `mod` does not, because non-integer operands are a legitimate use.

### `dot_product` — contraction

Operand ranks 1 and 2 only. Let `rank_a = len(a)`, `rank_b = len(b)`.

| `rank_a` | `rank_b` | Requirement | Output |
| --- | --- | --- | --- |
| 2 `(n, m)` | 2 `(p, q)` | `m == p` | `(n, q)` |
| 1 `(m,)` | 2 `(p, q)` | `m == p` | `(q,)` — operand treated as a row vector |
| 2 `(n, m)` | 1 `(p,)` | `m == p` | `(n,)` — operand treated as a column vector |
| 1 `(m,)` | 1 `(p,)` | `m == p` | `()` — rank-0 scalar, see below |
| 0, or > 2 | any | — | `DimensionalityError` (rank-0 operands and batched matmul are out of scope) |

When both ranks are supported but the contraction dimension disagrees → `ShapeMismatchError`. This
applies to the vector · vector row too: `(3,) @ (4,)` is a `ShapeMismatchError`, not a
dimensionality problem.

**Vector · vector yields a rank-0 scalar**, serialized as `output_shape: []`. Schema 1.1.0 was
amended specifically to permit this: an empty extent array denotes one scalar element, and the C++
side needs no special case because element count is the product of the extents and the product of an
empty list is 1 (`numpy.zeros(()).size == 1` is the same rule). Rank-0 results compose normally —
`scale` preserves rank 0, and `add` accepts two rank-0 operands — but they cannot feed `dot_product`
or `cross_product`, both of which require rank ≥ 1.

As of v1.2.0 rank 0 is also legal as a *declared* shape on `InitNode` (§7); schema 1.1.0's rank-1
floor on init was lifted once `multiply` and `mod` gave rank-0 sources something to feed.

### `cross_product` — 3-space cross product

Checked in this exact order, so error messages stay predictable:

1. `len(a) != 1 or len(b) != 1` → `DimensionalityError` (wrong tensor rank — this is the canonical
   trigger, e.g. a 2D matrix)
2. `a != b` → `ShapeMismatchError` (both rank-1 but different lengths)
3. `a != (3,)` → `DimensionalityError` (the cross product is only defined in 3-space)
4. otherwise → `(3,)`

The rank check precedes the length check because rank is the coarser, more informative failure: a
user who passed a matrix wants to hear "this needs a vector", not "this needs length 3".

### FLOP estimates

Used to populate the optional `hints.est_flops` field the schema defines. These are cost proxies for
the scheduler, not exact counts.

| Op | Estimate |
| --- | --- |
| `init` | `prod(shape)` — one PRNG draw per element |
| `add` | `prod(shape)` |
| `multiply` | `prod(shape)` |
| `scale` | `prod(shape)` |
| `mod` | `2 * prod(shape)` — a division and a conditional add per element |
| `dot_product` | `2 * prod(output_shape) * contraction_dim` — covers all four supported rank combinations, including the rank-0 result, because `math.prod(()) == 1` makes the vector · vector case fall out as `2 * m` with no special casing |
| `cross_product` | `9.0` (6 multiplies, 3 subtractions) |

---

## 5. `node.py` — the `Node` ABC

```python
class Node(ABC):
    """Abstract base for every node in a mathematical DAG.

    A node is an immutable-by-convention description of one operation. Its output shape and
    dtype are resolved eagerly at construction, so an ill-formed expression fails at the line
    that wrote it rather than at serialization time.

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
    ) -> None: ...
```

### Required behaviour

- **Do not make `Node` an `eq=True` dataclass.** Nodes are compared and hashed by *identity*; the
  reachability walk and topological sort both key dictionaries on node objects. A dataclass-generated
  `__eq__` would make two structurally identical nodes collide and would set `__hash__ = None`. If
  you use `@dataclass`, it must be `@dataclass(eq=False)`. Plain classes are simpler here.
- `name`, when given, is validated against the schema's node-ID pattern
  `^[A-Za-z_][A-Za-z0-9_.-]{0,63}$` and raises `ValueError` on violation. It becomes the node's
  serialized `id` and is never renumbered.
- `_provisional_id` is assigned from a module-level `itertools.count()` as `f"{OP}_{n}"`. It is for
  debugging and error messages only; `Graph` assigns the canonical serialized ID (§6).
- `_seq` records construction order from the same counter and is the tie-break key that makes the
  topological sort deterministic.

### Properties

```python
@property
def inputs(self) -> tuple[Node, ...]: ...  # immutable view of the operand list
@property
def output_shape(self) -> Shape: ...
@property
def dtype(self) -> DType: ...
@property
def label(self) -> str | None: ...
@property
def op(self) -> OpName: ...  # returns type(self).OP
```

### Methods

```python
def est_flops(self) -> float:
    """Return the estimated floating-point operation count for this node."""


def rewire(self, index: int, new_input: Node) -> None:
    """Replace one operand in place, re-running this node's shape inference.

    The advanced escape hatch for programmatically generated graphs. Shape and dtype are
    re-derived and may raise, but cycles cannot be detected here because reachability is a
    whole-graph property -- ``Graph.serialize`` is what catches those.

    Args:
        index: Position in ``inputs`` to replace.
        new_input: Replacement operand node.

    Raises:
        IndexError: If ``index`` is out of range.
        ShapeMismatchError: If the new operand does not align with the remaining operands.
        DimensionalityError: If the new operand has the wrong rank for this operation.
    """


@abstractmethod
def _payload(self) -> JsonDict:
    """Return the op-specific schema fields for this node.

    Returns:
        ``{"seed", "shape", "distribution"}`` for init nodes, ``{"factor"}`` for scale nodes,
        and an empty mapping for every other op.
    """


def to_dict(self, node_id: str, input_ids: Sequence[str], *, include_hints: bool) -> JsonDict:
    """Render this node as a schema-conformant JSON object."""
```

`to_dict` builds the common fields, then merges `_payload()`. Two schema rules it must respect:

1. **Omit `inputs` entirely for `init` nodes.** The schema sets `"inputs": false` under the init
   branch, so even `"inputs": []` is rejected. Guard with `if input_ids:` — `init` is the only
   zero-input op.
2. `label` and `hints` are omitted when absent/disabled; the schema sets
   `additionalProperties: false` at every level, so never emit a key with a `None` value.

### Operator overloading contract

| Expression | Result | Nodes | Notes |
| --- | --- | --- | --- |
| `a + b` | `AddNode(a, b)` | 1 | `b` not a `Node` → `NotImplemented` (Python raises `TypeError`) |
| `a - b` | `AddNode(a, ScaleNode(b, -1.0))` | 2 | **new in v1.2.0**, see P3 revision |
| `a * b` (both nodes) | `MultiplyNode(a, b)` | 1 | **flipped in v1.2.0**, see below |
| `a * k`, `k * a` | `ScaleNode(a, float(k))` | 1 | `__mul__` / `__rmul__`, `k` is `int \| float` |
| `a @ b` | `DotProductNode(a, b)` | 1 | `__matmul__` |
| `-a` | `ScaleNode(a, -1.0)` | 1 | `__neg__` |
| `a / k` | `ScaleNode(a, 1.0 / k)` | 1 | `__truediv__`, scalar divisor only |
| `a / b` (both nodes) | `NotImplemented` | — | no division primitive exists; §15 |
| `a ** n` | `tasks.math.pow(a, n)` expansion | O(log n) | **new in v1.2.0**, `n` a non-negative `int` |
| `a % m` | `ModNode(a, float(m))` | 1 | **new in v1.2.0**, `m` a positive scalar |
| `a % b` (both nodes) | `NotImplemented` | — | modulus is a scalar field, not an operand |
| `a.cross(b)` | `CrossProductNode(a, b)` | 1 | also exported as `tasks.cross(a, b)` |

`__mul__` dispatches on the operand type: `Node` → `MultiplyNode`, `int`/`float` → `ScaleNode`,
anything else → `NotImplemented`. Reject `bool` explicitly before the `int` branch, or `a * True`
silently becomes a scale by 1.0.

**Why `@` for dot and `*` for elementwise.** PEP 465 added `@` to Python specifically as the
matrix-multiplication operator, and NumPy follows it: `@` contracts, `*` is elementwise. Reusing that
split means `(a @ b) * 0.5` reads the same here as in NumPy, with no project-specific convention to
memorize.

**Verdict on `a * b` between two nodes: flipped to `MultiplyNode`.** My original design raised
`TypeError` here, and I stand by the reasoning as it was written — but its premise no longer holds. It
rested on there being no elementwise product in the vocabulary, which left only two options: reinterpret
`*` as a contraction, or reject. Rejecting was right, because `a * b` and `a @ b` building the same node
would train users to write dimension-sensitive code with the dimension-insensitive operator.

v1.2.0 adds `multiply`, so there is now a third option that is strictly better than both: `*` means
elementwise, exactly as it does in NumPy, and `@` keeps the contraction to itself. The objection I
raised is fully preserved — `*` still never means a contraction — while the NumPy-parity principle
that motivated it now *requires* the flip rather than forbidding it. Continuing to raise `TypeError`
would leave the builder in the odd position of having a Hadamard primitive that no operator can reach,
which is the sort of inconsistency users file bugs about.

**Why `a - b` is now provided.** See the P3 revision. The one-node rule it was justified by is gone,
and subtraction-as-add-with-negation surprises nobody. Its two-node expansion is documented and tested
like any other composite. Note the asymmetry it creates in reading DAG sizes: `a - b` costs two nodes
where `a + b` costs one, so a topology comparison between the two is not apples to apples.

**Why cross still has no operator.** The notation is `×`, not a Python operator, and every available
symbol (`^`, `%`) would be cryptic — `%` especially now that it means `mod`. A named method reads
better than a puzzle.


---

## 6. `graph.py` — `Graph`

```python
SCHEMA_VERSION: Final[str] = "1.2.0"


class Graph:
    """A closed, serializable mathematical DAG.

    A graph is constructed from its output nodes and discovers every contributing node by
    walking backwards through operand references. Nodes that were built but do not reach an
    output are excluded from serialization -- dead-code elimination is intentional and lets a
    task script explore alternatives without polluting the emitted DAG.
    """

    def __init__(
        self,
        outputs: Sequence[Node],
        *,
        dag_id: str,
        description: str | None = None,
    ) -> None:
        """Close a graph over the given output nodes.

        Args:
            outputs: Nodes whose tensors the engine must materialize. Must be non-empty and
                free of duplicates.
            dag_id: Stable identifier matching ``^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$``.
            description: Optional human-readable description for the metadata block.

        Raises:
            ValueError: If ``outputs`` is empty or contains duplicates, or ``dag_id`` does not
                match the schema pattern.
        """
```

### Public API

```python
@property
def outputs(self) -> tuple[Node, ...]: ...


def nodes(self) -> tuple[Node, ...]:
    """Return every node reachable from the outputs, in construction order."""


def topological_order(self) -> list[Node]:
    """Return all reachable nodes ordered so every operand precedes its consumer.

    Raises:
        CyclicDependencyError: If the reachable subgraph contains a cycle.
    """


def validate(self) -> None:
    """Run every whole-graph check without producing output.

    Raises:
        CyclicDependencyError: If the graph contains a cycle.
    """


def serialize(
    self,
    *,
    renumber: bool = True,
    include_hints: bool = True,
    include_timestamp: bool = True,
) -> JsonDict:
    """Render the graph as a schema-conformant document.

    Args:
        renumber: Assign canonical ``{op}_{index}`` IDs in topological order. User-supplied
            names are always preserved. Disable to keep provisional construction-time IDs.
        include_hints: Emit ``hints.est_flops`` on every node.
        include_timestamp: Emit ``metadata.created_at``. Pass ``False`` for byte-stable output.

    Returns:
        A mapping with ``metadata``, ``nodes``, and ``outputs`` keys.

    Raises:
        CyclicDependencyError: If the graph contains a cycle.
    """


def to_json(self, path: Path, *, indent: int = 2, **kwargs: Any) -> None:
    """Write the serialized graph to disk as UTF-8 JSON."""
```

### Reachability walk — must be iterative

The walk from outputs back through `.inputs` runs **before** cycle detection, so it will encounter
cyclic graphs. Use an explicit stack and a `visited` set keyed on `id(node)` or on the node itself;
a recursive walk hits `RecursionError` on a cycle and the user never sees the
`CyclicDependencyError` we promised them. This is the single easiest thing to get wrong in this file.

### Topological sort

Kahn's algorithm over the reachable set, seeding the queue with in-degree-zero nodes in ascending
`_seq` (construction) order and using a `deque`. Deterministic ordering matters: it makes serialized
output diffable across runs, which is what makes the research baseline reproducible.

If the emitted count is less than the reachable count, a cycle exists. Before raising, recover an
actual cycle path with a DFS over the unemitted remainder so the message names the nodes involved —
"there is a cycle somewhere" is not an actionable error.

### ID assignment

With `renumber=True`, walk the topological order and assign:

- a user-supplied `name` → used verbatim;
- otherwise → `f"{node.op}_{index}"` where `index` is the position in topological order.

Detect collisions across both sources and raise `ValueError` naming the duplicate. Build a
`dict[Node, str]` mapping and pass the resolved operand IDs into each `to_dict` call. Because IDs
derive from topological position rather than construction history, two runs of the same script emit
byte-identical documents.

### Metadata block

```python
{
    "schema_version": SCHEMA_VERSION,  # "1.2.0"
    "dag_id": self._dag_id,
    "ordering": "topological",  # schema const, always this literal
    "created_at": datetime.now(UTC).isoformat(),  # omitted when include_timestamp=False
    "description": ...,  # omitted when None
    "generator": f"tasks-builder {version('tasks')}",  # importlib.metadata
}
```

`from datetime import UTC, datetime` is available on 3.11. Wrap `version("tasks")` in
`try/except PackageNotFoundError` and fall back to `"unknown"` so an uninstalled source checkout
still serializes.

### `to_json` must reject non-finite floats

Call `json.dump(..., allow_nan=False)`. Python's default emits bare `NaN` / `Infinity` literals,
which are invalid JSON and will fail the schema's `number` type on the C++ side. `ScaleNode` already
rejects non-finite factors at construction (§7), so this is defence in depth.

---

## 7. `ops/` — concrete nodes

### `InitNode` (`ops/init_op.py`)

```python
class InitNode(Node):
    """A source node holding a PRNG-generated tensor."""

    OP: ClassVar[OpName] = "init"

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        *,
        seed: int | None = None,
        dtype: DType = "float64",
        distribution: Distribution = "uniform",
        name: str | None = None,
    ) -> None:
        """Declare a randomly initialized source tensor.

        Args:
            shape: Tensor extents; every entry must be a positive integer.
            seed: PRNG seed in ``[0, 2**64)``. Required even for the ``zeros`` and ``ones``
                distributions, which ignore it, so the engine needs no conditional logic.
            dtype: Element type of the buffer.
            distribution: PRNG distribution used to fill the tensor.
            name: Optional explicit node ID.

        Raises:
            UninitializedNodeError: If ``shape`` or ``seed`` is missing or invalid.
            ValueError: If ``dtype``, ``distribution``, or ``name`` is not a legal value.
        """
```

`shape` and `seed` both default to `None` *on purpose*: it makes `InitNode()` and `InitNode((4, 4))`
raise `UninitializedNodeError` rather than `TypeError`, which is precisely the behaviour CLAUDE.md
specifies for a missing seed or shape definition.

Validation order — fixed, so tests are deterministic:

1. `shape is None` → `UninitializedNodeError`
2. *(removed in v1.2.0 — a rank-0 `shape` of `()` is now legal; see the note below)*
3. `len(shape) > MAX_RANK` → `UninitializedNodeError`
4. any extent not an `int`, or `< 1`, or a `bool` → `UninitializedNodeError`
5. `seed is None` → `UninitializedNodeError`
6. `seed` not an `int`, or a `bool`, or outside `[0, UINT64_MAX]` → `UninitializedNodeError`
7. `dtype`/`distribution` not in their frozensets → `ValueError`

**Init may be rank 0 as of v1.2.0. This reverses the v1.1.0 restriction.** The original argument was
that a rank-0 `init` would be a random scalar no op could consume — `add` requires identical shapes,
`dot_product` and `cross_product` require rank ≥ 1, and `scale` takes a literal factor rather than a
node — so it could only ever be a dead-end output. Both halves of that premise are now false. The
elementwise `multiply` and `mod` ops happily consume rank-0 operands, and the composite expansions
positively *need* a rank-0 constant: `cos(u @ v)` on two vectors reduces to a rank-0 value, and its
`x**0` term is an `init`/`ones` node that must match that shape. So the floor is gone, `shape` and
`output_shape` accept ranks 0 through 8 uniformly, and the schema no longer pins `minItems: 1` on the
init branch.

One consequence to be aware of, since it removes a check that used to exist incidentally: the init
branch's rank floor was the only structural reason a document with `shape: []` and a non-rank-0
`output_shape` got rejected. Equality between those two fields was never schema-enforced — it has
always been a producer guarantee, listed among the scrutiny items from the first version of this
document — so this loses no *intended* validation, but `InitNode` is now the sole thing standing
between a typo and an inconsistent document. Keep the assertion that emitted `output_shape` equals
`shape` in the test suite.

`bool` is a subclass of `int`, so `isinstance(True, int)` is `True`. Reject `bool` explicitly at
steps 4 and 6; `mypy --strict` will not catch `seed=True` from untyped call sites such as parsed
config.

`_payload()` returns `{"seed": self._seed, "shape": list(self._shape), "distribution": self._dist}`.
Note `list`, not `tuple` — `json` handles both, but keeping the conversion here makes `to_dict`'s
output directly comparable to parsed JSON in tests.

### `AddNode` (`ops/arithmetic.py`)

```python
class AddNode(Node):
    OP: ClassVar[OpName] = "add"

    def __init__(self, left: Node, right: Node, *, name: str | None = None) -> None: ...
```

Output shape from `infer_add`; dtype from `promote`. `_payload()` returns `{}`.

### `MultiplyNode` (`ops/arithmetic.py`)

```python
class MultiplyNode(Node):
    """Elementwise (Hadamard) product of two tensors of identical shape."""

    OP: ClassVar[OpName] = "multiply"

    def __init__(self, left: Node, right: Node, *, name: str | None = None) -> None: ...
```

Output shape from `infer_multiply`, dtype from `promote`, `_payload()` returns `{}`. Structurally
identical to `AddNode`; the only difference is `OP`.

### `ModNode` (`ops/arithmetic.py`)

```python
class ModNode(Node):
    OP: ClassVar[OpName] = "mod"

    def __init__(self, operand: Node, modulus: float, *, name: str | None = None) -> None:
        """Reduce a tensor elementwise to the non-negative remainder modulo a scalar.

        The result lies in ``[0, modulus)`` -- floored semantics as in Python's ``%`` and
        ``numpy.mod``, not C's ``std::fmod``. See the schema's ``modulus`` description.

        Args:
            operand: Tensor to reduce.
            modulus: Strictly positive scalar modulus.
            name: Optional explicit node ID.

        Raises:
            TypeError: If ``modulus`` is not an ``int`` or ``float``, or is a ``bool``.
            ValueError: If ``modulus`` is not strictly positive, or is NaN or infinite.
        """
```

Shape and dtype are inherited from the operand. Store `float(modulus)`. Validation mirrors
`ScaleNode`'s: reject `bool`, reject non-finite via `math.isfinite`, and additionally reject
`modulus <= 0` — the schema sets `exclusiveMinimum: 0`, so a non-positive modulus would produce a
document the engine rejects at parse time, and catching it here gives the user a line number instead.
`_payload()` returns `{"modulus": self._modulus}`.

### `ScaleNode` (`ops/arithmetic.py`)

```python
class ScaleNode(Node):
    OP: ClassVar[OpName] = "scale"

    def __init__(self, operand: Node, factor: float, *, name: str | None = None) -> None:
        """Scale a tensor by a scalar constant.

        Raises:
            TypeError: If ``factor`` is not an ``int`` or ``float``.
            ValueError: If ``factor`` is NaN or infinite and therefore not serializable.
        """
```

Shape and dtype are inherited from the operand — a `float32` tensor scaled by a Python float stays
`float32`. Store `float(factor)`. Reject non-finite values with `math.isfinite`: JSON has no NaN or
Infinity literal, so a non-finite factor produces a document the engine cannot parse. (NaN arising
*at runtime* is the engine's problem; a NaN *constant* is ours.) `_payload()` returns
`{"factor": self._factor}`.

### `DotProductNode`, `CrossProductNode` (`ops/products.py`)

```python
class DotProductNode(Node):
    OP: ClassVar[OpName] = "dot_product"

    def __init__(self, left: Node, right: Node, *, name: str | None = None) -> None: ...


class CrossProductNode(Node):
    OP: ClassVar[OpName] = "cross_product"

    def __init__(self, left: Node, right: Node, *, name: str | None = None) -> None: ...
```

Shapes from `infer_dot` / `infer_cross`, dtype from `promote`, `_payload()` returns `{}`. Operand
order is significant for both and must be preserved in the serialized `inputs` array.

---

## 8. Exception reference

`exceptions.py` is already implemented and needs no changes. The hierarchy is
`DagBuildError(Exception)` with `ShapeMismatchError`, `DimensionalityError`,
`CyclicDependencyError`, and `UninitializedNodeError` as direct subclasses. Keep them as plain
subclasses with no custom `__init__` — that keeps them trivially picklable across the `hpcctl`
boundary and keeps `mypy --strict` quiet.

### Message format

One convention everywhere: `"{op}: {problem}, got {actual} ({where})"`, where `where` names the
operands. Provide a helper in `shapes.py`:

```python
def describe(*nodes: str) -> str:
    """Render a parenthetical operand reference for an error message.

    Args:
        *nodes: Provisional or user-assigned node IDs, in operand order.

    Returns:
        For example ``"nodes 'init_0' and 'init_1'"``.
    """
```

### Where each exception fires

| Exception | Fires at | Trigger | Example message |
| --- | --- | --- | --- |
| `ShapeMismatchError` | `AddNode.__init__` via `infer_add` | operand shapes not identical | `add: operand shapes must match exactly, got (2, 2) and (3, 3) (nodes 'init_0' and 'init_1')` |
| `ShapeMismatchError` | `DotProductNode.__init__` via `infer_dot` | inner dimensions disagree (including vector · vector of unequal length) | `dot_product: inner dimensions must agree, got (4, 3) @ (5, 2), 3 != 5 (nodes 'init_0' and 'init_1')` |
| `ShapeMismatchError` | `CrossProductNode.__init__` via `infer_cross` | both rank-1, different lengths | `cross_product: operand shapes must match exactly, got (3,) and (4,) (nodes 'u' and 'v')` |
| `ShapeMismatchError` | `MultiplyNode.__init__` via `infer_multiply` | operand shapes not identical | `multiply: operand shapes must match exactly, got (2, 2) and (3, 3) (nodes 'init_0' and 'init_1')` |
| `ShapeMismatchError` | `tasks.math.matpow` | operand is rank-2 but not square | `matpow: operand must be a square matrix, got (4, 3) (node 'init_0')` |
| `DimensionalityError` | `tasks.math.matpow` | operand is not rank-2 | `matpow: operand must be rank-2, got rank 1 (node 'init_0')` |
| `ShapeMismatchError` | `Node.rewire` | replacement operand breaks alignment | as above, for the recomputed op |
| `DimensionalityError` | `CrossProductNode.__init__` | either operand is not rank-1 | `cross_product: operands must be rank-1 vectors, got rank 2 and rank 1 (nodes 'm' and 'v')` |
| `DimensionalityError` | `CrossProductNode.__init__` | rank-1 but not length 3 | `cross_product: only defined for length-3 vectors, got length 4 (nodes 'u' and 'v')` |
| `DimensionalityError` | `DotProductNode.__init__` | either operand rank 0 or > 2 | `dot_product: operands must be rank-1 or rank-2, got rank 3 and rank 2 (nodes 't' and 'm')` |
| `UninitializedNodeError` | `InitNode.__init__` | `shape` missing | `InitNode 'init_0': shape is required for init nodes, got None` |
| `UninitializedNodeError` | `InitNode.__init__` | shape rank > 8 (rank 0 became legal in v1.2.0) | `InitNode 'init_0': shape must have rank 0 to 8, got rank 9` |
| `UninitializedNodeError` | `InitNode.__init__` | non-positive / non-integer extent | `InitNode 'init_0': shape extents must be positive integers, got (4, 0)` |
| `UninitializedNodeError` | `InitNode.__init__` | `seed` missing | `InitNode 'init_0': seed is required for init nodes, got None` |
| `UninitializedNodeError` | `InitNode.__init__` | seed out of `[0, 2**64)` or not an int | `InitNode 'init_0': seed must be an integer in [0, 2**64), got -1` |
| `CyclicDependencyError` | `Graph.topological_order` (reached from `validate` and `serialize`) | reachable subgraph contains a cycle | `Cyclic dependency detected: add_3 -> scale_4 -> add_3` |

Everything mathematical fires at construction. `CyclicDependencyError` is the only member of the
family that fires at serialization, because acyclicity is the only property that is not local to a
single expression.

### v1.2.0 additions require NO new exception class

`exceptions.py` still needs no changes. Every new failure mode lands on either an existing
`DagBuildError` subclass or on `ValueError`/`TypeError`, following the P2 split. The mapping, so nobody
is tempted to invent a fifth:

| New failure | Exception | Why this one |
| --- | --- | --- |
| `matpow` on a non-square rank-2 matrix | `ShapeMismatchError` | Dimensions fail to align for the operation — the same condition `dot_product` raises it for, since `matpow` *is* repeated `dot_product`. |
| `matpow` on a non-rank-2 tensor | `DimensionalityError` | Wrong tensor rank. Checked before squareness, matching the `cross_product` ordering: rank is the coarser, more informative failure. |
| Negative exponent (`pow`, `powmod`, `matpow`) | `ValueError` | Not a shape or rank problem — the operation is simply unsupported because no division or matrix-inverse primitive exists. API misuse. |
| `terms < 1`, or non-`int` / `bool` `terms` | `ValueError` | API misuse. Nothing to do with tensor shape or with init nodes. |
| `terms` so large the series coefficient is unrepresentable | `ValueError` | API misuse, with a computed bound (§14). |
| Non-integral or out-of-range modulus in `powmod` | `ValueError` | API misuse. |
| `modulus <= 0`, NaN, or infinite | `ValueError` | API misuse; the schema would reject the document anyway. |

The one judgement call worth naming: a negative exponent could arguably be a `DimensionalityError`
("this operation is not defined here"), but that exception means *wrong tensor rank* specifically, and
stretching it to cover unsupported scalar arguments would blur the one distinction that makes the
four-exception contract legible to the engine authors. `ValueError` it is.

---

## 9. `__init__.py` public surface

Extend the existing re-exports; keep `__all__` sorted (ruff `I`/`F` will check usage, not order, but
consistency helps review):

```python
from tasks.dtypes import DType, Distribution, OpName, Shape
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
    """Return the cross product of two length-3 vector nodes."""
```

**`pow` is deliberately NOT re-exported at package top level.** `from tasks import pow` would shadow
the builtin in the user's namespace, which is a nasty thing for a library to do to a caller. It is
reachable as `tasks.math.pow(x, n)` — mirroring stdlib `math.pow`, where the same name coexists with
the builtin for exactly this reason — or as `x ** n`, which is the idiomatic spelling anyway. Every
other composite is safe to export and is exported.

---

## 10. Worked example

```python
from tasks import Graph, InitNode

a = InitNode((64, 32), seed=42, distribution="normal", name="lhs")
b = InitNode((32, 16), seed=43)
c = (a @ b) * 0.5  # DotProductNode -> ScaleNode
d = c + c  # AddNode; both operands are the same node object

graph = Graph([d], dag_id="bench-matmul-001")
graph.to_json(Path("dag.json"))
```

Emits five nodes — `lhs`, `init_1`, `dot_product_2`, `scale_3`, `add_4` — with
`outputs: ["add_4"]`. Note `d = c + c`: a node may legally appear twice in one `inputs` array, and
the reachability walk must dedupe by identity so `c` is emitted once.

Rank-0 results, new in schema 1.1.0:

```python
u = InitNode((3,), seed=1)
v = InitNode((3,), seed=2)
s = u @ v  # DotProductNode, output_shape () -> serialized as []
t = s * 2.0 + s  # rank 0 composes through scale and add
```

Failure cases from the same session:

```python
InitNode((4, 4))  # UninitializedNodeError: seed is required
InitNode((4, 4), seed=1) + InitNode((3, 3), seed=2)  # ShapeMismatchError
InitNode((4, 4), seed=1) * InitNode((3, 3), seed=2)  # ShapeMismatchError (multiply)
InitNode((4, 4), seed=1) ** -2  # ValueError: no division primitive
InitNode((4, 3), seed=1) % 0  # ValueError: modulus must be positive
matpow(InitNode((4, 3), seed=1), 2)  # ShapeMismatchError: not square
InitNode((4, 4), seed=1).cross(InitNode((3,), seed=2))  # DimensionalityError: rank 2
InitNode((4, 3), seed=1) @ InitNode((5, 2), seed=2)  # ShapeMismatchError: 3 != 5
InitNode((3,), seed=1) @ InitNode((4,), seed=2)  # ShapeMismatchError: 3 != 4
(u @ v).cross(u)  # DimensionalityError: rank-0 operand
```

---

## 11. Test checklist for Builder

Structure tests to mirror the module layout; `shapes.py` should carry the densest coverage since it
holds the rules with no construction ceremony.

- **`test_shapes.py`** — every row of the `dot` table including the rank-0 result and rank-0
  *operands* being rejected, `add` equality including rank mismatch and the rank-0 + rank-0 case,
  all four `cross` branches in order, FLOP estimates.
- **`test_init_node.py`** — all seven validation steps in order, `seed=0` and `seed=UINT64_MAX`
  accepted, `seed=True` and `shape=(True, 2)` rejected, `shape=()` rejected, all four distributions.
- **`test_operators.py`** — each row of the operator table; `a * b` between nodes raises `TypeError`;
  `a - b` raises `TypeError`; `2 * a` and `a * 2` build equivalent nodes; `a / 0` raises
  `ZeroDivisionError`.
- **`test_graph.py`** — dead-node exclusion; diamond reuse (`c + c`) emitting one node; determinism
  (serialize twice with `include_timestamp=False`, assert byte equality); duplicate-name
  `ValueError`; empty-outputs `ValueError`.
- **`test_cycles.py`** — build `a -> b` then `b.rewire(0, ...)` back into `a`; assert
  `CyclicDependencyError` and that the message names both nodes. Assert no `RecursionError`.
- **`test_multiply_mod.py`** — `multiply` shape equality including rank mismatch and the rank-0 case;
  `MultiplyNode` and `AddNode` produce identical topology apart from `op`; `mod` preserves shape and
  dtype; `modulus` of 0, negative, NaN, infinity, and `True` all rejected; `mod` output payload carries
  `modulus`, never `factor`.
- **`test_operators_v12.py`** — `a * b` between nodes is a `MultiplyNode` (the flipped behaviour), while
  `a * 2` is still a `ScaleNode` and `a * True` raises; `a - b` is exactly two nodes, an `AddNode` over a
  `ScaleNode(-1.0)`; `a ** 3` matches `tasks.math.pow(a, 3)` node for node; `a % 7` is a `ModNode`;
  `a % b` and `a / b` between nodes both raise `TypeError`.
- **`test_composites.py`** — the node-count tables in §14 and §15 asserted exactly, for every function
  and every listed parameter. Specifically: `pow(x, 1024)` has **10** multiply nodes (this is the whole
  binary-exponentiation claim, so assert the number, not an inequality); `powmod(x, 1024, m)` has 21;
  `matpow(A, 64)` has 6; `sin(x, terms=10)` has 29. Also depth anchors from §14/§15, `pow(x, 1) is x`,
  `pow(x, 0)` emitting a lone `ones` node that does not reference `x`, `matpow(A, 0)` raising
  `ValueError`, negative exponents raising `ValueError`, `terms=0` raising, `terms=86` raising for `sin`
  but not for `cos`, and `matpow` on a `(4, 3)` matrix raising `ShapeMismatchError` while a rank-1 input
  raises `DimensionalityError`.
- **`test_numeric_expansion.py`** — the test that proves the expansions actually compute what they claim.
  Write a small reference interpreter over the *serialized* document (about forty lines, dispatching on
  `op` into numpy: `add` → `+`, `multiply` → `*`, `scale` → `* factor`, `mod` → `np.mod` (floored, which
  is why the schema chose floored), `dot_product` → `@`, `cross_product` → `np.cross`). Give it an
  `overrides: dict[str, np.ndarray]` parameter that supplies values for `init` nodes by ID, so the test
  never depends on matching the engine's PRNG — only `ones`/`zeros` inits are reproducible across
  languages, and substitution sidesteps the issue entirely. Then compare against `np.sin`, `np.cos`,
  `np.exp`, `np.sinh`, `np.cosh`, `xv ** n`, `np.linalg.matrix_power`, and Python's `pow(a, n, m)`.

  **Choose tolerances carefully — this is where a naive test goes flaky.** I hit all three traps while
  verifying this design:
  1. *Truncation error dominates for few terms.* `exp(x, terms=10)` at `|x| = 0.5` is off by 2.8e−10,
     which is not a bug: it is the series remainder. A fixed `atol=1e-12` fails.
  2. *Round-off dominates for many terms.* `sin(x, terms=10)` has a truncation bound of 9.3e−26, far
     below float64 epsilon; its actual error is 5.5e−17 of pure round-off. A tolerance set from the
     truncation bound alone fails.
  3. *The remainder is the whole tail, not just the first term.* `exp`'s actual error (2.8e−10) slightly
     **exceeds** the first-omitted-term bound (2.7e−10), because the tail sums.

  So use `atol = max(10 * |x|**P / P!, 1e-14 * max(1, max|expected|))`, where `P` is the first omitted
  power. For `pow` with large `n`, use a *relative* tolerance instead: `1.2 ** 1024` is about 1.2e81, so
  an absolute error of 1.4e67 is a relative error of 1.2e−14 — correct to full precision. Also note
  `x ** 1024` overflows `float32` for any `|x| > 1.0906`, so keep the numeric fixtures near 1.0.

  For `powmod`, assert **exact** equality (`atol=0`) against Python's `pow(a, n, m)` for integer inputs
  within the §15 bound — that bound exists precisely so this test can be exact.
- **`test_schema_conformance.py`** — the highest-value test in the suite. `uv add --dev jsonschema`,
  load `/shared/dag_schema.json`, and validate the output of every graph the other tests build —
  including every composite expansion, which is where an off-by-one in a payload would otherwise hide.
  Assert `metadata.schema_version == "1.2.0"` and that a vector · vector dot serializes with
  `"output_shape": []`. A ready-made harness with 71 accept/reject cases against that schema is at
  `/tmp/check_dag_schema.py`; lift its valid fixtures as a starting point. A prototype of every
  composite expansion, already validated against the schema and cross-checked numerically against
  numpy, is at `/tmp/proto_composites.py` — the node-count tables and reference interpreter in it are
  directly liftable.

---

## 12. Summary of decisions the Reviewer should scrutinize

1. ~~Mixed dtypes promote silently rather than raising.~~ **Resolved:** promotion accepted, no fifth
   exception (§3).
2. ~~Vector · vector dot is rejected.~~ **Resolved:** schema amended to 1.1.0, rank-0 results are
   representable as `[]`, and `(m,) @ (m,)` is now accepted (§4). The init rank-1 floor added at the
   same time was itself reversed in v1.2.0 — see item 10.
3. **No broadcasting in `add`** — exact shape equality only.
4. **`a - b` is deliberately unsupported** while `-a` and `a / k` are, following the
   one-operator-one-node rule (P3).
5. **IDs are reassigned at serialize time** from topological position, making output byte-stable at
   the cost of `Node.node_id` not being final until the graph is closed.
6. **Unreachable nodes are silently dropped** rather than warned about.

### v1.2.0 additions

7. **P3 was rewritten**, from "one operator, one node" to "an operator lowers to its own standard
   definition". Necessary once `**` existed; see §1 for the full argument.
8. **`a * b` between two nodes now builds a `MultiplyNode`** instead of raising. This reverses my own
   earlier call, because the premise it rested on (no elementwise primitive) is gone. §5 has the
   reasoning.
9. **`a - b` is now provided** as a two-node expansion, and consequently `a + b` and `a - b` no longer
   cost the same. Worth a second opinion if anyone is comparing DAG sizes across operators.
10. **`init` may now be rank 0**, reversing the v1.1.0 restriction, whose justification (nothing could
    consume a rank-0 source) was invalidated by `multiply`/`mod` and by composites needing rank-0
    constants.
11. **`pow(x, 0)` returns a graph that does not reference `x`** — mathematically right, potentially
    startling. §15.
12. **`matpow(A, 0)` is rejected** rather than adding an `eye` distribution, on Tier-1 minimality
    grounds.
13. **No cross-call CSE**: `sin(x) + cos(x)` builds `x**2` twice (§13). Deliberate for v1, and arguably
    desirable for scheduler realism, but it is a real duplication and someone should agree it is fine.
14. **`powmod` is float modular arithmetic**, exact only below a computed modulus bound (94906266 for
    `float64`, 4097 for `float32`). Honest fix is an integer dtype, which is a much bigger change.

---

## 13. The composite tier (`tasks/math.py`)

Composites are ordinary Python functions that build subgraphs of primitives and return the node
holding the result. They are Tier 2: nothing about them reaches the wire, and adding one costs no C++
work. Everything in §1–§12 still applies to the nodes they emit — shape inference runs, exceptions
fire eagerly, `Graph.serialize` sorts and validates the result like any other graph.

### Free functions, not `Node` methods

All composites live in `tasks/math.py` as free functions. Two reasons, and the second is the one that
matters:

1. `sin(x)` reads like mathematics; `x.sin()` reads like an object graph. `tasks.math.pow(x, n)`
   deliberately mirrors stdlib `math.pow`.
2. **A method implies atomicity.** `x.cross(y)` is one node, and every other `Node` method is O(1) in
   nodes. Putting a thirty-node expansion behind the same syntax would make the two indistinguishable
   at the call site. A module-qualified free function announces that something bigger is happening,
   which matters when DAG topology is the object of study.

`__pow__` and `__mod__` are the two sanctioned exceptions, per the P3 revision.

### Shared signature conventions

```python
def sin(x: Node, *, terms: int = 8, label_prefix: str = "sin") -> Node:
    """Expand sin(x) as a Maclaurin series of elementwise primitive ops.

    Computes sum over k in [0, terms) of (-1)**k * x**(2k+1) / (2k+1)!, reusing x**2 so the
    odd powers cost one multiply each rather than one per unit of exponent.

    Args:
        x: Operand node. Any rank, including rank 0; the expansion is elementwise.
        terms: Number of series terms. Must be an int >= 1.
        label_prefix: Prefix for the ``label`` of every emitted node, for traceability.

    Returns:
        The node holding the summed series.

    Raises:
        ValueError: If ``terms`` is not an int >= 1, or is so large that a coefficient is not
            representable as a float (see the caps below).
    """
```

`terms` is keyword-only so a bare `sin(x, 8)` cannot be misread as a second operand. Validation, in
order: `bool` rejected first (it is an `int` subclass), then non-`int`, then `< 1`.

**Every emitted node carries a structured `label`** of the form `f"{label_prefix}/{role}"` — e.g.
`sin/pow7`, `sin/coeff3`, `sin/sum1_0`, `exp/ones`, `powmod/sq4`. The schema's `label` is optional and
never load-bearing, but for this project it is close to free and directly useful: it lets a scheduler
trace group nodes by the composite that produced them, which is exactly the kind of structure the
research wants to correlate against makespan. Keep labels under the schema's 128-character limit.

### Shape and dtype

Every composite except `matpow` is **elementwise**: output shape equals input shape, at any rank from
0 to 8 inclusive. Rank 0 works and is tested — `cos(u @ v)` on two vectors is a legitimate expression,
and it is one of the two reasons the v1.2.0 schema lifted the rank-1 floor on `init` (§14). dtype is
inherited from `x` throughout; the scalar coefficients are Python floats folded into `scale` factors,
and per §3 a `factor` never promotes, so a `float32` input yields a `float32` expansion.

### Two policies worth stating up front

**No common-subexpression elimination across calls.** `sin(x) + cos(x)` emits two separate `x**2`
nodes. Within a single call, powers are aggressively reused; across calls, nothing is shared. This is
deliberate for v1: a CSE pass belongs on `Graph`, where it can see the whole DAG, not smeared across
composite implementations that cannot know what else exists. It is also not obviously desirable —
redundant subtrees are realistic scheduler input. If it is wanted later, the natural home is a
`Graph.deduplicate()` method operating on a structural hash of `(op, input_ids, payload)`. Flagged as
future work, not a defect.

**Unit coefficients are not optimized away.** `sin(x, terms=1)` still emits a `scale` by exactly 1.0.
Skipping it would make node count depend on the *values* of coefficients rather than on the parameters,
and a topology that changes on numeric coincidence is a poor experimental subject. The node-count
formulas below are exact because of this.

---

## 14. Series composites: `sin`, `cos`, `exp`, `sinh`, `cosh`

One shared private engine, five public wrappers:

```python
def _maclaurin(x: Node, terms: int, *, parity: str, alternate: bool, prefix: str) -> Node:
    """Expand a Maclaurin series over elementwise primitives.

    Args:
        parity: ``"odd"`` for powers 2k+1 (sin, sinh), ``"even"`` for 2k (cos, cosh),
            ``"all"`` for k (exp).
        alternate: Whether to apply the (-1)**k sign factor.
    """
```

| Function | Powers | Signs |
| --- | --- | --- |
| `sin` | 2k+1 | alternating |
| `sinh` | 2k+1 | all positive |
| `cos` | 2k | alternating |
| `cosh` | 2k | all positive |
| `exp` | k | all positive |

### Expansion algorithm

1. **Power cache.** Seed with `{1: x}`. If power 0 is needed (`cos`, `cosh`, `exp`), emit an `init`
   node with `distribution="ones"`, `seed=0`, and `shape == x.output_shape` — the contract's
   constant-tensor mechanism, recorded in the schema's `distribution` description.
2. **Stride.** For odd/even parity emit `x2 = multiply(x, x)` once and step powers by 2, so
   `x**3 = x**1 · x**2`, `x**5 = x**3 · x**2`, and so on: **one multiply per term**, not one per unit
   of exponent. For `exp`, step by 1 (`x**k = x**(k-1) · x`), which is optimal there because every
   intermediate power is itself a needed term.
3. **Coefficients.** For each k, `coeff = (-1)**k / factorial(p)` (or without the sign), computed in
   Python and folded into a single `scale` node on the cached power. Signs never need a `subtract`.
4. **Summation.** Balanced pairwise tree, left to right: repeatedly pair `(0,1), (2,3), …`, carrying a
   final odd element forward unpaired. Specify this exactly — it fixes both the node count and the
   floating-point summation order, so numerical tests are reproducible.

A balanced tree rather than a left-to-right chain because it has identical node count but logarithmic
depth, which exposes the parallelism a scheduler exists to exploit. Chained summation would make every
series a serial dependency, which would be a poor default for this project specifically.

### Node counts — verified

`N` = `terms`. Counts exclude `x` itself.

| Function | N = 1 | N ≥ 2 |
| --- | --- | --- |
| `sin`, `sinh` | 1 | **3N − 1** |
| `cos`, `cosh` | 2 | **3N − 1** |
| `exp` | 2 | **3N − 2** |

For N ≥ 2 the `sin` breakdown is: 1 (`x2`) + (N−1) (odd powers `x3 … x^(2N-1)`) + N (`scale`) +
(N−1) (`add`) = 3N − 1. `cos` trades one power multiply for the `ones` node, landing on the same total;
`exp` needs no `x2` seed, saving one.

Measured for N = 1…8, matching the formulas exactly:

```
sin, sinh :  1,  5,  8, 11, 14, 17, 20, 23
cos, cosh :  2,  5,  8, 11, 14, 17, 20, 23
exp       :  2,  4,  7, 10, 13, 16, 19, 22
```

So `sin(x, terms=10)` is 29 nodes — the "sine expands into many nodes" property the research wants,
from a five-op vocabulary.

### Depth — verified anchors

Depth is determined by the algorithm above but has no tidy closed form, because the deepest power
lands in the last slot of the summation tree and is carried unpaired through some levels. Assert
against these measured values rather than a formula. At `terms=10`: `sin` 13, `sinh` 13, `cos` 12,
`cosh` 12, `exp` 11.

### Term caps — computed, not guessed

`1.0 / float(math.factorial(p))` raises **`OverflowError`** (not a silent underflow to 0.0) once
`p >= 171`, because the exact integer factorial exceeds the float64 range during conversion. Builder
must catch that and re-raise as `ValueError`. The resulting caps:

| Parity | Highest power | Max `terms` |
| --- | --- | --- |
| odd (`sin`, `sinh`) | 2N−1 | **85** |
| even (`cos`, `cosh`) | 2(N−1) | **86** |
| all (`exp`) | N−1 | **171** |

Rather than hardcoding 85/86/171, compute the coefficient inside `try`/`except OverflowError` and
reject with a message naming the offending term. That way the check stays correct if the power schedule
ever changes.

### Accuracy: a caveat that must be documented, not hidden

**There is no range reduction.** Maclaurin series are accurate only for small `|x|`, and implementing
argument reduction would need `floor` and division primitives that deliberately do not exist (§16).
The truncation error is bounded by the first omitted term, `|x|**P / P!`. At the default `terms=8`:

| \|x\| | `sin` (P=17) | `cos` (P=16) | `exp` (P=8) |
| --- | --- | --- | --- |
| 0.5 | 2.1e−20 | 7.3e−19 | 9.7e−08 |
| 1.0 | 2.8e−15 | 4.8e−14 | 2.5e−05 |
| 2.0 | 3.7e−10 | 3.1e−09 | 6.4e−03 |
| π | 8.0e−07 | 4.3e−06 | 2.4e−01 |
| 10.0 | **2.8e+02** | **4.8e+02** | **2.5e+03** |

At `|x| = 10` the result is not merely inaccurate, it is meaningless. Say so in the docstrings. These
DAGs are workload generators for a scheduling study first and a numerics library second, and pretending
otherwise would be the kind of quiet inaccuracy that discredits a benchmark. Note also that `exp`
converges far more slowly than `sin`/`cos` at equal `terms`, because its omitted power is `N` rather
than roughly `2N` — so `exp` wants a larger `terms` for comparable accuracy.

---

## 15. Power composites: `pow`, `powmod`, `matpow`

All three share one left-to-right binary exponentiation walk; they differ only in the binary operation
used and whether a `mod` is interposed.

```python
def pow(x: Node, n: int, *, label_prefix: str = "pow") -> Node
def powmod(x: Node, n: int, m: float, *, label_prefix: str = "powmod") -> Node
def matpow(a: Node, n: int, *, label_prefix: str = "matpow") -> Node
```

### Algorithm

Process the bits of `n` from the most significant downward, skipping the leading 1:

```
acc = base
for bit in bin(n)[3:]:
    acc = op(acc, acc)              # square
    if bit == "1":
        acc = op(acc, base)         # multiply in
```

`op` is `multiply` for `pow`/`powmod` and `dot_product` for `matpow`. For `powmod`, `base` is
`mod(x, m)` and every `op` result is wrapped in a `mod` node — the user's explicit requirement that the
modulus be applied after *every* multiply, which is also what keeps intermediates bounded.

### Node counts — verified

Multiplies for exponent `n ≥ 1`:

```
multiplies(n) = floor(log2 n) + popcount(n) - 1     # == n.bit_length() - 1 + bin(n).count("1") - 1
```

| Composite | Nodes | n = 0 |
| --- | --- | --- |
| `pow(x, n)` | `multiplies(n)` | 1 (a `ones` init) |
| `powmod(x, n, m)` | `2 * multiplies(n) + 1` | 2 (`ones` then `mod`) |
| `matpow(A, n)` | `multiplies(n)` (all `dot_product`) | **rejected** |

Measured, matching exactly:

```
pow     n = 1, 2, 3, 7, 10, 16, 100, 1024  ->  0, 1, 2, 4, 4, 4, 8, 10
powmod  n = 1, 5, 10, 100, 1024            ->  1, 7, 9, 17, 21
matpow  n = 1, 2, 5, 16, 64                ->  0, 1, 3, 4, 6
```

**`pow(x, 1024)` is 10 multiply nodes, not 1023.** That is the binary-exponentiation claim, and §11
makes it an assertion rather than a comment.

### Depth

All three produce a **pure serial chain**: depth equals node count (`pow(x, 1024)` depth 10,
`powmod(x, 1024, m)` depth 21, `matpow(A, 64)` depth 6). This is not a flaw to fix — it is
scheduling-research value. The composite family now spans both extremes: series expansions are wide
and shallow (29 nodes, depth 13, lots of independent work), while `powmod` is narrow and deep (21
nodes, depth 21, zero parallelism). A scheduler that looks good on one may look bad on the other,
which is precisely the contrast a baseline should contain.

A note for later: left-to-right was chosen for simplicity. Right-to-left binary exponentiation lets
the squaring chain and the accumulation partially overlap, yielding slightly more parallelism at
identical node count. Worth adding as a second strategy if the research wants to vary that axis.

### Edge cases, with justifications

**`n = 1` returns `x` itself — zero new nodes.** `multiplies(1) = 0`, so the formula already predicts
this; emitting an identity `scale` would contradict it. `pow(x, 1) is x` holds and should be asserted.

**`n = 0` emits a `ones` node, and this is the one genuinely surprising case.** `x**0 = 1` for all `x`,
so the result is a constant tensor shaped like `x` and it **does not depend on `x` at all**. If `x` has
no other consumer, `Graph`'s reachability walk will drop it from the serialized DAG entirely (§6, dead
code elimination) — the user asks for a function of `x` and gets a graph that never mentions `x`.
Document this loudly in the docstring. I chose to support `n = 0` rather than reject it because it makes
`pow` total over `n >= 0`, so a parameter sweep over `n = 0…10` does not crash on its first iteration,
and because `init`/`ones` already exists to express it. Convention `0**0 == 1` matches NumPy.

**Negative `n` is rejected with `ValueError`** for all three: `x**-1` needs a division primitive and
`A**-1` needs a matrix inverse, neither of which exists (§16).

**`matpow(A, 0)` is rejected, unlike `pow(x, 0)`.** The identity matrix is not expressible: the `init`
distribution enum is `uniform | normal | zeros | ones`, with no `eye`. `ones((n,n))` is emphatically not
the multiplicative identity. Raise `ValueError` naming the reason. Adding `"eye"` to the enum would fix
it for one degenerate case at the cost of another C++ code path, which fails the Tier-1 minimality test
— so it is documented as a candidate for a future version rather than smuggled into v1.2.0. The schema's
`distribution` description records this decision so the C++ authors know the omission is intentional.

**`matpow` shape validation** runs before any expansion, in this order: rank ≠ 2 →
`DimensionalityError`; rank 2 but `shape[0] != shape[1]` → `ShapeMismatchError`. Same ordering
rationale as `cross_product`.

### `powmod` and float exactness — a hard, computed bound

`mod` is exact only while operands remain exactly representable integers. Because the modulus is
applied after every multiply, intermediates are bounded by `(m-1)**2`, giving a checkable precondition:

| dtype | Exact integers to | Largest safe modulus |
| --- | --- | --- |
| `float64` | 2\*\*53 = 9007199254740992 | **94906266** |
| `float32` | 2\*\*24 = 16777216 | **4097** |

Both verified: at `m = 94906266`, `(m-1)**2 = 9007199136250225 <= 2**53`, while `m**2` exceeds it;
`float32`'s bound is tight, with `4096**2 == 2**24` exactly.

`powmod` therefore validates: `n >= 0`; `m` integral, `> 0`, and satisfying `(m-1)**2 <= 2**53` (or
`2**24` for `float32`) — raising `ValueError` with the computed limit when it does not. Provide
`allow_inexact: bool = False` to bypass the last check for users who genuinely want float remainder
arithmetic. Be blunt in the docstring: **this is `float` modular arithmetic, not integer modpow.**
Within the bound it agrees with Python's `pow(a, n, m)` exactly, which is worth asserting in a test;
outside it, results silently drift. The correct long-term fix would be an integer dtype in the
contract, which is a much larger change than adding an op.

---

## 16. Deliberately out of scope

Excluded because each would require a **division** primitive, and division brings division-by-zero — a
runtime concern the engine owns, plus a build-time validation problem we cannot solve (we cannot know
whether a tensor contains a zero without evaluating it, which the lazy-evaluation contract forbids):

- `tan`, `cot`, `sec`, `csc` — quotients of series.
- `arcsin`, `arctan`, `log` — series with non-factorial denominators requiring per-term division, and
  in `log`'s case no useful expansion about 0 at all.
- `sqrt` / fractional powers — Newton iteration needs division.
- Negative exponents in `pow` / `matpow`.
- Range reduction for `sin`/`cos`, which needs `floor` and division (§14).

Also excluded, for the Tier-1 minimality reason rather than a numerical one:

- `subtract` as a **primitive** — a negative `scale` factor plus `add` covers it in two nodes with no
  new C++ path. The `a - b` operator lowers to exactly that.
- `eye` / identity as an `init` distribution — see `matpow(A, 0)` above.
- A dedicated `constant` op — `init` with `distribution="ones"` plus a `scale` expresses any constant
  tensor. It costs a full-size buffer of ones, which is real memory for large shapes and worth
  revisiting if constants ever appear in hot paths, but not worth an eighth primitive today.

If any of these becomes necessary, the decision is a schema version bump plus C++ work in `/engine`,
which is a conversation with the humans who own that directory — not a Python-side change.
