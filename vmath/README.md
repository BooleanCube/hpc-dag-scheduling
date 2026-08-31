# /vmath — the DAG math builder

Python library for writing mathematical DAGs that the C++ engine executes. You write ordinary
NumPy-flavoured expressions; the library records them as a graph, validates every logical
property eagerly, and serializes to the [`/shared/dag_schema.json`](../shared/dag_schema.json)
contract. **No arithmetic ever runs in Python** — construction records intent, the cluster does
the math.

```python
from pathlib import Path
from vmath import Graph, InitNode

a = InitNode((64, 32), seed=42, distribution="normal")
b = InitNode((32, 16), seed=43)
graph = Graph([(a @ b) * 0.5], dag_id="bench-matmul-001")
graph.to_json(Path("bench-matmul-001.json"))     # ready for `hpcctl submit ...`
```

Scripts meant for `hpcctl submit` end with `emit(graph)` instead: run as
`hpcctl submit my_case.py`, the document is written where hpcctl asks (the path in
`sys.argv[1]`); run standalone, it prints to stdout. The repository's benchmark
workloads live in [`/cases`](../cases/README.md) and all follow that convention.

## Design in four rules

1. **Lazy math, eager validation.** Building a node computes nothing, but shapes and dtypes
   resolve at construction — `a + b` with mismatched shapes raises *at that line*, not at
   serialization. The one exception is cycle detection, a whole-graph property checked when
   the graph serializes.
2. **Nodes are values, not graph members.** A node never references a graph, which is what
   lets bare `a + b` work. `Graph([outputs], dag_id=...)` closes over output nodes and
   discovers everything else by walking operand references backwards. Nodes that don't reach
   an output are silently dropped (intentional: explore alternatives freely in a script).
3. **Two tiers.** Seven *primitive* ops exist on the wire (each is a C++ code path). Everything
   else — `sin`, `matpow`, `**`, `-` — is a *composite*: a plain Python function that expands
   into primitives. Adding a composite costs zero C++ work.
4. **A method is one node; a function may be many.** `x.cross(y)` emits one node. `sin(x)`
   emits ~23. Composites are free functions so the two are distinguishable at the call site —
   DAG topology is the object of study here. (`**` and `%` are the sanctioned operator
   exceptions.)

## The seven primitives

| Op | Built by | Shape rule |
| -- | -------- | ---------- |
| `init` | `InitNode(shape, seed=…, dtype="float64", distribution="uniform")` | rank 0–8; every extent positive. `distribution`: `uniform`, `normal`, `zeros`, `ones`. Seed required (in `[0, 2**64)`) even for `zeros`/`ones` |
| `add` | `a + b` | shapes must match **exactly** — no broadcasting |
| `multiply` | `a * b` (elementwise/Hadamard) | shapes must match exactly. Never a contraction |
| `scale` | `a * 2`, `2 * a`, `a / 2`, `-a` | any shape; scalar constant lives on the node |
| `mod` | `a % m` | any shape; scalar modulus `m > 0`; result in `[0, m)` (Python/NumPy floored semantics) |
| `dot_product` | `a @ b` | rank 1 or 2 operands; trailing extent of `a` contracts with leading extent of `b`. Vector @ vector → rank-0 scalar |
| `cross_product` | `v.cross(w)` or `cross(v, w)` | both operands length-3 rank-1 vectors; result `(3,)` |

Dtypes are `float64` (default) and `float32`; mixing promotes to `float64`. Every node takes
optional `name=` (becomes the serialized `id`, must be unique) and `label=` (free-form
annotation, never load-bearing).

## Operators at a glance

```python
a + b        # add
a - b        # composite: add(a, scale(b, -1))         — costs 2 nodes, not 1
a * b        # multiply (elementwise; never matmul)
a * 2.0      # scale            (also 2.0 * a)
a / 2.0      # scale by 1/2     (node / node is NOT supported: no division primitive)
-a           # scale by -1
a % 97       # mod              (node % node is NOT supported: modulus is a scalar field)
a @ b        # dot_product      (matmul, as in NumPy)
a ** 8       # composite: binary exponentiation — 3 multiply nodes, not 7
v.cross(w)   # cross_product
```

There is deliberately **no `/` or `%` between two nodes** and **no negative exponent**: both
need a division primitive, and division brings divide-by-zero — a runtime concern that belongs
to the engine, not to a builder that never evaluates tensors.

## Composites (`vmath.math`)

| Function | Expands to | Nodes emitted |
| -------- | ---------- | ------------- |
| `sin(x, terms=8)` / `sinh` | Maclaurin series, odd powers | `3·terms − 1` |
| `cos(x, terms=8)` / `cosh` | Maclaurin series, even powers | `3·terms − 1` |
| `exp(x, terms=8)` | Maclaurin series, all powers | `3·terms − 2` |
| `pow(x, n)` / `x ** n` | binary exponentiation (elementwise) | `⌊log₂n⌋ + popcount(n) − 1` |
| `matpow(A, n)` | binary exponentiation over `@` | same formula, `dot_product` nodes |
| `powmod(x, n, m)` | multiply+mod after every step | `2·multiplies(n) + 1` |

Caveats that matter (all enforced or documented in `vmath/math.py`):

- **No range reduction.** The series are accurate for small `|x|` only; at `|x|=10` the result
  is meaningless. These are workload generators for a scheduling study first, a numerics
  library second. `exp` converges slower than `sin`/`cos` at equal `terms` — ask for more.
- **`terms=1` for `cos`/`cosh`/`exp`, and `x ** 0`, do not depend on `x` at all** (the only
  term is the constant `x⁰`). If `x` has no other consumer, dead-code elimination drops it.
- **`powmod` is float modular arithmetic**, exact only while `(m−1)² ≤ 2⁵³`, i.e.
  `m ≤ 94 906 266` for `float64` (`4097` for `float32`). Larger moduli are rejected unless you
  pass `allow_inexact=True`.
- **`matpow(A, 0)` is rejected**: the identity matrix is not expressible (no `eye`
  distribution), and `ones((n,n))` is emphatically not it.

## A tour of everything

This example runs as-is: it builds a 72-node DAG using all seven primitives and every
composite, and the output validates against the shared schema.

```python
from pathlib import Path
from vmath import Graph, InitNode, cross, exp, matpow, powmod, sin

a = InitNode((64, 32), seed=1, distribution="normal", name="a")
b = InitNode((32, 64), seed=2, distribution="uniform")
v = InitNode((3,), seed=3, name="v")
w = InitNode((3,), seed=4, dtype="float32")

product = a @ b                # dot_product: (64,32) @ (32,64) -> (64,64)
hadamard = product * product   # multiply (elementwise)
shifted = hadamard + product   # add
diff = shifted - product       # subtract: expands to add(x, scale(y, -1))
halved = 0.5 * diff            # scale (also diff * 0.5 and diff / 2)
reduced = halved % 97          # mod: remainder in [0, 97)
powered = product ** 3         # binary exponentiation: 2 multiply nodes
normal = cross(v, w)           # cross_product of two length-3 vectors

wavy = sin(product, terms=6)               # 17 primitive nodes
grown = exp(reduced, terms=10)             # 28 primitive nodes
eighth = matpow(a @ b, 8, label_prefix="A8")   # 3 dot_product nodes
residue = powmod(product, 12, 10_007)      # exact float modpow

graph = Graph(
    [halved, powered, normal, wavy, grown, eighth, residue],
    dag_id="tour-of-ops-001",
    description="one of everything",
)
graph.to_json(Path("tour-of-ops-001.json"))
```

Then run it on the cluster (see [`hpcctl/README.md`](../hpcctl/README.md)):

```bash
uv run hpcctl submit tour-of-ops-001.json -n 4
```

`Graph.serialize()` also takes `include_timestamp=False` for byte-stable output,
`include_hints=False` to drop the `est_flops` scheduler hints, and `renumber=False` to keep
construction-time IDs instead of canonical topological ones.

## Exceptions

All four logical errors derive from `vmath.DagBuildError`, so `except DagBuildError` catches
the category. A DAG that raises any of them **never reaches the engine** — that is the
project's core contract (the engine handles only runtime physics: OOM, MPI failures, schema
parsing, preemption, NaN/Inf).

| Exception | Meaning | Example trigger |
| --------- | ------- | --------------- |
| `ShapeMismatchError` | Operand dimensions don't align for the operation | `a + v` with shapes `(64,32)` and `(3,)`; `a @ b` with inner extents 32 vs 16; `matpow` on a non-square matrix |
| `DimensionalityError` | Operation applied to the wrong tensor *rank* | `cross(a, b)` on matrices (needs length-3 vectors); `a @ b` with a rank-3 operand; `matpow` on a vector |
| `UninitializedNodeError` | An `init` node is missing or has an invalid shape or seed | `InitNode((8, 8))` with no `seed`; a zero or negative extent; rank > 8 |
| `CyclicDependencyError` | The graph is not acyclic. **Only raised by `Graph.serialize()`** — the one property that can't be checked at the offending line. Normal operator use cannot create a cycle; `node.rewire()` can | rewiring a node to consume its own output, then serializing |

Everything raised at the line that wrote it, except the cycle check. Two neighbouring families
are *not* `DagBuildError` because they're API misuse rather than mathematical invalidity:

- `ValueError` / `TypeError` — bad knobs: `terms=0`, a negative exponent, a non-integral or
  oversized `powmod` modulus, a `name` that doesn't match the ID pattern, a duplicate output
  in `Graph(...)`, two nodes serializing to the same `id`.
- `ZeroDivisionError` — `a / 0`.

## Development

```bash
cd vmath
uv sync && uv run pytest && uv run ruff check . && uv run mypy .
```
