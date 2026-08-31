"""tiny-allops-001 — every wire primitive exactly once, on small tensors.

Size class: tiny graph (~14 nodes), small tensors ((8,8) matrices and (3,) vectors).

What it measures: correctness and per-op dispatch cost, not scheduling. This is the smoke
case: all seven primitive ops (init, add, multiply, scale, mod, dot_product, cross_product)
appear, with multiple outputs of differing rank (matrix, vector, and the engine's rank-0
buffer path is covered elsewhere). Run it first after any engine change; if this case is
wrong or slow, nothing else is worth measuring.

Suggested: -n 1, 2.
"""

from vmath import Graph, InitNode, cross, emit

a = InitNode((8, 8), seed=301, distribution="normal", name="a")
b = InitNode((8, 8), seed=302, distribution="uniform", name="b")
v = InitNode((3,), seed=303, name="v")
w = InitNode((3,), seed=304, dtype="float32", name="w")

product = a @ b
hadamard = a * b
summed = product + hadamard
scaled = summed * 0.25
reduced = scaled % 11.0
normal = cross(v, w)

emit(
    Graph(
        [reduced, normal],
        dag_id="tiny-allops-001",
        description="smoke case: all seven primitives once, mixed dtypes and ranks",
    )
)
