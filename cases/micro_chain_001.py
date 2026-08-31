"""micro-chain-001 — a pure sequential chain with zero exploitable parallelism.

Size class: tiny graph (13 nodes), tiny tensors ((16,) vectors, 128 B each).

What it measures: the per-node engine overhead and the MPI floor. Every node depends on the
previous one, so *no* scheduler can beat any other on placement — the makespan is pure
dependency-chain latency. Any scheduler that performs differently here is adding overhead
(e.g. bouncing the chain between ranks and paying a transfer per hop), which is exactly what
this case exists to expose. Expect identical runtimes at -n 1, 2, and 4; a scheduler whose
runtime *grows* with -n is shipping the chain around the cluster for no reason.

Suggested: -n 1, 2, 4.
"""

from vmath import Graph, InitNode, Node, emit

x = InitNode((16,), seed=101, distribution="normal", name="x")
y: Node = x
for _ in range(6):
    y = (y * 1.5) + x

emit(
    Graph(
        [y],
        dag_id="micro-chain-001",
        description="12-node serial chain on tiny vectors; measures per-node and MPI overhead",
    )
)
