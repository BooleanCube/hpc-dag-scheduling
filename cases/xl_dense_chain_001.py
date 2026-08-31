"""xl-dense-chain-001 — eight nodes, each huge: the small-graph/big-data extreme.

Size class: tiny graph (9 nodes), (2048,2048) matrices (32 MiB each, ~17 GFLOP per matmul).

What it measures: the memory/bandwidth-bound end of the spectrum, opposite of xs-swarm-001.
Two sequential 2048-cube matmuls followed by elementwise post-processing leave the scheduler
with almost no decisions — which is the point: the graph fits in one screen, the tensors do
not fit in cache, and the makespan is arithmetic plus any transfer the scheduler is careless
enough to introduce. One avoidable 32 MiB hop between ranks is directly legible in the
runtime. Also the first case to watch on memory: peak live set is a few hundred MiB, real
pressure on a c5.large. Compare against markov-power-001 (same shape of chain, 64x less data
per node) to separate compute floors from bandwidth floors.

Suggested: -n 1, 2 (there is nothing for a third instance to do; verify the scheduler agrees).
"""

from vmath import Graph, InitNode, emit

a = InitNode((2048, 2048), seed=10200, distribution="normal", name="a")
b = InitNode((2048, 2048), seed=10201, distribution="uniform", name="b")
c = InitNode((2048, 2048), seed=10202, distribution="normal", name="c")

chain = ((a @ b) * 1e-3) @ c
result = (chain % 997.0) + a

emit(
    Graph(
        [result],
        dag_id="xl-dense-chain-001",
        description="two sequential 2048^3 matmuls plus elementwise tail; 32 MiB per operand",
    )
)
