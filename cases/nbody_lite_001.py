"""nbody-lite-001 — hundreds of 24-byte tensor ops: communication cost dwarfs compute.

Size class: large graph (~220 nodes), every tensor a (3,) vector (24 bytes).

What it measures: overhead discipline. Thirty-two "bodies" produce ring cross products
(r_i = p_i x p_{i+1}), each torque mixes its own and its neighbour's result, and a global
reduction folds everything down — so the graph is dense with dependencies but each op is
maybe a dozen FLOPs. Shipping ANY of these tasks to another rank costs orders of magnitude
more in latency than executing it in place; the optimal schedule at every instance count is
close to "one rank does everything". A scheduler that proudly balances this workload across
four instances demonstrates exactly the pathology this case exists to catch. Real-world
analogue: small-N rigid-body physics steps, molecular torque accumulation.

Suggested: -n 1, 2, 4 (runtime should be flat or RISE with -n; rising is the lesson).
"""

from vmath import Graph, InitNode, Node, emit

BODIES = 32

positions = [InitNode((3,), seed=9700 + i, distribution="normal") for i in range(BODIES)]
crossed = [p.cross(positions[(i + 1) % BODIES]) for i, p in enumerate(positions)]
torques = [
    (r * float(i % 5 + 1)) + crossed[(i - 1) % BODIES] for i, r in enumerate(crossed)
]

total: Node = torques[0]
for torque in torques[1:]:
    total = total + torque

emit(
    Graph(
        [total],
        dag_id="nbody-lite-001",
        description="ring of cross products on (3,) vectors; punishes eager task distribution",
    )
)
