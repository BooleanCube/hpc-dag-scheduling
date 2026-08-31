"""xs-swarm-001 — ~1300 nodes of near-zero work: the big-graph/small-data extreme.

Size class: extra-large graph (~1300 nodes), every tensor a (4,) vector (32 bytes).

What it measures: raw graph-processing throughput — parse, topological bookkeeping, task
dispatch, and completion tracking at a node count where per-task fixed costs dominate
utterly. Forty chains of thirty tiny ops run in parallel, cross-linked to their neighbour
chain every ten steps so the graph is genuinely connected (a scheduler cannot trivially
treat it as 40 sealed silos), then folded by a final reduction. Wall-clock here is almost
pure scheduler-and-runtime overhead per node; divide the makespan by 1300 and track that
number across engine versions. Compare against xl-dense-chain-001 — the graphs are opposite
corners of the (nodes x bytes) plane, and a scheduler tuned for one often embarrasses
itself on the other.

Suggested: -n 1, 2, 4.
"""

from vmath import Graph, InitNode, Node, emit

CHAINS = 40
DEPTH = 30


def sum_tree(nodes: list[Node]) -> Node:
    """Reduce nodes pairwise so the combine is log-depth rather than a serial chain.

    Args:
        nodes: Nodes of identical shape to sum.

    Returns:
        The root of a balanced binary add tree.
    """
    layer = list(nodes)
    while len(layer) > 1:
        merged = [a + b for a, b in zip(layer[::2], layer[1::2], strict=False)]
        if len(layer) % 2:
            merged.append(layer[-1])
        layer = merged
    return layer[0]


heads: list[Node] = [
    InitNode((4,), seed=10300 + i, distribution="uniform") for i in range(CHAINS)
]
for step in range(1, DEPTH + 1):
    heads = [h * (1.0 + 0.001 * step) for h in heads]
    if step % 10 == 0:
        heads = [h + heads[(i + 1) % CHAINS] for i, h in enumerate(heads)]

emit(
    Graph(
        [sum_tree(heads)],
        dag_id="xs-swarm-001",
        description="40 cross-linked chains of 30 tiny ops (~1300 nodes); overhead throughput",
    )
)
