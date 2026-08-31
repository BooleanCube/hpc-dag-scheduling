"""trap-locality-001 — an all-to-all shuffle stage that no placement can make local.

Size class: medium graph (~150 nodes), (512,512) matrices (2 MiB each) crossing every edge.

What it measures: communication scheduling when locality is impossible. Stage 1 builds eight
independent 2 MiB producer tensors. Stage 2 has eight consumers, and EVERY consumer combines
scaled copies of ALL EIGHT producers — an 8x8 bipartite shuffle. A locality-greedy scheduler
that parks each consumer next to "its" producer still owes seven remote operands per
consumer; there is no clever placement, only good *overlap* (starting transfers early,
pipelining the reduction while data arrives). Schedulers that win here win on communication
scheduling, not placement luck. Compare -n 1 (zero transfers) against -n 4 to price the
shuffle itself.

Suggested: -n 1, 2, 4.
"""

from vmath import Graph, InitNode, Node, emit


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


producers = []
for i in range(8):
    a = InitNode((512, 512), seed=4000 + 2 * i, distribution="normal")
    b = InitNode((512, 512), seed=4001 + 2 * i, distribution="uniform")
    producers.append(a @ b)

consumers = [
    sum_tree([p * (1.0 / (i + j + 1)) for j, p in enumerate(producers)]) for i in range(8)
]

emit(
    Graph(
        consumers,
        dag_id="trap-locality-001",
        description="8x8 all-to-all shuffle of 2 MiB tensors; locality cannot win, overlap can",
    )
)
