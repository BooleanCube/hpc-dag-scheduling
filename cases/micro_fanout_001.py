"""micro-fanout-001 — one tiny source fanning out to 16 independent branches.

Size class: small graph (~48 nodes), tiny tensors ((64,) vectors, 512 B each).

What it measures: fan-out placement and small-message handling. All 16 branches become ready
the moment the single init completes, so the scheduler faces a pure placement decision with
negligible compute per task. Because every branch costs the same, this is a *fair* case for
round-robin — it is the reduction tree afterwards (log-depth, shrinking width) that separates
schedulers: the pairwise adds force cross-rank communication whose cost dwarfs the arithmetic.
A scheduler that scatters the branches maximally pays for it in the reduction.

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


x = InitNode((64,), seed=202, distribution="uniform", name="src")
branches = [(x * float(k + 1)) % 7.3 for k in range(16)]

emit(
    Graph(
        [sum_tree(branches)],
        dag_id="micro-fanout-001",
        description="16-way fan-out from one tiny source into a log-depth reduction",
    )
)
