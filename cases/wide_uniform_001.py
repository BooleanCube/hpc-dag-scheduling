"""wide-uniform-001 — 32 independent, perfectly uniform chains (the round-robin control).

Size class: medium graph (~160 nodes), medium tensors ((128,128) matrices, 128 KiB each).

What it measures: this is the deliberate CONTROL case in which naive schedulers are supposed
to look good. Thirty-two independent 4-node chains of identical cost become ready at once;
round-robin achieves a perfect balance by construction and locality-greedy keeps each chain
on one rank. A cost-aware scheduler must *match* them here — if it loses on the uniform case,
its wins on the skewed cases are buying imbalance insurance with real overhead. Compare
directly against wide-skewed-001, which differs only in the cost distribution.

Suggested: -n 2, 4 (speedup should be near-linear to 4).
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


chains = []
for i in range(32):
    left = InitNode((128, 128), seed=1000 + 2 * i, distribution="normal")
    right = InitNode((128, 128), seed=1001 + 2 * i, distribution="uniform")
    chains.append(((left @ right) * 1.01) + left)

emit(
    Graph(
        [sum_tree(chains)],
        dag_id="wide-uniform-001",
        description="32 identical independent chains; the control where round-robin is optimal",
    )
)
