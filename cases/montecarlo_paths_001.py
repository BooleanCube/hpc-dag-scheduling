"""montecarlo-paths-001 — 24 statistically identical simulation paths reduced to one scalar.

Size class: large graph (~290 nodes), (8192,) path vectors (64 KiB each), rank-0 outputs.

What it measures: throughput fairness on an embarrassingly parallel ensemble whose branches
are identical IN DISTRIBUTION but not in instruction stream (each path runs a cosh series,
a modular fold, then contracts to a scalar against a shared weight vector). With 24 equal
branches over 2, 3, or 4 instances the ideal split is exact, so ANY makespan spread across
ranks is scheduler noise — this is the fairness yardstick the trap cases are judged against.
The vector-vector dot at the end of each path exercises the engine's rank-0 buffer path 24
times. Real-world analogue: Monte-Carlo pricing or particle-transport batches.

Suggested: -n 2, 3, 4.
"""

from vmath import Graph, InitNode, Node, cosh, emit


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


weights = InitNode((8192,), seed=9500, distribution="uniform", name="payoff_weights")


def path(index: int) -> Node:
    """Simulate one Monte-Carlo path and contract it to a scalar payoff.

    Args:
        index: Path index, used to derive a distinct seed.

    Returns:
        A rank-0 payoff node.
    """
    shocks = InitNode((8192,), seed=9600 + index, distribution="normal")
    evolved = cosh(shocks * 0.01, terms=4, label_prefix=f"path{index}") % 1.0
    return evolved @ weights


emit(
    Graph(
        [sum_tree([path(i) for i in range(24)])],
        dag_id="montecarlo-paths-001",
        description="24 identical-cost simulation paths contracted to scalars; fairness yardstick",
    )
)
