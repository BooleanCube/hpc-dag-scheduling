"""poly-features-001 — polynomial feature expansion with geometrically growing branch costs.

Size class: medium graph (~40 nodes), (512,32) data matrix (128 KiB).

What it measures: the feature-engineering pattern (x, x^2, ..., x^9 of a design matrix,
then a weighted sum). All branches fork from the same source, but branch k costs O(log k)
multiplies via binary exponentiation, so the cost profile is a smooth known gradient rather
than uniform (fair to locality — every branch reads the same tensor) — the discriminator is
whether the scheduler finishes the cheap branches early and has the weighted-sum reduction
already half-built by the time x^9 lands, or serializes the reduction behind the slowest
branch. Real-world analogue: sklearn PolynomialFeatures ahead of a ridge regression.

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


data = InitNode((512, 32), seed=9100, distribution="uniform", name="design_matrix")
features = [(data**k) * (1.0 / float(k)) for k in range(1, 10)]

emit(
    Graph(
        [sum_tree(features)],
        dag_id="poly-features-001",
        description="powers x^1..x^9 of one design matrix folded into a weighted sum",
    )
)
