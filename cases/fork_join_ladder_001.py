"""fork-join-ladder-001 — six bulk-synchronous stages where stragglers compound.

Size class: medium graph (~160 nodes), (192,192) matrices (288 KiB each).

What it measures: barrier amplification. Each stage forks the previous join into eight
parallel matmul branches, then joins them again — the bulk-synchronous pattern of iterative
solvers and data-parallel training steps. Nothing in stage k+1 can start before stage k's
join, so ONE straggler per stage adds its full delay to the makespan, six times over. A
scheduler that is merely "usually balanced" (balanced in expectation, with variance) gets
punished sixfold; what wins is consistently tight stage packing and placing each join where
most of its operands already live. Compare per-stage times in the trace: they should be flat.

Suggested: -n 2, 4 (8 branches per stage: both divide evenly; imbalance is pure scheduling).
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


state: Node = InitNode((192, 192), seed=13000, distribution="normal", name="stage0")
for stage in range(6):
    branches = []
    for branch in range(8):
        mixer = InitNode((192, 192), seed=13100 + 8 * stage + branch, distribution="uniform")
        branches.append((state @ mixer) * (1.0 / 192.0))
    state = sum_tree(branches)

emit(
    Graph(
        [state],
        dag_id="fork-join-ladder-001",
        description="6 fork-join stages of 8 matmuls each; stage stragglers compound sixfold",
    )
)
