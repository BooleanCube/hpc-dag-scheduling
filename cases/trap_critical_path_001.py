"""trap-critical-path-001 — a long expensive spine buried under a cloud of cheap work.

Size class: medium graph (~130 nodes), spine on (768,768) matrices (4.5 MiB each).

What it measures: critical-path awareness. The spine is matpow-style: eight strictly
sequential (768,768) matmuls (~0.9 GFLOP each) — nothing downstream of it can start early,
so the makespan floor IS the spine. Alongside it, 40 independent (32,32) matmuls are ready
immediately and cost essentially nothing. A greedy scheduler that prioritizes "most ready
work first" (or fills ranks breadth-first) happily churns through the cheap cloud while the
spine's next link waits in queue — every such delay adds directly to the makespan. The
correct behaviour is to treat the spine as sacred and drip the cloud into idle cycles.

Suggested: -n 2, 4 (makespan should pin to the spine; excess over it is scheduling error).
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


spine = InitNode((768, 768), seed=3000, distribution="normal", name="spine_seed")
step = InitNode((768, 768), seed=3001, distribution="uniform", name="spine_step")
head: Node = spine
for _ in range(8):
    head = head @ step
head = head * 1e-3

cloud = []
for i in range(40):
    a = InitNode((32, 32), seed=3100 + 2 * i, distribution="normal")
    b = InitNode((32, 32), seed=3101 + 2 * i, distribution="uniform")
    cloud.append(a @ b)

emit(
    Graph(
        [head, sum_tree(cloud)],
        dag_id="trap-critical-path-001",
        description="8-deep expensive spine plus 40 cheap distractor tasks; starves greedy",
    )
)
