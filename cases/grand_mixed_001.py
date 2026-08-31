"""grand-mixed-001 — the final exam: four workload families in one heterogeneous graph.

Size class: extra-large graph (~350 nodes), tensors from 32 bytes to 32 MiB.

What it measures: everything at once, interacting. Four families share one graph:
a bandwidth-heavy (2048,2048) matmul spine; a 16-path Monte-Carlo cloud on (4096,) vectors;
a 6x6 wavefront of (128,128) cells; and exp/sin series expansions applied to a DOWNSTREAM
product of the spine, so the widest elementwise work only unlocks late. No single-policy
scheduler is right for all four regions simultaneously: the spine wants pinning, the cloud
wants spreading, the wavefront wants frontier chasing, and the late series wants the ranks
that the cloud finished with. Family results stay separate outputs so per-family completion
times are readable in the trace. This is the case to rank schedulers by once the focused
cases explain WHY each one wins or loses.

Suggested: -n 2, 4 (and 1, for the sequential reference line).
"""

from vmath import Graph, InitNode, Node, cosh, emit
from vmath import exp as vexp
from vmath import sin as vsin


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


# Family 1: the bandwidth-heavy spine (few nodes, 32 MiB operands).
spine_a = InitNode((2048, 2048), seed=11000, distribution="normal", name="spine_a")
spine_b = InitNode((2048, 2048), seed=11001, distribution="uniform", name="spine_b")
spine = ((spine_a @ spine_b) * 1e-3) % 613.0

# Family 2: the Monte-Carlo cloud (16 identical medium paths, ready immediately).
cloud_weights = InitNode((4096,), seed=11100, distribution="uniform", name="cloud_weights")
cloud_paths: list[Node] = []
for i in range(16):
    shocks = InitNode((4096,), seed=11200 + i, distribution="normal")
    cloud_paths.append((cosh(shocks * 0.01, terms=4, label_prefix=f"c{i}") % 1.0) @ cloud_weights)
cloud = sum_tree(cloud_paths)

# Family 3: the wavefront block (breathing parallelism on 128 KiB cells).
side = 6
grid: list[list[Node]] = []
for i in range(side):
    row: list[Node] = []
    for j in range(side):
        if i == 0 and j == 0:
            cell: Node = InitNode((128, 128), seed=11300, distribution="normal")
        elif i == 0:
            cell = row[j - 1] * 1.01
        elif j == 0:
            cell = grid[i - 1][0] * 0.99
        else:
            cell = grid[i - 1][j] + row[j - 1]
        row.append(cell)
    grid.append(row)
wavefront = grid[side - 1][side - 1]

# Family 4: late-unlocking wide elementwise series on a spine derivative. The scale keeps
# series inputs small enough for the Maclaurin expansions to be numerically meaningful.
late = vsin(spine * 1e-4, terms=5, label_prefix="late_sin") + vexp(
    spine * 1e-4, terms=6, label_prefix="late_exp"
)

emit(
    Graph(
        [spine, cloud, wavefront, late],
        dag_id="grand-mixed-001",
        description="spine + cloud + wavefront + late series; four regimes in one graph",
    )
)
