"""fourier-approx-001 — an eight-harmonic Fourier-style series over one signal vector.

Size class: large graph (~150 nodes), (4096,) signal vectors (32 KiB each).

What it measures: many medium-sized independent subgraphs hanging off one source — the
signal-processing shape (harmonic decomposition, wavelet banks). Each harmonic k scales the
signal by k and expands sin() into a ~17-node Maclaurin subgraph; the eight subgraphs are
mutually independent but internally sequential-ish, giving the scheduler mid-grained units
that are too big to treat as single tasks and too small to partition further. The final
weighted reduction forces the harmonics back together. Distinguishes schedulers that reason
about subgraph granularity from ones that see only individual nodes.

Suggested: -n 2, 4.
"""

from vmath import Graph, InitNode, Node, emit
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


signal = InitNode((4096,), seed=9200, distribution="uniform", name="signal")
harmonics = [
    vsin(signal * (0.1 * k), terms=6, label_prefix=f"h{k}") * (1.0 / k) for k in range(1, 9)
]

emit(
    Graph(
        [sum_tree(harmonics)],
        dag_id="fourier-approx-001",
        description="8 sin() harmonic subgraphs over one signal, folded into a weighted sum",
    )
)
