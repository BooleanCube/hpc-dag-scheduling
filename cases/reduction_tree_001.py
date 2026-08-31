"""reduction-tree-001 — a 64-leaf tournament whose parallelism halves every level.

Size class: large graph (~190 nodes), (256,256) matrices (512 KiB each).

What it measures: shrinking-parallelism management with real data volumes. Sixty-four init
leaves are pairwise multiplied then summed, level after level: 32-wide, 16, 8, 4, 2, 1.
Early levels saturate any instance count; late levels leave most ranks idle while 512 KiB
operands converge. The classic mistakes are visible immediately: keeping ranks "busy" at
level 5 by shipping both operands somewhere third (pure waste), or having scattered level-k
results so evenly that EVERY level-(k+1) op needs a remote operand. A good schedule looks
like a merge plan: pairs co-located one level ahead. Real-world analogue: distributed
aggregation, map-reduce combine phases, tournament reductions in ensemble scoring.

Suggested: -n 2, 4.
"""

from vmath import Graph, InitNode, Node, emit

leaves = [InitNode((256, 256), seed=10000 + i, distribution="normal") for i in range(64)]

layer: list[Node] = [a * b for a, b in zip(leaves[::2], leaves[1::2], strict=True)]
while len(layer) > 1:
    layer = [a + b for a, b in zip(layer[::2], layer[1::2], strict=True)]

emit(
    Graph(
        layer,
        dag_id="reduction-tree-001",
        description="64-leaf multiply/add tournament; width halves per level, transfers grow",
    )
)
