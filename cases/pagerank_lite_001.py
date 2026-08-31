"""pagerank-lite-001 — twelve damped power-iteration steps against one hot 32 MiB matrix.

Size class: small graph (~40 nodes), one (2048,2048) matrix (32 MiB) plus (2048,) vectors.

What it measures: hot-operand placement. Every one of the twelve iterations multiplies the
CURRENT rank vector (16 KiB) by the SAME link matrix (32 MiB). A scheduler that pins the
matrix once and routes the tiny vector to it does twelve 16 KiB transfers; one that treats
each step independently and moves work to "free" ranks re-ships 32 MiB per bounce — a
2000x difference in bytes moved for identical FLOPs. The damping add also consumes a fixed
teleport vector, a second (small) resident operand. Real-world analogue: PageRank/eigenvector
centrality, and any iterative solver with a fixed system matrix.

Suggested: -n 1, 2, 4 (more instances should NOT help; watch bytes-moved metrics).
"""

from vmath import Graph, InitNode, Node, emit

matrix = InitNode((2048, 2048), seed=9400, distribution="uniform", name="link_matrix")
teleport = InitNode((2048,), seed=9401, distribution="ones", name="teleport")

rank: Node = InitNode((2048,), seed=9402, distribution="uniform", name="rank0")
for _ in range(12):
    rank = ((rank @ matrix) * (0.85 / 2048.0)) + (teleport * (0.15 / 2048.0))

emit(
    Graph(
        [rank],
        dag_id="pagerank-lite-001",
        description="12 damped power-iteration steps reusing one 32 MiB matrix; hot-operand test",
    )
)
