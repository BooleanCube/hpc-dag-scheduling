"""wavefront-grid-001 — a 10x10 dependency lattice with breathing parallelism.

Size class: large graph (~120 nodes), (128,128) matrices (128 KiB each).

What it measures: frontier tracking. Cell (i,j) needs (i-1,j) and (i,j-1), so the ready set
sweeps anti-diagonals: width 1, 2, ... 10, ... 2, 1. No static assignment is right for the
whole run — a scheduler must widen onto more ranks mid-graph and contract back without
stranding cells away from both parents (each wrong placement costs a 128 KiB transfer on the
critical frontier). Round-robin fragments the diagonals; strict locality serializes them.
Real-world analogue: dynamic-programming alignments (Smith-Waterman), triangular solves,
LU-factorization panels.

Suggested: -n 2, 4 (peak frontier width is 10; speedup should saturate near min(-n, ~4)).
"""

from vmath import Graph, InitNode, Node, emit

SIDE = 10

grid: list[list[Node]] = []
for i in range(SIDE):
    row: list[Node] = []
    for j in range(SIDE):
        if i == 0 and j == 0:
            cell: Node = InitNode((128, 128), seed=9800, distribution="normal", name="corner")
        elif i == 0:
            cell = row[j - 1] * 1.01
        elif j == 0:
            cell = grid[i - 1][0] * 0.99
        else:
            cell = grid[i - 1][j] + row[j - 1]
        row.append(cell)
    grid.append(row)

emit(
    Graph(
        [grid[SIDE - 1][SIDE - 1]],
        dag_id="wavefront-grid-001",
        description="10x10 anti-diagonal wavefront; parallelism ramps 1..10..1 across the run",
    )
)
