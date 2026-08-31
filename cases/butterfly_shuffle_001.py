"""butterfly-shuffle-001 — four FFT-style butterfly stages over sixteen lanes.

Size class: large graph (~150 nodes), (4096,) lane vectors (32 KiB each).

What it measures: structured non-local communication. Stage s pairs lane i with lane
i XOR 2^s, so a lane's partner is 1 away, then 2, then 4, then 8 — any static clustering of
lanes onto ranks is perfect for some stages and maximally wrong for others, exactly like an
FFT or hypercube all-reduce. Locality-greedy schedulers thrash: the clusters they build in
stage 1 are torn apart by stage 3. The winning strategy is the textbook one (block lanes so
early stages are rank-internal and only late stages cross), which requires reasoning about
the WHOLE communication pattern rather than one edge at a time.

Suggested: -n 2, 4 (with 16 lanes on 4 ranks, exactly two stages must cross ranks).
"""

from vmath import Graph, InitNode, Node, emit

LANES = 16
STAGES = 4

lanes: list[Node] = [
    InitNode((4096,), seed=10100 + i, distribution="uniform") for i in range(LANES)
]
for stage in range(STAGES):
    stride = 1 << stage
    lanes = [
        (lane + (lanes[i ^ stride] * (1.0 if i & stride else -1.0))) % 8.0
        for i, lane in enumerate(lanes)
    ]

emit(
    Graph(
        lanes,
        dag_id="butterfly-shuffle-001",
        description="4 butterfly stages with stride-doubling partners; anti-clustering pattern",
    )
)
