"""xl-vector-stream-001 — eight bandwidth-bound streams of 16 MiB vectors.

Size class: medium graph (~70 nodes), (2097152,) vectors (16 MiB each).

What it measures: the large-VECTOR end of the spectrum (the matrix cases stress FLOPs; these
elementwise pipelines stress memory bandwidth — roughly one byte moved per FLOP executed).
Eight independent streams each run scale, add, and mod passes over a 16 MiB vector before
contracting against a shared weight vector. Streams are identical, so this doubles as a
bandwidth-fairness case; but unlike compute-bound clones, co-scheduling two streams on one
instance does NOT halve their combined time (they contend for the same memory bus), which
breaks schedulers whose cost model assumes task times are additive per rank.

Suggested: -n 1, 2, 4 (watch for sub-linear scaling from bus contention, not scheduling).
"""

from vmath import Graph, InitNode, Node, emit

LENGTH = 1 << 21

weights = InitNode((LENGTH,), seed=12000, distribution="uniform", name="stream_weights")


def stream(index: int) -> Node:
    """Run one elementwise pipeline over a 16 MiB vector and contract it to a scalar.

    Args:
        index: Stream index, used to derive a distinct seed.

    Returns:
        A rank-0 result node.
    """
    data = InitNode((LENGTH,), seed=12100 + index, distribution="normal")
    passes = ((data * 1.0001) + data) % 3.7
    return (passes * 0.5) @ weights


emit(
    Graph(
        [stream(i) for i in range(8)],
        dag_id="xl-vector-stream-001",
        description="8 identical 16 MiB elementwise streams; bandwidth-bound fairness",
    )
)
