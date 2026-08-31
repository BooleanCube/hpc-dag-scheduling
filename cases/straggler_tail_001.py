"""straggler-tail-001 — seventeen equal tasks: the off-by-one bin-packing probe.

Size class: small graph (~51 nodes), (512,512) matrices (2 MiB each).

What it measures: remainder handling. Seventeen identical independent matmul chains do not
divide evenly by 2, 3, or 4 — some rank always draws one extra task, and the makespan is
ceil(17 / n) task-times NO MATTER how clever the scheduler is. That makes the IDEAL runtime
exactly computable (9, 6, 5 task-times at n = 2, 3, 4), so this case turns scheduling
quality into a single crisp ratio: measured makespan over the ceil bound. It also exposes
schedulers that react badly to the tail — e.g. migrating the final task mid-flight (pointless
2 MiB shuffling) instead of just running it where it sits.

Suggested: -n 2, 3, 4 — compare each run against its ceil(17/n) bound.
"""

from vmath import Graph, InitNode, Node, emit


def chain(index: int) -> Node:
    """Build one of the seventeen identical matmul chains.

    Args:
        index: Chain index, used to derive distinct seeds.

    Returns:
        The head node of the chain.
    """
    a = InitNode((512, 512), seed=15000 + 2 * index, distribution="normal")
    b = InitNode((512, 512), seed=15001 + 2 * index, distribution="uniform")
    return ((a @ b) * (1.0 / 512.0)) % 5.0


emit(
    Graph(
        [chain(i) for i in range(17)],
        dag_id="straggler-tail-001",
        description="17 identical tasks; makespan bound is ceil(17/n), remainder is the test",
    )
)
