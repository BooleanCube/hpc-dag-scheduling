"""imbalanced-diamond-001 — diamonds whose two arms differ 12x in depth.

Size class: medium graph (~130 nodes), (256,256) matrices (512 KiB each).

What it measures: join-aware prioritization. Each of eight diamonds forks from a shared top
into a 12-node sequential arm and a 1-node arm, then joins with an add. The join can only
fire when the LONG arm finishes, so the short arm's result sits occupying memory while its
sibling grinds. A scheduler that treats both arms as equal-priority ready work (greedy FIFO)
delays long arms behind short ones across diamonds and serializes the joins; the right move
is to start every long arm first and fold the short arms in wherever. This is the classic
b-level/upward-rank test from the HEFT literature in its smallest honest form.

Suggested: -n 2, 4.
"""

from vmath import Graph, InitNode, Node, emit


def diamond(index: int) -> Node:
    """Build one diamond with a 12-deep left arm and a 1-node right arm.

    Args:
        index: Diamond index, used to derive distinct seeds.

    Returns:
        The join node of the diamond.
    """
    top_a = InitNode((256, 256), seed=6000 + 2 * index, distribution="normal")
    top_b = InitNode((256, 256), seed=6001 + 2 * index, distribution="uniform")
    top = top_a @ top_b
    long_arm = top
    for _ in range(6):
        long_arm = (long_arm * 0.999) + top
    short_arm = top * 0.5
    return long_arm + short_arm


emit(
    Graph(
        [diamond(i) for i in range(8)],
        dag_id="imbalanced-diamond-001",
        description="8 diamonds with 12x asymmetric arms; rewards upward-rank prioritization",
    )
)
