"""wide-skewed-001 — independent chains whose costs span four orders of magnitude.

Size class: medium graph (~64 nodes), tensors from (16,16) up to (1024,1024) (8 MiB).

What it measures: load balancing under heavy cost skew. Same topology as wide-uniform-001 —
independent 4-node chains, all ready at once — but one chain is a (1024,1024) matmul pipeline
(~2 GFLOP), a few are medium, and most are nearly free. A single 1024-cube matmul costs about
4000x a 16-cube one, so round-robin's "one task each" is a catastrophe: whichever rank draws
the giant finishes last while the rest idle, and the makespan equals the giant regardless of
-n unless the scheduler co-locates the *small* work away from the giant's rank. Cost-model
quality (the `est_flops` hints, or measured history) is exactly what this case rewards.

Suggested: -n 2, 4 (a good scheduler approaches the giant-chain floor; a bad one exceeds it).
"""

from vmath import Graph, InitNode, Node, emit

EXTENTS = [1024, 512, 512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16, 16, 16, 16]


def chain(extent: int, index: int) -> Node:
    """Build one independent 3-matmul chain of square matrices.

    Args:
        extent: Square matrix extent, the knob that sets this chain's cost.
        index: Chain index, used only to derive distinct seeds.

    Returns:
        The head node of the chain.
    """
    a = InitNode((extent, extent), seed=2000 + 3 * index, distribution="normal")
    b = InitNode((extent, extent), seed=2001 + 3 * index, distribution="uniform")
    return ((a @ b) @ a) * (1.0 / extent)


emit(
    Graph(
        [chain(extent, i) for i, extent in enumerate(EXTENTS)],
        dag_id="wide-skewed-001",
        description="16 independent chains, costs skewed ~4000x; punishes cost-blind balancing",
    )
)
