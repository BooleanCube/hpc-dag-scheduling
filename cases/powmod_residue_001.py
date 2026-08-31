"""powmod-residue-001 — twelve modular-exponentiation lanes, deep-sequential inside.

Size class: large graph (~240 nodes), (16384,) vectors (128 KiB each).

What it measures: depth-inside-width scheduling. Each lane computes x^257 mod 10007 by
binary exponentiation with a reduction after every squaring — a ~19-node strictly sequential
subchain (the number-theoretic workload shape: RSA-style exponentiation, hash residues).
Twelve lanes run in parallel, but no lane can be split, so the right schedule assigns whole
lanes to ranks and never migrates one mid-chain (each hop ships 128 KiB onto a serial
dependency path). With 12 lanes at -n 4 the packing is exact; the case pairs with
straggler-tail-001 (which makes packing inexact) and markov-power-001 (one lane, no width).

Suggested: -n 2, 3, 4.
"""

from vmath import Graph, InitNode, Node, emit, powmod


def lane(index: int) -> Node:
    """Build one modular-exponentiation lane.

    Args:
        index: Lane index, used to derive a distinct seed.

    Returns:
        The lane's residue node.
    """
    base = InitNode((16384,), seed=16000 + index, distribution="uniform")
    return powmod(base * 1000.0, 257, 10_007.0, label_prefix=f"lane{index}")


emit(
    Graph(
        [lane(i) for i in range(12)],
        dag_id="powmod-residue-001",
        description="12 independent powmod(x,257,10007) lanes; unsplittable serial subchains",
    )
)
