"""markov-power-001 — a transition matrix raised to the 32nd power, twice (f64 and f32).

Size class: tiny graph (~16 nodes), (512,512) matrices (2 MiB / 1 MiB).

What it measures: the hard scaling FLOOR. matpow expands by binary exponentiation into five
strictly sequential (512,512) matmuls — there is no parallelism to find inside a chain, and
the two dtype variants are the only concurrent work. Adding instances beyond 2 CANNOT help;
a scheduler that moves a chain's intermediate between ranks only adds a 2 MiB transfer per
hop to a critical path. Expect flat runtimes across -n; any slope is scheduler-inflicted.
Real-world analogue: Markov-chain steady-state estimation, graph reachability powers.

Suggested: -n 1, 2, 4 (the 4-instance run should match the 2-instance run).
"""

from vmath import Graph, InitNode, emit, matpow

p64 = InitNode((512, 512), seed=9300, distribution="uniform", name="p64")
p32 = InitNode((512, 512), seed=9301, distribution="uniform", dtype="float32", name="p32")

emit(
    Graph(
        [
            matpow(p64 * (1.0 / 512.0), 32, label_prefix="mp64"),
            matpow(p32 * (1.0 / 512.0), 32, label_prefix="mp32"),
        ],
        dag_id="markov-power-001",
        description="two sequential matpow(P,32) chains; a scaling floor no scheduler can beat",
    )
)
