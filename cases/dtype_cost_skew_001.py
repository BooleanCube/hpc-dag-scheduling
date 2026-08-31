"""dtype-cost-skew-001 — identical shapes, different dtypes: a trap for shape-based costing.

Size class: medium graph (~50 nodes), (768,768) matrices in float64 (4.5 MiB) and float32.

What it measures: cost-model fidelity beyond tensor extents. Twelve independent matmul
chains have IDENTICAL shapes and topology; six run in float64 and six in float32. On real
hardware the float32 chains finish meaningfully faster (half the bytes through the memory
system, twice the SIMD width), so a scheduler that estimates cost from shape alone sees
twelve equal tasks and packs them 6+6 — systematically loading whichever rank drew the
float64 half. The `est_flops` hints do not distinguish dtype either, making this a probe of
measured-history schedulers versus purely static ones. Placement is otherwise free of traps:
the chains are independent and uniform within each dtype class.

Suggested: -n 2, 4.
"""

from vmath import Graph, InitNode, Node, emit

outputs: list[Node] = []
for i in range(6):
    a64 = InitNode((768, 768), seed=14000 + 4 * i, distribution="normal")
    b64 = InitNode((768, 768), seed=14001 + 4 * i, distribution="uniform")
    outputs.append((a64 @ b64) * 1e-3)
    a32 = InitNode((768, 768), seed=14002 + 4 * i, distribution="normal", dtype="float32")
    b32 = InitNode((768, 768), seed=14003 + 4 * i, distribution="uniform", dtype="float32")
    outputs.append((a32 @ b32) * 1e-3)

emit(
    Graph(
        outputs,
        dag_id="dtype-cost-skew-001",
        description="12 shape-identical chains, half f32 half f64; breaks shape-only cost models",
    )
)
