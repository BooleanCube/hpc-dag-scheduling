"""trap-roundrobin-001 — heavy and light tasks interleaved to poison index-based assignment.

Size class: medium graph (~64 nodes), alternating (640,640) (3 MiB) and (16,16) matrices.

What it measures: whether assignment depends on anything smarter than the node index. The
graph is 16 independent matmul pairs created in strict heavy, light, heavy, light order, so
in the serialized topological ordering the heavy tasks occupy one parity class of indices.
A round-robin scheduler over an EVEN rank count maps every heavy task to the same residue
classes — with 2 ranks, one rank receives ALL sixteen heavy matmuls (~8.4 GFLOP) and the
other sixteen trivial ones, a worst-case imbalance manufactured from a perfectly balanced
workload. Any scheduler consulting cost hints, history, or even randomizing assignment
avoids the trap entirely. Run at -n 2 and -n 4 (even counts) and compare against -n 3.

Suggested: -n 2, 3, 4 — the even/odd contrast is the diagnostic.
"""

from vmath import Graph, InitNode, Node, emit

outputs: list[Node] = []
for i in range(16):
    heavy_a = InitNode((640, 640), seed=5000 + 4 * i, distribution="normal")
    heavy_b = InitNode((640, 640), seed=5001 + 4 * i, distribution="uniform")
    outputs.append((heavy_a @ heavy_b) * 1e-3)
    light_a = InitNode((16, 16), seed=5002 + 4 * i, distribution="normal")
    light_b = InitNode((16, 16), seed=5003 + 4 * i, distribution="uniform")
    outputs.append(light_a @ light_b)

emit(
    Graph(
        outputs,
        dag_id="trap-roundrobin-001",
        description="heavy/light tasks interleaved by index; index-mod-N assignment collapses",
    )
)
