"""mlp-forward-001 — a four-layer MLP inference pass (batch 256, 784-512-256-64-10).

Size class: medium graph (~60 nodes), realistic ML tensor shapes ((256,784) input, 1.5 MiB).

What it measures: the layer-pipeline pattern that dominates real inference workloads. Layers
are strictly sequential (each matmul consumes the previous activation), but *within* a layer
the sin() activation expands into a wide Maclaurin subgraph of elementwise ops that IS
parallelizable. So the parallelism profile oscillates: narrow (matmul) → wide (activation) →
narrow — a scheduler must repeatedly fan work out and gather it back without letting the
activation cloud drift away from the tensor it feeds. Biases are init tensors of the
activation shape (the wire format has no broadcasting, exactly like a fused/pre-tiled
runtime would see it).

Suggested: -n 1, 2, 4 (limited scaling is EXPECTED; measure where it saturates).
"""

from vmath import Graph, InitNode, Node, emit
from vmath import sin as vsin

BATCH = 256
LAYERS = [784, 512, 256, 64, 10]

x: Node = InitNode((BATCH, LAYERS[0]), seed=7000, distribution="normal", name="batch_input")
for layer, (fan_in, fan_out) in enumerate(zip(LAYERS, LAYERS[1:], strict=False)):
    weights = InitNode((fan_in, fan_out), seed=7100 + layer, distribution="normal")
    bias = InitNode((BATCH, fan_out), seed=7200 + layer, distribution="uniform")
    pre = ((x @ weights) * (1.0 / fan_in)) + bias
    x = vsin(pre, terms=4) if fan_out != LAYERS[-1] else pre

emit(
    Graph(
        [x],
        dag_id="mlp-forward-001",
        description="4-layer MLP forward pass; alternating narrow matmuls and wide activations",
    )
)
