"""mlp-ensemble-001 — six MLP forward passes sharing one input batch.

Size class: large graph (~360 nodes), (128,784) shared input, per-member 2-3 MiB weights.

What it measures: shared-source fan-out at realistic scale — the deep-ensemble / multi-head
serving pattern. One input tensor feeds six structurally identical but independently seeded
MLPs, so after a single init the graph is six heavy parallel pipelines. The interesting
scheduling question is what happens to the SHARED input: replicate it once to every rank up
front (right) or re-request it lazily per consumer (wrong, six transfers of the same bytes).
Members are identical in cost, so imbalance observed here is scheduler-inflicted, not
workload-inflicted — an honest fairness probe at a scale where mistakes are visible.

Suggested: -n 2, 3, 4 (6 members: -n 3 divides evenly, -n 4 does not — compare).
"""

from vmath import Graph, InitNode, Node, emit
from vmath import sin as vsin

BATCH = 128
LAYERS = [784, 256, 64, 10]


def member(index: int, batch_input: Node) -> Node:
    """Build one ensemble member's forward pass over the shared input.

    Args:
        index: Member index, used to derive distinct weight seeds.
        batch_input: The shared input batch node.

    Returns:
        The member's logits node.
    """
    x = batch_input
    for layer, (fan_in, fan_out) in enumerate(zip(LAYERS, LAYERS[1:], strict=False)):
        weights = InitNode((fan_in, fan_out), seed=8000 + 10 * index + layer)
        bias = InitNode((BATCH, fan_out), seed=8500 + 10 * index + layer)
        pre = ((x @ weights) * (1.0 / fan_in)) + bias
        x = vsin(pre, terms=4) if fan_out != LAYERS[-1] else pre
    return x


shared = InitNode((BATCH, LAYERS[0]), seed=8999, distribution="normal", name="shared_batch")

emit(
    Graph(
        [member(i, shared) for i in range(6)],
        dag_id="mlp-ensemble-001",
        description="6-member MLP ensemble over one shared input; probes source replication",
    )
)
