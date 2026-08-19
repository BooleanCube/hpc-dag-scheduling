"""The :class:`Graph` view: reachability, topological ordering, and serialization.

A graph is a *view* over nodes rather than a container they belong to. It is constructed from
its output nodes and discovers everything else by walking backwards through operand
references, which is what lets ``a + b`` work with no graph in scope.

This module owns the only :class:`~tasks.exceptions.CyclicDependencyError` in the library:
acyclicity is the one mathematical property that is not local to a single expression, so it
cannot be checked when the offending edge is written.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final

from tasks.dtypes import JsonDict
from tasks.exceptions import CyclicDependencyError
from tasks.node import Node

SCHEMA_VERSION: Final[str] = "1.2.0"
"""Version of ``/shared/dag_schema.json`` this builder targets.

1.1.0 widened ``shape`` to allow the empty array, which is what makes a vector-vector
``dot_product`` (a rank-0 result) representable on the wire. 1.2.0 added the ``multiply`` and
``mod`` ops and lifted the rank-1 floor on ``init``, whose justification -- that nothing could
consume a rank-0 source -- the elementwise ops invalidated.
"""

DAG_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
"""Schema pattern for ``metadata.dag_id``."""


def _generator() -> str:
    """Return the ``metadata.generator`` string identifying this builder.

    Returns:
        ``"tasks-builder <version>"``, falling back to ``"unknown"`` for an uninstalled
        source checkout.
    """
    try:
        return f"tasks-builder {version('tasks')}"
    except PackageNotFoundError:  # pragma: no cover - only in an uninstalled checkout
        return "tasks-builder unknown"


class Graph:
    """A closed, serializable mathematical DAG.

    A graph is constructed from its output nodes and discovers every contributing node by
    walking backwards through operand references. Nodes that were built but do not reach an
    output are excluded from serialization -- dead-code elimination is intentional and lets a
    task script explore alternatives without polluting the emitted DAG.
    """

    def __init__(
        self,
        outputs: Sequence[Node],
        *,
        dag_id: str,
        description: str | None = None,
    ) -> None:
        """Close a graph over the given output nodes.

        Args:
            outputs: Nodes whose tensors the engine must materialize. Must be non-empty and
                free of duplicates.
            dag_id: Stable identifier matching ``^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$``.
            description: Optional human-readable description for the metadata block.

        Raises:
            TypeError: If ``outputs`` contains something that is not a :class:`Node`.
            ValueError: If ``outputs`` is empty or contains duplicates, or ``dag_id`` does not
                match the schema pattern.
        """
        resolved = tuple(outputs)
        if not resolved:
            raise ValueError("outputs must contain at least one node")
        for candidate in resolved:
            if not isinstance(candidate, Node):
                raise TypeError(
                    f"outputs must contain Node instances, got {type(candidate).__name__}"
                )
        seen: set[int] = set()
        for candidate in resolved:
            if id(candidate) in seen:
                raise ValueError(
                    f"duplicate output node {candidate.display_id!r}; "
                    "each output must appear exactly once"
                )
            seen.add(id(candidate))
        if not DAG_ID_PATTERN.match(dag_id):
            raise ValueError(f"dag_id must match {DAG_ID_PATTERN.pattern!r}, got {dag_id!r}")

        self._outputs = resolved
        self._dag_id = dag_id
        self._description = description

    @property
    def outputs(self) -> tuple[Node, ...]:
        """Nodes whose tensors the engine must materialize."""
        return self._outputs

    @property
    def dag_id(self) -> str:
        """Stable identifier for this DAG instance."""
        return self._dag_id

    @property
    def description(self) -> str | None:
        """Free-text description recorded in the metadata block."""
        return self._description

    def nodes(self) -> tuple[Node, ...]:
        """Return every node reachable from the outputs, in construction order.

        Returns:
            The reachable nodes sorted by construction sequence. Unreachable nodes are
            excluded.
        """
        return tuple(sorted(self._reachable(), key=lambda node: node._seq))

    def _reachable(self) -> list[Node]:
        """Collect every node reachable from the outputs via an iterative walk.

        The walk runs *before* cycle detection and will therefore encounter cyclic graphs. It
        uses an explicit stack rather than recursion: a recursive walk would hit
        ``RecursionError`` on a cycle and the caller would never see the
        :class:`~tasks.exceptions.CyclicDependencyError` we promised them.

        Returns:
            The reachable nodes, deduplicated by identity, in discovery order.
        """
        visited: dict[int, Node] = {}
        stack: list[Node] = list(self._outputs)
        while stack:
            node = stack.pop()
            if id(node) in visited:
                continue
            visited[id(node)] = node
            stack.extend(node.inputs)
        return list(visited.values())

    def topological_order(self) -> list[Node]:
        """Return all reachable nodes ordered so every operand precedes its consumer.

        Uses Kahn's algorithm, seeding the queue with in-degree-zero nodes in ascending
        construction order. Deterministic ordering matters: it is what makes serialized output
        diffable across runs, and therefore what makes the research baseline reproducible.

        Returns:
            The reachable nodes in a deterministic topological order.

        Raises:
            CyclicDependencyError: If the reachable subgraph contains a cycle.
        """
        reachable = self._reachable()
        members = {id(node) for node in reachable}

        # Edges are counted with multiplicity: `c + c` gives that AddNode in-degree 2, and
        # emitting `c` decrements it twice, so the arithmetic stays balanced.
        indegree: dict[int, int] = {id(node): len(node.inputs) for node in reachable}
        consumers: dict[int, list[Node]] = {id(node): [] for node in reachable}
        for node in reachable:
            for operand in node.inputs:
                consumers[id(operand)].append(node)
        for consumer_list in consumers.values():
            consumer_list.sort(key=lambda node: node._seq)

        ready = deque(
            sorted(
                (node for node in reachable if indegree[id(node)] == 0),
                key=lambda node: node._seq,
            )
        )
        order: list[Node] = []
        while ready:
            node = ready.popleft()
            order.append(node)
            for consumer in consumers[id(node)]:
                indegree[id(consumer)] -= 1
                if indegree[id(consumer)] == 0:
                    ready.append(consumer)

        if len(order) < len(reachable):
            emitted = {id(node) for node in order}
            remaining = [node for node in reachable if id(node) not in emitted]
            cycle = _find_cycle(remaining, members)
            path = " -> ".join(node.display_id for node in cycle)
            raise CyclicDependencyError(f"Cyclic dependency detected: {path}")
        return order

    def _ordered(self) -> list[Node]:
        """Return the topological order with every node's shape and dtype refreshed.

        :meth:`~tasks.node.Node.rewire` re-infers the node it edits, but nodes hold no
        back-references to their consumers, so a downstream node keeps the shape and dtype it
        was built with. Refreshing in topological order means every operand is up to date
        before its consumer re-reads it; without this pass a rewired graph can emit a
        ``scale`` declaring ``output_shape`` ``[3, 5]`` above a ``dot_product`` that now
        produces ``[3, 7]`` -- a schema-valid but mathematically impossible document, which is
        exactly what the engine is promised never to receive.

        Returns:
            The reachable nodes in deterministic topological order.

        Raises:
            CyclicDependencyError: If the reachable subgraph contains a cycle.
            ShapeMismatchError: If refreshed operands no longer align.
            DimensionalityError: If a refreshed operand has the wrong rank.
        """
        order = self.topological_order()
        for node in order:
            node._refresh()
        return order

    def validate(self) -> None:
        """Run every whole-graph check without producing output.

        Raises:
            CyclicDependencyError: If the graph contains a cycle.
            ShapeMismatchError: If a rewired operand left a consumer misaligned.
            DimensionalityError: If a rewired operand left a consumer at the wrong rank.
            ValueError: If two nodes would serialize to the same ID.
        """
        order = self._ordered()
        self._assign_ids(order, renumber=True)

    def _assign_ids(self, order: Sequence[Node], *, renumber: bool) -> dict[int, str]:
        """Resolve the canonical serialized ID of every node.

        Args:
            order: Nodes in topological order.
            renumber: Assign ``{op}_{index}`` IDs from topological position. User-supplied
                names always win regardless.

        Returns:
            A mapping from node identity to serialized ID.

        Raises:
            ValueError: If two nodes resolve to the same ID.
        """
        ids: dict[int, str] = {}
        used: dict[str, Node] = {}
        for index, node in enumerate(order):
            if node.name is not None:
                node_id = node.name
            elif renumber:
                node_id = f"{node.op}_{index}"
            else:
                node_id = node._provisional_id
            if node_id in used:
                raise ValueError(
                    f"duplicate node ID {node_id!r}: assigned to both "
                    f"{used[node_id]!r} and {node!r}"
                )
            used[node_id] = node
            ids[id(node)] = node_id
        return ids

    def _metadata(self, *, include_timestamp: bool) -> JsonDict:
        """Build the metadata block.

        Args:
            include_timestamp: Whether to emit ``created_at``.

        Returns:
            A mapping conforming to the schema's ``metadata`` definition, with absent optional
            fields omitted rather than set to ``null``.
        """
        meta: JsonDict = {
            "schema_version": SCHEMA_VERSION,
            "dag_id": self._dag_id,
            "ordering": "topological",
        }
        if include_timestamp:
            meta["created_at"] = datetime.now(UTC).isoformat()
        if self._description is not None:
            meta["description"] = self._description
        meta["generator"] = _generator()
        return meta

    def serialize(
        self,
        *,
        renumber: bool = True,
        include_hints: bool = True,
        include_timestamp: bool = True,
    ) -> JsonDict:
        """Render the graph as a schema-conformant document.

        Args:
            renumber: Assign canonical ``{op}_{index}`` IDs in topological order. User-supplied
                names are always preserved. Disable to keep provisional construction-time IDs.
            include_hints: Emit ``hints.est_flops`` on every node.
            include_timestamp: Emit ``metadata.created_at``. Pass ``False`` for byte-stable
                output.

        Returns:
            A mapping with ``metadata``, ``nodes``, and ``outputs`` keys.

        Raises:
            CyclicDependencyError: If the graph contains a cycle.
            ShapeMismatchError: If a rewired operand left a consumer misaligned.
            DimensionalityError: If a rewired operand left a consumer at the wrong rank.
            ValueError: If two nodes would serialize to the same ID.
        """
        order = self._ordered()
        ids = self._assign_ids(order, renumber=renumber)
        return {
            "metadata": self._metadata(include_timestamp=include_timestamp),
            "nodes": [
                node.to_dict(
                    ids[id(node)],
                    [ids[id(operand)] for operand in node.inputs],
                    include_hints=include_hints,
                )
                for node in order
            ],
            "outputs": [ids[id(node)] for node in self._outputs],
        }

    def to_json(self, path: Path, *, indent: int = 2, **kwargs: Any) -> None:
        """Write the serialized graph to disk as UTF-8 JSON.

        Args:
            path: Destination file path.
            indent: Indentation passed to :func:`json.dumps`.
            **kwargs: Forwarded to :meth:`serialize`.

        Raises:
            CyclicDependencyError: If the graph contains a cycle.
            ValueError: If two nodes would serialize to the same ID, or if the document
                contains a non-finite float.
        """
        document = self.serialize(**kwargs)
        # allow_nan=False: Python's default emits bare NaN/Infinity literals, which are
        # invalid JSON and would fail the schema's `number` type on the C++ side. ScaleNode
        # already rejects non-finite factors at construction, so this is defence in depth.
        text = json.dumps(document, indent=indent, allow_nan=False)
        path.write_text(text + "\n", encoding="utf-8")


def _find_cycle(remaining: Sequence[Node], members: set[int]) -> list[Node]:
    """Recover one concrete cycle from the nodes Kahn's algorithm could not emit.

    "There is a cycle somewhere" is not an actionable error, so the message names the nodes
    involved. The search is an iterative three-colour DFS; recursion would risk a
    ``RecursionError`` on precisely the input this function exists to describe.

    Args:
        remaining: Nodes left unemitted by the topological sort. Every cycle is contained in
            this set.
        members: Identities of the reachable nodes, used to ignore edges leaving the set.

    Returns:
        Nodes forming a cycle, with the entry node repeated at the end, each followed by the
        operand it depends on. Empty if no cycle is found, which should be unreachable.
    """
    pool = {id(node) for node in remaining} & members
    colour: dict[int, int] = {}  # 0/absent = unvisited, 1 = on stack, 2 = finished
    for start in sorted(remaining, key=lambda node: node._seq):
        # Unreachable today: starts are visited in ascending construction order and operand
        # edges strictly decrease it, so a start can never have been coloured by an earlier
        # walk -- except via a cycle, which returns before the loop advances. Kept because
        # dropping it would silently break this DFS if the iteration order ever changes.
        if colour.get(id(start), 0) != 0:  # pragma: no cover
            continue
        path: list[Node] = [start]
        colour[id(start)] = 1
        stack: list[tuple[Node, int]] = [(start, 0)]
        while stack:
            node, cursor = stack[-1]
            operands = [operand for operand in node.inputs if id(operand) in pool]
            if cursor < len(operands):
                stack[-1] = (node, cursor + 1)
                nxt = operands[cursor]
                state = colour.get(id(nxt), 0)
                if state == 1:
                    entry = next(index for index, seen in enumerate(path) if id(seen) == id(nxt))
                    return [*path[entry:], nxt]
                if state == 0:
                    colour[id(nxt)] = 1
                    path.append(nxt)
                    stack.append((nxt, 0))
            else:
                stack.pop()
                colour[id(node)] = 2
                path.pop()
    return []
