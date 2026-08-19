"""Tests for cycle detection.

``CyclicDependencyError`` is the only member of the ``DagBuildError`` family that fires at
serialization rather than at the offending expression, because acyclicity is the one property
that is not local to a single expression.

The recursion tests matter more than they look: the reachability walk runs *before* cycle
detection and therefore meets cyclic graphs head-on. A recursive walk would raise
``RecursionError`` and the user would never see the error we promised them.
"""

from pathlib import Path

import pytest

from tasks import Graph, InitNode, Node
from tasks.exceptions import CyclicDependencyError, DagBuildError
from tasks.graph import _find_cycle


def _self_loop() -> Node:
    """Build a scale node whose only operand is itself.

    Returns:
        The self-referential node.
    """
    scale = InitNode((2,), seed=1) * 2.0
    scale.rewire(0, scale)
    return scale


def _two_cycle() -> tuple[Node, Node]:
    """Build a two-node cycle by rewiring the first scale onto the second.

    Returns:
        The ``(first, second)`` pair, each depending on the other.
    """
    first = InitNode((2,), seed=1) * 2.0
    second = first * 3.0
    first.rewire(0, second)
    return first, second


class TestCycleDetection:
    def test_self_loop_raises(self) -> None:
        with pytest.raises(CyclicDependencyError):
            Graph([_self_loop()], dag_id="loop").serialize()

    def test_two_node_cycle_raises(self) -> None:
        _, second = _two_cycle()
        with pytest.raises(CyclicDependencyError):
            Graph([second], dag_id="cycle").serialize()

    def test_message_names_both_nodes(self) -> None:
        first, second = _two_cycle()
        with pytest.raises(CyclicDependencyError) as excinfo:
            Graph([second], dag_id="cycle").serialize()
        message = str(excinfo.value)
        assert message.startswith("Cyclic dependency detected: ")
        assert first.display_id in message
        assert second.display_id in message

    def test_message_renders_a_closed_path(self) -> None:
        """The first and last entries are the same node, so the cycle reads as a loop."""
        _, second = _two_cycle()
        with pytest.raises(CyclicDependencyError) as excinfo:
            Graph([second], dag_id="cycle").serialize()
        path = str(excinfo.value).removeprefix("Cyclic dependency detected: ").split(" -> ")
        assert len(path) >= 2
        assert path[0] == path[-1]

    def test_raised_from_topological_order(self) -> None:
        _, second = _two_cycle()
        with pytest.raises(CyclicDependencyError):
            Graph([second], dag_id="cycle").topological_order()

    def test_raised_from_validate(self) -> None:
        _, second = _two_cycle()
        with pytest.raises(CyclicDependencyError):
            Graph([second], dag_id="cycle").validate()

    def test_raised_from_to_json(self, tmp_path: Path) -> None:
        _, second = _two_cycle()
        with pytest.raises(CyclicDependencyError):
            Graph([second], dag_id="cycle").to_json(tmp_path / "dag.json")

    def test_is_a_dag_build_error(self) -> None:
        _, second = _two_cycle()
        with pytest.raises(DagBuildError):
            Graph([second], dag_id="cycle").serialize()

    def test_three_node_cycle(self) -> None:
        first = InitNode((2,), seed=1) * 2.0
        second = first * 3.0
        third = second * 4.0
        first.rewire(0, third)
        with pytest.raises(CyclicDependencyError) as excinfo:
            Graph([third], dag_id="cycle").serialize()
        for node in (first, second, third):
            assert node.display_id in str(excinfo.value)


class TestCycleIsScopedToReachableNodes:
    def test_unreachable_cycle_is_ignored(self) -> None:
        """Dead-code elimination runs first, so a cycle nobody depends on is not an error."""
        live = InitNode((2,), seed=1)
        _, _unreachable = _two_cycle()
        Graph([live], dag_id="live-only").serialize()

    def test_cycle_reachable_from_one_of_several_outputs_still_raises(self) -> None:
        live = InitNode((2,), seed=1)
        _, second = _two_cycle()
        with pytest.raises(CyclicDependencyError):
            Graph([live, second], dag_id="mixed").serialize()


class TestFindCycleDirectly:
    """``_find_cycle`` is a general DFS, exercised here beyond what Kahn's remainder reaches.

    Its call site only ever passes a node set that is guaranteed to contain a cycle, so the
    backtracking and already-visited paths are unreachable from ``serialize``. Testing the
    function directly covers them without weakening the code to a ``pragma: no cover``.
    """

    def test_returns_empty_for_an_acyclic_input(self) -> None:
        source = InitNode((2,), seed=1)
        first = source * 2.0
        second = first * 3.0
        nodes = [second, first, source]
        members = {id(node) for node in nodes}
        assert _find_cycle(nodes, members) == []

    def test_ignores_operands_outside_the_member_set(self) -> None:
        source = InitNode((2,), seed=1)
        scaled = source * 2.0
        assert _find_cycle([scaled], {id(scaled)}) == []

    def test_finds_a_cycle_among_extra_acyclic_nodes(self) -> None:
        first, second = _two_cycle()
        downstream = second * 5.0
        nodes = [downstream, first, second]
        members = {id(node) for node in nodes}
        cycle = _find_cycle(nodes, members)
        assert cycle[0] is cycle[-1]
        assert {id(node) for node in cycle} == {id(first), id(second)}


class TestNoRecursionError:
    """A long cycle must produce our error, not a stack overflow."""

    CHAIN = 3000

    def test_long_cycle_raises_cyclic_dependency_not_recursion_error(self) -> None:
        head = InitNode((2,), seed=1) * 1.0
        tail = head
        for _ in range(self.CHAIN):
            tail = tail * 1.0
        head.rewire(0, tail)
        with pytest.raises(CyclicDependencyError):
            Graph([tail], dag_id="long-cycle").serialize()

    def test_long_acyclic_chain_serializes(self) -> None:
        tail: Node = InitNode((2,), seed=1)
        for _ in range(self.CHAIN):
            tail = tail * 1.0
        document = Graph([tail], dag_id="long-chain").serialize(include_timestamp=False)
        assert len(document["nodes"]) == self.CHAIN + 1

    def test_wide_diamond_of_shared_nodes(self) -> None:
        """Repeated reuse must not blow up the walk or double-count in-degrees."""
        source = InitNode((2,), seed=1)
        total: Node = source
        for _ in range(500):
            total = total + source
        document = Graph([total], dag_id="wide").serialize(include_timestamp=False)
        assert len(document["nodes"]) == 501
