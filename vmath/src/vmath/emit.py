"""The case-script output contract between ``/cases`` scripts and ``hpcctl submit``.

``hpcctl submit <case.py>`` compiles a case by running the script with the destination path
as its first command-line argument; a case script ends with a single :func:`emit` call and
never chooses its own output location. Run standalone (``python case.py``), the same script
prints the document to stdout, so ``python case.py > dag.json`` and piping both work without
any hpcctl involvement.
"""

import sys
from pathlib import Path
from typing import Any

from vmath.graph import Graph


def emit(graph: Graph, **kwargs: Any) -> None:
    """Serialize a graph to the destination this script was invoked with.

    Writes to the path in ``sys.argv[1]`` when present (the ``hpcctl submit`` calling
    convention), otherwise to stdout.

    Args:
        graph: The closed graph to serialize.
        **kwargs: Forwarded to :meth:`vmath.Graph.serialize` (``renumber``,
            ``include_hints``, ``include_timestamp``).

    Raises:
        CyclicDependencyError: If the graph contains a cycle.
        ValueError: If two nodes would serialize to the same ID.
    """
    if len(sys.argv) > 1:
        graph.to_json(Path(sys.argv[1]), **kwargs)
    else:
        import json

        sys.stdout.write(json.dumps(graph.serialize(**kwargs), indent=2, allow_nan=False) + "\n")
