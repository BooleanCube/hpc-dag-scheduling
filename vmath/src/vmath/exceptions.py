"""Build-time exceptions raised by the DAG builder.

These cover every logical error the Python layer is responsible for catching. The C++
engine handles only runtime physics (OOM, MPI deadlocks, schema parse failures, Slurm
preemption, NaN/Inf), so no DAG that fails one of these checks may reach it.
"""


class DagBuildError(Exception):
    """Base class for every error detected while building a DAG."""


class ShapeMismatchError(DagBuildError):
    """Operand dimensions do not align for the requested operation."""


class DimensionalityError(DagBuildError):
    """Operation was applied to a tensor of the wrong rank."""


class CyclicDependencyError(DagBuildError):
    """Graph contains a cycle and is therefore not a valid DAG."""


class UninitializedNodeError(DagBuildError):
    """An ``init`` node is missing its PRNG seed or shape definition."""
