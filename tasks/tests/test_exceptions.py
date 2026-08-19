"""Smoke tests for the build-time exception hierarchy."""

import pytest

from tasks import (
    CyclicDependencyError,
    DagBuildError,
    DimensionalityError,
    ShapeMismatchError,
    UninitializedNodeError,
)


@pytest.mark.parametrize(
    "exc",
    [
        ShapeMismatchError,
        DimensionalityError,
        CyclicDependencyError,
        UninitializedNodeError,
    ],
)
def test_build_errors_share_a_common_base(exc: type[DagBuildError]) -> None:
    assert issubclass(exc, DagBuildError)


def test_build_error_carries_its_message() -> None:
    with pytest.raises(ShapeMismatchError, match="3x4 vs 5x6"):
        raise ShapeMismatchError("3x4 vs 5x6")
