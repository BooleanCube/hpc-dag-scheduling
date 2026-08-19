"""Evaluate serialized composite expansions with numpy and compare against reference functions.

This is the test that proves the expansions compute what they claim rather than merely having the
right node count. It works over the *serialized document*, not the node objects, so it also checks
that every payload the wire carries is the one the maths needs.

``init`` values come from an ``overrides`` mapping keyed by node ID, so the test never depends on
reproducing the engine's PRNG -- only ``ones`` and ``zeros`` are reproducible across languages, and
substitution sidesteps the issue entirely.

**Tolerances are computed, not guessed.** Three traps make a naive fixed tolerance flaky:

1. *Truncation error dominates for few terms.* ``exp(x, terms=10)`` at ``|x| = 0.5`` is off by
   2.8e-10 -- that is the series remainder, not a bug, and ``atol=1e-12`` would fail.
2. *Round-off dominates for many terms.* ``sin(x, terms=10)`` has a truncation bound of 9.3e-26,
   far below float64 epsilon, but an actual error of about 5.5e-17 of pure round-off. A tolerance
   derived from the truncation bound alone would fail.
3. *The remainder is the whole tail, not just the first term.* ``exp``'s actual error slightly
   **exceeds** the first-omitted-term bound, because the tail sums.

Hence ``atol = max(10 * |x|**P / P!, 1e-14 * max(1, max|expected|))``, with ``P`` the first omitted
power. ``pow`` uses a *relative* tolerance instead, and ``powmod`` is asserted **exactly**, which is
the entire point of the modulus bound.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from tasks import Graph, InitNode, Node, cos, cosh, exp, matpow, powmod, sin, sinh
from tasks.math import pow as tpow

Array = npt.NDArray[Any]
Document = dict[str, Any]


def evaluate(document: Document, overrides: dict[str, Array]) -> dict[str, Array]:
    """Interpret a serialized DAG with numpy, one forward pass over ``nodes``.

    The document declares ``ordering: topological``, so a single pass resolves every reference
    before it is used -- the same guarantee the C++ engine relies on.

    Args:
        document: A serialized, schema-conformant DAG.
        overrides: Values to substitute for ``init`` nodes, keyed by node ID.

    Returns:
        A mapping from node ID to computed array.
    """
    env: dict[str, Array] = {}
    for node in document["nodes"]:
        op = node["op"]
        node_id = node["id"]
        if op == "init":
            if node_id in overrides:
                env[node_id] = np.asarray(overrides[node_id], dtype=node["dtype"])
            else:
                shape = tuple(node["shape"])
                distribution = node["distribution"]
                if distribution == "ones":
                    env[node_id] = np.ones(shape, dtype=node["dtype"])
                elif distribution == "zeros":
                    env[node_id] = np.zeros(shape, dtype=node["dtype"])
                else:
                    rng = np.random.default_rng(node["seed"])
                    drawn = (
                        rng.uniform(size=shape)
                        if distribution == "uniform"
                        else rng.standard_normal(size=shape)
                    )
                    env[node_id] = drawn.astype(node["dtype"])
            continue
        left = env[node["inputs"][0]]
        if op == "add":
            env[node_id] = left + env[node["inputs"][1]]
        elif op == "multiply":
            env[node_id] = left * env[node["inputs"][1]]
        elif op == "scale":
            env[node_id] = left * node["factor"]
        elif op == "mod":
            # np.mod is floored, which is exactly what the schema specifies -- not fmod.
            env[node_id] = np.mod(left, node["modulus"])
        elif op == "dot_product":
            env[node_id] = left @ env[node["inputs"][1]]
        elif op == "cross_product":
            env[node_id] = np.cross(left, env[node["inputs"][1]])
        else:  # pragma: no cover - the op enum is closed
            raise AssertionError(f"unhandled op {op!r}")
    return env


def run(output: Node, overrides: dict[str, Array]) -> Array:
    """Serialize a graph, validate nothing was lost, and evaluate it.

    Args:
        output: Node to close the graph over.
        overrides: Values for ``init`` nodes, keyed by node ID.

    Returns:
        The array the output node computes.
    """
    document = Graph([output], dag_id="numeric").serialize(include_timestamp=False)
    env = evaluate(document, overrides)
    return env[document["outputs"][0]]


def series_tolerance(x: Array, expected: Array, first_omitted_power: int) -> float:
    """Compute the prescribed tolerance for a truncated series.

    Args:
        x: Input values.
        expected: Reference result.
        first_omitted_power: The power of the first term the series drops.

    Returns:
        ``max(10 * |x|**P / P!, 1e-14 * max(1, max|expected|))``.
    """
    peak = float(np.max(np.abs(x)))
    truncation = 10.0 * peak**first_omitted_power / math.factorial(first_omitted_power)
    roundoff = 1e-14 * max(1.0, float(np.max(np.abs(expected))))
    return max(truncation, roundoff)


def first_omitted_power(kind: str, terms: int) -> int:
    """Return the power of the first term a series omits.

    Args:
        kind: ``"odd"``, ``"even"``, or ``"all"``.
        terms: Number of terms included.

    Returns:
        ``2N+1`` for odd, ``2N`` for even, ``N`` for all.
    """
    if kind == "odd":
        return 2 * terms + 1
    if kind == "even":
        return 2 * terms
    return terms


SERIES: list[tuple[str, Callable[..., Node], Callable[[Array], Array], str]] = [
    ("sin", sin, np.sin, "odd"),
    ("sinh", sinh, np.sinh, "odd"),
    ("cos", cos, np.cos, "even"),
    ("cosh", cosh, np.cosh, "even"),
    ("exp", exp, np.exp, "all"),
]


class TestSeriesAccuracy:
    @pytest.mark.parametrize(
        ("name", "build", "reference", "kind"), SERIES, ids=[s[0] for s in SERIES]
    )
    @pytest.mark.parametrize("terms", [4, 8, 10, 14])
    def test_matches_numpy_within_the_computed_tolerance(
        self,
        name: str,
        build: Callable[..., Node],
        reference: Callable[[Array], Array],
        kind: str,
        terms: int,
    ) -> None:
        values = np.linspace(-0.5, 0.5, 7)
        x = InitNode((7,), seed=1, name="x")
        got = run(build(x, terms=terms), {"x": values})
        expected = reference(values)
        tolerance = series_tolerance(values, expected, first_omitted_power(kind, terms))
        assert np.allclose(got, expected, rtol=0.0, atol=tolerance)

    @pytest.mark.parametrize(
        ("name", "build", "reference", "kind"), SERIES, ids=[s[0] for s in SERIES]
    )
    def test_accuracy_improves_with_more_terms(
        self,
        name: str,
        build: Callable[..., Node],
        reference: Callable[[Array], Array],
        kind: str,
    ) -> None:
        values = np.linspace(-1.0, 1.0, 5)
        expected = reference(values)
        errors = []
        for terms in (2, 4, 6):
            x = InitNode((5,), seed=1, name="x")
            got = run(build(x, terms=terms), {"x": values})
            errors.append(float(np.max(np.abs(got - expected))))
        assert errors[0] > errors[1] > errors[2]

    @pytest.mark.parametrize(
        ("name", "build", "reference", "kind"), SERIES, ids=[s[0] for s in SERIES]
    )
    def test_works_on_a_rank_zero_input(
        self,
        name: str,
        build: Callable[..., Node],
        reference: Callable[[Array], Array],
        kind: str,
    ) -> None:
        """cos(u @ v) is the motivating case for the v1.2.0 init rank-0 relaxation."""
        u = InitNode((3,), seed=1, name="u")
        v = InitNode((3,), seed=2, name="v")
        uv = np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])
        scalar = np.asarray(float(np.dot(uv[0], uv[1])))
        got = run(build(u @ v, terms=8), {"u": uv[0], "v": uv[1]})
        expected = reference(scalar)
        tolerance = series_tolerance(scalar, expected, first_omitted_power(kind, 8))
        assert np.allclose(got, expected, rtol=0.0, atol=tolerance)
        assert np.asarray(got).shape == ()

    @pytest.mark.parametrize(
        ("name", "build", "reference", "kind"), SERIES, ids=[s[0] for s in SERIES]
    )
    def test_works_on_a_matrix_elementwise(
        self,
        name: str,
        build: Callable[..., Node],
        reference: Callable[[Array], Array],
        kind: str,
    ) -> None:
        values = np.linspace(-0.4, 0.4, 6).reshape(2, 3)
        x = InitNode((2, 3), seed=1, name="x")
        got = run(build(x, terms=10), {"x": values})
        expected = reference(values)
        tolerance = series_tolerance(values, expected, first_omitted_power(kind, 10))
        assert np.allclose(got, expected, rtol=0.0, atol=tolerance)
        assert np.asarray(got).shape == (2, 3)

    def test_single_term_series_are_their_leading_terms(self) -> None:
        values = np.linspace(-0.3, 0.3, 5)
        x = InitNode((5,), seed=1, name="x")
        assert np.allclose(run(sin(x, terms=1), {"x": values}), values)
        y = InitNode((5,), seed=1, name="x")
        assert np.allclose(run(cos(y, terms=1), {"x": values}), np.ones(5))

    def test_exp_converges_more_slowly_than_sin_at_equal_terms(self) -> None:
        """Its omitted power is N rather than roughly 2N, so it wants more terms."""
        values = np.linspace(-1.0, 1.0, 5)
        x1 = InitNode((5,), seed=1, name="x")
        x2 = InitNode((5,), seed=1, name="x")
        sin_error = float(np.max(np.abs(run(sin(x1, terms=6), {"x": values}) - np.sin(values))))
        exp_error = float(np.max(np.abs(run(exp(x2, terms=6), {"x": values}) - np.exp(values))))
        assert exp_error > sin_error

    def test_large_argument_is_documented_as_meaningless(self) -> None:
        """No range reduction: at |x| = 10 the series is not merely inaccurate."""
        values = np.array([10.0])
        x = InitNode((1,), seed=1, name="x")
        got = run(sin(x, terms=8), {"x": values})
        assert float(np.max(np.abs(got - np.sin(values)))) > 1.0


class TestPowAccuracy:
    @pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 16, 100, 1024])
    def test_matches_numpy_within_a_relative_tolerance(self, n: int) -> None:
        """1.2**1024 is about 1.2e81, so only a relative tolerance is meaningful."""
        values = np.linspace(0.8, 1.2, 5)
        x = InitNode((5,), seed=1, name="x")
        got = run(tpow(x, n), {"x": values})
        expected = values**n
        assert np.allclose(got, expected, rtol=1e-12, atol=0.0)

    def test_exponent_zero_is_all_ones(self) -> None:
        values = np.linspace(0.5, 2.0, 4)
        x = InitNode((4,), seed=1, name="x")
        assert np.array_equal(run(tpow(x, 0), {"x": values}), np.ones(4))

    def test_zero_to_the_zero_is_one_matching_numpy(self) -> None:
        values = np.zeros(3)
        x = InitNode((3,), seed=1, name="x")
        assert np.array_equal(run(tpow(x, 0), {"x": values}), np.ones(3))

    def test_float32_stays_near_one_to_avoid_overflow(self) -> None:
        """X ** 1024 overflows float32 for any |x| > 1.0906."""
        values = np.linspace(0.99, 1.01, 5).astype(np.float32)
        x = InitNode((5,), seed=1, dtype="float32", name="x")
        got = run(tpow(x, 1024), {"x": values})
        expected = values.astype(np.float64) ** 1024
        assert np.allclose(got, expected, rtol=1e-3, atol=0.0)

    def test_exact_for_small_integer_bases(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        x = InitNode((3,), seed=1, name="x")
        assert np.array_equal(run(tpow(x, 10), {"x": values}), values**10)


class TestPowmodExactness:
    """Within the computed modulus bound this agrees with Python's pow(a, n, m) exactly."""

    @pytest.mark.parametrize(
        ("n", "m"), [(1, 7), (5, 13), (10, 97), (100, 101), (1024, 1000003), (37, 94906266)]
    )
    def test_exact_against_python_pow(self, n: int, m: int) -> None:
        bases = np.array([2.0, 3.0, 5.0, 7.0])
        x = InitNode((4,), seed=1, name="x")
        got = run(powmod(x, n, m), {"x": bases})
        expected = np.array([float(pow(int(b), n, m)) for b in bases])
        assert np.array_equal(got, expected)

    def test_result_lies_in_the_half_open_interval(self) -> None:
        bases = np.array([2.0, 3.0, 5.0, 7.0])
        x = InitNode((4,), seed=1, name="x")
        got = run(powmod(x, 50, 97), {"x": bases})
        assert np.all(got >= 0)
        assert np.all(got < 97)

    def test_floored_semantics_on_negative_input(self) -> None:
        """np.mod, not fmod: the result is non-negative even for a negative dividend."""
        values = np.array([-7.0, -1.0, 3.0])
        x = InitNode((3,), seed=1, name="x")
        got = run(x % 5, {"x": values})
        assert np.array_equal(got, np.array([3.0, 4.0, 3.0]))
        assert np.array_equal(got, np.mod(values, 5.0))

    def test_fmod_would_have_given_a_different_answer(self) -> None:
        """Pins the semantic choice: truncated remainder differs on negative inputs."""
        values = np.array([-7.0])
        x = InitNode((1,), seed=1, name="x")
        got = run(x % 5, {"x": values})
        assert got[0] == 3.0
        assert math.fmod(-7.0, 5.0) == -2.0

    def test_exponent_zero_is_one_mod_m(self) -> None:
        bases = np.array([2.0, 3.0])
        x = InitNode((2,), seed=1, name="x")
        assert np.array_equal(run(powmod(x, 0, 7), {"x": bases}), np.ones(2))

    def test_modulus_one_collapses_to_zero(self) -> None:
        bases = np.array([2.0, 3.0, 5.0])
        x = InitNode((3,), seed=1, name="x")
        assert np.array_equal(run(powmod(x, 5, 1), {"x": bases}), np.zeros(3))


class TestMatpowAccuracy:
    @pytest.mark.parametrize("n", [1, 2, 5, 16, 64])
    def test_matches_numpy_matrix_power(self, n: int) -> None:
        matrix = np.array(
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.05, 0.9, 0.05, 0.0],
                [0.0, 0.1, 0.8, 0.1],
                [0.0, 0.0, 0.2, 0.8],
            ]
        )
        a = InitNode((4, 4), seed=1, name="a")
        got = run(matpow(a, n), {"a": matrix})
        expected = np.linalg.matrix_power(matrix, n)
        assert np.allclose(got, expected, rtol=0.0, atol=1e-12)

    def test_is_not_elementwise_power(self) -> None:
        """Guards against a dot_product/multiply mix-up, which node counts alone would miss."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = InitNode((2, 2), seed=1, name="a")
        got = run(matpow(a, 2), {"a": matrix})
        assert np.allclose(got, matrix @ matrix)
        assert not np.allclose(got, matrix * matrix)

    def test_identity_matrix_is_its_own_power(self) -> None:
        identity = np.eye(3)
        a = InitNode((3, 3), seed=1, name="a")
        assert np.allclose(run(matpow(a, 32), {"a": identity}), identity)


class TestOperatorNumerics:
    def test_multiply_is_elementwise_not_a_contraction(self) -> None:
        left_values = np.array([[1.0, 2.0], [3.0, 4.0]])
        right_values = np.array([[5.0, 6.0], [7.0, 8.0]])
        left = InitNode((2, 2), seed=1, name="l")
        right = InitNode((2, 2), seed=2, name="r")
        got = run(left * right, {"l": left_values, "r": right_values})
        assert np.array_equal(got, left_values * right_values)
        assert not np.array_equal(got, left_values @ right_values)

    def test_subtraction_expansion_computes_a_difference(self) -> None:
        left_values = np.array([5.0, 7.0, 9.0])
        right_values = np.array([1.0, 2.0, 3.0])
        left = InitNode((3,), seed=1, name="l")
        right = InitNode((3,), seed=2, name="r")
        got = run(left - right, {"l": left_values, "r": right_values})
        assert np.array_equal(got, left_values - right_values)

    def test_pow_operator_matches_numpy(self) -> None:
        values = np.array([1.5, 2.0, 2.5])
        x = InitNode((3,), seed=1, name="x")
        assert np.allclose(run(x**7, {"x": values}), values**7)

    def test_mod_operator_matches_numpy(self) -> None:
        values = np.array([1.0, 7.0, 13.0, -2.0])
        x = InitNode((4,), seed=1, name="x")
        assert np.array_equal(run(x % 5, {"x": values}), np.mod(values, 5.0))


class TestInterpreterCoversEveryOp:
    def test_all_seven_ops_evaluate(self) -> None:
        """A graph exercising the whole enum, so the interpreter cannot silently skip one."""
        m = InitNode((3, 3), seed=1, name="m")
        u = InitNode((3,), seed=2, name="u")
        v = InitNode((3,), seed=3, name="v")
        contracted = m @ u
        crossed = u.cross(v)
        combined = ((contracted + crossed) * crossed) % 5
        document = Graph([combined - contracted], dag_id="all").serialize(include_timestamp=False)
        ops = {node["op"] for node in document["nodes"]}
        assert ops == {"init", "add", "multiply", "scale", "mod", "dot_product", "cross_product"}
        env = evaluate(
            document,
            {
                "m": np.eye(3),
                "u": np.array([1.0, 2.0, 3.0]),
                "v": np.array([4.0, 5.0, 6.0]),
            },
        )
        assert np.asarray(env[document["outputs"][0]]).shape == (3,)

    def test_unseeded_inits_are_reproducible_from_the_document(self) -> None:
        """No overrides: ones and zeros are language-independent."""
        x = InitNode((4,), seed=0, distribution="ones", name="x")
        assert np.array_equal(run(x * 3.0, {}), np.full(4, 3.0))

    def test_zeros_distribution(self) -> None:
        x = InitNode((4,), seed=0, distribution="zeros", name="x")
        assert np.array_equal(run(x + x, {}), np.zeros(4))
