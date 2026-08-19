"""Adversarial tests for the v1.2.0 composite tier: numerics, counts, hygiene, migration.

The other composite suites confirm documented behaviour. This one attacks it along five axes:

1. **Numerics.** Every expansion is evaluated with numpy against the reference function, under the
   design's computed tolerance rather than a guessed constant. The attacks are chosen to break
   specific assumptions: negative inputs (odd/even symmetry must hold exactly), values straddling
   zero in one tensor, the ``terms=1``/``terms=2`` boundaries, ``float32``, rank-0 operands,
   exponents at popcount extremes, and ``powmod`` at both edges of its exactness bound.
2. **Node-count and depth contract.** The formulas are asserted across a sweep, not spot-checked,
   and the depth claims are asserted where the design commits to a number.
3. **Expansion hygiene.** Every expanded graph must validate against the real contract; ``x**2``
   must actually be reused; composites must compose; large builds must not blow up.
4. **Migration completeness.** No stale assertion or docstring may still claim ``node * node``
   raises, that ``init`` has a rank floor, or that the wire version is 1.1.0.
5. **Cross-project.** A serialized v1.2.0 document carrying ``multiply`` and ``mod`` must satisfy
   the same contract ``hpcctl submit`` validates against, and 1.0.0/1.1.0-era documents must
   still pass.

:class:`TestConstantTermDropsOperand` pins the one finding from this pass: ``cos``, ``cosh``, and
``exp`` at ``terms=1`` produce a graph that never mentions ``x``, exactly as ``pow(x, 0)`` does.
The behaviour is mathematically right and the node counts are as documented; only the docstrings
were silent about it.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from jsonschema import Draft202012Validator

import tasks
from tasks import (
    DagBuildError,
    DimensionalityError,
    Graph,
    InitNode,
    ModNode,
    MultiplyNode,
    Node,
    ShapeMismatchError,
    cos,
    cosh,
    cross,
    exp,
    matpow,
    powmod,
    sin,
    sinh,
)
from tasks.dtypes import DType
from tasks.graph import SCHEMA_VERSION
from tasks.math import multiplies, safe_modulus_limit
from tasks.math import pow as tpow

Array = npt.NDArray[Any]
Document = dict[str, Any]

REPO = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO / "shared" / "dag_schema.json"

SERIES: list[tuple[str, Callable[..., Node], Callable[[Array], Array], str]] = [
    ("sin", sin, np.sin, "odd"),
    ("sinh", sinh, np.sinh, "odd"),
    ("cos", cos, np.cos, "even"),
    ("cosh", cosh, np.cosh, "even"),
    ("exp", exp, np.exp, "all"),
]

# Names the input node explicitly so an override key cannot land on the `ones` constant an even
# or all-parity series also emits. Getting this wrong silently substitutes the wrong tensor.
X = "x"


def evaluate(document: Document, overrides: dict[str, Array]) -> dict[str, Array]:
    """Interpret a serialized DAG with numpy in one forward pass.

    Args:
        document: A serialized, schema-conformant DAG.
        overrides: Values to substitute for ``init`` nodes, keyed by node ID.

    Returns:
        A mapping from node ID to computed array.
    """
    env: dict[str, Array] = {}
    for node in document["nodes"]:
        op = node["op"]
        key = node["id"]
        if op == "init":
            if key in overrides:
                env[key] = np.asarray(overrides[key], dtype=node["dtype"])
            else:
                shape = tuple(node["shape"])
                dist = node["distribution"]
                if dist == "ones":
                    env[key] = np.ones(shape, dtype=node["dtype"])
                elif dist == "zeros":
                    env[key] = np.zeros(shape, dtype=node["dtype"])
                else:
                    rng = np.random.default_rng(node["seed"])
                    drawn = (
                        rng.uniform(size=shape)
                        if dist == "uniform"
                        else rng.standard_normal(size=shape)
                    )
                    env[key] = drawn.astype(node["dtype"])
            continue
        left = env[node["inputs"][0]]
        if op == "add":
            env[key] = left + env[node["inputs"][1]]
        elif op == "multiply":
            env[key] = left * env[node["inputs"][1]]
        elif op == "scale":
            env[key] = left * node["factor"]
        elif op == "mod":
            # np.mod is floored, which is what the schema specifies -- never fmod.
            env[key] = np.mod(left, node["modulus"])
        elif op == "dot_product":
            env[key] = left @ env[node["inputs"][1]]
        elif op == "cross_product":
            env[key] = np.cross(left, env[node["inputs"][1]])
        else:  # pragma: no cover - the op enum is closed
            raise AssertionError(f"unhandled op {op!r}")
    return env


def serialize(output: Node, *, dag_id: str = "adv") -> Document:
    """Close a graph over one node and serialize it.

    Args:
        output: Node to close the graph over.
        dag_id: Identifier for the document.

    Returns:
        The serialized document.
    """
    return Graph([output], dag_id=dag_id).serialize(include_timestamp=False)


def run(output: Node, values: Array, *, key: str = X) -> Array:
    """Serialize, evaluate, and return the output array.

    Args:
        output: Node to evaluate.
        values: Values to substitute for the named input node.
        key: Node ID to override.

    Returns:
        The array the output node computes.
    """
    document = serialize(output)
    return evaluate(document, {key: values})[document["outputs"][0]]


def tolerance(x: Array, expected: Array, omitted_power: int) -> float:
    """Compute the design's prescribed tolerance for a truncated series.

    Args:
        x: Input values.
        expected: Reference result.
        omitted_power: Power of the first term the series drops.

    Returns:
        ``max(10 * |x|**P / P!, 1e-14 * max(1, max|expected|))``.
    """
    peak = float(np.max(np.abs(x)))
    truncation = 10.0 * peak**omitted_power / math.factorial(omitted_power)
    roundoff = 1e-14 * max(1.0, float(np.max(np.abs(expected))))
    return max(truncation, roundoff)


def omitted(kind: str, terms: int) -> int:
    """Return the power of the first term a series omits.

    Args:
        kind: ``"odd"``, ``"even"``, or ``"all"``.
        terms: Number of terms included.

    Returns:
        ``2N+1`` for odd, ``2N`` for even, ``N`` for all.
    """
    return {"odd": 2 * terms + 1, "even": 2 * terms}.get(kind, terms)


def node_count(output: Node, base: Node) -> int:
    """Count reachable nodes excluding the base operand.

    Args:
        output: Node closing the graph.
        base: Operand to exclude from the count, if it is reachable at all.

    Returns:
        The number of nodes the expansion contributed.
    """
    reachable = Graph([output], dag_id="count").nodes()
    return len(reachable) - (1 if base in reachable else 0)


def depth_of(output: Node) -> int:
    """Return the longest path from ``output`` back to a source, iteratively.

    Args:
        output: Node to measure from.

    Returns:
        Edge count along the longest chain; 0 for a source node.
    """
    order = Graph([output], dag_id="depth").topological_order()
    best: dict[int, int] = {}
    for node in order:
        best[id(node)] = 0 if not node.inputs else 1 + max(best[id(i)] for i in node.inputs)
    return best[id(output)]


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    """Build a validator over the real serialization contract.

    Returns:
        A draft 2020-12 validator for ``/shared/dag_schema.json``.
    """
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def assert_conforms(validator: Draft202012Validator, document: Document) -> None:
    """Assert a document satisfies the contract, listing every violation.

    Args:
        validator: Contract validator.
        document: Serialized DAG.
    """
    errors = sorted(validator.iter_errors(document), key=str)
    if errors:
        detail = "\n".join(f"  at {list(e.absolute_path)}: {e.message}" for e in errors)
        raise AssertionError(f"document violates the contract:\n{detail}")


class TestSeriesNumericAttacks:
    """Series accuracy under inputs chosen to break symmetry or cancellation assumptions."""

    @pytest.mark.parametrize(("name", "fn", "ref", "kind"), SERIES)
    def test_negative_inputs_match_the_reference(
        self, name: str, fn: Callable[..., Node], ref: Callable[[Array], Array], kind: str
    ) -> None:
        values = np.array([-1.5, -1.0, -0.5, -0.25, -0.01])
        got = run(fn(InitNode(values.shape, seed=1, name=X), terms=12), values)
        want = ref(values)
        assert np.allclose(got, want, atol=tolerance(values, want, omitted(kind, 12)), rtol=0)

    @pytest.mark.parametrize(("name", "fn"), [("sin", sin), ("sinh", sinh)])
    def test_odd_functions_are_odd(self, name: str, fn: Callable[..., Node]) -> None:
        """``f(-x) == -f(x)`` must hold in the expansion, not just in the reference."""
        values = np.array([0.25, 0.5, 1.0, 1.75])
        positive = run(fn(InitNode(values.shape, seed=1, name=X), terms=12), values)
        negative = run(fn(InitNode(values.shape, seed=1, name=X), terms=12), -values)
        assert np.allclose(negative, -positive, atol=0, rtol=0)

    @pytest.mark.parametrize(("name", "fn"), [("cos", cos), ("cosh", cosh)])
    def test_even_functions_are_even(self, name: str, fn: Callable[..., Node]) -> None:
        values = np.array([0.25, 0.5, 1.0, 1.75])
        positive = run(fn(InitNode(values.shape, seed=1, name=X), terms=12), values)
        negative = run(fn(InitNode(values.shape, seed=1, name=X), terms=12), -values)
        assert np.allclose(negative, positive, atol=0, rtol=0)

    @pytest.mark.parametrize(("name", "fn", "ref", "kind"), SERIES)
    def test_mixed_signs_and_values_near_zero_in_one_tensor(
        self, name: str, fn: Callable[..., Node], ref: Callable[[Array], Array], kind: str
    ) -> None:
        """Straddling zero exercises the sign-alternating terms against catastrophic values."""
        values = np.array([-2.0, -0.5, -1e-8, 0.0, 1e-8, 0.5, 2.0])
        got = run(fn(InitNode(values.shape, seed=1, name=X), terms=14), values)
        want = ref(values)
        assert np.allclose(got, want, atol=tolerance(values, want, omitted(kind, 14)), rtol=0)

    @pytest.mark.parametrize(("name", "fn", "ref", "kind"), SERIES)
    def test_exactly_zero_input(
        self, name: str, fn: Callable[..., Node], ref: Callable[[Array], Array], kind: str
    ) -> None:
        values = np.zeros(3)
        got = run(fn(InitNode(values.shape, seed=1, name=X), terms=6), values)
        assert np.allclose(got, ref(values), atol=1e-15, rtol=0)

    @pytest.mark.parametrize(("name", "fn", "ref", "kind"), SERIES)
    @pytest.mark.parametrize("terms", [2, 3])
    def test_low_term_boundaries(
        self,
        name: str,
        fn: Callable[..., Node],
        ref: Callable[[Array], Array],
        kind: str,
        terms: int,
    ) -> None:
        """``terms=1`` is covered separately: for even/all parity it drops ``x`` entirely."""
        values = np.array([0.1, -0.1, 0.05])
        got = run(fn(InitNode(values.shape, seed=1, name=X), terms=terms), values)
        want = ref(values)
        assert np.allclose(got, want, atol=tolerance(values, want, omitted(kind, terms)), rtol=0)

    @pytest.mark.parametrize(("name", "fn"), [("sin", sin), ("sinh", sinh)])
    def test_odd_series_at_one_term_is_just_x(self, name: str, fn: Callable[..., Node]) -> None:
        """One odd term is ``x/1!``, so the expansion is a single unit scale of ``x``."""
        values = np.array([0.3, -0.7, 0.0])
        assert np.allclose(run(fn(InitNode(values.shape, seed=1, name=X), terms=1), values), values)

    @pytest.mark.parametrize(("name", "fn", "ref", "kind"), SERIES)
    def test_float32_expansion_stays_float32_and_tracks_the_reference(
        self, name: str, fn: Callable[..., Node], ref: Callable[[Array], Array], kind: str
    ) -> None:
        values = np.array([0.5, -0.5, 0.25, -0.125], dtype=np.float32)
        output = fn(InitNode(values.shape, seed=1, dtype="float32", name=X), terms=8)
        document = serialize(output)
        assert {node["dtype"] for node in document["nodes"]} == {"float32"}
        got = evaluate(document, {X: values})[document["outputs"][0]]
        assert got.dtype == np.float32
        want = ref(values.astype(np.float64))
        assert np.allclose(got, want, atol=1e-6, rtol=0)

    @pytest.mark.parametrize(("name", "fn", "ref", "kind"), SERIES)
    def test_rank_zero_operand_from_a_vector_dot(
        self, name: str, fn: Callable[..., Node], ref: Callable[[Array], Array], kind: str
    ) -> None:
        """``sin(u @ v)`` is the expression that motivated lifting the rank-1 floor."""
        u_values = np.array([0.3, 0.2, 0.1, 0.4])
        v_values = np.array([0.5, 0.25, 0.2, 0.1])
        u = InitNode(u_values.shape, seed=1, name="u")
        v = InitNode(v_values.shape, seed=2, name="v")
        output = fn(u @ v, terms=12)
        document = serialize(output)
        result_shape = next(
            node["output_shape"]
            for node in document["nodes"]
            if node["id"] == document["outputs"][0]
        )
        assert result_shape == []
        got = evaluate(document, {"u": u_values, "v": v_values})[document["outputs"][0]]
        assert np.allclose(got, ref(u_values @ v_values), atol=1e-13, rtol=0)

    @pytest.mark.parametrize(("name", "fn", "ref", "kind"), SERIES)
    def test_rank_zero_init_operand(
        self, name: str, fn: Callable[..., Node], ref: Callable[[Array], Array], kind: str
    ) -> None:
        """A rank-0 ``init`` is legal as of v1.2.0 and must flow through an elementwise series."""
        values = np.array(0.4)
        got = run(fn(InitNode((), seed=1, name=X), terms=12), values)
        assert np.allclose(got, ref(values), atol=1e-13, rtol=0)


class TestPowNumericAttacks:
    """Binary exponentiation accuracy, with exponents chosen at popcount extremes."""

    @pytest.mark.parametrize("k", list(range(1, 12)))
    @pytest.mark.parametrize("offset", [-1, 0, 1])
    def test_popcount_extremes(self, k: int, offset: int) -> None:
        """``2**k`` is all-squares; ``2**k - 1`` is all-ones, the worst case for the walk."""
        exponent = 2**k + offset
        if exponent < 1:
            pytest.skip("exponent below 1 is covered by the n=0 case")
        values = np.array([1.0009, 0.9995, -1.0007, 1.0])
        got = run(tpow(InitNode(values.shape, seed=1, name=X), exponent), values)
        assert np.allclose(got, values**exponent, rtol=1e-12, atol=0)

    def test_negative_base_sign_follows_exponent_parity(self) -> None:
        values = np.array([-2.0, -1.0, -0.5])
        for exponent in (2, 3, 7, 8):
            got = run(tpow(InitNode(values.shape, seed=1, name=X), exponent), values)
            assert np.allclose(got, values**exponent, rtol=1e-13, atol=0)
            assert np.all(np.sign(got) == np.sign(values**exponent))

    def test_zero_base(self) -> None:
        values = np.zeros(3)
        assert np.allclose(run(tpow(InitNode(values.shape, seed=1, name=X), 5), values), 0.0)

    def test_matpow_matches_numpy_matrix_power(self) -> None:
        matrix = np.array([[0.9, 0.1, 0.0], [0.05, 0.9, 0.05], [0.0, 0.2, 0.8]])
        for exponent in (1, 2, 3, 5, 7, 16, 17, 64):
            got = run(matpow(InitNode(matrix.shape, seed=1, name=X), exponent), matrix)
            want = np.linalg.matrix_power(matrix, exponent)
            assert np.allclose(got, want, rtol=1e-11, atol=1e-13)


class TestPowmodExactness:
    """``powmod`` must agree with Python's ``pow(a, n, m)`` exactly inside its bound."""

    MODULI = (2, 3, 97, 1000, 65537)
    EXPONENTS = (0, 1, 2, 3, 5, 7, 10, 16, 17, 100, 1024)

    @pytest.mark.parametrize("modulus", MODULI)
    @pytest.mark.parametrize("exponent", EXPONENTS)
    def test_exact_including_negative_bases(self, modulus: int, exponent: int) -> None:
        """Negative bases are the floored-mod test: ``fmod`` would carry the sign and fail."""
        values = np.array(
            [0, 1, 2, 5, -5, -1, -123, 7, modulus - 1, -(modulus - 1)], dtype=np.float64
        )
        got = run(powmod(InitNode(values.shape, seed=1, name=X), exponent, float(modulus)), values)
        want = np.array([pow(int(v), exponent, modulus) for v in values], dtype=np.float64)
        assert np.array_equal(got, want)

    @pytest.mark.parametrize("exponent", [1, 2, 3, 5, 10, 100, 1024])
    def test_exact_at_the_float64_limit(self, exponent: int) -> None:
        modulus = safe_modulus_limit("float64")
        values = np.array([modulus - 1, modulus - 2, 12345678, 1, 0, -(modulus - 1)])
        got = run(powmod(InitNode(values.shape, seed=1, name=X), exponent, float(modulus)), values)
        want = np.array([pow(int(v), exponent, modulus) for v in values], dtype=np.float64)
        assert np.array_equal(got, want)

    @pytest.mark.parametrize("exponent", [1, 2, 3, 5, 7, 10, 16, 17, 64, 100, 1024])
    def test_exact_at_the_tight_float32_limit(self, exponent: int) -> None:
        """``float32``'s bound is exactly tight: ``4096**2 == 2**24``."""
        modulus = safe_modulus_limit("float32")
        values = np.array(
            [modulus - 1, modulus - 2, 4096, 2048, 3, 1, 0, -(modulus - 1)], dtype=np.float32
        )
        output = powmod(
            InitNode(values.shape, seed=1, dtype="float32", name=X), exponent, float(modulus)
        )
        got = run(output, values)
        want = np.array([pow(int(v), exponent, modulus) for v in values], dtype=np.float32)
        assert np.array_equal(got, want)

    def test_no_intermediate_leaves_the_exact_integer_range(self) -> None:
        """The bound's whole justification: intermediates peak at ``(m-1)**2``."""
        modulus = safe_modulus_limit("float64")
        values = np.array([modulus - 1, modulus - 2, 12345678, 1, 0, -(modulus - 1)])
        document = serialize(powmod(InitNode(values.shape, seed=1, name=X), 1024, float(modulus)))
        env = evaluate(document, {X: values})
        peak = max(float(np.max(np.abs(env[node["id"]]))) for node in document["nodes"])
        assert peak == float((modulus - 1) ** 2)
        assert peak <= float(2**53)

    def test_every_mod_result_lies_in_the_half_open_interval(self) -> None:
        """Floored semantics: even from negative input, every ``mod`` output is in ``[0, m)``."""
        modulus = 97.0
        values = np.array([-5.0, -123.0, -1.0, -96.0])
        document = serialize(powmod(InitNode(values.shape, seed=1, name=X), 10, modulus))
        env = evaluate(document, {X: values})
        mod_nodes = [node for node in document["nodes"] if node["op"] == "mod"]
        assert mod_nodes
        for node in mod_nodes:
            result = env[node["id"]]
            assert np.all(result >= 0.0)
            assert np.all(result < modulus)

    @pytest.mark.parametrize("dtype", ["float64", "float32"])
    def test_the_bound_is_enforced_at_its_exact_edge(self, dtype: DType) -> None:
        limit = safe_modulus_limit(dtype)
        operand = InitNode((2,), seed=1, dtype=dtype)
        powmod(operand, 3, float(limit))
        with pytest.raises(ValueError, match="exceeds the largest value exact"):
            powmod(operand, 3, float(limit + 1))

    @pytest.mark.parametrize("dtype", ["float64", "float32"])
    def test_the_bound_arithmetic_is_what_the_design_computed(self, dtype: DType) -> None:
        limit = safe_modulus_limit(dtype)
        bits = 53 if dtype == "float64" else 24
        assert (limit - 1) ** 2 <= 2**bits
        assert limit**2 > 2**bits

    def test_allow_inexact_bypasses_the_bound_and_genuinely_drifts(self) -> None:
        """Documented behaviour: outside the bound results silently disagree."""
        modulus = 100000
        values = np.array([modulus - 1, modulus - 2, 7], dtype=np.float32)
        output = powmod(
            InitNode(values.shape, seed=1, dtype="float32", name=X),
            16,
            float(modulus),
            allow_inexact=True,
        )
        got = run(output, values)
        want = np.array([pow(int(v), 16, modulus) for v in values], dtype=np.float64)
        assert not np.array_equal(got.astype(np.float64), want)


class TestNodeCountContract:
    """The formulas are the product; assert them across a sweep, not at spot checks."""

    SWEEP = [*range(1, 65), 1023, 1024, 1025]

    @pytest.mark.parametrize("exponent", [*SWEEP, 2**20, 2**30])
    def test_multiplies_matches_the_documented_closed_form(self, exponent: int) -> None:
        closed = (exponent.bit_length() - 1) + bin(exponent).count("1") - 1
        assert multiplies(exponent) == closed

    @pytest.mark.parametrize("exponent", [*SWEEP, 2**20])
    def test_multiplies_matches_a_brute_force_walk_of_the_algorithm(self, exponent: int) -> None:
        """Independent derivation: count the steps the documented bit walk actually takes."""
        counted = 0
        for bit in bin(exponent)[3:]:
            counted += 1
            if bit == "1":
                counted += 1
        assert multiplies(exponent) == counted

    @pytest.mark.parametrize("exponent", SWEEP)
    def test_pow_node_count(self, exponent: int) -> None:
        base = InitNode((3,), seed=1)
        assert node_count(tpow(base, exponent), base) == multiplies(exponent)

    @pytest.mark.parametrize("exponent", SWEEP)
    def test_powmod_node_count(self, exponent: int) -> None:
        base = InitNode((3,), seed=1)
        assert node_count(powmod(base, exponent, 97.0), base) == 2 * multiplies(exponent) + 1

    @pytest.mark.parametrize("exponent", SWEEP)
    def test_matpow_node_count(self, exponent: int) -> None:
        base = InitNode((3, 3), seed=1)
        assert node_count(matpow(base, exponent), base) == multiplies(exponent)

    def test_the_headline_claim(self) -> None:
        """1024 costs ten multiplies, not 1023. This is the whole point of the tier."""
        base = InitNode((3,), seed=1)
        assert node_count(tpow(base, 1024), base) == 10
        assert node_count(powmod(base, 1024, 97.0), base) == 21
        square = InitNode((3, 3), seed=2)
        assert node_count(matpow(square, 64), square) == 6

    @pytest.mark.parametrize("terms", [*range(1, 65), 85])
    @pytest.mark.parametrize(("name", "fn"), [("sin", sin), ("sinh", sinh)])
    def test_odd_series_counts(self, name: str, fn: Callable[..., Node], terms: int) -> None:
        base = InitNode((3,), seed=1)
        expected = 1 if terms == 1 else 3 * terms - 1
        assert node_count(fn(base, terms=terms), base) == expected

    @pytest.mark.parametrize("terms", [*range(1, 65), 86])
    @pytest.mark.parametrize(("name", "fn"), [("cos", cos), ("cosh", cosh)])
    def test_even_series_counts(self, name: str, fn: Callable[..., Node], terms: int) -> None:
        base = InitNode((3,), seed=1)
        expected = 2 if terms == 1 else 3 * terms - 1
        assert node_count(fn(base, terms=terms), base) == expected

    @pytest.mark.parametrize("terms", [*range(1, 65), 171])
    def test_exp_counts(self, terms: int) -> None:
        base = InitNode((3,), seed=1)
        expected = 2 if terms == 1 else 3 * terms - 2
        assert node_count(exp(base, terms=terms), base) == expected

    def test_the_measured_tables_in_the_design(self) -> None:
        base = InitNode((3,), seed=1)
        assert [node_count(sin(base, terms=n), base) for n in range(1, 9)] == [
            1,
            5,
            8,
            11,
            14,
            17,
            20,
            23,
        ]
        assert [node_count(cos(base, terms=n), base) for n in range(1, 9)] == [
            2,
            5,
            8,
            11,
            14,
            17,
            20,
            23,
        ]
        assert [node_count(exp(base, terms=n), base) for n in range(1, 9)] == [
            2,
            4,
            7,
            10,
            13,
            16,
            19,
            22,
        ]

    def test_multiplies_is_not_the_node_count_at_zero(self) -> None:
        """``multiplies(0) == 0`` but ``pow(x, 0)`` still emits the ``ones`` node."""
        base = InitNode((3,), seed=1)
        assert multiplies(0) == 0
        assert node_count(tpow(base, 0), base) == 1
        assert node_count(powmod(base, 0, 97.0), base) == 2


class TestDepthContract:
    """Where the design commits to a depth, assert it."""

    def test_power_composites_are_pure_serial_chains(self) -> None:
        """Depth equals node count: zero parallelism, which is the research contrast."""
        base = InitNode((3,), seed=1)
        for output in (tpow(base, 1024), powmod(base, 1024, 97.0)):
            assert depth_of(output) == node_count(output, base)
        square = InitNode((3, 3), seed=2)
        assert depth_of(matpow(square, 64)) == node_count(matpow(square, 64), square)

    def test_measured_power_depths(self) -> None:
        base = InitNode((3,), seed=1)
        assert depth_of(tpow(base, 1024)) == 10
        assert depth_of(powmod(base, 1024, 97.0)) == 21
        assert depth_of(matpow(InitNode((3, 3), seed=2), 64)) == 6

    @pytest.mark.parametrize(
        ("name", "fn", "expected"),
        [
            ("sin", sin, 13),
            ("sinh", sinh, 13),
            ("cos", cos, 12),
            ("cosh", cosh, 12),
            ("exp", exp, 11),
        ],
    )
    def test_measured_series_depths_at_ten_terms(
        self, name: str, fn: Callable[..., Node], expected: int
    ) -> None:
        assert depth_of(fn(InitNode((3,), seed=1), terms=10)) == expected

    def test_series_are_wide_and_shallow_while_powmod_is_narrow_and_deep(self) -> None:
        """The contrast the baseline exists to contain, asserted rather than asserted in prose."""
        base = InitNode((3,), seed=1)
        series = sin(base, terms=10)
        chain = powmod(base, 1024, 97.0)
        assert node_count(series, base) == 29
        assert depth_of(series) == 13
        assert node_count(chain, base) == 21
        assert depth_of(chain) == 21
        assert depth_of(series) < node_count(series, base)


class TestTermCaps:
    """The float-range caps, asserted at their exact edges."""

    @pytest.mark.parametrize(
        ("name", "fn", "cap"),
        [
            ("sin", sin, 85),
            ("sinh", sinh, 85),
            ("cos", cos, 86),
            ("cosh", cosh, 86),
            ("exp", exp, 171),
        ],
    )
    def test_cap_edges(self, name: str, fn: Callable[..., Node], cap: int) -> None:
        base = InitNode((2,), seed=1)
        fn(base, terms=cap)
        with pytest.raises(ValueError, match="not representable as a float"):
            fn(base, terms=cap + 1)

    def test_a_wildly_large_term_count_is_rejected_not_silently_truncated(self) -> None:
        """500 terms exceeds every cap; the coefficient must not underflow to zero in silence."""
        with pytest.raises(ValueError, match="not representable as a float"):
            sin(InitNode((2,), seed=1), terms=500)

    def test_the_error_names_the_offending_term(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            sin(InitNode((2,), seed=1), terms=500)
        assert re.search(r"term \d+ needs 1/\d+!", str(excinfo.value))


class TestExpansionHygiene:
    """Every expansion must be a legal document, reuse its powers, and compose."""

    def _catalogue(self) -> dict[str, Node]:
        """Build one representative of every expansion shape.

        Returns:
            A mapping of label to output node.
        """
        cases: dict[str, Node] = {}
        for terms in (1, 2, 3, 8, 10, 40):
            for name, fn, _, _ in SERIES:
                cases[f"{name}-{terms}"] = fn(InitNode((3, 2), seed=1, name=X), terms=terms)
        for exponent in (0, 1, 2, 3, 7, 10, 16, 100, 1024):
            cases[f"pow-{exponent}"] = tpow(InitNode((3,), seed=1, name=X), exponent)
            cases[f"powmod-{exponent}"] = powmod(InitNode((3,), seed=1, name=X), exponent, 97.0)
        for exponent in (1, 2, 5, 16, 64):
            cases[f"matpow-{exponent}"] = matpow(InitNode((3, 3), seed=1, name=X), exponent)
        cases["rank0-dot"] = sin(InitNode((4,), seed=1) @ InitNode((4,), seed=2), terms=6)
        cases["rank0-init"] = cos(InitNode((), seed=1), terms=5)
        cases["rank8"] = sin(InitNode((1,) * 8, seed=1), terms=4)
        return cases

    def test_every_expansion_conforms_to_the_contract(
        self, validator: Draft202012Validator
    ) -> None:
        for label, output in self._catalogue().items():
            document = serialize(output, dag_id="hygiene")
            try:
                assert_conforms(validator, document)
            except AssertionError as exc:  # pragma: no cover - only on a real violation
                raise AssertionError(f"{label}: {exc}") from exc

    def test_every_expansion_declares_the_current_wire_version(self) -> None:
        for output in self._catalogue().values():
            assert serialize(output)["metadata"]["schema_version"] == "1.2.0"

    def test_expansions_emit_only_primitive_ops(self, validator: Draft202012Validator) -> None:
        """A composite must never appear on the wire; the engine implements primitives only."""
        allowed = set(
            validator.schema["$defs"]["node"]["properties"]["op"]["enum"]  # type: ignore[index]
        )
        for output in self._catalogue().values():
            assert {node["op"] for node in serialize(output)["nodes"]} <= allowed

    @pytest.mark.parametrize(
        ("name", "fn", "terms"),
        [("sin", sin, 10), ("cos", cos, 10), ("sinh", sinh, 12), ("exp", exp, 10)],
    )
    def test_no_multiply_duplicates_another_multiplys_operand_pair(
        self, name: str, fn: Callable[..., Node], terms: int
    ) -> None:
        """Power reuse, stated as a property: two multiplies never compute the same thing."""
        document = serialize(fn(InitNode((3,), seed=1, name=X), terms=terms))
        pairs = Counter(
            tuple(node["inputs"]) for node in document["nodes"] if node["op"] == "multiply"
        )
        assert all(count == 1 for count in pairs.values()), pairs

    @pytest.mark.parametrize(("name", "fn"), [("sin", sin), ("sinh", sinh)])
    def test_odd_series_squares_x_exactly_once(self, name: str, fn: Callable[..., Node]) -> None:
        document = serialize(fn(InitNode((3,), seed=1, name=X), terms=10))
        squares = [
            node
            for node in document["nodes"]
            if node["op"] == "multiply" and node["inputs"] == [X, X]
        ]
        assert len(squares) == 1

    def test_odd_series_uses_one_multiply_per_term_not_one_per_exponent(self) -> None:
        """The stride trick: ``sin(x, terms=10)`` reaches ``x**19`` in ten multiplies."""
        document = serialize(sin(InitNode((3,), seed=1, name=X), terms=10))
        assert sum(1 for node in document["nodes"] if node["op"] == "multiply") == 10

    def test_composites_compose(self, validator: Draft202012Validator) -> None:
        base = InitNode((3,), seed=1, name=X)
        square = InitNode((3, 3), seed=1, name="a")
        compositions = {
            "sin(pow(x,3))": sin(tpow(base, 3), terms=6),
            "pow(sin(x),4)": tpow(sin(base, terms=5), 4),
            "exp(cos(x))": exp(cos(base, terms=4), terms=4),
            "powmod(pow(x,3),5)": powmod(tpow(base, 3), 5, 97.0),
            "matpow(a*b,5)": matpow(square * InitNode((3, 3), seed=2), 5),
            "powmod -> add": powmod(base, 7, 97.0) + InitNode((3,), seed=9),
            "sin(cross(u,v))": sin(cross(base, InitNode((3,), seed=2)), terms=5),
            "mod(sin(x))": ModNode(sin(base, terms=5), 3.0),
            "multiply(sin,cos)": MultiplyNode(sin(base, terms=4), cos(base, terms=4)),
        }
        for label, output in compositions.items():
            document = serialize(output, dag_id="compose")
            try:
                assert_conforms(validator, document)
            except AssertionError as exc:  # pragma: no cover
                raise AssertionError(f"{label}: {exc}") from exc

    def test_composition_is_numerically_right(self) -> None:
        """``sin(x**3)`` must equal ``numpy.sin(x**3)``, not merely serialize."""
        values = np.array([0.4, -0.3, 0.1])
        got = run(sin(tpow(InitNode(values.shape, seed=1, name=X), 3), terms=12), values)
        assert np.allclose(got, np.sin(values**3), atol=1e-13, rtol=0)

    def test_a_shared_operand_is_emitted_once_with_several_consumers(self) -> None:
        """Diamond sharing across two composites: one ``x``, four consumers."""
        base = InitNode((3,), seed=1, name=X)
        document = serialize(sin(base, terms=5) + cos(base, terms=5))
        assert sum(1 for node in document["nodes"] if node["id"] == X) == 1
        consumers = [node["id"] for node in document["nodes"] if X in node.get("inputs", [])]
        assert len(consumers) > 1

    def test_no_cse_across_calls_is_the_documented_policy(self) -> None:
        """``sin(x) + cos(x)`` deliberately emits two separate ``x**2`` nodes."""
        base = InitNode((3,), seed=1, name=X)
        document = serialize(sin(base, terms=5) + cos(base, terms=5))
        squares = [
            node
            for node in document["nodes"]
            if node["op"] == "multiply" and node["inputs"] == [X, X]
        ]
        assert len(squares) == 2

    @pytest.mark.parametrize(
        ("label", "builder", "expected"),
        [
            ("pow 2**30", lambda b: tpow(b, 2**30), 30),
            ("powmod 2**30", lambda b: powmod(b, 2**30, 97.0), 61),
            ("sin 85 terms", lambda b: sin(b, terms=85), 254),
            ("exp 171 terms", lambda b: exp(b, terms=171), 511),
        ],
    )
    def test_large_expansions_stay_at_their_formula_size(
        self,
        validator: Draft202012Validator,
        label: str,
        builder: Callable[[Node], Node],
        expected: int,
    ) -> None:
        """A 2**30 exponent must cost 30 nodes, not blow up or recurse."""
        base = InitNode((3,), seed=1, name=X)
        output = builder(base)
        assert node_count(output, base) == expected
        assert_conforms(validator, serialize(output, dag_id="large"))

    @pytest.mark.parametrize(
        ("label", "builder"),
        [
            ("sin", lambda b: sin(b, terms=10)),
            ("powmod", lambda b: powmod(b, 1024, 97.0)),
            ("matpow", lambda b: matpow(InitNode((3, 3), seed=5, name="m"), 64)),
        ],
    )
    def test_expansions_serialize_byte_identically_on_rebuild(
        self, label: str, builder: Callable[[Node], Node]
    ) -> None:
        """Topology-derived IDs make a rebuilt identical script emit identical bytes."""
        first = json.dumps(serialize(builder(InitNode((3,), seed=1, name=X))), sort_keys=True)
        second = json.dumps(serialize(builder(InitNode((3,), seed=1, name=X))), sort_keys=True)
        assert first == second


class TestConstantTermDropsOperand:
    """The ``x`` vanishes surprise, which applies to four functions, not just ``pow``.

    ``pow(x, 0)`` is documented loudly as producing a graph that never mentions ``x``. The same
    thing happens for ``cos``, ``cosh``, and ``exp`` at ``terms == 1``, because the sole term is
    ``x**0``, the constant 1. That was undocumented before this pass. The behaviour is correct and
    the node counts match the published formulas; only the docstrings were silent.
    """

    @pytest.mark.parametrize(
        ("label", "builder"),
        [
            ("pow n=0", lambda b: tpow(b, 0)),
            ("cos terms=1", lambda b: cos(b, terms=1)),
            ("cosh terms=1", lambda b: cosh(b, terms=1)),
            ("exp terms=1", lambda b: exp(b, terms=1)),
        ],
    )
    def test_the_operand_is_dropped_as_unreachable(
        self, label: str, builder: Callable[[Node], Node]
    ) -> None:
        base = InitNode((3,), seed=7, name=X)
        document = serialize(builder(base))
        assert X not in [node["id"] for node in document["nodes"]]

    @pytest.mark.parametrize(
        ("label", "builder"),
        [
            ("sin terms=1", lambda b: sin(b, terms=1)),
            ("sinh terms=1", lambda b: sinh(b, terms=1)),
            ("cos terms=2", lambda b: cos(b, terms=2)),
            ("cosh terms=2", lambda b: cosh(b, terms=2)),
            ("exp terms=2", lambda b: exp(b, terms=2)),
            ("pow n=1", lambda b: tpow(b, 1)),
            ("pow n=2", lambda b: tpow(b, 2)),
        ],
    )
    def test_the_operand_survives_everywhere_else(
        self, label: str, builder: Callable[[Node], Node]
    ) -> None:
        base = InitNode((3,), seed=7, name=X)
        document = serialize(builder(base))
        assert X in [node["id"] for node in document["nodes"]]

    @pytest.mark.parametrize(("name", "fn"), [("cos", cos), ("cosh", cosh), ("exp", exp)])
    def test_the_constant_result_is_exactly_one(self, name: str, fn: Callable[..., Node]) -> None:
        """Whatever ``x`` was, one term of an even or all-parity series is the constant 1."""
        document = serialize(fn(InitNode((3,), seed=7, name=X), terms=1))
        assert np.allclose(evaluate(document, {})[document["outputs"][0]], 1.0)

    @pytest.mark.parametrize(("name", "fn"), [("cos", cos), ("cosh", cosh), ("exp", exp)])
    def test_the_docstring_warns_about_it(self, name: str, fn: Callable[..., Node]) -> None:
        """Regression guard on the documentation fix, per the design's own instruction."""
        assert fn.__doc__ is not None
        assert "does not depend on ``x`` at all" in fn.__doc__

    def test_pow_at_zero_still_warns_too(self) -> None:
        assert tpow.__doc__ is not None
        assert "does not depend on ``x`` at all" in tpow.__doc__


class TestScalarArgumentErrorsAreNotBuildErrors:
    """Bad scalar arguments are API misuse, never a ``DagBuildError``."""

    @pytest.mark.parametrize(
        ("label", "call"),
        [
            ("pow n=-1", lambda: tpow(InitNode((3,), seed=1), -1)),
            ("pow n=1.5", lambda: tpow(InitNode((3,), seed=1), 1.5)),  # type: ignore[arg-type]
            ("pow n=True", lambda: tpow(InitNode((3,), seed=1), True)),
            ("terms=0", lambda: sin(InitNode((3,), seed=1), terms=0)),
            ("terms=-3", lambda: sin(InitNode((3,), seed=1), terms=-3)),
            ("terms=True", lambda: sin(InitNode((3,), seed=1), terms=True)),
            ("terms=2.5", lambda: sin(InitNode((3,), seed=1), terms=2.5)),  # type: ignore[arg-type]
            ("terms=500", lambda: sin(InitNode((3,), seed=1), terms=500)),
            ("powmod m=0", lambda: powmod(InitNode((3,), seed=1), 3, 0.0)),
            ("powmod m=-7", lambda: powmod(InitNode((3,), seed=1), 3, -7.0)),
            ("powmod m=2.5", lambda: powmod(InitNode((3,), seed=1), 3, 2.5)),
            ("powmod m=inf", lambda: powmod(InitNode((3,), seed=1), 3, float("inf"))),
            ("powmod m=nan", lambda: powmod(InitNode((3,), seed=1), 3, float("nan"))),
            ("powmod m too big", lambda: powmod(InitNode((3,), seed=1), 3, 1e9)),
            ("matpow n=0", lambda: matpow(InitNode((3, 3), seed=1), 0)),
            ("matpow n=-2", lambda: matpow(InitNode((3, 3), seed=1), -2)),
        ],
    )
    def test_raises_value_error_and_not_a_dag_build_error(
        self, label: str, call: Callable[[], Node]
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            call()
        assert not isinstance(excinfo.value, DagBuildError)

    def test_a_non_numeric_modulus_is_a_type_error(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            powmod(InitNode((3,), seed=1), 3, True)
        assert not isinstance(excinfo.value, DagBuildError)

    def test_matpow_zero_names_the_identity_matrix_reason(self) -> None:
        with pytest.raises(ValueError, match="identity matrix"):
            matpow(InitNode((3, 3), seed=1), 0)


class TestMatpowShapeValidation:
    """Rank before squareness, and shape before the exponent."""

    @pytest.mark.parametrize(
        "operand",
        [InitNode((3,), seed=1), InitNode((2, 2, 2), seed=1), InitNode((), seed=1)],
    )
    def test_wrong_rank_is_a_dimensionality_error(self, operand: Node) -> None:
        with pytest.raises(DimensionalityError, match="rank-2 matrix"):
            matpow(operand, 2)

    def test_non_square_rank_two_is_a_shape_mismatch(self) -> None:
        with pytest.raises(ShapeMismatchError, match="square"):
            matpow(InitNode((2, 3), seed=1), 2)

    def test_rank_is_reported_before_squareness(self) -> None:
        """A caller who passed a vector wants "this needs a matrix", not "this needs square"."""
        with pytest.raises(DimensionalityError):
            matpow(InitNode((5,), seed=1), 2)

    def test_shape_is_reported_before_the_exponent(self) -> None:
        """Documented ordering consequence: a bad shape outranks a bad exponent."""
        with pytest.raises(DimensionalityError):
            matpow(InitNode((3,), seed=1), -1)
        with pytest.raises(ShapeMismatchError):
            matpow(InitNode((2, 3), seed=1), 0)


class TestMigrationCompleteness:
    """No stale claim from v1.0.0 or v1.1.0 may survive in code, tests, or docstrings."""

    def test_the_wire_version_is_current_everywhere(self) -> None:
        assert SCHEMA_VERSION == "1.2.0"
        document = serialize(sin(InitNode((3,), seed=1), terms=3))
        assert document["metadata"]["schema_version"] == "1.2.0"

    def test_node_times_node_builds_a_multiply_rather_than_raising(self) -> None:
        """The v1.1.0 behaviour was ``TypeError``; v1.2.0 flipped it."""
        left = InitNode((2, 2), seed=1)
        right = InitNode((2, 2), seed=2)
        product = left * right
        assert isinstance(product, MultiplyNode)
        assert product.op == "multiply"

    def test_star_still_never_means_a_contraction(self) -> None:
        """The original objection survives the flip: ``*`` is elementwise, ``@`` contracts."""
        left = InitNode((2, 3), seed=1)
        right = InitNode((3, 4), seed=2)
        with pytest.raises(ShapeMismatchError):
            left * right
        assert (left @ right).op == "dot_product"

    def test_init_has_no_rank_floor(self) -> None:
        assert InitNode((), seed=1).output_shape == ()
        document = serialize(InitNode((), seed=1, name=X))
        assert document["nodes"][0]["shape"] == []
        assert document["nodes"][0]["output_shape"] == []

    def test_subtraction_exists_and_lowers_to_two_nodes(self) -> None:
        """``a - b`` costs an ``add`` over a negating ``scale``, so two nodes above the operands."""
        left = InitNode((2,), seed=1)
        right = InitNode((2,), seed=2)
        difference = left - right
        assert difference.op == "add"
        emitted = Graph([difference], dag_id="sub").nodes()
        assert len(emitted) - 2 == 2
        assert sum(1 for node in emitted if node.op == "scale") == 1

    @pytest.mark.parametrize("source", sorted((REPO / "tasks" / "src").rglob("*.py")))
    def test_no_shipped_module_claims_the_old_behaviour(self, source: Path) -> None:
        text = source.read_text(encoding="utf-8")
        assert "no elementwise (Hadamard) product is supported" not in text
        assert 'SCHEMA_VERSION: Final[str] = "1.1.0"' not in text

    def test_dtype_promotes_through_multiply(self) -> None:
        narrow = InitNode((3,), seed=1, dtype="float32")
        wide = InitNode((3,), seed=2, dtype="float64")
        assert MultiplyNode(narrow, wide).dtype == "float64"
        assert MultiplyNode(wide, narrow).dtype == "float64"
        assert MultiplyNode(narrow, InitNode((3,), seed=3, dtype="float32")).dtype == "float32"

    def test_mod_never_promotes(self) -> None:
        """``mod``'s modulus is a scalar field, and a scalar never widens its operand."""
        assert ModNode(InitNode((3,), seed=1, dtype="float32"), 7.0).dtype == "float32"
        assert ModNode(InitNode((3,), seed=1, dtype="float64"), 7.0).dtype == "float64"

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_composites_preserve_the_operand_dtype(self, dtype: DType) -> None:
        output = sin(InitNode((3,), seed=1, dtype=dtype), terms=6)
        assert {node["dtype"] for node in serialize(output)["nodes"]} == {dtype}

    def test_powmod_picks_its_bound_from_the_operand_dtype(self) -> None:
        with pytest.raises(ValueError, match="exact in float32"):
            powmod(InitNode((3,), seed=1, dtype="float32"), 3, 5000.0)
        powmod(InitNode((3,), seed=1, dtype="float64"), 3, 5000.0)

    def test_pow_is_not_re_exported_and_cannot_shadow_the_builtin(self) -> None:
        assert "pow" not in tasks.__all__
        assert not hasattr(tasks, "pow")

    def test_the_new_primitives_are_publicly_reachable(self) -> None:
        expected = (
            "MultiplyNode",
            "ModNode",
            "sin",
            "cos",
            "exp",
            "sinh",
            "cosh",
            "matpow",
            "powmod",
        )
        for name in expected:
            assert name in tasks.__all__


class TestCrossProjectContract:
    """The document hpcctl validates is the document tasks emits."""

    def test_a_v120_document_with_multiply_and_mod_conforms(
        self, validator: Draft202012Validator
    ) -> None:
        base = InitNode((4,), seed=11, name=X)
        document = serialize(powmod(base, 10, 97.0) + sin(base, terms=4), dag_id="v120-cross")
        assert_conforms(validator, document)
        ops = {node["op"] for node in document["nodes"]}
        assert {"multiply", "mod"} <= ops

    def test_mod_nodes_always_carry_a_positive_modulus(self) -> None:
        document = serialize(powmod(InitNode((3,), seed=1, name=X), 10, 97.0))
        mods = [node for node in document["nodes"] if node["op"] == "mod"]
        assert mods
        for node in mods:
            assert node["modulus"] > 0
            assert "factor" not in node

    def test_multiply_nodes_carry_neither_factor_nor_modulus(self) -> None:
        document = serialize(sin(InitNode((3,), seed=1, name=X), terms=6))
        for node in document["nodes"]:
            if node["op"] == "multiply":
                assert "factor" not in node
                assert "modulus" not in node
                assert len(node["inputs"]) == 2

    @pytest.mark.parametrize("version", ["1.0.0", "1.1.0", "1.2.0"])
    def test_earlier_era_documents_still_validate(
        self, validator: Draft202012Validator, version: str
    ) -> None:
        """1.2.0 is additive, so every older document must remain legal."""
        document = {
            "metadata": {
                "schema_version": version,
                "dag_id": "legacy",
                "ordering": "topological",
            },
            "nodes": [
                {
                    "id": "a",
                    "op": "init",
                    "output_shape": [3],
                    "dtype": "float64",
                    "seed": 1,
                    "shape": [3],
                    "distribution": "ones",
                },
                {
                    "id": "b",
                    "op": "init",
                    "output_shape": [3],
                    "dtype": "float64",
                    "seed": 2,
                    "shape": [3],
                    "distribution": "ones",
                },
                {
                    "id": "c",
                    "op": "add",
                    "output_shape": [3],
                    "dtype": "float64",
                    "inputs": ["a", "b"],
                },
            ],
            "outputs": ["c"],
        }
        assert_conforms(validator, document)

    def test_a_rank_zero_dot_result_from_the_1_1_0_era_still_validates(
        self, validator: Draft202012Validator
    ) -> None:
        document = serialize(InitNode((4,), seed=1) @ InitNode((4,), seed=2), dag_id="legacy-r0")
        assert_conforms(validator, document)

    def test_the_op_enum_gained_exactly_two_members(self, validator: Draft202012Validator) -> None:
        enum = validator.schema["$defs"]["node"]["properties"]["op"]["enum"]  # type: ignore[index]
        assert set(enum) == {
            "init",
            "add",
            "multiply",
            "scale",
            "mod",
            "dot_product",
            "cross_product",
        }
