"""Node-count and structural tests for the Tier-2 composites.

**The node counts here are contractual, not incidental.** ``pow(x, 1024)`` costing exactly ten
multiply nodes *is* the binary-exponentiation claim, so it is asserted as an equality rather than an
inequality. The same goes for the series formulas: they are exact only because unit coefficients are
deliberately not optimized away, so a change that "helpfully" elides a scale by 1.0 must fail here.

Depth is asserted against measured anchors rather than a formula. It has no tidy closed form for the
series, because the deepest power lands in the last slot of the summation tree and is carried
unpaired through some levels.
"""

from collections.abc import Callable

import pytest

from tasks import Graph, InitNode, Node, cos, cosh, exp, matpow, powmod, sin, sinh
from tasks.exceptions import DimensionalityError, ShapeMismatchError
from tasks.math import multiplies, safe_modulus_limit
from tasks.math import pow as tpow

Series = Callable[..., Node]


def emitted(build: Callable[..., Node], *, shape: tuple[int, ...] = (3,)) -> int:
    """Count the nodes a composite adds, excluding its operand.

    Args:
        build: Callable taking the operand node and returning the result node. Typed loosely
            so call sites may bind loop variables as lambda defaults.
        shape: Shape of the operand.

    Returns:
        The number of nodes the composite emitted.
    """
    x = InitNode(shape, seed=1)
    nodes = Graph([build(x)], dag_id="c").nodes()
    return len(nodes) - (1 if x in nodes else 0)


def depth(output: Node) -> int:
    """Return the longest path from any source to ``output``.

    Args:
        output: Node to measure to.

    Returns:
        Edge count of the longest input chain.
    """
    memo: dict[int, int] = {}

    def walk(node: Node) -> int:
        if id(node) in memo:
            return memo[id(node)]
        memo[id(node)] = 0 if not node.inputs else 1 + max(walk(i) for i in node.inputs)
        return memo[id(node)]

    return walk(output)


class TestSeriesNodeCounts:
    """The measured tables from the design, asserted exactly for N = 1..8."""

    @pytest.mark.parametrize(
        ("name", "fn", "expected"),
        [
            ("sin", sin, [1, 5, 8, 11, 14, 17, 20, 23]),
            ("sinh", sinh, [1, 5, 8, 11, 14, 17, 20, 23]),
            ("cos", cos, [2, 5, 8, 11, 14, 17, 20, 23]),
            ("cosh", cosh, [2, 5, 8, 11, 14, 17, 20, 23]),
            ("exp", exp, [2, 4, 7, 10, 13, 16, 19, 22]),
        ],
    )
    def test_measured_table(self, name: str, fn: Series, expected: list[int]) -> None:
        got = [emitted(lambda x, n=n, fn=fn: fn(x, terms=n)) for n in range(1, 9)]
        assert got == expected

    @pytest.mark.parametrize("fn", [sin, sinh], ids=["sin", "sinh"])
    @pytest.mark.parametrize("terms", range(2, 12))
    def test_odd_formula_is_3n_minus_1(self, fn: Series, terms: int) -> None:
        assert emitted(lambda x, fn=fn, t=terms: fn(x, terms=t)) == 3 * terms - 1

    @pytest.mark.parametrize("fn", [cos, cosh], ids=["cos", "cosh"])
    @pytest.mark.parametrize("terms", range(2, 12))
    def test_even_formula_is_3n_minus_1(self, fn: Series, terms: int) -> None:
        assert emitted(lambda x, fn=fn, t=terms: fn(x, terms=t)) == 3 * terms - 1

    @pytest.mark.parametrize("terms", range(2, 12))
    def test_exp_formula_is_3n_minus_2(self, terms: int) -> None:
        assert emitted(lambda x, t=terms: exp(x, terms=t)) == 3 * terms - 2

    def test_sin_terms_10_is_29_nodes(self) -> None:
        """The headline 'a sine expands into many nodes' figure."""
        assert emitted(lambda x: sin(x, terms=10)) == 29

    def test_unit_coefficients_are_not_elided(self) -> None:
        """sin(x, terms=1) still emits a scale by exactly 1.0; the formulas depend on it."""
        x = InitNode((3,), seed=1)
        document = Graph([sin(x, terms=1)], dag_id="one").serialize(include_timestamp=False)
        scales = [n for n in document["nodes"] if n["op"] == "scale"]
        assert len(scales) == 1
        assert scales[0]["factor"] == 1.0


class TestSeriesStructure:
    def test_default_terms_is_eight(self) -> None:
        assert emitted(sin) == 3 * 8 - 1

    @pytest.mark.parametrize("fn", [sin, sinh, cos, cosh, exp])
    def test_shape_is_preserved(self, fn: Series) -> None:
        x = InitNode((2, 3), seed=1)
        assert fn(x, terms=4).output_shape == (2, 3)

    @pytest.mark.parametrize("fn", [sin, sinh, cos, cosh, exp])
    def test_rank_zero_operand_works(self, fn: Series) -> None:
        """cos(u @ v) is the motivating case for lifting the init rank floor."""
        u, v = InitNode((3,), seed=1), InitNode((3,), seed=2)
        assert fn(u @ v, terms=4).output_shape == ()

    @pytest.mark.parametrize("fn", [sin, sinh, cos, cosh, exp])
    def test_dtype_is_inherited_not_promoted(self, fn: Series) -> None:
        """Coefficients are folded into scale factors, and a factor never promotes."""
        x = InitNode((3,), seed=1, dtype="float32")
        assert fn(x, terms=5).dtype == "float32"

    @pytest.mark.parametrize(("fn", "power_op"), [(cos, "init"), (cosh, "init"), (exp, "init")])
    def test_even_and_exp_emit_a_ones_node(self, fn: Series, power_op: str) -> None:
        x = InitNode((3,), seed=1)
        document = Graph([fn(x, terms=3)], dag_id="ones").serialize(include_timestamp=False)
        ones = [
            n for n in document["nodes"] if n["op"] == "init" and n.get("distribution") == "ones"
        ]
        assert len(ones) == 1
        assert ones[0]["shape"] == [3]

    def test_sin_emits_no_ones_node(self) -> None:
        """Odd parity never needs power 0."""
        x = InitNode((3,), seed=1)
        document = Graph([sin(x, terms=3)], dag_id="noones").serialize(include_timestamp=False)
        assert not [n for n in document["nodes"] if n.get("distribution") == "ones"]

    @pytest.mark.parametrize("fn", [sin, sinh, cos, cosh, exp])
    def test_every_node_is_labelled_with_the_prefix(self, fn: Series) -> None:
        x = InitNode((3,), seed=1, name="x")
        document = Graph([fn(x, terms=4)], dag_id="lbl").serialize(include_timestamp=False)
        emitted_nodes = [n for n in document["nodes"] if n["id"] != "x"]
        assert emitted_nodes
        for node in emitted_nodes:
            assert node["label"].startswith(f"{fn.__name__}/")

    def test_label_prefix_is_configurable(self) -> None:
        x = InitNode((3,), seed=1, name="x")
        document = Graph([sin(x, terms=3, label_prefix="wave")], dag_id="lp").serialize(
            include_timestamp=False
        )
        for node in (n for n in document["nodes"] if n["id"] != "x"):
            assert node["label"].startswith("wave/")

    def test_labels_stay_within_the_schema_limit(self) -> None:
        x = InitNode((3,), seed=1)
        for node in Graph([sin(x, terms=40)], dag_id="long").nodes():
            if node.label is not None:
                assert len(node.label) <= 128

    def test_no_cross_call_cse(self) -> None:
        """sin(x) + cos(x) builds x**2 twice; deliberate for v1."""
        x = InitNode((3,), seed=1)
        document = Graph([sin(x, terms=4) + cos(x, terms=4)], dag_id="cse").serialize(
            include_timestamp=False
        )
        pow2_labels = [n for n in document["nodes"] if n.get("label", "").endswith("/pow2")]
        assert len(pow2_labels) == 2


class TestSeriesDepth:
    """Measured anchors from the design at terms=10."""

    @pytest.mark.parametrize(
        ("fn", "expected"),
        [(sin, 13), (sinh, 13), (cos, 12), (cosh, 12), (exp, 11)],
        ids=["sin", "sinh", "cos", "cosh", "exp"],
    )
    def test_depth_anchor(self, fn: Series, expected: int) -> None:
        x = InitNode((3,), seed=1)
        assert depth(fn(x, terms=10)) == expected

    def test_series_are_wide_and_shallow_relative_to_pow(self) -> None:
        """The contrast the composite family exists to provide."""
        x = InitNode((3,), seed=1)
        series = sin(x, terms=10)
        chain = powmod(x, 1024, 97)
        assert emitted(lambda y: sin(y, terms=10)) == 29
        assert depth(series) == 13
        assert emitted(lambda y: powmod(y, 1024, 97)) == 21
        assert depth(chain) == 21


class TestTermsValidation:
    @pytest.mark.parametrize("fn", [sin, sinh, cos, cosh, exp])
    def test_zero_terms_is_rejected(self, fn: Series) -> None:
        with pytest.raises(ValueError, match="int >= 1"):
            fn(InitNode((3,), seed=1), terms=0)

    @pytest.mark.parametrize("fn", [sin, sinh, cos, cosh, exp])
    def test_negative_terms_is_rejected(self, fn: Series) -> None:
        with pytest.raises(ValueError, match="int >= 1"):
            fn(InitNode((3,), seed=1), terms=-3)

    @pytest.mark.parametrize("fn", [sin, sinh, cos, cosh, exp])
    def test_bool_terms_is_rejected(self, fn: Series) -> None:
        """Bool is an int subclass, so terms=True would silently mean terms=1."""
        with pytest.raises(ValueError, match="int >= 1"):
            fn(InitNode((3,), seed=1), terms=True)

    @pytest.mark.parametrize("fn", [sin, sinh, cos, cosh, exp])
    def test_float_terms_is_rejected(self, fn: Series) -> None:
        with pytest.raises(ValueError, match="int >= 1"):
            fn(InitNode((3,), seed=1), terms=4.0)

    def test_terms_must_be_keyword_only(self) -> None:
        """So a bare sin(x, 8) cannot be misread as a second operand."""
        with pytest.raises(TypeError):
            sin(InitNode((3,), seed=1), 8)  # type: ignore[call-arg]

    def test_sin_caps_at_85_terms(self) -> None:
        """Power 2N-1 hits 1/171! and overflows the float range."""
        x = InitNode((3,), seed=1)
        assert sin(x, terms=85).output_shape == (3,)
        with pytest.raises(ValueError, match="not representable"):
            sin(x, terms=86)

    def test_cos_accepts_86_terms_where_sin_does_not(self) -> None:
        """Even parity's highest power is 2(N-1), one step behind odd's 2N-1."""
        x = InitNode((3,), seed=1)
        assert cos(x, terms=86).output_shape == (3,)
        with pytest.raises(ValueError, match="not representable"):
            cos(x, terms=87)

    def test_exp_caps_at_171_terms(self) -> None:
        x = InitNode((3,), seed=1)
        assert exp(x, terms=171).output_shape == (3,)
        with pytest.raises(ValueError, match="not representable"):
            exp(x, terms=172)

    def test_overflow_message_names_the_term(self) -> None:
        with pytest.raises(ValueError, match=r"term \d+ needs 1/\d+!"):
            sin(InitNode((3,), seed=1), terms=86)


class TestPow:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [(1, 0), (2, 1), (3, 2), (7, 4), (10, 4), (16, 4), (100, 8), (1024, 10)],
    )
    def test_measured_node_counts(self, n: int, expected: int) -> None:
        assert emitted(lambda x, n=n: tpow(x, n)) == expected
        assert expected == multiplies(n)

    def test_1024_is_exactly_ten_multiplies(self) -> None:
        """The whole binary-exponentiation claim, as an equality."""
        assert emitted(lambda x: tpow(x, 1024)) == 10

    def test_exponent_one_returns_the_operand_itself(self) -> None:
        x = InitNode((3,), seed=1)
        assert tpow(x, 1) is x

    def test_exponent_zero_emits_a_lone_ones_node(self) -> None:
        x = InitNode((3,), seed=1)
        result = tpow(x, 0)
        assert result.op == "init"
        nodes = Graph([result], dag_id="zero").nodes()
        assert len(nodes) == 1

    def test_exponent_zero_does_not_reference_the_operand(self) -> None:
        """Documented surprise: a function of x yields a DAG that never mentions x."""
        x = InitNode((3,), seed=1)
        assert x not in Graph([tpow(x, 0)], dag_id="zero").nodes()

    def test_exponent_zero_matches_shape_and_dtype(self) -> None:
        x = InitNode((2, 3), seed=1, dtype="float32")
        result = tpow(x, 0)
        assert result.output_shape == (2, 3)
        assert result.dtype == "float32"

    @pytest.mark.parametrize("n", [-1, -2, -1024])
    def test_negative_exponent_is_rejected(self, n: int) -> None:
        with pytest.raises(ValueError, match="int >= 0"):
            tpow(InitNode((3,), seed=1), n)

    def test_depth_equals_node_count(self) -> None:
        """A pure serial chain, which is the scheduling-research value."""
        x = InitNode((3,), seed=1)
        assert depth(tpow(x, 1024)) == 10

    def test_all_nodes_are_multiplies(self) -> None:
        x = InitNode((3,), seed=1, name="x")
        document = Graph([tpow(x, 100)], dag_id="p").serialize(include_timestamp=False)
        assert [n["op"] for n in document["nodes"] if n["id"] != "x"] == ["multiply"] * 8

    def test_labels_are_prefixed(self) -> None:
        x = InitNode((3,), seed=1, name="x")
        document = Graph([tpow(x, 10)], dag_id="p").serialize(include_timestamp=False)
        for node in (n for n in document["nodes"] if n["id"] != "x"):
            assert node["label"].startswith("pow/")

    def test_works_on_rank_zero(self) -> None:
        u, v = InitNode((3,), seed=1), InitNode((3,), seed=2)
        assert tpow(u @ v, 8).output_shape == ()

    def test_not_exported_at_package_top_level(self) -> None:
        """`from tasks import pow` would shadow the builtin in the caller's namespace."""
        import tasks

        assert not hasattr(tasks, "pow")
        assert "pow" not in tasks.__all__


class TestPowmod:
    @pytest.mark.parametrize(("n", "expected"), [(1, 1), (5, 7), (10, 9), (100, 17), (1024, 21)])
    def test_measured_node_counts(self, n: int, expected: int) -> None:
        assert emitted(lambda x, n=n: powmod(x, n, 97)) == expected
        assert expected == 2 * multiplies(n) + 1

    def test_1024_is_21_nodes(self) -> None:
        assert emitted(lambda x: powmod(x, 1024, 97)) == 21

    def test_depth_equals_node_count(self) -> None:
        x = InitNode((3,), seed=1)
        assert depth(powmod(x, 1024, 97)) == 21

    def test_mod_follows_every_multiply(self) -> None:
        x = InitNode((3,), seed=1, name="x")
        document = Graph([powmod(x, 100, 97)], dag_id="pm").serialize(include_timestamp=False)
        ops = [n["op"] for n in document["nodes"] if n["id"] != "x"]
        assert ops.count("multiply") == multiplies(100)
        assert ops.count("mod") == multiplies(100) + 1

    def test_every_multiply_is_immediately_reduced(self) -> None:
        """Structural check: each multiply's only consumer is a mod."""
        x = InitNode((3,), seed=1, name="x")
        document = Graph([powmod(x, 100, 97)], dag_id="pm").serialize(include_timestamp=False)
        by_id = {n["id"]: n for n in document["nodes"]}
        consumers: dict[str, list[str]] = {n["id"]: [] for n in document["nodes"]}
        for node in document["nodes"]:
            for ref in node.get("inputs", []):
                consumers[ref].append(node["id"])
        for node in document["nodes"]:
            if node["op"] == "multiply":
                assert all(by_id[c]["op"] == "mod" for c in consumers[node["id"]])

    def test_base_is_reduced_first(self) -> None:
        x = InitNode((3,), seed=1, name="x")
        document = Graph([powmod(x, 8, 97)], dag_id="pm").serialize(include_timestamp=False)
        first = next(n for n in document["nodes"] if n["id"] != "x")
        assert first["op"] == "mod"
        assert first["inputs"] == ["x"]

    def test_exponent_zero_is_ones_then_mod(self) -> None:
        assert emitted(lambda x: powmod(x, 0, 97)) == 2
        x = InitNode((3,), seed=1)
        document = Graph([powmod(x, 0, 97)], dag_id="pz").serialize(include_timestamp=False)
        assert [n["op"] for n in document["nodes"]] == ["init", "mod"]

    def test_negative_exponent_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="int >= 0"):
            powmod(InitNode((3,), seed=1), -1, 97)

    @pytest.mark.parametrize("m", [0, -1, -97])
    def test_non_positive_modulus_is_rejected(self, m: float) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            powmod(InitNode((3,), seed=1), 5, m)

    def test_non_integral_modulus_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="integral"):
            powmod(InitNode((3,), seed=1), 5, 7.5)

    def test_bool_modulus_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="int or float"):
            powmod(InitNode((3,), seed=1), 5, True)

    def test_non_finite_modulus_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            powmod(InitNode((3,), seed=1), 5, float("inf"))


class TestPowmodExactnessBound:
    def test_float64_limit_is_94906266(self) -> None:
        assert safe_modulus_limit("float64") == 94906266

    def test_float32_limit_is_4097(self) -> None:
        assert safe_modulus_limit("float32") == 4097

    @pytest.mark.parametrize("dtype", ["float64", "float32"])
    def test_limit_squared_minus_one_fits(self, dtype: str) -> None:
        from tasks.math import EXACT_INTEGER_BITS

        limit = safe_modulus_limit(dtype)  # type: ignore[arg-type]
        cap = 2 ** EXACT_INTEGER_BITS[dtype]
        assert (limit - 1) ** 2 <= cap
        assert limit**2 > cap

    def test_modulus_at_the_limit_is_accepted(self) -> None:
        x = InitNode((3,), seed=1)
        assert powmod(x, 4, safe_modulus_limit("float64")).output_shape == (3,)

    def test_modulus_past_the_limit_is_rejected(self) -> None:
        x = InitNode((3,), seed=1)
        with pytest.raises(ValueError, match="exceeds the largest value exact"):
            powmod(x, 4, safe_modulus_limit("float64") + 1)

    def test_float32_bound_is_tighter(self) -> None:
        x32 = InitNode((3,), seed=1, dtype="float32")
        assert powmod(x32, 4, 4097).output_shape == (3,)
        with pytest.raises(ValueError, match="float32"):
            powmod(x32, 4, 4098)

    def test_allow_inexact_bypasses_the_bound(self) -> None:
        x = InitNode((3,), seed=1)
        result = powmod(x, 4, 10**12, allow_inexact=True)
        assert result.op == "mod"

    def test_error_names_the_computed_limit(self) -> None:
        with pytest.raises(ValueError, match="94906266"):
            powmod(InitNode((3,), seed=1), 4, 10**9)


class TestMatpow:
    @pytest.mark.parametrize(("n", "expected"), [(1, 0), (2, 1), (5, 3), (16, 4), (64, 6)])
    def test_measured_node_counts(self, n: int, expected: int) -> None:
        a = InitNode((4, 4), seed=1)
        nodes = Graph([matpow(a, n)], dag_id="m").nodes()
        assert len(nodes) - 1 == expected
        assert expected == multiplies(n)

    def test_64_is_six_dot_products(self) -> None:
        a = InitNode((4, 4), seed=1)
        document = Graph([matpow(a, 64)], dag_id="m").serialize(include_timestamp=False)
        assert [n["op"] for n in document["nodes"]].count("dot_product") == 6

    def test_exponent_one_returns_the_operand(self) -> None:
        a = InitNode((4, 4), seed=1)
        assert matpow(a, 1) is a

    def test_shape_is_preserved(self) -> None:
        a = InitNode((5, 5), seed=1)
        assert matpow(a, 8).output_shape == (5, 5)

    def test_depth_equals_node_count(self) -> None:
        a = InitNode((4, 4), seed=1)
        assert depth(matpow(a, 64)) == 6

    def test_exponent_zero_is_rejected(self) -> None:
        """No identity matrix is expressible: the distribution enum has no 'eye'."""
        a = InitNode((4, 4), seed=1)
        with pytest.raises(ValueError, match="identity matrix"):
            matpow(a, 0)

    def test_zero_rejection_mentions_the_enum_gap(self) -> None:
        a = InitNode((4, 4), seed=1)
        with pytest.raises(ValueError, match="eye"):
            matpow(a, 0)

    def test_negative_exponent_is_rejected(self) -> None:
        a = InitNode((4, 4), seed=1)
        with pytest.raises(ValueError, match="int >= 0"):
            matpow(a, -1)

    def test_non_square_is_a_shape_mismatch(self) -> None:
        a = InitNode((4, 3), seed=1)
        with pytest.raises(ShapeMismatchError, match="must be square"):
            matpow(a, 2)

    @pytest.mark.parametrize("shape", [(), (4,), (2, 2, 2)])
    def test_wrong_rank_is_a_dimensionality_error(self, shape: tuple[int, ...]) -> None:
        a = InitNode(shape, seed=1)
        with pytest.raises(DimensionalityError, match="rank-2 matrix"):
            matpow(a, 2)

    def test_rank_check_precedes_the_squareness_check(self) -> None:
        """Same ordering rationale as cross_product: rank is the coarser failure."""
        a = InitNode((4,), seed=1)
        with pytest.raises(DimensionalityError):
            matpow(a, 2)

    def test_labels_are_prefixed(self) -> None:
        a = InitNode((4, 4), seed=1, name="a")
        document = Graph([matpow(a, 10)], dag_id="m").serialize(include_timestamp=False)
        for node in (n for n in document["nodes"] if n["id"] != "a"):
            assert node["label"].startswith("matpow/")

    def test_emits_only_dot_products(self) -> None:
        a = InitNode((4, 4), seed=1, name="a")
        document = Graph([matpow(a, 100)], dag_id="m").serialize(include_timestamp=False)
        assert {n["op"] for n in document["nodes"] if n["id"] != "a"} == {"dot_product"}


class TestMultipliesFormula:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 2),
            (7, 4),
            (8, 3),
            (10, 4),
            (16, 4),
            (100, 8),
            (1024, 10),
        ],
    )
    def test_known_values(self, n: int, expected: int) -> None:
        assert multiplies(n) == expected

    @pytest.mark.parametrize("n", range(1, 200))
    def test_matches_the_documented_expression(self, n: int) -> None:
        assert multiplies(n) == (n.bit_length() - 1) + bin(n).count("1") - 1

    @pytest.mark.parametrize("power", range(1, 15))
    def test_powers_of_two_cost_log2_multiplies(self, power: int) -> None:
        assert multiplies(2**power) == power

    @pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 16, 31, 100, 1024])
    def test_agrees_with_the_actual_expansion(self, n: int) -> None:
        assert emitted(lambda x, n=n: tpow(x, n)) == multiplies(n)


class TestDirectCallValidation:
    """The composites are also callable directly, bypassing the operators' type dispatch.

    ``x ** 2.5`` returns ``NotImplemented`` from ``__pow__`` before reaching ``pow``, so these
    branches are only reachable through a direct call -- which untyped callers do make.
    """

    @pytest.mark.parametrize("bad", [2.5, "3", None, [2]])
    def test_pow_rejects_a_non_integer_exponent(self, bad: object) -> None:
        with pytest.raises(ValueError, match="int >= 0"):
            tpow(InitNode((3,), seed=1), bad)  # type: ignore[arg-type]

    def test_pow_rejects_a_bool_exponent(self) -> None:
        """Rejects True: bool is an int subclass, so it would otherwise mean 1."""
        with pytest.raises(ValueError, match="int >= 0"):
            tpow(InitNode((3,), seed=1), True)

    @pytest.mark.parametrize("bad", [2.5, "3"])
    def test_powmod_rejects_a_non_integer_exponent(self, bad: object) -> None:
        with pytest.raises(ValueError, match="int >= 0"):
            powmod(InitNode((3,), seed=1), bad, 97)  # type: ignore[arg-type]

    def test_matpow_rejects_a_non_integer_exponent(self) -> None:
        with pytest.raises(ValueError, match="int >= 0"):
            matpow(InitNode((4, 4), seed=1), 2.5)  # type: ignore[arg-type]

    def test_matpow_rejects_a_bool_exponent(self) -> None:
        with pytest.raises(ValueError, match="int >= 0"):
            matpow(InitNode((4, 4), seed=1), True)

    def test_error_names_the_offending_type(self) -> None:
        with pytest.raises(ValueError, match="got str"):
            tpow(InitNode((3,), seed=1), "3")  # type: ignore[arg-type]
