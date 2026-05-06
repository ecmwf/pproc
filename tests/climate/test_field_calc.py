"""Tests for pproc.climate.field_calc — mir-compute-compatible formula evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from pproc.climate.field_calc import (
    evaluate_formula,
    parse_variables,
)
from pproc.climate.field_calc import (
    tokenize,
    Token,
    TokenKind,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_numeric_literals(self):
        toks = tokenize("2 3.14 0.00000001 1e-8")
        kinds = [t.kind for t in toks if t.kind is not TokenKind.EOF]
        assert kinds == [TokenKind.NUMBER] * 4
        values = [t.value for t in toks if t.kind is TokenKind.NUMBER]
        assert values == [2.0, 3.14, 1e-8, 1e-8]

    def test_identifiers(self):
        toks = tokenize("a gradx K_Lprime_gt_0 f1 land_mask pi orog_N2000")
        idents = [t.text for t in toks if t.kind is TokenKind.IDENT]
        assert idents == [
            "a",
            "gradx",
            "K_Lprime_gt_0",
            "f1",
            "land_mask",
            "pi",
            "orog_N2000",
        ]

    def test_operators(self):
        toks = tokenize("+ - * / ^")
        ops = [t.text for t in toks if t.kind is TokenKind.OP]
        assert ops == ["+", "-", "*", "/", "^"]

    def test_comparison_operators(self):
        toks = tokenize("> < >= <= ==")
        ops = [t.text for t in toks if t.kind is TokenKind.OP]
        assert ops == [">", "<", ">=", "<=", "=="]

    def test_parens_and_comma(self):
        toks = tokenize("atan2(1, 2)")
        kinds = [t.kind for t in toks if t.kind is not TokenKind.EOF]
        assert kinds == [
            TokenKind.IDENT,
            TokenKind.LPAREN,
            TokenKind.NUMBER,
            TokenKind.COMMA,
            TokenKind.NUMBER,
            TokenKind.RPAREN,
        ]

    def test_whitespace_and_newlines(self):
        toks = tokenize(" a\n+\n  b\t")
        kinds = [t.kind for t in toks if t.kind is not TokenKind.EOF]
        assert kinds == [TokenKind.IDENT, TokenKind.OP, TokenKind.IDENT]

    def test_eof_terminator(self):
        toks = tokenize("1")
        assert toks[-1].kind is TokenKind.EOF

    def test_token_positions(self):
        toks = tokenize("a + b")
        positions = [t.pos for t in toks if t.kind is not TokenKind.EOF]
        assert positions == [0, 2, 4]

    def test_unknown_character_errors(self):
        with pytest.raises(SyntaxError) as excinfo:
            tokenize("a @ b")
        assert "@" in str(excinfo.value)

    def test_token_repr_smoke(self):
        # Smoke test: Token is a dataclass-like with repr
        t = Token(kind=TokenKind.NUMBER, text="2", pos=0, value=2.0)
        assert "NUMBER" in repr(t)


# ---------------------------------------------------------------------------
# Arithmetic, precedence, associativity
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_simple_add(self):
        assert evaluate_formula("1 + 2", {}) == 3.0

    def test_precedence_mul_over_add(self):
        # 2 + 3 * 4 == 14
        assert evaluate_formula("2 + 3 * 4", {}) == 14.0

    def test_subtraction_and_division(self):
        assert evaluate_formula("10 - 4 / 2", {}) == 8.0

    def test_left_assoc_subtraction(self):
        # 10 - 3 - 2 == 5, not 9
        assert evaluate_formula("10 - 3 - 2", {}) == 5.0

    def test_left_assoc_division(self):
        # 100 / 5 / 2 == 10, not 40
        assert evaluate_formula("100 / 5 / 2", {}) == 10.0

    def test_power_is_power_not_xor(self):
        assert evaluate_formula("2 ^ 3", {}) == 8.0

    def test_power_right_associative(self):
        # 2 ^ 3 ^ 2 == 2 ** (3 ** 2) == 512
        assert evaluate_formula("2 ^ 3 ^ 2", {}) == 512.0

    def test_unary_minus(self):
        assert evaluate_formula("-3", {}) == -3.0

    def test_unary_minus_on_paren(self):
        assert evaluate_formula("-(1 + 2)", {}) == -3.0

    def test_unary_minus_with_power(self):
        # Convention: unary - binds tighter than ^ per task spec
        # so -2 ^ 2 evaluates as (-2) ^ 2 == 4
        assert evaluate_formula("-2 ^ 2", {}) == 4.0

    def test_parens_override_precedence(self):
        assert evaluate_formula("(2 + 3) * 4", {}) == 20.0

    def test_float_literal(self):
        assert evaluate_formula("0.5 * 4", {}) == 2.0


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


class TestFunctions:
    def test_sqrt(self):
        assert evaluate_formula("sqrt(4)", {}) == 2.0

    def test_abs(self):
        assert evaluate_formula("abs(-3)", {}) == 3.0

    def test_atan2(self):
        np.testing.assert_allclose(evaluate_formula("atan2(1, 1)", {}), np.pi / 4)

    def test_min(self):
        assert evaluate_formula("min(2, 3)", {}) == 2.0

    def test_max(self):
        assert evaluate_formula("max(2, 3)", {}) == 3.0

    def test_min_with_arrays(self):
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([2.0, 4.0, 3.0])
        result = evaluate_formula("min(a, b)", {"a": a, "b": b})
        np.testing.assert_array_equal(result, [1.0, 4.0, 3.0])

    def test_max_with_arrays(self):
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([2.0, 4.0, 3.0])
        result = evaluate_formula("max(a, b)", {"a": a, "b": b})
        np.testing.assert_array_equal(result, [2.0, 5.0, 3.0])

    def test_sqrt_of_array(self):
        a = np.array([1.0, 4.0, 9.0])
        result = evaluate_formula("sqrt(a)", {"a": a})
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_pi(self):
        assert evaluate_formula("pi", {}) == np.pi

    def test_pi_in_expression(self):
        np.testing.assert_allclose(evaluate_formula("2 * pi", {}), 2 * np.pi)


# ---------------------------------------------------------------------------
# Comparisons (return float64 0/1 arrays)
# ---------------------------------------------------------------------------


class TestComparisons:
    def test_gt_returns_float64(self):
        a = np.array([-1.0, 0.0, 1.0])
        result = evaluate_formula("a > 0", {"a": a})
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])

    def test_lt(self):
        a = np.array([-1.0, 0.0, 1.0])
        result = evaluate_formula("a < 0", {"a": a})
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 0.0, 0.0])

    def test_ge(self):
        a = np.array([-1.0, 0.0, 1.0])
        result = evaluate_formula("a >= 0", {"a": a})
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [0.0, 1.0, 1.0])

    def test_le(self):
        a = np.array([-1.0, 0.0, 1.0])
        result = evaluate_formula("a <= 0", {"a": a})
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 1.0, 0.0])

    def test_eq(self):
        a = np.array([-1.0, 0.0, 1.0])
        result = evaluate_formula("a == 0", {"a": a})
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [0.0, 1.0, 0.0])

    def test_comparison_arithmetic_combination(self):
        # The ksh-script idiom: (K - Lprime) * ((K - Lprime) > 0)
        K = np.array([1.0, 2.0, -1.0])
        Lprime = np.array([0.0, 3.0, 0.5])
        result = evaluate_formula(
            "(K - Lprime) * ((K - Lprime) > 0)",
            {"K": K, "Lprime": Lprime},
        )
        np.testing.assert_array_equal(result, [1.0, 0.0, 0.0])

    def test_scalar_comparison_is_float64(self):
        result = evaluate_formula("3 > 2", {})
        # Scalar comparisons should also produce a 0/1 float64 (array or scalar).
        arr = np.asarray(result)
        assert arr.dtype == np.float64
        assert float(arr) == 1.0


# ---------------------------------------------------------------------------
# ksh-script formula coverage (verbatim from sso-migration learnings)
# ---------------------------------------------------------------------------


class TestKshScriptFormulae:
    def _grid(self, seed: int, shape=(3, 4)) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(-1.0, 1.0, size=shape)

    def test_formula_1_difference(self):
        a = self._grid(1)
        b = self._grid(2)
        result = evaluate_formula(
            "orog_N2000 - orog_egrid_N2000",
            {"orog_N2000": a, "orog_egrid_N2000": b},
        )
        np.testing.assert_array_equal(result, a - b)

    def test_formula_2_squared_difference(self):
        a = self._grid(1)
        b = self._grid(2)
        result = evaluate_formula(
            "(orog_N2000 - orog_egrid_N2000)^2",
            {"orog_N2000": a, "orog_egrid_N2000": b},
        )
        np.testing.assert_allclose(result, (a - b) ** 2)

    def test_formula_3_gradx_sq(self):
        gx = self._grid(3)
        gy = self._grid(4)
        result = evaluate_formula("gradx * gradx", {"gradx": gx, "grady": gy})
        np.testing.assert_array_equal(result, gx * gx)

    def test_formula_4_grady_sq(self):
        gx = self._grid(3)
        gy = self._grid(4)
        result = evaluate_formula("grady * grady", {"gradx": gx, "grady": gy})
        np.testing.assert_array_equal(result, gy * gy)

    def test_formula_5_gradxy(self):
        gx = self._grid(3)
        gy = self._grid(4)
        result = evaluate_formula("gradx * grady", {"gradx": gx, "grady": gy})
        np.testing.assert_array_equal(result, gx * gy)

    def test_formula_6_stdgwd(self):
        diff_sq = np.abs(self._grid(5))  # ensure non-negative for sqrt
        mask = np.array(
            np.random.default_rng(6).uniform(0, 1, diff_sq.shape) > 0.5,
            dtype=np.float64,
        )
        result = evaluate_formula(
            "sqrt(orog_mgrid_diff_sq) * land_mask",
            {"orog_mgrid_diff_sq": diff_sq, "land_mask": mask},
        )
        np.testing.assert_allclose(result, np.sqrt(diff_sq) * mask)

    def test_formula_7_K_expression(self):
        # K = 0.5*(gradxx+gradyy)
        gxx = self._grid(7)
        gyy = self._grid(8)
        result = evaluate_formula("0.5*(gradxx+gradyy)", {"gradxx": gxx, "gradyy": gyy})
        np.testing.assert_allclose(result, 0.5 * (gxx + gyy))

    def test_formula_7_L_expression(self):
        gxx = self._grid(7)
        gyy = self._grid(8)
        result = evaluate_formula("0.5*(gradxx-gradyy)", {"gradxx": gxx, "gradyy": gyy})
        np.testing.assert_allclose(result, 0.5 * (gxx - gyy))

    def test_formula_7_Lprime_expression(self):
        gxx = self._grid(7)
        gyy = self._grid(8)
        gxy = self._grid(9)
        result = evaluate_formula(
            "sqrt((0.5*(gradxx-gradyy))^2+(gradxy)^2)",
            {"gradxx": gxx, "gradyy": gyy, "gradxy": gxy},
        )
        expected = np.sqrt((0.5 * (gxx - gyy)) ** 2 + gxy**2)
        np.testing.assert_allclose(result, expected)

    def test_formula_8_slogwd(self):
        K = np.abs(self._grid(10))
        Lprime = np.abs(self._grid(11))
        mask = np.array(
            np.random.default_rng(12).uniform(0, 1, K.shape) > 0.5,
            dtype=np.float64,
        )
        result = evaluate_formula(
            "sqrt(K + Lprime) * land_mask",
            {"K": K, "Lprime": Lprime, "land_mask": mask},
        )
        np.testing.assert_allclose(result, np.sqrt(K + Lprime) * mask)

    def test_formula_9a_K_minus_Lprime_gt_0(self):
        K = self._grid(10)
        Lprime = self._grid(11)
        result = evaluate_formula("(K - Lprime) > 0", {"K": K, "Lprime": Lprime})
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, (K - Lprime > 0).astype(np.float64))

    def test_formula_9b_K_plus_Lprime_gt_epsilon(self):
        K = self._grid(10)
        Lprime = self._grid(11)
        result = evaluate_formula(
            "(K + Lprime) > 0.00000001", {"K": K, "Lprime": Lprime}
        )
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, (K + Lprime > 1e-8).astype(np.float64))

    def test_formula_10_isogwd_mixed_named_and_positional(self):
        # f1, f4 are passed as named entries in the variables dict (the CLI
        # layer is responsible for translating positional refs to named ones).
        # Use f1 >= f4 >= 0 to keep the sqrt argument non-negative.
        a = np.abs(self._grid(13)) + 0.5
        b = np.abs(self._grid(14)) * 0.4  # b < a guaranteed
        f1 = a
        f4 = b
        gt0 = np.array(
            np.random.default_rng(15).uniform(0, 1, f1.shape) > 0.5,
            dtype=np.float64,
        )
        gt_eps = np.ones_like(f1)  # always 1, so denominator is positive
        mask = np.array(
            np.random.default_rng(17).uniform(0, 1, f1.shape) > 0.5,
            dtype=np.float64,
        )
        formula = (
            "sqrt( ((f1 - f4) * K_Lprime_gt_0) / "
            "((f1 + f4) * K_Lprime_gt_epsilon + 0.00000001) ) * land_mask"
        )
        result = evaluate_formula(
            formula,
            {
                "f1": f1,
                "f4": f4,
                "K_Lprime_gt_0": gt0,
                "K_Lprime_gt_epsilon": gt_eps,
                "land_mask": mask,
            },
        )
        expected = np.sqrt(((f1 - f4) * gt0) / ((f1 + f4) * gt_eps + 1e-8)) * mask
        np.testing.assert_allclose(result, expected)

    def test_formula_11_anggwd(self):
        L = self._grid(18)
        M = self._grid(19)
        mask = np.array(
            np.random.default_rng(20).uniform(0, 1, L.shape) > 0.5,
            dtype=np.float64,
        )
        result = evaluate_formula(
            "0.5 * atan2(M, L) * land_mask",
            {"L": L, "M": M, "land_mask": mask},
        )
        np.testing.assert_allclose(result, 0.5 * np.arctan2(M, L) * mask)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_undefined_variable(self):
        with pytest.raises(NameError) as excinfo:
            evaluate_formula("a + b", {"a": np.array([1.0])})
        msg = str(excinfo.value)
        assert "b" in msg
        # formula text should appear in the error
        assert "a + b" in msg

    def test_unknown_function(self):
        with pytest.raises((SyntaxError, NameError)) as excinfo:
            evaluate_formula("foobar(1)", {})
        assert "foobar" in str(excinfo.value)

    def test_unbalanced_parens_open(self):
        with pytest.raises(SyntaxError) as excinfo:
            evaluate_formula("(1 + 2", {})
        # position info should be present
        assert "(1 + 2" in str(excinfo.value) or "paren" in str(excinfo.value).lower()

    def test_unbalanced_parens_close(self):
        with pytest.raises(SyntaxError):
            evaluate_formula("1 + 2)", {})

    def test_lone_operator(self):
        with pytest.raises(SyntaxError):
            evaluate_formula("+", {})

    def test_missing_operand(self):
        with pytest.raises(SyntaxError):
            evaluate_formula("1 +", {})

    def test_missing_function_arg(self):
        with pytest.raises(SyntaxError):
            evaluate_formula("sqrt()", {})

    def test_wrong_function_arity(self):
        with pytest.raises(SyntaxError):
            evaluate_formula("atan2(1)", {})

    def test_too_many_function_args(self):
        with pytest.raises(SyntaxError):
            evaluate_formula("sqrt(1, 2)", {})

    def test_shape_mismatch(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError) as excinfo:
            evaluate_formula("a + b", {"a": a, "b": b})
        msg = str(excinfo.value)
        # both variable names should be mentioned
        assert "a" in msg and "b" in msg

    def test_chained_comparison_disallowed(self):
        # comparisons are non-chaining
        with pytest.raises(SyntaxError):
            evaluate_formula("1 < 2 < 3", {})

    def test_unknown_character(self):
        with pytest.raises(SyntaxError):
            evaluate_formula("a $ b", {"a": np.array([1.0]), "b": np.array([2.0])})

    def test_eval_is_not_used(self):
        # Defensive: ensure no python eval/exec lurking in source
        import inspect
        from pproc.climate import field_calc

        src = inspect.getsource(field_calc)
        # plain `eval(` or `exec(` should not appear
        # (substring checks; sufficient to deter accidental use)
        assert "eval(" not in src
        assert "exec(" not in src


# ---------------------------------------------------------------------------
# parse_variables
# ---------------------------------------------------------------------------


class TestParseVariables:
    def test_simple(self):
        assert parse_variables("a;b;c") == ["a", "b", "c"]

    def test_whitespace_tolerant(self):
        assert parse_variables(" a ; b ;c ") == ["a", "b", "c"]

    def test_empty_string(self):
        assert parse_variables("") == []

    def test_single_name(self):
        assert parse_variables("a") == ["a"]

    def test_empty_segment_raises(self):
        with pytest.raises(ValueError):
            parse_variables("a;;b")

    def test_trailing_semicolon_raises(self):
        with pytest.raises(ValueError):
            parse_variables("a;b;")

    def test_only_whitespace_segment_raises(self):
        with pytest.raises(ValueError):
            parse_variables("a; ;b")
