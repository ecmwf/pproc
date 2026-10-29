"""Formula parser and numpy evaluator.

This module implements a small expression language for per-grid arithmetic
over GRIB fields, used by the ``pproc-formula`` CLI and by pproc workflows
(such as the SSO pipeline) to evaluate expressions over numpy arrays. The
Python builtin for dynamic expression evaluation is *not* used; a
hand-written recursive-descent parser produces an AST that a numpy-based
visitor walks.

Grammar (informal, lowest to highest precedence):

    expr        := compare
    compare     := addsub ( ('>' | '<' | '>=' | '<=' | '==') addsub )?     # non-chaining
    addsub      := muldiv ( ('+' | '-') muldiv )*                          # left-assoc
    muldiv      := power  ( ('*' | '/') power )*                           # left-assoc
    power       := unary ( '^' power )?                                    # right-assoc; unary binds tighter
    unary       := '-' unary | primary                                     # unary higher than '^'
    primary     := NUMBER
                 | IDENT                                                   # variable or `pi`
                 | IDENT '(' arglist? ')'                                  # function call
                 | '(' expr ')'
    arglist     := expr (',' expr)*

Operators ``+ - * /`` and the comparisons map to the obvious numpy
ufuncs. ``^`` is **power** (``numpy.power``), not XOR. Comparisons
return ``float64`` arrays of ``0.0`` / ``1.0`` (not ``bool``) so that
results compose with subsequent arithmetic — this matches the ksh-script
idiom ``(K - Lprime) * ((K - Lprime) > 0)``.

The constant ``pi`` evaluates to ``numpy.pi``. Functions: ``sqrt``,
``abs``, ``atan2``, ``min``, ``max``.

Public API:

* :func:`evaluate_formula` — parse and evaluate a single expression.
* :func:`parse_variables` — split a ``--variables a;b;c`` string.

The parser is hand-written; ``eval`` / ``exec`` are not used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

__all__ = [
    "evaluate_formula",
    "parse_variables",
    "tokenize",
    "Token",
    "TokenKind",
]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TokenKind(Enum):
    """Lexical token kinds emitted by :func:`tokenize`."""

    NUMBER = "NUMBER"
    IDENT = "IDENT"
    OP = "OP"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    EOF = "EOF"


@dataclass
class Token:
    """A single lexical token produced by :func:`tokenize`."""

    kind: TokenKind
    text: str
    pos: int
    value: Optional[float] = None


_OPERATOR_CHARS = set("+-*/^<>=")
# Two-character operators (longest match first). Single-char fallbacks below.
_TWO_CHAR_OPS = {">=", "<=", "=="}
_ONE_CHAR_OPS = {"+", "-", "*", "/", "^", ">", "<"}


def tokenize(source: str) -> list[Token]:
    """Tokenise ``source`` into a list of :class:`Token` ending with EOF.

    Whitespace (including newlines and tabs) is ignored. Unknown characters
    raise :class:`SyntaxError` with the offending position.
    """
    tokens: list[Token] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        # Numeric literal (int or float, with optional exponent).
        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            start = i
            j = i
            seen_dot = False
            seen_exp = False
            while j < n:
                cj = source[j]
                if cj.isdigit():
                    j += 1
                elif cj == "." and not seen_dot and not seen_exp:
                    seen_dot = True
                    j += 1
                elif cj in ("e", "E") and not seen_exp:
                    seen_exp = True
                    j += 1
                    if j < n and source[j] in ("+", "-"):
                        j += 1
                else:
                    break
            text = source[start:j]
            try:
                value = float(text)
            except ValueError as exc:
                raise SyntaxError(
                    f"invalid numeric literal {text!r} at position {start} in {source!r}"
                ) from exc
            tokens.append(
                Token(kind=TokenKind.NUMBER, text=text, pos=start, value=value)
            )
            i = j
            continue
        # Identifier (variable name, function name, or `pi`).
        if ch.isalpha() or ch == "_":
            start = i
            j = i
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            text = source[start:j]
            tokens.append(Token(kind=TokenKind.IDENT, text=text, pos=start))
            i = j
            continue
        # Parentheses and comma.
        if ch == "(":
            tokens.append(Token(kind=TokenKind.LPAREN, text="(", pos=i))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token(kind=TokenKind.RPAREN, text=")", pos=i))
            i += 1
            continue
        if ch == ",":
            tokens.append(Token(kind=TokenKind.COMMA, text=",", pos=i))
            i += 1
            continue
        # Operators (two-char, then one-char).
        if ch in _OPERATOR_CHARS:
            two = source[i : i + 2]
            if two in _TWO_CHAR_OPS:
                tokens.append(Token(kind=TokenKind.OP, text=two, pos=i))
                i += 2
                continue
            if ch in _ONE_CHAR_OPS:
                tokens.append(Token(kind=TokenKind.OP, text=ch, pos=i))
                i += 1
                continue
            # `=` on its own is not allowed.
            raise SyntaxError(
                f"unexpected character {ch!r} at position {i} in {source!r}"
            )
        raise SyntaxError(f"unexpected character {ch!r} at position {i} in {source!r}")
    tokens.append(Token(kind=TokenKind.EOF, text="", pos=n))
    return tokens


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------


@dataclass
class Number:
    value: float
    pos: int = 0


@dataclass
class Variable:
    name: str
    pos: int = 0


@dataclass
class UnaryOp:
    op: str
    operand: "Node"
    pos: int = 0


@dataclass
class BinOp:
    op: str
    left: "Node"
    right: "Node"
    pos: int = 0


@dataclass
class Compare:
    op: str
    left: "Node"
    right: "Node"
    pos: int = 0


@dataclass
class FunctionCall:
    name: str
    args: list["Node"] = field(default_factory=list)
    pos: int = 0


Node = object  # Union of the above; kept loose to avoid Python <3.10 syntax.


# ---------------------------------------------------------------------------
# Parser (recursive descent)
# ---------------------------------------------------------------------------


_COMPARE_OPS = {">", "<", ">=", "<=", "=="}
_ADDSUB_OPS = {"+", "-"}
_MULDIV_OPS = {"*", "/"}

_FUNCTION_ARITY: dict[str, int] = {
    "sqrt": 1,
    "abs": 1,
    "atan2": 2,
    "min": 2,
    "max": 2,
}


class _Parser:
    def __init__(self, tokens: list[Token], source: str) -> None:
        self.tokens = tokens
        self.source = source
        self.idx = 0

    # ------------- helpers -------------

    @property
    def cur(self) -> Token:
        return self.tokens[self.idx]

    def advance(self) -> Token:
        tok = self.tokens[self.idx]
        self.idx += 1
        return tok

    def expect(self, kind: TokenKind, what: str) -> Token:
        tok = self.cur
        if tok.kind is not kind:
            raise SyntaxError(
                f"expected {what} but found {tok.text!r} at position {tok.pos} "
                f"in formula {self.source!r}"
            )
        return self.advance()

    def error(self, msg: str, pos: Optional[int] = None) -> SyntaxError:
        if pos is None:
            pos = self.cur.pos
        return SyntaxError(f"{msg} at position {pos} in formula {self.source!r}")

    # ------------- grammar -------------

    def parse(self) -> Node:
        node = self.parse_expr()
        if self.cur.kind is not TokenKind.EOF:
            raise self.error(f"unexpected token {self.cur.text!r}")
        return node

    def parse_expr(self) -> Node:
        return self.parse_compare()

    def parse_compare(self) -> Node:
        left = self.parse_addsub()
        if self.cur.kind is TokenKind.OP and self.cur.text in _COMPARE_OPS:
            op_tok = self.advance()
            right = self.parse_addsub()
            # Disallow chaining: `a < b < c` is a parse error.
            if self.cur.kind is TokenKind.OP and self.cur.text in _COMPARE_OPS:
                raise self.error(
                    f"chained comparison {self.cur.text!r} is not supported"
                )
            return Compare(op=op_tok.text, left=left, right=right, pos=op_tok.pos)
        return left

    def parse_addsub(self) -> Node:
        node = self.parse_muldiv()
        while self.cur.kind is TokenKind.OP and self.cur.text in _ADDSUB_OPS:
            op_tok = self.advance()
            right = self.parse_muldiv()
            node = BinOp(op=op_tok.text, left=node, right=right, pos=op_tok.pos)
        return node

    def parse_muldiv(self) -> Node:
        node = self.parse_power()
        while self.cur.kind is TokenKind.OP and self.cur.text in _MULDIV_OPS:
            op_tok = self.advance()
            right = self.parse_power()
            node = BinOp(op=op_tok.text, left=node, right=right, pos=op_tok.pos)
        return node

    def parse_power(self) -> Node:
        # Unary binds tighter than '^', so the base is a unary expression.
        # Right-associative: the exponent recurses into parse_power.
        base = self.parse_unary()
        if self.cur.kind is TokenKind.OP and self.cur.text == "^":
            op_tok = self.advance()
            exponent = self.parse_power()
            return BinOp(op="^", left=base, right=exponent, pos=op_tok.pos)
        return base

    def parse_unary(self) -> Node:
        if self.cur.kind is TokenKind.OP and self.cur.text == "-":
            op_tok = self.advance()
            operand = self.parse_unary()
            return UnaryOp(op="-", operand=operand, pos=op_tok.pos)
        if self.cur.kind is TokenKind.OP and self.cur.text == "+":
            # unary plus is a no-op
            self.advance()
            return self.parse_unary()
        return self.parse_primary()

    def parse_primary(self) -> Node:
        tok = self.cur
        if tok.kind is TokenKind.NUMBER:
            self.advance()
            return Number(value=tok.value, pos=tok.pos)
        if tok.kind is TokenKind.LPAREN:
            self.advance()
            inner = self.parse_expr()
            if self.cur.kind is not TokenKind.RPAREN:
                raise self.error("missing closing parenthesis", pos=tok.pos)
            self.advance()
            return inner
        if tok.kind is TokenKind.IDENT:
            self.advance()
            # Function call?
            if self.cur.kind is TokenKind.LPAREN:
                lparen = self.advance()
                args: list[Node] = []
                if self.cur.kind is not TokenKind.RPAREN:
                    args.append(self.parse_expr())
                    while self.cur.kind is TokenKind.COMMA:
                        self.advance()
                        args.append(self.parse_expr())
                if self.cur.kind is not TokenKind.RPAREN:
                    raise self.error(
                        "missing closing parenthesis in function call",
                        pos=lparen.pos,
                    )
                self.advance()
                return FunctionCall(name=tok.text, args=args, pos=tok.pos)
            return Variable(name=tok.text, pos=tok.pos)
        if tok.kind is TokenKind.OP:
            raise self.error(f"unexpected operator {tok.text!r}; expected a value")
        if tok.kind is TokenKind.RPAREN:
            raise self.error("unmatched closing parenthesis")
        if tok.kind is TokenKind.EOF:
            raise self.error("unexpected end of formula; expected a value")
        raise self.error(f"unexpected token {tok.text!r}")


def _parse(formula: str) -> tuple[Node, str]:
    tokens = tokenize(formula)
    parser = _Parser(tokens, formula)
    return parser.parse(), formula


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


_BINOP_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.true_divide,
    "^": np.power,
}

_COMPARE_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "==": np.equal,
}


def _collect_variable_names(node: Node) -> list[str]:
    """Return identifiers used as variables in the AST, in source order."""
    found: list[str] = []
    seen: set[str] = set()

    def walk(n: Node) -> None:
        if isinstance(n, Variable):
            if n.name != "pi" and n.name not in seen:
                seen.add(n.name)
                found.append(n.name)
        elif isinstance(n, UnaryOp):
            walk(n.operand)
        elif isinstance(n, (BinOp, Compare)):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, FunctionCall):
            for a in n.args:
                walk(a)
        # Number: nothing to do

    walk(node)
    return found


def _evaluate(node: Node, variables: dict, formula: str):
    if isinstance(node, Number):
        return node.value
    if isinstance(node, Variable):
        if node.name == "pi":
            return np.pi
        if node.name not in variables:
            raise NameError(
                f"undefined variable {node.name!r} at position {node.pos} "
                f"in formula {formula!r}"
            )
        return variables[node.name]
    if isinstance(node, UnaryOp):
        operand = _evaluate(node.operand, variables, formula)
        if node.op == "-":
            return np.negative(operand)
        raise SyntaxError(f"unknown unary operator {node.op!r} at position {node.pos}")
    if isinstance(node, BinOp):
        left = _evaluate(node.left, variables, formula)
        right = _evaluate(node.right, variables, formula)
        try:
            return _BINOP_FUNCS[node.op](left, right)
        except ValueError as exc:
            # Most likely shape mismatch in broadcasting.
            raise _shape_error(node, variables, formula, exc) from exc
    if isinstance(node, Compare):
        left = _evaluate(node.left, variables, formula)
        right = _evaluate(node.right, variables, formula)
        try:
            result = _COMPARE_FUNCS[node.op](left, right)
        except ValueError as exc:
            raise _shape_error(node, variables, formula, exc) from exc
        return np.asarray(result, dtype=np.float64)
    if isinstance(node, FunctionCall):
        return _evaluate_function(node, variables, formula)
    raise TypeError(f"unknown AST node type: {type(node).__name__}")


def _evaluate_function(node: "FunctionCall", variables: dict, formula: str):
    name = node.name
    if name not in _FUNCTION_ARITY:
        raise SyntaxError(
            f"unknown function {name!r} at position {node.pos} in formula {formula!r}"
        )
    expected = _FUNCTION_ARITY[name]
    if len(node.args) != expected:
        raise SyntaxError(
            f"function {name!r} expects {expected} argument(s), got "
            f"{len(node.args)} at position {node.pos} in formula {formula!r}"
        )
    args = [_evaluate(a, variables, formula) for a in node.args]
    try:
        if name == "sqrt":
            return np.sqrt(args[0])
        if name == "abs":
            return np.abs(args[0])
        if name == "atan2":
            return np.arctan2(args[0], args[1])
        if name == "min":
            return np.minimum(args[0], args[1])
        if name == "max":
            return np.maximum(args[0], args[1])
    except ValueError as exc:
        raise _shape_error(node, variables, formula, exc) from exc
    raise AssertionError(f"unhandled function {name!r}")  # pragma: no cover


def _shape_error(
    node: Node, variables: dict, formula: str, original: BaseException
) -> ValueError:
    """Build a descriptive shape-mismatch error.

    Reports which variable names participate in the offending sub-expression
    along with their shapes, so the caller can see at a glance which two
    inputs broadcast-clash.
    """
    names = _collect_variable_names(node)
    shape_info = []
    for name in names:
        if name in variables:
            arr = variables[name]
            shape = getattr(arr, "shape", None)
            if shape is None:
                shape_info.append(f"{name}=scalar")
            else:
                shape_info.append(f"{name}{shape}")
    detail = ", ".join(shape_info) if shape_info else "(no named variables)"
    return ValueError(
        f"shape mismatch evaluating sub-expression at position {getattr(node, 'pos', 0)} "
        f"in formula {formula!r}: variables involved: {detail}. "
        f"Underlying error: {original}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_formula(formula: str, variables: dict) -> np.ndarray:
    """Evaluate a single formula expression.

    Parameters
    ----------
    formula:
        The expression text. May contain newlines and arbitrary whitespace.
        A single formula returns a single value; the multi-output ``;``
        separator accepted by ``pproc-formula --formula`` is the
        responsibility of the CLI layer (it splits and calls this function
        once per sub-expression).
    variables:
        Mapping from identifier name to numpy array (or scalar). The constant
        ``pi`` is always available and need not be in ``variables``.

    Returns
    -------
    numpy.ndarray
        The evaluation result. For comparison operators the result is a
        ``float64`` array of ``0.0`` / ``1.0`` so it composes with subsequent
        arithmetic.

    Raises
    ------
    SyntaxError
        Tokeniser or parser errors (unbalanced parens, lone operator,
        unknown function, malformed numeric literal, …). The message
        includes the offending position and the formula text.
    NameError
        An identifier in the expression is not in ``variables`` (and is
        not the ``pi`` constant). The message includes the missing name
        and the formula text.
    ValueError
        Numpy broadcasting failed. The message names the variables involved
        in the clashing sub-expression along with their shapes.
    """
    if not isinstance(formula, str):
        raise TypeError(f"formula must be a str, got {type(formula).__name__}")
    ast, src = _parse(formula)
    return _evaluate(ast, variables, src)


def parse_variables(variables_str: str) -> list[str]:
    """Split a ``--variables a;b;c`` string into a list of names.

    Whitespace around names is stripped. An empty input string returns an
    empty list. Empty segments (e.g. ``"a;;b"`` or a trailing ``";"``) raise
    :class:`ValueError`; empty variable names are not allowed.
    """
    if not isinstance(variables_str, str):
        raise TypeError(
            f"variables_str must be a str, got {type(variables_str).__name__}"
        )
    if variables_str.strip() == "":
        return []
    parts = variables_str.split(";")
    out: list[str] = []
    for raw in parts:
        name = raw.strip()
        if name == "":
            raise ValueError(
                f"empty variable name in {variables_str!r}; "
                "--variables does not allow empty entries"
            )
        out.append(name)
    return out
