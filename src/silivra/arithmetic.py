"""Pure bounded arithmetic: exact rational operations, 50-digit decimal output.

Only numeric literals, parentheses and + - * / // % with unary + or - are
accepted. AST nodes are interpreted directly; no generated code is executed.
Floor division and remainder follow Python's floor/sign semantics. Repeating or
long decimal answers are explicitly marked rounded at 50 significant digits.
"""
from __future__ import annotations

import ast
from decimal import Context, Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import operator


MAX_EXPRESSION_CHARS = 512
MAX_AST_NODES = 64
MAX_DEPTH = 16
MAX_OPERATIONS = 64
MAX_RATIONAL_BITS = 4096
PRECISION_DIGITS = 50
_MAX_MAGNITUDE = Fraction(10**100)
_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
              ast.Mod: operator.mod}


class CalculationError(ValueError):
    """A stable public failure code without executing or exposing host state."""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def calculate(expression: str) -> dict:
    """Evaluate one bounded arithmetic expression and return decimal text.

    Inputs are decimal Python numeric literals; hexadecimal, complex, boolean,
    nonfinite and excessively small/large literals are rejected. Rational storage
    is bounded after every operation, even when near-cancellation hides its size.
    """
    if (not isinstance(expression, str) or not expression.strip()
            or len(expression) > MAX_EXPRESSION_CHARS):
        raise CalculationError("invalid_arguments", "expression must contain 1–512 characters of arithmetic.")
    expression = expression.strip()
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, OverflowError):
        raise CalculationError("invalid_expression", "Expected one numeric arithmetic expression.") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise CalculationError("calculation_budget", "Expression exceeds the 64-node arithmetic budget.")
    remaining = MAX_OPERATIONS

    def bounded(value):
        value = Fraction(value)
        if (value.numerator.bit_length() > MAX_RATIONAL_BITS
                or value.denominator.bit_length() > MAX_RATIONAL_BITS
                or abs(value) > _MAX_MAGNITUDE):
            raise CalculationError("calculation_budget", "Numeric value exceeds the arithmetic size budget.")
        return value

    def interpret(node, depth=0):
        nonlocal remaining
        if depth > MAX_DEPTH:
            raise CalculationError("calculation_budget", "Expression exceeds the 16-level arithmetic depth budget.")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            source = ast.get_source_segment(expression, node)
            try:
                value = Decimal(source.replace("_", ""))
            except (InvalidOperation, ValueError, AttributeError):
                raise CalculationError("invalid_expression", "Only decimal numeric literals are supported.") from None
            # Validate the exponent before constructing powers of ten in Fraction.
            if (not value.is_finite() or (value and not -100 <= value.adjusted() <= 100)
                    or value.copy_abs() > Decimal("1e100")):
                raise CalculationError("calculation_budget", "Numeric literal exceeds the supported magnitude/exponent range.")
            return bounded(Fraction(value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            remaining -= 1
            if remaining < 0:
                raise CalculationError("calculation_budget", "Expression exceeds the 64-operation arithmetic budget.")
            value = interpret(node.operand, depth + 1)
            return bounded(-value if isinstance(node.op, ast.USub) else value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            remaining -= 1
            if remaining < 0:
                raise CalculationError("calculation_budget", "Expression exceeds the 64-operation arithmetic budget.")
            left, right = interpret(node.left, depth + 1), interpret(node.right, depth + 1)
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise CalculationError("division_by_zero", "Division or remainder by zero is undefined.")
            return bounded(_OPERATORS[type(node.op)](left, right))
        raise CalculationError("invalid_expression", "Only numeric literals and + - * / // % arithmetic are supported.")

    value = interpret(tree.body)
    with localcontext(Context(prec=PRECISION_DIGITS, rounding=ROUND_HALF_EVEN,
                              Emin=-10_000, Emax=10_000)):
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
        rendered = format(decimal, "f")
        rounded = Fraction(decimal) != value
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return {"expression": expression, "result": rendered,
            "rounded": rounded, "precision_digits": PRECISION_DIGITS}
