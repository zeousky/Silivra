"""Bounded pure arithmetic behavior, precision and rejection of non-arithmetic."""
from decimal import localcontext, ROUND_UP
import unittest

from silivra.arithmetic import CalculationError, calculate


class ArithmeticTests(unittest.TestCase):
    def assertResult(self, expression, expected, *, rounded=False):
        result = calculate(expression)
        self.assertEqual(result["result"], expected)
        self.assertEqual(result["rounded"], rounded)
        self.assertEqual(result["precision_digits"], 50)
        self.assertEqual(result["expression"], expression.strip())

    def test_precedence_parentheses_and_observed_model_failures(self):
        self.assertResult("37 * 14 - 59", "459")
        self.assertResult("3 * 7 + 2 * 4", "29")
        self.assertResult("  (125 - 38) * 3  ", "261")
        self.assertResult("10 / (2 + 3)", "2")

    def test_decimal_literals_preserve_base_ten_arithmetic(self):
        self.assertResult("0.1 + 0.2", "0.3")
        self.assertResult("1.25 * 0.8", "1")
        self.assertResult("0.00000000000000000001 * 100", "0.000000000000000001")
        self.assertResult("1_000 + 2.50e1", "1025")

    def test_exact_rational_intermediates_and_explicit_rounding(self):
        self.assertResult("1 / 3 * 3", "1")
        self.assertResult("1 / 3", "0." + "3" * 50, rounded=True)
        self.assertResult("1 / 8", "0.125")
        self.assertResult("-0.0", "0")

    def test_floor_division_and_modulo_have_python_sign_semantics(self):
        for expression, expected in (("-7 // 2", "-4"), ("-7 % 2", "1"),
                                     ("7 // -2", "-4"), ("7 % -2", "-1"),
                                     ("7.5 % 2", "1.5"), ("--4 + +2", "6")):
            with self.subTest(expression=expression):
                self.assertResult(expression, expected)

    def test_result_is_independent_of_callers_decimal_context(self):
        with localcontext() as context:
            context.prec = 2
            context.Emax = 2
            context.rounding = ROUND_UP
            self.assertResult("1 / 3", "0." + "3" * 50, rounded=True)
            self.assertResult("1e100", "1" + "0" * 100)

    def test_division_by_zero_is_explicit_for_each_division_operator(self):
        for expression in ("1/0", "1//0", "1%0", "1/(2-2)"):
            with self.subTest(expression=expression), self.assertRaises(CalculationError) as raised:
                calculate(expression)
            self.assertEqual(raised.exception.code, "division_by_zero")

    def test_names_calls_attributes_collections_and_powers_are_refused(self):
        for expression in ("x + 2", "__import__('os').getcwd()", "open('anything')",
                           "(1).__class__", "[1, 2]", "{'n': 1}", "2 ** 10",
                           "True + 1", "1j", "1 < 2", "1 if 1 else 2", "0xff"):
            with self.subTest(expression=expression), self.assertRaises(CalculationError) as raised:
                calculate(expression)
            self.assertEqual(raised.exception.code, "invalid_expression")

    def test_argument_type_text_and_ast_budgets(self):
        for expression in (None, 3, True, "", "   ", "1" * 513):
            with self.subTest(expression=expression), self.assertRaises(CalculationError) as raised:
                calculate(expression)
            self.assertEqual(raised.exception.code, "invalid_arguments")
        for expression in ("+".join(["1"] * 30), "-" * 17 + "1"):
            with self.subTest(expression=expression), self.assertRaises(CalculationError) as raised:
                calculate(expression)
            self.assertEqual(raised.exception.code, "calculation_budget")
        for expression in ("1 +", "1; 2", "1\x00+2"):
            with self.subTest(expression=expression), self.assertRaises(CalculationError):
                calculate(expression)

    def test_literal_and_intermediate_numeric_budgets(self):
        self.assertResult("1e100", "1" + "0" * 100)
        for expression in ("1e101", "1e100 * 10", "1e-101", "1e-999999999",
                           "*".join(["1e-100"] * 13)):
            with self.subTest(expression=expression), self.assertRaises(CalculationError) as raised:
                calculate(expression)
            self.assertEqual(raised.exception.code, "calculation_budget")


if __name__ == "__main__":
    unittest.main()
