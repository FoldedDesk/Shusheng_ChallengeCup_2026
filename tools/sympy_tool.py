from __future__ import annotations

import re
from math import gcd
from typing import Any, Optional


class SympyTool:
    """Optional local symbolic helper. Tool failures never block model solving."""

    def __init__(self) -> None:
        try:
            import sympy as sympy_module
        except ImportError:
            sympy_module = None
        self.sympy = sympy_module

    def derivative(self, expression: str, variable: str = "x") -> Optional[str]:
        return self._run(lambda s: s.diff(self._parse(expression), s.Symbol(variable)))

    def integral(self, expression: str, variable: str = "x") -> Optional[str]:
        return self._run(lambda s: s.integrate(self._parse(expression), s.Symbol(variable)))

    def definite_integral(
        self,
        expression: str,
        variable: str,
        lower: str,
        upper: str,
    ) -> Optional[str]:
        return self._run(
            lambda s: s.integrate(
                self._parse(expression),
                (s.Symbol(variable), self._parse(lower), self._parse(upper)),
            )
        )

    def solve_equation(self, expression: str, variable: str = "x") -> Optional[list]:
        if not self.sympy:
            return None
        try:
            return [self._format(item) for item in self.sympy.solve(self._parse(expression), self.sympy.Symbol(variable))]
        except Exception:
            return None

    def matrix(self, rows: list[list[Any]]) -> Optional[list[list[str]]]:
        if not self.sympy:
            return None
        try:
            return [[self._format(item) for item in row] for row in self.sympy.Matrix(rows).tolist()]
        except Exception:
            return None

    def limit(self, expression: str, variable: str, point: str) -> Optional[str]:
        return self._run(
            lambda s: s.limit(self._parse(expression), s.Symbol(variable), self._parse(point))
        )

    def evaluate(self, expression: str) -> Optional[str]:
        return self._run(lambda _: self._parse(expression))

    def hints_for(self, problem: str) -> list[str]:
        """Return safe, deterministic hints for elementary symbolic subproblems.

        This deliberately handles only unambiguous LaTex or plain-text forms.
        Anything it cannot parse is left to the model solver.
        """
        if not self.sympy:
            return []

        hints: list[str] = []
        arithmetic = re.search(
            r"(?:计算|求值)\s*([0-9A-Za-z_+\-*/^().,\s]+?)[。？?]?$", problem
        )
        if arithmetic and not re.search(r"积分|导数|极限|方程", problem):
            result = self.evaluate(arithmetic.group(1))
            if result is not None:
                hints.append(f"SymPy 计算: {result}")

        congruence = self._congruence_hint(problem)
        if congruence:
            hints.append(congruence)
        modular_power = self._modular_power_hint(problem)
        if modular_power:
            hints.append(modular_power)

        if re.search(r"导数|求导|微分", problem):
            partial = re.search(
                r"f\s*\(\s*[A-Za-z]\s*,\s*[A-Za-z]\s*\)\s*=\s*(?P<expression>[^，。；;]+?)\s*关于\s*\$?(?P<variable>[A-Za-z])\$?\s*的?(?:偏导|导数)",
                problem,
            )
            match = partial or re.search(
                r"(?:f\s*\(\s*(?P<variable>[A-Za-z])\s*\)|y)\s*=\s*(?P<expression>[^，。；;]+?)(?=\s*(?:的(?:导数|微分)|[,，。；;]|$))",
                problem,
            )
            if match:
                variable = match.group("variable") or "x"
                result = self.derivative(self._latex_to_sympy(match.group("expression")), variable)
                if result is not None:
                    label = "偏导数" if partial else "导数"
                    hints.append(f"SymPy {label}: {result}")

        math_parts = re.findall(r"\$([^$]+)\$", problem)
        math_parts.extend(self._raw_latex_parts(problem))
        math_parts.extend(self._plain_equations(problem))
        if re.search(r"积分|\\int", problem):
            for part in math_parts:
                definite = re.search(
                    r"\\int_\{?([^}\s]+)\}?\^\{?([^}\s]+)\}?\s*(.+?)(?:\\,|\s)*d([A-Za-z])\b",
                    part,
                )
                if definite:
                    result = self.definite_integral(
                        self._latex_to_sympy(definite.group(3)),
                        definite.group(4),
                        self._latex_to_sympy(definite.group(1)),
                        self._latex_to_sympy(definite.group(2)),
                    )
                    if result is not None:
                        hints.append(f"SymPy 定积分: {result}")
                    break
                match = re.search(r"\\int\s*(.+?)(?:\\,|\s)*d([A-Za-z])\b", part)
                if match:
                    result = self.integral(self._latex_to_sympy(match.group(1)), match.group(2))
                    if result is not None:
                        hints.append(f"SymPy 不定积分: {result}")
                    break

        if re.search(r"极限|\\lim", problem):
            for part in math_parts:
                match = re.search(r"\\lim_\{?\s*([A-Za-z])\s*\\to\s*([^}\s]+)\}?\s*(.+)", part)
                if match:
                    result = self.limit(
                        self._latex_to_sympy(match.group(3)), match.group(1), self._latex_to_sympy(match.group(2))
                    )
                    if result is not None:
                        hints.append(f"SymPy 极限: {result}")
                    break

        if re.search(r"方程|求解", problem):
            for part in math_parts:
                if "=" not in part or r"\begin" in part:
                    continue
                left, right = part.split("=", 1)
                variable = re.search(r"\b([xyz])\b", left + right)
                if variable:
                    expression = f"({self._latex_to_sympy(left)})-({self._latex_to_sympy(right)})"
                    result = self.solve_equation(expression, variable.group(1))
                    if result is not None:
                        if result:
                            answer = "，".join(f"{variable.group(1)}={item}" for item in result)
                        else:
                            answer = "无解"
                        hints.append(f"SymPy 方程解: {answer}")
                    break

        if re.search(r"矩阵|\\begin\{[pb]?matrix\}", problem):
            for part in math_parts:
                match = re.search(r"\\begin\{[pb]?matrix\}(.+?)\\end\{[pb]?matrix\}", part, re.DOTALL)
                if match:
                    rows = [
                        [self._latex_to_sympy(cell) for cell in row.split("&")]
                        for row in re.split(r"\\\\", match.group(1))
                    ]
                    result = self.matrix(rows)
                    if result is not None:
                        hints.append(f"SymPy 矩阵: {result}")
                    break
        return hints

    @staticmethod
    def _plain_equations(problem: str) -> list[str]:
        """Extract only short, ASCII-style equations outside LaTex delimiters."""
        if "$" in problem or not re.search(r"方程|求解", problem):
            return []
        matches = re.findall(
            r"([0-9xyzXYZ(][0-9A-Za-z_+\-*/^().,\s]{0,120}=[0-9A-Za-z_+\-*/^().,\s]{1,120})",
            problem,
        )
        return [match.strip() for match in matches]

    @staticmethod
    def _congruence_hint(problem: str) -> Optional[str]:
        normalized = (
            problem.replace(r"\equiv", "≡")
            .replace(r"\pmod{", " mod ")
            .replace(r"\pmod", " mod ")
            .replace("}", "")
        )
        match = re.search(
            r"(-?\d+)\s*\*?\s*x\s*≡\s*(-?\d+)\s*(?:mod\s*|\b)(\d+)",
            normalized,
        )
        if not match:
            return None
        coefficient, constant, modulus = map(int, match.groups())
        divisor = gcd(coefficient, modulus)
        if constant % divisor:
            return "本地同余方程：无解"
        if divisor != 1:
            return None
        solution = (pow(coefficient % modulus, -1, modulus) * constant) % modulus
        return f"本地同余方程解: x={solution} (mod {modulus})"

    @staticmethod
    def _modular_power_hint(problem: str) -> Optional[str]:
        normalized = problem.replace(r"\bmod", "mod").replace("{", "").replace("}", "")
        match = re.search(r"(-?\d+)\s*\^\s*(\d+)\s*mod\s*(\d+)", normalized)
        if not match:
            return None
        base, exponent, modulus = map(int, match.groups())
        if modulus == 0:
            return None
        return f"本地模幂计算: {pow(base, exponent, modulus)}"

    @staticmethod
    def _raw_latex_parts(problem: str) -> list[str]:
        """Find standalone raw LaTex limits and integrals without `$...$`."""
        return re.findall(r"(\\(?:lim|int)[^。？?\n]+)", problem)

    def _parse(self, expression: str):
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        if not re.fullmatch(r"[0-9A-Za-z_+\-*/^().,\s]+", expression):
            raise ValueError("unsupported symbolic expression")
        identifiers = set(re.findall(r"[A-Za-z]+", expression))
        allowed = {"sin", "cos", "tan", "asin", "acos", "atan", "log", "exp", "sqrt", "pi", "oo"}
        if any(identifier not in allowed and len(identifier) != 1 for identifier in identifiers):
            raise ValueError("unsupported symbolic identifier")
        return parse_expr(
            expression,
            transformations=standard_transformations + (implicit_multiplication_application,),
        )

    @staticmethod
    def _latex_to_sympy(expression: str) -> str:
        value = expression.strip().replace("$", "")
        value = SympyTool._replace_fractions(value)
        value = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", value)
        function_names = {
            "arctan": "atan",
            "arcsin": "asin",
            "arccos": "acos",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "log": "log",
            "ln": "log",
            "exp": "exp",
        }
        value = re.sub(
            r"\\(arctan|arcsin|arccos|sin|cos|tan|log|ln|exp)",
            lambda m: f" {function_names[m.group(1)]}",
            value,
        )
        value = (
            value.replace(r"\left", "")
            .replace(r"\right", "")
            .replace(r"\!", "")
            .replace(r"\pi", "pi")
            .replace(r"\infty", "oo")
            .replace(r"\,", "")
        )
        value = value.replace("^", "**").replace("{", "(").replace("}", ")")
        value = re.sub(r"(?<=[xyzXYZ])(?=[xyzXYZ])", "*", value)
        return re.sub(r"(?<![A-Za-z])e(?=\s*\*\*)", "E", value)

    @staticmethod
    def _replace_fractions(value: str) -> str:
        """Convert nested LaTex fractions without relying on a full TeX parser."""
        marker = r"\frac"
        while marker in value:
            start = value.find(marker)
            numerator = SympyTool._braced_group(value, start + len(marker))
            if numerator is None:
                break
            numerator_text, after_numerator = numerator
            denominator = SympyTool._braced_group(value, after_numerator)
            if denominator is None:
                break
            denominator_text, after_denominator = denominator
            replacement = f"({numerator_text})/({denominator_text})"
            value = value[:start] + replacement + value[after_denominator:]
        return value

    @staticmethod
    def _braced_group(value: str, start: int) -> Optional[tuple[str, int]]:
        while start < len(value) and value[start].isspace():
            start += 1
        if start >= len(value) or value[start] != "{":
            return None
        depth = 0
        for index in range(start, len(value)):
            if value[index] == "{":
                depth += 1
            elif value[index] == "}":
                depth -= 1
                if depth == 0:
                    return value[start + 1:index], index + 1
        return None

    @staticmethod
    def _format(value: Any) -> str:
        text = str(value).replace("**", "^")
        text = re.sub(r"\blog\(", "ln(", text)
        text = re.sub(r"\batan\(", "arctan(", text)
        text = re.sub(r"\basin\(", "arcsin(", text)
        text = re.sub(r"\bacos\(", "arccos(", text)
        text = re.sub(r"\bexp\(x\)", "e^x", text)
        text = re.sub(r"\bexp\(([^()]+)\)", r"e^(\1)", text)
        return re.sub(r"(?<![A-Za-z])oo(?![A-Za-z])", "∞", text)

    def _run(self, operation) -> Optional[str]:
        if not self.sympy:
            return None
        try:
            return self._format(operation(self.sympy))
        except Exception:
            return None
