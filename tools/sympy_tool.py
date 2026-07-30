from __future__ import annotations

import re
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

    def solve_equation(self, expression: str, variable: str = "x") -> Optional[list]:
        if not self.sympy:
            return None
        try:
            return [str(item) for item in self.sympy.solve(self._parse(expression), self.sympy.Symbol(variable))]
        except Exception:
            return None

    def matrix(self, rows: list[list[Any]]) -> Optional[list[list[str]]]:
        if not self.sympy:
            return None
        try:
            return [[str(item) for item in row] for row in self.sympy.Matrix(rows).tolist()]
        except Exception:
            return None

    def limit(self, expression: str, variable: str, point: str) -> Optional[str]:
        return self._run(
            lambda s: s.limit(self._parse(expression), s.Symbol(variable), self._parse(point))
        )

    def hints_for(self, problem: str) -> list[str]:
        """Return safe, deterministic hints for elementary symbolic subproblems.

        This deliberately handles only unambiguous LaTex or plain-text forms.
        Anything it cannot parse is left to the model solver.
        """
        if not self.sympy:
            return []

        hints: list[str] = []
        if re.search(r"导数|求导|微分", problem):
            match = re.search(
                r"(?:f|y)\s*\(\s*([A-Za-z])\s*\)\s*=\s*([^，。；;]+?)(?=\s*(?:的(?:导数|微分)|[,，。；;]|$))",
                problem,
            )
            if match:
                result = self.derivative(self._latex_to_sympy(match.group(2)), match.group(1))
                if result is not None:
                    hints.append(f"SymPy 导数: {result}")

        math_parts = re.findall(r"\$([^$]+)\$", problem)
        if re.search(r"积分|\\int", problem):
            for part in math_parts:
                match = re.search(r"\\int\s*(.+?)(?:\\,|\s)*d([A-Za-z])\b", part)
                if match and "_" not in part[:match.start(1)]:
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
                        hints.append(f"SymPy 方程解: {result}")
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

    def _parse(self, expression: str):
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        if not re.fullmatch(r"[0-9A-Za-z_+\-*/^().,\s]+", expression):
            raise ValueError("unsupported symbolic expression")
        identifiers = set(re.findall(r"[A-Za-z]+", expression))
        allowed = {"sin", "cos", "tan", "log", "exp", "sqrt", "pi", "oo"}
        if any(identifier not in allowed and len(identifier) != 1 for identifier in identifiers):
            raise ValueError("unsupported symbolic identifier")
        return parse_expr(
            expression,
            transformations=standard_transformations + (implicit_multiplication_application,),
        )

    @staticmethod
    def _latex_to_sympy(expression: str) -> str:
        value = expression.strip()
        while True:
            updated = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", value)
            if updated == value:
                break
            value = updated
        value = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", value)
        value = re.sub(r"\\(sin|cos|tan|log|ln|exp)", lambda m: "log" if m.group(1) == "ln" else m.group(1), value)
        value = value.replace(r"\pi", "pi").replace(r"\,", "")
        return value.replace("^", "**").replace("{", "(").replace("}", ")")

    def _run(self, operation) -> Optional[str]:
        if not self.sympy:
            return None
        try:
            return str(operation(self.sympy))
        except Exception:
            return None
