from __future__ import annotations

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
        return self._run(lambda s: s.diff(s.sympify(expression), s.Symbol(variable)))

    def integral(self, expression: str, variable: str = "x") -> Optional[str]:
        return self._run(lambda s: s.integrate(s.sympify(expression), s.Symbol(variable)))

    def solve_equation(self, expression: str, variable: str = "x") -> Optional[list]:
        if not self.sympy:
            return None
        try:
            return [str(item) for item in self.sympy.solve(expression, self.sympy.Symbol(variable))]
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
            lambda s: s.limit(s.sympify(expression), s.Symbol(variable), s.sympify(point))
        )

    def _run(self, operation) -> Optional[str]:
        if not self.sympy:
            return None
        try:
            return str(operation(self.sympy))
        except Exception:
            return None
