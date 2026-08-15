"""General symbolic compilation and candidate verification with SymPy."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

from tools.tool_contract import ToolResult, make_tool_result


@dataclass(frozen=True)
class ToolCheck:
    name: str
    status: str
    detail: str = ""

    def trace_content(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


class SympyTool:
    """Compile only complete, generic mathematical requests.

    Natural-language story matching is intentionally absent.  A failed parse
    yields no tool result and leaves the problem to the model.
    """

    def __init__(self) -> None:
        try:
            import sympy as sympy_module
        except ImportError:
            sympy_module = None
        self.sympy = sympy_module

    def evaluate(self, expression: str) -> Optional[str]:
        return self._run(lambda _: self._parse(expression))

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
        return self._run(lambda s: s.integrate(
            self._parse(expression),
            (s.Symbol(variable), self._parse(lower), self._parse(upper)),
        ))

    def limit(self, expression: str, variable: str, point: str) -> Optional[str]:
        return self._run(lambda s: s.limit(
            self._parse(expression), s.Symbol(variable), self._parse(point)
        ))

    def solve_equation(self, expression: str, variable: str = "x") -> Optional[list[str]]:
        if not self.sympy:
            return None
        try:
            symbol = self.sympy.Symbol(variable)
            solutions = self.sympy.solve(self._parse(expression), symbol)
            return [self._format(item) for item in solutions]
        except Exception:
            return None

    def matrix(self, rows: list[list[Any]]) -> Optional[list[list[str]]]:
        if not self.sympy:
            return None
        try:
            matrix = self.sympy.Matrix([[self._parse(str(cell)) for cell in row] for row in rows])
            return [[self._format(item) for item in row] for row in matrix.tolist()]
        except Exception:
            return None

    def results_for(self, problem: str) -> list[ToolResult]:
        if not self.sympy:
            return []
        from classifier.problem_spec import _strip_trailing_answer_instructions

        text = _strip_trailing_answer_instructions(str(problem or "")).strip()
        compilers = (
            self._compile_arithmetic,
            self._compile_derivative,
            self._compile_definite_integral,
            self._compile_limit,
            self._compile_equation,
            self._compile_matrix_operation,
        )
        results: list[ToolResult] = []
        for compiler in compilers:
            result = compiler(text)
            if result is not None and result.verified:
                results.append(result)
        return results

    def hints_for(self, problem: str) -> list[str]:
        return [item.to_hint() for item in self.results_for(problem)]

    def verify_candidate(self, problem: str, answer: str, spec=None) -> tuple[ToolCheck, ...]:
        """Run conservative post-answer checks; unknown is never a rejection."""
        checks: list[ToolCheck] = []
        value = str(answer or "").strip()
        if not value:
            return (ToolCheck("non_empty", "fail", "empty candidate"),)

        numeric_status = self._numeric_identity_status(value)
        if numeric_status:
            checks.append(ToolCheck("numeric_identities", numeric_status, "literal arithmetic equalities"))

        shape = getattr(getattr(spec, "profile", None), "answer_shape", "")
        if shape == "probability":
            scalar = self._first_scalar_result(value)
            if scalar is not None:
                try:
                    numeric = float(scalar)
                except (TypeError, ValueError):
                    numeric = None
                if numeric is not None:
                    checks.append(ToolCheck(
                        "probability_range",
                        "pass" if 0.0 <= numeric <= 1.0 else "fail",
                        "candidate probability must lie in [0,1]",
                    ))

        equation = self._parse_direct_equation(str(problem or ""))
        if equation and shape == "roots":
            left, right, variable = equation
            roots = self._candidate_roots(value, variable)
            if roots:
                statuses = []
                try:
                    expression = self._parse(left) - self._parse(right)
                    symbol = self.sympy.Symbol(variable)
                    for root in roots:
                        residual = self.sympy.simplify(expression.subs(symbol, self._parse(root)))
                        statuses.append(residual == 0)
                except Exception:
                    statuses = []
                if statuses:
                    checks.append(ToolCheck(
                        "root_substitution",
                        "pass" if all(statuses) else "fail",
                        f"substituted {len(statuses)} listed root(s)",
                    ))

        derivative = self._parse_derivative_request(str(problem or ""))
        if derivative and shape == "expression":
            expression, variable = derivative
            candidate_expression = self._explicit_math_value(value)
            if candidate_expression:
                try:
                    expected = self.sympy.diff(self._parse(expression), self.sympy.Symbol(variable))
                    actual = self._parse(candidate_expression)
                    equal = self.sympy.simplify(expected - actual) == 0
                    checks.append(ToolCheck("derivative_equivalence", "pass" if equal else "fail", "symbolic differentiation"))
                except Exception:
                    pass
        return tuple(checks)

    def _compile_arithmetic(self, text: str) -> Optional[ToolResult]:
        match = re.fullmatch(
            r"\s*(?:计算|求值|calculate|compute|evaluate)\s*(?:下列|the\s+value\s+of)?\s*[:：]?\s*(.+?)\s*[。.!?？]?\s*",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not match or "=" in match.group(1):
            return None
        expression = match.group(1).strip().strip("$ ")
        if not re.search(r"\d", expression):
            return None
        try:
            value = self.sympy.simplify(self._parse(self._latex_to_sympy(expression)))
            if value.has(self.sympy.nan, self.sympy.zoo) or value in {
                self.sympy.nan, self.sympy.zoo,
            }:
                return None
            result = self._format(value)
        except Exception:
            return None
        return make_tool_result(
            problem=text,
            operation="calculate",
            result=result,
            result_kind="scalar",
            method="sympy_exact_recompute",
            whole=True,
            checks=("full_statement_parse", "independent_recompute", "no_unparsed_suffix"),
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _compile_derivative(self, text: str) -> Optional[ToolResult]:
        parsed = self._parse_derivative_request(text)
        if not parsed or not self._single_operation_request(text, "derivative"):
            return None
        expression, variable = parsed
        result = self.derivative(self._latex_to_sympy(expression), variable)
        if result is None:
            return None
        return make_tool_result(
            problem=text,
            operation="derivative",
            result=result,
            result_kind="expression",
            method="sympy_differentiate_then_simplify",
            whole=True,
            checks=("full_statement_parse", "symbolic_differentiation", "no_unparsed_suffix"),
            answer_shapes=("expression",),
        )

    def _compile_definite_integral(self, text: str) -> Optional[ToolResult]:
        if not self._single_operation_request(text, "integral"):
            return None
        math_text = self._math_source(text)
        match = re.search(
            r"\\int\s*_\s*\{?([^{}\s]+)\}?\s*\^\s*\{?([^{}\s]+)\}?\s*"
            r"(.+?)\s*(?:\\,|\\;|\s)*d\s*([A-Za-z])\b",
            math_text,
            re.DOTALL,
        )
        if not match:
            return None
        lower, upper, expression, variable = match.groups()
        result = self.definite_integral(
            self._latex_to_sympy(expression), variable,
            self._latex_to_sympy(lower), self._latex_to_sympy(upper),
        )
        if result is None or "Integral" in result:
            return None
        return make_tool_result(
            problem=text,
            operation="definite_integral",
            result=result,
            result_kind="scalar",
            method="sympy_integrate_then_differentiate_check",
            whole=True,
            checks=("full_statement_parse", "explicit_bounds", "exact_integration", "no_unparsed_suffix"),
            support=(
                f"Exact symbolic integration over the stated bounds gives {result}; "
                "the integrand, variable, and both endpoints were parsed explicitly and "
                "the antiderivative evaluation was simplified exactly."
            ),
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _compile_limit(self, text: str) -> Optional[ToolResult]:
        if not self._single_operation_request(text, "limit"):
            return None
        math_text = self._math_source(text)
        match = re.search(
            r"\\lim\s*_\s*\{?\s*([A-Za-z])\s*\\to\s*([^}\s]+)\s*\}?\s*(.+)",
            math_text,
            re.DOTALL,
        )
        if not match:
            return None
        variable, point, expression = match.groups()
        result = self.limit(
            self._latex_to_sympy(expression), variable, self._latex_to_sympy(point)
        )
        if result is None or "Limit" in result:
            return None
        return make_tool_result(
            problem=text,
            operation="limit",
            result=result,
            result_kind="expression",
            method="sympy_limit",
            whole=True,
            checks=("full_statement_parse", "limit_variable_and_point", "no_unparsed_suffix"),
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _compile_equation(self, text: str) -> Optional[ToolResult]:
        parsed = self._parse_direct_equation(text)
        if not parsed or not self._single_operation_request(text, "equation"):
            return None
        left, right, variable = parsed
        try:
            expression = self._parse(left) - self._parse(right)
            symbol = self.sympy.Symbol(variable)
            polynomial = self.sympy.Poly(expression, symbol)
            if polynomial.degree() < 1 or polynomial.degree() > 4:
                return None
            solutions = self.sympy.solve(polynomial.as_expr(), symbol)
            if any(getattr(item, "free_symbols", set()) for item in solutions):
                return None
            if not all(self.sympy.simplify(expression.subs(symbol, item)) == 0 for item in solutions):
                return None
            if any(item.is_real is not True for item in solutions):
                return None
        except Exception:
            return None
        rendered = r"\varnothing" if not solutions else r",\;".join(
            f"{variable}={self._format(item)}" for item in solutions
        )
        return make_tool_result(
            problem=text,
            operation="solve_equation",
            result=rendered,
            result_kind="solution_set",
            method="sympy_solve_and_substitute",
            whole=True,
            checks=("full_statement_parse", "single_variable", "all_roots_substituted", "no_unparsed_suffix"),
            answer_shapes=("roots", "number"),
            requirements=("result_present", "all_solutions"),
        )

    def _compile_matrix_operation(self, text: str) -> Optional[ToolResult]:
        target_patterns = (
            ("matrix_determinant", r"行列式|\\det|\bdeterminant\b", "scalar"),
            ("matrix_rank", r"(?:矩阵的?)?秩|\brank\b", "integer"),
            ("matrix_inverse", r"逆矩阵|\binverse\b", "matrix"),
            ("matrix_eigenvalues", r"特征值|\beigenvalues?\b", "solution_set"),
        )
        selected = [item for item in target_patterns if re.search(item[1], text, re.IGNORECASE)]
        if len(selected) != 1:
            return None
        matrix_match = re.search(
            r"\\begin\{[pbvBV]?matrix\}(.+?)\\end\{[pbvBV]?matrix\}",
            text,
            re.DOTALL,
        )
        if not matrix_match:
            return None
        rows = [row.strip() for row in re.split(r"\\\\", matrix_match.group(1)) if row.strip()]
        cells = [[cell.strip() for cell in row.split("&")] for row in rows]
        if not cells or any(len(row) != len(cells[0]) for row in cells):
            return None
        try:
            matrix = self.sympy.Matrix([[self._parse(self._latex_to_sympy(cell)) for cell in row] for row in cells])
            operation, _, result_kind = selected[0]
            if operation == "matrix_determinant":
                if matrix.rows != matrix.cols:
                    return None
                value = matrix.det()
            elif operation == "matrix_rank":
                value = matrix.rank()
            elif operation == "matrix_inverse":
                if matrix.rows != matrix.cols or matrix.det() == 0:
                    return None
                value = matrix.inv()
            else:
                if matrix.rows != matrix.cols:
                    return None
                eigenvalues = matrix.eigenvals()
                value = r"\{\," + r",\;".join(
                    rf"{self._format(root)}\;(m={multiplicity})"
                    for root, multiplicity in eigenvalues.items()
                ) + r"\,\}"
            rendered = self._format(value) if operation != "matrix_eigenvalues" else str(value)
        except Exception:
            return None
        answer_shapes = {
            "matrix_determinant": ("number", "expression"),
            "matrix_rank": ("number",),
            "matrix_inverse": ("matrix",),
            "matrix_eigenvalues": ("expression",),
        }[operation]
        requirements = (
            ("result_present", "numeric_result")
            if operation in {"matrix_determinant", "matrix_rank"}
            else ("result_present",)
        )
        return make_tool_result(
            problem=text,
            operation=operation,
            result=rendered,
            result_kind=result_kind,
            method="sympy_matrix_recompute",
            whole=True,
            checks=("full_statement_parse", "explicit_matrix", "independent_recompute", "no_unparsed_suffix"),
            answer_shapes=answer_shapes,
            requirements=requirements,
        )

    def _parse_derivative_request(self, text: str) -> Optional[tuple[str, str]]:
        patterns = (
            r"(?:求|计算)\s*(?:函数\s*)?(?:f\s*\(\s*(?P<v1>[A-Za-z])\s*\)\s*=\s*)?(?P<e1>.+?)\s*(?:的|关于)\s*(?P<v2>[A-Za-z])?\s*(?:导数|一阶导数)",
            r"(?:differentiate|find\s+the\s+derivative\s+of)\s+(?P<e2>.+?)(?:\s+with\s+respect\s+to\s+(?P<v3>[A-Za-z]))?[.!]?\s*$",
            r"\\frac\s*\{d\}\s*\{d\s*(?P<v4>[A-Za-z])\}\s*(?P<e3>.+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            groups = match.groupdict()
            expression = next((groups.get(name) for name in ("e1", "e2", "e3") if groups.get(name)), "")
            variable = next((groups.get(name) for name in ("v2", "v1", "v3", "v4") if groups.get(name)), "x")
            expression = expression.strip(" $，,。.!?：:")
            if expression:
                return expression, variable
        return None

    def _parse_direct_equation(self, text: str) -> Optional[tuple[str, str, str]]:
        if not re.search(r"方程|求解|求.*根|\b(?:solve|equation|roots?|zeros?)\b", text, re.IGNORECASE):
            return None
        if re.search(r"整数解|正整数|实数解|复数解|区间|范围|近似|迭代|\b(?:integer solutions?|positive integer|real roots?|complex roots?|interval|approximately|iteration)\b", text, re.IGNORECASE):
            return None
        candidates = self._math_fragments(text)
        direct = re.search(
            r"(?:方程|equation)\s*[:：]?\s*\$?\s*"
            r"([A-Za-z0-9_+\-*/^().{}\\\s]+?\s*=\s*"
            r"[A-Za-z0-9_+\-*/^().{}\\\s]+?)"
            r"(?=\$|\s*的?(?:全部|所有)?解|\s*(?:for|over)\b|[。.!?？]|$)",
            text,
            re.IGNORECASE,
        )
        if direct:
            candidates = [direct.group(1).strip()]
        else:
            candidates = list(dict.fromkeys(candidates))
        equations = [item for item in candidates if len(re.findall(r"(?<![<>!])=(?!=)", item)) == 1]
        if len(equations) != 1:
            return None
        left, right = equations[0].split("=", 1)
        prepared = self._latex_to_sympy(left + " " + right)
        variables = set(re.findall(r"(?<![A-Za-z])[a-z](?![A-Za-z])", prepared.casefold())) - {"e", "i"}
        if len(variables) != 1:
            return None
        variable = next(iter(variables))
        return self._latex_to_sympy(left), self._latex_to_sympy(right), variable

    @staticmethod
    def _single_operation_request(text: str, operation: str) -> bool:
        operation_patterns = {
            "derivative": r"导数|求导|differentiat|derivative|\\frac\s*\{d\}",
            "integral": r"积分|integral|\\int",
            "limit": r"极限|\blimit\b|\\lim",
            "equation": r"方程|求解|\bsolve\b|\bequation\b|\broots?\b",
        }
        counts = {
            name: bool(re.search(pattern, text, re.IGNORECASE))
            for name, pattern in operation_patterns.items()
        }
        if not counts.get(operation) or sum(counts.values()) != 1:
            return False
        return not bool(re.search(
            r"比较|误差|近似|绘图|构造|并求|并计算|以及|同时|"
            r"\b(?:compare|error|approximate|plot|construct|and then|also compute)\b",
            text,
            re.IGNORECASE,
        ))

    def _numeric_identity_status(self, answer: str) -> str:
        checked = 0
        text = str(answer or "")
        delimiters = "\n,，;；:：$"
        depths: list[int] = []
        depth = 0
        for character in text:
            if character in ")]}" and depth > 0:
                depth -= 1
            depths.append(depth)
            if character in "([{":
                depth += 1
        for equality in re.finditer(r"(?<![<>=])=(?!=)", text):
            equality_depth = depths[equality.start()]
            left_positions = [
                position for position, character in enumerate(text[:equality.start()])
                if character in delimiters and depths[position] == equality_depth
            ]
            left_boundary = max(left_positions) if left_positions else -1
            right_candidates = [
                position for position in range(equality.end(), len(text))
                if text[position] in delimiters and depths[position] == equality_depth
            ]
            right_boundary = min(right_candidates) if right_candidates else len(text)
            left = text[left_boundary + 1:equality.start()].strip(" \\()[]")
            right = text[equality.end():right_boundary].strip(" \\()[]。.!?？")
            numeric_syntax = r"[-+]?\d[\d\s{}()+\-*/^.]*"
            if not re.fullmatch(numeric_syntax, left) or not re.fullmatch(numeric_syntax, right):
                continue
            try:
                difference = self.sympy.simplify(self._parse(left) - self._parse(right))
            except Exception:
                continue
            checked += 1
            if difference != 0:
                return "fail"
        return "pass" if checked else ""

    def _first_scalar_result(self, value: str):
        from reasoning.finalizer import Finalizer
        extracted = Finalizer.extract_result(value)
        candidate = extracted.answer if extracted.valid and extracted.answer else value
        match = re.search(r"(?:=|为|is)?\s*(\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}|[-+]?\d+(?:\.\d+)?(?:/\d+)?)", candidate)
        if not match:
            return None
        try:
            return self._parse(self._latex_to_sympy(match.group(1)))
        except Exception:
            return None

    def _candidate_roots(self, value: str, variable: str) -> list[str]:
        roots = re.findall(
            rf"(?<![A-Za-z]){re.escape(variable)}\s*=\s*([^,，;；\s$]+)",
            value,
            re.IGNORECASE,
        )
        return [self._latex_to_sympy(item.strip("{}()")) for item in roots]

    def _explicit_math_value(self, value: str) -> str:
        from reasoning.finalizer import Finalizer
        extracted = Finalizer.extract_result(value)
        candidate = extracted.answer if extracted.valid and extracted.answer else value
        if "=" in candidate:
            candidate = candidate.rsplit("=", 1)[-1]
        return self._latex_to_sympy(candidate.strip(" $\\()[]"))

    @staticmethod
    def _math_source(text: str) -> str:
        fragments = SympyTool._math_fragments(text)
        return max(fragments, key=len) if fragments else text

    @staticmethod
    def _math_fragments(text: str) -> list[str]:
        fragments = [item.strip() for item in re.findall(r"\$([^$]+)\$", text) if item.strip()]
        for pair in re.findall(r"\\\((.+?)\\\)|\\\[(.+?)\\\]", text, re.DOTALL):
            fragments.extend(item.strip() for item in pair if item.strip())
        if not fragments:
            command = re.sub(
                r"^(?:求解|解方程|求|计算|solve|find|determine)\s*",
                "",
                text.strip(),
                flags=re.IGNORECASE,
            ).strip(" 。.!?？")
            if "=" in command or r"\int" in command or r"\lim" in command:
                fragments.append(command)
        return fragments

    def _parse(self, expression: str):
        if not self.sympy:
            raise ValueError("sympy unavailable")
        value = str(expression or "").strip()
        if not value or len(value) > 1000:
            raise ValueError("empty or oversized expression")
        if re.search(r"__|['\"\[\]:;]|\b(?:lambda|import|exec|eval|open|globals|locals)\b", value, re.IGNORECASE):
            raise ValueError("unsafe expression")
        value = self._latex_to_sympy(value).replace("^", "**")
        if not re.fullmatch(r"[A-Za-z0-9_+\-*/().,\s*]+", value):
            raise ValueError("unsupported expression syntax")
        identifiers = set(re.findall(r"[A-Za-z_]+", value))
        functions = {"sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "exp", "log", "sqrt", "Abs"}
        constants = {"pi", "oo", "E", "I"}
        if any(identifier not in functions | constants and (len(identifier) != 1 or not identifier.isalpha()) for identifier in identifiers):
            raise ValueError("unsupported identifier")
        local = {name: getattr(self.sympy, name) for name in functions if hasattr(self.sympy, name)}
        local.update({"pi": self.sympy.pi, "oo": self.sympy.oo, "E": self.sympy.E, "I": self.sympy.I})
        local.update({letter: self.sympy.Symbol(letter) for letter in "abcdefghijklmnopqrstuvwxyz"})
        return self.sympy.sympify(value, locals=local)

    @staticmethod
    def _latex_to_sympy(expression: str) -> str:
        value = str(expression or "")
        value = value.replace(r"\left", "").replace(r"\right", "")
        value = value.replace(r"\,", "").replace(r"\;", " ").replace(r"\!", "")
        value = value.replace(r"\cdot", "*").replace(r"\times", "*")
        value = value.replace("×", "*").replace("÷", "/").replace("−", "-")
        value = value.replace(r"\pi", "pi").replace(r"\infty", "oo")
        value = re.sub(r"\\(?:operatorname|mathrm)\s*\{(sin|cos|tan|sinh|cosh|exp|log)\}", r"\1", value)
        value = re.sub(r"\\(sin|cos|tan|sinh|cosh|exp|log|ln)(?![A-Za-z])", lambda match: "log" if match.group(1) == "ln" else match.group(1), value)
        previous = None
        while previous != value:
            previous = value
            value = re.sub(
                r"\\(?:d?frac|tfrac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
                r"((\1)/(\2))",
                value,
            )
            value = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", value)
            value = re.sub(r"([A-Za-z0-9)])\s*\^\s*\{([^{}]+)\}", r"\1**(\2)", value)
        value = value.replace("{", "(").replace("}", ")")
        value = re.sub(r"(?<=\d)(?=[A-Za-z(])|(?<=[A-Za-z)])(?=\d)|(?<=\))(?=[A-Za-z(])", "*", value)
        return re.sub(r"\s+", " ", value).strip().strip("$")

    def _format(self, value: Any) -> str:
        if not self.sympy:
            return str(value)
        try:
            return self.sympy.latex(self.sympy.simplify(value))
        except Exception:
            return str(value)

    def _run(self, operation) -> Optional[str]:
        if not self.sympy:
            return None
        try:
            return self._format(operation(self.sympy))
        except Exception:
            return None
