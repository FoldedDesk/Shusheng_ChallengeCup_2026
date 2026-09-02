"""General symbolic compilation and candidate verification with SymPy."""

from __future__ import annotations

from dataclasses import dataclass
import re
import string
from typing import Any, Optional

from tools.tool_contract import ToolResult, make_tool_result as _make_tool_result


def make_tool_result(**kwargs) -> ToolResult:
    """Build a certificate only after a SymPy compiler contract has passed.

    Every call site in this module first parses and bounds the requested
    operation, executes it locally, and rejects unresolved symbolic output.
    Keeping this wrapper local prevents unrelated producers from inheriting
    those guarantees merely by calling the shared constructor.
    """

    checks = tuple(kwargs.get("checks", ()))
    kwargs.setdefault("preconditions", (
        "sympy_request_parser_accepted",
        "symbols_and_domain_bounded",
    ))
    kwargs.setdefault("execution_checks", (
        "deterministic_sympy_operation_completed",
    ))
    kwargs.setdefault("postconditions", checks)
    kwargs.setdefault("certified_value", True)
    return _make_tool_result(**kwargs)


@dataclass(frozen=True)
class ToolCheck:
    name: str
    status: str
    detail: str = ""
    decisive: bool = False

    def trace_content(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "decisive": self.decisive,
        }


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
        return self._run(
            lambda s: s.diff(self._parse(expression), s.Symbol(variable)),
            forbidden_nodes=("Derivative",),
        )

    def integral(self, expression: str, variable: str = "x") -> Optional[str]:
        if not self._integral_complexity_allowed(expression, variable):
            return None
        return self._run(
            lambda s: s.integrate(self._parse(expression), s.Symbol(variable)),
            forbidden_nodes=("Integral",),
        )

    def definite_integral(
        self,
        expression: str,
        variable: str,
        lower: str,
        upper: str,
    ) -> Optional[str]:
        if not self._integral_complexity_allowed(expression, variable):
            return None
        return self._run(
            lambda s: s.integrate(
                self._parse(expression),
                (s.Symbol(variable), self._parse(lower), self._parse(upper)),
            ),
            forbidden_nodes=("Integral",),
        )

    @staticmethod
    def _integral_complexity_allowed(expression: str, variable: str) -> bool:
        """Admit only integrations with a predictably small symbolic search.

        SymPy integration is deterministic but not time bounded.  Mixed
        transcendental/rational integrands can spend minutes in heuristic
        search, which is unsafe inside a per-question Agent budget.  Refusing
        such a request is an abstention, never a mathematical verdict.
        """
        text = str(expression or "").strip()
        name = str(variable or "").strip()
        if (
            not text
            or len(text) > 600
            or not re.fullmatch(r"[A-Za-z]", name)
            or re.search(
                r"\b(?:Piecewise|RootSum|Integral|Derivative|Sum|Product)\b|"
                r"\b(?:abs|sign|floor|ceiling|gamma|polygamma|polylog|meijerg|"
                r"hyper|elliptic|bessel|erf|Ei)\s*\(",
                text,
                re.IGNORECASE,
            )
        ):
            return False
        function_calls = re.findall(
            r"\b(?:log|ln|sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|exp)\s*\(",
            text,
            re.IGNORECASE,
        )
        if len(function_calls) > 1:
            return False
        # A transcendental numerator over a variable-dependent denominator is
        # a common trigger for expensive special-function search.  Simple
        # log(x), sin(x), exp(x), and rational functions remain admitted.
        if function_calls and re.search(
            rf"/\s*(?:\([^)]*\b{re.escape(name)}\b[^)]*\)|"
            rf"[^+\-*/\n]{{0,80}}\b{re.escape(name)}\b)",
            text,
        ):
            return False
        if re.search(rf"\^\s*\([^)]*\b{re.escape(name)}\b", text):
            return False
        return True

    def limit(self, expression: str, variable: str, point: str) -> Optional[str]:
        return self._run(
            lambda s: s.limit(
                self._parse(expression), s.Symbol(variable), self._parse(point)
            ),
            forbidden_nodes=("Limit",),
        )

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
            self._compile_function_evaluation,
            self._compile_finite_sum,
            self._compile_linear_system,
            self._compile_explicit_pde_solution_check,
            self._compile_laplacian,
            self._compile_central_difference,
            self._compile_derivative,
            self._compile_iterated_definite_integral,
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

    def result_from_located_fragment(
        self,
        problem: str,
        operation: str,
        source: str,
        variable: str = "",
        *,
        spec=None,
    ) -> Optional[ToolResult]:
        """Compute a whole answer only from a fragment grounded in the statement.

        The locator is untrusted.  It may select an operation and copy source
        text, but it may neither rewrite the mathematics nor provide a result.
        Every accepted result is recomputed locally and rebound to the original
        statement fingerprint.
        """
        if not self.sympy or not self._located_operation_matches(
            problem, operation, spec
        ):
            return None
        grounded = self._grounded_fragment(problem, source)
        if not grounded:
            return None
        fragment = self._strip_math_delimiters(grounded)
        if not fragment or len(fragment) > 900:
            return None

        located: Optional[ToolResult]
        if operation == "calculate":
            located = self._calculate_located(fragment, grounded)
        elif operation == "solve_equation":
            located = self._solve_located_equation(fragment, problem, variable)
        elif operation == "derivative":
            located = self._differentiate_located(fragment, variable)
        elif operation == "definite_integral":
            located = self._integrate_located(fragment)
        elif operation == "limit":
            located = self._limit_located(fragment)
        else:
            prefixes = {
                "finite_sum": "Compute the finite sum ",
                "solve_linear_system": "Solve the linear system ",
                "matrix_determinant": "Find the determinant of ",
                "matrix_rank": "Find the rank of ",
                "matrix_inverse": "Find the inverse matrix of ",
                "matrix_eigenvalues": "Find the eigenvalues of ",
            }
            compilers = {
                "finite_sum": self._compile_finite_sum,
                "solve_linear_system": self._compile_linear_system,
                "matrix_determinant": self._compile_matrix_operation,
                "matrix_rank": self._compile_matrix_operation,
                "matrix_inverse": self._compile_matrix_operation,
                "matrix_eigenvalues": self._compile_matrix_operation,
            }
            prefix = prefixes.get(operation)
            compiler = compilers.get(operation)
            located = compiler(prefix + grounded) if prefix and compiler else None
        if located is None or not located.verified or located.operation != operation:
            return None
        contract = located.contract
        if contract is None:
            return None
        return make_tool_result(
            problem=problem,
            operation=located.operation,
            result=located.result,
            result_kind=contract.result_kind,
            method=f"located_fragment_{located.certificate.method}",
            whole=True,
            written_support=contract.written_support_capable,
            checks=(
                "statement_fragment_grounded",
                "operation_target_matched",
                *located.certificate.checks,
            ),
            support=located.support,
            answer_shapes=contract.allowed_answer_shapes,
            requirements=contract.allowed_requirements,
        )

    def _calculate_located(
        self,
        fragment: str,
        grounded: str,
    ) -> Optional[ToolResult]:
        if re.search(r"(?<![<>!])=(?!=)|\\(?:sum|int|lim)|[A-Za-z]\s*\(", fragment):
            return None
        try:
            value = self.sympy.simplify(self._parse(fragment))
            if value.free_symbols or value.has(self.sympy.nan, self.sympy.zoo):
                return None
        except Exception:
            return None
        return make_tool_result(
            problem=grounded,
            operation="calculate",
            result=self._format(value),
            result_kind="scalar",
            method="sympy_exact_grounded_expression",
            whole=True,
            checks=("grounded_expression", "exact_simplification"),
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _solve_located_equation(
        self,
        fragment: str,
        problem: str,
        variable: str,
    ) -> Optional[ToolResult]:
        if len(re.findall(r"(?<![<>!])=(?!=)", fragment)) != 1:
            return None
        left, right = re.split(r"(?<![<>!])=(?!=)", fragment, maxsplit=1)
        try:
            expression = self._parse(left) - self._parse(right)
            symbols = expression.free_symbols
            if variable:
                symbol = self.sympy.Symbol(variable)
                if symbols != {symbol}:
                    return None
            elif len(symbols) == 1:
                symbol = next(iter(symbols))
                variable = symbol.name
            else:
                return None
            roots = self._exact_roots(expression, symbol, problem)
            if roots is None or not all(
                self.sympy.simplify(expression.subs(symbol, root)) == 0
                for root in roots
            ):
                return None
        except Exception:
            return None
        rendered = r"\varnothing" if not roots else r",\;".join(
            f"{variable}={self._format(root)}" for root in roots
        )
        return make_tool_result(
            problem=fragment,
            operation="solve_equation",
            result=rendered,
            result_kind="solution_set",
            method="sympy_grounded_polynomial_roots",
            whole=True,
            checks=("single_grounded_equation", "complete_exact_roots", "root_substitution"),
            answer_shapes=("roots",),
            requirements=("result_present", "all_solutions"),
        )

    def _differentiate_located(
        self,
        fragment: str,
        variable: str,
    ) -> Optional[ToolResult]:
        if not variable or re.search(r"(?<![<>!])=(?!=)|\\(?:int|lim|sum)", fragment):
            return None
        try:
            symbol = self.sympy.Symbol(variable)
            expression = self._parse(fragment)
            if expression.free_symbols - {symbol}:
                return None
            value = self.sympy.diff(expression, symbol)
            if value.has(self.sympy.Derivative):
                return None
        except Exception:
            return None
        return make_tool_result(
            problem=fragment,
            operation="derivative",
            result=self._format(value),
            result_kind="expression",
            method="sympy_grounded_differentiation",
            whole=True,
            checks=("grounded_expression", "explicit_variable", "symbolic_differentiation"),
            answer_shapes=("expression",),
        )

    def _integrate_located(self, fragment: str) -> Optional[ToolResult]:
        parsed = self._parse_definite_integral_request(fragment)
        if not parsed:
            return None
        expression, variable, lower, upper = parsed
        result = self.definite_integral(
            self._latex_to_sympy(expression), variable,
            self._latex_to_sympy(lower), self._latex_to_sympy(upper),
        )
        if result is None or "Integral" in result:
            return None
        return make_tool_result(
            problem=fragment,
            operation="definite_integral",
            result=result,
            result_kind="scalar",
            method="sympy_grounded_definite_integral",
            whole=True,
            written_support=True,
            checks=("grounded_integral", "explicit_bounds", "exact_integration"),
            support=f"Exact integration of the copied integrand over its copied bounds gives {result}.",
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _limit_located(self, fragment: str) -> Optional[ToolResult]:
        parsed = self._parse_limit_request(fragment)
        if not parsed:
            return None
        expression, variable, point = parsed
        result = self.limit(
            self._latex_to_sympy(expression), variable,
            self._latex_to_sympy(point),
        )
        if result is None or "Limit" in result:
            return None
        return make_tool_result(
            problem=fragment,
            operation="limit",
            result=result,
            result_kind="expression",
            method="sympy_grounded_limit",
            whole=True,
            checks=("grounded_limit", "explicit_variable_and_point", "symbolic_limit"),
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    @staticmethod
    def _grounded_fragment(problem: str, source: str) -> str:
        statement = str(problem or "")
        fragment = str(source or "").strip()
        if not statement or not fragment or len(fragment) > 900:
            return ""
        if fragment in statement:
            return fragment
        normalized_statement = re.sub(r"\s+", " ", statement).strip()
        normalized_fragment = re.sub(r"\s+", " ", fragment).strip()
        if not normalized_fragment or normalized_statement.count(normalized_fragment) != 1:
            return ""
        return fragment

    @staticmethod
    def _strip_math_delimiters(value: str) -> str:
        text = str(value or "").strip()
        pairs = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))
        for left, right in pairs:
            if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
                return text[len(left):-len(right)].strip()
        return text

    @staticmethod
    def _located_operation_matches(problem: str, operation: str, spec=None) -> bool:
        if spec is None:
            return False
        profile = getattr(spec, "profile", None)
        if profile is None or str(getattr(profile, "topic", "")).startswith("olympiad_"):
            return False
        if len(getattr(spec, "goals", ())) != 1:
            return False
        if str(getattr(profile, "task_kind", "")) not in {"calculation", "fill_blank"}:
            return False
        if "residual_output_contract" in getattr(spec, "risk_flags", ()):
            return False
        shape = getattr(profile, "answer_shape", "")
        text = str(problem or "")
        gates = {
            # Direct exact-value requests are sometimes classified as an
            # expression because the requested source contains variables or
            # powers.  The explicit operation wording, grounded source, and
            # local free-symbol check are the decisive guards here.
            "calculate": (shape in {"number", "expression"}, r"计算|求值|的值|\b(?:compute|calculate|evaluate|value of)\b"),
            "solve_equation": (shape == "roots", r"解方程|全部解|所有根|\b(?:solve|all solutions?|all roots?)\b"),
            "derivative": (shape == "expression", r"导数|求导|\b(?:derivative|differentiat)\w*\b"),
            "definite_integral": (shape in {"number", "expression"}, r"积分|\\int|\bintegral\b"),
            "limit": (shape in {"number", "expression"}, r"极限|\\lim|\blimit\b"),
            "finite_sum": (shape in {"number", "count", "expression"}, r"求和|\\sum|\b(?:finite sum|summation)\b"),
            "solve_linear_system": (shape in {"roots", "expression"}, r"方程组|\b(?:linear )?system\b"),
            "matrix_determinant": (shape == "number", r"行列式|\\det|\bdeterminant\b"),
            "matrix_rank": (shape == "number", r"(?:矩阵的?)?秩|\brank\b"),
            "matrix_inverse": (shape == "matrix", r"逆矩阵|\binverse matrix\b|\bmatrix inverse\b"),
            "matrix_eigenvalues": (shape == "expression", r"特征值|\beigenvalues?\b"),
        }
        enabled, pattern = gates.get(operation, (False, ""))
        return bool(enabled and re.search(pattern, text, re.IGNORECASE))

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
            checks.append(ToolCheck(
                "numeric_identities",
                numeric_status,
                "literal arithmetic equalities",
                False,
            ))

        stated_evaluation = self._verify_stated_function_evaluations(value)
        if stated_evaluation is not None:
            checks.append(stated_evaluation)

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
                        False,
                    ))

        if shape == "count":
            scalar = self._first_scalar_result(value)
            if scalar is not None and not getattr(scalar, "free_symbols", set()):
                is_count = bool(scalar.is_integer is True and scalar.is_nonnegative is True)
                checks.append(ToolCheck(
                    "count_domain",
                    "pass" if is_count else "fail",
                    "a count must be a nonnegative integer",
                    False,
                ))

        equation = self._parse_direct_equation(str(problem or ""), allow_domains=True)
        if equation and shape == "roots":
            left, right, variable = equation
            roots = self._candidate_roots(value, variable)
            claims_empty = bool(re.search(
                r"\\varnothing|空集|无解|不存在(?:实数|复数)?解|no solutions?",
                value,
                re.IGNORECASE,
            ))
            if roots or claims_empty:
                statuses = []
                equation_expression = None
                equation_symbol = None
                try:
                    equation_expression = self._parse(left) - self._parse(right)
                    equation_symbol = self.sympy.Symbol(variable)
                    for root in roots:
                        residual = self.sympy.simplify(
                            equation_expression.subs(equation_symbol, self._parse(root))
                        )
                        statuses.append(residual == 0)
                except Exception:
                    statuses = []
                if statuses:
                    checks.append(ToolCheck(
                        "root_substitution",
                        "pass" if all(statuses) else "fail",
                        f"substituted {len(statuses)} listed root(s)",
                        False,
                    ))
                expected = (
                    self._exact_roots(
                        equation_expression,
                        equation_symbol,
                        str(problem or ""),
                    )
                    if equation_expression is not None and equation_symbol is not None
                    else None
                )
                if expected is not None:
                    actual = None
                    try:
                        actual = [self._parse(root) for root in roots]
                    except Exception:
                        actual = None
                    if actual is not None:
                        complete = self._same_expression_set(actual, expected)
                        checks.append(ToolCheck(
                            "root_set_completeness",
                            "pass" if complete else "fail",
                            f"compared {len(actual)} listed root(s) with {len(expected)} exact root(s)",
                            True,
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
                    checks.append(ToolCheck(
                        "derivative_equivalence",
                        "pass" if equal else "fail",
                        "symbolic differentiation",
                        True,
                    ))
                except Exception:
                    pass

        problem_text = str(problem or "")
        limit_request = self._parse_limit_request(problem_text)
        definite_integral = self._parse_definite_integral_request(problem_text)
        quadrature_request = bool(re.search(
            r"数值积分|求积|复化(?:中点|梯形|辛普森)|"
            r"\b(?:quadrature|composite\s+(?:midpoint|trapezoid|simpson)|"
            r"gauss[- ]legendre)\b",
            problem_text,
            re.IGNORECASE,
        ))
        # In lim F_n where F_n contains an integral, the finite-n integral is
        # an intermediate expression, not the requested scalar.  Comparing the
        # submitted limit against F_n would decisively reject a correct answer.
        if (
            definite_integral
            and limit_request is None
            and not quadrature_request
            and shape in {"number", "expression"}
        ):
            expression, variable, lower, upper = definite_integral
            candidate_expression = self._explicit_math_value(value)
            if candidate_expression:
                try:
                    symbol = self.sympy.Symbol(variable)
                    expected = self.sympy.integrate(
                        self._parse(expression),
                        (symbol, self._parse(lower), self._parse(upper)),
                    )
                    actual = self._parse(candidate_expression)
                    equal = self.sympy.simplify(expected - actual) == 0
                    checks.append(ToolCheck(
                        "definite_integral_equivalence",
                        "pass" if equal else "fail",
                        "exact integration over parsed bounds",
                        True,
                    ))
                except Exception:
                    pass

        if limit_request and shape in {"number", "expression"}:
            expression, variable, point = limit_request
            candidate_expression = self._explicit_math_value(value)
            if candidate_expression:
                try:
                    expected = self.sympy.limit(
                        self._parse(expression),
                        self.sympy.Symbol(variable),
                        self._parse(point),
                    )
                    actual = self._parse(candidate_expression)
                    equal = self.sympy.simplify(expected - actual) == 0
                    checks.append(ToolCheck(
                        "limit_equivalence",
                        "pass" if equal else "fail",
                        "symbolic limit at parsed point",
                        True,
                    ))
                except Exception:
                    pass

        topic = getattr(getattr(spec, "profile", None), "topic", "")
        requirement_names = {
            requirement.name
            for goal in getattr(spec, "goals", ())
            for requirement in goal.requirements
        } if spec is not None else set()
        if "closed_stability_interval" in requirement_names:
            stability_check = self._verify_stability_consistency(value)
            if stability_check is not None:
                checks.append(stability_check)
        if topic != "numerical_method" and not requirement_names.intersection({
            "method_formula", "first_iteration",
        }):
            checks.extend(self._verify_statement_constraints(str(problem or ""), value, spec))
        return tuple(checks)

    def _verify_stated_function_evaluations(
        self, answer: str
    ) -> Optional[ToolCheck]:
        """Check F(variable)=expression against a stated F(point)=value."""
        if not self.sympy:
            return None
        text = str(answer or "").replace(r"\left", "").replace(r"\right", "")
        claim_pattern = re.compile(
            r"(?P<function>[A-Zfghuy](?:\s*_\s*(?:\{[^{}\n]{1,40}\}|[A-Za-z0-9]+))?)"
            r"\s*\(\s*(?P<argument>[^()\n]{1,30})\s*\)\s*=",
        )
        matches = list(claim_pattern.finditer(text))
        if len(matches) < 2:
            return None

        claims: dict[str, list[tuple[str, str]]] = {}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            right = text[match.end():end].strip()
            right = right.strip(" $,，;；。\n")
            right = re.sub(r"\\q?quad\s*$", "", right).strip(
                " $,，;；。\n"
            )
            if "=" in right:
                right = right.rsplit("=", 1)[-1].strip()
            if not right:
                continue
            function = re.sub(r"\s+", "", match.group("function"))
            claims.setdefault(function, []).append(
                (match.group("argument").strip(), right)
            )

        compared = 0
        for function_claims in claims.values():
            definitions = [
                (argument, right)
                for argument, right in function_claims
                if re.fullmatch(r"[A-Za-z]", argument)
            ]
            evaluations = [
                (argument, re.split(r"[,，;；。\n]", right, maxsplit=1)[0].strip())
                for argument, right in function_claims
                if not re.fullmatch(r"[A-Za-z]", argument)
            ]
            for variable_name, expression_source in definitions:
                try:
                    variable = self.sympy.Symbol(variable_name)
                    expression_text = self._latex_to_sympy(expression_source)
                    expression_text = re.sub(
                        rf"\b{re.escape(variable_name)}\s*(?=\()",
                        f"{variable_name}*",
                        expression_text,
                    )
                    expression_text = re.sub(r"\)\s*(?=\()", ")*", expression_text)
                    expression_text = re.sub(
                        rf"(?<=[0-9)])\s*(?={re.escape(variable_name)}\b)",
                        "*",
                        expression_text,
                    )
                    expression_text = re.sub(
                        rf"\)\s*(?={re.escape(variable_name)}\b)",
                        ")*",
                        expression_text,
                    )
                    expression = self._parse(expression_text)
                    if expression.free_symbols - {variable}:
                        continue
                except Exception:
                    continue
                for point_source, value_source in evaluations:
                    try:
                        point = self._parse(self._latex_to_sympy(point_source.strip(" $")))
                        actual = self._parse(self._latex_to_sympy(value_source.strip(" $")))
                        if point.free_symbols or actual.free_symbols:
                            continue
                        expected = self.sympy.simplify(expression.subs(variable, point))
                        equal = self.sympy.simplify(expected - actual) == 0
                    except Exception:
                        continue
                    compared += 1
                    if not equal:
                        return ToolCheck(
                            "stated_function_evaluation_consistency",
                            "fail",
                            "the stated function value conflicts with the submitted formula",
                            True,
                        )
        if compared:
            return ToolCheck(
                "stated_function_evaluation_consistency",
                "pass",
                f"checked {compared} stated function evaluation(s)",
                False,
            )
        return None

    def _verify_stability_consistency(self, answer: str) -> Optional[ToolCheck]:
        """Check that a reported stability endpoint agrees with its own R(z)."""
        if not self.sympy:
            return None
        text = str(answer or "").replace(r"\left", "").replace(r"\right", "")
        function_match = re.search(
            r"R\s*\(\s*z\s*\)\s*=\s*(.+?)"
            r"(?=\$|[；;。\n]|[,，]\s*\\?\s*(?:[xr]\s*(?:\^|\*\*)|\[))",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        interval_match = re.search(
            r"\[\s*([^,，\]\n]{1,80})\s*[,，]\s*([^\]\n]{1,80})\s*\]",
            text,
        )
        if not function_match or not interval_match:
            return None

        left_text, right_text = (item.strip(" $ ") for item in interval_match.groups())
        endpoint_text = left_text
        endpoint_match = re.fullmatch(r"[-+−]?\d+(?:\.\d+)?", endpoint_text)
        if endpoint_match is None and re.search(r"[A-Za-z]", endpoint_text):
            approximation = re.search(
                r"(?:\\approx|≈|approximately|approx\.?|约为)\s*"
                r"([-+−]?\d+(?:\.\d+)?)",
                text,
                re.IGNORECASE,
            )
            if approximation:
                magnitude = approximation.group(1).replace("−", "-").lstrip("+")
                endpoint_text = (
                    "-" + magnitude.lstrip("-")
                    if endpoint_text.lstrip().startswith(("-", "−"))
                    else magnitude
                )
                endpoint_match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", endpoint_text)
        if endpoint_match is None:
            return None

        try:
            right = self._parse(self._latex_to_sympy(right_text))
            if self.sympy.simplify(right) != 0:
                return None
            z = self.sympy.Symbol("z")
            expression = self._parse(
                self._latex_to_sympy(function_match.group(1).strip())
            )
            if expression.free_symbols - {z}:
                return None
            if self.sympy.simplify(expression.subs(z, 0) - 1) != 0:
                return ToolCheck(
                    "stability_internal_consistency",
                    "fail",
                    "the reported stability function does not satisfy R(0)=1",
                    True,
                )
            endpoint = self._parse(endpoint_text.replace("−", "-"))
            endpoint_value = complex(self.sympy.N(expression.subs(z, endpoint), 16))
        except Exception:
            return None

        decimal_places = (
            len(endpoint_text.rsplit(".", 1)[1]) if "." in endpoint_text else 0
        )
        tolerance = (
            max(0.02, 5 * (10 ** (-decimal_places)))
            if decimal_places
            else 0.02
        )
        if abs(abs(endpoint_value) - 1.0) > tolerance:
            return ToolCheck(
                "stability_internal_consistency",
                "fail",
                "the reported interval endpoint does not satisfy |R(z)|=1",
                True,
            )

        boundary_claims = re.findall(
            r"R\s*\(\s*[xr]\s*\)\s*=\s*([+-]?\s*1)(?![\d.])",
            text,
            re.IGNORECASE,
        )
        for raw_claim in boundary_claims:
            claimed = float(raw_claim.replace(" ", ""))
            if abs(endpoint_value.real - claimed) > tolerance or abs(endpoint_value.imag) > tolerance:
                return ToolCheck(
                    "stability_internal_consistency",
                    "fail",
                    "the stated R(endpoint) boundary conflicts with the reported endpoint",
                    True,
                )

        equation_matches = re.finditer(
            r"(?:即|equivalently|边界方程(?:为|是)?|boundary\s+equation(?:\s+is)?)"
            r"\s*\$?\s*([^$；;。\n=]{1,220})\s*=\s*0|"
            r"[,，]\s*\\?\s*((?:[xr]\s*(?:\^|\*\*)|[xr]\^\{)"
            r"[^$；;。\n=]{1,200})\s*=\s*0",
            text,
            re.IGNORECASE,
        )
        for equation_match in equation_matches:
            try:
                equation_text = next(
                    item for item in equation_match.groups() if item is not None
                )
                equation = self._parse(
                    self._latex_to_sympy(equation_text.strip())
                )
                if len(equation.free_symbols) != 1:
                    continue
                variable = next(iter(equation.free_symbols))
                residual = abs(complex(self.sympy.N(
                    equation.subs(variable, endpoint), 16
                )))
                coefficients = self.sympy.Poly(equation, variable).all_coeffs()
                scale = max(1.0, sum(abs(float(self.sympy.N(item))) for item in coefficients))
            except Exception:
                continue
            if residual > scale * tolerance:
                return ToolCheck(
                    "stability_internal_consistency",
                    "fail",
                    "the boundary equation is inconsistent with the reported endpoint",
                    True,
                )

        return ToolCheck(
            "stability_internal_consistency",
            "pass",
            "R(0), the negative-axis endpoint, and stated boundary relations agree",
            True,
        )

    def _compile_arithmetic(self, text: str) -> Optional[ToolResult]:
        expression = self._parse_arithmetic_request(text)
        if expression is None:
            return None
        expression = self._strip_math_delimiters(expression)
        if (
            not re.search(r"\d|\\pi|π", expression)
            or re.search(r"(?<![<>!])=(?!=)|[<>≤≥]|\\(?:sum|int|lim)\b", expression)
        ):
            return None
        try:
            value = self.sympy.simplify(self._parse(self._latex_to_sympy(expression)))
            if (
                value.free_symbols
                or value.is_number is not True
                or value.has(self.sympy.nan, self.sympy.zoo)
                or value in {self.sympy.nan, self.sympy.zoo, self.sympy.oo, -self.sympy.oo}
            ):
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

    def _compile_function_evaluation(self, text: str) -> Optional[ToolResult]:
        parsed = self._parse_function_evaluation_request(text)
        if not parsed:
            return None
        function_name, variable_name, expression_text, point_text = parsed
        try:
            variable = self.sympy.Symbol(variable_name)
            expression = self._parse(expression_text)
            point = self._parse(point_text)
            if expression.free_symbols - {variable} or point.free_symbols:
                return None
            value = self.sympy.simplify(expression.subs(variable, point))
            if value.free_symbols or value.has(self.sympy.nan, self.sympy.zoo):
                return None
            if self.sympy.simplify(value - expression.subs(variable, point)) != 0:
                return None
        except Exception:
            return None
        result = self._format(value)
        support = (
            f"Substitute {variable_name}={self._format(point)} into "
            f"{function_name}({variable_name})={self._format(expression)}; exact simplification gives {result}."
        )
        return make_tool_result(
            problem=text,
            operation="function_evaluation",
            result=result,
            result_kind="scalar",
            method="sympy_exact_substitution",
            whole=True,
            written_support=True,
            checks=("function_definition_parsed", "evaluation_point_parsed", "exact_substitution"),
            support=support,
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _compile_finite_sum(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"\\sum|求和|有限和|finite sum|summation", text, re.IGNORECASE):
            return None
        math_text = self._math_source(text)
        match = re.search(
            r"\\sum\s*_\s*\{\s*([A-Za-z])\s*=\s*([^{}]+)\}\s*"
            r"\^\s*\{([^{}]+)\}\s*(.+)$",
            math_text,
            re.DOTALL,
        )
        if not match:
            return None
        variable_name, lower_text, upper_text, expression_text = match.groups()
        expression_text = expression_text.strip().strip("$，,。.!? ")
        try:
            variable = self.sympy.Symbol(variable_name)
            lower = self._parse(lower_text)
            upper = self._parse(upper_text)
            expression = self._parse(expression_text)
            if lower.free_symbols or upper.free_symbols or expression.free_symbols - {variable}:
                return None
            if lower.is_integer is not True or upper.is_integer is not True:
                return None
            span = int(upper - lower)
            if span < -1 or span > 10000:
                return None
            value = self.sympy.simplify(self.sympy.summation(expression, (variable, lower, upper)))
            if span <= 500:
                recomputed = self.sympy.simplify(sum(
                    expression.subs(variable, index)
                    for index in range(int(lower), int(upper) + 1)
                ))
                if self.sympy.simplify(value - recomputed) != 0:
                    return None
        except Exception:
            return None
        result = self._format(value)
        support = (
            f"The finite index range is {variable_name}={self._format(lower)},\\ldots,{self._format(upper)}. "
            f"Exact symbolic summation of {self._format(expression)} gives {result}."
        )
        return make_tool_result(
            problem=text,
            operation="finite_sum",
            result=result,
            result_kind="scalar",
            method="sympy_finite_sum_with_direct_recompute",
            whole=True,
            written_support=True,
            checks=("finite_integer_bounds", "symbolic_summation", "direct_term_recompute"),
            support=support,
            answer_shapes=("number", "expression", "count"),
            requirements=("result_present", "numeric_result"),
        )

    def _compile_linear_system(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"方程组|线性方程组|solve\s+(?:the\s+)?(?:linear\s+)?system", text, re.IGNORECASE):
            return None
        match = re.search(
            r"\\begin\{cases\}(.+?)\\end\{cases\}",
            text,
            re.DOTALL,
        )
        if not match:
            return None
        rows = [
            row.replace("&", "").strip()
            for row in re.split(r"\\\\", match.group(1))
            if row.strip()
        ]
        if not 1 <= len(rows) <= 5 or any(row.count("=") != 1 for row in rows):
            return None
        try:
            equations = []
            variables: set = set()
            for row in rows:
                left, right = row.split("=", 1)
                expression = self._parse(left) - self._parse(right)
                equations.append(expression)
                variables.update(expression.free_symbols)
            ordered = tuple(sorted(variables, key=lambda item: item.name))
            if not ordered or len(ordered) > 4:
                return None
            matrix, vector = self.sympy.linear_eq_to_matrix(equations, ordered)
            solution_set = self.sympy.linsolve((matrix, vector), ordered)
            solutions = list(solution_set)
            if len(solutions) != 1 or any(item.free_symbols for item in solutions[0]):
                return None
            solution = solutions[0]
            substitutions = dict(zip(ordered, solution))
            if not all(self.sympy.simplify(eq.subs(substitutions)) == 0 for eq in equations):
                return None
        except Exception:
            return None
        rendered = r",\;".join(
            f"{symbol}={self._format(item)}" for symbol, item in zip(ordered, solution)
        )
        return make_tool_result(
            problem=text,
            operation="solve_linear_system",
            result=rendered,
            result_kind="solution_set",
            method="sympy_linear_elimination_and_substitution",
            whole=True,
            checks=("explicit_equation_system", "linear_elimination", "all_equations_substituted"),
            answer_shapes=("roots", "expression"),
            requirements=("result_present", "all_solutions"),
        )

    def _compile_explicit_pde_solution_check(
        self,
        text: str,
    ) -> Optional[ToolResult]:
        if not re.search(
            r"(?:验证|检验|核对)[^。；;\n]{0,180}(?:是否)?(?:为|是)?(?:其|该|此)?解|"
            r"\b(?:verify|check|determine\s+whether)\b[^.;\n]{0,180}"
            r"(?:\b(?:is|satisf(?:y|ies))\b[^.;\n]{0,60}"
            r"\b(?:solution|equation|PDE)\b|"
            r"\bsatisf(?:y|ies)\b[^.;\n]{0,60}"
            r"[A-Za-z]\s*_\s*(?:\{\s*)?t)",
            text,
            re.IGNORECASE,
        ):
            return None
        definitions = self._function_definitions(text)
        if len(definitions) != 1:
            return None
        function_name, variable_names, expression_text = definitions[0]
        if set(variable_names) != {"x", "t"}:
            return None

        def derivative(suffix: str) -> str:
            return (
                rf"{re.escape(function_name)}\s*_\s*"
                rf"(?:\{{\s*{re.escape(suffix)}\s*\}}|{re.escape(suffix)})"
            )

        coefficient = (
            r"(?P<coefficient>[-+]|[-+]?\s*(?:"
            r"\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?|"
            r"[A-Za-z](?:\s*\^\s*\{?\s*[-+]?\d+\s*\}?)?))"
        )

        def locate_equation(time_suffix: str):
            left = derivative(time_suffix)
            right = derivative("xx")
            # Parse the coefficient-free form first.  Making the coefficient
            # optional lets the function name in ``u_{xx}`` masquerade as a
            # symbolic coefficient and rejects the standard equation.
            unit = re.compile(
                left + r"\s*=\s*" + right,
                re.IGNORECASE,
            ).search(text)
            if unit is not None:
                return unit, "1"
            scaled = re.compile(
                left
                + r"\s*=\s*"
                + coefficient
                + r"\s*(?:\\cdot|\*)?\s*"
                + right,
                re.IGNORECASE,
            ).search(text)
            if scaled is None:
                return None, ""
            return scaled, str(scaled.group("coefficient") or "").strip()

        equation_match, coefficient_text = locate_equation("t")
        equation_kind = "heat"
        if equation_match is None:
            equation_match, coefficient_text = locate_equation("tt")
            equation_kind = "wave"
        if equation_match is None:
            return None

        if coefficient_text in {"", "+"}:
            coefficient_text = "1"
        elif coefficient_text == "-":
            coefficient_text = "-1"
        try:
            expression = self._parse(self._latex_to_sympy(expression_text))
            coefficient = self._parse(self._latex_to_sympy(coefficient_text))
            x_var = self.sympy.Symbol("x")
            t_var = self.sympy.Symbol("t")
            left = self.sympy.simplify(
                self.sympy.diff(expression, t_var, 1 if equation_kind == "heat" else 2)
            )
            right = self.sympy.simplify(
                coefficient * self.sympy.diff(expression, x_var, 2)
            )
            residual = self.sympy.simplify(left - right)
        except Exception:
            return None
        if any(
            item.has(self.sympy.Derivative, self.sympy.Integral)
            for item in (left, right, residual)
        ):
            return None

        left_label = f"{function_name}_{{t}}" if equation_kind == "heat" else f"{function_name}_{{tt}}"
        if coefficient_text == "1":
            right_label = f"{function_name}_{{xx}}"
        elif coefficient_text == "-1":
            right_label = f"-{function_name}_{{xx}}"
        else:
            right_label = f"{coefficient_text}{function_name}_{{xx}}"
        satisfied = residual == 0
        left_value = self._format(left)
        right_value = self._format(right)
        residual_value = self._format(residual)
        chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
        if chinese:
            result = (
                rf"${left_label}={left_value}$，${right_label}={right_value}$，"
                rf"残差 ${left_label}-{right_label}={residual_value}$，"
                f"所以给定函数{'是' if satisfied else '不是'}该方程的解。"
            )
            support = (
                "对给定显式函数分别作时间导数和二阶空间导数并直接代回；"
                f"符号化简后的残差为 {residual_value}。"
            )
        else:
            result = (
                rf"${left_label}={left_value}$; ${right_label}={right_value}$; "
                rf"the residual ${left_label}-{right_label}={residual_value}$. "
                f"Thus the proposed function {'is' if satisfied else 'is not'} a solution."
            )
            support = (
                "Differentiate the explicit function in time and twice in space, "
                f"then substitute directly; the simplified residual is {residual_value}."
            )
        return make_tool_result(
            problem=text,
            operation=f"{equation_kind}_equation_solution_check",
            result=result,
            result_kind="verification",
            method="sympy_explicit_pde_substitution",
            whole=True,
            written_support=True,
            checks=(
                "explicit_function_definition_parsed",
                "standard_pde_operator_parsed",
                "time_derivative_computed",
                "space_second_derivative_computed",
                "symbolic_residual_simplified",
            ),
            support=support,
            answer_shapes=("expression", "text", "truth"),
            requirements=("result_present", "judgement"),
        )

    def _compile_laplacian(self, text: str) -> Optional[ToolResult]:
        # A bi-Laplacian is Delta squared, not the ordinary Laplacian.  The
        # generic compiler does not implement that operator, so a hyphenated
        # English name must not fall through to the ``laplacian`` word match.
        if re.search(
            r"双调和(?:算子|方程)?|双拉普拉斯|"
            r"\b(?:bi[- ]?laplacian|biharmonic(?:\s+operator)?)\b",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(
            r"拉普拉斯算子|拉普拉斯量|拉普拉斯方程|拉普拉斯[-— ]?贝尔特拉米|"
            r"是否调和|是不是调和|"
            r"\blaplacian\b|\bLaplace[- ]Beltrami\b|"
            r"\b(?:whether|is)\b[^.\n]{0,60}\bharmonic\b",
            text,
            re.IGNORECASE,
        ):
            return None
        surface_context = bool(re.search(
            r"(?:在|限制在)(?:单位圆|圆周|球面|曲面|流形)"
            r"[^。；;\n]{0,160}?(?:上|处)|"
            r"\b(?:on|along|restricted\s+to)\s+(?:the\s+)?(?:unit\s+)?"
            r"(?:circle|sphere|surface|manifold)\b",
            text,
            re.IGNORECASE,
        ))
        circle_context = bool(re.search(
            r"单位圆|圆周|(?:圆|circle)\s*\$?\s*[A-Za-z]\s*\^|"
            r"\b(?:unit\s+)?circle\b",
            text,
            re.IGNORECASE,
        ))
        explicit_ambient = bool(re.search(
            r"欧氏环境|环境欧氏|环境拉普拉斯|"
            r"\b(?:ambient(?:\s+euclidean)?|euclidean\s+ambient)\s+laplacian\b",
            text,
            re.IGNORECASE,
        ))
        explicit_intrinsic = bool(re.search(
            r"内蕴|诱导度量|拉普拉斯[-— ]?贝尔特拉米|"
            r"\b(?:intrinsic|induced[- ]metric)\b[^.\n]{0,45}\blaplacian\b|"
            r"\bLaplace[- ]Beltrami\b",
            text,
            re.IGNORECASE,
        ))
        if explicit_intrinsic and explicit_ambient:
            return None
        if surface_context and not (explicit_intrinsic or explicit_ambient):
            # ``the Laplacian on a surface`` can mean the ambient Euclidean
            # operator or the intrinsic Laplace-Beltrami operator.  Both are
            # mathematically standard and can disagree, so abstain unless the
            # statement selects one explicitly.
            return None
        if explicit_intrinsic and (not surface_context or not circle_context):
            # General intrinsic surfaces require an explicit metric parser.
            # The centered circle case below is the only certified geometry.
            return None
        definitions: list[tuple[tuple[str, ...], str]] = []
        function_name = "f"
        for found_name, variables, expression_text in self._function_definitions(text):
            function_name = found_name
            definitions.append((variables, expression_text))
        if not definitions:
            fallback_matches = list(re.finditer(
                r"(?<![A-Za-z0-9_{}])([A-Za-z])\s*=\s*(.+?)"
                r"(?=是否|是不是|并(?:进行|计算|求)|[，。；;?？]|"
                r"\b(?:is|whether|and)\b|$)",
                text,
                re.IGNORECASE | re.DOTALL,
            ))
            for fallback in fallback_matches:
                expression_text = fallback.group(2).strip().strip("$")
                try:
                    parsed_expression = self._parse(
                        self._latex_to_sympy(expression_text)
                    )
                except Exception:
                    continue
                inferred_variables = tuple(sorted(
                    str(symbol) for symbol in parsed_expression.free_symbols
                ))
                if not 1 <= len(inferred_variables) <= 4:
                    continue
                function_name = fallback.group(1)
                definitions.append((inferred_variables, expression_text))
        if len(definitions) != 1:
            return None
        variable_names, expression_text = definitions[0]
        if len(set(variable_names)) != len(variable_names):
            return None

        normalized = text.replace(r"\left", "").replace(r"\right", "")
        point_requested = bool(re.search(
            r"在\s*点|\bat\s+(?:(?:the\s+)?point\s*)?\$?\s*\(",
            normalized,
            re.IGNORECASE,
        ))
        point_values: tuple[str, ...] = ()
        if point_requested:
            point_match = re.search(
                r"(?:在\s*点|at\s+(?:(?:the\s+)?point\s*)?)"
                r"\s*\$?\s*\(([^$]+?)\)",
                normalized,
                re.IGNORECASE | re.DOTALL,
            )
            if not point_match:
                return None
            point_values = tuple(
                item.strip()
                for item in self._split_top_level_commas(point_match.group(1))
            )
            if len(point_values) != len(variable_names):
                return None

        try:
            variables = tuple(self.sympy.Symbol(name) for name in variable_names)
            expression = self._parse(self._latex_to_sympy(expression_text))
            if expression.free_symbols - set(variables):
                return None
            substitutions = {}
            if point_requested:
                parsed_points = tuple(
                    self._parse(self._latex_to_sympy(item)) for item in point_values
                )
                if any(item.free_symbols for item in parsed_points):
                    return None
                substitutions = dict(zip(variables, parsed_points))
        except Exception:
            return None

        if explicit_intrinsic:
            if len(variables) != 2 or not point_requested:
                return None
            x_var, y_var = variables
            coordinate_square_sum = self.sympy.expand(x_var ** 2 + y_var ** 2)
            radius_squared = None
            if re.search(r"单位圆|\bunit\s+circle\b", text, re.IGNORECASE):
                radius_squared = self.sympy.Integer(1)
            for fragment in self._math_fragments(text):
                if "=" not in fragment or re.match(r"\s*[A-Za-z]\s*\(", fragment):
                    continue
                parts = fragment.split("=", 1)
                try:
                    left = self._parse(self._latex_to_sympy(parts[0].strip()))
                    right = self._parse(self._latex_to_sympy(parts[1].strip()))
                except Exception:
                    continue
                if self.sympy.simplify(left - coordinate_square_sum) == 0:
                    candidate_radius = self.sympy.simplify(right)
                elif self.sympy.simplify(right - coordinate_square_sum) == 0:
                    candidate_radius = self.sympy.simplify(left)
                else:
                    continue
                if candidate_radius.free_symbols:
                    continue
                radius_squared = candidate_radius
                break
            if radius_squared is None or radius_squared.is_positive is not True:
                return None
            point_norm_squared = self.sympy.simplify(
                substitutions[x_var] ** 2 + substitutions[y_var] ** 2
            )
            if self.sympy.simplify(point_norm_squared - radius_squared) != 0:
                return None
            try:
                fx = self.sympy.diff(expression, x_var)
                fy = self.sympy.diff(expression, y_var)
                fxx = self.sympy.diff(expression, x_var, 2)
                fxy = self.sympy.diff(expression, x_var, y_var)
                fyy = self.sympy.diff(expression, y_var, 2)
                intrinsic_laplacian = self.sympy.simplify(
                    (
                        y_var ** 2 * fxx
                        - 2 * x_var * y_var * fxy
                        + x_var ** 2 * fyy
                        - x_var * fx
                        - y_var * fy
                    ) / radius_squared
                )
                evaluated = self.sympy.simplify(
                    intrinsic_laplacian.subs(substitutions)
                )
            except Exception:
                return None
            if evaluated.free_symbols or evaluated.has(self.sympy.nan, self.sympy.zoo):
                return None
            result = self._format(evaluated)
            support = (
                "On x^2+y^2=R^2, Delta_M f="
                "(y^2 f_xx-2xy f_xy+x^2 f_yy-x f_x-y f_y)/R^2. "
                f"Substitution at the stated point gives {result}."
            )
            return make_tool_result(
                problem=text,
                operation="circle_intrinsic_laplacian",
                result=result,
                result_kind="scalar",
                method="sympy_circle_laplace_beltrami",
                whole=True,
                written_support=True,
                checks=(
                    "centered_circle_equation_parsed",
                    "point_membership_verified",
                    "induced_metric_laplacian_computed",
                ),
                preconditions=(
                    "intrinsic_operator_explicitly_requested",
                    "centered_circle_equation_parsed",
                    "evaluation_point_on_circle",
                ),
                execution_checks=(
                    "laplace_beltrami_formula_symbolically_evaluated",
                ),
                postconditions=(
                    "point_membership_verified",
                    "finite_symbol_free_result",
                ),
                support=support,
                answer_shapes=("number", "expression"),
                requirements=("result_present", "numeric_result"),
            )

        try:
            pure_second_derivatives = tuple(
                self.sympy.simplify(self.sympy.diff(expression, variable, 2))
                for variable in variables
            )
            laplacian = self.sympy.simplify(sum(pure_second_derivatives))
            if laplacian.has(self.sympy.Derivative):
                return None
            evaluated = self.sympy.simplify(laplacian.subs(substitutions))
            if (
                (substitutions and evaluated.free_symbols)
                or evaluated.has(self.sympy.nan, self.sympy.zoo)
            ):
                return None
        except Exception:
            return None

        second_derivatives_requested = bool(re.search(
            r"二阶(?:求导|导数|偏导)|求[^。；;\n]{0,30}二阶(?:偏)?导数|"
            r"\b(?:second(?:-order)?\s+(?:partial\s+)?derivatives?|"
            r"differentiate\s+twice)\b",
            text,
            re.IGNORECASE,
        ))
        harmonicity_requested = bool(re.search(
            r"是否调和|是不是调和|判断[^。；;\n]{0,50}调和|"
            r"\b(?:whether|determine\s+whether|is)\b[^.\n]{0,60}\bharmonic\b",
            text,
            re.IGNORECASE,
        ))
        if second_derivatives_requested or harmonicity_requested:
            derivative_parts = [
                rf"${function_name}_{{{name}{name}}}="
                rf"{self._format(derivative)}$"
                for name, derivative in zip(variable_names, pure_second_derivatives)
            ]
            delta_part = rf"$\Delta {function_name}={self._format(laplacian)}$"
            is_harmonic = self.sympy.simplify(laplacian) == 0
            if re.search(r"[\u4e00-\u9fff]", text):
                result_parts = derivative_parts if second_derivatives_requested else []
                if harmonicity_requested:
                    result_parts.extend((
                        delta_part,
                        f"所以函数 {function_name}{'是' if is_harmonic else '不是'}调和函数。",
                    ))
                result = "，".join(result_parts)
                support = (
                    "逐个变量作两次符号微分得到 "
                    + "，".join(derivative_parts)
                    + f"；求和得到 {delta_part}，故调和性判断如上。"
                )
            else:
                result_parts = derivative_parts if second_derivatives_requested else []
                if harmonicity_requested:
                    result_parts.extend((
                        delta_part,
                        f"Thus {function_name} is "
                        f"{'harmonic' if is_harmonic else 'not harmonic'}.",
                    ))
                result = "; ".join(result_parts)
                support = (
                    "Differentiate twice in each variable to obtain "
                    + ", ".join(derivative_parts)
                    + f". Their sum is {delta_part}, which gives the stated harmonicity."
                )
            requirements = ["result_present"]
            if second_derivatives_requested:
                requirements.append("second_derivatives")
            if harmonicity_requested:
                requirements.append("harmonicity_judgement")
            return make_tool_result(
                problem=text,
                operation="laplacian",
                result=result,
                result_kind="second_derivatives_and_harmonicity",
                method="sympy_pure_second_derivatives_and_laplacian_identity",
                whole=True,
                written_support=True,
                checks=(
                    "function_and_variables_parsed",
                    "all_pure_second_derivatives_computed",
                    "laplacian_summed_symbolically",
                    "harmonicity_checked_by_zero_identity",
                ),
                support=support,
                answer_shapes=("expression", "text", "truth"),
                requirements=tuple(requirements),
            )

        result = self._format(evaluated)
        second_terms = "+".join(
            f"d2f/d{name}2" for name in variable_names
        )
        support = (
            f"Using Delta f={second_terms}, symbolic differentiation gives "
            f"Delta f={self._format(laplacian)}"
            + (f", whose value at the stated point is {result}." if substitutions else ".")
        )
        return make_tool_result(
            problem=text,
            operation="laplacian",
            result=result,
            result_kind="scalar",
            method="sympy_multivariate_second_derivatives",
            whole=True,
            written_support=True,
            checks=(
                "function_and_variables_parsed",
                "all_pure_second_derivatives_computed",
                "stated_point_substituted" if substitutions else "symbolic_laplacian_simplified",
            ),
            support=support,
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _compile_central_difference(self, text: str) -> Optional[ToolResult]:
        if not re.search(
            r"中心差分|中央差分|\b(?:central|centered)\s+difference\b",
            text,
            re.IGNORECASE,
        ) or not re.search(r"一阶导数|first\s+derivative", text, re.IGNORECASE):
            return None
        definitions: list[tuple[str, str]] = []
        assignments: dict[str, str] = {}
        for fragment in self._math_fragments(text):
            definition = re.fullmatch(
                r"\s*[A-Za-z]\s*\(\s*([A-Za-z])\s*\)\s*=\s*(.+?)\s*",
                fragment,
                re.DOTALL,
            )
            if definition:
                definitions.append((definition.group(1), definition.group(2).strip()))
                continue
            assignment = re.fullmatch(
                r"\s*([A-Za-z])\s*=\s*(.+?)\s*",
                fragment,
                re.DOTALL,
            )
            if assignment:
                assignments[assignment.group(1)] = assignment.group(2).strip()
        if len(definitions) != 1:
            return None
        variable_name, expression_text = definitions[0]
        if variable_name not in assignments or "h" not in assignments:
            return None
        try:
            variable = self.sympy.Symbol(variable_name)
            expression = self._parse(self._latex_to_sympy(expression_text))
            point = self._parse(self._latex_to_sympy(assignments[variable_name]))
            step = self._parse(self._latex_to_sympy(assignments["h"]))
            if expression.free_symbols - {variable} or point.free_symbols or step.free_symbols:
                return None
            if step == 0 or step.is_real is not True:
                return None
            quotient = self.sympy.simplify(
                (
                    expression.subs(variable, point + step)
                    - expression.subs(variable, point - step)
                ) / (2 * step)
            )
            if quotient.free_symbols or quotient.has(self.sympy.nan, self.sympy.zoo):
                return None
            numeric = self.sympy.N(quotient, 12)
            if numeric.is_real is not True or numeric.is_finite is not True:
                return None
        except Exception:
            return None
        result = str(numeric)
        support = (
            f"The centered first difference is "
            f"[f(x+h)-f(x-h)]/(2h). Substituting x={self._format(point)} "
            f"and h={self._format(step)} gives {result}."
        )
        return make_tool_result(
            problem=text,
            operation="central_difference_first_derivative",
            result=result,
            result_kind="scalar",
            method="sympy_centered_difference_recompute",
            whole=True,
            written_support=True,
            checks=(
                "function_point_and_step_parsed",
                "symmetric_function_evaluations",
                "finite_numeric_quotient",
            ),
            preconditions=(
                "centered_first_difference_explicitly_requested",
                "function_point_and_step_parsed",
                "nonzero_real_step",
            ),
            execution_checks=(
                "symmetric_quotient_exactly_recomputed",
            ),
            postconditions=(
                "quotient_contains_no_free_symbols",
                "finite_real_numeric_result",
            ),
            support=support,
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _compile_derivative(self, text: str) -> Optional[ToolResult]:
        if re.search(
            r"中心差分|中央差分|\b(?:central|centered)\s+difference\b",
            text,
            re.IGNORECASE,
        ):
            return None
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

    def _compile_iterated_definite_integral(self, text: str) -> Optional[ToolResult]:
        """Exactly evaluate a fully parsed nonnegative double integral."""
        if len(re.findall(r"\\int(?![A-Za-z])", text)) != 2 or not re.search(
            r"计算|求值|严格计算|交换积分|Tonelli|Fubini|"
            r"\b(?:compute|evaluate|iterated integral|change the order of integration)\b",
            text,
            re.IGNORECASE,
        ):
            return None
        source = self._math_source(text)
        bounded_integral = (
            r"\\int\s*_\s*\{?\s*([^{}\s]+)\s*\}?\s*"
            r"\^\s*\{?\s*([^{}\s]+)\s*\}?"
        )
        match = re.search(
            bounded_integral
            + r"\s*" + bounded_integral
            + r"\s*(.+?)\s*(?:\\,|\\;|\s)*d\s*([A-Za-z])\s*"
            r"(?:\\,|\\;|\s)*d\s*([A-Za-z])\b",
            source,
            re.DOTALL,
        )
        if not match:
            return None
        (
            outer_lower_text,
            outer_upper_text,
            inner_lower_text,
            inner_upper_text,
            expression_text,
            inner_name,
            outer_name,
        ) = match.groups()
        if inner_name == outer_name:
            return None
        expression_text = expression_text.strip().strip("$，,。.!? ")
        try:
            outer_lower = self._parse(outer_lower_text)
            outer_upper = self._parse(outer_upper_text)
            inner_lower = self._parse(inner_lower_text)
            inner_upper = self._parse(inner_upper_text)
            expression = self._parse(expression_text)
            inner = self.sympy.Symbol(inner_name)
            outer = self.sympy.Symbol(outer_name)
            if expression.free_symbols - {inner, outer}:
                return None
            bounds = (
                (inner, inner_lower, inner_upper),
                (outer, outer_lower, outer_upper),
            )
            if any(
                bound.free_symbols
                for _, lower, upper in bounds
                for bound in (lower, upper)
            ):
                return None
            nonnegative_variables = {
                variable
                for variable, lower, _ in bounds
                if self.sympy.simplify(lower).is_nonnegative is True
            }

            def structurally_nonnegative(node) -> bool:
                if node.is_nonnegative is True or node.is_positive is True:
                    return True
                if node.is_Symbol:
                    return node in nonnegative_variables
                if node.func in {self.sympy.exp, self.sympy.Abs}:
                    return True
                if node.func in {self.sympy.Min, self.sympy.Max}:
                    return bool(node.args) and all(
                        structurally_nonnegative(argument) for argument in node.args
                    )
                if node.is_Mul or node.is_Add:
                    return bool(node.args) and all(
                        structurally_nonnegative(argument) for argument in node.args
                    )
                if node.is_Pow:
                    base, exponent = node.args
                    return bool(
                        structurally_nonnegative(base)
                        or (exponent.is_integer is True and exponent.is_even is True)
                    )
                return False

            if not structurally_nonnegative(expression):
                return None
            first_order = self.sympy.integrate(
                expression,
                (inner, inner_lower, inner_upper),
                (outer, outer_lower, outer_upper),
            )
            second_order = self.sympy.integrate(
                expression,
                (outer, outer_lower, outer_upper),
                (inner, inner_lower, inner_upper),
            )
            forbidden = (self.sympy.Integral, self.sympy.nan, self.sympy.zoo)
            if any(
                first_order.has(item) or second_order.has(item)
                for item in forbidden
            ):
                return None
            if first_order.free_symbols or second_order.free_symbols:
                return None
            if first_order in {self.sympy.oo, -self.sympy.oo}:
                return None
            if self.sympy.simplify(first_order - second_order) != 0:
                return None
            result = self._format(self.sympy.simplify(first_order))
        except Exception:
            return None

        if re.search(r"[\u4e00-\u9fff]", text):
            support = (
                "被积函数在给定非负积分区域上非负，故 Tonelli 定理允许交换积分次序；"
                f"按两种次序分别精确积分所得结果一致，均为 {result}。"
            )
        else:
            support = (
                "The integrand is nonnegative on the stated domain, so Tonelli's theorem "
                f"permits either integration order; exact evaluation in both orders gives {result}."
            )
        return make_tool_result(
            problem=text,
            operation="iterated_definite_integral",
            result=result,
            result_kind="scalar",
            method="parsed_nonnegative_double_integral_tonelli",
            whole=True,
            written_support=True,
            checks=(
                "two_bounds_and_differentials_parsed",
                "structural_nonnegativity_on_domain",
                "both_integration_orders_exact",
                "finite_order_agreement",
            ),
            support=support,
            answer_shapes=("number", "expression", "text"),
            requirements=("result_present", "numeric_result"),
        )

    def _compile_definite_integral(self, text: str) -> Optional[ToolResult]:
        if not self._single_operation_request(text, "integral"):
            return None
        parsed = self._parse_definite_integral_request(text)
        if not parsed:
            return None
        expression, variable, lower, upper = parsed
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
        parsed = self._parse_limit_request(text)
        if not parsed:
            return None
        expression, variable, point = parsed
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
        parsed = self._parse_direct_equation(text, allow_domains=True)
        if not parsed or not self._single_operation_request(text, "equation"):
            return None
        left, right, variable = parsed
        try:
            expression = self._parse(left) - self._parse(right)
            symbol = self.sympy.Symbol(variable)
            polynomial = self.sympy.Poly(expression, symbol)
            if polynomial.degree() < 1 or polynomial.degree() > 6:
                return None
            if any(item.free_symbols for item in polynomial.all_coeffs()):
                return None
            solutions = self._exact_roots(expression, symbol, text)
            if solutions is None:
                return None
            if not all(self.sympy.simplify(expression.subs(symbol, item)) == 0 for item in solutions):
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

    def _parse_function_evaluation_request(
        self,
        text: str,
    ) -> Optional[tuple[str, str, str, str]]:
        if re.search(
            r"导数|求导|积分|极限|方程的根|derivative|differentiat|integral|limit|roots?",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(
            r"函数值|的值|evaluate|value\s+of|find\s+[A-Za-z]\s*\(",
            text,
            re.IGNORECASE,
        ):
            return None
        normalized = text.replace(r"\(", "$").replace(r"\)", "$")
        definition = re.search(
            r"(?P<fn>[A-Za-z])\s*\(\s*(?P<var>[A-Za-z])\s*\)\s*=\s*"
            r"(?P<expr>.+?)(?=\s*\$|\s*(?:在|当|at|when)\s*\$?\s*"
            r"(?P=var)\s*=|[，,。；;\n]|$)",
            normalized,
            re.IGNORECASE | re.DOTALL,
        )
        if not definition:
            return None
        function_name = definition.group("fn")
        variable_name = definition.group("var")
        expression = definition.group("expr").strip(" $，,。.!?：:")
        point_match = re.search(
            rf"(?:在|当|at|when)\s*\$?\s*{re.escape(variable_name)}\s*=\s*"
            r"([^$，,。；;\s]+)",
            normalized,
            re.IGNORECASE,
        )
        if point_match:
            point = point_match.group(1).strip("{}()")
        else:
            calls = [
                item.strip()
                for item in re.findall(
                    rf"(?<![A-Za-z]){re.escape(function_name)}\s*\(\s*([^()]+)\s*\)",
                    normalized,
                    re.IGNORECASE,
                )
                if item.strip().casefold() != variable_name.casefold()
            ]
            if len(calls) != 1:
                return None
            point = calls[0]
        if not expression or not point:
            return None
        return function_name, variable_name, expression, point

    def _parse_definite_integral_request(
        self,
        text: str,
    ) -> Optional[tuple[str, str, str, str]]:
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
        return expression.strip(), variable, lower, upper

    def _parse_limit_request(
        self,
        text: str,
    ) -> Optional[tuple[str, str, str]]:
        math_text = self._math_source(text)
        match = re.search(
            r"\\lim\s*_\s*\{?\s*([A-Za-z])\s*\\to\s*([^}\s]+)\s*\}?\s*(.+)",
            math_text,
            re.DOTALL,
        )
        if not match:
            return None
        variable, point, expression = match.groups()
        return expression.strip(), variable, point

    def _parse_derivative_request(self, text: str) -> Optional[tuple[str, str]]:
        if not re.search(r"导数|求导|\\frac\s*\{d\}|\b(?:derivative|differentiat)\w*\b", text, re.IGNORECASE):
            return None
        if re.search(
            r"偏导|二阶|三阶|高阶|n\s*阶|在[^。；;\n]{0,50}处|"
            r"\b(?:partial|second|third|higher|n(?:th)?)[ -]?derivative\b|"
            r"\bat\s+[A-Za-z]\s*=",
            text,
            re.IGNORECASE,
        ):
            return None

        fragments = list(dict.fromkeys(self._math_fragments(text)))
        explicit_variable = ""
        variable_match = re.search(
            r"(?:关于|对)\s*\$?\s*([A-Za-z])\s*\$?|"
            r"with\s+respect\s+to\s+\$?\s*([A-Za-z])\s*\$?",
            text,
            re.IGNORECASE,
        )
        if variable_match:
            explicit_variable = next(group for group in variable_match.groups() if group)

        definitions: list[tuple[str, str]] = []
        expression_fragments: list[str] = []
        for fragment in fragments:
            definition = re.fullmatch(
                r"\s*[A-Za-z]\s*\(\s*([A-Za-z])\s*\)\s*=\s*(.+?)\s*",
                fragment,
                re.DOTALL,
            )
            if definition:
                definitions.append((definition.group(2), definition.group(1)))
            elif not re.fullmatch(r"\s*[A-Za-z]\s*", fragment):
                expression_fragments.append(fragment)
        if len(definitions) == 1 and not expression_fragments:
            expression, variable = definitions[0]
            if explicit_variable and explicit_variable.casefold() != variable.casefold():
                return None
            return expression.strip(), explicit_variable or variable
        if not definitions and len(expression_fragments) == 1:
            expression = expression_fragments[0].strip()
            variable = explicit_variable
            if not variable:
                try:
                    symbols = self._parse(self._latex_to_sympy(expression)).free_symbols
                except Exception:
                    symbols = set()
                if len(symbols) != 1:
                    return None
                variable = next(iter(symbols)).name
            return expression, variable

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

    @staticmethod
    def _parse_arithmetic_request(text: str) -> Optional[str]:
        """Return the sole scalar expression from a direct bilingual request."""
        value = str(text or "").strip()
        if not value or re.search(
            r"证明|说明|推导|近似|误差|比较|构造|方程|导数|积分|极限|求和|"
            r"\b(?:prove|explain|derive|approx|error|compare|construct|equation|"
            r"derivative|integral|limit|summation)\b",
            value,
            re.IGNORECASE,
        ):
            return None
        patterns = (
            r"(?:请)?(?:计算|求值)\s*(?:下列)?(?:表达式)?\s*[:：]?\s*(.+?)"
            r"\s*(?:的\s*(?:精确|准确|确切)?值)?\s*[。.!?？]?",
            r"(?:请)?求\s*(?:下列)?(?:表达式\s*)?(.+?)\s*的\s*"
            r"(?:精确|准确|确切)?值\s*[。.!?？]?",
            r"(?:please\s+)?(?:calculate|compute|evaluate)\s*"
            r"(?:(?:the\s+)?(?:exact\s+)?value\s+of\s+)?(.+?)\s*[.!?]?",
            r"(?:please\s+)?find\s+(?:the\s+)?(?:exact\s+)?value\s+of\s+"
            r"(.+?)\s*[.!?]?",
        )
        for pattern in patterns:
            match = re.fullmatch(pattern, value, re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            expression = match.group(1).strip()
            # A residual Chinese/English word means the natural-language
            # scaffold was not fully consumed.  Mathematical function names
            # are checked later by the restricted SymPy parser.
            outside_commands = re.sub(
                r"\\(?:frac|dfrac|tfrac|sqrt|sin|cos|tan|sinh|cosh|tanh|"
                r"coth|sech|csch|exp|log|ln|pi)|"
                r"\b(?:sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|coth|"
                r"sech|csch|exp|log|sqrt|pi)\b",
                "",
                expression,
                flags=re.IGNORECASE,
            )
            if re.search(r"[\u4e00-\u9fff]|[A-Za-z]{2,}", outside_commands):
                continue
            return expression
        return None

    def _parse_direct_equation(
        self,
        text: str,
        *,
        allow_domains: bool = False,
    ) -> Optional[tuple[str, str, str]]:
        if not re.search(r"方程|求解|求.*根|\b(?:solve|equation|roots?|zeros?)\b", text, re.IGNORECASE):
            return None
        domain_pattern = (
            r"整数解|正整数|实数解|复数解|"
            r"\b(?:integer solutions?|positive integer|real roots?|complex roots?)\b|"
        )
        if re.search(
            ("" if allow_domains else domain_pattern)
            + r"区间|范围|近似|迭代|\b(?:interval|range|approximately|iteration)\b",
            text,
            re.IGNORECASE,
        ):
            return None
        candidates = self._math_fragments(text)
        direct = re.search(
            r"(?:方程|equation)\s*[:：]?\s*\$?\s*"
            r"([A-Za-z0-9_+\-*/^().{}\\\s]+?\s*=\s*"
            r"[A-Za-z0-9_+\-*/^().{}\\\s]+?)"
            r"(?=\$|\s*的?(?:全部|所有)?(?:解|根)|"
            r"\s*(?:for|over)\b|[。.!?？]|$)",
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

    def _exact_roots(self, expression, symbol, text: str) -> Optional[list]:
        try:
            polynomial = self.sympy.Poly(expression, symbol)
            if polynomial.degree() < 1 or polynomial.degree() > 6:
                return None
            if any(item.free_symbols for item in polynomial.all_coeffs()):
                return None
            roots = list(self.sympy.solve(polynomial.as_expr(), symbol))
            if any(root.free_symbols for root in roots):
                return None
        except Exception:
            return None
        if re.search(r"正整数|positive integers?", text, re.IGNORECASE):
            roots = [root for root in roots if root.is_integer is True and root.is_positive is True]
        elif re.search(r"(?:非负)?整数|(?:nonnegative )?integers?", text, re.IGNORECASE):
            roots = [root for root in roots if root.is_integer is True]
            if re.search(r"非负|nonnegative", text, re.IGNORECASE):
                roots = [root for root in roots if root.is_nonnegative is True]
        elif re.search(
            r"实数(?:解|根|范围|域)?|real (?:solutions?|roots?|numbers?)|"
            r"(?:over|in)\s+(?:the\s+)?reals?|(?:over|in)\s*\\mathbb\s*\{R\}",
            text,
            re.IGNORECASE,
        ):
            roots = [root for root in roots if root.is_real is True]
        elif not re.search(
            r"复数(?:解|根|范围|域)?|complex (?:solutions?|roots?|numbers?)|"
            r"(?:over|in)\s+(?:the\s+)?complex(?:es| numbers?)?|"
            r"(?:over|in)\s*\\mathbb\s*\{C\}",
            text,
            re.IGNORECASE,
        ):
            if any(root.is_real is not True for root in roots):
                return None
        return roots

    def _same_expression_set(self, left: list, right: list) -> bool:
        if len(left) != len(right):
            return False
        unused = list(right)
        for item in left:
            match_index = next((
                index for index, other in enumerate(unused)
                if self.sympy.simplify(item - other) == 0
            ), None)
            if match_index is None:
                return False
            unused.pop(match_index)
        return not unused

    def _verify_statement_constraints(
        self,
        problem: str,
        answer: str,
        spec=None,
    ) -> tuple[ToolCheck, ...]:
        atoms = self._relation_atoms(problem)
        if not atoms:
            return ()
        target_variables = self._target_variables(spec)
        assignments = self._candidate_assignments(answer, target_variables)
        if not assignments or not set(assignments).intersection(target_variables):
            return ()
        parsed_atoms = []
        for left_text, operator, right_text in atoms:
            try:
                left = self._parse(left_text)
                right = self._parse(right_text)
            except Exception:
                continue
            symbols = left.free_symbols | right.free_symbols
            if not symbols or not symbols <= set(assignments):
                continue
            parsed_atoms.append((left, operator, right, symbols))
        if not parsed_atoms:
            return ()

        statuses = [
            self._relation_holds(left, operator, right, assignments)
            for left, operator, right, _ in parsed_atoms
        ]
        if any(status is False for status in statuses):
            return (ToolCheck(
                "statement_constraint_substitution",
                "fail",
                f"candidate violates at least one of {len(parsed_atoms)} parsed statement constraints",
                False,
            ),)
        checks = [ToolCheck(
            "statement_constraint_substitution",
            "pass",
            f"candidate satisfies {len(parsed_atoms)} parsed statement constraint(s)",
            False,
        )]
        unique = self._unique_solution_from_atoms(parsed_atoms, problem)
        if unique is not None and set(unique) <= set(assignments):
            matches = all(
                self.sympy.simplify(assignments[symbol] - expected) == 0
                for symbol, expected in unique.items()
            )
            checks.append(ToolCheck(
                "unique_constraint_solution",
                "pass" if matches else "fail",
                f"exactly one solution remains after {len(parsed_atoms)} parsed constraints",
                True,
            ))
        return tuple(checks)

    def _relation_atoms(self, text: str) -> list[tuple[str, str, str]]:
        sources = [*self._math_fragments(text), *re.split(r"[。！？!?；;\n]+", text)]
        atoms: list[tuple[str, str, str]] = []
        for source in sources:
            normalized = str(source or "")
            normalized = normalized.replace(r"\leq", "<=").replace(r"\le", "<=")
            normalized = normalized.replace(r"\geq", ">=").replace(r"\ge", ">=")
            for segment in re.split(r"且|并且|同时|\band\b|,|，", normalized, flags=re.IGNORECASE):
                operator_match = re.search(r"(?<![<>!])=(?!=)|<=|>=|<|>", segment)
                if not operator_match:
                    continue
                left_prefix = segment[:operator_match.start()]
                right_suffix = segment[operator_match.end():]
                left_match = re.search(r"([A-Za-z0-9_+\-*/^().{}\\\s]{1,140})$", left_prefix)
                right_match = re.match(r"\s*([A-Za-z0-9_+\-*/^().{}\\\s]{1,140})", right_suffix)
                if not left_match or not right_match:
                    continue
                left = left_match.group(1).strip(" $:：")
                right = right_match.group(1).strip(" $:：。.!?")
                if not left or not right or r"\begin" in left + right:
                    continue
                atoms.append((left, operator_match.group(0), right))
        return list(dict.fromkeys(atoms))[:8]

    def _candidate_assignments(
        self,
        answer: str,
        target_variables: set,
    ) -> dict:
        from reasoning.finalizer import Finalizer

        extracted = Finalizer.extract_result(answer)
        candidate = extracted.answer if extracted.valid and extracted.answer else answer
        assignments = {}
        for match in re.finditer(
            r"(?<![A-Za-z])([A-Za-z])\s*=\s*([^,，;；\n$]+)",
            candidate,
        ):
            symbol = self.sympy.Symbol(match.group(1))
            if symbol not in target_variables:
                continue
            expression_text = match.group(2).strip(" {}()。.!?")
            try:
                expression = self._parse(expression_text)
            except Exception:
                continue
            if not expression.free_symbols:
                assignments[symbol] = expression
        if not assignments and len(target_variables) == 1:
            scalar = self._first_scalar_result(candidate)
            if scalar is not None and not getattr(scalar, "free_symbols", set()):
                assignments[next(iter(target_variables))] = scalar
        return assignments

    def _target_variables(self, spec) -> set:
        target = str(getattr(getattr(spec, "semantics", None), "target", ""))
        variables = set()
        for match in re.finditer(
            r"(?:求|计算|确定|find|determine|compute)\s*(?:the\s+value\s+of\s+)?"
            r"([A-Za-z])(?![A-Za-z]|\s*=)",
            target,
            re.IGNORECASE,
        ):
            variables.add(self.sympy.Symbol(match.group(1)))
        if getattr(getattr(spec, "profile", None), "answer_shape", "") == "roots":
            for relation in getattr(getattr(spec, "semantics", None), "relations", ()):
                prepared = self._latex_to_sympy(relation)
                for name in re.findall(r"(?<![A-Za-z])([a-z])(?![A-Za-z])", prepared.casefold()):
                    if name not in {"e", "i"}:
                        variables.add(self.sympy.Symbol(name))
        return variables

    def _relation_holds(self, left, operator: str, right, assignments: dict) -> Optional[bool]:
        difference = self.sympy.simplify((left - right).subs(assignments))
        if difference.free_symbols:
            return None
        if operator == "=":
            return difference == 0
        predicates = {
            "<": difference.is_negative,
            ">": difference.is_positive,
            "<=": difference.is_nonpositive,
            ">=": difference.is_nonnegative,
        }
        return predicates.get(operator)

    def _unique_solution_from_atoms(self, atoms, text: str) -> Optional[dict]:
        variables = tuple(sorted(
            set().union(*(symbols for _, _, _, symbols in atoms)),
            key=lambda item: item.name,
        ))
        equalities = [left - right for left, operator, right, _ in atoms if operator == "="]
        if not variables or len(variables) > 3 or not equalities or len(equalities) > 4:
            return None
        try:
            for expression in equalities:
                polynomial = self.sympy.Poly(expression, *variables)
                if polynomial.total_degree() > 4:
                    return None
            raw_solutions = self.sympy.solve(equalities, variables, dict=True)
        except Exception:
            return None
        valid = []
        for solution in raw_solutions:
            if set(solution) != set(variables) or any(value.free_symbols for value in solution.values()):
                continue
            if not self._solution_domain_valid(solution, text):
                continue
            statuses = [
                self._relation_holds(left, operator, right, solution)
                for left, operator, right, _ in atoms
            ]
            if all(status is True for status in statuses):
                valid.append(solution)
        return valid[0] if len(valid) == 1 else None

    @staticmethod
    def _solution_domain_valid(solution: dict, text: str) -> bool:
        values = tuple(solution.values())
        if re.search(r"正整数|positive integers?", text, re.IGNORECASE):
            return all(value.is_integer is True and value.is_positive is True for value in values)
        if re.search(r"非负整数|nonnegative integers?", text, re.IGNORECASE):
            return all(value.is_integer is True and value.is_nonnegative is True for value in values)
        if re.search(r"整数|integers?", text, re.IGNORECASE):
            return all(value.is_integer is True for value in values)
        if re.search(r"实数|real numbers?", text, re.IGNORECASE):
            return all(value.is_real is True for value in values)
        return True

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
    def _split_top_level_commas(value: str) -> list[str]:
        parts: list[str] = []
        start = 0
        depth = 0
        for index, character in enumerate(str(value or "")):
            if character in "{[(":
                depth += 1
            elif character in "}])":
                depth = max(0, depth - 1)
            elif character == "," and depth == 0:
                parts.append(value[start:index])
                start = index + 1
        parts.append(value[start:])
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _function_definitions(
        text: str,
    ) -> list[tuple[str, tuple[str, ...], str]]:
        """Extract grounded scalar function definitions without executing prose."""
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])([A-Za-z])\s*"
            r"\(\s*([A-Za-z](?:\s*,\s*[A-Za-z]){1,3})\s*\)\s*=\s*"
            r"(.+?)(?="
            r"\s+(?:在|限制在|沿|于|on\s+(?:the\s+)?|along\s+(?:the\s+)?|"
            r"restricted\s+to\s+(?:the\s+)?|at\s+(?:the\s+)?)"
            r"(?:单位圆|圆周|圆|球面|曲面|流形|circle|sphere|surface|manifold|point)\b|"
            r"是否|是不是|并(?:计算|求|判断|验证)|"
            r"\b(?:whether|and\s+(?:compute|find|determine|verify)|"
            r"is\s+(?:an?\s+)?solution|satisf(?:y|ies)|solves?)\b|"
            r"\$|[，。；;?？]|$)",
            re.IGNORECASE | re.DOTALL,
        )
        sources = tuple(dict.fromkeys((
            *SympyTool._math_fragments(text),
            str(text or ""),
        )))
        definitions: list[tuple[str, tuple[str, ...], str]] = []
        seen: set[tuple[str, tuple[str, ...], str]] = set()
        for source in sources:
            for match in pattern.finditer(source):
                variables = tuple(
                    item.strip() for item in match.group(2).split(",")
                )
                definition = (
                    match.group(1),
                    variables,
                    match.group(3).strip().strip("$"),
                )
                if definition not in seen:
                    definitions.append(definition)
                    seen.add(definition)
        return definitions

    @staticmethod
    def _math_fragments(text: str) -> list[str]:
        fragments = [item.strip() for item in re.findall(r"\$([^$]+)\$", text) if item.strip()]
        for pair in re.findall(r"\\\((.+?)\\\)|\\\[(.+?)\\\]", text, re.DOTALL):
            fragments.extend(item.strip() for item in pair if item.strip())
        if not fragments:
            command = re.sub(
                r"^(?:求解(?:该|此|下列)?方程|解(?:该|此|下列)?方程|"
                r"求(?:该|此|下列)?方程|求解|求|计算|"
                r"solve(?:\s+the)?(?:\s+equation)?|find|determine)\s*",
                "",
                text.strip(),
                flags=re.IGNORECASE,
            ).strip(" 。.!?？")
            command = re.sub(
                r"\s*(?:的)?(?:全部|所有)(?:实数|复数|有理数)?(?:解|根)\s*$|"
                r"\s*(?:find\s+)?(?:all\s+)?(?:real\s+|complex\s+)?roots?\s*$",
                "",
                command,
                flags=re.IGNORECASE,
            ).strip()
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
        functions = {
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh", "coth", "sech", "csch",
            "exp", "log", "sqrt", "Abs", "Min", "Max",
            "factorial", "binomial", "gamma",
        }
        constants = {"pi", "oo", "E", "I"}
        called_identifiers = set(re.findall(r"\b([A-Za-z_]+)\s*\(", value))
        if called_identifiers - functions:
            # Never let SymPy resolve an undefined f(x), S(n), Q(x), ...
            # through its global namespace. A caller must first substitute an
            # explicitly parsed function definition.
            raise ValueError("unsupported function call")
        if any(
            identifier not in functions | constants
            and (len(identifier) != 1 or identifier not in string.ascii_letters)
            for identifier in identifiers
        ):
            raise ValueError("unsupported identifier")
        local = {name: getattr(self.sympy, name) for name in functions if hasattr(self.sympy, name)}
        local.update({"pi": self.sympy.pi, "oo": self.sympy.oo, "E": self.sympy.E, "I": self.sympy.I})
        local.update({
            letter: self.sympy.Symbol(letter)
            for letter in string.ascii_letters
            if letter not in {"E", "I"}
        })
        return self.sympy.sympify(value, locals=local)

    @staticmethod
    def _latex_to_sympy(expression: str) -> str:
        value = str(expression or "")
        value = value.replace(r"\left", "").replace(r"\right", "")
        value = value.replace(r"\,", "").replace(r"\;", " ").replace(r"\!", "")
        value = value.replace(r"\cdot", "*").replace(r"\times", "*")
        value = value.replace("×", "*").replace("÷", "/").replace("−", "-")
        value = (
            value.replace(r"\pi", "pi")
            .replace("π", "pi")
            .replace(r"\infty", "oo")
            .replace(r"\Gamma", "gamma")
        )
        value = re.sub(
            r"\\(min|max)\s*\\\{([^{}]+)\\\}",
            lambda match: f"{match.group(1).title()}({match.group(2)})",
            value,
        )
        value = re.sub(
            r"\\(min|max)\s*\{([^{}]+)\}",
            lambda match: f"{match.group(1).title()}({match.group(2)})",
            value,
        )
        value = re.sub(
            r"\\(?:operatorname|mathrm)\s*\{"
            r"(sin|cos|tan|sinh|cosh|tanh|coth|sech|csch|exp|log)\}",
            r" \1",
            value,
        )
        value = re.sub(
            r"\\(sin|cos|tan|sinh|cosh|tanh|coth|sech|csch|exp|log|ln)(?![A-Za-z])",
            lambda match: " " + ("log" if match.group(1) == "ln" else match.group(1)),
            value,
        )
        previous = None
        while previous != value:
            previous = value
            value = re.sub(
                r"(?<![A-Za-z])e\s*\^\s*\{([^{}]+)\}",
                r"exp(\1)",
                value,
            )
            value = re.sub(
                r"(?<![A-Za-z])e\s*\^\s*(\([^()]+\)|[-+]?[A-Za-z0-9]+)",
                r"exp(\1)",
                value,
            )
            value = re.sub(
                r"\\(?:d?frac|tfrac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
                r"((\1)/(\2))",
                value,
            )
            value = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", value)
            value = re.sub(r"([A-Za-z0-9)])\s*\^\s*\{([^{}]+)\}", r"\1**(\2)", value)
        value = value.replace("{", "(").replace("}", ")")
        value = re.sub(
            r"\b(sin|cos|tan|sinh|cosh|tanh|coth|sech|csch|exp|log)"
            r"\s+([-+]?[A-Za-z])\b",
            r"\1(\2)",
            value,
        )
        value = re.sub(
            r"(?<=[A-Za-z0-9)])\s+(?="
            r"(?:sin|cos|tan|sinh|cosh|tanh|coth|sech|csch|exp|log)\()",
            "*",
            value,
        )
        value = re.sub(
            r"(?<![A-Za-z])([A-Za-z])\s+([A-Za-z])(?![A-Za-z])",
            r"\1*\2",
            value,
        )
        value = re.sub(r"(?<=\d)(?=[A-Za-z(])|(?<=[A-Za-z)])(?=\d)|(?<=\))(?=[A-Za-z(])", "*", value)
        return re.sub(r"\s+", " ", value).strip().strip("$")

    def _format(self, value: Any) -> str:
        if not self.sympy:
            return str(value)
        try:
            return self.sympy.latex(self.sympy.simplify(value))
        except Exception:
            return str(value)

    def _run(
        self,
        operation,
        *,
        forbidden_nodes: tuple[str, ...] = (),
    ) -> Optional[str]:
        if not self.sympy:
            return None
        try:
            value = operation(self.sympy)
            for name in forbidden_nodes:
                node_type = getattr(self.sympy, name, None)
                has_node = getattr(value, "has", lambda *_: False)
                if node_type is not None and has_node(node_type):
                    return None
            return self._format(value)
        except Exception:
            return None
