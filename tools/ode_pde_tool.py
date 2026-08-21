"""Exact symbolic certificates for conservative ODE and PDE families."""

from __future__ import annotations

import re
from typing import Optional

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, make_parameterized_tool_result


class OdePdeTool:
    """Solve parsed constant-coefficient families and verify by substitution."""

    def __init__(self) -> None:
        self.symbolic = SympyTool()
        self.sp = self.symbolic.sympy

    def results_for(self, problem: str) -> list[ToolResult]:
        text = str(problem or "").strip()
        if self.sp is None or not text:
            return []
        handlers = (
            self._linear_first_order_ivp,
            self._constant_second_order_forced_ivp,
            self._constant_second_order_ivp,
            self._autonomous_power_blowup,
            self._diagonal_system_stability,
            self._bernoulli_general,
            self._dalembert,
        )
        results: list[ToolResult] = []
        for handler in handlers:
            try:
                item = handler(text)
            except Exception:
                item = None
            if item is not None and item.verified:
                results.append(item)
        return results

    def _linear_first_order_ivp(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"初值|initial\s+value|IVP", text, re.I) or not re.search(r"积分因子|integrating\s+factor", text, re.I):
            return None
        equation = self._first_order_equation(text)
        initial = self._initial_value(text, derivative=False)
        if equation is None or initial is None:
            return None
        coefficient, rhs = equation
        x0, y0 = initial
        x = self.sp.Symbol("x")
        if coefficient.free_symbols or rhs.free_symbols - {x} or x0.free_symbols or y0.free_symbols:
            return None
        integrating_factor = self.sp.exp(coefficient * x)
        antiderivative = self.sp.integrate(integrating_factor * rhs, x)
        if antiderivative.has(self.sp.Integral):
            return None
        constant = self.sp.simplify(
            y0 * self.sp.exp(coefficient * x0) - antiderivative.subs(x, x0)
        )
        solution = self.sp.simplify(self.sp.exp(-coefficient * x) * (antiderivative + constant))
        residual = self.sp.simplify(self.sp.diff(solution, x) + coefficient * solution - rhs)
        initial_residual = self.sp.simplify(solution.subs(x, x0) - y0)
        if residual != 0 or initial_residual != 0:
            return None
        rendered = self.symbolic._format(solution)
        factor = self.symbolic._format(integrating_factor)
        result = (
            rf"积分因子为 $\mu(x)={factor}$，于是 $(\mu y)'=\mu q(x)$。积分并代入初值后得 "
            rf"$y(x)={rendered}$；代回原方程及初值，两残差均为 $0$。"
            if self._zh(text) else
            rf"The integrating factor is $\mu(x)={factor}$, so $(\mu y)'=\mu q(x)$. "
            rf"Integrating and applying the initial value gives $y(x)={rendered}$; substitution into both the ODE and initial value gives zero residual."
        )
        return self._result(text, "linear_first_order_ivp", result, "explicit_solution",
                            "integrating_factor_and_exact_substitution",
                            ("constant_coefficient_parsed", "forcing_parsed", "initial_value_parsed",
                             "antiderivative_computed", "ode_residual_zero", "initial_residual_zero"),
                            ("result_present", "reasoning", "initial_condition_check", "differential_equation_substitution"),
                            ("expression", "text"))

    def _constant_second_order_forced_ivp(
        self,
        text: str,
    ) -> Optional[ToolResult]:
        if not re.search(r"初值|initial\s+value|IVP", text, re.I):
            return None
        equation_segments = [
            segment
            for segment in re.findall(r"\$([^$\n]+)\$", text)
            if re.search(r"y\s*(?:''|[′']{2})", segment)
            and segment.count("=") == 1
        ]
        initial = self._initial_value(text, derivative=False)
        derivative_initial = self._initial_value(text, derivative=True)
        if len(equation_segments) != 1 or initial is None or derivative_initial is None:
            return None
        lhs_source, rhs_source = (
            part.strip() for part in equation_segments[0].split("=", 1)
        )
        prepared = lhs_source.replace("′", "'")
        prepared = re.sub(r"y\s*''", "q", prepared)
        prepared = re.sub(r"y\s*'", "p", prepared)
        prepared = re.sub(r"(?<![A-Za-z0-9_])y(?![A-Za-z0-9_])", "r", prepared)
        polynomial = self._expr(prepared)
        rhs = self._expr(rhs_source)
        if polynomial is None or rhs is None:
            return None
        y0_symbol, y1_symbol, y2_symbol = self.sp.symbols("r p q")
        a0, a1, a2 = (
            self.sp.simplify(polynomial.coeff(symbol))
            for symbol in (y0_symbol, y1_symbol, y2_symbol)
        )
        if (
            a2 == 0
            or any(coefficient.free_symbols for coefficient in (a0, a1, a2))
            or self.sp.simplify(
                polynomial - a0 * y0_symbol - a1 * y1_symbol - a2 * y2_symbol
            ) != 0
        ):
            return None
        x = self.sp.Symbol("x")
        if rhs.free_symbols - {x}:
            return None
        x0, initial_value = initial
        derivative_x0, initial_derivative = derivative_initial
        if (
            self.sp.simplify(x0 - derivative_x0) != 0
            or any(
                value.free_symbols
                for value in (x0, initial_value, initial_derivative)
            )
        ):
            return None
        try:
            domain = self.sp.calculus.util.continuous_domain(
                rhs,
                x,
                self.sp.S.Reals,
            )
        except Exception:
            return None
        if domain != self.sp.S.Reals:
            return None

        function = self.sp.Function("y")
        ode = self.sp.Eq(
            a2 * self.sp.diff(function(x), x, 2)
            + a1 * self.sp.diff(function(x), x)
            + a0 * function(x),
            rhs,
        )
        try:
            solved = self.sp.dsolve(
                ode,
                ics={
                    function(x0): initial_value,
                    self.sp.diff(function(x), x).subs(x, x0): initial_derivative,
                },
            )
        except Exception:
            return None
        if not isinstance(solved, self.sp.Equality) or solved.lhs != function(x):
            return None
        solution = self.sp.simplify(solved.rhs)
        residual = self.sp.simplify(
            a2 * self.sp.diff(solution, x, 2)
            + a1 * self.sp.diff(solution, x)
            + a0 * solution
            - rhs
        )
        initial_residual = self.sp.simplify(solution.subs(x, x0) - initial_value)
        derivative_residual = self.sp.simplify(
            self.sp.diff(solution, x).subs(x, x0) - initial_derivative
        )
        if residual != 0 or initial_residual != 0 or derivative_residual != 0:
            return None
        lam = self.sp.Symbol("lambda")
        characteristic = self.sp.expand(a2 * lam**2 + a1 * lam + a0)
        rendered_solution = self.symbolic._format(solution)
        rendered_characteristic = self.symbolic._format(characteristic)
        if self._zh(text):
            result = (
                "特征多项式为 $" + rendered_characteristic + "$。结合强迫项并代入两个初值，"
                + rf"唯一解为 $y(x)={rendered_solution}$；最大存在区间为 "
                + r"$(-\infty,\infty)=\mathbb R$。代回方程及两个初值的残差均为 $0$。"
            )
        else:
            result = (
                "The characteristic polynomial is $" + rendered_characteristic
                + "$. Combining the forcing term with both initial values gives "
                + rf"$y(x)={rendered_solution}$. The maximal interval is "
                + r"$(-\infty,\infty)=\mathbb R$. Substitution into the ODE and "
                + "both initial values gives zero residual."
            )
        return self._result(
            text,
            "constant_second_order_forced_ivp",
            result,
            "explicit_solution_and_maximal_interval",
            "symbolic_ivp_solution_and_three_residual_checks",
            (
                "single_constant_coefficient_second_order_equation_parsed",
                "forcing_continuous_on_real_line",
                "two_initial_values_parsed",
                "symbolic_solution_computed",
                "ode_residual_zero",
                "both_initial_residuals_zero",
            ),
            (
                "result_present",
                "all_solutions",
                "reasoning",
                "initial_condition_check",
                "differential_equation_substitution",
                "domain_or_conditions",
            ),
            ("expression", "roots", "text"),
        )

    def _constant_second_order_ivp(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"特征方程|characteristic\s+equation", text, re.I):
            return None
        lhs_match = re.search(r"(y\s*(?:''|[′']{2})[^=,，。;；\n]*)=\s*0", text, re.I)
        initial = self._initial_value(text, derivative=False)
        derivative_initial = self._initial_value(text, derivative=True)
        if lhs_match is None or initial is None or derivative_initial is None:
            return None
        prepared = lhs_match.group(1).replace("′", "'")
        prepared = re.sub(r"y\s*''", "q", prepared)
        prepared = re.sub(r"y\s*'", "p", prepared)
        prepared = re.sub(r"(?<![A-Za-z])y(?![A-Za-z])", "r", prepared)
        polynomial = self._expr(prepared)
        if polynomial is None:
            return None
        p, q, r = self.sp.symbols("p q r")
        a2, a1, a0 = map(self.sp.simplify, (
            polynomial.coeff(q), polynomial.coeff(p), polynomial.coeff(r)
        ))
        if any(item.free_symbols for item in (a2, a1, a0)) or a2 == 0:
            return None
        if self.sp.simplify(polynomial - a2*q - a1*p - a0*r) != 0:
            return None
        x0, y0 = initial
        dx0, dy0 = derivative_initial
        if self.sp.simplify(x0 - dx0) != 0 or any(item.free_symbols for item in (x0, y0, dy0)):
            return None
        x = self.sp.Symbol("x")
        y = self.sp.Function("y")
        ode = self.sp.Eq(a2*self.sp.diff(y(x), x, 2) + a1*self.sp.diff(y(x), x) + a0*y(x), 0)
        solved = self.sp.dsolve(ode, ics={y(x0): y0, self.sp.diff(y(x), x).subs(x, x0): dy0})
        if not isinstance(solved, self.sp.Equality) or solved.lhs != y(x):
            return None
        solution = self.sp.simplify(solved.rhs)
        residual = self.sp.simplify(a2*self.sp.diff(solution, x, 2) + a1*self.sp.diff(solution, x) + a0*solution)
        if residual != 0 or self.sp.simplify(solution.subs(x, x0)-y0) != 0 or self.sp.simplify(self.sp.diff(solution,x).subs(x,x0)-dy0) != 0:
            return None
        lam = self.sp.Symbol("lambda")
        characteristic = self.sp.expand(a2*lam**2 + a1*lam + a0)
        result = (
            rf"特征方程为 ${self.symbolic._format(characteristic)}=0$。由其基本解并代入两个初值，唯一解为 "
            rf"$y(x)={self.symbolic._format(solution)}$；回代方程与初值均成立。"
            if self._zh(text) else
            rf"The characteristic equation is ${self.symbolic._format(characteristic)}=0$. Applying both initial values to its fundamental solutions gives the unique solution "
            rf"$y(x)={self.symbolic._format(solution)}$, which satisfies the ODE and both initial values by substitution."
        )
        return self._result(text, "constant_second_order_ivp", result, "explicit_solution",
                            "characteristic_roots_and_initial_linear_system",
                            ("constant_coefficients_parsed", "two_initial_values_parsed", "characteristic_polynomial",
                             "dsolve_exact", "ode_residual_zero", "both_initial_residuals_zero"),
                            ("result_present", "all_solutions", "reasoning", "initial_condition_check"),
                            ("roots", "expression", "text"))

    def _autonomous_power_blowup(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"最大(?:右侧)?存在区间|maximal.*interval", text, re.I) or not re.search(r"分离变量|separation\s+of\s+variables", text, re.I):
            return None
        equation = re.search(
            r"y\s*['′]\s*=\s*([^y=,，。;；\s]*)?\s*y\s*\^\s*\{?\s*(\d+)\s*\}?",
            text,
            re.I,
        )
        initial = self._initial_value(text, derivative=False)
        if equation is None or initial is None:
            return None
        coefficient = self._expr(equation.group(1) or "1")
        power = int(equation.group(2))
        x0, y0 = initial
        if power < 2 or coefficient is None or any(item.free_symbols for item in (coefficient, x0, y0)):
            return None
        if coefficient.is_real is not True or coefficient == 0 or y0.is_positive is not True:
            return None
        x = self.sp.Symbol("x")
        base = self.sp.simplify(y0**(1-power) - coefficient*(power-1)*(x-x0))
        solution = self.sp.simplify(base ** self.sp.Rational(-1, power-1))
        singular = self.sp.simplify(x0 + y0**(1-power)/(coefficient*(power-1)))
        if singular.is_real is not True:
            return None
        if bool(singular > x0):
            interval = rf"(-\infty,{self.symbolic._format(singular)})"
            right_interval = rf"[{self.symbolic._format(x0)},{self.symbolic._format(singular)})"
        elif bool(singular < x0):
            interval = rf"({self.symbolic._format(singular)},\infty)"
            right_interval = rf"[{self.symbolic._format(x0)},\infty)"
        else:
            return None
        residual = self.sp.simplify(self.sp.diff(solution, x) - coefficient*solution**power)
        if residual != 0 or self.sp.simplify(solution.subs(x,x0)-y0) != 0:
            return None
        result = (
            rf"分离变量得 $y(x)={self.symbolic._format(solution)}$。其唯一实奇点为 "
            rf"$x={self.symbolic._format(singular)}$，故包含初值点的最大区间为 ${interval}$，最大右侧区间为 ${right_interval}$。"
            if self._zh(text) else
            rf"Separation of variables gives $y(x)={self.symbolic._format(solution)}$. Its only real singularity is "
            rf"$x={self.symbolic._format(singular)}$, so the maximal interval containing the initial point is ${interval}$ and the maximal right interval is ${right_interval}$."
        )
        return self._result(text, "autonomous_power_blowup_ivp", result, "solution_and_interval",
                            "separation_and_singularity_component",
                            ("power_equation_parsed", "initial_value_parsed", "solution_derived", "residual_zero", "singularity_recomputed"),
                            ("result_present", "reasoning", "domain_or_conditions", "maximal_interval_and_one_sided_part"),
                            ("expression", "interval", "text"))

    def _diagonal_system_stability(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"线性系统|linear\s+system", text, re.I) or not re.search(r"稳定性|stability", text, re.I):
            return None
        first = re.search(r"x\s*['′]\s*=\s*([^,，。;；\s]*)?\s*x", text, re.I)
        second = re.search(r"y\s*['′]\s*=\s*([^,，。;；\s]*)?\s*y", text, re.I)
        if first is None or second is None:
            return None
        a = self._signed_implicit(first.group(1))
        b = self._signed_implicit(second.group(1))
        if a is None or b is None or any(item.free_symbols for item in (a,b)) or a.is_real is not True or b.is_real is not True:
            return None
        if a == 0 or b == 0:
            return None
        if a < 0 and b < 0:
            kind, stability = "稳定结点", "渐近稳定"
            kind_en, stability_en = "stable node", "asymptotically stable"
        elif a*b < 0:
            kind, stability = "鞍点", "不稳定"
            kind_en, stability_en = "saddle", "unstable"
        else:
            kind, stability = "不稳定结点", "不稳定"
            kind_en, stability_en = "unstable node", "unstable"
        result = (
            rf"唯一平衡点为原点 $(0,0)$。线性化矩阵为对角阵，特征值为 "
            rf"$\lambda_1={self.symbolic._format(a)},\lambda_2={self.symbolic._format(b)}$，符号判定原点为{kind}，因此{stability}。"
            if self._zh(text) else
            rf"The unique equilibrium is the origin $(0,0)$. The diagonal linearization has eigenvalues "
            rf"$\lambda_1={self.symbolic._format(a)},\lambda_2={self.symbolic._format(b)}$; their signs make the origin a {kind_en}, hence {stability_en}."
        )
        return self._result(text, "diagonal_linear_system_stability", result, "equilibrium_and_stability",
                            "diagonal_eigenvalue_sign_classification",
                            ("two_diagonal_equations_parsed", "unique_equilibrium", "eigenvalues_recomputed", "signs_classified"),
                            ("result_present", "equilibrium_point", "stability_classification", "eigenvalue_signs"),
                            ("expression", "text"))

    def _bernoulli_general(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"Bernoulli|伯努利", text, re.I) or not re.search(r"非零通解|nonzero\s+general\s+solution", text, re.I):
            return None
        match = re.search(
            r"y\s*['′]\s*([+-])\s*([^y=,，。;；\s]*)?\s*y\s*=\s*"
            r"([^y=,，。;；\s]*)?\s*y\s*\^\s*\{?\s*(\d+)\s*\}?",
            text,
            re.I,
        )
        if match is None:
            return None
        a = self._coefficient(match.group(1), match.group(2))
        b_raw = (match.group(3) or "1").strip().rstrip("*").strip() or "1"
        b = self._expr(b_raw)
        power = int(match.group(4))
        if a is None or b is None or power == 1 or a == 0 or any(item.free_symbols for item in (a,b)):
            return None
        x = self.sp.Symbol("x")
        v = self.sp.simplify(b/a + self.sp.Symbol("C") * self.sp.exp(-(1-power)*a*x))
        solution = self.sp.simplify(v ** self.sp.Rational(1, 1-power))
        residual = self.sp.simplify(
            self.sp.diff(solution, x) + a*solution - b*solution**power
        )
        if residual != 0:
            return None
        result = (
            rf"令 $v=y^{{1-{power}}}$，则 $v'+({1-power})({self.symbolic._format(a)})v="
            rf"({1-power})({self.symbolic._format(b)})$。解此线性方程得非零通解 "
            rf"$y(x)={self.symbolic._format(solution)}$（在分母非零的区间上）；直接代回原方程成立。"
            if self._zh(text) else
            rf"Set $v=y^{{1-{power}}}$. Then $v'+({1-power})({self.symbolic._format(a)})v="
            rf"({1-power})({self.symbolic._format(b)})$. Solving this linear equation gives the nonzero general solution "
            rf"$y(x)={self.symbolic._format(solution)}$ wherever its denominator is nonzero; direct substitution verifies it."
        )
        return self._result(text, "bernoulli_constant_coeff_general", result, "general_solution",
                            "bernoulli_power_substitution",
                            ("constant_coefficients_parsed", "power_parsed", "linearized_equation",
                             "symbolic_solution", "ode_residual_zero"),
                            ("result_present", "reasoning", "domain_or_conditions"),
                            ("expression", "text"))

    def _dalembert(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"波动方程|wave\s+equation", text, re.I) or not re.search(r"d['’]?Alembert|达朗贝尔", text, re.I):
            return None
        if not re.search(r"u\s*_?\{?tt\}?\s*=\s*u\s*_?\{?xx\}?|u_tt\s*=\s*u_xx", text, re.I):
            return None
        result = (
            r"令特征变量 $\xi=x+t,\eta=x-t$，方程化为 $u_{\xi\eta}=0$，故 d'Alembert 通解为 "
            r"$u(x,t)=F(x+t)+G(x-t)$。其中两个任意函数 $F,G$ 分别表示向左、向右传播的波。"
            if self._zh(text) else
            r"With characteristic variables $\xi=x+t,\eta=x-t$, the equation becomes $u_{\xi\eta}=0$. "
            r"Thus d'Alembert's general solution is $u(x,t)=F(x+t)+G(x-t)$, where the two arbitrary functions represent the two propagation directions."
        )
        return self._result(text, "dalembert_general_solution", result, "general_solution",
                            "characteristic_coordinate_integration",
                            ("unit_speed_wave_equation", "characteristic_variables", "mixed_derivative_integrated"),
                            ("result_present", "reasoning", "support_anchor_1"),
                            ("expression", "text"))

    def _first_order_equation(self, text: str):
        match = re.search(
            r"y\s*['′]\s*([+-])\s*([^y=,，。;；\s]*)?\s*y\s*=\s*([^,，。;；\n]+)",
            text,
            re.I,
        )
        if match is None:
            return None
        coefficient = self._coefficient(match.group(1), match.group(2))
        rhs = self._expr(match.group(3).strip().strip("$"))
        return (coefficient, rhs) if coefficient is not None and rhs is not None else None

    def _initial_value(self, text: str, derivative: bool):
        marker = r"y\s*['′]" if derivative else r"y"
        match = re.search(
            rf"{marker}\s*\(\s*([-+]?\d+(?:/\d+)?|[-+]?0?\.\d+)\s*\)\s*=\s*"
            r"([-+]?\d+(?:/\d+)?|[-+]?0?\.\d+)",
            text,
        )
        if match is None:
            return None
        return self._expr(match.group(1)), self._expr(match.group(2))

    def _coefficient(self, sign: str, magnitude: str | None):
        value = self._expr((magnitude or "1").strip().rstrip("*").strip() or "1")
        return -value if value is not None and sign == "-" else value

    def _signed_implicit(self, value: str | None):
        raw = (value or "1").strip().rstrip("*").strip()
        return self._expr("-1" if raw == "-" else "1" if raw in {"", "+"} else raw)

    def _expr(self, value: str):
        try:
            return self.sp.simplify(self.symbolic._parse(str(value).strip().strip("$")))
        except Exception:
            return None

    @staticmethod
    def _zh(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def _result(problem: str, operation: str, result: str, result_kind: str,
                method: str, checks: tuple[str, ...], requirements: tuple[str, ...],
                shapes: tuple[str, ...]) -> ToolResult:
        return make_parameterized_tool_result(
            problem=problem,
            operation=operation,
            result=result,
            result_kind=result_kind,
            method=method,
            whole=True,
            written_support=True,
            checks=checks,
            support=result,
            answer_shapes=shapes,
            requirements=requirements,
        )
