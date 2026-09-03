"""Certified parameterized solvers for explicitly requested numerical methods.

Every value is parsed from the current statement.  The tool deliberately
abstains unless the method, data, and requested step are unambiguous; it never
uses a stored problem or answer and never executes model-generated code.
"""

from __future__ import annotations

import math
import re
from typing import Optional

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, make_parameterized_tool_result


class NumericalMethodTool:
    """Run a small whitelist of textbook numerical algorithms exactly."""

    _SCALAR = r"([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*\d+)?)"

    def __init__(self) -> None:
        self.symbolic = SympyTool()
        self.sp = self.symbolic.sympy

    def results_for(self, problem: str) -> list[ToolResult]:
        if not self.sp:
            return []
        text = str(problem or "").strip()
        results: list[ToolResult] = []
        for compiler in (
            self._one_step_bisection,
            self._bisection_approximation,
            self._newton_first_iteration,
            self._newton_approximation,
            self._secant_first_iteration,
            self._secant_approximation,
            self._polynomial_interpolation,
            self._gauss_legendre_quadrature,
            self._jacobi_exact_iterations,
            self._composite_trapezoid,
            self._composite_simpson,
            self._forward_euler,
            self._improved_euler,
            self._explicit_runge_kutta_stability,
            self._runge_kutta_4,
            self._taylor_polynomial,
        ):
            try:
                result = compiler(text)
            except Exception:
                result = None
            if result is not None and result.verified:
                results.append(result)
        return results

    def _one_step_bisection(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"二分法|bisection", text, re.IGNORECASE):
            return None
        if not re.search(
            r"一次|第一步|首次|one\s+(?:bisection\s+)?(?:step|iteration)|first\s+(?:step|iteration)",
            text,
            re.IGNORECASE,
        ):
            return None
        expression = self._function_expression(text, "x")
        interval = self._interval(text)
        if expression is None or interval is None:
            return None
        x = self.sp.Symbol("x")
        lower, upper = interval
        if expression.free_symbols - {x} or not self._strictly_less(lower, upper):
            return None
        midpoint = self.sp.simplify((lower + upper) / 2)
        values = tuple(self.sp.simplify(expression.subs(x, p)) for p in (lower, midpoint, upper))
        if any(v.has(self.sp.nan, self.sp.zoo) or v.is_real is not True for v in values):
            return None
        signs = tuple(self.sp.sign(v) for v in values)
        if signs[0] * signs[2] > 0:
            return None
        if signs[0] == 0:
            new_interval = (lower, lower)
        elif signs[1] == 0:
            new_interval = (midpoint, midpoint)
        elif signs[0] * signs[1] == -1:
            new_interval = (lower, midpoint)
        elif signs[1] * signs[2] == -1:
            new_interval = (midpoint, upper)
        else:
            return None
        a, b = (self.sp.latex(item) for item in new_interval)
        midpoint_text = self.sp.latex(midpoint)
        zh = self._is_chinese(text)
        result = (
            rf"中点 $m=(a+b)/2={midpoint_text}$，一次二分后的区间为 $[{a},{b}]$。"
            if zh else
            rf"The midpoint is $m=(a+b)/2={midpoint_text}$, and after one bisection the interval is $[{a},{b}]$."
        )
        support = (
            rf"端点函数值及中点函数值的符号为 {tuple(str(s) for s in signs)}，故保留函数值异号的一半。"
            if zh else
            rf"The signs at the two endpoints and midpoint are {tuple(str(s) for s in signs)}, so the sign-changing half is retained."
        )
        return self._result(
            text, "one_step_bisection", result, "interval",
            "exact_midpoint_sign_bracket",
            ("method_explicit", "function_parsed", "initial_interval_parsed", "all_three_signs_exact"),
            support,
            ("interval", "expression", "number"),
            ("result_present", "numeric_result", "method_formula", "first_iteration"),
        )

    def _bisection_approximation(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"二分法|bisection", text, re.IGNORECASE):
            return None
        tolerance = self._tolerance(text)
        expression = self._function_expression(text, "x")
        interval = self._interval(text)
        if tolerance is None or expression is None or interval is None:
            return None
        if not self._strictly_less(0, tolerance):
            return None
        x = self.sp.Symbol("x")
        lower, upper = interval
        if expression.free_symbols - {x} or not self._strictly_less(lower, upper):
            return None
        if not self._continuous_on_closed_interval(expression, x, lower, upper):
            return None
        f_lower = self.sp.simplify(expression.subs(x, lower))
        f_upper = self.sp.simplify(expression.subs(x, upper))
        lower_sign, upper_sign = self._numeric_sign(f_lower), self._numeric_sign(f_upper)
        if lower_sign is None or upper_sign is None or lower_sign * upper_sign > 0:
            return None
        iterations = 0
        if lower_sign == 0:
            midpoint = lower
            upper = lower
        elif upper_sign == 0:
            midpoint = upper
            lower = upper
        else:
            midpoint = self.sp.simplify((lower + upper) / 2)
            while self._strictly_less(tolerance, self.sp.Abs(upper - lower) / 2):
                if iterations >= 10_000:
                    return None
                midpoint = self.sp.simplify((lower + upper) / 2)
                middle_sign = self._numeric_sign(expression.subs(x, midpoint))
                if middle_sign is None:
                    return None
                iterations += 1
                if middle_sign == 0:
                    lower = upper = midpoint
                    break
                if lower_sign * middle_sign < 0:
                    upper = midpoint
                else:
                    lower, lower_sign = midpoint, middle_sign
            midpoint = self.sp.simplify((lower + upper) / 2)
        digits = self._display_digits(tolerance)
        approximation = self._decimal(midpoint, digits)
        error_bound = self.sp.simplify(self.sp.Abs(upper - lower) / 2)
        zh = self._is_chinese(text)
        result = (
            rf"二分法得到根 $x\approx {approximation}$；最终区间为 $[{self.sp.latex(lower)},{self.sp.latex(upper)}]$，误差上界为 ${self.sp.latex(error_bound)}\le {self.sp.latex(tolerance)}$。"
            if zh else
            rf"Bisection gives $x\approx {approximation}$; the final bracket is $[{self.sp.latex(lower)},{self.sp.latex(upper)}]$, with error bound ${self.sp.latex(error_bound)}\le {self.sp.latex(tolerance)}$."
        )
        support = (
            f"初始端点异号；每步保留异号子区间，共执行 {iterations} 次并用半区间长度作误差证书。"
            if zh else
            f"The initial endpoint values have opposite signs; {iterations} sign-preserving halvings were performed and the half-width certifies the error."
        )
        return self._result(
            text, "bisection_approximation", result, "certified_approximation",
            "sign_preserving_bisection_with_interval_error_bound",
            ("method_explicit", "function_interval_tolerance_parsed", "initial_sign_change", "bounded_iterations", "final_half_width_within_tolerance"),
            support,
            ("number", "interval", "expression"),
            ("result_present", "numeric_result", "method_formula"),
        )

    def _newton_first_iteration(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"牛顿法|Newton(?:'s)?\s+(?:method|iteration)", text, re.IGNORECASE):
            return None
        if not re.search(r"x\s*_?\s*\{?\s*1\s*\}?|第一次|第一步|first\s+(?:iterate|iteration|step)", text, re.IGNORECASE):
            return None
        x0 = self._indexed_value(text, "x", 0)
        expression = self._function_expression(text, "x")
        if x0 is None or expression is None:
            return None
        x = self.sp.Symbol("x")
        if expression.free_symbols - {x}:
            return None
        derivative = self.sp.diff(expression, x)
        denominator = self.sp.simplify(derivative.subs(x, x0))
        if denominator == 0 or denominator.has(self.sp.nan, self.sp.zoo):
            return None
        x1 = self.sp.simplify(x0 - expression.subs(x, x0) / denominator)
        if x1.has(self.sp.nan, self.sp.zoo):
            return None
        xn = self.sp.Symbol("x_n")
        iteration = self.sp.simplify(xn - expression.subs(x, xn) / derivative.subs(x, xn))
        formula = rf"x_{{n+1}}={self.sp.latex(iteration)}"
        zh = self._is_chinese(text)
        result = (
            rf"Newton 迭代公式为 ${formula}$；代入 $x_0={self.sp.latex(x0)}$ 得 $x_1={self.sp.latex(x1)}$。"
            if zh else
            rf"Newton's iteration is ${formula}$; with $x_0={self.sp.latex(x0)}$, $x_1={self.sp.latex(x1)}$."
        )
        support = (
            r"使用 $x_{n+1}=x_n-f(x_n)/f'(x_n)$，并已精确回代检查分母非零。"
            if zh else
            r"This uses $x_{n+1}=x_n-f(x_n)/f'(x_n)$; exact substitution verifies the denominator is nonzero."
        )
        return self._result(
            text, "newton_iteration", result, "formula_and_iterate",
            "symbolic_newton_substitution",
            ("method_explicit", "equation_parsed", "derivative_computed", "denominator_nonzero", "first_iterate_recomputed"),
            support,
            ("expression", "number"),
            ("result_present", "numeric_result", "method_formula", "first_iteration"),
        )

    def _newton_approximation(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"牛顿法|Newton(?:'s)?\s+(?:method|iteration)", text, re.IGNORECASE):
            return None
        tolerance = self._tolerance(text)
        x0 = self._indexed_value(text, "x", 0)
        expression = self._function_expression(text, "x")
        if tolerance is None or x0 is None or expression is None or not self._strictly_less(0, tolerance):
            return None
        x = self.sp.Symbol("x")
        if expression.free_symbols - {x}:
            return None
        derivative = self.sp.diff(expression, x)
        current = self.sp.N(x0, 60)
        iterations = 0
        converged = False
        for iterations in range(1, 101):
            denominator = self.sp.N(derivative.subs(x, current), 60)
            if self._near_zero(denominator, tolerance / 100):
                return None
            next_value = self.sp.N(current - expression.subs(x, current) / denominator, 60)
            if not self._finite(next_value):
                return None
            if self._numeric_leq(self.sp.Abs(next_value - current), tolerance):
                current, converged = next_value, True
                break
            if self._numeric_abs(next_value) > 10**50:
                return None
            current = next_value
        if not converged:
            return None
        residual = self.sp.N(self.sp.Abs(expression.subs(x, current)), 40)
        if not self._credible_root_residual(residual, tolerance, current):
            return None
        digits = self._display_digits(tolerance)
        zh = self._is_chinese(text)
        result = (
            rf"Newton 法经 {iterations} 次迭代得到 $x\approx {self._decimal(current, digits)}$，末步改变量不超过 ${self.sp.latex(tolerance)}$。"
            if zh else
            rf"After {iterations} Newton iterations, $x\approx {self._decimal(current, digits)}$; the final update is at most ${self.sp.latex(tolerance)}$."
        )
        support = (
            rf"逐步使用 $x_{{n+1}}=x_n-f(x_n)/f'(x_n)$ 并检查每个分母；末点残差约为 ${self._decimal(residual, digits)}$。"
            if zh else
            rf"Each denominator in $x_{{n+1}}=x_n-f(x_n)/f'(x_n)$ was checked; the final residual is approximately ${self._decimal(residual, digits)}$."
        )
        return self._result(
            text, "newton_approximation", result, "numeric_approximation",
            "bounded_high_precision_newton_iteration",
            ("method_explicit", "function_initial_tolerance_parsed", "all_derivatives_nonzero", "bounded_iterations", "final_update_within_tolerance", "residual_recomputed"),
            support,
            ("number", "expression"),
            ("result_present", "numeric_result", "method_formula"),
        )

    def _secant_first_iteration(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"割线法|弦截法|secant\s+(?:method|iteration)", text, re.IGNORECASE):
            return None
        if not re.search(r"x\s*_?\s*\{?\s*2\s*\}?|下一次|第一步|first\s+(?:iterate|iteration|step)", text, re.IGNORECASE):
            return None
        x0 = self._indexed_value(text, "x", 0)
        x1 = self._indexed_value(text, "x", 1)
        expression = self._function_expression(text, "x")
        if x0 is None or x1 is None or expression is None:
            return None
        x = self.sp.Symbol("x")
        if expression.free_symbols - {x}:
            return None
        f0 = self.sp.simplify(expression.subs(x, x0))
        f1 = self.sp.simplify(expression.subs(x, x1))
        denominator = self.sp.simplify(f1 - f0)
        if denominator == 0 or denominator.has(self.sp.nan, self.sp.zoo):
            return None
        x2 = self.sp.simplify(x1 - f1 * (x1 - x0) / denominator)
        zh = self._is_chinese(text)
        formula = r"x_{n+1}=x_n-f(x_n)\frac{x_n-x_{n-1}}{f(x_n)-f(x_{n-1})}"
        result = (
            rf"割线迭代公式为 ${formula}$；代入初值可得 $x_2={self.sp.latex(x2)}$。"
            if zh else
            rf"The secant iteration is ${formula}$; substituting the initial values gives $x_2={self.sp.latex(x2)}$."
        )
        support = (
            r"已精确计算两个函数值并验证它们之差非零，再代入割线公式。"
            if zh else
            r"Both function values were computed exactly, their difference was checked nonzero, and the secant formula was applied."
        )
        return self._result(
            text, "secant_iteration", result, "formula_and_iterate",
            "exact_two_point_secant_substitution",
            ("method_explicit", "equation_parsed", "two_initial_values", "denominator_nonzero", "next_iterate_recomputed"),
            support,
            ("expression", "number"),
            ("result_present", "numeric_result", "method_formula", "first_iteration"),
        )

    def _secant_approximation(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"割线法|弦截法|secant\s+(?:method|iteration)", text, re.IGNORECASE):
            return None
        tolerance = self._tolerance(text)
        previous = self._indexed_value(text, "x", 0)
        current = self._indexed_value(text, "x", 1)
        expression = self._function_expression(text, "x")
        if any(item is None for item in (tolerance, previous, current, expression)):
            return None
        if not self._strictly_less(0, tolerance):
            return None
        x = self.sp.Symbol("x")
        if expression.free_symbols - {x}:
            return None
        previous, current = self.sp.N(previous, 60), self.sp.N(current, 60)
        converged = False
        iterations = 0
        for iterations in range(1, 101):
            f_previous = self.sp.N(expression.subs(x, previous), 60)
            f_current = self.sp.N(expression.subs(x, current), 60)
            denominator = self.sp.N(f_current - f_previous, 60)
            if self._near_zero(denominator, tolerance / 100):
                return None
            next_value = self.sp.N(
                current - f_current * (current - previous) / denominator, 60
            )
            if not self._finite(next_value):
                return None
            if self._numeric_leq(self.sp.Abs(next_value - current), tolerance):
                current, converged = next_value, True
                break
            if self._numeric_abs(next_value) > 10**50:
                return None
            previous, current = current, next_value
        if not converged:
            return None
        residual = self.sp.N(self.sp.Abs(expression.subs(x, current)), 40)
        if not self._credible_root_residual(residual, tolerance, current):
            return None
        digits = self._display_digits(tolerance)
        zh = self._is_chinese(text)
        result = (
            rf"割线法经 {iterations} 次更新得到 $x\approx {self._decimal(current, digits)}$，末步改变量不超过 ${self.sp.latex(tolerance)}$。"
            if zh else
            rf"After {iterations} secant updates, $x\approx {self._decimal(current, digits)}$; the final update is at most ${self.sp.latex(tolerance)}$."
        )
        support = (
            rf"每步均检查 $f(x_n)-f(x_{{n-1}})\ne0$；末点残差约为 ${self._decimal(residual, digits)}$。"
            if zh else
            rf"Every denominator $f(x_n)-f(x_{{n-1}})$ was checked nonzero; the final residual is approximately ${self._decimal(residual, digits)}$."
        )
        return self._result(
            text, "secant_approximation", result, "numeric_approximation",
            "bounded_high_precision_secant_iteration",
            ("method_explicit", "function_initial_values_tolerance_parsed", "all_denominators_nonzero", "bounded_iterations", "final_update_within_tolerance", "residual_recomputed"),
            support,
            ("number", "expression"),
            ("result_present", "numeric_result", "method_formula"),
        )

    def _polynomial_interpolation(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"插值|interpolat", text, re.IGNORECASE):
            return None
        points = []
        for left, right in re.findall(
            r"\(\s*([^,，()]+)\s*[,，]\s*([^()]+?)\s*\)", text
        ):
            x_value, y_value = self._math_expr(left), self._math_expr(right)
            if (
                x_value is None or y_value is None
                or x_value.free_symbols or y_value.free_symbols
            ):
                continue
            points.append((x_value, y_value))
        if not 2 <= len(points) <= 20 or len({item[0] for item in points}) != len(points):
            return None
        x = self.sp.Symbol("x")
        polynomial = self.sp.expand(self.sp.interpolate(points, x))
        if self.sp.degree(polynomial, x) > len(points) - 1:
            return None
        if any(
            self.sp.simplify(polynomial.subs(x, xv) - yv) != 0
            for xv, yv in points
        ):
            return None
        target = self._interpolation_target(text)
        zh = self._is_chinese(text)
        if target is not None:
            value = self.sp.simplify(polynomial.subs(x, target))
            result = (
                rf"插值多项式为 $p(x)={self.sp.latex(polynomial)}$，故 $p({self.sp.latex(target)})={self.sp.latex(value)}$。"
                if zh else
                rf"The interpolation polynomial is $p(x)={self.sp.latex(polynomial)}$, hence $p({self.sp.latex(target)})={self.sp.latex(value)}$."
            )
            result_kind = "polynomial_and_value"
            requirements = ("result_present", "numeric_result", "method_formula")
        else:
            if not re.search(r"多项式|polynomial|公式|formula", text, re.IGNORECASE):
                return None
            result = rf"p(x)={self.sp.latex(polynomial)}"
            result_kind = "polynomial"
            requirements = ("result_present", "method_formula")
        support = (
            "由 Lagrange/Newton 插值的唯一性构造，并已把所得多项式逐点代回全部数据。"
            if zh else
            "The unique Lagrange/Newton interpolant was constructed and substituted back at every supplied node."
        )
        return self._result(
            text, "polynomial_interpolation", result, result_kind,
            "symbolic_interpolation_with_all_node_checks",
            ("method_explicit", "distinct_nodes", "degree_bound", "all_nodes_recovered"),
            support,
            ("expression", "number"),
            requirements,
        )

    def _gauss_legendre_quadrature(self, text: str) -> Optional[ToolResult]:
        if not re.search(
            r"Gauss\s*[- ]?\s*Legendre|高斯\s*[-—]?\s*勒让德",
            text,
            re.IGNORECASE,
        ):
            return None
        point_match = re.search(
            r"(?P<count>\d+|一|二|两|三|四|五)\s*点"
            r"[^。；;\n]{0,30}?(?:Gauss|高斯)|"
            r"(?P<en_count>\d+)\s*[- ]point\s+Gauss|"
            r"(?:Gauss\s*[- ]?\s*Legendre|高斯\s*[-—]?\s*勒让德)"
            r"[^。；;\n]{0,30}?(?:n\s*=\s*)?(?P<tail_count>\d+)\s*"
            r"(?:点|points?)",
            text,
            re.IGNORECASE,
        )
        if point_match is None:
            return None
        token = (
            point_match.group("count")
            or point_match.group("en_count")
            or point_match.group("tail_count")
        )
        point_count = self._small_integer(token)
        integral = self._definite_integral(text)
        if point_count is None or integral is None or not 1 <= point_count <= 5:
            return None
        expression, variable, lower, upper = integral
        if (
            expression.free_symbols - {variable}
            or lower.free_symbols
            or upper.free_symbols
            or not self._strictly_less(lower, upper)
        ):
            return None
        try:
            self.sp.Poly(expression, variable)
        except Exception:
            return None

        standard = self.sp.Symbol("_gauss_x", real=True)
        polynomial = self.sp.legendre(point_count, standard)
        roots = tuple(self.sp.solve(polynomial, standard))
        if len(roots) != point_count:
            return None
        roots = tuple(sorted(
            (self.sp.simplify(root) for root in roots),
            key=lambda item: float(self.sp.N(item, 40)),
        ))
        if any(
            root.is_real is not True
            or not self._strictly_less(-1, root)
            or not self._strictly_less(root, 1)
            for root in roots
        ):
            return None
        derivative = self.sp.diff(polynomial, standard)
        standard_weights = tuple(
            self.sp.simplify(
                2 / ((1 - root**2) * derivative.subs(standard, root) ** 2)
            )
            for root in roots
        )
        midpoint = self.sp.simplify((lower + upper) / 2)
        half_width = self.sp.simplify((upper - lower) / 2)
        nodes = tuple(
            self.sp.simplify(midpoint + half_width * root) for root in roots
        )
        weights = tuple(
            self.sp.simplify(half_width * weight)
            for weight in standard_weights
        )
        if any(
            weight.is_positive is not True
            or not self._finite(node)
            or not self._finite(weight)
            for node, weight in zip(nodes, weights)
        ):
            return None

        # Moment exactness is recomputed independently of the target integral.
        for degree in range(2 * point_count):
            quadrature_moment = self.sp.simplify(sum(
                weight * node**degree
                for node, weight in zip(nodes, weights)
            ))
            exact_moment = self.sp.simplify(
                (upper ** (degree + 1) - lower ** (degree + 1)) / (degree + 1)
            )
            if self.sp.simplify(quadrature_moment - exact_moment) != 0:
                return None

        quadrature = self.sp.simplify(sum(
            weight * expression.subs(variable, node)
            for node, weight in zip(nodes, weights)
        ))
        exact = self.sp.integrate(expression, (variable, lower, upper))
        if exact.has(self.sp.Integral, self.sp.nan, self.sp.zoo):
            return None
        error_iq = self.sp.simplify(exact - quadrature)
        if re.search(r"Q\s*-\s*I", text, re.IGNORECASE):
            error_label = "Q-I"
            error = -error_iq
        else:
            error_label = "I-Q"
            error = error_iq

        rendered_nodes = ",\\;".join(self.sp.latex(node) for node in nodes)
        rendered_weights = ",\\;".join(self.sp.latex(weight) for weight in weights)
        rendered_q = self.sp.latex(quadrature)
        rendered_i = self.sp.latex(exact)
        rendered_error = self.sp.latex(self.sp.simplify(error))
        zh = self._is_chinese(text)
        if zh:
            result = (
                rf"节点为 $({rendered_nodes})$，权重为 $({rendered_weights})$；"
                rf"求积值 $Q={rendered_q}$，精确积分 $I={rendered_i}$，故 $"
                + error_label
                + rf"={rendered_error}$。"
            )
            support = (
                rf"节点取 $P_{{{point_count}}}$ 的根并仿射映射到题设区间，权重由"
                r"$2/((1-\xi_j^2)[P_n'(\xi_j)]^2)$ 同步缩放得到；"
                f"已逐次精确核对 0 到 {2 * point_count - 1} 次矩，并另作符号积分。"
            )
        else:
            result = (
                rf"The nodes are $({rendered_nodes})$ and the weights are "
                rf"$({rendered_weights})$; $Q={rendered_q}$, $I={rendered_i}$, so $"
                + error_label
                + rf"={rendered_error}$."
            )
            support = (
                rf"The roots of $P_{{{point_count}}}$ were affinely mapped to the "
                r"stated interval and the weights obtained from "
                r"$2/((1-\xi_j^2)[P_n'(\xi_j)]^2)$ with the same scaling. "
                f"Moments of degrees 0 through {2 * point_count - 1} and the "
                "symbolic integral were independently checked."
            )
        return self._result(
            text,
            "gauss_legendre_quadrature",
            result,
            "nodes_weights_value_and_error",
            "legendre_roots_weights_moment_and_integral_crosscheck",
            (
                "method_and_point_count_parsed",
                "single_definite_polynomial_integral_parsed",
                "legendre_roots_recomputed",
                "positive_weights_recomputed",
                "all_exactness_moments_verified",
                "quadrature_sum_and_symbolic_integral_recomputed",
            ),
            support,
            ("expression", "number"),
            (
                "result_present",
                "numeric_result",
                "reasoning",
                "quadrature_nodes",
                "quadrature_weights",
                "quadrature_value",
                "quadrature_error",
                "exact_and_approximate",
            ),
        )

    def _jacobi_exact_iterations(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"\bJacobi\b|雅可比", text, re.IGNORECASE):
            return None
        matrices = self._latex_matrices(text)
        square = [matrix for matrix in matrices if matrix.rows == matrix.cols]
        if len(square) != 1 or not 2 <= square[0].rows <= 6:
            return None
        matrix = square[0]
        dimension = matrix.rows
        rhs = self._jacobi_rhs(text, dimension)
        initial = self._jacobi_initial(text, dimension)
        requested = {
            int(value)
            for value in re.findall(
                r"x\s*\^\s*\{\s*\(\s*(\d{1,2})\s*\)\s*\}",
                text,
                re.IGNORECASE,
            )
            if int(value) > 0
        }
        if rhs is None or initial is None or not requested:
            return None
        last_iteration = max(requested)
        if last_iteration > 20:
            return None
        diagonal_values = [matrix[index, index] for index in range(dimension)]
        if any(value == 0 or value.is_zero is not False for value in diagonal_values):
            return None
        diagonal = self.sp.diag(*diagonal_values)
        remainder = matrix - diagonal
        iteration_matrix = self.sp.simplify(-diagonal.inv() * remainder)
        constant = self.sp.simplify(diagonal.inv() * rhs)

        characteristic = iteration_matrix.charpoly()
        charpoly = characteristic.as_expr()
        eigenvalues = iteration_matrix.eigenvals()
        if sum(eigenvalues.values()) != dimension:
            return None
        moduli = [self.sp.simplify(self.sp.Abs(value)) for value in eigenvalues]
        if any(not self._finite(value) for value in moduli):
            return None
        spectral_radius = max(
            moduli,
            key=lambda value: float(self.sp.N(value, 60)),
        )
        rho_numeric = float(self.sp.N(spectral_radius, 80))
        if abs(rho_numeric - 1.0) < 1e-30:
            convergence = "boundary"
        elif rho_numeric < 1:
            convergence = "convergent"
        else:
            convergence = "divergent"
        if any(
            self.sp.simplify(charpoly.subs(characteristic.gen, value)) != 0
            for value in eigenvalues
        ):
            return None

        iterates = [initial]
        for _ in range(last_iteration):
            following = self.sp.simplify(iteration_matrix * iterates[-1] + constant)
            if self.sp.simplify(
                diagonal * following - (rhs - remainder * iterates[-1])
            ) != self.sp.zeros(dimension, 1):
                return None
            iterates.append(following)

        rendered_matrix = self.sp.latex(iteration_matrix)
        rendered_rho = self.sp.latex(spectral_radius)
        rendered_iterates = "，".join(
            rf"$x^{{({index})}}={self.sp.latex(iterates[index])}$"
            for index in sorted(requested)
        )
        zh = self._is_chinese(text)
        if zh:
            if convergence == "convergent":
                convergence_text = r"因 $\rho(B_J)<1$，Jacobi 迭代对任意初值收敛。"
            elif convergence == "divergent":
                convergence_text = r"因 $\rho(B_J)>1$，Jacobi 迭代并非对任意初值收敛。"
            else:
                convergence_text = r"$\rho(B_J)=1$，谱半径判据不能给出严格收敛。"
            result = (
                rf"迭代矩阵 $B_J={rendered_matrix}$，谱半径"
                rf"$\rho(B_J)={rendered_rho}$；{rendered_iterates}。"
                + convergence_text
            )
            support = (
                r"按 $A=D+(L+U)$ 精确重算 $B_J=-D^{-1}(L+U)$ 与"
                r"$c=D^{-1}b$；每个迭代向量均同时代回"
                r"$Dx^{(k+1)}=b-(L+U)x^{(k)}$ 核验，并由特征多项式重算谱半径。"
            )
        else:
            if convergence == "convergent":
                convergence_text = (
                    r"Since $\rho(B_J)<1$, Jacobi iteration converges from every "
                    "initial vector."
                )
            elif convergence == "divergent":
                convergence_text = (
                    r"Since $\rho(B_J)>1$, Jacobi iteration does not converge from "
                    "every initial vector."
                )
            else:
                convergence_text = (
                    r"Here $\rho(B_J)=1$, so the spectral-radius criterion gives no "
                    "strict convergence."
                )
            rendered_iterates_en = ", ".join(
                rf"$x^{{({index})}}={self.sp.latex(iterates[index])}$"
                for index in sorted(requested)
            )
            result = (
                rf"The iteration matrix is $B_J={rendered_matrix}$ and the spectral "
                rf"radius is $\rho(B_J)={rendered_rho}$; {rendered_iterates_en}. "
                + convergence_text
            )
            support = (
                r"With $A=D+(L+U)$, both $B_J=-D^{-1}(L+U)$ and $c=D^{-1}b$ "
                r"were recomputed exactly. Every iterate was independently checked in "
                r"$Dx^{(k+1)}=b-(L+U)x^{(k)}$, and the spectral radius was recomputed "
                "from the characteristic polynomial."
            )
        return self._result(
            text,
            "jacobi_exact_iterations",
            result,
            "iteration_matrix_radius_and_iterates",
            "exact_jacobi_split_charpoly_and_step_substitution",
            (
                "single_square_matrix_parsed",
                "rhs_and_initial_vector_parsed",
                "nonzero_diagonal_verified",
                "iteration_matrix_recomputed",
                "spectral_radius_recomputed_from_charpoly",
                "every_requested_iterate_substituted",
            ),
            support,
            ("expression", "matrix"),
            (
                "result_present",
                "reasoning",
                "iteration_matrix",
                "spectral_radius",
                "requested_iterates",
                "convergence_judgement",
            ),
        )

    def _composite_trapezoid(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"复化梯形|复合梯形|composite\s+trapezoid", text, re.IGNORECASE):
            return None
        n = self._subinterval_count(text)
        integral = self._definite_integral(text)
        if n is None or integral is None or not 1 <= n <= 10_000:
            return None
        expression, variable, lower, upper = integral
        if expression.free_symbols - {variable} or not self._strictly_less(lower, upper):
            return None
        step = self.sp.simplify((upper - lower) / n)
        values = tuple(
            self.sp.simplify(expression.subs(variable, lower + index * step))
            for index in range(n + 1)
        )
        if any(value.has(self.sp.nan, self.sp.zoo) for value in values):
            return None
        approximate = self.sp.simplify(
            step * (values[0] / 2 + sum(values[1:-1], self.sp.S.Zero) + values[-1] / 2)
        )
        exact = self.sp.integrate(expression, (variable, lower, upper))
        if exact.has(self.sp.Integral, self.sp.nan, self.sp.zoo):
            return None
        difference = self.sp.simplify(approximate - exact)
        relation = "=" if difference == 0 else (">" if self._strictly_less(0, difference) else "<" if self._strictly_less(difference, 0) else "")
        if not relation:
            return None
        zh = self._is_chinese(text)
        compare_exact = self._requests_exact_comparison(text)
        result = (
            (
                rf"步长 $h={self.sp.latex(step)}$，复化梯形近似值 $T_n={self.sp.latex(approximate)}$；精确值 $I={self.sp.latex(exact)}$，故 $T_n{relation}I$。"
                if zh else
                rf"With $h={self.sp.latex(step)}$, the composite trapezoid value is $T_n={self.sp.latex(approximate)}$; the exact integral is $I={self.sp.latex(exact)}$, so $T_n{relation}I$."
            )
            if compare_exact
            else rf"T_n={self.sp.latex(approximate)}"
        )
        support = (
            r"已在全部等距节点上精确求值并代入复化梯形公式，另以符号积分独立核对。"
            if zh else
            r"All equally spaced node values were evaluated exactly in the composite trapezoid formula and checked against an independent symbolic integral."
        )
        return self._result(
            text, "composite_trapezoid", result, "approximation_and_exact",
            "exact_composite_trapezoid_sum_and_integral",
            ("method_explicit", "integral_parsed", "subinterval_count_parsed", "all_nodes_evaluated", "exact_integral_recomputed"),
            support,
            ("expression", "number"),
            (
                "result_present", "numeric_result", "quadrature_value",
                "method_formula", "exact_and_approximate",
            ),
        )

    def _composite_simpson(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"辛普森|Simpson", text, re.IGNORECASE):
            return None
        n = self._subinterval_count(text)
        integral = self._definite_integral(text)
        if n is None or integral is None or not 2 <= n <= 10_000 or n % 2:
            return None
        expression, variable, lower, upper = integral
        if expression.free_symbols - {variable} or not self._strictly_less(lower, upper):
            return None
        step = self.sp.simplify((upper - lower) / n)
        values = tuple(
            self.sp.simplify(expression.subs(variable, lower + index * step))
            for index in range(n + 1)
        )
        if any(value.has(self.sp.nan, self.sp.zoo) for value in values):
            return None
        odd_sum = sum(values[index] for index in range(1, n, 2))
        even_sum = sum(values[index] for index in range(2, n, 2))
        approximate = self.sp.simplify(
            step * (values[0] + values[-1] + 4 * odd_sum + 2 * even_sum) / 3
        )
        exact = self.sp.integrate(expression, (variable, lower, upper))
        if exact.has(self.sp.Integral, self.sp.nan, self.sp.zoo):
            return None
        difference = self.sp.simplify(approximate - exact)
        relation = "=" if difference == 0 else (">" if self._strictly_less(0, difference) else "<" if self._strictly_less(difference, 0) else "")
        if not relation:
            return None
        zh = self._is_chinese(text)
        compare_exact = self._requests_exact_comparison(text)
        result = (
            (
                rf"步长 $h={self.sp.latex(step)}$，复化 Simpson 近似值 $S_n={self.sp.latex(approximate)}$；精确值 $I={self.sp.latex(exact)}$，故 $S_n{relation}I$。"
                if zh else
                rf"With $h={self.sp.latex(step)}$, the composite Simpson value is $S_n={self.sp.latex(approximate)}$; the exact integral is $I={self.sp.latex(exact)}$, so $S_n{relation}I$."
            )
            if compare_exact
            else rf"S_n={self.sp.latex(approximate)}"
        )
        support = (
            "分段数为偶数；全部节点按端点、奇节点权 4、偶节点权 2 精确求和，并以符号积分核对。"
            if zh else
            "The subinterval count is even; all endpoint, odd-node (weight 4), and even-node (weight 2) terms were summed exactly and checked by symbolic integration."
        )
        return self._result(
            text, "composite_simpson", result, "approximation_and_exact",
            "exact_composite_simpson_sum_and_integral",
            ("method_explicit", "integral_parsed", "positive_even_subinterval_count", "all_nodes_evaluated", "exact_integral_recomputed"),
            support,
            ("expression", "number"),
            (
                "result_present", "numeric_result", "quadrature_value",
                "method_formula", "exact_and_approximate",
            ),
        )

    def _forward_euler(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"前向欧拉|显式欧拉|forward\s+Euler|explicit\s+Euler", text, re.IGNORECASE):
            return None
        rhs = self._ode_rhs(text)
        initial = self._initial_condition(text)
        step = self._named_scalar(text, "h")
        target = self._target_x(text)
        if rhs is None or initial is None or step is None or target is None:
            return None
        x0, y0 = initial
        if not self._strictly_less(0, step):
            return None
        count_value = self.sp.simplify((target - x0) / step)
        if count_value.is_integer is not True:
            return None
        count = int(count_value)
        if not 1 <= count <= 10_000:
            return None
        x, y = self.sp.symbols("x y")
        if rhs.free_symbols - {x, y}:
            return None
        current_x, current_y = x0, y0
        iterates = []
        for index in range(1, count + 1):
            current_y = self.sp.simplify(current_y + step * rhs.subs({x: current_x, y: current_y}))
            current_x = self.sp.simplify(current_x + step)
            if current_y.has(self.sp.nan, self.sp.zoo):
                return None
            iterates.append((index, current_y))
        if self.sp.simplify(current_x - target) != 0:
            return None
        zh = self._is_chinese(text)
        result = (
            rf"前向 Euler 公式为 $y_{{k+1}}=y_k+h f(x_k,y_k)$；经 {count} 步得 $y({self.sp.latex(target)})\approx {self.sp.latex(current_y)}$。"
            if zh else
            rf"Forward Euler uses $y_{{k+1}}=y_k+h f(x_k,y_k)$; after {count} steps, $y({self.sp.latex(target)})\approx {self.sp.latex(current_y)}$."
        )
        rendered = r",\;".join(rf"y_{i}={self.sp.latex(v)}" for i, v in iterates)
        support = (
            rf"由题面初值逐步精确代入，得到 ${rendered}$。"
            if zh else
            rf"Exact stepwise substitution from the stated initial value gives ${rendered}$."
        )
        return self._result(
            text, "forward_euler", result, "numeric_approximation",
            "exact_explicit_euler_recurrence",
            ("method_explicit", "ode_rhs_parsed", "initial_condition_parsed", "positive_step", "integer_step_count", "all_steps_recomputed"),
            support,
            ("number", "expression"),
            ("result_present", "numeric_result", "method_formula", "first_iteration"),
        )

    def _improved_euler(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"改进欧拉|修正欧拉|Heun|improved\s+Euler|modified\s+Euler", text, re.IGNORECASE):
            return None
        data = self._ode_grid_data(text)
        if data is None:
            return None
        rhs, x, y, current_x, current_y, step, target, count = data
        iterates = []
        for index in range(1, count + 1):
            first_slope = self.sp.simplify(rhs.subs({x: current_x, y: current_y}))
            predictor = self.sp.simplify(current_y + step * first_slope)
            second_slope = self.sp.simplify(
                rhs.subs({x: current_x + step, y: predictor})
            )
            current_y = self.sp.simplify(
                current_y + step * (first_slope + second_slope) / 2
            )
            current_x = self.sp.simplify(current_x + step)
            if current_y.has(self.sp.nan, self.sp.zoo):
                return None
            iterates.append((index, current_y))
        zh = self._is_chinese(text)
        result = (
            rf"改进 Euler 采用预测 $\widetilde y_{{k+1}}=y_k+h f(x_k,y_k)$ 与校正 $y_{{k+1}}=y_k+\frac h2[f(x_k,y_k)+f(x_{{k+1}},\widetilde y_{{k+1}})]$；经 {count} 步得 $y({self.sp.latex(target)})\approx {self.sp.latex(current_y)}$。"
            if zh else
            rf"Improved Euler uses predictor $\widetilde y_{{k+1}}=y_k+h f(x_k,y_k)$ and corrector $y_{{k+1}}=y_k+\frac h2[f(x_k,y_k)+f(x_{{k+1}},\widetilde y_{{k+1}})]$; after {count} steps, $y({self.sp.latex(target)})\approx {self.sp.latex(current_y)}$."
        )
        support = self._iterate_support(text, iterates)
        return self._result(
            text, "improved_euler", result, "numeric_approximation",
            "exact_heun_predictor_corrector_recurrence",
            ("method_explicit", "ode_grid_data_parsed", "integer_step_count", "predictor_and_corrector_each_step", "target_reached"),
            support,
            ("number", "expression"),
            ("result_present", "numeric_result", "method_formula"),
        )

    def _explicit_runge_kutta_stability(self, text: str) -> Optional[ToolResult]:
        """Derive the negative-axis interval from a named explicit RK tableau."""
        if not re.search(
            r"稳定函数|稳定域|绝对稳定|stability\s+(?:function|region)|"
            r"absolute\s+stability",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(
            r"负实轴|negative\s+real\s+axis",
            text,
            re.IGNORECASE,
        ):
            return None

        half = self.sp.Rational(1, 2)
        sixth = self.sp.Rational(1, 6)
        methods = (
            (
                r"经典四阶\s*(?:Runge[- ]?Kutta|龙格[-— ]?库塔)|"
                r"classical\s+fourth[- ]order\s+Runge[- ]?Kutta|\bRK4\b",
                "classical RK4",
                self.sp.Matrix([
                    [0, 0, 0, 0],
                    [half, 0, 0, 0],
                    [0, half, 0, 0],
                    [0, 0, 1, 0],
                ]),
                self.sp.Matrix([sixth, self.sp.Rational(1, 3), self.sp.Rational(1, 3), sixth]),
            ),
            (
                r"(?:经典三阶|Kutta\s+third[- ]order|classical\s+third[- ]order)"
                r"\s*(?:Runge[- ]?Kutta|龙格[-— ]?库塔)?|\bRK3\b",
                "classical RK3",
                self.sp.Matrix([
                    [0, 0, 0],
                    [half, 0, 0],
                    [-1, 2, 0],
                ]),
                self.sp.Matrix([sixth, self.sp.Rational(2, 3), sixth]),
            ),
            (
                r"改进欧拉|修正欧拉|Heun|improved\s+Euler|explicit\s+trapezoid",
                "Heun RK2",
                self.sp.Matrix([[0, 0], [1, 0]]),
                self.sp.Matrix([half, half]),
            ),
            (
                r"显式中点|explicit\s+midpoint",
                "explicit midpoint RK2",
                self.sp.Matrix([[0, 0], [half, 0]]),
                self.sp.Matrix([0, 1]),
            ),
            (
                r"前向欧拉|显式欧拉|forward\s+Euler|explicit\s+Euler",
                "forward Euler",
                self.sp.Matrix([[0]]),
                self.sp.Matrix([1]),
            ),
        )
        selected = next(
            (
                (name, matrix, weights)
                for pattern, name, matrix, weights in methods
                if re.search(pattern, text, re.IGNORECASE)
            ),
            None,
        )
        if selected is None:
            return None
        method_name, matrix, weights = selected

        z = self.sp.Symbol("z")
        stages = matrix.rows
        ones = self.sp.ones(stages, 1)
        stability = self.sp.cancel(
            1 + z * (weights.T * (self.sp.eye(stages) - z * matrix).inv() * ones)[0]
        )
        stability = self.sp.expand(stability)
        if stability.free_symbols != {z} or self.sp.simplify(stability.subs(z, 0) - 1) != 0:
            return None

        boundary = self._negative_real_stability_boundary(stability, z)
        if boundary is None:
            return None
        endpoint, target, boundary_equation = boundary
        endpoint_text = format(float(endpoint), ".12g")
        equation_text = self.sp.latex(boundary_equation)
        stability_text = self.sp.latex(stability)
        zh = self._is_chinese(text)
        result = (
            rf"稳定函数为 $R(z)={stability_text}$；令 $z_*<0$ 为方程 "
            rf"${equation_text}=0$ 中与 $0$ 相邻的负实根，则 "
            rf"$z_*\approx -{endpoint_text}$，负实轴上的闭绝对稳定区间为 "
            rf"$[z_*,0]\approx[-{endpoint_text},0]$。"
            if zh else
            rf"The stability function is $R(z)={stability_text}$. Let $z_*<0$ be "
            rf"the negative root nearest zero of ${equation_text}=0$. Then "
            rf"$z_*\approx -{endpoint_text}$ and the closed absolute-stability "
            rf"interval on the negative real axis is $[z_*,0]\approx[-{endpoint_text},0]$."
        )
        relation = "1" if target == 1 else "-1"
        support = (
            rf"由 {method_name} 的 Butcher 数据按公式 "
            rf"$R(z)=1+zb^T(I-zA)^{{-1}}\mathbf 1$ 作符号运算得到上述多项式。"
            rf"沿负实轴从 $0$ 连续向左检查 $R(z)=\pm1$ 的全部实根，第一个边界满足 "
            rf"$R(z_*)={relation}$；区间内部没有别的 $|R|=1$ 交点，边界外侧 "
            rf"$|R|>1$，故闭区间如上。"
            if zh else
            rf"Using the {method_name} Butcher data in "
            rf"$R(z)=1+zb^T(I-zA)^{{-1}}\mathbf 1$ gives the displayed polynomial "
            rf"symbolically. All real roots of $R(z)=\pm1$ are checked from zero "
            rf"leftward; the first boundary has $R(z_*)={relation}$, there is no other "
            rf"$|R|=1$ crossing inside, and immediately outside one has $|R|>1$."
        )
        return self._result(
            text,
            "explicit_runge_kutta_stability",
            result,
            "stability_function_and_interval",
            "butcher_resolvent_and_exact_negative_axis_boundary",
            (
                "named_explicit_runge_kutta_method_matched",
                "butcher_matrix_and_weights_instantiated",
                "stability_function_symbolically_derived",
                "R_at_zero_equals_one",
                "all_negative_axis_unit_modulus_roots_enumerated",
                "connected_stability_component_certified",
                "outside_endpoint_is_unstable",
            ),
            support,
            ("number", "expression", "interval"),
            (
                "result_present",
                "numeric_result",
                "reasoning",
                "stability_function",
                "stability_boundary_equation",
                "closed_stability_interval",
            ),
        )

    def _negative_real_stability_boundary(self, stability, variable):
        """Return the first positive r where |R(-r)| leaves the unit interval."""
        r = self.sp.Symbol("r", real=True)
        negative_axis = self.sp.Poly(
            self.sp.together(stability.subs(variable, -r)), r
        )
        candidates: list[tuple[float, int, object]] = []
        for target in (-1, 1):
            equation = self.sp.Poly(negative_axis.as_expr() - target, r)
            for root in self.sp.nroots(equation, maxsteps=200):
                numeric = complex(root)
                if abs(numeric.imag) <= 1e-10 and numeric.real > 1e-9:
                    candidates.append((numeric.real, target, equation.as_expr()))
        candidates.sort(key=lambda item: item[0])
        for endpoint, target, equation_r in candidates:
            inside = max(endpoint * (1 - 1e-7), endpoint - 1e-7)
            outside = endpoint * (1 + 1e-7)
            inside_value = abs(float(self.sp.N(negative_axis.as_expr().subs(r, inside), 18)))
            outside_value = abs(float(self.sp.N(negative_axis.as_expr().subs(r, outside), 18)))
            if inside_value > 1 + 1e-8 or outside_value <= 1 + 1e-9:
                continue
            earlier = [item[0] for item in candidates if item[0] < endpoint - 1e-8]
            if earlier:
                continue
            residual = abs(float(self.sp.N(equation_r.subs(r, endpoint), 18)))
            if residual > 1e-8:
                continue
            equation_z = self.sp.Poly(
                self.sp.expand(stability - target), variable
            )
            while equation_z.eval(0) == 0 and equation_z.degree() > 0:
                quotient, remainder = self.sp.div(
                    equation_z, self.sp.Poly(variable, variable)
                )
                if not remainder.is_zero:
                    break
                equation_z = quotient
            _, primitive = equation_z.clear_denoms(convert=True)
            primitive = primitive.primitive()[1]
            if primitive.LC() < 0:
                primitive = -primitive
            return endpoint, target, self.sp.expand(primitive.as_expr())
        return None

    def _runge_kutta_4(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"四阶\s*(?:Runge[- ]?Kutta|龙格[-— ]?库塔)|RK4|fourth[- ]order\s+Runge[- ]?Kutta", text, re.IGNORECASE):
            return None
        data = self._ode_grid_data(text)
        if data is None:
            return None
        rhs, x, y, current_x, current_y, step, target, count = data
        iterates = []
        for index in range(1, count + 1):
            k1 = self.sp.simplify(rhs.subs({x: current_x, y: current_y}))
            k2 = self.sp.simplify(rhs.subs({
                x: current_x + step / 2,
                y: current_y + step * k1 / 2,
            }))
            k3 = self.sp.simplify(rhs.subs({
                x: current_x + step / 2,
                y: current_y + step * k2 / 2,
            }))
            k4 = self.sp.simplify(rhs.subs({
                x: current_x + step,
                y: current_y + step * k3,
            }))
            current_y = self.sp.simplify(
                current_y + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            )
            current_x = self.sp.simplify(current_x + step)
            if current_y.has(self.sp.nan, self.sp.zoo):
                return None
            iterates.append((index, current_y))
        zh = self._is_chinese(text)
        result = (
            rf"按经典四阶 Runge--Kutta 的 $k_1,k_2,k_3,k_4$ 加权公式迭代 {count} 步，得到 $y({self.sp.latex(target)})\approx {self.sp.latex(current_y)}$。"
            if zh else
            rf"Applying the classical fourth-order Runge--Kutta $k_1,k_2,k_3,k_4$ weighted update for {count} steps gives $y({self.sp.latex(target)})\approx {self.sp.latex(current_y)}$."
        )
        support = self._iterate_support(text, iterates)
        return self._result(
            text, "runge_kutta_4", result, "numeric_approximation",
            "exact_classical_rk4_recurrence",
            ("method_explicit", "ode_grid_data_parsed", "integer_step_count", "four_stages_each_step", "target_reached"),
            support,
            ("number", "expression"),
            ("result_present", "numeric_result", "method_formula"),
        )

    def _taylor_polynomial(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"Taylor|泰勒", text, re.IGNORECASE):
            return None
        if not re.search(r"多项式|polynomial", text, re.IGNORECASE):
            return None
        expression = self._function_expression(text, "x")
        center = self._taylor_center(text)
        order = self._taylor_order(text)
        if expression is None or center is None or order is None or not 0 <= order <= 30:
            return None
        x = self.sp.Symbol("x")
        if expression.free_symbols - {x}:
            return None
        polynomial = self.sp.expand(
            self.sp.series(expression, x, center, order + 1).removeO()
        )
        for derivative_order in range(order + 1):
            original_value = self.sp.diff(expression, x, derivative_order).subs(x, center)
            polynomial_value = self.sp.diff(polynomial, x, derivative_order).subs(x, center)
            if self.sp.simplify(original_value - polynomial_value) != 0:
                return None
        zh = self._is_chinese(text)
        result = (
            rf"{order} 阶 Taylor 多项式为 $T_{order}(x)={self.sp.latex(polynomial)}$。"
            if zh else
            rf"The Taylor polynomial of order {order} is $T_{order}(x)={self.sp.latex(polynomial)}$."
        )
        support = (
            rf"按 Taylor 系数公式展开，并已核对在 $x={self.sp.latex(center)}$ 处从 0 到 {order} 阶的全部导数。"
            if zh else
            rf"Taylor coefficients were computed symbolically and derivatives of orders 0 through {order} were checked at $x={self.sp.latex(center)}$."
        )
        return self._result(
            text, "taylor_polynomial", result, "polynomial",
            "symbolic_series_with_derivative_jet_check",
            ("method_explicit", "function_center_order_parsed", "series_truncated", "all_requested_derivatives_rechecked"),
            support,
            ("expression",),
            ("result_present", "method_formula"),
        )

    def _function_expression(self, text: str, variable_name: str):
        symbol = self.sp.Symbol(variable_name)
        for segment in self._math_segments(text):
            cleaned = self._clean_math(segment)
            if re.search(rf"{re.escape(variable_name)}\s*_\s*\{{?\s*\d", cleaned):
                continue
            function = re.fullmatch(
                rf"f\s*\(\s*{re.escape(variable_name)}\s*\)\s*=\s*(.+)",
                cleaned,
                re.IGNORECASE,
            )
            if function:
                expression = self._math_expr(function.group(1))
                if expression is not None and not expression.free_symbols - {symbol}:
                    return self.sp.simplify(expression)
            if len(re.findall(r"(?<![<>!])=(?!=)", cleaned)) == 1:
                left, right = re.split(r"(?<![<>!])=(?!=)", cleaned, maxsplit=1)
                left_expr, right_expr = self._math_expr(left), self._math_expr(right)
                if left_expr is None or right_expr is None:
                    continue
                expression = self.sp.simplify(left_expr - right_expr)
                if expression.free_symbols and not expression.free_symbols - {symbol}:
                    return expression
        boundary = (
            r"(?=\s*(?:[,，。;；\n]|在\s*(?:区间|点|x\s*=)|由\s*x\s*_?\s*\{?0|"
            r"并(?:由|取|从|计算|写|说明)|且(?:由|取)|"
            r"\b(?:on|over|with|using|where|starting|from|at)\b|$))"
        )
        plain = re.search(
            rf"f\s*\(\s*{re.escape(variable_name)}\s*\)\s*=\s*(.+?){boundary}",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if plain:
            expression = self._math_expr(plain.group(1))
            if expression is not None and not expression.free_symbols - {symbol}:
                return self.sp.simplify(expression)
        equation = re.search(
            r"(?:方程|equation)\s*[:：]?\s*(.+?)\s*=\s*(.+?)"
            r"(?=\s*(?:的(?:迭代|根|解|正根|负根)|[,，。;；\n]|"
            r"\b(?:with|using|where|starting|from|for)\b|$))",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if equation is None and re.search(
            r"二分法|牛顿法|割线法|bisection|Newton(?:'s)?\s+method|"
            r"secant\s+method",
            text,
            re.IGNORECASE,
        ):
            equation = re.search(
                r"(?:求|solve)\s*(?:方程\s*)?(.+?)\s*=\s*(.+?)"
                r"(?=\s*(?:[,，。;；\n]|的(?:根|解|迭代)|"
                r"\b(?:with|using|where|starting|from|for)\b|$))",
                text,
                re.IGNORECASE | re.DOTALL,
            )
        if equation:
            left = self._math_expr(equation.group(1))
            right = self._math_expr(equation.group(2))
            if left is not None and right is not None:
                expression = self.sp.simplify(left - right)
                if expression.free_symbols and not expression.free_symbols - {symbol}:
                    return expression
        return None

    def _interval(self, text: str):
        for left, right in re.findall(r"[\[【]\s*([^,，\]】]+)\s*[,，]\s*([^\]】]+)\s*[\]】]", text):
            lower, upper = self._math_expr(left), self._math_expr(right)
            if lower is not None and upper is not None and not lower.free_symbols and not upper.free_symbols:
                return lower, upper
        return None

    def _definite_integral(self, text: str):
        finite_bound = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*\d+)?"
        patterns = (
            re.compile(
                r"(?:∫|\\int)\s*_\s*\{\s*([^{}]+)\s*\}\s*"
                r"\^\s*\{\s*([^{}]+)\s*\}\s*(.+?)\s*"
                r"(?:\\?d\s*([A-Za-z])|\\,\s*d\s*([A-Za-z]))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:∫|\\int)\s*_\s*\{\s*([^{}]+)\s*\}\s*"
                r"\^\s*(" + finite_bound + r")\s*(.+?)\s*"
                r"(?:\\?d\s*([A-Za-z])|\\,\s*d\s*([A-Za-z]))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:∫|\\int)\s*_\s*(" + finite_bound + r")\s*"
                r"\^\s*(" + finite_bound + r")\s*(.+?)\s*"
                r"(?:\\?d\s*([A-Za-z])|\\,\s*d\s*([A-Za-z]))",
                re.IGNORECASE,
            ),
        )
        matches = [match for pattern in patterns if (match := pattern.search(text))]
        if len(matches) != 1:
            return None
        match = matches[0]
        lower = self._math_expr(match.group(1))
        upper = self._math_expr(match.group(2))
        expression = self._math_expr(match.group(3))
        variable = self.sp.Symbol(match.group(4) or match.group(5))
        if any(item is None for item in (lower, upper, expression)):
            return None
        return expression, variable, lower, upper

    def _latex_matrices(self, text: str) -> tuple:
        matrices = []
        for match in re.finditer(
            r"\\begin\{[pbvBV]?matrix\}(.+?)\\end\{[pbvBV]?matrix\}",
            text,
            re.DOTALL,
        ):
            rows = [
                row.strip()
                for row in re.split(r"\\\\", match.group(1))
                if row.strip()
            ]
            cells = [[cell.strip() for cell in row.split("&")] for row in rows]
            if (
                not cells
                or len(cells) > 6
                or len(cells[0]) > 6
                or any(len(row) != len(cells[0]) for row in cells)
            ):
                return ()
            parsed_rows = []
            for row in cells:
                parsed_row = []
                for cell in row:
                    value = self._math_expr(cell)
                    if (
                        value is None
                        or value.free_symbols
                        or value.is_real is not True
                    ):
                        return ()
                    parsed_row.append(value)
                parsed_rows.append(parsed_row)
            matrices.append(self.sp.Matrix(parsed_rows))
        return tuple(matrices)

    def _jacobi_rhs(self, text: str, dimension: int):
        matrices = self._latex_matrices(text)
        columns = [
            matrix for matrix in matrices
            if matrix.rows == dimension and matrix.cols == 1
        ]
        candidates = list(columns)
        if dimension == 2:
            braced = re.search(
                r"\\binom\s*\{\s*([^{}]+)\s*\}\s*\{\s*([^{}]+)\s*\}",
                text,
            )
            shorthand = re.search(
                r"\\binom\s*([-+]?\d)\s*([-+]?\d)(?!\d)",
                text,
            )
            match = braced or shorthand
            if match:
                values = [self._math_expr(match.group(1)), self._math_expr(match.group(2))]
                if all(
                    value is not None
                    and not value.free_symbols
                    and value.is_real is True
                    for value in values
                ):
                    candidates.append(self.sp.Matrix(values))
        tuple_match = re.search(
            r"(?:\\end\{[pbvBV]?matrix\}|A)\s*x\s*=\s*"
            r"\(\s*([^()\n]+)\s*\)\s*\^?\s*\{?T\}?",
            text,
            re.IGNORECASE,
        )
        if tuple_match:
            parts = [part.strip() for part in re.split(r"[,，]", tuple_match.group(1))]
            values = [self._math_expr(part) for part in parts]
            if (
                len(values) == dimension
                and all(
                    value is not None
                    and not value.free_symbols
                    and value.is_real is True
                    for value in values
                )
            ):
                candidates.append(self.sp.Matrix(values))
        unique = []
        for candidate in candidates:
            if not any(candidate == existing for existing in unique):
                unique.append(candidate)
        return unique[0] if len(unique) == 1 else None

    def _jacobi_initial(self, text: str, dimension: int):
        prefix = r"x\s*\^\s*\{\s*\(\s*0\s*\)\s*\}\s*=\s*"
        if re.search(prefix + r"0(?:\.0+)?(?=\s*[$,，。;；\n]|$)", text, re.IGNORECASE):
            return self.sp.zeros(dimension, 1)
        tuple_match = re.search(
            prefix + r"\(\s*([^()]+)\s*\)\s*\^?\s*\{?T\}?",
            text,
            re.IGNORECASE,
        )
        if tuple_match:
            parts = [part.strip() for part in re.split(r"[,，]", tuple_match.group(1))]
            values = [self._math_expr(part) for part in parts]
            if (
                len(values) == dimension
                and all(
                    value is not None
                    and not value.free_symbols
                    and value.is_real is True
                    for value in values
                )
            ):
                return self.sp.Matrix(values)
        if dimension == 2:
            braced = re.search(
                prefix + r"\\binom\s*\{\s*([^{}]+)\s*\}\s*\{\s*([^{}]+)\s*\}",
                text,
            )
            if braced:
                values = [
                    self._math_expr(braced.group(1)),
                    self._math_expr(braced.group(2)),
                ]
                if all(
                    value is not None
                    and not value.free_symbols
                    and value.is_real is True
                    for value in values
                ):
                    return self.sp.Matrix(values)
        return None

    def _ode_rhs(self, text: str):
        match = re.search(r"y\s*(?:'|′|\\prime)\s*=\s*([^,，。;；\n$]+)", text)
        if not match:
            for segment in self._math_segments(text):
                match = re.search(r"y\s*(?:'|′|\\prime)\s*=\s*(.+)", self._clean_math(segment))
                if match:
                    break
        return self._math_expr(match.group(1)) if match else None

    def _initial_condition(self, text: str):
        match = re.search(r"y\s*\(\s*" + self._SCALAR + r"\s*\)\s*=\s*" + self._SCALAR, text)
        if not match:
            return None
        x0, y0 = self._math_expr(match.group(1)), self._math_expr(match.group(2))
        return (x0, y0) if x0 is not None and y0 is not None else None

    def _target_x(self, text: str):
        patterns = (
            r"(?:近似(?:计算|求)?|计算|estimate|approximate)[^。.;\n]{0,30}?y\s*\(\s*" + self._SCALAR + r"\s*\)",
            r"(?:在|at)\s*x\s*=\s*" + self._SCALAR,
        )
        for pattern in patterns:
            matches = tuple(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                return self._math_expr(matches[-1].group(1))
        return None

    def _tolerance(self, text: str):
        probe = str(text or "")
        for marker in ("$", r"\(", r"\)", r"\[", r"\]"):
            probe = probe.replace(marker, "")
        labels = r"(?:tol(?:erance)?|eps(?:ilon)?|\\epsilon|ε|精度|误差(?:限|不超过)?)"
        patterns = (
            labels + r"\s*(?:=|为|is|不超过|<=|≤)\s*([^,，。;；\s$]+(?:\s*\^\s*\{?\s*[+-]?\d+\s*\}?)?)",
            r"(?:误差|error)[^,，。;；\n]{0,20}?(?:不超过|at\s+most|<=|≤)\s*([^,，。;；\s$]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, probe, re.IGNORECASE)
            if match:
                value = self._math_expr(match.group(1))
                if value is not None and not value.free_symbols:
                    return value
        for segment in self._math_segments(text):
            match = re.search(
                labels + r"\s*=\s*(.+)$",
                self._clean_math(segment),
                re.IGNORECASE,
            )
            if match:
                value = self._math_expr(match.group(1))
                if value is not None and not value.free_symbols:
                    return value
        return None

    def _ode_grid_data(self, text: str):
        rhs = self._ode_rhs(text)
        initial = self._initial_condition(text)
        step = self._named_scalar(text, "h")
        target = self._target_x(text)
        if rhs is None or initial is None or step is None or target is None:
            return None
        current_x, current_y = initial
        if not self._strictly_less(0, step):
            return None
        count_value = self.sp.simplify((target - current_x) / step)
        if count_value.is_integer is not True:
            return None
        count = int(count_value)
        if not 1 <= count <= 10_000:
            return None
        x, y = self.sp.symbols("x y")
        if rhs.free_symbols - {x, y}:
            return None
        return rhs, x, y, current_x, current_y, step, target, count

    def _iterate_support(self, text: str, iterates) -> str:
        rendered = r",\;".join(
            rf"y_{index}={self.sp.latex(value)}" for index, value in iterates
        )
        return (
            rf"由题面初值逐步精确代入，得到 ${rendered}$。"
            if self._is_chinese(text) else
            rf"Exact stepwise substitution from the stated initial value gives ${rendered}$."
        )

    def _subinterval_count(self, text: str) -> Optional[int]:
        patterns = (
            r"(?:分成|分为|划分为|取|with|using|into)\s*"
            r"(\d+|一|二|两|三|四|五|六|七|八|九|十|"
            r"one|two|three|four|five|six|seven|eight|nine|ten)\s*"
            r"(?:个)?\s*(?:equal(?:ly)?\s+)?"
            r"(?:等分|子区间|小区间|段|subintervals?)",
            r"\bn\s*=\s*(\d+)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._small_integer(match.group(1))
        return None

    @staticmethod
    def _requests_exact_comparison(text: str) -> bool:
        return bool(re.search(
            r"与[^。；;\n]{0,50}精确(?:值|积分)[^。；;\n]{0,30}比较|"
            r"比较[^。；;\n]{0,50}精确(?:值|积分)|"
            r"(?:给出|求|计算|报告)[^。；;\n]{0,60}精确积分|"
            r"\bcompare\b[^.;\n]{0,80}\b(?:exact|true)\s+(?:value|integral)\b|"
            r"\b(?:give|find|compute|report)\b[^.;\n]{0,80}\bexact\s+integral\b",
            str(text or ""),
            re.IGNORECASE,
        ))

    def _interpolation_target(self, text: str):
        matches = tuple(re.finditer(
            r"(?:求|计算|evaluate|find)[^。.;\n]{0,30}?"
            r"(?:p|P|f)\s*\(\s*" + self._SCALAR + r"\s*\)",
            text,
            re.IGNORECASE,
        ))
        return self._math_expr(matches[-1].group(1)) if matches else None

    def _taylor_center(self, text: str):
        patterns = (
            r"(?:在|at|about|centered\s+at)\s*x\s*=\s*" + self._SCALAR,
            r"(?:在|about|centered\s+at)\s*" + self._SCALAR + r"\s*(?:处|附近)?",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._math_expr(match.group(1))
        for segment in self._math_segments(text):
            match = re.fullmatch(
                r"x\s*=\s*" + self._SCALAR,
                self._clean_math(segment),
                re.IGNORECASE,
            )
            if match:
                return self._math_expr(match.group(1))
        if re.search(r"Maclaurin|麦克劳林", text, re.IGNORECASE):
            return self.sp.S.Zero
        return None

    def _taylor_order(self, text: str) -> Optional[int]:
        patterns = (
            r"(\d+|一|二|两|三|四|五|六|七|八|九|十)\s*阶",
            r"(?:degree|order)\s*(\d+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._small_integer(match.group(1))
        return None

    def _indexed_value(self, text: str, name: str, index: int):
        match = re.search(
            rf"{re.escape(name)}\s*_?\s*\{{?\s*{index}\s*\}}?\s*=\s*{self._SCALAR}",
            text,
            re.IGNORECASE,
        )
        return self._math_expr(match.group(1)) if match else None

    def _named_scalar(self, text: str, name: str):
        match = re.search(rf"(?<![A-Za-z]){re.escape(name)}\s*=\s*{self._SCALAR}", text, re.IGNORECASE)
        return self._math_expr(match.group(1)) if match else None

    def _math_expr(self, source: str):
        value = self._clean_math(source)
        value = re.sub(r"\bln\b", "log", value, flags=re.IGNORECASE)
        value = re.sub(r"\be(?=\s*\^)", "E", value)
        value = re.sub(
            r"\\(sin|cos|tan|sinh|cosh|tanh|exp|log)\s*\{?\s*([A-Za-z])\s*\}?",
            r"\1(\2)", value,
        )
        value = re.sub(
            r"\b(sin|cos|tan|sinh|cosh|tanh|exp|log)\s+([A-Za-z])\b",
            r"\1(\2)", value, flags=re.IGNORECASE,
        )
        value = re.sub(r"(?<=[0-9A-Za-z)])(?=(?:sin|cos|tan|exp|log)\()", "*", value, flags=re.IGNORECASE)
        value = re.sub(r"(?<![A-Za-z])([A-Za-z])([xyz])(?![A-Za-z])", r"\1*\2", value)
        try:
            parsed = self.symbolic._parse(value)
            return self.sp.nsimplify(parsed, rational=True, full=False)
        except Exception:
            return None

    def _numeric_sign(self, value) -> Optional[int]:
        exact = self.sp.sign(self.sp.simplify(value))
        if exact in (-1, 0, 1):
            return int(exact)
        try:
            numeric = float(self.sp.N(value, 60))
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(numeric):
            return None
        threshold = 1e-40
        return 1 if numeric > threshold else -1 if numeric < -threshold else 0

    def _numeric_leq(self, left, right) -> bool:
        try:
            return float(self.sp.N(left - right, 60)) <= 0
        except (TypeError, ValueError, OverflowError):
            return False

    def _numeric_abs(self, value) -> float:
        try:
            numeric = float(self.sp.N(self.sp.Abs(value), 60))
            return numeric if math.isfinite(numeric) else math.inf
        except (TypeError, ValueError, OverflowError):
            return math.inf

    def _near_zero(self, value, threshold) -> bool:
        return self._numeric_leq(self.sp.Abs(value), self.sp.Abs(threshold))

    def _credible_root_residual(self, residual, tolerance, point) -> bool:
        """Reject false convergence where a tiny update is not near a root."""
        try:
            threshold = self.sp.sqrt(self.sp.Abs(tolerance)) * (
                1 + self.sp.Abs(point)
            )
            return self._numeric_leq(self.sp.Abs(residual), threshold)
        except Exception:
            return False

    def _continuous_on_closed_interval(self, expression, symbol, lower, upper) -> bool:
        """Certify the bisection continuity hypothesis or abstain."""
        try:
            interval = self.sp.Interval(lower, upper)
            domain = self.sp.calculus.util.continuous_domain(
                expression, symbol, self.sp.S.Reals
            )
            return bool(interval.is_subset(domain))
        except Exception:
            return False

    def _finite(self, value) -> bool:
        if value.has(self.sp.nan, self.sp.zoo, self.sp.oo, -self.sp.oo):
            return False
        try:
            return math.isfinite(float(self.sp.N(value, 30)))
        except (TypeError, ValueError, OverflowError):
            return False

    def _display_digits(self, tolerance) -> int:
        try:
            magnitude = max(0, math.ceil(-math.log10(float(tolerance))))
        except (TypeError, ValueError, OverflowError):
            magnitude = 8
        return min(30, max(8, magnitude + 3))

    def _decimal(self, value, digits: int) -> str:
        return str(self.sp.N(value, max(8, min(int(digits), 30))))

    @staticmethod
    def _math_segments(text: str) -> tuple[str, ...]:
        segments = []
        for match in re.finditer(
            r"\$(?P<dollar>[^$\n]+)\$|\\\((?P<paren>.*?)\\\)|\\\[(?P<bracket>.*?)\\\]",
            text,
            re.DOTALL,
        ):
            value = match.group("dollar") or match.group("paren") or match.group("bracket") or ""
            if value.strip():
                segments.append(value.strip())
        return tuple(segments)

    @staticmethod
    def _clean_math(value: str) -> str:
        text = str(value or "").strip().strip("$")
        text = text.replace(r"\left", "").replace(r"\right", "")
        text = text.replace(r"\,", " ").replace("−", "-")
        return text.strip()

    @staticmethod
    def _small_integer(token: str) -> Optional[int]:
        words = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                 "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
                 "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        value = str(token or "").strip().casefold()
        return int(value) if value.isdigit() else words.get(value)

    @staticmethod
    def _strictly_less(left, right) -> bool:
        comparison = left < right
        return bool(comparison is True or comparison == True)  # noqa: E712

    @staticmethod
    def _is_chinese(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def _result(
        problem: str,
        operation: str,
        result: str,
        result_kind: str,
        method: str,
        checks: tuple[str, ...],
        support: str,
        shapes: tuple[str, ...],
        requirements: tuple[str, ...],
    ) -> ToolResult:
        return make_parameterized_tool_result(
            problem=problem,
            operation=operation,
            result=result,
            result_kind=result_kind,
            method=method,
            whole=True,
            written_support=True,
            checks=checks,
            support=support,
            answer_shapes=shapes,
            requirements=requirements,
        )
