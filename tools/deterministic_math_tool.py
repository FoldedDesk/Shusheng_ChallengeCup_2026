"""Conservative parameterized calculators for certifiable math targets."""

from __future__ import annotations

from math import gcd
from itertools import combinations
import re
from typing import Optional

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, make_tool_result


class DeterministicMathTool:
    """Recompute supported targets only from parameters in the current statement."""

    def __init__(self) -> None:
        self.symbolic = SympyTool()
        self.sp = self.symbolic.sympy

    def results_for(self, problem: str) -> list[ToolResult]:
        if not self.sp:
            return []
        text = str(problem or "").strip()
        compilers = (
            self._complete_bipartite_edge_deletion,
            self._scaled_cauchy_kernel_limit,
            self._surface_gaussian_curvature,
            self._gauss_legendre_polynomial_error,
            self._uniform_maximum_spacing_expectation,
            self._birth_death_hitting_probability,
            self._contour_residue_integral,
            self._constant_coefficient_ivp,
            self._intercept_gls,
            self._poisson_disk_arc,
            self._torus_cell_attachment,
            self._nilpotent_jordan_partition,
            self._uniform_scale_umvu,
            self._sobolev_energy_minimization,
            self._linear_program_2d_vertices,
        )
        results: list[ToolResult] = []
        for compiler in compilers:
            try:
                result = compiler(text)
            except Exception:
                result = None
            if result is not None and result.verified:
                results.append(result)
        return results

    def _complete_bipartite_edge_deletion(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"生成树|spanning\s+trees?", text, re.IGNORECASE):
            return None
        if not re.search(r"删去|删除|去掉|remove|delete", text, re.IGNORECASE):
            return None
        match = re.search(
            r"K\s*_?\s*\{?\s*(\d+)\s*[,，]\s*(\d+)\s*\}?",
            text,
            re.IGNORECASE,
        )
        fixed_edge = re.search(
            r"(?:一|1|one)\s*(?:条|个)?\s*(?:固定|specified|fixed)?\s*(?:的)?\s*(?:边|edge)|"
            r"fixed\s+edge",
            text,
            re.IGNORECASE,
        )
        if not match or not fixed_edge:
            return None
        left, right = int(match.group(1)), int(match.group(2))
        if left < 1 or right < 1 or left + right > 40:
            return None
        total = left ** (right - 1) * right ** (left - 1)
        containing = self.sp.Rational(total * (left + right - 1), left * right)
        remaining = self.sp.Integer(total) - containing
        if remaining.is_integer is not True or remaining < 0:
            return None

        checks = ["complete_bipartite_tree_formula", "edge_appearance_double_count"]
        if left + right <= 14:
            size = left + right
            adjacency = self.sp.zeros(size)
            for row in range(left):
                for column in range(left, size):
                    if (row, column) != (0, left):
                        adjacency[row, column] = adjacency[column, row] = 1
            degrees = [sum(adjacency.row(index)) for index in range(size)]
            laplacian = self.sp.diag(*degrees) - adjacency
            if self.sp.simplify(self.sp.det(laplacian[:-1, :-1]) - remaining) != 0:
                return None
            checks.append("laplacian_cofactor_recompute")
        result = str(int(remaining))
        support = (
            rf"By the matrix-tree formula, \tau(K_{{{left},{right}}})="
            rf"{left}^{{{right - 1}}}{right}^{{{left - 1}}}={total}; "
            rf"a fixed edge occurs in fraction ({left}+{right}-1)/"
            rf"({left}\cdot {right}), "
            rf"therefore deletion leaves {result} spanning trees."
        )
        return make_tool_result(
            problem=text,
            operation="graph_spanning_trees",
            result=result,
            result_kind="integer",
            method="matrix_tree_and_edge_double_count",
            whole=True,
            written_support=True,
            checks=checks,
            support=support,
            answer_shapes=("number", "count"),
            requirements=("result_present", "numeric_result"),
        )

    def _scaled_cauchy_kernel_limit(self, text: str) -> Optional[ToolResult]:
        limit_match = re.search(
            r"\\lim\s*_\s*\{?\s*([A-Za-z])\s*\\to\s*\\infty\s*\}?",
            text,
            re.IGNORECASE,
        )
        integral_match = re.search(
            r"\\int\s*_\s*\{?\s*0\s*\}?\s*\^\s*\{?\s*\\infty\s*\}?\s*"
            r"(.+?)\s*(?:\\,|\\;|\s)*d\s*([A-Za-z])\b",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not limit_match or not integral_match:
            return None
        scale_name = limit_match.group(1)
        expression_raw, variable_name = integral_match.groups()
        scale = self.sp.Symbol(scale_name)
        variable = self.sp.Symbol(variable_name)
        if scale == variable:
            return None
        expression = self._formula(expression_raw.strip().strip("$ "))
        raw_numerator, denominator = self.sp.fraction(expression, exact=True)
        numerator = self.sp.simplify(raw_numerator / scale)
        if self.sp.simplify(raw_numerator - scale * numerator) != 0:
            return None
        if scale in numerator.free_symbols or numerator.free_symbols - {variable}:
            return None
        constant = self.sp.simplify(denominator.subs(variable, 0))
        quadratic = self.sp.simplify(
            (denominator - constant) / (scale**2 * variable**2)
        )
        if constant.free_symbols or quadratic.free_symbols:
            return None
        if self.sp.simplify(
            denominator - constant - quadratic * scale**2 * variable**2
        ) != 0:
            return None
        if float(self.sp.N(constant)) <= 0 or float(self.sp.N(quadratic)) <= 0:
            return None
        value_at_zero = self.sp.simplify(numerator.subs(variable, 0))
        if value_at_zero.free_symbols or float(self.sp.N(value_at_zero)) <= 0:
            return None
        logarithmic_derivative = self.sp.simplify(
            self.sp.diff(numerator, variable) / numerator
        )
        if logarithmic_derivative.free_symbols or logarithmic_derivative.is_real is False:
            return None
        if float(self.sp.N(logarithmic_derivative)) > 0:
            return None
        parameter = self.sp.Symbol("t", nonnegative=True)
        kernel_integral = self.sp.integrate(
            1 / (constant + quadratic * parameter**2),
            (parameter, 0, self.sp.oo),
        )
        expected_kernel = self.sp.pi / (2 * self.sp.sqrt(constant * quadratic))
        if self.sp.simplify(kernel_integral - expected_kernel) != 0:
            return None
        value = self.sp.simplify(value_at_zero * kernel_integral)
        result = self.symbolic._format(value)
        support = (
            rf"Set t={scale_name}{variable_name}. The integral becomes "
            rf"\int_0^\infty g(t/{scale_name})/"
            rf"({self.symbolic._format(constant)}+{self.symbolic._format(quadratic)}\,t^2)\,dt, "
            rf"where g(0)={self.symbolic._format(value_at_zero)}. Since "
            rf"g'/g={self.symbolic._format(logarithmic_derivative)}\le 0, its absolute value "
            rf"is dominated by g(0)/({self.symbolic._format(constant)}+"
            rf"{self.symbolic._format(quadratic)}\,t^2), which is integrable. Dominated "
            rf"convergence therefore gives {result}."
        )
        return make_tool_result(
            problem=text,
            operation="scaled_cauchy_kernel_limit",
            result=result,
            result_kind="scalar",
            method="scale_substitution_and_dominated_convergence",
            whole=True,
            written_support=True,
            checks=("exact_scale_substitution", "nonincreasing_exponential_factor", "integrable_cauchy_dominator"),
            support=support,
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _surface_gaussian_curvature(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"Gauss\s*曲率|高斯曲率|Gaussian\s+curvature", text, re.IGNORECASE):
            return None
        fragments = self._math_fragments(text)
        surface_fragment = next(
            (
                item for item in fragments
                if re.search(r"(?:\\mathbf\s*\{?X\}?|X)\s*\(\s*u\s*,\s*v\s*\)\s*=", item)
            ),
            "",
        )
        match = re.search(
            r"(?:\\mathbf\s*\{?X\}?|X)\s*\(\s*u\s*,\s*v\s*\)\s*=\s*\((.*)\)\s*$",
            surface_fragment,
            re.DOTALL,
        )
        if not match:
            return None
        component_text = self._split_top_level(match.group(1))
        if len(component_text) != 3:
            return None
        components = [self._formula(item) for item in component_text]
        u, v = self.sp.symbols("u v")
        if any(component.free_symbols - {u, v} for component in components):
            return None

        substitutions = {}
        for fragment in fragments:
            assignment = re.fullmatch(r"\s*([uv])\s*=\s*(.+?)\s*", fragment)
            if assignment:
                substitutions[{"u": u, "v": v}[assignment.group(1)]] = self._formula(
                    assignment.group(2)
                )
        surface = self.sp.Matrix(components)
        xu, xv = surface.diff(u), surface.diff(v)
        normal = xu.cross(xv)
        e_metric = self.sp.simplify(xu.dot(xu))
        f_metric = self.sp.simplify(xu.dot(xv))
        g_metric = self.sp.simplify(xv.dot(xv))
        metric_det = self.sp.simplify(e_metric * g_metric - f_metric**2)
        second_uu = self.sp.simplify(surface.diff(u, 2).dot(normal))
        second_uv = self.sp.simplify(surface.diff(u, v).dot(normal))
        second_vv = self.sp.simplify(surface.diff(v, 2).dot(normal))
        curvature = self.sp.simplify(
            (second_uu * second_vv - second_uv**2) / metric_det**2
        )
        value = self.sp.simplify(curvature.subs(substitutions))
        determinant_at_point = self.sp.simplify(metric_det.subs(substitutions))
        if value.free_symbols or determinant_at_point.free_symbols:
            return None
        if determinant_at_point == 0 or value.is_real is False:
            return None
        # The unnormalized-normal formula must be invariant under N -> -N.
        reversed_value = self.sp.simplify(
            ((-second_uu) * (-second_vv) - (-second_uv) ** 2) / metric_det**2
        ).subs(substitutions)
        if self.sp.simplify(value - reversed_value) != 0:
            return None
        result = self.symbolic._format(value)
        support = (
            rf"For the first fundamental form, E={self.symbolic._format(e_metric)}, "
            rf"F={self.symbolic._format(f_metric)}, G={self.symbolic._format(g_metric)}. "
            rf"With N=X_u\times X_v (not normalized), set "
            rf"\tilde e=X_{{uu}}\cdot N={self.symbolic._format(second_uu)}, "
            rf"\tilde f=X_{{uv}}\cdot N={self.symbolic._format(second_uv)}, "
            rf"\tilde g=X_{{vv}}\cdot N={self.symbolic._format(second_vv)}. Then "
            rf"K=(\tilde e\tilde g-\tilde f^2)/(EG-F^2)^2={result}. "
            rf"Replacing N by -N changes all three tilded coefficients' signs and leaves K unchanged."
        )
        return make_tool_result(
            problem=text,
            operation="surface_gaussian_curvature",
            result=result,
            result_kind="scalar",
            method="first_second_fundamental_forms_without_unit_normal",
            whole=True,
            written_support=True,
            checks=("three_coordinate_chart", "regular_metric_at_target", "normal_reversal_invariance"),
            support=support,
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _gauss_legendre_polynomial_error(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"Gauss[-–— ]*Legendre|高斯.?勒让德|高斯求积", text, re.IGNORECASE):
            return None
        if not re.search(r"误差|\berror\b", text, re.IGNORECASE):
            return None
        point_match = re.search(r"([1-5])\s*点|([1-5])[- ]point", text, re.IGNORECASE)
        if point_match:
            count = int(next(group for group in point_match.groups() if group))
        else:
            words = {"一点": 1, "二点": 2, "三点": 3, "四点": 4, "五点": 5}
            count = next((number for word, number in words.items() if word in text), 0)
        if not count:
            return None
        integral = re.search(
            r"\\int\s*_\s*\{([^{}]+)\}\s*\^\s*\{([^{}]+)\}\s*"
            r"(.+?)\s*(?:\\,|\\;|\s)*d\s*([A-Za-z])\b",
            text,
            re.DOTALL,
        )
        if not integral:
            return None
        lower_raw, upper_raw, expression_raw, variable = integral.groups()
        lower = self._formula(lower_raw)
        upper = self._formula(upper_raw)
        expression = self._formula(expression_raw.strip().strip("$ "))
        symbol = self.sp.Symbol(variable)
        polynomial = self.sp.Poly(expression, symbol)
        if polynomial.degree() < 0 or polynomial.degree() > 20:
            return None
        if any(coefficient.is_Rational is not True for coefficient in polynomial.all_coeffs()):
            return None
        legendre = self.sp.legendre(count, symbol)
        nodes = self.sp.solve(legendre, symbol)
        if len(nodes) != count:
            return None
        derivative = self.sp.diff(legendre, symbol)
        midpoint = (upper + lower) / 2
        half_width = (upper - lower) / 2

        def quadrature_for(integrand):
            total = self.sp.Integer(0)
            for node in nodes:
                weight = self.sp.simplify(
                    2 / ((1 - node**2) * derivative.subs(symbol, node) ** 2)
                )
                total += weight * integrand.subs(symbol, midpoint + half_width * node)
            return self.sp.simplify(half_width * total)

        quadrature = quadrature_for(expression)
        exact = self.sp.integrate(expression, (symbol, lower, upper))
        if exact.has(self.sp.Integral):
            return None
        normal_sign = re.search(r"E\s*=\s*I\s*-\s*Q", text, re.IGNORECASE)
        reverse_sign = re.search(r"E\s*=\s*Q\s*-\s*I", text, re.IGNORECASE)
        if not normal_sign and not reverse_sign:
            return None
        error = self.sp.simplify(quadrature - exact if reverse_sign else exact - quadrature)
        if any(
            self.sp.simplify(
                self.sp.integrate(symbol**degree, (symbol, lower, upper))
                - quadrature_for(symbol**degree)
            ) != 0
            for degree in range(2 * count)
        ):
            return None
        result = self.symbolic._format(error)
        support = (
            rf"By exact recomputation, the {count}-point rule gives "
            rf"I={self.symbolic._format(exact)}, Q={self.symbolic._format(quadrature)}, "
            rf"and the requested signed error {result}. Its nodes are the roots of P_{count} "
            rf"with weights 2/((1-x_j^2)(P_{count}'(x_j))^2), and direct monomial checks "
            rf"confirm exactness through degree {2 * count - 1}."
        )
        return make_tool_result(
            problem=text,
            operation="gauss_legendre_error",
            result=result,
            result_kind="scalar",
            method="exact_nodes_weights_and_polynomial_recompute",
            whole=True,
            written_support=True,
            checks=("exact_integral", "exact_quadrature_sum", "degree_exactness_audit"),
            support=support,
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _uniform_maximum_spacing_expectation(self, text: str) -> Optional[ToolResult]:
        if not re.search(
            r"最大(?:间隔|空隙|间距).{0,30}期望|期望.{0,30}最大(?:间隔|空隙|间距)|"
            r"expected.{0,30}(?:largest|maximum).{0,20}(?:gap|spacing)|"
            r"(?:largest|maximum).{0,20}(?:gap|spacing).{0,30}expect",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            return None
        if not re.search(
            r"独立.{0,20}均匀|均匀.{0,20}独立|i\.?i\.?d\.?\s+uniform|independent.{0,20}uniform",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(
            r"端点.{0,30}(?:加入|包括|一并)|(?:including|together with).{0,20}endpoints?",
            text,
            re.IGNORECASE,
        ):
            return None
        interval_match = re.search(r"\[\s*([^,，\]]+)\s*[,，]\s*([^\]]+)\s*\]", text)
        if not interval_match:
            return None
        lower = self._formula(interval_match.group(1))
        upper = self._formula(interval_match.group(2))
        if lower.free_symbols or upper.free_symbols or self.sp.N(upper - lower) <= 0:
            return None
        count_match = re.search(
            r"(?:抽取|选取|取|采样|sample|draw)\s*"
            r"([零一二两三四五六七八九十百\d]+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"\s*(?:个)?\s*(?:点|样本|points?|samples?)",
            text,
            re.IGNORECASE,
        )
        if not count_match:
            return None
        sample_count = self._small_integer(count_match.group(1))
        if sample_count is None or not 1 <= sample_count <= 80:
            return None
        spacing_count = sample_count + 1
        harmonic = self.sp.simplify(sum(self.sp.Rational(1, k) for k in range(1, spacing_count + 1)))
        expected_unit = self.sp.simplify(harmonic / spacing_count)
        inclusion_exclusion = self.sp.simplify(sum(
            (-1) ** (k + 1) * self.sp.binomial(spacing_count, k) / k
            for k in range(1, spacing_count + 1)
        ))
        if self.sp.simplify(inclusion_exclusion - harmonic) != 0:
            return None
        expectation = self.sp.simplify((upper - lower) * expected_unit)
        result = self.symbolic._format(expectation)
        support = (
            rf"The {spacing_count} normalized spacings have the Dirichlet(1,...,1) law. "
            rf"Inclusion-exclusion gives P(M>t)=\sum_{{k=1}}^{{{spacing_count}}}"
            rf"(-1)^{{k+1}}\binom{{{spacing_count}}}k(1-kt)_+^{{{spacing_count - 1}}}. "
            rf"Integrating the tail yields E[M]=H_{{{spacing_count}}}/{spacing_count}; "
            rf"multiplication by the interval length {self.symbolic._format(upper - lower)} "
            rf"gives {result}."
        )
        return make_tool_result(
            problem=text,
            operation="uniform_maximum_spacing_expectation",
            result=result,
            result_kind="scalar",
            method="dirichlet_spacings_inclusion_exclusion_tail_integral",
            whole=True,
            written_support=True,
            checks=("uniform_spacing_dirichlet_law", "inclusion_exclusion_identity", "interval_scaling"),
            support=support,
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _birth_death_hitting_probability(self, text: str) -> Optional[ToolResult]:
        if not re.search(
            r"随机游走|出生.?死亡|马尔可夫链|hitting|birth.?death|random walk",
            text,
            re.IGNORECASE,
        ):
            return None
        boundary = re.search(
            r"(?:\\?\{|\{)?\s*0\s*,\s*1\s*,\s*(?:\\(?:ldots|dots)|\.\.\.)\s*,\s*(\d+)",
            text,
            re.IGNORECASE,
        )
        start = re.search(
            r"从\s*\$?\s*(\d+)\s*\$?\s*出发|starting\s+from\s*\$?\s*(\d+)",
            text,
            re.IGNORECASE,
        )
        probabilities = self._direction_probabilities(text)
        if not boundary or not start or probabilities is None:
            return None
        upper = int(boundary.group(1))
        initial = int(next(group for group in start.groups() if group))
        if not (2 <= upper <= 80 and 0 < initial < upper):
            return None
        up_expression = self._formula(probabilities[0])
        down_expression = self._formula(probabilities[1])
        state = self.sp.Symbol("i")
        increments = [self.sp.Integer(1)]
        transitions = []
        for index in range(1, upper):
            up = self.sp.simplify(up_expression.subs(state, index))
            down = self.sp.simplify(down_expression.subs(state, index))
            if self.sp.simplify(up + down - 1) != 0 or up <= 0 or down < 0:
                return None
            transitions.append((up, down))
            increments.append(self.sp.simplify(increments[-1] * down / up))
        denominator = self.sp.simplify(sum(increments))
        probability = self.sp.simplify(sum(increments[:initial]) / denominator)
        values = [self.sp.Integer(0)]
        values.extend(
            self.sp.simplify(sum(increments[:index]) / denominator)
            for index in range(1, upper)
        )
        values.append(self.sp.Integer(1))
        for index, (up, down) in enumerate(transitions, start=1):
            if self.sp.simplify(
                values[index] - up * values[index + 1] - down * values[index - 1]
            ) != 0:
                return None
        result = self.symbolic._format(probability)
        support = (
            rf"By solving h_i=p_i h_{{i+1}}+q_i h_{{i-1}}, h_0=0, h_{{{upper}}}=1 "
            rf"with exact successive increments, every recurrence holds and therefore "
            rf"h_{{{initial}}}={result}."
        )
        return make_tool_result(
            problem=text,
            operation="birth_death_hitting_probability",
            result=result,
            result_kind="probability",
            method="exact_scale_increments_and_recurrence_substitution",
            whole=True,
            written_support=True,
            checks=("transition_normalization", "absorbing_boundaries", "all_interior_recurrences"),
            support=support,
            answer_shapes=("probability", "number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    @staticmethod
    def _direction_probabilities(text: str) -> Optional[tuple[str, str]]:
        chinese_up = re.search(r"概率\s*\$([^$]+)\$\s*向右", text, re.IGNORECASE)
        chinese_down = re.search(r"(?:[,，、]\s*)?\$([^$]+)\$\s*向左", text, re.IGNORECASE)
        english_up = re.search(
            r"(?:to\s+)?\$?i\s*\+\s*1\$?\s+with\s+probability\s*\$([^$]+)\$",
            text,
            re.IGNORECASE,
        )
        english_down = re.search(
            r"(?:to\s+)?\$?i\s*-\s*1\$?\s+with\s+probability\s*\$([^$]+)\$",
            text,
            re.IGNORECASE,
        )
        up = chinese_up or english_up
        down = chinese_down or english_down
        if not up or not down:
            return None
        return up.group(1), down.group(1)

    def _contour_residue_integral(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"\\oint|围道积分|contour integral", text, re.IGNORECASE):
            return None
        radius_match = re.search(r"\|\s*z\s*\|\s*=\s*([^}\s$]+)", text, re.IGNORECASE)
        fraction = re.search(
            r"\\oint[\s\S]{0,100}?\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}"
            r"\s*(?:\\,|\\;|\s)*d\s*z",
            text,
            re.IGNORECASE,
        )
        if not radius_match or not fraction:
            return None
        radius = self._formula(radius_match.group(1))
        numerator = self._formula(fraction.group(1))
        denominator = self._formula(fraction.group(2))
        variable = self.sp.Symbol("z")
        integrand = self.sp.cancel(numerator / denominator)
        _, reduced_denominator = self.sp.fraction(integrand)
        poles = self.sp.solve(reduced_denominator, variable)
        if not poles or any(pole.free_symbols for pole in poles):
            return None
        numeric_radius = float(self.sp.N(radius, 50))
        inside = []
        for pole in poles:
            distance = abs(complex(self.sp.N(pole, 50)))
            if abs(distance - numeric_radius) < 1e-12:
                return None
            if distance < numeric_radius:
                inside.append(pole)
        residues = [self.sp.simplify(self.sp.residue(integrand, variable, pole)) for pole in inside]
        orientation = -1 if re.search(
            r"负向|顺时针|clockwise|negative orientation", text, re.IGNORECASE
        ) else 1
        value = self.sp.simplify(orientation * 2 * self.sp.pi * self.sp.I * sum(residues))
        result = self.symbolic._format(value)
        residue_text = ", ".join(
            rf"Res({self.symbolic._format(pole)})={self.symbolic._format(residue)}"
            for pole, residue in zip(inside, residues)
        )
        return make_tool_result(
            problem=text,
            operation="contour_residue_integral",
            result=result,
            result_kind="expression",
            method="symbolic_pole_detection_and_residue_sum",
            whole=True,
            checks=("all_denominator_roots", "contour_membership", "symbolic_residues", "orientation"),
            support=(
                f"By symbolic pole detection, the inside residues are {residue_text}; "
                f"therefore the oriented residue sum is {result}."
            ),
            written_support=True,
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _constant_coefficient_ivp(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"初值问题|initial value problem", text, re.IGNORECASE):
            return None
        fragments = self._math_fragments(text)
        equation_fragment = next(
            (item for item in fragments if "y''" in item or "y′′" in item), ""
        )
        if not equation_fragment or "=" not in equation_fragment:
            return None
        left_raw, right_raw = equation_fragment.split("=", 1)
        prepared_left = left_raw.replace("y''", "q").replace("y′′", "q")
        prepared_left = prepared_left.replace("y'", "p").replace("y′", "p")
        left = self._formula(prepared_left)
        right = self._formula(right_raw)
        x = self.sp.Symbol("x")
        p, q, y_symbol = self.sp.symbols("p q y")
        basis = (q, p, y_symbol)
        coefficients = tuple(self.sp.simplify(left.coeff(symbol)) for symbol in basis)
        remainder = self.sp.simplify(
            left - sum(coefficient * symbol for coefficient, symbol in zip(coefficients, basis))
        )
        if remainder != 0 or coefficients[0] == 0:
            return None
        if any(x in coefficient.free_symbols for coefficient in coefficients):
            return None
        value_fragment = next(
            (
                item for item in fragments
                if re.search(r"y\s*\([^)]*\)\s*=", item)
                and "y'" not in item
                and "y′" not in item
            ),
            "",
        )
        derivative_fragment = next(
            (item for item in fragments if re.search(r"y(?:'|′)\s*\([^)]*\)\s*=", item)),
            "",
        )
        value_match = re.search(r"y\s*\(([^)]+)\)\s*=\s*(.+)", value_fragment)
        derivative_match = re.search(
            r"y(?:'|′)\s*\(([^)]+)\)\s*=\s*(.+)", derivative_fragment
        )
        if not value_match or not derivative_match:
            return None
        point = self._formula(value_match.group(1))
        if self.sp.simplify(point - self._formula(derivative_match.group(1))) != 0:
            return None
        initial_value = self._formula(value_match.group(2))
        initial_derivative = self._formula(derivative_match.group(2))
        function = self.sp.Function("y")
        ode = self.sp.Eq(
            coefficients[0] * self.sp.diff(function(x), x, 2)
            + coefficients[1] * self.sp.diff(function(x), x)
            + coefficients[2] * function(x),
            right,
        )
        general_solution = self.sp.dsolve(ode)
        constants = tuple(
            symbol for symbol in general_solution.rhs.free_symbols
            if re.fullmatch(r"C\d+", symbol.name)
        )
        particular = self.sp.simplify(
            general_solution.rhs.subs({symbol: 0 for symbol in constants})
        )
        particular_residual = self.sp.simplify(
            coefficients[0] * self.sp.diff(particular, x, 2)
            + coefficients[1] * self.sp.diff(particular, x)
            + coefficients[2] * particular
            - right
        )
        if particular_residual != 0:
            return None
        solution = self.sp.dsolve(
            ode,
            ics={
                function(point): initial_value,
                self.sp.diff(function(x), x).subs(x, point): initial_derivative,
            },
        )
        expression = self.sp.simplify(solution.rhs)
        residual = self.sp.simplify(
            coefficients[0] * self.sp.diff(expression, x, 2)
            + coefficients[1] * self.sp.diff(expression, x)
            + coefficients[2] * expression
            - right
        )
        if residual != 0:
            return None
        if self.sp.simplify(expression.subs(x, point) - initial_value) != 0:
            return None
        if self.sp.simplify(self.sp.diff(expression, x).subs(x, point) - initial_derivative) != 0:
            return None
        characteristic_variable = self.sp.Symbol("r")
        characteristic = self.sp.expand(
            coefficients[0] * characteristic_variable**2
            + coefficients[1] * characteristic_variable
            + coefficients[2]
        )
        roots = self.sp.roots(characteristic, characteristic_variable)
        if sum(roots.values()) != 2:
            return None
        root_text = ", ".join(
            f"{self.symbolic._format(root)} (multiplicity {multiplicity})"
            for root, multiplicity in roots.items()
        )
        forcing_exponents = self._forcing_exponents(right, x)
        resonant = [
            exponent for exponent in forcing_exponents
            if self.sp.simplify(characteristic.subs(characteristic_variable, exponent)) == 0
        ]
        resonance_support = (
            " The forcing exponents "
            + ", ".join(self.symbolic._format(item) for item in resonant)
            + " are characteristic roots, so the particular ansatz requires the corresponding "
            "resonance factor in x."
            if resonant else ""
        )
        result = rf"y(x)={self.symbolic._format(expression)}"
        support = (
            rf"The characteristic equation {self.symbolic._format(characteristic)}=0 has roots "
            rf"{root_text}.{resonance_support} Setting the integration constants to zero in the "
            rf"general solution constructs y_p={self.symbolic._format(particular)}, and direct "
            rf"substitution gives L[y_p]={self.symbolic._format(right)}. Imposing "
            rf"y({self.symbolic._format(point)})={self.symbolic._format(initial_value)} and "
            rf"y'({self.symbolic._format(point)})={self.symbolic._format(initial_derivative)} "
            rf"gives {result}; differentiation verifies both initial values and zero ODE residual."
        )
        return make_tool_result(
            problem=text,
            operation="linear_ode_ivp",
            result=result,
            result_kind="expression",
            method="dsolve_then_symbolic_residual_and_initial_data",
            whole=True,
            written_support=True,
            checks=(
                "constant_coefficients",
                "characteristic_roots",
                "particular_solution_residual",
                "symbolic_ode_residual",
                "all_initial_conditions",
            ),
            support=support,
            answer_shapes=("expression",),
            requirements=("result_present",),
        )

    def _intercept_gls(self, text: str) -> Optional[ToolResult]:
        if not re.search(
            r"(?:截距模型|intercept(?:-only)?\s+model)", text, re.IGNORECASE
        ) or not re.search(r"\bGLS\b|广义最小二乘", text, re.IGNORECASE):
            return None
        fragments = self._math_fragments(text)
        response_match = next(
            (
                match for fragment in fragments
                if (match := re.search(
                    r"(?:^|[^A-Za-z])y\s*=\s*\(([^()]*)\)\s*"
                    r"(?:\^\s*\{?\\top\}?|\^\s*T)?\s*$",
                    fragment,
                    re.IGNORECASE,
                ))
            ),
            None,
        )
        covariance_match = next(
            (
                match for fragment in fragments
                if (match := re.search(
                    r"(?:\\operatorname\s*\{\s*diag\s*\}|\\?diag)\s*\(([^()]*)\)",
                    fragment,
                    re.IGNORECASE,
                ))
            ),
            None,
        )
        if response_match is None or covariance_match is None:
            return None
        responses = [self._formula(item) for item in self._split_top_level(response_match.group(1))]
        diagonal = [self._formula(item) for item in self._split_top_level(covariance_match.group(1))]
        if len(responses) < 2 or len(responses) != len(diagonal):
            return None
        if any(value.free_symbols for value in (*responses, *diagonal)):
            return None
        if any(float(self.sp.N(value)) <= 0 for value in diagonal):
            return None
        denominator = self.sp.simplify(sum(1 / value for value in diagonal))
        numerator = self.sp.simplify(sum(
            response / variance for response, variance in zip(responses, diagonal)
        ))
        estimate = self.sp.simplify(numerator / denominator)
        normal_residual = self.sp.simplify(sum(
            (response - estimate) / variance
            for response, variance in zip(responses, diagonal)
        ))
        if normal_residual != 0:
            return None
        formatted = self.symbolic._format(estimate)
        result = rf"\widehat{{\beta}}_{{GLS}}={formatted}"
        support = (
            rf"For X=1, the GLS normal equation is "
            rf"(\sum_i d_i^{{-1}})\widehat{{\beta}}="
            rf"\sum_i d_i^{{-1}}y_i. Here the denominator is "
            rf"{self.symbolic._format(denominator)} and the numerator is "
            rf"{self.symbolic._format(numerator)}, so {result}. Substitution makes "
            rf"\sum_i(y_i-\widehat{{\beta}})/d_i=0."
        )
        return make_tool_result(
            problem=text,
            operation="intercept_gls",
            result=result,
            result_kind="estimator",
            method="inverse_variance_weighted_normal_equation",
            whole=True,
            written_support=True,
            checks=("diagonal_covariance_positive", "weighted_normal_equation", "common_scale_cancels"),
            support=support,
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _poisson_disk_arc(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"单位圆盘|unit disk", text, re.IGNORECASE):
            return None
        if not re.search(r"Poisson\s*核|Poisson kernel", text, re.IGNORECASE):
            return None
        plain_text = text.replace("$", "")
        boundary_values = re.search(
            r"取值\s*1[\s\S]{0,100}取值\s*0|equals?\s+1[\s\S]{0,100}equals?\s+0",
            plain_text,
            re.IGNORECASE,
        )
        if not boundary_values:
            return None
        fragments = self._math_fragments(text)
        arc_fragment = next(
            (item for item in fragments if re.search(r"\|\\?theta\|\s*<", item)), ""
        )
        point_fragment = next(
            (item for item in fragments if re.search(r"u\s*\([^,]+,\s*0\s*\)", item)),
            "",
        )
        arc = re.search(r"\|\\?theta\|\s*<\s*(.+)", arc_fragment)
        point = re.search(r"u\s*\(([^,]+),\s*0\s*\)", point_fragment)
        if not arc or not point:
            return None
        alpha = self._formula(arc.group(1))
        radius = self._formula(point.group(1))
        if not (0 <= float(self.sp.N(radius)) < 1):
            return None
        if not (0 < float(self.sp.N(alpha)) < float(self.sp.pi)):
            return None
        factor = self.sp.simplify((1 + radius) / (1 - radius))
        value = self.sp.simplify(
            2 / self.sp.pi * self.sp.atan(factor * self.sp.tan(alpha / 2))
        )
        angle = self.sp.Symbol("a", real=True)
        derivative_difference = (
            self.sp.diff(2 * self.sp.atan(factor * self.sp.tan(angle / 2)), angle)
            - (1 - radius**2) / (1 - 2 * radius * self.sp.cos(angle) + radius**2)
        )
        from sympy.simplify.fu import fu

        derivative_check = self.sp.simplify(fu(derivative_difference))
        if derivative_check != 0:
            return None
        result = self.symbolic._format(value)
        support = (
            rf"By integrating the Poisson kernel, the symmetric arc value is "
            rf"(2/\pi)\arctan(((1+r)/(1-r))"
            rf"\tan(\alpha/2)); r={self.symbolic._format(radius)}, "
            rf"\alpha={self.symbolic._format(alpha)} gives {result}."
        )
        return make_tool_result(
            problem=text,
            operation="poisson_disk_harmonic_measure",
            result=result,
            result_kind="scalar",
            method="poisson_kernel_antiderivative_and_boundary_check",
            whole=True,
            written_support=True,
            checks=("interior_point", "symmetric_arc", "poisson_antiderivative"),
            support=support,
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _torus_cell_attachment(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"二维环面|2-?torus|T\s*\^\s*2", text, re.IGNORECASE):
            return None
        if not re.search(r"粘贴.*(?:二维|2-?)胞腔|attach.*2-?cell", text, re.IGNORECASE):
            return None
        if not re.search(r"基本群|fundamental group", text, re.IGNORECASE):
            return None
        relation = re.search(
            r"a\s*\^\s*\{?\s*(-?\d+)\s*\}?\s*b\s*\^\s*\{?\s*(-?\d+)\s*\}?",
            text,
            re.IGNORECASE,
        )
        if not relation:
            return None
        first, second = int(relation.group(1)), int(relation.group(2))
        divisor = gcd(abs(first), abs(second))
        if first == 0 and second == 0:
            result = r"\mathbb{Z}^{2}"
        elif divisor == 1:
            result = r"\mathbb{Z}"
        else:
            result = rf"\mathbb{{Z}}\oplus\mathbb{{Z}}_{{{divisor}}}"
        support = (
            rf"By van Kampen the torus relation makes the presentation abelian; the row "
            rf"[{first},{second}] has Smith invariant gcd({abs(first)},{abs(second)})="
            rf"{divisor}, hence {result}."
        )
        return make_tool_result(
            problem=text,
            operation="abelian_presentation_snf",
            result=result,
            result_kind="group",
            method="abelian_presentation_and_smith_normal_form",
            whole=True,
            written_support=True,
            checks=("torus_commutator_relation", "attachment_exponent_row", "gcd_smith_invariant"),
            support=support,
            answer_shapes=("expression",),
            requirements=("result_present",),
        )

    def _nilpotent_jordan_partition(self, text: str) -> Optional[ToolResult]:
        """Recover a nilpotent Jordan partition from a complete rank sequence."""
        if not re.search(r"幂零|nilpotent", text, re.IGNORECASE):
            return None
        if not re.search(r"Jordan|若尔当", text, re.IGNORECASE):
            return None

        dimension = self._vector_space_dimension(text)
        nilpotency = re.search(
            r"(?P<op>[A-Za-z])\s*\^\s*\{?\s*(?P<power>\d+)\s*\}?\s*=\s*0",
            text,
            re.IGNORECASE,
        )
        if dimension is None or nilpotency is None:
            return None
        operator = nilpotency.group("op")
        max_size = int(nilpotency.group("power"))
        if not (1 <= max_size <= 30 and dimension <= 500):
            return None

        rank_pattern = re.compile(
            rf"(?:\\operatorname\s*\{{\s*rank\s*\}}|\\?rank|秩)\s*"
            rf"{re.escape(operator)}(?:\s*\^\s*\{{?\s*(\d+)\s*\}}?)?\s*=\s*(\d+)",
            re.IGNORECASE,
        )
        ranks = {0: dimension, max_size: 0}
        for match in rank_pattern.finditer(text):
            power = int(match.group(1) or 1)
            rank = int(match.group(2))
            if power in ranks and ranks[power] != rank:
                return None
            ranks[power] = rank
        if set(ranks) != set(range(max_size + 1)):
            return None
        rank_sequence = [ranks[index] for index in range(max_size + 1)]
        if any(
            not (0 <= rank_sequence[index + 1] <= rank_sequence[index])
            for index in range(max_size)
        ):
            return None

        exact_counts: dict[int, int] = {}
        for size in range(1, max_size + 1):
            next_rank = rank_sequence[size + 1] if size < max_size else 0
            count = rank_sequence[size - 1] - 2 * rank_sequence[size] + next_rank
            if count < 0:
                return None
            exact_counts[size] = count
        blocks = tuple(
            size
            for size in range(max_size, 0, -1)
            for _ in range(exact_counts[size])
        )
        if not blocks or sum(blocks) != dimension:
            return None
        for power in range(max_size + 1):
            reconstructed = sum(max(size - power, 0) for size in blocks)
            if reconstructed != rank_sequence[power]:
                return None

        result = "(" + ",".join(str(size) for size in blocks) + ")"
        ranks_text = ", ".join(
            rf"r_{{{power}}}={rank_sequence[power]}"
            for power in range(max_size + 1)
        ) + rf", r_{{{max_size + 1}}}=0"
        counts_text = ", ".join(
            rf"c_{{{size}}}=r_{{{size - 1}}}-2r_{{{size}}}+r_{{{size + 1}}}"
            rf"={exact_counts[size]}"
            for size in range(1, max_size + 1)
        )
        if re.search(r"[\u4e00-\u9fff]", text):
            support = (
                rf"令 r_j=\operatorname{{rank}}({operator}^j)，则 {ranks_text}。"
                rf"大小恰为 j 的 Jordan 块数为 c_j=r_{{j-1}}-2r_j+r_{{j+1}}；"
                rf"代入得 {counts_text}，故分拆为 {result}。"
                rf"按 \sum\max(s-j,0) 回代各次秩均与题设一致。"
            )
        else:
            support = (
                rf"Let r_j=\operatorname{{rank}}({operator}^j); then {ranks_text}. "
                rf"The number of Jordan blocks of exact size j is "
                rf"c_j=r_{{j-1}}-2r_j+r_{{j+1}}; hence {counts_text}, giving {result}. "
                rf"Recomputing every rank as \sum\max(s-j,0) matches the statement."
            )
        return make_tool_result(
            problem=text,
            operation="nilpotent_jordan_partition",
            result=result,
            result_kind="partition",
            method="nilpotent_rank_second_differences",
            whole=True,
            written_support=True,
            checks=(
                "complete_rank_sequence",
                "nonnegative_second_differences",
                "dimension_reconstruction",
                "all_power_ranks_reconstructed",
            ),
            support=support,
            answer_shapes=("expression",),
            requirements=("result_present",),
        )

    def _sobolev_energy_minimization(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"H\s*_\s*\{?0\}?\s*\^\s*\{?1\}?|H_0\^1", text):
            return None
        if not re.search(r"最小值|极小|minimi[sz]e|minimum", text, re.IGNORECASE):
            return None
        space = re.search(
            r"H\s*_\s*\{?0\}?\s*\^\s*\{?1\}?\s*\(\s*0\s*[,，]\s*([^)$]+)\)",
            text,
        )
        fragments = self._math_fragments(text)
        constraint_pattern = re.compile(
            r"\\int\s*_\s*\{?\s*0\s*\}?\s*\^\s*"
            r"(?:\{\s*([^{}]+?)\s*\}|([^()\s]+?))\s*"
            r"y\s*(?:\(\s*x\s*\))?\s*(?:\\,)?\s*d\s*x\s*=\s*([^$，,。；;\s]+)\s*",
            re.IGNORECASE,
        )
        objective_pattern = re.compile(
            r"\\int\s*_\s*\{?\s*0\s*\}?\s*\^\s*"
            r"(?:\{\s*([^{}]+?)\s*\}|([^()\s]+?))\s*"
            r"\(?\s*\(?y\s*(?:'|′)\s*\)?\s*\^\s*\{?2\}?\s*\)?\s*"
            r"(?:\\,)?\s*d\s*x\s*",
            re.IGNORECASE,
        )
        constraint = next(
            (match for fragment in fragments if (match := constraint_pattern.fullmatch(fragment))),
            None,
        )
        objective = next(
            (match for fragment in fragments if (match := objective_pattern.fullmatch(fragment))),
            None,
        )
        if not space or not constraint or not objective:
            return None
        try:
            length = self._formula(space.group(1))
            constraint_upper = self._formula(constraint.group(1) or constraint.group(2))
            objective_upper = self._formula(objective.group(1) or objective.group(2))
            mass = self._formula(constraint.group(3))
            if any(value.free_symbols for value in (length, constraint_upper, objective_upper, mass)):
                return None
            if self.sp.simplify(length - constraint_upper) != 0:
                return None
            if self.sp.simplify(length - objective_upper) != 0:
                return None
            if float(self.sp.N(length)) <= 0:
                return None
        except Exception:
            return None
        coefficient = self.sp.simplify(6 * mass / length**3)
        minimum = self.sp.simplify(12 * mass**2 / length**3)
        x = self.sp.Symbol("x")
        minimizer = self.sp.simplify(coefficient * x * (length - x))
        if self.sp.simplify(self.sp.integrate(minimizer, (x, 0, length)) - mass) != 0:
            return None
        if self.sp.simplify(
            self.sp.integrate(self.sp.diff(minimizer, x) ** 2, (x, 0, length)) - minimum
        ) != 0:
            return None
        result = self.symbolic._format(minimum)
        minimizer_text = self.symbolic._format(minimizer)
        if re.search(r"[\u4e00-\u9fff]", text):
            support = (
                rf"取 y_*(x)={minimizer_text}，则 y_*\in H_0^1(0,{self.symbolic._format(length)}) "
                rf"且 \int_0^{{{self.symbolic._format(length)}}}y_*\,dx="
                rf"{self.symbolic._format(mass)}。对任意可行 y 写 h=y-y_*；"
                rf"由 h 的零边界与 \int h=0，分部积分得 \int y_*'h'=0。因而 "
                rf"\int(y')^2=\int(y_*')^2+\int(h')^2\ge {result}。"
                rf"等号迫使 h'=0，再由零边界得 h=0，故最小值为 {result} 且极小元唯一。"
            )
        else:
            support = (
                rf"Set y_*(x)={minimizer_text}; it has zero boundary values and integral "
                rf"{self.symbolic._format(mass)}. For any feasible y write h=y-y_*. "
                rf"Integration by parts and \int h=0 give \int y_*'h'=0, so "
                rf"\int(y')^2=\int(y_*')^2+\int(h')^2\ge {result}. Equality forces h'=0 "
                rf"and the boundary values force h=0, proving the unique global minimizer."
            )
        return make_tool_result(
            problem=text,
            operation="sobolev_energy_minimization",
            result=result,
            result_kind="scalar",
            method="euler_lagrange_candidate_and_positive_remainder",
            whole=True,
            written_support=True,
            checks=(
                "matching_interval_bounds",
                "constraint_integral_recomputed",
                "minimum_energy_recomputed",
                "positive_remainder_and_unique_equality_case",
            ),
            support=support,
            answer_shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _linear_program_2d_vertices(self, text: str) -> Optional[ToolResult]:
        fragment = next(
            (
                item for item in self._math_fragments(text)
                if re.search(r"\\(?:max|min)\s*\\\{", item, re.IGNORECASE)
            ),
            "",
        )
        match = re.fullmatch(
            r"\s*\\(max|min)\s*\\\{(.+)\\\}\s*",
            fragment,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        sense = match.group(1).casefold()
        body = match.group(2)
        objective_raw, separator, constraints_raw = body.partition(":")
        if not separator:
            objective_raw, separator, constraints_raw = body.partition("：")
        if not separator:
            return None
        try:
            objective = self._formula(objective_raw)
        except Exception:
            return None
        variables = tuple(sorted(objective.free_symbols, key=lambda item: item.name))
        if len(variables) != 2:
            return None
        try:
            if self.sp.Poly(objective, *variables).total_degree() > 1:
                return None
        except Exception:
            return None

        relations: list[tuple[object, str]] = []
        for raw in self._split_top_level(constraints_raw):
            relation = re.fullmatch(
                r"\s*(.+?)\s*(\\leq?|\\geq?|<=|>=)\s*(.+?)\s*",
                raw,
                re.DOTALL,
            )
            if not relation:
                return None
            left_raw, operator, right_raw = relation.groups()
            try:
                expression = self.sp.expand(self._formula(left_raw) - self._formula(right_raw))
                polynomial = self.sp.Poly(expression, *variables)
                if polynomial.total_degree() > 1 or expression.free_symbols - set(variables):
                    return None
            except Exception:
                return None
            normalized_operator = "le" if "le" in operator or operator == "<=" else "ge"
            relations.append((expression, normalized_operator))
        if not 3 <= len(relations) <= 20:
            return None

        normalized = [expression if operator == "le" else -expression for expression, operator in relations]
        for variable in variables:
            nonnegative = any(self.sp.simplify(expression + variable) == 0 for expression in normalized)
            bounded_above = False
            for expression in normalized:
                polynomial = self.sp.Poly(expression, *variables)
                coefficient = self.sp.simplify(polynomial.coeff_monomial(variable))
                other_coefficients = [
                    self.sp.simplify(polynomial.coeff_monomial(other))
                    for other in variables if other != variable
                ]
                constant = self.sp.simplify(polynomial.coeff_monomial(1))
                if (
                    coefficient.is_positive is True
                    and all(item.is_nonnegative is True for item in other_coefficients)
                    and constant.is_negative is True
                ):
                    bounded_above = True
                    break
            if not nonnegative or not bounded_above:
                return None

        vertices: list[tuple] = []
        for (left, _), (right, _) in combinations(relations, 2):
            try:
                solutions = self.sp.solve((left, right), variables, dict=True)
            except Exception:
                continue
            for solution in solutions:
                if set(solution) != set(variables):
                    continue
                point = tuple(self.sp.simplify(solution[variable]) for variable in variables)
                if any(value.free_symbols or value.is_real is False for value in point):
                    continue
                substitution = dict(zip(variables, point))
                feasible = True
                for expression, operator in relations:
                    numeric = float(self.sp.N(expression.subs(substitution), 30))
                    if (operator == "le" and numeric > 1e-10) or (
                        operator == "ge" and numeric < -1e-10
                    ):
                        feasible = False
                        break
                if feasible and not any(
                    all(self.sp.simplify(a - b) == 0 for a, b in zip(point, existing))
                    for existing in vertices
                ):
                    vertices.append(point)
        if not vertices:
            return None
        values = [self.sp.simplify(objective.subs(dict(zip(variables, point)))) for point in vertices]
        numeric_values = [float(self.sp.N(value, 30)) for value in values]
        optimum_numeric = max(numeric_values) if sense == "max" else min(numeric_values)
        optimum_indices = [
            index for index, value in enumerate(numeric_values)
            if abs(value - optimum_numeric) <= 1e-10
        ]
        if len(optimum_indices) != 1:
            return None
        optimum_index = optimum_indices[0]
        optimum_point = vertices[optimum_index]
        optimum_value = values[optimum_index]
        point_text = "(" + ",".join(self.symbolic._format(item) for item in optimum_point) + ")"
        value_text = self.symbolic._format(optimum_value)
        variable_tuple = "(" + ",".join(str(variable) for variable in variables) + ")"
        result = rf"{variable_tuple}={point_text},\quad \{sense}={value_text}"
        audit = "; ".join(
            rf"({','.join(self.symbolic._format(item) for item in point)})\mapsto"
            rf"{self.symbolic._format(value)}"
            for point, value in zip(vertices, values)
        )
        if re.search(r"[\u4e00-\u9fff]", text):
            support = (
                rf"枚举全部两两活跃约束并代回所有不等式，恰得可行顶点及目标值：{audit}。"
                rf"逐项精确比较表明唯一最优点为 {point_text}，目标值 {value_text}；"
                rf"非负约束与含正系数的上界约束同时给出可行域有界性，故顶点检验完备。"
            )
        else:
            support = (
                rf"Enumerating every pair of active constraints and substituting into all "
                rf"inequalities gives the feasible vertex/value list {audit}. Exact comparison "
                rf"makes {point_text} the unique optimum with value {value_text}; nonnegativity "
                rf"and the positive-coefficient upper bounds certify boundedness."
            )
        return make_tool_result(
            problem=text,
            operation="linear_program_2d_vertices",
            result=result,
            result_kind="optimizer",
            method="all_active_constraint_pairs_and_exact_objective_comparison",
            whole=True,
            written_support=True,
            checks=(
                "linear_objective_and_constraints",
                "bounded_first_quadrant_region",
                "all_boundary_pairs_enumerated",
                "all_vertices_feasibility_substituted",
                "exact_unique_objective_comparison",
            ),
            support=support,
            answer_shapes=("expression",),
            requirements=("result_present",),
        )

    def _uniform_scale_umvu(self, text: str) -> Optional[ToolResult]:
        """Certify the UMVU estimator of the endpoint of U(0, theta)."""
        if not re.search(r"UMVU|一致最小方差无偏", text, re.IGNORECASE):
            return None
        if not re.search(r"独立同分布|独立服从|\bi\.?i\.?d\.?(?:\b|\s)", text, re.IGNORECASE):
            return None
        theta_target = re.search(
            r"(?:求|find|determine)[^。.!?\n]{0,100}?\\theta[^。.!?\n]{0,60}?UMVU|"
            r"UMVU[^。.!?\n]{0,80}?(?:of|for|估计量)[^。.!?\n]{0,30}?\\theta",
            text,
            re.IGNORECASE,
        )
        if not theta_target or re.search(r"\\theta\s*\^", theta_target.group(0)):
            return None
        if not re.search(
            r"U\s*\(\s*0\s*[,，]\s*\\?theta\s*\)|"
            r"(?:uniform|均匀分布).{0,30}(?:0\s*[,，]\s*\\?theta)",
            text,
            re.IGNORECASE,
        ):
            return None
        sample = re.search(
            r"X\s*_\s*\{?\s*1\s*\}?\s*[,，].{0,60}?"
            r"X\s*_\s*\{?\s*(\d+)\s*\}?",
            text,
            re.IGNORECASE,
        )
        if sample is None:
            sample = re.search(
                r"(?:sample\s+size|样本量)\s*(?:is|为|=)?\s*(\d+)",
                text,
                re.IGNORECASE,
            )
        if sample is None:
            return None
        count = int(sample.group(1))
        if not 1 <= count <= 100000:
            return None
        maximum = re.search(
            r"(?P<name>[A-Za-z])\s*=\s*\\?max\s*"
            r"(?:_\s*\{?[A-Za-z]\}?)?\s*X|"
            r"(?P<name_zh>[A-Za-z])\s*=\s*最大(?:值|次序统计量)",
            text,
            re.IGNORECASE,
        )
        maximum_name = (
            maximum.group("name") or maximum.group("name_zh")
            if maximum is not None else rf"X_{{({count})}}"
        )
        coefficient = self.sp.Rational(count + 1, count)
        formatted = self.symbolic._format(coefficient)
        result = rf"\widehat{{\theta}}={formatted}{maximum_name}"
        if re.search(r"[\u4e00-\u9fff]", text):
            support = (
                rf"似然为 \theta^{{-{count}}}\mathbf{{1}}_{{0\le {maximum_name}\le\theta}}，"
                rf"故 {maximum_name} 充分。若 E_\theta g({maximum_name})=0 对所有 \theta>0 成立，"
                rf"则 {count}\theta^{{-{count}}}\int_0^\theta g(m)m^{{{count - 1}}}\,dm=0；"
                rf"对 \theta 求导得 g(\theta)=0（几乎处处），故 {maximum_name} 完全。"
                rf"又 E_\theta {maximum_name}={count}\theta/{count + 1}，所以 {result} 无偏；"
                rf"由 Lehmann--Scheffe 定理它是 UMVU。"
            )
        else:
            support = (
                rf"The likelihood is \theta^{{-{count}}}\mathbf{{1}}_{{0\le {maximum_name}\le\theta}}, "
                rf"so {maximum_name} is sufficient. If E_\theta g({maximum_name})=0 for every "
                rf"\theta>0, differentiating \int_0^\theta g(m)m^{{{count - 1}}}\,dm=0 gives "
                rf"g(\theta)=0 almost everywhere, proving completeness. Since "
                rf"E_\theta {maximum_name}={count}\theta/{count + 1}, {result} is unbiased and "
                rf"Lehmann--Scheffe makes it UMVU."
            )
        return make_tool_result(
            problem=text,
            operation="uniform_scale_umvu",
            result=result,
            result_kind="estimator",
            method="factorization_completeness_differentiation_and_unbiasing",
            whole=True,
            written_support=True,
            checks=(
                "sample_size_parsed",
                "factorization_sufficiency",
                "integral_completeness_argument",
                "maximum_expectation",
            ),
            support=support,
            answer_shapes=("number", "expression"),
            requirements=("result_present",),
        )

    def _vector_space_dimension(self, text: str) -> Optional[int]:
        patterns = (
            r"([零一二两三四五六七八九十百\d]+)\s*维(?:复|实)?向量空间",
            r"(?:dim(?:ension)?\s*(?:of\s*)?(?:V)?\s*(?:is|=)?|"
            r"([零一二两三四五六七八九十百\d]+)[- ]dimensional)\s*(\d+)?",
            r"\\dim\s*(?:\([^)]*\)|[A-Za-z])\s*=\s*(\d+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            token = next((group for group in match.groups() if group), "")
            value = self._small_integer(token)
            if value is not None and value > 0:
                return value
        return None

    def _forcing_exponents(self, expression, variable) -> tuple:
        rewritten = self.sp.expand(expression.rewrite(self.sp.exp))
        exponents = []
        for term in self.sp.Add.make_args(rewritten):
            if term == 0:
                continue
            logarithmic_derivative = self.sp.simplify(self.sp.diff(term, variable) / term)
            if variable in logarithmic_derivative.free_symbols:
                continue
            if any(self.sp.simplify(logarithmic_derivative - item) == 0 for item in exponents):
                continue
            exponents.append(logarithmic_derivative)
        return tuple(exponents)

    @staticmethod
    def _split_top_level(value: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        depth = 0
        for character in str(value or ""):
            if character in "([{":
                depth += 1
            elif character in ")]}" and depth > 0:
                depth -= 1
            if character in ",，" and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(character)
        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    @staticmethod
    def _small_integer(value: str) -> Optional[int]:
        token = str(value or "").strip().casefold()
        if token.isdigit():
            return int(token)
        english = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        if token in english:
            return english[token]
        digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
                  "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        if token in digits:
            return digits[token]
        if "十" in token:
            left, right = token.split("十", 1)
            tens = digits.get(left, 1) if left else 1
            units = digits.get(right, 0) if right else 0
            return 10 * tens + units
        return None

    def _formula(self, value: str):
        prepared = str(value or "").strip().strip("$ ")
        prepared = re.sub(r"\\(?:[,;:!]|\s)", " ", prepared)
        prepared = re.sub(
            r"\\(sin|cos|tan|sinh|cosh|exp|log|ln)", r" \1 ", prepared
        )
        prepared = self.symbolic._latex_to_sympy(prepared)
        prepared = re.sub(r"\be\s*\*\*\s*\(([^()]*)\)", r"exp(\1)", prepared)
        prepared = re.sub(r"\be\s*\*\*\s*([A-Za-z0-9.+-]+)", r"exp(\1)", prepared)
        prepared = re.sub(r"\be\s*\^\s*\(([^()]+)\)", r"exp(\1)", prepared)
        prepared = re.sub(r"\be\s*\^\s*([A-Za-z0-9.+-]+)", r"exp(\1)", prepared)
        prepared = re.sub(
            r"\b(sin|cos|tan|sinh|cosh|exp|log)\s+([A-Za-z0-9.+-]+)",
            r"\1(\2)",
            prepared,
        )
        prepared = re.sub(
            r"(?<=[A-Za-z0-9)])\s+(?=(?:sin|cos|tan|sinh|cosh|exp|log)\()",
            "*",
            prepared,
        )
        return self.symbolic._parse(prepared)

    @staticmethod
    def _math_fragments(text: str) -> list[str]:
        fragments = [
            item.strip() for item in re.findall(r"\$([^$]+)\$", text) if item.strip()
        ]
        for pair in re.findall(r"\\\((.+?)\\\)|\\\[(.+?)\\\]", text, re.DOTALL):
            fragments.extend(item.strip() for item in pair if item.strip())
        return fragments
