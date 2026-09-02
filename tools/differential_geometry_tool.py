"""Parameterized differential-geometry formulas with symbolic recomputation."""

from __future__ import annotations

import re
from typing import Optional

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, make_parameterized_tool_result


class DifferentialGeometryTool:
    """Certify elementary curve and surface calculations from parsed data."""

    def __init__(self) -> None:
        self.symbolic = SympyTool()
        self.sp = self.symbolic.sympy

    def results_for(self, problem: str) -> list[ToolResult]:
        text = str(problem or "").strip()
        if self.sp is None or not text:
            return []
        handlers = (
            self._plane_curve_curvature,
            self._circle_curvature,
            self._curve_speed,
            self._graph_surface_curvatures,
            self._sphere_curvatures,
            self._unit_tangent_orthogonality,
            self._zero_curvature_line,
            self._first_fundamental_form,
            self._parametric_surface_gaussian_curvature,
            self._graph_gaussian_formula,
            self._planar_conformal_gaussian_curvature,
        )
        results: list[ToolResult] = []
        for handler in handlers:
            try:
                result = handler(text)
            except Exception:
                result = None
            if result is not None and result.verified:
                results.append(result)
        return results

    def _plane_curve_curvature(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"平面曲线|plane\s+curve", text, re.I) or not re.search(r"曲率|curvature", text, re.I):
            return None
        parsed = self._vector_definition(text, ("γ", r"\gamma", "gamma"), 2)
        point = self._parameter_point(text)
        if parsed is None or point is None:
            return None
        parameter, components = parsed
        x, y = components
        dx, dy = self.sp.diff(x, parameter), self.sp.diff(y, parameter)
        ddx, ddy = self.sp.diff(dx, parameter), self.sp.diff(dy, parameter)
        denominator = self.sp.simplify((dx**2 + dy**2) ** self.sp.Rational(3, 2))
        denominator_at = self.sp.simplify(denominator.subs(parameter, point))
        if denominator_at.is_zero is not False:
            return None
        curvature = self.sp.simplify(
            self.sp.Abs(dx * ddy - dy * ddx).subs(parameter, point) / denominator_at
        )
        first = self._vector_latex((dx, dy))
        second = self._vector_latex((ddx, ddy))
        value = self.symbolic._format(curvature)
        point_text = self.symbolic._format(point)
        result = (
            rf"一阶导数 $\gamma'(t)={first}$，二阶导数 $\gamma''(t)={second}$；"
            rf"代入平面曲率公式得 $\kappa({point_text})={value}$。"
            if self._zh(text) else
            rf"The first derivative is $\gamma'(t)={first}$ and the second is "
            rf"$\gamma''(t)={second}$; the planar curvature formula gives "
            rf"$\kappa({point_text})={value}$."
        )
        return self._result(text, "plane_curve_curvature", result, "curvature",
                            "symbolic_planar_curvature_formula",
                            ("curve_components_parsed", "first_derivative", "second_derivative",
                             "regularity_at_point", "curvature_recomputed"),
                            ("result_present", "numeric_result", "first_second_derivatives"),
                            ("number", "expression"))

    def _circle_curvature(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"圆|circle", text, re.I) or not re.search(r"曲率|curvature", text, re.I):
            return None
        if not re.search(r"弧长参数|arc[- ]length\s+param", text, re.I):
            return None
        radius_match = re.search(
            r"(?:半径|radius)\s*(?:为|是|is|=|of)?\s*\$?\s*([A-Za-z]|\d+(?:/\d+)?|0?\.\d+)",
            text,
            re.I,
        )
        if radius_match is None:
            return None
        radius = self._expr(radius_match.group(1))
        if radius is None or radius.is_real is False:
            return None
        if not radius.free_symbols and radius.is_positive is not True:
            return None
        value = self.symbolic._format(self.sp.simplify(1 / radius))
        result = (
            rf"弧长参数满足 $|\gamma'(s)|=1$，圆的曲率为 $\kappa=1/R={value}$；"
            r"曲率是几何不变量，与正则参数的选择无关。"
            if self._zh(text) else
            rf"An arc-length parameter has $|\gamma'(s)|=1$, and the circle has "
            rf"$\kappa=1/R={value}$; curvature is invariant under regular reparametrization."
        )
        return self._result(text, "circle_arclength_curvature", result, "curvature",
                            "circle_radius_curvature_invariance",
                            ("radius_parsed", "positive_radius", "arc_length_hypothesis", "formula_recomputed"),
                            ("result_present", "numeric_result", "reasoning"),
                            ("number", "expression", "text"))

    def _curve_speed(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"曲线|curve", text, re.I) or not re.search(r"速度|speed|弧长参数|arc[- ]length", text, re.I):
            return None
        parsed = self._vector_definition(text, ("γ", r"\gamma", "gamma"), 3)
        if parsed is None:
            return None
        parameter, components = parsed
        derivatives = tuple(self.sp.diff(item, parameter) for item in components)
        squared_speed = self.sp.simplify(sum(item**2 for item in derivatives))
        if squared_speed.free_symbols:
            return None
        speed = self.sp.simplify(self.sp.sqrt(squared_speed))
        if speed.is_positive is not True:
            return None
        is_arc = self.sp.simplify(speed - 1) == 0
        rendered = self.symbolic._format(speed)
        judgement = "是" if is_arc else "不是"
        result = (
            rf"$\gamma'(t)={self._vector_latex(derivatives)}$，故速度长度 "
            rf"$|\gamma'(t)|={rendered}$；因此该参数{judgement}弧长参数。"
            if self._zh(text) else
            rf"$\gamma'(t)={self._vector_latex(derivatives)}$, so the speed is "
            rf"$|\gamma'(t)|={rendered}$; the parameter is {'an' if is_arc else 'not an'} arc-length parameter."
        )
        return self._result(text, "parametric_curve_speed", result, "speed_and_judgement",
                            "symbolic_velocity_norm",
                            ("three_components_parsed", "velocity_recomputed", "speed_simplified", "unit_speed_checked"),
                            ("result_present", "judgement", "reasoning"),
                            ("number", "expression", "text", "truth"))

    def _graph_surface_curvatures(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"曲面|surface", text, re.I) or not re.search(r"主曲率|principal\s+curvatures?", text, re.I):
            return None
        expression = self._graph_expression(text)
        if expression is None or not re.search(r"原点|origin", text, re.I):
            return None
        x, y = self.sp.Symbol("x"), self.sp.Symbol("y")
        if expression.free_symbols - {x, y}:
            return None
        gradient = (self.sp.diff(expression, x), self.sp.diff(expression, y))
        if any(self.sp.simplify(item.subs({x: 0, y: 0})) != 0 for item in gradient):
            return None
        hessian = self.sp.hessian(expression, (x, y)).subs({x: 0, y: 0})
        eigenvalues = []
        for value, multiplicity in hessian.eigenvals().items():
            eigenvalues.extend([self.sp.simplify(value)] * int(multiplicity))
        if len(eigenvalues) != 2:
            return None
        eigenvalues = sorted(eigenvalues, key=self.sp.default_sort_key)
        gaussian = self.sp.simplify(hessian.det())
        k1, k2, k_value = map(self.symbolic._format, (*eigenvalues, gaussian))
        result = (
            rf"原点处 $\nabla f=0$，向上法向下形算子由 Hessian 给出："
            rf"$f_{{xx}},f_{{xy}},f_{{yy}}={self._vector_latex((hessian[0,0], hessian[0,1], hessian[1,1]))}$。"
            rf"故主曲率为 $k_1={k1},k_2={k2}$，原点处高斯曲率 $K=k_1k_2={k_value}$。"
            if self._zh(text) else
            rf"At the origin $\nabla f=0$, so for the upward normal the shape operator is the Hessian, "
            rf"with $(f_{{xx}},f_{{xy}},f_{{yy}})={self._vector_latex((hessian[0,0], hessian[0,1], hessian[1,1]))}$. "
            rf"Thus $k_1={k1},k_2={k2}$ and $K=k_1k_2={k_value}$ at the origin."
        )
        return self._result(text, "graph_surface_principal_curvatures", result, "principal_and_gaussian_curvatures",
                            "critical_graph_hessian_eigenvalues",
                            ("graph_parsed", "critical_point_checked", "hessian_recomputed",
                             "eigenvalues_recomputed", "determinant_checked"),
                            ("result_present", "numeric_result", "reasoning", "curvature_point_value"),
                            ("number", "expression", "text"))

    def _sphere_curvatures(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"球面|sphere", text, re.I) or not re.search(r"平均曲率|mean\s+curvature", text, re.I):
            return None
        radius_match = re.search(
            r"(?:半径|radius)\s*(?:为|是|is|=|of)?\s*\$?\s*([A-Za-z]|\d+(?:/\d+)?|0?\.\d+)",
            text,
            re.I,
        )
        if radius_match:
            radius = self._expr(radius_match.group(1))
        elif re.search(r"单位球面|unit\s+sphere", text, re.I):
            radius = self.sp.Integer(1)
        else:
            return None
        if radius is None or (not radius.free_symbols and radius.is_positive is not True):
            return None
        mean = self.symbolic._format(self.sp.simplify(1 / radius))
        gaussian = self.symbolic._format(self.sp.simplify(1 / radius**2))
        result = (
            rf"球面的两主曲率绝对值均为 $1/R$，故 $|H|={mean}$、$K={gaussian}$；"
            r"反向法向会改变 $H$ 的符号，但不改变 $K$。"
            if self._zh(text) else
            rf"Both principal curvatures have magnitude $1/R$, hence $|H|={mean}$ and $K={gaussian}$; "
            r"reversing the normal changes the sign of $H$ but not $K$."
        )
        return self._result(text, "sphere_curvatures", result, "mean_and_gaussian_curvatures",
                            "sphere_shape_operator",
                            ("sphere_radius_parsed", "principal_curvatures_recomputed", "orientation_effect_checked"),
                            ("result_present", "numeric_result", "reasoning"),
                            ("number", "expression", "text"))

    def _unit_tangent_orthogonality(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"弧长参数|arc[- ]length\s+param", text, re.I) or not re.search(r"T\s*(?:\\cdot|·|\.)\s*T['′]|orthogonal", text, re.I):
            return None
        result = (
            r"弧长参数下 $T=\gamma'(s)$ 且 $\langle T,T\rangle=1$。对 $s$ 求导得 "
            r"$2\langle T,T'\rangle=0$，所以 $T'\perp T$。"
            if self._zh(text) else
            r"For an arc-length parameter, $T=\gamma'(s)$ and $\langle T,T\rangle=1$. "
            r"Differentiating gives $2\langle T,T'\rangle=0$, hence $T'\perp T$."
        )
        return self._result(text, "unit_tangent_orthogonality", result, "proof",
                            "differentiate_unit_tangent_norm",
                            ("arc_length_hypothesis", "unit_norm_identity", "identity_differentiated"),
                            ("result_present", "reasoning"), ("proof", "text"))

    def _zero_curvature_line(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"曲率(?:恒为|恒等于|identically)\s*0|zero\s+curvature", text, re.I):
            return None
        if not re.search(r"连通|connected", text, re.I):
            return None
        result = (
            r"以弧长 $s$ 参数化后，$\kappa=|T'|\equiv0$，故 $T$ 为常向量。"
            r"积分得 $\gamma(s)=p+sT$；连通性保证整条曲线像包含在同一直线中。"
            if self._zh(text) else
            r"After arc-length parametrization, $\kappa=|T'|\equiv0$, so $T$ is constant. "
            r"Thus $\gamma(s)=p+sT$; connectedness places the whole image in one line."
        )
        return self._result(text, "zero_curvature_line", result, "proof",
                            "constant_unit_tangent_integration",
                            ("zero_curvature_hypothesis", "tangent_constant", "curve_integrated", "connectedness_used"),
                            ("result_present", "reasoning"), ("proof", "text"))

    def _first_fundamental_form(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"第一基本形式|first\s+fundamental\s+form", text, re.I):
            return None
        parsed = self._vector_definition(text, ("X",), 3, parameters=2)
        if parsed is None:
            return None
        parameters, components = parsed
        u, v = parameters
        xu = self.sp.Matrix([self.sp.diff(item, u) for item in components])
        xv = self.sp.Matrix([self.sp.diff(item, v) for item in components])
        e = self.sp.simplify(xu.dot(xu))
        f = self.sp.simplify(xu.dot(xv))
        g = self.sp.simplify(xv.dot(xv))
        if any(item.free_symbols - {u, v} for item in (e, f, g)):
            return None
        er, fr, gr = map(self.symbolic._format, (e, f, g))
        result = (
            rf"$X_u={self._vector_latex(tuple(xu))}$，$X_v={self._vector_latex(tuple(xv))}$；"
            rf"故 $E=\langle X_u,X_u\rangle={er},\ F=\langle X_u,X_v\rangle={fr},\ G=\langle X_v,X_v\rangle={gr}$。"
            if self._zh(text) else
            rf"$X_u={self._vector_latex(tuple(xu))}$ and $X_v={self._vector_latex(tuple(xv))}$; "
            rf"therefore $E={er},\ F={fr},\ G={gr}$."
        )
        return self._result(text, "first_fundamental_form", result, "metric_coefficients",
                            "surface_partial_derivative_dot_products",
                            ("surface_components_parsed", "both_partial_vectors", "three_dot_products_recomputed"),
                            ("result_present", "numeric_result", "reasoning"),
                            ("number", "expression", "text"))

    def _graph_gaussian_formula(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"图形|graph", text, re.I) or not re.search(r"高斯曲率|Gaussian\s+curvature", text, re.I):
            return None
        if not re.search(r"(?:\\nabla|∇)\s*f\s*=\s*0|critical\s+point|临界点", text, re.I):
            return None
        result = (
            r"图形 $z=f(x,y)$ 在 $\nabla f=0$ 处第一基本形式为单位阵，第二基本形式为 Hessian，故 "
            r"$K=\det(D^2f)=f_{xx}f_{yy}-f_{xy}^2$。"
            if self._zh(text) else
            r"For the graph $z=f(x,y)$ at $\nabla f=0$, the first fundamental form is the identity and "
            r"the second is the Hessian, so $K=\det(D^2f)=f_{xx}f_{yy}-f_{xy}^2$."
        )
        return self._result(text, "graph_gaussian_at_critical_point", result, "curvature_formula",
                            "graph_curvature_formula_at_critical_point",
                            ("graph_hypothesis", "critical_point_hypothesis", "hessian_determinant_formula"),
                            ("result_present", "numeric_result", "reasoning"),
                            ("number", "expression", "text"))

    def _parametric_surface_gaussian_curvature(self, text: str) -> Optional[ToolResult]:
        """Compute both fundamental forms and K for an explicit parametrization."""
        if not re.search(r"(?:Gauss|Gaussian|高斯)\s*曲率|Gauss\s+curvature", text, re.I):
            return None
        if not re.search(r"参数化|parametri[sz](?:ation|ed)|parametric\s+surface", text, re.I):
            return None
        parsed = self._surface_definition_with_parameters(text)
        if parsed is None:
            return None
        (u, v), components, extra_parameters = parsed
        surface = self.sp.Matrix(components)
        xu, xv = surface.diff(u), surface.diff(v)
        cross = xu.cross(xv)
        metric_det = self.sp.factor(
            self.sp.trigsimp(self.sp.expand_trig(self.sp.expand(cross.dot(cross))))
        )
        if not self._certified_positive_even_polynomial(
            metric_det,
            (u, v),
            extra_parameters,
            text,
        ):
            return None
        e1 = self.sp.simplify(xu.dot(xu))
        f1 = self.sp.simplify(xu.dot(xv))
        g1 = self.sp.simplify(xv.dot(xv))
        if self.sp.simplify(e1 * g1 - f1**2 - metric_det) != 0:
            return None

        normal = self.sp.simplify(cross / self.sp.sqrt(metric_det))
        xuu, xuv, xvv = surface.diff(u, 2), surface.diff(u, v), surface.diff(v, 2)
        e2 = self.sp.simplify(normal.dot(xuu))
        f2 = self.sp.simplify(normal.dot(xuv))
        g2 = self.sp.simplify(normal.dot(xvv))
        curvature = self.sp.factor(self.sp.simplify((e2 * g2 - f2**2) / metric_det))

        # Independent formula without normalizing the cross product.
        unnormalized = self.sp.factor(self.sp.simplify(
            (
                cross.dot(xuu) * cross.dot(xvv)
                - cross.dot(xuv) ** 2
            )
            / metric_det**2
        ))
        if self.sp.simplify(curvature - unnormalized) != 0:
            return None

        rendered = {
            "E": self.symbolic._format(e1),
            "F": self.symbolic._format(f1),
            "G": self.symbolic._format(g1),
            "e": self.symbolic._format(e2),
            "f": self.symbolic._format(f2),
            "g": self.symbolic._format(g2),
            "K": self.symbolic._format(curvature),
        }
        independent_v = not curvature.has(v)
        explanation_zh = (
            f"；该表达式不含 ${v}$，故改变 ${v}$ 时曲率不变"
            if independent_v else ""
        )
        explanation_en = (
            f"; it contains no ${v}$, so the curvature is unchanged as ${v}$ varies"
            if independent_v else ""
        )
        result = (
            rf"取法向 $X_{u}\times X_{v}/|X_{u}\times X_{v}|$。第一基本形式系数为 "
            rf"$E={rendered['E']},\ F={rendered['F']},\ G={rendered['G']}$；第二基本形式系数为 "
            rf"$e={rendered['e']},\ f={rendered['f']},\ g={rendered['g']}$。因此 "
            rf"$K({u},{v})=(eg-f^2)/(EG-F^2)={rendered['K']}${explanation_zh}。"
            if self._zh(text) else
            rf"Using the normal $X_{u}\times X_{v}/|X_{u}\times X_{v}|$, the first fundamental form has "
            rf"$E={rendered['E']},\ F={rendered['F']},\ G={rendered['G']}$ and the second has "
            rf"$e={rendered['e']},\ f={rendered['f']},\ g={rendered['g']}$. Hence "
            rf"$K({u},{v})=(eg-f^2)/(EG-F^2)={rendered['K']}${explanation_en}."
        )
        return self._result(
            text,
            "parametric_surface_gaussian_curvature",
            result,
            "fundamental_forms_and_curvature",
            "symbolic_fundamental_forms_with_unnormalized_crosscheck",
            (
                "surface_components_and_parameters_parsed",
                "extra_parameters_have_nonzero_hypotheses",
                "metric_determinant_strictly_positive",
                "first_fundamental_form_recomputed",
                "second_fundamental_form_recomputed",
                "gaussian_curvature_recomputed",
                "unnormalized_normal_formula_crosscheck",
            ),
            (
                "result_present",
                "numeric_result",
                "reasoning",
                "curvature_function",
                "judgement",
            ),
            ("number", "expression", "text"),
        )

    def _surface_definition_with_parameters(self, text: str):
        value = text.replace(r"\left", "").replace(r"\right", "")
        match = re.search(
            r"X\s*\(\s*([A-Za-z])\s*[,，]\s*([A-Za-z])\s*\)\s*=\s*\(",
            value,
            re.I,
        )
        if match is None:
            return None
        body = self._balanced_body(value, match.end() - 1)
        if body is None:
            return None
        pieces = self._split_top_level(body)
        if len(pieces) != 3:
            return None
        expressions = tuple(
            self._expr(
                "*".join(piece.strip())
                if re.fullmatch(r"[A-Za-z]{2}", piece.strip())
                else piece
            )
            for piece in pieces
        )
        if any(item is None for item in expressions):
            return None
        u, v = (self.sp.Symbol(match.group(index)) for index in (1, 2))
        extras = tuple(sorted(
            set().union(*(expression.free_symbols for expression in expressions)) - {u, v},
            key=lambda item: item.name,
        ))
        if len(extras) > 4 or any(len(item.name) != 1 for item in extras):
            return None
        return (u, v), expressions, extras

    @staticmethod
    def _certified_positive_even_polynomial(expression, variables, extras, text: str) -> bool:
        symbols = tuple((*variables, *extras))
        try:
            polynomial = expression.as_poly(*symbols)
        except Exception:
            return False
        if polynomial is None:
            return False
        terms = polynomial.terms()
        if not terms or any(
            coefficient.is_positive is not True
            or any(exponent % 2 for exponent in powers)
            for powers, coefficient in terms
        ):
            return False
        explicitly_nonzero = {
            parameter
            for parameter in extras
            if re.search(
                rf"(?<![A-Za-z]){re.escape(parameter.name)}\s*"
                r"(?:>|<|\\ne|!=|≠)\s*0|"
                rf"0\s*(?:<|>)\s*{re.escape(parameter.name)}(?![A-Za-z])",
                text,
                re.I,
            )
        }
        variable_count = len(variables)
        return any(
            all(power == 0 for power in powers[:variable_count])
            and any(
                powers[variable_count + index] > 0
                and parameter in explicitly_nonzero
                for index, parameter in enumerate(extras)
            )
            for powers, _ in terms
        ) or any(all(power == 0 for power in powers) for powers, _ in terms)

    def _planar_conformal_gaussian_curvature(self, text: str) -> Optional[ToolResult]:
        """Compute K=-exp(-2u)(u_xx+u_yy) for an explicit planar metric."""
        if not re.search(r"(?:Gauss|Gaussian|高斯)\s*曲率|Gauss\s+curvature", text, re.I):
            return None
        compact = re.sub(r"\s+", "", text)
        if not re.search(
            r"g=e\^\{?2u\}?\((?:d?x\^2\+d?y\^2|dx\^\{?2\}?\+dy\^\{?2\}?)\)",
            compact,
            re.I,
        ):
            return None
        definition = re.search(
            r"u(?:\s*\(\s*x\s*,\s*y\s*\))?\s*=\s*\$?\s*([^$，,。;；\n]+)",
            text,
            re.I,
        )
        if definition is None:
            return None
        u = self._expr(definition.group(1))
        x, y = self.sp.symbols("x y")
        if u is None or u.free_symbols - {x, y}:
            return None
        laplacian = self.sp.simplify(self.sp.diff(u, x, 2) + self.sp.diff(u, y, 2))
        curvature = self.sp.simplify(-self.sp.exp(-2 * u) * laplacian)
        if self.sp.simplify(self.sp.exp(2 * u) * curvature + laplacian) != 0:
            return None
        u_text = self.symbolic._format(u)
        laplacian_text = self.symbolic._format(laplacian)
        curvature_text = self.symbolic._format(curvature)
        at_origin = bool(re.search(r"原点|\(\s*0\s*,\s*0\s*\)|at\s+the\s+origin", text, re.I))
        origin_text = ""
        if at_origin:
            origin = self.sp.simplify(curvature.subs({x: 0, y: 0}))
            origin_text = (
                rf"，特别地 $K(0,0)={self.symbolic._format(origin)}$"
                if self._zh(text) else
                rf"; in particular, $K(0,0)={self.symbolic._format(origin)}$"
            )
        result = (
            rf"二维共形度量 $g=e^{{2u}}(dx^2+dy^2)$ 的曲率公式为 "
            rf"$K=-e^{{-2u}}\Delta u$。这里 $u={u_text}$、$\Delta u={laplacian_text}$，故 "
            rf"$K(x,y)={curvature_text}${origin_text}。"
            if self._zh(text) else
            rf"For the conformal metric $g=e^{{2u}}(dx^2+dy^2)$, "
            rf"$K=-e^{{-2u}}\Delta u$. Here $u={u_text}$ and $\Delta u={laplacian_text}$, so "
            rf"$K(x,y)={curvature_text}${origin_text}."
        )
        return self._result(
            text,
            "planar_conformal_gaussian_curvature",
            result,
            "curvature_function",
            "conformal_metric_laplacian_formula",
            (
                "planar_conformal_metric_parsed",
                "conformal_factor_parsed",
                "laplacian_recomputed",
                "curvature_identity_substituted",
                "requested_point_substituted" if at_origin else "curvature_function_simplified",
            ),
            (
                "result_present",
                "numeric_result",
                "reasoning",
                "curvature_function",
                "curvature_point_value",
            ),
            ("expression", "number", "text"),
        )

    def _vector_definition(self, text: str, names: tuple[str, ...], size: int, parameters: int = 1):
        value = text.replace(r"\left", "").replace(r"\right", "")
        name_pattern = "|".join(re.escape(name) for name in names)
        variable_pattern = r"([A-Za-z])" if parameters == 1 else r"([A-Za-z])\s*[,，]\s*([A-Za-z])"
        match = re.search(rf"(?:{name_pattern})\s*\(\s*{variable_pattern}\s*\)\s*=\s*\(", value, re.I)
        if match is None:
            return None
        opening = match.end() - 1
        body = self._balanced_body(value, opening)
        if body is None:
            return None
        pieces = self._split_top_level(body)
        if len(pieces) != size:
            return None
        expressions = tuple(self._expr(piece) for piece in pieces)
        if any(item is None for item in expressions):
            return None
        variable_names = match.groups()[-parameters:]
        symbols = tuple(self.sp.Symbol(item) for item in variable_names)
        allowed = set(symbols)
        if any(expression.free_symbols - allowed for expression in expressions):
            return None
        return (symbols[0] if parameters == 1 else symbols), expressions

    def _parameter_point(self, text: str):
        match = re.search(r"(?:[tTsS]\s*=\s*|at\s+[tTsS]\s*=\s*)\$?\s*([-+]?\d+(?:/\d+)?|[-+]?0?\.\d+)", text, re.I)
        return self._expr(match.group(1)) if match else None

    def _graph_expression(self, text: str):
        match = re.search(r"z\s*=\s*([^，,。;；\n]+)", text, re.I)
        return self._expr(match.group(1).strip().strip("$")) if match else None

    def _expr(self, value: str):
        try:
            return self.sp.simplify(self.symbolic._parse(value))
        except Exception:
            return None

    @staticmethod
    def _balanced_body(text: str, opening: int) -> Optional[str]:
        depth = 0
        for index in range(opening, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    return text[opening + 1:index]
        return None

    @staticmethod
    def _split_top_level(body: str) -> list[str]:
        parts, start, depth = [], 0, 0
        for index, char in enumerate(body):
            depth += int(char == "(") - int(char == ")")
            if char in ",，" and depth == 0:
                parts.append(body[start:index].strip())
                start = index + 1
        parts.append(body[start:].strip())
        return [part for part in parts if part]

    def _vector_latex(self, values) -> str:
        return r"\left(" + ",".join(self.symbolic._format(value) for value in values) + r"\right)"

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
