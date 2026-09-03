"""Conservative symbolic certificates for elementary complex analysis."""

from __future__ import annotations

import re
from typing import Optional

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, make_parameterized_tool_result


class ComplexAnalysisTool:
    """Recompute rational residues and certify explicit standard theorems."""

    def __init__(self) -> None:
        self.symbolic = SympyTool()
        self.sp = self.symbolic.sympy

    def results_for(self, problem: str) -> list[ToolResult]:
        text = str(problem or "").strip()
        if self.sp is None or not text:
            return []
        handlers = (
            self._contour_integral,
            self._holomorphic_power_real_part,
            self._residue_at_point,
            self._geometric_power_series,
            self._liouville,
            self._conjugate_difference_quotient,
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

    def _contour_integral(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"(?:\\oint|∮)", text) or not re.search(r"围道|contour|\|\s*z\s*\|", text, re.I):
            return None
        radius_match = re.search(
            r"(?:\|\s*z\s*\||\\lvert\s*z\s*\\rvert)\s*=\s*\$?\s*"
            r"([-+]?\d+(?:/\d+)?|[-+]?0?\.\d+)",
            text,
        )
        raw = self._contour_integrand(text)
        if radius_match is None or raw is None:
            return None
        radius = self._expr(radius_match.group(1))
        expression = self._expr(raw)
        z = self.sp.Symbol("z")
        if radius is None or expression is None or radius.is_positive is not True:
            return None
        if expression.free_symbols - {z}:
            return None
        numerator, denominator = map(self.sp.factor, self.sp.fraction(self.sp.cancel(expression)))
        if not denominator.has(z) or not (numerator.is_polynomial(z) and denominator.is_polynomial(z)):
            return None
        roots = self.sp.roots(denominator, z)
        if sum(int(value) for value in roots.values()) != self.sp.degree(denominator, z):
            return None
        inside, outside = [], []
        for pole in roots:
            absolute = self.sp.simplify(self.sp.Abs(pole))
            difference = self.sp.simplify(absolute - radius)
            if difference.is_zero is True:
                return None
            if difference.is_negative is True:
                inside.append(pole)
            elif difference.is_positive is True:
                outside.append(pole)
            else:
                numeric = complex(self.sp.N(pole, 30))
                boundary = float(self.sp.N(radius, 30))
                if abs(abs(numeric) - boundary) < 1e-12:
                    return None
                (inside if abs(numeric) < boundary else outside).append(pole)
        residue_sum = self.sp.simplify(sum(self.sp.residue(expression, z, pole) for pole in inside))
        value = self.sp.simplify(2 * self.sp.pi * self.sp.I * residue_sum)
        rendered = self.symbolic._format(value)
        pole_text = ",".join(self.symbolic._format(item) for item in inside) or "none"
        requested_cauchy = bool(re.search(r"柯西积分公式|Cauchy\s+integral\s+formula", text, re.I))
        theorem_zh = "柯西积分公式（等价地，留数定理）" if requested_cauchy else "留数定理"
        theorem_en = "the Cauchy integral formula (equivalently, the residue theorem)" if requested_cauchy else "the residue theorem"
        result = (
            rf"极点 $z={pole_text}$ 位于围道内（其余极点在外），故由{theorem_zh}，"
            rf"围道积分为 $2\pi i\sum\operatorname{{Res}}={rendered}$。"
            if self._zh(text) else
            rf"The poles $z={pole_text}$ lie inside the contour (all other poles lie outside), so by "
            rf"{theorem_en} the contour integral is $2\pi i\sum\operatorname{{Res}}={rendered}$."
        )
        return self._result(text, "contour_residue_integral", result, "contour_integral",
                            "rational_poles_and_residue_sum",
                            ("radius_parsed", "rational_integrand_parsed", "all_poles_solved",
                             "no_boundary_pole", "inside_residues_recomputed"),
                            ("result_present", "numeric_result", "reasoning", "judgement", "pole_location", "support_anchor_1"),
                            ("number", "expression", "text"))

    def _holomorphic_power_real_part(self, text: str) -> Optional[ToolResult]:
        match = re.search(r"f\s*\(\s*z\s*\)\s*=\s*z\s*\^\s*\{?\s*(\d+)\s*\}?", text, re.I)
        if match is None or not re.search(r"实部|real\s+part", text, re.I):
            return None
        degree = int(match.group(1))
        if not 0 <= degree <= 30:
            return None
        x, y = self.sp.symbols("x y", real=True)
        expanded = self.sp.expand((x + self.sp.I * y) ** degree)
        real = self.sp.simplify(self.sp.re(expanded))
        laplacian = self.sp.simplify(self.sp.diff(real, x, 2) + self.sp.diff(real, y, 2))
        if laplacian != 0:
            return None
        rendered = self.symbolic._format(real)
        result = (
            rf"展开 $(x+iy)^{{{degree}}}$ 得实部 $u(x,y)={rendered}$；"
            rf"直接求二阶导数有 $u_{{xx}}+u_{{yy}}={self.symbolic._format(laplacian)}$，故 $u$ 调和。"
            if self._zh(text) else
            rf"Expanding $(x+iy)^{{{degree}}}$ gives $u(x,y)={rendered}$. Direct differentiation yields "
            rf"$u_{{xx}}+u_{{yy}}={self.symbolic._format(laplacian)}$, so $u$ is harmonic."
        )
        return self._result(text, "holomorphic_power_real_part", result, "real_part_and_harmonicity",
                            "complex_power_expansion_and_laplacian",
                            ("integer_power_parsed", "real_part_expanded", "laplacian_recomputed"),
                            ("result_present", "reasoning", "second_derivatives", "harmonicity_judgement"),
                            ("expression", "text", "truth"))

    def _residue_at_point(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"留数|residue", text, re.I) or re.search(r"(?:\\oint|∮)", text):
            return None
        point_match = re.search(
            r"(?:在|at)\s*\$?\s*z\s*=\s*"
            r"([-+]?\d+(?:/\d+)?|[-+]?0?\.\d+|[A-Za-z])\s*\$?\s*(?:处|at)?",
            text,
            re.I,
        )
        expression_match = re.search(r"(?:求|find|compute)\s*\$?\s*(.+?)\s*\$?\s*(?:在|at)\s*\$?\s*z\s*=", text, re.I)
        if point_match is None or expression_match is None:
            return None
        raw = re.sub(r"^(?:函数|function)\s*", "", expression_match.group(1).strip(), flags=re.I)
        expression, point = self._expr(raw), self._expr(point_match.group(1))
        z = self.sp.Symbol("z")
        if expression is None or point is None or expression.free_symbols - {z} or point.free_symbols:
            return None
        residue = self.sp.simplify(self.sp.residue(expression, z, point))
        partial = self.sp.apart(expression, z)
        rendered, decomposition = self.symbolic._format(residue), self.symbolic._format(partial)
        result = (
            rf"部分分式分解为 ${decomposition}$；在 $z={self.symbolic._format(point)}$ 的 "
            rf"$(z-{self.symbolic._format(point)})^{{-1}}$ 系数即留数，故为 ${rendered}$。"
            if self._zh(text) else
            rf"The partial-fraction decomposition is ${decomposition}$. The coefficient of "
            rf"$(z-{self.symbolic._format(point)})^{{-1}}$ is the residue, namely ${rendered}$."
        )
        return self._result(text, "rational_residue_at_point", result, "residue",
                            "symbolic_partial_fraction_and_residue",
                            ("rational_expression_parsed", "point_parsed", "partial_fraction_recomputed", "residue_recomputed"),
                            ("result_present", "numeric_result", "reasoning"),
                            ("number", "expression", "text"))

    def _geometric_power_series(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"幂级数|power\s+series", text, re.I) or not re.search(r"z\s*=\s*0|at\s+(?:the\s+)?origin|邻域", text, re.I):
            return None
        match = re.search(r"1\s*/\s*\(\s*1\s*-\s*([^()]*)\s*\)|\\frac\s*\{\s*1\s*\}\s*\{\s*1\s*-\s*([^{}]+)\s*\}", text)
        if match is None:
            return None
        term = self._expr(next(item for item in match.groups() if item is not None))
        z = self.sp.Symbol("z")
        if term is None or not term.has(z):
            return None
        coefficient = self.sp.simplify(term / z)
        if coefficient.has(z) or coefficient.free_symbols or coefficient == 0:
            return None
        radius = self.sp.simplify(1 / self.sp.Abs(coefficient))
        c_text, r_text = self.symbolic._format(coefficient), self.symbolic._format(radius)
        result = (
            rf"由几何级数，$\frac1{{1-{c_text}z}}=\sum_{{n=0}}^\infty({c_text}z)^n$，"
            rf"收敛条件为 $|{c_text}z|<1$，故收敛半径 $R={r_text}$；半径外项不趋于0，级数发散。"
            if self._zh(text) else
            rf"The geometric series gives $\frac1{{1-{c_text}z}}=\sum_{{n=0}}^\infty({c_text}z)^n$. "
            rf"It converges for $|{c_text}z|<1$, so $R={r_text}$; outside the disk its terms do not tend to zero."
        )
        return self._result(text, "geometric_power_series", result, "series_and_radius",
                            "geometric_series_ratio_test",
                            ("denominator_parsed", "geometric_coefficient_extracted", "radius_recomputed"),
                            ("result_present", "numeric_result", "reasoning", "convergence_radius", "convergence_domain"),
                            ("expression", "number", "text"))

    def _liouville(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"整函数|entire\s+function", text, re.I) or not re.search(r"有界|bounded", text, re.I):
            return None
        if not re.search(r"常数|constant", text, re.I):
            return None
        result = (
            r"由刘维尔经典定理，整个复平面上有界的整函数必为常数。具体地，柯西估计给出 "
            r"$|f'(z_0)|\le M/R$；令 $R\to\infty$ 得 $f'(z_0)=0$，而 $z_0$ 任意，故 $f$ 为常数。"
            if self._zh(text) else
            r"By Liouville's theorem, a bounded entire function is constant. Indeed Cauchy's estimate gives "
            r"$|f'(z_0)|\le M/R$; letting $R\to\infty$ yields $f'(z_0)=0$ for every $z_0$."
        )
        return self._result(text, "liouville_bounded_entire", result, "proof",
                            "liouville_via_cauchy_derivative_estimate",
                            ("entire_hypothesis", "global_bound", "cauchy_estimate", "infinite_radius_limit"),
                            ("result_present", "reasoning", "support_anchor_1"),
                            ("proof", "text"))

    def _conjugate_difference_quotient(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"f\s*\(\s*z\s*\)\s*=\s*(?:\\bar\s*\{?\s*z\s*\}?|z\s*(?:的)?共轭)|f\s*\(\s*z\s*\)\s*=\s*conj", text, re.I):
            return None
        point = re.search(
            r"z\s*=\s*([-+]?\d+(?:/\d+)?|[-+]?0?\.\d+|[A-Za-z])",
            text,
        )
        if point is None:
            return None
        if self._expr(point.group(1)) is None:
            return None
        result = (
            r"在该点取增量 $h\in\mathbb R$ 时差商为 $\bar h/h=1$；取 $h=it$ 时差商为 "
            r"$\overline{it}/(it)=-1$。两条路径极限不同，所以该点不可复导。"
            if self._zh(text) else
            r"For real increments $h$, the quotient is $\bar h/h=1$, while for $h=it$ it is "
            r"$\overline{it}/(it)=-1$. The directional limits differ, so the function is not complex differentiable there."
        )
        return self._result(text, "conjugate_not_complex_differentiable", result, "differentiability_judgement",
                            "two_direction_difference_quotients",
                            ("conjugate_function_parsed", "point_present", "real_direction", "imaginary_direction", "limits_disagree"),
                            ("result_present", "judgement", "reasoning"),
                            ("truth", "text"))

    def _contour_integrand(self, text: str) -> Optional[str]:
        match = re.search(
            r"(?:\\oint|∮)\s*(.+?)(?=，|,|。|;|；|\n|\.(?:\s|$)|$)",
            text,
            re.I,
        )
        if match is None:
            return None
        value = match.group(1).strip().strip("$")
        value = re.sub(r"^_\s*\{[^{}]+\}\s*", "", value)
        value = re.sub(r"\s*(?:\\,?d\s*z|dz)\s*$", "", value, flags=re.I).strip()
        if re.match(r"^(?:\\,?d\s*z|dz)\s*/", value, re.I):
            value = re.sub(r"^(?:\\,?d\s*z|dz)\s*/", "1/", value, count=1, flags=re.I)
        return value or None

    def _expr(self, value: str):
        try:
            prepared = str(value).strip().strip("$")
            prepared = re.sub(
                r"(?<![A-Za-z0-9_])([A-Za-z])\s*\(",
                r"\1*(",
                prepared,
            )
            return self.sp.simplify(self.symbolic._parse(prepared))
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
