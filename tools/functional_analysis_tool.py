"""Conservative certificates for explicit elementary functional-analysis operators."""

from __future__ import annotations

import re
from typing import Optional

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, make_parameterized_tool_result


class FunctionalAnalysisTool:
    """Certify affine multiplication operators on a finite real interval."""

    def __init__(self) -> None:
        self.symbolic = SympyTool()
        self.sp = self.symbolic.sympy

    def results_for(self, problem: str) -> list[ToolResult]:
        text = str(problem or "").strip()
        if self.sp is None or not text:
            return []
        try:
            result = self._affine_multiplication_operator(text)
        except Exception:
            result = None
        return [result] if result is not None and result.verified else []

    def _affine_multiplication_operator(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"L\s*\^\s*\{?\s*2\s*\}?", text, re.I):
            return None
        if not re.search(r"乘法算子|multiplication\s+operator", text, re.I):
            return None
        if not re.search(r"算子范数|operator\s+norm|\\?\|\s*T\s*\\?\|", text, re.I):
            return None
        if not re.search(r"点谱|point\s+spectrum", text, re.I):
            return None
        clean = text.replace(r"\left", "").replace(r"\right", "")
        definition = re.search(
            r"\(\s*T\s*f\s*\)\s*\(\s*(?P<variable>[A-Za-z])\s*\)\s*=\s*"
            r"(?P<multiplier>[^$，,。;；\n]+?)\s*f\s*\(\s*(?P=variable)\s*\)",
            clean,
            re.I,
        )
        interval = re.search(
            r"L\s*\^\s*\{?\s*2\s*\}?\s*(?:\(\s*)?\[\s*"
            r"(?P<lower>[-+]?\d+(?:\s*/\s*\d+)?|[-+]?\.\d+)\s*[,，]\s*"
            r"(?P<upper>[-+]?\d+(?:\s*/\s*\d+)?|[-+]?\.\d+)\s*\](?:\s*\))?",
            clean,
            re.I,
        )
        if definition is None or interval is None:
            return None
        variable = self.sp.Symbol(definition.group("variable"))
        multiplier = self._expr(definition.group("multiplier"))
        lower = self._expr(interval.group("lower"))
        upper = self._expr(interval.group("upper"))
        if multiplier is None or lower is None or upper is None:
            return None
        if multiplier.free_symbols - {variable} or lower.free_symbols or upper.free_symbols:
            return None
        if self.sp.simplify(upper - lower).is_positive is not True:
            return None
        polynomial = self.sp.Poly(multiplier, variable)
        if polynomial.degree() != 1:
            return None
        slope = polynomial.coeff_monomial(variable)
        if slope.is_real is not True or slope.is_zero is not False:
            return None
        intercept = polynomial.coeff_monomial(1)
        if intercept.is_real is not True:
            return None

        endpoint_values = tuple(
            self.sp.simplify(multiplier.subs(variable, point))
            for point in (lower, upper)
        )
        if any(value.is_real is not True for value in endpoint_values):
            return None
        spectral_lower = min(endpoint_values)
        spectral_upper = max(endpoint_values)
        norm = max(tuple(self.sp.Abs(value) for value in endpoint_values))
        if self.sp.simplify(
            norm - max(self.sp.Abs(spectral_lower), self.sp.Abs(spectral_upper))
        ) != 0:
            return None

        norm_text = self.symbolic._format(norm)
        lower_text = self.symbolic._format(spectral_lower)
        upper_text = self.symbolic._format(spectral_upper)
        multiplier_text = self.symbolic._format(multiplier)
        zh = bool(re.search(r"[\u4e00-\u9fff]", text))
        result = (
            rf"乘子为 $m({variable})={multiplier_text}$。由 "
            rf"$\|Tf\|_2\le\|m\|_\infty\|f\|_2$ 得上界；取支撑逐渐集中到使 "
            rf"$|m|$ 达到最大值的端点附近的归一化示性函数可逼近取等，故 "
            rf"$\|T\|={norm_text}$。连续仿射乘子的本质值域为 "
            rf"$[{lower_text},{upper_text}]$，所以 $\sigma(T)=[{lower_text},{upper_text}]$。"
            rf"若 $Tf=\lambda f$，则 $(m({variable})-\lambda)f=0$；每个水平集至多为单点、"
            rf"Lebesgue 测度为零，故 $\sigma_p(T)=\varnothing$。"
            if zh else
            rf"The multiplier is $m({variable})={multiplier_text}$. The bound "
            rf"$\|Tf\|_2\le\|m\|_\infty\|f\|_2$, together with normalized indicators supported "
            rf"on shrinking neighborhoods of a maximizing endpoint, gives $\|T\|={norm_text}$. "
            rf"The essential range of the continuous affine multiplier is "
            rf"$[{lower_text},{upper_text}]$, hence $\sigma(T)=[{lower_text},{upper_text}]$. "
            rf"If $Tf=\lambda f$, then $(m({variable})-\lambda)f=0$; every level set is at most a "
            rf"singleton and has Lebesgue measure zero, so $\sigma_p(T)=\varnothing$."
        )
        return make_parameterized_tool_result(
            problem=text,
            operation="affine_multiplication_operator_spectrum",
            result=result,
            result_kind="operator_norm_spectrum_point_spectrum",
            method="essential_range_with_endpoint_concentration_and_level_sets",
            whole=True,
            written_support=True,
            checks=(
                "l2_interval_and_multiplier_parsed",
                "affine_nonconstant_multiplier_checked",
                "endpoint_essential_supremum_recomputed",
                "near_endpoint_approximate_equality_argument",
                "continuous_essential_range_recomputed",
                "level_sets_have_measure_zero",
            ),
            support=result,
            answer_shapes=("expression", "text", "set", "interval"),
            requirements=(
                "result_present",
                "numeric_result",
                "reasoning",
                "operator_norm",
                "operator_spectrum",
                "point_spectrum",
            ),
        )

    def _expr(self, value: str):
        try:
            return self.sp.simplify(self.symbolic._parse(value))
        except Exception:
            return None
