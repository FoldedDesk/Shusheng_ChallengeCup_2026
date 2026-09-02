"""Conservative certificates for explicit elementary analysis families."""

from __future__ import annotations

import re
from typing import Optional

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, make_parameterized_tool_result


class AnalysisTool:
    """Certify parameterized series only when their full scope is explicit."""

    def __init__(self) -> None:
        self.symbolic = SympyTool()
        self.sp = self.symbolic.sympy

    def results_for(self, problem: str) -> list[ToolResult]:
        text = str(problem or "").strip()
        if self.sp is None or not text:
            return []
        try:
            result = self._logarithmic_power_series(text)
        except Exception:
            result = None
        return [result] if result is not None and result.verified else []

    def _logarithmic_power_series(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"函数项级数|power\s+series|function\s+series", text, re.I):
            return None
        if not re.search(r"一致收敛|uniform(?:ly)?\s+conver", text, re.I):
            return None
        if not re.search(r"和函数|求和|sum\s+function|find\s+(?:its|the)\s+sum", text, re.I):
            return None

        summand = self._summand(text)
        if summand is None:
            return None
        base, variable = summand
        polynomial = self.sp.Poly(base, variable)
        if polynomial.degree() != 1 or polynomial.coeff_monomial(1) != 0:
            return None
        scale = self.sp.simplify(polynomial.coeff_monomial(variable))
        if scale.free_symbols or scale.is_positive is not True:
            return None
        radius = self.sp.simplify(1 / scale)

        intervals = self._intervals(text)
        local = next((item for item in intervals if item[1] == "]" and item[0].is_Symbol), None)
        global_scope = next((item for item in intervals if item[1] == ")" and not item[0].free_symbols), None)
        if local is None or global_scope is None:
            return None
        local_upper = local[0]
        global_upper = self.sp.simplify(global_scope[0])
        if self.sp.simplify(global_upper - radius) != 0:
            return None
        if not self._local_range_stated(text, local_upper, radius):
            return None

        variable_text = str(variable)
        local_text = str(local_upper)
        radius_text = self.symbolic._format(radius)
        scaled_local = self.symbolic._format(self.sp.simplify(scale * local_upper))
        scale_text = self.symbolic._format(scale)
        scaled_variable = variable_text if scale == 1 else f"{scale_text}{variable_text}"
        sum_text = rf"-\log(1-{variable_text})" if scale == 1 else rf"-\log(1-{scale_text}{variable_text})"
        zh = bool(re.search(r"[\u4e00-\u9fff]", text))
        result = (
            rf"在每个 $[0,{local_text}]$（$0<{local_text}<{radius_text}$）上，"
            rf"$\left|({scaled_variable})^n/n\right|\le({scaled_local})^n/n$，"
            rf"而 $\sum_{{n\ge1}}({scaled_local})^n/n$ 收敛，故由 Weierstrass M 判别法一致收敛。"
            rf"当 $0\le {variable_text}<{radius_text}$ 时逐项积分几何级数得到和函数 "
            rf"$S({variable_text})={sum_text}$。但在整个 $[0,{radius_text})$ 上不一致收敛："
            rf"每个部分和均有界，而和函数 $S({variable_text})$ 在 "
            rf"${variable_text}\uparrow {radius_text}$ 时无界；"
            rf"一致极限若存在则必须仍有界。"
            if zh else
            rf"On every $[0,{local_text}]$ with $0<{local_text}<{radius_text}$, "
            rf"$|({scaled_variable})^n/n|\le({scaled_local})^n/n$, and "
            rf"$\sum_{{n\ge1}}({scaled_local})^n/n$ converges, so the Weierstrass M-test gives "
            rf"uniform convergence. Termwise integration of the geometric series gives "
            rf"$S({variable_text})={sum_text}$ for $0\le {variable_text}<{radius_text}$. "
            rf"The convergence is not uniform on the whole $[0,{radius_text})$: every partial sum is "
            rf"bounded, whereas the sum function $S({variable_text})$ is unbounded as "
            rf"${variable_text}\uparrow {radius_text}$."
        )
        return make_parameterized_tool_result(
            problem=text,
            operation="logarithmic_power_series_uniform_convergence",
            result=result,
            result_kind="sum_function_and_uniform_convergence_scopes",
            method="m_test_geometric_antiderivative_and_unbounded_limit",
            whole=True,
            written_support=True,
            checks=(
                "power_series_summand_parsed",
                "positive_linear_scale_checked",
                "local_closed_interval_and_range_checked",
                "global_half_open_radius_checked",
                "m_test_majorant_recomputed",
                "sum_function_recomputed_by_geometric_antiderivative",
                "bounded_partial_sums_unbounded_limit_obstruction",
            ),
            support=result,
            answer_shapes=("expression", "text", "proof", "interval"),
            requirements=(
                "result_present",
                "reasoning",
                "local_uniform_convergence",
                "global_nonuniform_convergence",
                "series_sum_function",
                "uniform_convergence_scope_reason",
            ),
        )

    def _summand(self, text: str):
        match = re.search(
            r"\\sum\s*_\s*\{?\s*n\s*=\s*1\s*\}?\s*\^\s*"
            r"\{?\s*\\?infty\s*\}?\s*(?P<term>[^$，,。;；\n]+)",
            text,
            re.I,
        )
        if match is None:
            return None
        term = match.group("term").strip()
        fraction = re.fullmatch(
            r"(?P<base>\([^()]+\)|(?:[-+]?\d+(?:/\d+)?)?\s*[A-Za-z])"
            r"\s*\^\s*\{?\s*n\s*\}?\s*/\s*n",
            term,
        )
        if fraction is None:
            return None
        base_text = fraction.group("base").strip()
        if base_text.startswith("(") and base_text.endswith(")"):
            base_text = base_text[1:-1]
        base = self._expr(base_text)
        if base is None or len(base.free_symbols) != 1:
            return None
        variable = next(iter(base.free_symbols))
        return base, variable

    def _intervals(self, text: str):
        intervals = []
        for match in re.finditer(
            r"\[\s*0\s*[,，]\s*(?P<upper>[^\]\)\n]+)\s*(?P<right>[\]\)])",
            text,
        ):
            upper = self._expr(match.group("upper"))
            if upper is not None:
                intervals.append((upper, match.group("right")))
        return tuple(intervals)

    def _local_range_stated(self, text: str, symbol, radius) -> bool:
        compact = re.sub(r"\s+", "", text).replace(r"\frac", "frac")
        symbol_text = re.escape(str(symbol))
        radius_sources = {
            re.sub(r"\s+", "", self.symbolic._format(radius)).replace(r"\frac", "frac"),
            re.sub(r"\s+", "", str(radius)),
        }
        return any(re.search(rf"0<{symbol_text}<{re.escape(source)}", compact) for source in radius_sources)

    def _expr(self, value: str):
        try:
            clean = str(value or "").strip().replace(r"\,", "")
            clean = re.sub(r"(?<=\d)(?=[A-Za-z])", "*", clean)
            return self.sp.simplify(self.symbolic._parse(clean))
        except Exception:
            return None
